"""The record tap in run_teleop.py and scripts/teleop/recorder.py, against the simulated arms.

Nothing here touches the robot, the CAN buses or the real configs: the supervisors get
SimArmDriver, scratch copies of the profile and claw limits, ports of their own and no
servos, and recorder.py writes into a temporary folder ($TMPDIR).

  .venv/bin/python tools/test_recorder.py

a. The tap changes nothing. Off, there is no socket and no per-cycle work. On, a supervisor
   commands exactly what one without it does, tick for tick, even while its sends fail.
   With a fake Nano: the claws' last pulses outlive a re-arm, a NaN goes out as null, and
   there is no camera angle while the Nano is gone. The servo bookkeeping the tap reads
   changes no pulse: the same bytes at the same moments as ServoLink before it.
b. recorder.py runs as its own process, fed by a supervisor ticking at ~50 Hz and a fake
   camera writing 1280x480 JPEGs at 30 Hz, and driven with record_ctl.send and the way
   voice_typer.py sends "agent, start episode". It checks the refused starts, datagrams
   nested too deep for json on both ports, one episode and its files byte for byte (the end
   answered before the save), discard, primes used by voice starts, and the automatic aborts
   (rows stop, zero taken again, recorder killed with SIGTERM, its terminal closed: SIGHUP).
c. tools/lfd/zero_check.py's decision rule, whose result every episode carries, at its edges.
d. recorder.py's Recorder in this process: an end answered before its save, a prime used
   once, one that expires, and a disk that runs low mid-episode.
e. A full disk under run_teleop.py's own teleop log never stops the control loop.
"""

from __future__ import annotations

import collections
import contextlib
import hashlib
import io
import json
import math
import os
import queue
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/teleop"))
sys.path.insert(0, str(REPO / "tools/lfd"))
try:
    import berkeley_humanoid_lite_lowlevel  # noqa: E402, F401
except ImportError:   # the rig's .venv installs it from source/; a bare environment has only the tree
    sys.path.insert(0, str(REPO / "source/berkeley_humanoid_lite_lowlevel"))

import run_teleop as rt  # noqa: E402
from arm_driver import SimArmDriver  # noqa: E402
from arm_power import ARM_JOINTS, GravityModel, PowerProfile  # noqa: E402
import record_ctl  # noqa: E402
import recorder  # noqa: E402
import servos  # noqa: E402
import voice_typer  # noqa: E402
import zero_check  # noqa: E402

# docs/LFD_RECORDING_FORMAT.md, field for field
ROW_KEYS = {"v", "seq", "t", "state", "motors", "src", "q", "qc", "tau", "tff", "lim", "zero", "held", "grip",
            "trig", "hand", "head", "tgt", "swap", "scale", "claw", "cam", "cam_mode", "intervention"}
SESSION_KEYS = {"format", "version", "session_id", "created_wall", "host", "repo_commit", "joint_names",
                "urdf_joint_names", "camera", "tap_port", "control_port"}
CAMERA_KEYS = {"frame_file", "layout", "rotated_180", "calibration", "camera_latency_s"}
EPISODE_KEYS = {"format", "version", "episode_index", "session_id", "task", "mode", "policy", "operator",
                "arrangement", "trial", "start", "end", "outcome", "end_reason", "discarded", "notes", "counts",
                "cam_mode_at_start", "cam_at_start", "zero_at_start", "zero_changed", "zero_convention",
                "zero_check", "profile_sha256", "claw_limits", "warnings"}
COUNT_KEYS = {"rows", "frames", "row_gaps_over_50ms", "frame_gaps_over_67ms"}
FRAME_KEYS = {"i", "off", "len", "t_cap", "t_seen", "mtime_ns"}
TASK = "Put the foam block in the bowl"
# a tap row as run_teleop.py sends one with the motors off, for feeding a recorder by hand
FAKE_ROW = {"v": 1, "seq": 0, "t": 0.0, "state": "STOPPED", "motors": False, "src": "idle",
            "q": [0.0] * 10, "qc": [0.0] * 10, "tau": [0.0] * 10, "tff": [0.0] * 10, "lim": [1.0] * 10,
            "zero": [0.0] * 10, "held": [False, False], "grip": [False, False], "trig": [0.0, 0.0],
            "hand": [None, None], "head": None, "tgt": [None, None], "swap": False, "scale": 0.6,
            "claw": [None, None], "cam": [0.0, 0.0], "cam_mode": "track", "intervention": False}
# configs/claw_limits.json as it is on the rig: the servo scenarios do not depend on the file
LIMITS = {"D3": {"open_us": 1000, "closed_us": 1640}, "D7": {"open_us": 1000, "closed_us": 1670},
          "D9": {"center_us": 1500}, "D5": {"level_us": 1500, "up_us": 1200, "down_us": 1850}}
DEEP = b"[" * 5000 + b"]" * 5000      # valid JSON, nested past what json.loads can recurse into


def free_port() -> int:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def pose(x: float, y: float, z: float, yaw: float = 0.0) -> list:
    c, s = math.cos(yaw), math.sin(yaw)
    return [[c, -s, 0.0, x], [s, c, 0.0, y], [0.0, 0.0, 1.0, z], [0.0, 0.0, 0.0, 1.0]]


def packet(t: float) -> dict:
    """What the headset sends: both controllers tracked, both grips held, the hands circling slowly."""
    a = 2 * math.pi * 0.5 * t
    return {"left": {"tracked": True, "button_pressed": True, "trigger": 0.25,
                     "pose": pose(0.30, 0.20 + 0.05 * math.cos(a), 1.00 + 0.05 * math.sin(a), 0.2 * math.sin(a))},
            "right": {"tracked": True, "button_pressed": True, "trigger": 0.75,
                      "pose": pose(0.30, -0.20 + 0.05 * math.cos(a), 1.00 - 0.05 * math.sin(a))},
            "head": pose(0.0, 0.0, 1.6, 0.1), "stick": [0.0, 0.0]}


def supervisor(tmp: Path, gravity: GravityModel, profile: Path, record_port: int | None):
    """A supervisor as run_teleop.py builds one, on the simulated arms, a scratch profile and its own ports."""
    return rt.ArmSupervisor(SimArmDriver(gravity), PowerProfile.load(profile), gravity, rt.TeleopIkSolver(),
                            servos=None, person_port=0, tune_port=0, record_port=record_port)


def arm(sup, now: float) -> None:
    """Motors on and ARMED, as _arm_teleop does, minus the teleop log it would open inside the repo."""
    sup.start(now)
    sup.solver.reset()
    sup._motors_on(now, sup.driver.q.copy())
    sup.state = "ARMED"


def scratch_profile(tmp: Path, name: str) -> Path:
    path = tmp / f"profile_{name}.json"
    shutil.copy(REPO / "configs/arm_power_profile.json", path)
    return path


class RefusingSocket:
    """A tap socket whose every send fails, as a full buffer would."""

    def sendto(self, *args):
        raise BlockingIOError("send buffer full")


def same_state(on, off) -> str:
    """'' when the two supervisors command and hold exactly the same, else what differs."""
    for name in ("q_cmd", "tau_ff", "limits"):
        if not np.array_equal(getattr(on, name), getattr(off, name)):
            return name
    if not np.array_equal(on.limiter.sent, off.limiter.sent, equal_nan=True):
        return "limits sent"
    if not np.array_equal(on.driver.q, off.driver.q, equal_nan=True):
        return "measured q"
    if (on.state, on.motors) != (off.state, off.motors):
        return f"state {on.state}/{on.motors} vs {off.state}/{off.motors}"
    return ""


def test_no_change(tmp: Path, gravity: GravityModel, failures: list[str]) -> None:
    """a. Same packets, same clock: the tap-on supervisor commands what the tap-off one does."""
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
    receiver.bind(("127.0.0.1", 0))
    receiver.setblocking(False)
    off = supervisor(tmp, gravity, scratch_profile(tmp, "off"), None)
    on = supervisor(tmp, gravity, scratch_profile(tmp, "on"), receiver.getsockname()[1])
    before = len(failures)
    if off.tap is not None or off.record_port is not None:
        failures.append("with record_port None a tap socket exists")
    calls: list[int] = []
    off._record_tap = lambda: calls.append(1)        # any per-cycle tap work would land here
    rows, q_cmd_at, first_difference = [], {}, ""
    t0, real_tap = 1000.0, on.tap
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        arm(off, t0)
        arm(on, t0)
        q_start = on.q_cmd.copy()
        for k in range(1, 201):
            now = t0 + 0.02 * k                       # one synthetic clock for both
            sent = packet(0.02 * k)
            if k == 100:
                on.tap = RefusingSocket()             # 20 cycles whose rows cannot be sent
            elif k == 120:
                on.tap = real_tap
            off.tick(now, sent, 0.0, {})
            on.tick(now, sent, 0.0, {})
            q_cmd_at[k - 1] = (on.q_cmd.copy(), sent)
            if not first_difference:
                what = same_state(on, off)
                if what:
                    first_difference = f"tick {k}: {what}"
            while True:
                try:
                    rows.append(json.loads(receiver.recv(65536)))
                except BlockingIOError:
                    break
    receiver.close()
    moved = float(np.max(np.abs(on.q_cmd - q_start)))
    seqs = [row.get("seq") for row in rows]
    if calls:
        failures.append(f"with the tap off, _record_tap still ran {len(calls)} times")
    elif first_difference:
        failures.append(f"the tap changed what the loop does, first at {first_difference}")
    elif on.state != "ARMED" or moved < 0.05:
        failures.append(f"the comparison proves little: state {on.state}, q_cmd moved only {moved:.3f} rad")
    elif on.tap_errors != 20 or "tap errors 20" not in on.summary_line() or on.status().get("tap_errors") != 20 \
            or "tap errors" in off.summary_line() or "tap_errors" in off.status():
        failures.append(f"20 failed sends were counted as {on.tap_errors}, or the summary line and status do not say "
                        f"so (or say it with the tap off): {on.summary_line()!r}")
    elif seqs != list(range(0, 99)) + list(range(119, 200)):
        failures.append(f"rows went missing, or seq does not count every cycle: {len(rows)} rows, seq {seqs[:3]}...")
    else:
        print(f"ok    tap off: no socket, no per-cycle call; tap on: q_cmd, tau_ff, limits, state identical for "
              f"200 armed ticks (q_cmd moved {moved:.2f} rad), 20 failed sends counted (in the summary line and "
              "status), never raised")
    if len(failures) > before:
        print(printed.getvalue()[-2000:])
        return

    # the rows themselves: every field of the spec, and what the loop really had
    problems = []
    for row in rows:
        q_cmd, sent = q_cmd_at[row["seq"]]
        left = np.array(sent["left"]["pose"])[:3, 3]
        hand = row["hand"][0]
        if set(row) != ROW_KEYS:
            problems.append(f"fields {sorted(set(row) ^ ROW_KEYS)} differ from the spec")
        elif row["qc"] != np.round(q_cmd, 5).tolist():
            problems.append(f"seq {row['seq']}: qc is not the q_cmd that cycle sent")
        elif hand is None or [hand[3], hand[7], hand[11]] != np.round(left, 5).tolist() or hand[12:] != [0, 0, 0, 1]:
            problems.append(f"seq {row['seq']}: the left hand pose is not row-major 4x4: {hand}")
        elif (row["v"], row["src"], row["state"], row["motors"], row["intervention"]) != (1, "teleop", "ARMED",
                                                                                          True, False):
            problems.append(f"seq {row['seq']}: v/src/state/motors/intervention wrong: {row}")
        elif row["grip"] != [True, True] or row["trig"] != [0.25, 0.75] or row["held"] != [True, True]:
            problems.append(f"seq {row['seq']}: grips, triggers or held wrong: {row['grip']} {row['trig']}")
        elif any(len(row[k]) != len(ARM_JOINTS) for k in ("q", "qc", "tau", "tff", "lim", "zero")):
            problems.append(f"seq {row['seq']}: a joint vector is not 10 long")
        elif len(row["head"]) != 9 or any(t is None or len(t) != 3 for t in row["tgt"]):
            problems.append(f"seq {row['seq']}: head or IK targets missing")
        elif row["claw"] != [None, None] or row["cam"] is not None or row["cam_mode"] is not None:
            problems.append(f"seq {row['seq']}: without servos claw/cam should be null: {row['claw']} {row['cam']}")
        elif row["zero"] != np.round(on.driver.zero_fw, 5).tolist() or row["swap"] != on.profile.swap_hands:
            problems.append(f"seq {row['seq']}: zero or swap wrong")
        if problems:
            break
    times = [row["t"] for row in rows]
    size = max(len(json.dumps(row, separators=(",", ":"))) for row in rows)
    if problems:
        failures.append("tap rows: " + problems[0])
    elif any(b < a for a, b in zip(times, times[1:])):
        failures.append("tap row times go backwards")
    else:
        print(f"ok    tap rows: exactly the spec's fields, qc is the cycle's q_cmd, hands row-major 4x4, "
              f"monotonic t, at most {size} bytes")


class FakePort:
    """The Nano's serial port: every write kept, with the (fake) time it was made."""

    def __init__(self, link: "FakeNanoLink"):
        self.link = link

    def write(self, data: bytes) -> int:
        if self.link.unplug_next:
            self.link.unplug_next = False
            raise OSError("the Nano was unplugged")
        self.link.writes.append((self.link.clock[0], data))
        return len(data)

    def read(self, size: int) -> bytes:
        return b""

    def close(self) -> None:
        pass


class FakeNanoLink(servos.ServoLink):
    """servos.ServoLink itself (move, off, _send) on a FakePort: it never looks for a real Nano."""

    def __init__(self, clock: list[float]):
        self.clock = clock
        self.writes: list[tuple[float, bytes]] = []
        self.unplug_next = False
        super().__init__()
        self.plug()

    def _open(self) -> None:                 # the background search for the Nano: nothing to find here
        pass

    def plug(self) -> None:
        self.port = FakePort(self)


class OldMovesLink(FakeNanoLink):
    """move and off exactly as servos.ServoLink had them before the record tap's last_us."""

    def move(self, pin: int, us: int) -> bool:
        return self._send(f"P {pin} {max(servos.MIN_US, min(servos.MAX_US, int(us)))}")

    def off(self) -> bool:
        return self._send("OFF")


def last_pulses(writes: list[tuple[float, bytes]]) -> dict[int, int]:
    """Pin -> the last pulse written to it, from the bytes the Nano got."""
    last = {}
    for _, data in writes:
        if data.startswith(b"P "):
            _, pin, us = data.split()
            last[int(pin)] = int(us)
    return last


def turned(yaw: float, pitch: float) -> np.ndarray:
    """A head rotation in the robot frame: yaw about z, then pitch about y."""
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    return np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]]) @ np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0],
                                                                                [-sp, 0.0, cp]])


def servo_scenario(link: FakeNanoLink) -> list[bool]:
    """Twelve seconds of claws (re-armed halfway) and camera (tracking, following the head, still), the
    Nano unplugged and found again, then direct moves at the edges, OFF and a failed move. What each
    direct call returned."""
    clock = link.clock
    teleop, camera = servos.ServoTeleop(link), servos.CameraAim(link)
    teleop.start(None)
    teleop.limits, camera.limits = LIMITS, LIMITS
    for k in range(600):
        clock[0] = now = 100.0 + 0.02 * k
        if k == 300:
            teleop.start(None)                     # a re-arm: a new _Sender, its rate limits afresh
            teleop.limits = LIMITS
        if k in (200, 400):
            camera.set_mode("head" if k == 200 else "still")
        if k == 350:
            link.unplug_next = True                # the next write fails and the port is dropped...
        if k == 380:
            link.plug()                            # ...until the Nano is found again
        grips = [k % 200 < 120, 30 < k % 150]
        triggers = [0.5 + 0.5 * math.sin(k / 17), 0.5 + 0.5 * math.cos(k / 23)]
        head = turned(0.4 * math.sin(k / 40), 0.2 * math.sin(k / 60))
        person = {"type": "person", "seen": k % 90 < 70, "t": k // 8, "yaw": 25 * math.sin(k / 70),
                  "pitch": 3.0, "top": 9.0}
        teleop.update(now, grips, triggers, head)
        camera.update(now, head, person)
    clock[0] = 120.0
    returns = [link.move(pin, us) for pin, us in ((3, 1320.7), (7, 1450), (9, 99999), (5, -5), (3, 1320),
                                                   (7, 2600.2), (5, 1500.0), (9, "1700"))]
    returns.append(link.off())
    link.unplug_next = True
    returns.append(link.move(3, 1600))             # fails: neither sent nor kept
    return returns


def test_servo_bytes(failures: list[str]) -> None:
    """a. ServoLink.last_us, which the tap reads, changes no pulse: same pins, same values, same moments."""
    new, old = FakeNanoLink([0.0]), OldMovesLink([0.0])
    with contextlib.redirect_stdout(io.StringIO()):          # "Servos: claw/camera Nano lost ..."
        returns_new, returns_old = servo_scenario(new), servo_scenario(old)
    pins = set(last_pulses(new.writes))
    if new.writes != old.writes or returns_new != returns_old:
        first = next((i for i, (a, b) in enumerate(zip(new.writes, old.writes)) if a != b),
                     min(len(new.writes), len(old.writes)))
        failures.append(f"the servo bytes changed: {len(new.writes)} writes, {len(old.writes)} before; first "
                        f"difference at write {first}; returns {returns_new} vs {returns_old}")
    elif pins != {3, 5, 7, 9} or len(new.writes) < 100 or (120.0, b"OFF\n") not in new.writes \
            or False not in returns_new:
        failures.append(f"the servo scenario proves little: {len(new.writes)} writes to pins {sorted(pins)}")
    elif new.last_us != last_pulses(new.writes):
        failures.append(f"ServoLink.last_us {new.last_us} is not the last pulse written per pin "
                        f"{last_pulses(new.writes)}")
    else:
        print(f"ok    ServoLink.last_us changes no pulse: {len(new.writes)} writes to the Nano (claws through a "
              "re-arm, camera tracking, on the head and still, an unplug, moves at the edges, OFF) are the same "
              "bytes at the same moments as before it, and it holds the last pulse sent to each pin")


def reject(token: str):
    raise ValueError(f"{token} is not JSON")


def test_tap_port(failures: list[str]) -> None:
    """a. --record-tap-port is checked when parsed: a port every send would fail on is refused at once."""
    taken, refused = [], []
    for text in ("1024", "11014", "65535", "1023", "80", "65536", "70000", "-1", "x"):
        try:
            taken.append(rt.tap_port(text))
        except (ValueError, rt.argparse.ArgumentTypeError):
            refused.append(text)
    if taken != [1024, 11014, 65535] or refused != ["1023", "80", "65536", "70000", "-1", "x"]:
        failures.append(f"--record-tap-port took {taken} and refused {refused}")
    else:
        print("ok    --record-tap-port takes 1024..65535 and refuses 1023, 80, 65536, 70000, -1 and 'x'")


def test_tap_values(tmp: Path, gravity: GravityModel, failures: list[str]) -> None:
    """a. The tap with a fake Nano: claws through a re-arm, NaN as null, no camera angle without the Nano."""
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
    receiver.bind(("127.0.0.1", 0))
    receiver.setblocking(False)
    link = FakeNanoLink([0.0])
    sup = rt.ArmSupervisor(SimArmDriver(gravity), PowerProfile.load(scratch_profile(tmp, "claws")), gravity,
                           rt.TeleopIkSolver(), servos=link, person_port=0, tune_port=0,
                           record_port=receiver.getsockname()[1])
    sup.camera.limits = LIMITS

    def rows() -> list[bytes]:
        got = []
        while True:
            try:
                got.append(receiver.recv(65536))
            except BlockingIOError:
                return got

    released = packet(0.0)
    released["left"]["button_pressed"] = released["right"]["button_pressed"] = False
    t0 = 1000.0
    with contextlib.redirect_stdout(io.StringIO()):
        sup._record_tap()
        before = [json.loads(raw) for raw in rows()]
        arm(sup, t0)
        sup.servo_teleop.start(None)                 # as _arm_teleop does
        sup.servo_teleop.limits = LIMITS
        for k in range(1, 61):                       # both grips held, the triggers at 0.25 and 0.75
            link.clock[0] = t0 + 0.02 * k
            sup.tick(t0 + 0.02 * k, packet(0.02 * k), 0.0, {})
        driving = [json.loads(raw) for raw in rows()]
        last = last_pulses(link.writes)
        sent, writes_then = [last.get(3), last.get(7)], len(link.writes)
        sup.servo_teleop.start(None)                 # stopped and armed again: a new _Sender
        sup.servo_teleop.limits = LIMITS
        for k in range(61, 91):                      # grips let go: the claws keep what they hold
            link.clock[0] = t0 + 0.02 * k
            sup.tick(t0 + 0.02 * k, released, 0.0, {})
        rearmed = [json.loads(raw) for raw in rows()]
        claw_writes = [data for _, data in link.writes[writes_then:] if data.startswith((b"P 3 ", b"P 7 "))]
        sup.driver.p_fw[2] = float("nan")            # a joint not heard from yet
        sup.tau_ff[0] = float("inf")
        sup._record_tap()
        odd = rows()
        link.port = None                             # the Nano unplugged
        sup._record_tap()
        unplugged = [json.loads(raw) for raw in rows()]
    receiver.close()
    try:
        strict = json.loads(odd[0], parse_constant=reject) if len(odd) == 1 else None
    except ValueError as exc:
        strict = f"{exc}"
    if not before or before[-1]["claw"] != [None, None]:
        failures.append(f"before any pulse the tap's claw is not null: {before}")
    elif None in sent or not driving or driving[-1]["claw"] != sent:
        failures.append(f"while driving, the tap's claw {driving[-1]['claw'] if driving else None} is not the last "
                        f"pulse sent {sent}")
    elif claw_writes or not rearmed or any(row["claw"] != sent for row in rearmed):
        failures.append(f"after a re-arm the claw reads {[row['claw'] for row in rearmed][-3:]}, not the {sent} "
                        f"the claws still hold ({len(claw_writes)} claw pulses since)")
    elif not isinstance(strict, dict) or strict["q"][2] is not None or strict["tff"][0] is not None \
            or not recorder.usable_row(strict):
        failures.append(f"a NaN or inf is not sent as null in strict JSON: {strict if strict else odd}")
    elif not (isinstance(driving[-1]["cam"], list) and len(driving[-1]["cam"]) == 2) \
            or not unplugged or unplugged[0]["cam"] is not None or unplugged[0]["cam_mode"] != "track":
        failures.append(f"camera angle with the Nano {driving[-1]['cam']}, without it {unplugged}")
    elif sup.tap_errors:
        failures.append(f"{sup.tap_errors} tap rows failed")
    else:
        print(f"ok    tap with a Nano: claw null before any pulse, then the last pulse sent {sent}, still so after a "
              "re-arm; q NaN and tff inf go out as null (strict JSON, usable); no camera angle without the Nano")


class Camera(threading.Thread):
    """camera_share.py's stand-in: a new 1280x480 JPEG every 1/30 s, written atomically as ffmpeg's is."""

    def __init__(self, path: Path, fps: float = 30.0):
        super().__init__(daemon=True)
        self.path, self.period = path, 1.0 / fps
        self.kept: collections.deque = collections.deque(maxlen=200)    # (number, bytes) of the last 200
        self.running = True
        self.error: Exception | None = None

    def run(self) -> None:
        rng = np.random.default_rng(7)
        background = np.full((480, 1280, 3), (40, 60, 80), np.uint8)
        tmp = self.path.with_name(self.path.name + ".tmp")
        n, due = 0, time.perf_counter()
        try:
            while self.running:
                image = background.copy()
                x = (n * 17) % 1200
                image[200:280, x:x + 80] = rng.integers(0, 256, (80, 80, 3), dtype=np.uint8)   # every frame differs
                cv2.putText(image, str(n), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
                _, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
                data = jpeg.tobytes()
                tmp.write_bytes(data)
                os.replace(tmp, self.path)
                self.kept.append((n, data))
                n += 1
                due += self.period
                if time.perf_counter() - due > 0.2:
                    due = time.perf_counter()
                time.sleep(max(0.0, due - time.perf_counter()))
        except Exception as exc:
            self.error = exc


class Ticker(threading.Thread):
    """run_teleop.py's loop at ~50 Hz: the armed supervisor fed the synthetic headset. Jobs run between ticks."""

    def __init__(self, sup):
        super().__init__(daemon=True)
        self.sup = sup
        self.running = True
        self.paused = threading.Event()
        self.jobs: queue.Queue = queue.Queue()
        self.error: Exception | None = None

    def run(self) -> None:
        t0 = due = time.perf_counter()
        try:
            arm(self.sup, t0)
            while self.running:
                while not self.jobs.empty():
                    self.jobs.get()()
                if not self.paused.is_set():
                    now = time.perf_counter()
                    self.sup.tick(now, packet(now - t0), 0.0, {})
                due += 0.02
                if time.perf_counter() - due > 0.1:
                    due = time.perf_counter()
                time.sleep(max(0.0, due - time.perf_counter()))
        except Exception as exc:
            self.error = exc


def wait_until(predicate, timeout: float, step: float = 0.05) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(step)
    return predicate()


def check_episode_files(folder: Path, episode: dict, camera: Camera) -> str:
    """rows.jsonl and the frames: '' when they are right, else what is wrong."""
    rows = [json.loads(line) for line in (folder / "rows.jsonl").read_text().splitlines()]
    if len(rows) != episode["counts"]["rows"]:
        return f"{len(rows)} rows in rows.jsonl, episode.json says {episode['counts']['rows']}"
    if any(set(row) != ROW_KEYS for row in rows):
        return "a row in rows.jsonl does not have exactly the spec's fields"
    seqs, times = [row["seq"] for row in rows], [row["t"] for row in rows]
    if any(b <= a for a, b in zip(seqs, seqs[1:])) or any(b < a for a, b in zip(times, times[1:])):
        return "rows.jsonl is not in order"
    start, end = episode["start"]["t"], episode["end"]["t"]
    if not (start - 0.05 <= times[0] and times[-1] <= end + 0.05):
        return f"rows from outside the episode: {times[0]:.3f}..{times[-1]:.3f} vs {start:.3f}..{end:.3f}"
    if any(row["zero"] != episode["zero_at_start"] for row in rows):
        return "a row's zero differs from zero_at_start"
    index = [json.loads(line) for line in (folder / "frames.jsonl").read_text().splitlines()]
    blob = (folder / "frames.mjpeg").read_bytes()
    written = {data: n for n, data in camera.kept}
    if len(index) != episode["counts"]["frames"] or not index:
        return f"{len(index)} lines in frames.jsonl, episode.json says {episode['counts']['frames']}"
    offset, numbers = 0, []
    for i, entry in enumerate(index):
        if set(entry) != FRAME_KEYS or entry["i"] != i or entry["off"] != offset:
            return f"frames.jsonl line {i} is not {sorted(FRAME_KEYS)} with contiguous offsets: {entry}"
        data = blob[entry["off"]:entry["off"] + entry["len"]]
        if data not in written:
            return f"frame {i} ({entry['len']} bytes at {entry['off']}) is not a JPEG the camera wrote"
        numbers.append(written[data])
        if not -0.002 <= entry["t_seen"] - entry["t_cap"] <= 0.25:
            return f"frame {i}: seen {entry['t_seen'] - entry['t_cap']:+.3f} s after its capture time"
        offset += entry["len"]
    if offset != len(blob):
        return f"frames.mjpeg holds {len(blob)} bytes, the index covers {offset}"
    if any(b <= a for a, b in zip(numbers, numbers[1:])):
        return f"frames out of order or repeated: {numbers[:10]}"
    if not (start - 0.1 <= index[0]["t_cap"] and index[-1]["t_seen"] <= end + 0.05):
        return "frames from outside the episode"
    return ""


def test_recorder(tmp: Path, gravity: GravityModel, failures: list[str]) -> None:
    """b. recorder.py as its own process, end to end."""
    root, frame_file = tmp / "recordings", tmp / "bhl_camera.jpg"
    profile = scratch_profile(tmp, "rec")
    claws = tmp / "claw_limits.json"
    shutil.copy(REPO / "configs/claw_limits.json", claws)
    zero_file = tmp / "zero_check.json"
    with contextlib.redirect_stdout(io.StringIO()):
        zero_check.main(["--torso-deg", "0.5", "--left-upper-arm-deg", "1.0", "--right-upper-arm-deg", "-0.5",
                         "--profile", str(profile), "--out", str(zero_file)])
    tap_port, ctl_port = free_port(), free_port()
    camera = Camera(frame_file)
    sup = supervisor(tmp, gravity, profile, tap_port)
    sup._say = lambda text: None                       # its alerts would only clutter this output
    ticker = Ticker(sup)
    log = open(tmp / "recorder.log", "w+")
    proc = None
    before = len(failures)

    def ctl(cmd: dict) -> dict:
        return record_ctl.send(cmd, ctl_port)

    def fail(text: str) -> None:
        log.flush()
        log.seek(0)
        failures.append(f"{text}\n      recorder.log: ...{log.read()[-1500:]}")

    def episode_json(name: str, where: Path | None = None) -> dict:
        return json.loads(((where or session) / name / "episode.json").read_text())

    try:
        camera.start()
        wait_until(lambda: frame_file.exists(), 5.0)
        proc = subprocess.Popen(
            [sys.executable, str(REPO / "scripts/teleop/recorder.py"), "--root", str(root),
             "--tap-port", str(tap_port), "--control-port", str(ctl_port), "--frame-file", str(frame_file),
             "--profile", str(profile), "--claw-limits", str(claws), "--zero-check", str(zero_file)],
            cwd=REPO, stdout=log, stderr=subprocess.STDOUT, text=True)
        if not wait_until(lambda: proc.poll() is not None or ctl({"cmd": "status"}).get("ok"), 30.0, 0.2) \
                or proc.poll() is not None:
            return fail("recorder.py did not come up")
        sessions = [p for p in root.iterdir() if p.is_dir()]
        session = sessions[0]
        time.sleep(0.4)                                  # frames flow; no tap rows yet

        # refused: no rows; then rows from a camera that is not still
        no_rows = ctl({"cmd": "start", "task": TASK, "mode": "teleop"})
        fake = dict(FAKE_ROW)
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for n in range(15):
            fake.update(seq=n, t=time.monotonic())
            sender.sendto(json.dumps(fake).encode(), ("127.0.0.1", tap_port))
            time.sleep(0.02)
        sender.close()
        tracking = ctl({"cmd": "start", "task": TASK, "mode": "teleop"})
        if no_rows.get("ok") or "tap rows" not in no_rows.get("said", ""):
            return fail(f"a start before any tap row was not refused for that: {no_rows}")
        if tracking.get("ok") or "still" not in tracking.get("said", ""):
            return fail(f"a start while the camera tracks was not refused for that: {tracking}")
        print(f"ok    start refused before any tap row ({no_rows['said']!r}) and while the camera moves")
        junk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        junk.settimeout(2.0)
        junk.sendto(b"not json", ("127.0.0.1", ctl_port))
        garbled = json.loads(junk.recvfrom(65536)[0])
        junk.close()
        odd = [ctl({"cmd": "end", "outcome": "success"}), ctl({"cmd": "fly"}), ctl({"cmd": "task", "task": " "})]
        if garbled.get("ok") is not False or any(reply.get("ok") is not False for reply in odd):
            return fail(f"garbage, an end with nothing open, an unknown command or an empty task was accepted: "
                        f"{garbled} {odd}")
        # JSON nested too deep for json.loads (RecursionError, not a ValueError) on both ports, and a row
        # whose time is an integer too big for a float: each is refused or ignored, and the recorder lives on
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.settimeout(2.0)
        probe.sendto(DEEP, ("127.0.0.1", ctl_port))
        try:
            nested = json.loads(probe.recvfrom(65536)[0])
        except socket.timeout:
            nested = {"no answer": "in 2 s"}
        probe.sendto(DEEP, ("127.0.0.1", tap_port))
        probe.sendto(json.dumps(dict(FAKE_ROW, t=10 ** 400)).encode(), ("127.0.0.1", tap_port))
        probe.close()
        alive = ctl({"cmd": "status"})
        if nested.get("ok") is not False or proc.poll() is not None or not alive.get("ok") \
                or "2 unreadable tap rows ignored" not in alive.get("warnings", []):
            return fail(f"nested JSON or a huge time: reply {nested}, recorder exit {proc.poll()}, status {alive}")
        print("ok    JSON nested 5000 deep on the command and tap ports, and a row whose t is 10**400: refused or "
              "ignored, the recorder lives on")

        # one episode: the default task, rows and frames flowing, then success
        ticker.start()
        time.sleep(0.6)
        task_set = ctl({"cmd": "task", "task": TASK})
        started = ctl({"cmd": "start", "mode": "teleop", "operator": "tester", "arrangement": "A", "trial": 3})
        again = ctl({"cmd": "start", "mode": "teleop"})
        status = ctl({"cmd": "status"})
        time.sleep(2.0)
        ended = ctl({"cmd": "end", "outcome": "success", "notes": "test notes"})
        saved = ctl({"cmd": "status"})                   # answered only once the end's save is done
        if not task_set.get("ok") or not started.get("ok"):
            return fail(f"the start was refused with rows and frames flowing: {task_set} {started}")
        if "no camera servos: camera angle unknown" not in started.get("warnings", []):
            return fail(f"no warning that the camera angle is unknown: {started}")
        if again.get("ok"):
            return fail("a second start was accepted while an episode was open")
        if (status.get("episode") or {}).get("name") != "ep_0000" or not status.get("rows_per_s"):
            return fail(f"status does not show the open episode and its rates: {status}")
        if not ended.get("ok") or not ended.get("said", "").endswith("saving"):
            return fail(f"end was refused, or not answered before its save: {ended}")
        if saved.get("last_episode") != {"name": "ep_0000", "outcome": "success", "end_reason": "operator",
                                         "saved": True, "discarded": False}:
            return fail(f"the status after the end does not show ep_0000 saved: {saved}")
        print(f"ok    start with rows and frames flowing ({started['said']!r}); a second start refused; "
              f"status {status['rows_per_s']:.0f} rows/s, {status['frames_per_s']:.0f} frames/s; end answered "
              f"before its save ({ended['said']!r}), the next status after it")

        folder = session / "ep_0000"
        session_json = json.loads((session / "session.json").read_text())
        episode = episode_json("ep_0000")
        missing = [name for name in ("episode.json", "rows.jsonl", "frames.mjpeg", "frames.jsonl")
                   if not (folder / name).is_file()]
        seconds = episode["end"]["t"] - episode["start"]["t"]
        row_rate, frame_rate = episode["counts"]["rows"] / seconds, episode["counts"]["frames"] / seconds
        claw_file = json.loads(claws.read_text())["claws"]
        want = {
            "format": "bhl-lfd-episode", "version": 1, "episode_index": 0, "session_id": session.name,
            "task": TASK, "mode": "teleop", "policy": None, "operator": "tester", "arrangement": "A", "trial": 3,
            "outcome": "success", "end_reason": "operator", "discarded": False, "notes": "test notes",
            "cam_mode_at_start": None, "cam_at_start": None, "zero_changed": False, "zero_convention": "hanging=0",
            "zero_at_start": np.round(sup.driver.zero_fw, 5).tolist(),
            "zero_check": json.loads(zero_file.read_text()),
            "profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
            "claw_limits": {side: {k: claw_file[pin][k] for k in ("open_us", "closed_us")}
                            for pin, side in (("D3", "left"), ("D7", "right"))},
        }
        wrong = {k: episode.get(k) for k, v in want.items() if episode.get(k) != v}
        commit = session_json.get("repo_commit")
        if missing:
            return fail(f"episode files missing: {missing}")
        if set(episode) != EPISODE_KEYS or set(episode["counts"]) != COUNT_KEYS:
            return fail(f"episode.json fields differ from the spec: {sorted(set(episode) ^ EPISODE_KEYS)}")
        if wrong:
            return fail(f"episode.json says {wrong}")
        if set(session_json) != SESSION_KEYS or set(session_json["camera"]) != CAMERA_KEYS:
            return fail(f"session.json fields differ from the spec: {sorted(set(session_json) ^ SESSION_KEYS)}")
        if (session_json["joint_names"] != [j.key for j in ARM_JOINTS]
                or session_json["urdf_joint_names"] != [j.urdf for j in ARM_JOINTS]
                or (session_json["tap_port"], session_json["control_port"]) != (tap_port, ctl_port)
                or (commit is not None and len(commit) != 40)):
            return fail(f"session.json is wrong: {session_json}")
        if not (40 <= row_rate <= 55 and 25 <= frame_rate <= 33):
            return fail(f"rates {row_rate:.1f} rows/s, {frame_rate:.1f} frames/s over {seconds:.2f} s "
                        "(want 40-55 and 25-33)")
        problem = check_episode_files(folder, episode, camera)
        if problem:
            return fail(problem)
        print(f"ok    ep_0000: every spec field in session.json and episode.json, {row_rate:.1f} rows/s, "
              f"{frame_rate:.1f} frames/s; all {episode['counts']['frames']} frames byte-identical to the JPEGs "
              "written, in order, contiguous")

        # discard: no episode open, so the last one moves
        discarded = ctl({"cmd": "discard"})
        moved = session / "_discarded" / "ep_0000"
        twice = ctl({"cmd": "discard"})
        if not discarded.get("ok") or folder.exists() or not moved.is_dir():
            return fail(f"discard did not move ep_0000 to _discarded/: {discarded}")
        if not episode_json("ep_0000", session / "_discarded")["discarded"] or twice.get("ok"):
            return fail(f"the discarded episode is not marked so, or a second discard reached further: {twice}")
        print("ok    discard moves the last episode to _discarded/ (marked discarded); a second discard refused")

        # "agent, start episode" as voice_typer sends it (no mode, no task), then the ticks stop
        # for more than 2 s: aborted by itself
        voice_typer.RECORDER_PORT = ctl_port
        unsure = voice_typer.run_episode(voice_typer.parse_command("end episode"))
        spoken = voice_typer.run_episode(voice_typer.parse_command("start episode"))
        if unsure.get("ok") is not False or not spoken.get("ok") or spoken.get("do") != "recorder":
            return fail(f"the voice start did not reach the recorder, or 'end episode' alone was not asked again: "
                        f"{spoken} {unsure}")
        time.sleep(0.4)
        ticker.paused.set()
        closed = wait_until(lambda: (ctl({"cmd": "status"}).get("episode") is None), 4.0, 0.2)
        ticker.paused.clear()
        stalled = episode_json("ep_0001")
        if (stalled["mode"], stalled["task"]) != ("teleop", TASK):
            return fail(f"the voice start was not a teleop episode of the default task: {stalled['mode']} "
                        f"{stalled['task']}")
        if not closed or stalled["outcome"] != "aborted" or "no tap rows" not in str(stalled["end_reason"]):
            return fail(f"ep_0001 did not abort when the rows stopped: {stalled['outcome']} {stalled['end_reason']}")
        print(f"ok    'agent, start episode' (voice_typer) opens a teleop episode of the default task; rows stop "
              f"for 2 s: ep_0001 aborted by itself ({stalled['end_reason']!r}, {stalled['counts']['rows']} rows kept)")

        # an evaluation trial: primed, then opened by "agent, start episode"; refused fields leave the prime
        # alone, and the start uses it up
        time.sleep(0.6)
        trial = {"mode": "policy", "trial": 17, "arrangement": "A07", "policy": "pi05"}
        primed = ctl(dict(trial, cmd="prime"))
        shown = ctl({"cmd": "status"})
        wrong = [ctl({"cmd": "prime", "trial": "17"}), ctl({"cmd": "prime", "arangement": "A07"}),
                 ctl({"cmd": "prime", "mode": "fly"}), ctl({"cmd": "start", "mode": "teleop", "trial": 3.5}),
                 ctl({"cmd": "start", "mode": "teleop", "arrangement": 7}), ctl({"cmd": "start", "policy": ["x"]})]
        spoken = voice_typer.run_episode(voice_typer.parse_command("start episode"))
        used = ctl({"cmd": "status"})
        if not primed.get("ok") or shown.get("prime") != trial or not 590 <= (shown.get("prime_expires_in_s") or 0):
            return fail(f"the prime was not taken, or status does not show it: {primed} {shown}")
        if any(reply.get("ok") is not False for reply in wrong):
            return fail(f"a prime or start with a field of the wrong kind was accepted: {wrong}")
        if not spoken.get("ok") or "trial 17" not in spoken.get("said", "") or used.get("prime") is not None:
            return fail(f"the voice start did not use the prime, or the prime outlived it: {spoken} {used}")
        print(f"ok    prime -> status shows it -> 'agent, start episode' opens the trial ({spoken['said']!r}); "
              "the prime is used up; wrong kinds of field refused without touching it")

        # zero taken again mid-episode (after the arms moved): aborted
        time.sleep(0.5)
        old_zero = sup.driver.zero_fw.copy()
        ticker.jobs.put(sup.driver.capture_zero)
        closed = wait_until(lambda: (ctl({"cmd": "status"}).get("episode") is None), 3.0, 0.1)
        rezeroed = episode_json("ep_0002")
        if np.allclose(old_zero, sup.driver.zero_fw, atol=1e-5):
            return fail("capture_zero did not change the zero: the arms had not moved")
        if not closed or rezeroed["outcome"] != "aborted" or not rezeroed["zero_changed"] \
                or rezeroed["end_reason"] != "zero changed":
            return fail(f"ep_0002 did not abort on the new zero: {rezeroed['outcome']} {rezeroed['end_reason']}")
        if {k: rezeroed[k] for k in trial} != trial or rezeroed["task"] != TASK:
            return fail(f"the primed fields did not reach ep_0002: {[rezeroed[k] for k in trial]}")
        print("ok    zero taken again mid-episode: ep_0002 aborted, zero_changed true; it carries the primed fields")

        # the recorder stopped (SIGTERM) with an episode open: finalized as aborted. Its start names its own
        # mode and trial, which win over the prime's; the arrangement it leaves out comes from the prime.
        time.sleep(0.3)
        ctl({"cmd": "prime", "mode": "guided", "trial": 18, "arrangement": "B"})
        if not ctl({"cmd": "start", "mode": "teleop", "trial": 19}).get("ok"):
            return fail("could not start ep_0003")
        time.sleep(0.5)
        proc.send_signal(signal.SIGTERM)
        try:
            code = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            code = None
        last = episode_json("ep_0003")
        if code != 0 or last["outcome"] != "aborted" or last["end_reason"] != "recorder stopped" \
                or last["end"] is None:
            return fail(f"SIGTERM: exit {code}, ep_0003 {last['outcome']} {last['end_reason']}")
        if (last["mode"], last["trial"], last["arrangement"]) != ("teleop", 19, "B"):
            return fail(f"a start's own fields do not win over the prime's: {last['mode']} {last['trial']} "
                        f"{last['arrangement']}")
        print("ok    SIGTERM with an episode open: ep_0003 finalized as aborted, 'recorder stopped', exit 0; its "
              "start's mode and trial won over the prime, which gave the arrangement")

        # a second recorder on the same ports, its terminal closed (SIGHUP) with an episode open
        proc = subprocess.Popen(
            [sys.executable, str(REPO / "scripts/teleop/recorder.py"), "--root", str(root),
             "--tap-port", str(tap_port), "--control-port", str(ctl_port), "--frame-file", str(frame_file),
             "--profile", str(profile), "--claw-limits", str(claws), "--zero-check", str(zero_file)],
            cwd=REPO, stdout=log, stderr=subprocess.STDOUT, text=True)
        if not wait_until(lambda: proc.poll() is not None or ctl({"cmd": "status"}).get("ok"), 30.0, 0.2) \
                or proc.poll() is not None:
            return fail("the second recorder.py did not come up")
        time.sleep(0.6)                                  # rows and frames reach it
        second = next(p for p in root.iterdir() if p.is_dir() and p != session)
        # record_ctl.py itself: a prime, a start, and an end that returns once the episode is on disk
        cli = []
        for argv in (["prime", "--trial", "21", "--arrangement", "D"], ["start", "--mode", "teleop", "--task", TASK],
                     ["end", "--outcome", "failure", "--notes", "by record_ctl"]):
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                cli.append((record_ctl.main(["--port", str(ctl_port), *argv]), printed.getvalue()))
        by_cli = episode_json("ep_0000", second)
        if [code for code, _ in cli] != [0, 0, 0] or "ok    ep_0000 saved" not in cli[2][1] \
                or [by_cli[k] for k in ("outcome", "trial", "arrangement", "notes")] != ["failure", 21, "D",
                                                                                        "by record_ctl"]:
            return fail(f"record_ctl.py prime/start/end: {cli}")
        print("ok    record_ctl.py prime, start, end: exit 0 each; end returns once ep_0000 is saved, and the "
              "prime reached it")
        hung = ctl({"cmd": "start", "mode": "teleop", "task": TASK})
        if not hung.get("ok"):
            return fail(f"the second recorder refused a start: {hung}")
        time.sleep(0.5)
        proc.send_signal(signal.SIGHUP)
        try:
            code = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            code = None
        last = episode_json("ep_0001", second)
        if code != 0 or last["outcome"] != "aborted" or last["end_reason"] != "recorder stopped" \
                or last["end"] is None:
            return fail(f"SIGHUP: exit {code}, {second.name}/ep_0001 {last['outcome']} {last['end_reason']}")
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            gone = record_ctl.main(["--port", str(ctl_port), "status"])
        if gone != 1 or "not running" not in printed.getvalue():
            return fail(f"record_ctl.py with no recorder: exit {gone}, {printed.getvalue()!r}")
        print("ok    SIGHUP (terminal closed) with an episode open: finalized as aborted, 'recorder stopped', exit 0; "
              "record_ctl.py then says it is not running")
    finally:
        ticker.running = False
        camera.running = False
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        log.close()
        for name, thread in (("ticker", ticker), ("camera", camera)):
            if thread.is_alive():
                thread.join(timeout=2)
            if thread.error is not None and len(failures) == before:
                failures.append(f"the {name} thread failed: {thread.error!r}")


def test_zero_check(tmp: Path, failures: list[str]) -> None:
    """c. The preregistered rule, per arm and overall, and a torso that leans."""
    cases = (((0, 0, 0), "holds"), ((0, 3, -3), "holds"), ((0, 15, 15), "rejected"), ((0, 12, 19), "rejected"),
             ((0, 15, 1), "report"), ((0, 8, 8), "report"), ((0, -15, -15), "report"), ((0, 20, 15), "report"),
             ((5, -5, 5), "holds"),            # the torso leans 5 deg; both arms hang along it
             ((-4, 19, 11), "rejected"))       # leaning the other way: 15 deg out from the torso on both
    wrong = [(readings, zero_check.check(*readings)["decision"], want) for readings, want in cases
             if zero_check.check(*readings)["decision"] != want]
    # Readings at 0.1 deg steps whose angle from the torso is exactly a boundary are decided as that
    # boundary, whatever the float sum says (-4.9 + 1.9 is -3.0000000000000004, still "within 3 deg").
    for tenths in range(-60, 61):
        torso = tenths / 10
        for angle, want in ((3.0, "holds"), (-3.0, "holds"), (12.0, "rejected"), (19.0, "rejected"),
                            (3.1, "report"), (-3.1, "report"), (11.9, "report"), (19.1, "report")):
            left, right = round(angle - torso, 1), round(angle + torso, 1)     # left + torso, right - torso
            got = zero_check.check(torso, left, right)
            if got["decision"] != want or got["relative_to_torso_deg"] != {"left": angle, "right": angle}:
                wrong.append(((torso, left, right), got["decision"], want))
    out = tmp / "zero_check_test.json"
    with contextlib.redirect_stdout(io.StringIO()) as printed:
        zero_check.main(["--torso-deg", "0", "--left-upper-arm-deg", "14.5", "--right-upper-arm-deg", "16",
                         "--method", "inclinometer", "--notes", "test", "--out", str(out)])
    written = json.loads(out.read_text())
    keys = {"date", "readings_deg", "relative_to_torso_deg", "decisions", "decision", "rule", "profile_sha256",
            "zero_fix_applied", "method", "notes"}
    if wrong:
        failures.append(f"zero_check decides {len(wrong)} wrongly (readings, got, want): {wrong[:6]}")
    elif not keys <= set(written) or written["decision"] != "rejected" or "REJECTED" not in printed.getvalue():
        failures.append(f"zero_check wrote or said the wrong thing: {written}")
    else:
        print("ok    zero_check: +-3 deg holds, 12-19 deg out rejected, anything else or a split report; "
              "a leaning torso is taken out; readings that sum exactly to a boundary are decided as it")


def test_full_disk(tmp: Path, gravity: GravityModel, failures: list[str]) -> None:
    """e. A full disk (recordings fill it) never stops teleop: run_teleop's own teleop log stops instead, once."""
    sup = supervisor(tmp, gravity, scratch_profile(tmp, "full"), None)
    said: list[str] = []
    sup._say = said.append

    def stopped() -> int:
        return sum(text.startswith("Teleop log stopped") for text in said)

    problem = ""
    with contextlib.redirect_stdout(io.StringIO()):
        arm(sup, 1000.0)
        sup.log_file, sup.log_last = open("/dev/full", "w"), -math.inf      # _open_log's file, on a full disk
        for k in range(1, 200):
            try:
                sup.tick(1000.0 + 0.02 * k, packet(0.02 * k), 0.0, {})
            except Exception as exc:
                problem = f"tick {k} raised {exc!r} (motors {sup.motors}, {sup.state})"
                break
        cycle = (sup.log_file, sup.state, sup.motors, stopped())
        sup.log_file = open("/dev/full", "w")
        sup._log_row({"tune": {"what": "power"}, "said": "x" * 10000})          # past the write buffer
        row = (sup.log_file, stopped())
        sup.log_file = open("/dev/full", "w")
        sup._log_row({"tune": {"what": "undo"}})                                 # buffered: fails at the close
        try:
            sup._stop("triple X")
        except Exception as exc:
            problem = problem or f"the stop raised {exc!r}"
    if problem:
        failures.append(f"a full disk under the teleop log: {problem}")
    elif cycle != (None, "ARMED", True, 1) or row != (None, 2) or stopped() != 3 or sup.log_file is not None:
        failures.append(f"a full disk under the teleop log: after the ticks {cycle}, after a tuning line {row}, "
                        f"said {said}")
    else:
        print("ok    a full disk under the teleop log (/dev/full): 200 armed ticks carry on, the log is closed and "
              "said once; the same for a tuning line and for the close at a stop")


def test_in_process(tmp: Path, failures: list[str]) -> None:
    """d. A Recorder in this process, fed by hand: a prime used once, one that expires, and low disk."""
    frame_file = tmp / "inproc_camera.jpg"
    jpeg = cv2.imencode(".jpg", np.zeros((48, 128, 3), np.uint8))[1].tobytes()
    tap_port = free_port()
    args = recorder.parse_args(["--root", str(tmp / "inproc"), "--tap-port", str(tap_port),
                                "--control-port", str(free_port()), "--frame-file", str(frame_file),
                                "--profile", str(scratch_profile(tmp, "inproc")), "--task", TASK,
                                "--claw-limits", str(tmp / "no_claw_limits.json"),
                                "--zero-check", str(tmp / "no_zero_check.json")])
    feeder = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    seq = iter(range(10 ** 6))
    with contextlib.redirect_stdout(io.StringIO()):
        rec = recorder.Recorder(args)

    def fresh() -> None:
        """A tap row and a new frame, as the two streams bring them, taken in."""
        row = dict(FAKE_ROW, seq=next(seq), t=time.monotonic(), cam_mode="still")
        feeder.sendto(json.dumps(row).encode(), ("127.0.0.1", tap_port))
        tmp_frame = frame_file.with_name(frame_file.name + ".tmp")
        tmp_frame.write_bytes(jpeg)
        os.replace(tmp_frame, frame_file)
        time.sleep(0.01)
        rec._take_rows()
        rec._poll_frame()

    def command(cmd: dict) -> dict:
        """As _take_commands runs one: the reply, then whatever was held back until it was out."""
        reply = rec._command(cmd)
        rec._after_reply()
        return reply

    def episode(name: str) -> dict:
        return json.loads((rec.session_dir / name / "episode.json").read_text())

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fresh()
            command({"cmd": "prime", "trial": 6, "arrangement": "C"})
            once = [command({"cmd": "start"}), command({"cmd": "status"})]
            # the end's reply comes first: when it is built, nothing is saved yet and nothing more is written
            ending = rec._command({"cmd": "end", "outcome": "success"})
            unsaved = (episode("ep_0000")["outcome"], rec.episode, len(rec.after_reply))
            rec._after_reply()
            once.append(ending)
            fresh()
            once += [command({"cmd": "start"}), command({"cmd": "end", "outcome": "failure"})]
            command({"cmd": "prime", "trial": 7})
            kept = command({"cmd": "status"})
            rec.prime_at -= recorder.PRIME_S + 1          # ten minutes pass, and nobody starts
            expired = command({"cmd": "status"})
            fresh()
            late = command({"cmd": "start"})
            cleared = [command({"cmd": "prime", "trial": 8}), command({"cmd": "prime"}), command({"cmd": "status"})]
            rec._free_bytes = lambda: 1.5e9               # the disk fills while ep_0002 records
            rec.next_disk_check = 0.0
            fresh()
            rec._watch(time.monotonic())
            low = [rec.episode, command({"cmd": "status"})]
            fresh()
            low.append(command({"cmd": "start"}))
        if once[1].get("prime") is not None or not all(reply.get("ok") for reply in once):
            failures.append(f"a primed start went wrong, or the prime outlived it: {once}")
        elif unsaved != (None, None, 1) or episode("ep_0000")["outcome"] != "success":
            failures.append(f"the end's save did not wait for its reply (outcome, open episode, held back: "
                            f"{unsaved}), or never came ({episode('ep_0000')['outcome']})")
        elif (episode("ep_0000")["trial"], episode("ep_0000")["arrangement"]) != (6, "C") \
                or (episode("ep_0001")["trial"], episode("ep_0001")["arrangement"]) != (None, None):
            failures.append("a prime did not reach the next start, or reached the one after it too")
        elif kept.get("prime") != {"trial": 7} or expired.get("prime") is not None or not late.get("ok") \
                or episode("ep_0002")["trial"] is not None:
            failures.append(f"a prime did not expire after {recorder.PRIME_S:g} s unused: {kept} {expired} {late}")
        elif cleared[1].get("said") != "prime cleared" or cleared[2].get("prime") is not None:
            failures.append(f"a prime with no fields does not clear the prime: {cleared}")
        elif low[0] is not None or episode("ep_0002")["outcome"] != "aborted" \
                or episode("ep_0002")["end_reason"] != "low disk" \
                or not any("below the 2 GB floor" in w for w in episode("ep_0002")["warnings"]):
            failures.append(f"low disk did not end ep_0002 as aborted, 'low disk': {episode('ep_0002')}")
        elif low[2].get("ok") is not False or "GB free" not in low[2].get("said", "") \
                or not any("below the 2 GB floor" in w for w in low[1].get("warnings", [])):
            failures.append(f"a start below the free-space floor was not refused, or status hides it: {low[1:]}")
        else:
            print("ok    in process: an end is answered before its save; a prime goes to the next start only, "
                  "expires after 10 min unused, and an empty prime clears it; 1.5 GB free mid-episode ends it as "
                  f"aborted ('low disk') and refuses starts ({low[2]['said']!r})")
    finally:
        rec.tap.close()
        rec.ctl.close()
        feeder.close()


def main() -> int:
    failures: list[str] = []
    gravity = GravityModel()
    with tempfile.TemporaryDirectory(prefix="test_recorder_") as tmp:
        test_no_change(Path(tmp), gravity, failures)
        test_tap_port(failures)
        test_servo_bytes(failures)
        test_tap_values(Path(tmp), gravity, failures)
        test_recorder(Path(tmp), gravity, failures)
        test_zero_check(Path(tmp), failures)
        test_in_process(Path(tmp), failures)
        test_full_disk(Path(tmp), gravity, failures)
    print()
    for failure in failures:
        print("FAIL  " + failure)
    print("recording is sound" if not failures else f"{len(failures)} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
