"""The claw and camera servos on the Arduino Nano (scripts/arduino/claw_controller).

Text protocol at 115200: `P <pin> <us>` (1000..2000, slewed by the Nano),
`OFF` (all limp), `STATUS`. Pins: D3 left claw, D7 right claw, D9 camera pan
(left-right), D5 camera tilt (up-down). Positions marked in the headset setup are kept in
configs/claw_limits.json (tools/claw_jog.py writes the same file).

Everything here is optional: without the Nano, teleop runs exactly as before.
Writes never block the arm loop, and a vanished port is simply closed.
"""

from __future__ import annotations

import glob
import math
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LIMITS_PATH = REPO_ROOT / "configs" / "claw_limits.json"
MIN_US, MAX_US = 500, 2500   # widest any servo allows; each Servo has its own


@dataclass(frozen=True)
class Servo:
    key: str        # key in claw_limits.json
    pin: int
    label: str
    marks: tuple    # (field, what the operator should set) in the order asked
    min_us: int = 1000
    max_us: int = 2000


SERVOS = (
    # The first mark of each is its home, where it sits when the rig is at rest.
    Servo("D3", 3, "LEFT CLAW", (("closed_us", "CLOSED - home (jaws just touch, not squeezing)"),
                                 ("open_us", "fully OPEN"))),
    Servo("D7", 7, "RIGHT CLAW", (("closed_us", "CLOSED - home (jaws just touch, not squeezing)"),
                                  ("open_us", "fully OPEN"))),
    Servo("D9", 9, "CAMERA PAN", (("center_us", "CENTER - home (facing the robot's front)"),
                                  ("left_us", "furthest LEFT it should go"),
                                  ("right_us", "furthest RIGHT it should go")), 500, 2500),
    Servo("D5", 5, "CAMERA TILT", (("level_us", "LEVEL - home (looking straight ahead)"),
                                   ("up_us", "furthest UP it should go"),
                                   ("down_us", "furthest DOWN it should go")), 500, 2500),
)
ROLES = {"D3": "left_claw", "D7": "right_claw", "D9": "camera_pan", "D5": "camera_tilt"}


def load_limits(path: Path = LIMITS_PATH) -> dict:
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        data = {}
    data.setdefault("claws", {})
    return data


def save_limits(data: dict, path: Path = LIMITS_PATH) -> None:
    data["saved"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    data["note"] = ("Pulse widths (us) for scripts/arduino/claw_controller.ino; written by the "
                    "headset setup (triple Y -> B) or tools/claw_jog.py.")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2)
    tmp.replace(path)


def ch340_ports() -> list[tuple[str, tuple[str, str]]]:
    """Every CH340 serial port, straight from the kernel: [(/dev/ttyUSBn, (USB port, devnum))].

    Not /dev/serial/by-id: CH340s have no serial number, so two of them (the claw Nano
    and the IMU, 29 Sep) share ONE by-id name, which pointed at the IMU - the Nano was
    never found and the camera and claws went dead.
    """
    found = []
    for tty in sorted(glob.glob("/sys/class/tty/ttyUSB*")):
        interface = Path(tty, "device").resolve().parent          # .../3-1.4:1.0
        usb = interface.parent                                   # .../3-1.4
        try:
            if (usb / "idVendor").read_text().strip() != "1a86":
                continue
            devnum = (usb / "devnum").read_text().strip()        # changes when replugged
        except OSError:
            continue
        found.append((f"/dev/{Path(tty).name}", (usb.name, devnum)))
    return found


class ServoLink:
    """Opens the Nano in the background (opening resets it for ~2 s)."""

    def __init__(self):
        self.port = None
        self.status = "claw/camera Nano: looking for it"
        self.lock = threading.Lock()
        self._closed = False
        self.not_nano: set = set()        # (USB port, device number) of CH340s that are not it
        # pin -> the last pulse (us) a move() sent successfully since start-up, for run_teleop's
        # record tap. Bookkeeping only: nothing reads it to decide what to send. Kept through a
        # re-arm, OFF and a Nano reset alike, so it is a record of commands, not of what a servo holds.
        self.last_us: dict[int, int] = {}
        threading.Thread(target=self._open, daemon=True).start()

    @property
    def ready(self) -> bool:
        return self.port is not None

    def _open(self) -> None:
        """Keep looking until the Nano turns up: plugged in late, or back after a drop,
        it is picked up within seconds, with no restart."""
        told = False
        while not self._closed:
            if self._try_open():
                return
            if not told:
                told = True
                print(f"Servos: {self.status}; still looking every 3 s", flush=True)
            time.sleep(3.0)

    def _try_open(self) -> bool:
        import serial  # only needed when the Nano is actually used
        for path, ident in ch340_ports():
            if ident in self.not_nano:
                continue                      # answered wrongly before: leave it alone until replugged
            try:
                # exclusive: a port someone else holds (the IMU, in sensor_share.py) fails here
                port = serial.Serial(path, 115200, timeout=0, write_timeout=0, exclusive=True)
            except (OSError, serial.SerialException):
                continue
            time.sleep(2.2)
            reply = port.read(512).decode(errors="replace")
            if "_READY" not in reply:
                port.write(b"STATUS\n")
                time.sleep(0.3)
                reply += port.read(512).decode(errors="replace")
            if "_READY" in reply or "D3 " in reply:
                with self.lock:
                    self.port = port
                self.status = f"claw/camera Nano on {path} (USB {ident[0]})"
                print(f"Servos: {self.status}", flush=True)
                return True
            port.close()
            self.not_nano.add(ident)
        self.status = "claw/camera Nano not found (USB CH340 with claw_controller)"
        return False

    def _send(self, text: str) -> bool:
        with self.lock:
            if self.port is None:
                return False
            try:
                self.port.write((text + "\n").encode())
                self.port.read(4096)          # drain the Nano's replies
                return True
            except Exception as exc:          # unplugged: look for it again
                self.status = f"claw/camera Nano lost ({exc})"
                print(f"Servos: {self.status}", flush=True)
                try:
                    self.port.close()
                finally:
                    self.port = None
                threading.Thread(target=self._open, daemon=True).start()
                return False

    def move(self, pin: int, us: int) -> bool:
        us = max(MIN_US, min(MAX_US, int(us)))
        ok = self._send(f"P {pin} {us}")
        if ok:
            self.last_us[pin] = us            # what was sent, for the record tap only
        return ok

    def off(self) -> bool:
        return self._send("OFF")

    def close(self) -> None:
        self._closed = True
        self.off()
        with self.lock:
            if self.port is not None:
                self.port.close()
                self.port = None


SG90_DEG_PER_US = 0.09   # ~180 deg over 500..2500 us


class _Sender:
    """Rate-limited writes to the Nano: a pulse only when it changed enough, and not too often."""

    SEND_EVERY = 0.04   # s between writes to one servo
    MIN_CHANGE = 6      # us worth sending

    def __init__(self, link: ServoLink):
        self.link = link
        self.sent: dict[int, tuple[int, float]] = {}

    def send(self, pin: int, us: float, now: float) -> None:
        us = int(round(us))
        last = self.sent.get(pin)
        if last and (abs(last[0] - us) < self.MIN_CHANGE or now - last[1] < self.SEND_EVERY):
            return
        if self.link.move(pin, us):
            self.sent[pin] = (us, now)


class ServoTeleop:
    """The claws while teleop is armed.

    Claw k (0 = left arm, 1 = right arm, after any hand swap): while that arm's grip
    is held, its trigger sets the claw between the saved open (trigger released) and
    closed (fully squeezed) pulses; letting go of the grip keeps the claw where it is,
    so nothing held is dropped. The camera is CameraAim's, in every state.
    """

    def __init__(self, link: ServoLink):
        self.link = link
        self.limits = {}
        self.out = _Sender(link)

    def start(self, head_rotation=None) -> None:
        self.limits = load_limits().get("claws", {})
        self.out = _Sender(self.link)

    def stop(self) -> None:
        pass                                              # claws keep what they hold

    def update(self, now: float, grips, triggers, head_rotation=None) -> None:
        if not self.link.ready:
            return
        for k, key in enumerate(("D3", "D7")):
            claw = self.limits.get(key, {})
            if grips[k] and "open_us" in claw and "closed_us" in claw:
                t = min(1.0, max(0.0, (float(triggers[k]) - 0.05) / 0.9))
                self.out.send(int(key[1:]), claw["open_us"] + t * (claw["closed_us"] - claw["open_us"]), now)


class CameraAim:
    """The stereo camera's pan/tilt, in every state but the claws-and-camera setup menu.

    track (the default): keep the operator centred in the picture, from person_track.py.
    head:  follow the headset, starting from wherever the camera is (it never jumps).
    still: hold where it is - for debugging, when the picture must not move.

    A x2 in the headset switches track <-> head; "agent, camera still / track / head"
    picks one. Angles are degrees, + = left / up, around the saved center / level,
    clamped to the saved extremes.
    """

    MODES = ("track", "head", "still")
    # Tracking aims at where the operator IS (camera angle when the frame was taken +
    # where they were in it), not "a bit more that way" from where the camera is now:
    # a detection is ~0.35 s old, and correcting the present by it overshot, so the
    # camera swung from side to side and blurred the frames it was tracking with.
    DEADBAND = 6.0      # deg: a target this close to the last one is not worth chasing
    TRACK_RATE = 20.0   # deg/s the camera turns toward the operator
    # Tilt too, so a seated operator is centred like a standing one: it aims BELOW_TOP
    # under the top of their head (person_track's "top"), not at the head point, whose
    # height jumps about with a headset on (face, headset and box guess take turns).
    TRACK_TILT = True
    BELOW_TOP = 12.0    # deg: puts the face a little above the middle of the picture
    AGREE = 3           # of the last 5 readings (2 s) must agree before the camera moves...
    AGREE_DEG = 8.0     # ...to within this of their median; the rest are outliers
    # Until the extremes are recorded (Y x3 -> B, claws + camera setup), stay well inside
    # the servo's range: an SG90 driven into its mount's stop buzzes and shakes.
    SAFE_PAN = 35.0
    SAFE_TILT = 15.0
    LOST_S = 8.0        # nobody seen this long: drift back to straight ahead
    HOME_RATE = 10.0    # deg/s, that drift

    def __init__(self, link: ServoLink):
        self.link = link
        self.limits = load_limits().get("claws", {})
        self.out = _Sender(link)
        self.mode = "track"
        self.angle = (0.0, 0.0)       # yaw, pitch last sent (what the servo can reach)
        self.hold = (0.0, 0.0)        # where head mode started from
        self.head_ref = None
        self.person_t = None
        self.seen_at = 0.0
        self.last = 0.0
        self.goal = None              # tracking: where the operator is, from the robot
        self.readings: list[tuple[float, float]] = []    # (time, world yaw), the last few
        self.history: list[tuple[float, tuple[float, float]]] = []   # (wall time, angle)

    def set_mode(self, mode: str) -> str:
        if mode in self.MODES and mode != self.mode:
            self.mode = mode
            self.hold = self.angle    # head mode carries on from here...
            self.head_ref = None      # ...measured from where the head is now
            self.seen_at = self.last  # tracking does not drift home straight away
        return self.mode

    def toggle(self) -> str:
        """A x2: track <-> head (still -> track)."""
        return self.set_mode("head" if self.mode == "track" else "track")

    def update(self, now: float, head_rotation, person: dict | None) -> None:
        dt = min(0.2, now - self.last) if self.last else 0.0
        self.last = now
        if not self.link.ready or self.mode == "still":
            return
        if self.mode == "head":
            if head_rotation is None:
                return
            if self.head_ref is None:
                self.head_ref = head_rotation.copy()
            rel = self.head_ref.T @ head_rotation         # head turn since head mode began
            f = rel[:, 0]                                  # where the face points (+X forward)
            target = (self.hold[0] + math.degrees(math.atan2(f[1], f[0])),
                      self.hold[1] + math.degrees(math.atan2(f[2], math.hypot(f[0], f[1]))))
        else:
            if person and person.get("seen") and person.get("t") != self.person_t:
                self.person_t = person.get("t")            # each detection counts once
                self.seen_at = now
                self.readings = [r for r in self.readings if now - r[0] < 2.0][-4:] + [(now, *self.aim_at(person))]
                goal, moved = list(self.goal or self.angle), False
                for axis in (0, 1):                         # pan and tilt settle separately
                    values = sorted(r[1 + axis] for r in self.readings)
                    middle = values[len(values) // 2]
                    agree = [v for v in values if abs(v - middle) <= self.AGREE_DEG]
                    steady = sum(agree) / len(agree)
                    if len(agree) >= self.AGREE and (self.goal is None or abs(steady - goal[axis]) > self.DEADBAND):
                        goal[axis], moved = steady, True
                if moved:
                    self.goal = tuple(goal)
            if now - self.seen_at > self.LOST_S:
                self.goal = (0.0, 0.0)                      # nobody: back to straight ahead
                rate = self.HOME_RATE
            else:
                rate = self.TRACK_RATE
            goal = self.goal or self.angle
            turn = rate * dt                                # smoothly, never in jumps
            target = tuple(a + max(-turn, min(turn, g - a)) for a, g in zip(self.angle, goal))
        self.angle = self._aim(now, *target)
        self.history.append((time.time(), self.angle))
        cutoff = time.time() - 3.0
        while self.history and self.history[0][0] < cutoff:
            self.history.pop(0)

    def angle_at(self, when: float | None) -> tuple[float, float]:
        """Where the camera pointed at wall time `when` (the frame's capture time)."""
        if when is None or not self.history:
            return self.angle
        best = min(self.history, key=lambda item: abs(item[0] - when))
        return best[1]

    def world(self, person: dict) -> tuple[float, float]:
        """Where the operator is from the robot, not from the (turned) camera: the
        camera's angle when their frame was taken, plus where they were in it."""
        at = self.angle_at(person.get("captured"))
        return at[0] + float(person["yaw"]), at[1] + float(person["pitch"])

    def aim_at(self, person: dict) -> tuple[float, float]:
        """Where tracking points the camera, from the robot: across, at the operator's
        head; up-down, BELOW_TOP under the top of it (a tracker without "top": the head)."""
        at = self.angle_at(person.get("captured"))
        top = person.get("top")
        pitch = float(top) - self.BELOW_TOP if top is not None else float(person["pitch"])
        return at[0] + float(person["yaw"]), (at[1] + pitch) if self.TRACK_TILT else 0.0

    def _aim(self, now: float, yaw: float, pitch: float) -> tuple[float, float]:
        # no recorded extremes: a conservative range, never the servo's full swing
        if not {"left_us", "right_us"} <= set(self.limits.get("D9", {})):
            yaw = max(-self.SAFE_PAN, min(self.SAFE_PAN, yaw))
        if not {"up_us", "down_us"} <= set(self.limits.get("D5", {})):
            pitch = max(-self.SAFE_TILT, min(self.SAFE_TILT, pitch))
        reached = {"D9": yaw, "D5": pitch}
        for key, center, pos_end, neg_end, angle in (("D9", "center_us", "left_us", "right_us", yaw),
                                                      ("D5", "level_us", "up_us", "down_us", pitch)):
            cam = self.limits.get(key, {})
            if center not in cam:
                continue
            c = cam[center]
            # which way is "left"/"up" in pulse terms comes from the marked extremes;
            # unmarked, assume increasing pulses and the servo's own 500..2500
            hi = cam.get(pos_end, 2500)
            lo = cam.get(neg_end, 500 if hi > c else 2500)
            sign = 1 if hi >= c else -1
            us = min(max(c + sign * angle / SG90_DEG_PER_US, min(lo, hi)), max(lo, hi))
            reached[key] = sign * (us - c) * SG90_DEG_PER_US     # what the servo can actually do
            self.out.send(int(key[1:]), us, now)
        return reached["D9"], reached["D5"]
