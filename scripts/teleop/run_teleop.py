"""Quest teleop for the Berkeley Humanoid Lite arms, with per-joint power.

  .venv/bin/python scripts/teleop/quest_bridge.py        # headset page + status relay
  .venv/bin/python scripts/teleop/run_teleop.py          # arms on can0/can1
  .venv/bin/python scripts/teleop/run_teleop.py --sim    # no robot, same headset flow
  .venv/bin/python scripts/teleop/run_teleop.py --record-tap   # + one row per cycle to recorder.py (UDP 11014)

Headset:
  triple-click X      arm / stop the motors (stop = damping)
  hold grip           that arm follows the controller (dead-man)
  triple-click Y      calibration menu (motors stopped first): A = arm power,
                      B = claws + camera servos (right stick jogs, A records)
  A / B               answer calibration prompts
  right stick press   re-center the status panel

Each joint gets gravity feed-forward (PDO-3) and a torque limit that follows its
gravity load, from configs/arm_power_profile.json (arm_power.py). Start with
both arms hanging straight down: that pose becomes zero.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import signal
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pink
import pinocchio as pin
import qpsolvers
from loop_rate_limiters import RateLimiter
from pink import solve_ik
from pink.limits import ConfigurationLimit, VelocityLimit
from pink.tasks import FrameTask

from arm_calibration import CalInputs, PowerCalibration
from servo_calibration import CalibrationChooser, ServoCalibration
from servos import CameraAim, ServoLink, ServoTeleop
from reach_check import ReachCheck
from arm_driver import ENCODER_FAULT, ArmDriver, Mode, SimArmDriver, describe_state
from arm_power import (
    ABSOLUTE_MAX_TORQUE, ARM_JOINTS, KIND_DEFAULTS, N_JOINTS, PROFILE_PATH, URDF_PATH,
    GravityModel, PowerProfile, joint_index,
)

np.set_printoptions(precision=2, suppress=True)

ARM_SLICES = (slice(0, 5), slice(5, 10))
EVENTS = ("x3", "y3", "a", "b", "cam", "camtrack", "camhead", "camstill")
PERSON_PORT = 11011   # person_track.py -> here: where the operator is, from the camera
FACE_PORT = 11010     # here -> face_display.py: where the operator is, from the robot
TUNE_PORT = 11012     # tools/teleop_tune.py, a Claude session, "agent, more power ..." -> here: live changes
RECORD_TAP_PORT = 11014   # here -> recorder.py: one JSON row per control cycle, only with --record-tap
TERMINAL_LINES = collections.deque(maxlen=200)


class LogTee:
    """Pass terminal output through, keeping the recent lines for the headset.

    The headset panel is fourteen lines tall, so anything repetitive crowds out
    everything worth reading: identical lines are folded into a count, and the
    once-a-second heartbeat skips this entirely (see local_only).
    """

    def __init__(self, stream):
        self.stream = stream
        self.pending = ""

    def write(self, text: str) -> int:
        written = self.stream.write(HEARTBEAT.close() + text)
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            line = line.rstrip()[:130]
            if not line:
                continue
            if TERMINAL_LINES:
                last = TERMINAL_LINES[-1]
                base, marker, count = last.rpartition("  x")
                if last == line:
                    TERMINAL_LINES[-1] = f"{line}  x2"
                    continue
                if marker and base == line and count.isdigit():
                    TERMINAL_LINES[-1] = f"{base}  x{int(count) + 1}"
                    continue
            TERMINAL_LINES.append(line)
        return written

    def flush(self) -> None:
        self.stream.flush()

    def isatty(self) -> bool:
        return self.stream.isatty()


class Heartbeat:
    """The once-a-second status line, rewritten in place instead of scrolling.

    A new line every second is thousands of identical `STOPPED ... tau/lim ...`
    lines an hour, which is all anyone reading that terminal can see - including
    through the headset, where this terminal is the shared screen. It keeps one
    line and steps aside whenever something with anything to say comes along.
    """

    def __init__(self):
        self.open = False

    def close(self) -> str:
        """What to print before ordinary output, to take the live line down."""
        if not self.open:
            return ""
        self.open = False
        return "\r\033[K"


HEARTBEAT = Heartbeat()


def local_only(text: str) -> None:
    """To the PC terminal, not to the headset.

    The status panel already shows the state, the rate and every joint's torque
    against its limit, so echoing the same heartbeat into the log panel filled
    all fourteen lines with `tau/lim ...` and hid the messages that matter.
    """
    stream = sys.stdout
    raw = stream.stream if isinstance(stream, LogTee) else stream
    try:
        live = raw.isatty()
    except (AttributeError, ValueError):
        live = False
    if live:                      # one line, rewritten; a log file still gets lines
        raw.write("\r\033[K" + text)
        raw.flush()
        HEARTBEAT.open = True
        return
    raw.write(text + "\n")
    raw.flush()


class TeleopIkSolver:
    """Controller motion -> arm joint targets.

    While a grip is held, that hand frame follows the controller's motion since
    the grip was pressed, starting from where the robot hand was at that moment.
    A released arm holds still. The IK integrates the commanded configuration,
    capped at max_speed and at max_lead ahead of the measured joints.
    """

    FRAMES = ("arm_left_elbow_roll", "arm_right_elbow_roll")  # hand frames

    def __init__(self, urdf_path: Path = URDF_PATH, orientation_cost: float = 0.1,
                 max_speed: float = 1.5, max_lead: float = 0.35):
        full = pin.buildModelFromUrdf(str(urdf_path))
        others = [full.getJointId(name) for name in full.names[1:] if not name.startswith("arm_")]
        self.model = pin.buildReducedModel(full, others, pin.neutral(full))
        names = list(self.model.names[1:])
        if names != [joint.urdf for joint in ARM_JOINTS]:
            raise RuntimeError(f"URDF arm joint order differs from ARM_JOINTS: {names}")
        self.data = self.model.createData()
        self.model.velocityLimit = np.full(self.model.nv, max_speed)
        self.max_speed = max_speed
        self.max_lead = max_lead
        self.lower = self.model.lowerPositionLimit.copy()
        self.upper = self.model.upperPositionLimit.copy()
        self.limits = [ConfigurationLimit(self.model), VelocityLimit(self.model)]
        self.tasks = [
            FrameTask(frame, position_cost=1.0, orientation_cost=orientation_cost, lm_damping=1.0)
            for frame in self.FRAMES
        ]
        self.solver = "quadprog" if "quadprog" in qpsolvers.available_solvers else qpsolvers.available_solvers[0]
        self.held = [False, False]
        self.vr_ref = [None, None]
        self.smooth = [None, None]   # low-passed controller pose: hand tremor is not a command
        # The headset's room frame keeps the heading it had when the session began;
        # set_heading() turns hand motion into the operator's own frame at arming.
        self.heading = np.eye(3)
        self.last_d_pos = [None, None]
        self.ee_ref = [None, None]
        self.targets = [None, None]

    def reset(self) -> None:
        self.held = [False, False]

    def set_heading(self, head_rotation) -> float:
        """Operator forward = where the head faces now (yaw only). Returns that yaw, deg."""
        if head_rotation is None:
            self.heading = np.eye(3)
            return 0.0
        yaw = math.atan2(head_rotation[1, 0], head_rotation[0, 0])
        c, s_ = math.cos(-yaw), math.sin(-yaw)
        self.heading = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
        return math.degrees(yaw)

    def hand_positions(self, q) -> list:
        configuration = pink.Configuration(self.model, self.data, np.asarray(q, dtype=float))
        return [configuration.get_transform_frame_to_world(f).translation.copy() for f in self.FRAMES]

    def set_range(self, lower: np.ndarray, upper: np.ndarray) -> None:
        """Teleop soft limits (profile range_lower / range_upper, else URDF)."""
        self.lower = np.array(lower, dtype=float)
        self.upper = np.array(upper, dtype=float)

    # Facing the robot (hands swapped), it moves like your reflection: reach toward it and
    # its hand comes toward you (its forward is yours), move a hand to your right and its
    # hand goes to its left; up is up. Only the sideways axis flips. Until 30 Sep this also
    # flipped forward (a 180 deg turn), so reaching toward the robot sent its hand away,
    # and bringing the hands inward - which carries them forward too - read as backward.
    FACING = np.diag([1.0, -1.0, 1.0])
    SMOOTH_TAU = 0.12   # s; filters tremor, still follows a deliberate move

    def update(self, q_cmd, q_meas, grips, hands, dt: float, motion_scale: float = 1.0,
               facing: bool = False) -> np.ndarray:
        q = np.array(q_cmd, dtype=float)
        # A joint that starts slightly outside its URDF range may stay there,
        # but the IK can only move it back toward the range.
        self.model.lowerPositionLimit = np.minimum(self.lower, q)
        self.model.upperPositionLimit = np.maximum(self.upper, q)
        configuration = pink.Configuration(self.model, self.data, q)
        for k, frame in enumerate(self.FRAMES):
            current = configuration.get_transform_frame_to_world(frame)
            if grips[k] and hands[k] is not None:
                if not self.held[k] or self.smooth[k] is None:
                    self.smooth[k] = hands[k].copy()
                else:
                    a = 1.0 - math.exp(-max(dt, 1e-3) / self.SMOOTH_TAU)
                    step = pin.log6(self.smooth[k].actInv(hands[k]))
                    self.smooth[k] = self.smooth[k] * pin.exp6(step * a)
                hand = self.smooth[k]
                if not self.held[k]:
                    self.vr_ref[k] = hand.copy()
                    self.ee_ref[k] = current.copy()
                    self.held[k] = True
                d_rot = hand.rotation @ self.vr_ref[k].rotation.T
                d_pos = (hand.translation - self.vr_ref[k].translation) * motion_scale
                d_pos = self.heading @ d_pos
                d_rot = self.heading @ d_rot @ self.heading.T
                self.last_d_pos[k] = d_pos.copy()
                if facing:
                    d_pos = self.FACING @ d_pos
                    d_rot = self.FACING @ d_rot @ self.FACING
                target = pin.SE3(d_rot @ self.ee_ref[k].rotation, self.ee_ref[k].translation + d_pos)
            else:
                self.held[k] = False
                target = current
            self.tasks[k].set_target(target)
            self.targets[k] = target
        if not any(self.held):
            return q
        dt = float(np.clip(dt, 0.005, 0.1))
        try:
            velocity = solve_ik(configuration, self.tasks, dt, solver=self.solver,
                                limits=self.limits, safety_break=False)
            q_new = pin.integrate(self.model, q, velocity * dt)
        except Exception as exc:  # infeasible QP: hold this cycle
            print(f"IK failed, holding: {exc}", flush=True)
            q_new = q.copy()
        for k, part in enumerate(ARM_SLICES):
            if not self.held[k]:
                q_new[part] = q[part]
        step = self.max_speed * dt
        q_new = np.clip(q_new, q - step, q + step)
        known = np.isfinite(q_meas)
        q_new[known] = np.clip(q_new[known], q_meas[known] - self.max_lead, q_meas[known] + self.max_lead)
        return np.clip(q_new, self.model.lowerPositionLimit, self.model.upperPositionLimit)


class MeshcatView:
    """Browser view of the measured arms and the IK hand targets, 10 Hz."""

    def __init__(self, urdf_path: Path = URDF_PATH):
        import meshcat_shapes
        from pink.visualization import start_meshcat_visualizer

        self.robot = pin.RobotWrapper.BuildFromURDF(
            str(urdf_path), package_dirs=[str(Path(urdf_path).parent)],
            root_joint=pin.JointModelFreeFlyer())
        self.visualizer = start_meshcat_visualizer(self.robot, open=False)
        self.viewer = self.visualizer.viewer
        self.q = pin.neutral(self.robot.model)
        self.q[2] = 0.5
        model = self.robot.model
        self.idx = [model.joints[model.getJointId(joint.urdf)].idx_q for joint in ARM_JOINTS]
        for name in ("left_target", "right_target"):
            meshcat_shapes.frame(self.viewer[name], opacity=0.5)
        self.last = 0.0
        print(f"Meshcat: {self.viewer.url()}", flush=True)

    def show(self, q_arm: np.ndarray, targets, now: float) -> None:
        if now - self.last < 0.1 or not np.all(np.isfinite(q_arm)):
            return
        self.last = now
        self.q[self.idx] = q_arm
        self.visualizer.display(self.q)
        for name, target in zip(("left_target", "right_target"), targets):
            if target is not None:
                pose = target.homogeneous.copy()
                pose[2, 3] += 0.5
                self.viewer[name].set_transform(pose)


class OperatorLink:
    """Headset packets in (UDP from quest_bridge.py), status for the headset out.

    Button presses arrive as monotonically increasing counters, so a lost UDP
    packet cannot lose or repeat a press.
    """

    def __init__(self, listen_port: int = 11005, status_port: int = 11006):
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.bind(("0.0.0.0", listen_port))
        self.rx.settimeout(0.5)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.status_addr = ("127.0.0.1", status_port)
        self.lock = threading.Lock()
        self.packet = None
        self.packet_time = -math.inf
        self.page = None
        self.counters = None
        self.pending = dict.fromkeys(EVENTS, 0)
        self.running = True
        self.thread = threading.Thread(target=self._receive, daemon=True)
        self.thread.start()

    def _receive(self) -> None:
        announced = False
        while self.running:
            try:
                data, _ = self.rx.recvfrom(65536)
                packet = json.loads(data)
            except socket.timeout:
                continue
            except (OSError, ValueError):
                if not self.running:
                    return
                continue
            if not isinstance(packet, dict):
                continue
            ev = packet.get("ev") or {}
            with self.lock:
                counts = {k: int(ev.get(k, 0) or 0) for k in EVENTS}
                page = packet.get("page")
                if self.counters is not None and page == self.page:
                    for k in EVENTS:
                        # A smaller counter means the page reloaded: new baseline.
                        self.pending[k] += max(0, counts[k] - self.counters[k])
                # A different page instance (reload, second tab) starts a new
                # baseline, so its counters can never read as presses.
                self.page = page
                self.counters = counts
                self.packet = packet
                self.packet_time = time.monotonic()
            if not announced:
                print("Headset packets arriving.", flush=True)
                announced = True

    def snapshot(self) -> tuple[dict | None, float]:
        with self.lock:
            return self.packet, time.monotonic() - self.packet_time

    def take_events(self) -> dict[str, int]:
        with self.lock:
            events = dict(self.pending)
            self.pending = dict.fromkeys(EVENTS, 0)
        return events

    def send_status(self, status: dict) -> None:
        try:
            self.tx.sendto(json.dumps(status).encode(), self.status_addr)
        except OSError:
            pass

    def close(self) -> None:
        self.running = False
        self.rx.close()
        self.tx.close()


def parse_packet(packet: dict | None, age: float, timeout: float):
    """Controller poses (robot frame), grips and the right thumbstick.

    Everything is dropped while the headset link is stale.
    """
    hands, grips, stick = [None, None], [False, False], (0.0, 0.0)
    if packet is None or age > timeout:
        return hands, grips, stick
    raw = packet.get("stick")
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        try:
            values = [float(v) for v in raw]
            if all(math.isfinite(v) for v in values):
                stick = (max(-1.0, min(1.0, values[0])), max(-1.0, min(1.0, values[1])))
        except (TypeError, ValueError):
            pass
    for k, side in enumerate(("left", "right")):
        hand = packet.get(side) or {}
        if not hand.get("tracked", False):
            continue
        try:
            pose = np.array(hand.get("pose"), dtype=float)
        except (TypeError, ValueError):
            continue
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            continue
        u, _, vt = np.linalg.svd(pose[:3, :3])
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            continue
        hands[k] = pin.SE3(rotation, pose[:3, 3].copy())
        grips[k] = bool(hand.get("button_pressed"))
    return hands, grips, stick


def parse_extras(packet: dict | None, age: float, timeout: float):
    """Triggers (left, right, 0..1) and the head rotation (robot frame), or zeros/None."""
    triggers, head = [0.0, 0.0], None
    if packet is None or age > timeout:
        return triggers, head
    for k, side in enumerate(("left", "right")):
        try:
            triggers[k] = max(0.0, min(1.0, float((packet.get(side) or {}).get("trigger") or 0.0)))
        except (TypeError, ValueError):
            pass
    try:
        pose = np.array(packet.get("head"), dtype=float)
        if pose.shape == (4, 4) and np.all(np.isfinite(pose)):
            u, _, vt = np.linalg.svd(pose[:3, :3])
            head = u @ vt
    except (TypeError, ValueError):
        pass
    return triggers, head


class LimitWriter:
    """Send torque limits by SDO only when they change: raise quickly, lower lazily."""

    def __init__(self, driver):
        self.driver = driver
        self.sent = np.full(N_JOINTS, np.nan)
        self.when = np.full(N_JOINTS, -math.inf)

    def update(self, want: np.ndarray, now: float, urgent=()) -> None:
        """urgent: joints whose exact value must go out now if it changed."""
        changes = {}
        for i in range(N_JOINTS):
            sent, value = self.sent[i], float(want[i])
            if not math.isfinite(sent):
                due, ready = True, True
            elif i in urgent:
                due, ready = abs(value - sent) > 1e-6, True
            else:
                due = (
                    value > sent + 0.1
                    or value < sent - 0.5
                    or (abs(value - sent) > 0.05 and now - self.when[i] > 1.0)
                )
                ready = now - self.when[i] >= 0.05
            if due and ready:
                changes[i] = value
                self.sent[i] = value
                self.when[i] = now
        if changes:
            self.driver.write_limits(changes)


class JointHealth:
    """Per-joint flags for the headset: no CAN reply, torque maxed out, not following."""

    def __init__(self):
        self.maxed_time = np.zeros(N_JOINTS)
        self.lag_time = np.zeros(N_JOINTS)
        self.tags = ["" for _ in range(N_JOINTS)]

    def update(self, dt, q_cmd, q, tau, limits, miss, motors_on: bool) -> None:
        err = np.abs(np.nan_to_num(q_cmd - q))
        maxed = motors_on & (np.abs(tau) >= 0.92 * limits)
        lag = motors_on & (err > 0.15) & ~maxed
        self.maxed_time = np.where(maxed, self.maxed_time + dt, 0.0)
        self.lag_time = np.where(lag, self.lag_time + dt, 0.0)
        for i in range(N_JOINTS):
            if miss[i] > 5:
                self.tags[i] = "NO CAN"
            elif self.maxed_time[i] > 0.3:
                self.tags[i] = "MAXED"
            elif self.lag_time[i] > 0.5:
                self.tags[i] = "LAG"
            else:
                self.tags[i] = ""


def joint_named(text: str) -> int:
    """'L yaw', 'left yaw', 'left_shoulder_yaw', 'right elbow' -> its index in ARM_JOINTS."""
    said = str(text or "").strip().lower()
    for i, joint in enumerate(ARM_JOINTS):
        if said in (joint.key, joint.label.lower(), joint.urdf.lower()):
            return i
    words = set(said.replace("_", " ").replace("-", " ").split())
    side = "left" if words & {"left", "l"} else "right" if words & {"right", "r"} else None
    kind = next((k for k in ("pitch", "roll", "yaw", "elbow", "wrist") if k in words), None)
    if side is None or kind is None:
        raise ValueError(f"which joint is {text!r}? a side and one of pitch, roll, yaw, elbow, wrist")
    return joint_index(side, kind)


def _cal_brief(result) -> dict | None:
    """The last calibration attempt of a joint, small enough to send five times a second."""
    if not isinstance(result, dict):
        return None
    attempts = [a for a in result.get("attempts") or [] if isinstance(a, dict)]
    last = attempts[-1] if attempts else {}
    return {"ok": bool(last.get("pass")), "why": str(last.get("why") or "")[:80],
            "at": _num(float(last.get("level") or 0.0)), "moved": _num(float(last.get("progress") or 0.0)),
            "err": _num(float(last.get("max_err") or 0.0))}


def _num(value: float, digits: int = 2):
    return round(float(value), digits) if math.isfinite(value) else None


class ArmSupervisor:
    """States: STOPPED (damping), ARMED (teleop), CAL (power calibration)."""

    def __init__(self, driver, profile: PowerProfile, gravity: GravityModel, solver: TeleopIkSolver,
                 *, gravity_comp: bool = True, caps: np.ndarray | None = None,
                 link_timeout: float = 0.25, link_lost_stop: float = 5.0, viz: MeshcatView | None = None,
                 servos: ServoLink | None = None, person_port: int = PERSON_PORT, tune_port: int = TUNE_PORT,
                 record_port: int | None = None):
        self.driver = driver
        self.servos = servos
        self.servo_teleop = ServoTeleop(servos) if servos is not None else None
        self.camera = CameraAim(servos) if servos is not None else None
        # person_track.py reports where the operator is; the camera keeps them centred,
        # and the robot's face turns to them (it needs the camera's angle added).
        self.person = None
        self.person_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.person_in.bind(("127.0.0.1", person_port))
        self.person_in.setblocking(False)
        self.face_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # live tuning (tools/teleop_tune.py, a Claude session, "agent, more power ...")
        self.tune_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.tune_in.bind(("127.0.0.1", tune_port))
        self.tune_in.setblocking(False)
        # --record-tap: one row per cycle to recorder.py (docs/LFD_RECORDING_FORMAT.md, section 2).
        # Off (None) means no socket and nothing done per cycle: the loop is exactly as before.
        self.record_port = record_port
        self.tap = None
        self.tap_seq = 0
        self.tap_errors = 0
        if record_port is not None:
            self.tap = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.tap.setblocking(False)
        self.tune_history: list[tuple] = []          # (key, field, old, new), for undo
        self.reach_at = -math.inf                    # hands sent out of reach, for the hint
        self.reach_since = [0.0, 0.0]
        self.reach_cm = [0.0, 0.0]
        self.triggers = [0.0, 0.0]
        self.head = None
        self.profile = profile
        self.gravity = gravity
        self.solver = solver
        self.gravity_comp = gravity_comp
        self.caps = caps
        self.link_timeout = link_timeout
        self.link_lost_stop = link_lost_stop
        self.viz = viz
        self.state = "STOPPED"
        self.motors = False
        self.q_cmd = np.zeros(N_JOINTS)
        self.tau_ff = np.zeros(N_JOINTS)
        self.limits = profile.column("floor")
        self.limiter = LimitWriter(driver)
        self.health = JointHealth()
        self.cal: PowerCalibration | None = None
        self.alert = ""
        self.alert_until = 0.0
        self.last_diag = -math.inf
        self.last_poll = -math.inf
        self.poll_index = 0
        self.faulted: dict[int, str] = {}
        self.motors_since = -math.inf
        self.ff_start = 0.0
        self.last = None
        self.hz = 0.0
        self.link_age = math.inf
        self.hands = [None, None]
        self.grips = [False, False]
        self.stick = (0.0, 0.0)
        self.now = 0.0

    # -- host API used by PowerCalibration
    def capture_zero(self) -> None:
        self.driver.capture_zero()
        self._say("Zero set: arms hanging = 0")

    def apply_signs(self) -> None:
        self.driver.set_signs(self.profile.axis_signs())

    def apply_ranges(self) -> None:
        lower, upper = self.profile.ranges(self.gravity.lower, self.gravity.upper)
        self.solver.set_range(lower, upper)

    def joint_status(self, i: int) -> str:
        try:
            return self.driver.joint_status(i)
        except Exception as exc:
            return f"status read failed: {exc}"

    # -- lifecycle
    def start(self, now: float) -> None:
        self.apply_ranges()
        self.apply_signs()
        self.driver.capture_zero()
        self.driver.write_gains(self.profile.column("kp"), self.profile.column("kd"))
        self.limiter.update(self.profile.column("floor"), now, urgent=range(N_JOINTS))
        self.driver.exchange(self.driver.q, np.zeros(N_JOINTS), now)
        self.driver.set_mode(Mode.DAMPING)
        self.q_cmd = self.driver.q.copy()

    def shutdown(self) -> None:
        if self.cal is not None:
            self.cal.abort("shutdown")
        self.driver.set_mode(Mode.DAMPING)
        self.motors = False
        self.state = "STOPPED"

    def tick(self, now: float, packet: dict | None, age: float, events: dict[str, int]) -> None:
        dt = 0.02 if self.last is None else max(1e-4, now - self.last)
        self.last = now
        self.now = now
        self.hz = 0.9 * self.hz + 0.1 / dt if self.hz else 1.0 / dt
        self.link_age = age
        self.hands, self.grips, self.stick = parse_packet(packet, age, self.link_timeout)
        self.triggers, self.head = parse_extras(packet, age, self.link_timeout)
        q_meas = self.driver.q.copy()
        tau_meas = self.driver.tau.copy()

        if events.get("x3"):
            if self.motors or self.state != "STOPPED":
                self._stop("triple X")
            else:
                # reach first (sitting vs standing), then arm when it closes
                self.cal = ReachCheck(self.profile)
                self.state = "CAL"
                self._say("Reach check, then the motors arm")
        if events.get("y3"):
            if self.state == "STOPPED":
                self.cal = CalibrationChooser(self.servos)
                self.state = "CAL"
                self._say("Calibration menu: A = arm power, B = claws + camera")
            elif self.state == "CAL":
                self._stop("calibration closed")
            else:
                self._say("Stop the motors first (triple X), then triple Y")
        # The camera: A x2 switches track <-> head; the agent can also pick "still".
        wanted = ("toggle" if events.get("cam") else "track" if events.get("camtrack")
                  else "head" if events.get("camhead") else "still" if events.get("camstill") else None)
        if wanted and self.camera is not None:
            mode = self.camera.toggle() if wanted == "toggle" else self.camera.set_mode(wanted)
            self._say({"track": "Camera TRACKS you: it keeps you centred",
                       "head": "Camera FOLLOWS your head",
                       "still": "Camera holds STILL"}[mode])
        elif wanted:
            self._say("No camera servos (the Nano is not connected)")
        if events.get("b") and self.state == "STOPPED":
            self.profile.swap_hands = not self.profile.swap_hands
            try:
                self.profile.save()
            except OSError as exc:
                self._say(f"could not save the profile: {exc}")
            self._say("MIRROR (face the robot): left controller drives the RIGHT arm, motion reflected"
                      if self.profile.swap_hands else
                      "STRAIGHT (stand the way the robot faces): left controller drives the LEFT arm")
        self._take_tuning()
        if self.motors and age > self.link_lost_stop:
            self._stop("headset link lost")
        silent = np.flatnonzero(self.driver.miss > 25)
        if self.motors and silent.size:
            # Its own watchdog would drop only that joint; stop the whole arm set.
            self._stop(f"{ARM_JOINTS[silent[0]].label} stopped answering on CAN")

        self._aim_camera(now)
        if self.state == "ARMED":
            hands, grips, triggers = self.hands, self.grips, self.triggers
            if self.profile.swap_hands:
                hands, grips, triggers = [hands[1], hands[0]], [grips[1], grips[0]], [triggers[1], triggers[0]]
            if self.servo_teleop is not None:
                self.servo_teleop.update(now, grips, triggers, self.head)
            self._tune_sensitivity()
            self._log_cycle(now, q_meas)
            self.q_cmd = self.solver.update(self.q_cmd, q_meas, grips, hands, dt,
                                            self.profile.motion_scale,
                                            facing=self.profile.swap_hands)
            self._watch_reach(now)
        elif self.state == "CAL":
            self._tick_calibration(now, dt, q_meas, tau_meas, events)
        if not self.motors:
            self.q_cmd = q_meas

        if self.motors:
            ramp = min(1.0, (now - self.ff_start) / 0.5)
            if self.gravity_comp:
                self.tau_ff = self.profile.feed_forward(self.gravity.torque(self.q_cmd)) * ramp
            else:
                self.tau_ff = np.zeros(N_JOINTS)
            self.limits = self.profile.dynamic_limits(self.tau_ff)
            overrides = self.cal.limit_override if self.cal is not None else {}
            for i, value in overrides.items():
                self.limits[i] = value
            self.limiter.update(self.limits, now, urgent=tuple(overrides))
        else:
            self.tau_ff = np.zeros(N_JOINTS)
            self.limits = self.profile.column("floor")
            self.limiter.update(self.limits, now)

        self.driver.exchange(self.q_cmd, self.tau_ff, now)
        if self.tap is not None:
            self._record_tap()
        self.health.update(dt, self.q_cmd, self.driver.q, self.driver.tau, self.limits,
                           self.driver.miss, self.motors)
        self._diagnose(now)
        self._poll_state(now)
        if self.viz is not None:
            self.viz.show(self.driver.q, self.solver.targets, now)

    # -- transitions
    def _arming_problem(self, q: np.ndarray) -> str:
        if self.faulted:
            i = next(iter(self.faulted))
            return f"{ARM_JOINTS[i].label}: {self.faulted[i]}"
        if self.link_age > self.link_timeout:
            return "No headset link: cannot arm"
        if not np.all(np.isfinite(q)):
            return "Joint positions unknown: cannot arm"
        silent = np.flatnonzero(self.driver.miss > 5)
        if silent.size:
            return f"{ARM_JOINTS[silent[0]].label} not replying: cannot arm"
        outside = np.flatnonzero((q < self.gravity.lower - 0.1) | (q > self.gravity.upper + 0.1))
        if outside.size:
            i = outside[0]
            return (f"{ARM_JOINTS[i].label} reads {q[i]:+.2f} rad, outside its range: "
                    "hang the arms and set zero (triple Y)")
        return ""

    def _motors_on(self, now: float, q_start: np.ndarray) -> None:
        self.q_cmd = q_start.copy()
        self.ff_start = now
        self.limiter.update(self.profile.column("floor"), now, urgent=range(N_JOINTS))
        # Refresh the stored targets (feed-forward 0) before leaving damping.
        self.driver.exchange(self.q_cmd, np.zeros(N_JOINTS), now)
        self.driver.set_mode(Mode.POSITION)
        self.motors = True
        self.motors_since = now

    def _motors_off(self) -> None:
        if self.motors:
            self.driver.set_mode(Mode.DAMPING)
        self.motors = False

    def _open_log(self, heading_deg: float) -> None:
        """One JSON line per 40 ms while armed: arm_validation/teleop_logs/<time>.jsonl."""
        folder = Path(__file__).resolve().parents[2] / "arm_validation" / "teleop_logs"
        self.log_path = folder / time.strftime("%Y%m%d_%H%M%S.jsonl")
        self.log_last = -math.inf
        try:
            folder.mkdir(parents=True, exist_ok=True)
            self.log_file = open(self.log_path, "w")
            self.log_file.write(json.dumps({"start": time.time(), "heading_deg": round(heading_deg, 1),
                                            "mirror": self.profile.swap_hands,
                                            "scale": self.profile.motion_scale}) + "\n")
        except OSError as exc:                  # a full disk: teleop goes on without its log
            self._say(f"Heading {heading_deg:+.0f} deg taken as forward")
            self._close_log(exc)
            return
        self._say(f"Heading {heading_deg:+.0f} deg taken as forward; recording to {self.log_path.name}")

    def _log_cycle(self, now: float, q_meas) -> None:
        f = getattr(self, "log_file", None)
        if f is None or now - self.log_last < 0.04 or not any(self.solver.held):
            return
        self.log_last = now
        got = self.solver.hand_positions(np.where(np.isfinite(q_meas), q_meas, self.q_cmd))
        sent = self.solver.hand_positions(self.q_cmd)
        # tau and lim tell "at its power limit" from anything else; "commanded" (where the
        # IK put the hand) tells "sent out of reach" from "the arm did not follow"
        row = {"t": round(now, 3), "held": self.solver.held,
               "q_cmd": np.round(self.q_cmd, 4).tolist(), "q_meas": np.round(q_meas, 4).tolist(),
               "tau": np.round(self.driver.tau, 2).tolist(), "lim": np.round(self.limits, 2).tolist()}
        for k, side in enumerate(("L", "R")):
            if self.solver.held[k] and self.solver.targets[k] is not None:
                row[side] = {"op_move": np.round(self.solver.last_d_pos[k], 4).tolist(),
                             "target": np.round(self.solver.targets[k].translation, 4).tolist(),
                             "commanded": np.round(sent[k], 4).tolist(),
                             "reached": np.round(got[k], 4).tolist()}
        try:
            f.write(json.dumps(row) + "\n")
        except OSError as exc:
            self._close_log(exc)

    def _close_log(self, failed: OSError | None = None) -> None:
        """At a stop, or at once when a write fails (a full disk): said once, and the loop carries on."""
        f, self.log_file = getattr(self, "log_file", None), None   # first: a closed file would raise ValueError
        if f is not None:
            try:
                f.close()
            except OSError as exc:              # its last lines could not be written
                failed = failed or exc
        if failed is not None:
            self._say(f"Teleop log stopped: {failed}")

    def _log_row(self, row: dict) -> None:
        f = getattr(self, "log_file", None)
        if f is not None:
            try:
                f.write(json.dumps(dict(row, t=round(self.now, 3))) + "\n")
            except OSError as exc:
                self._close_log(exc)

    @staticmethod
    def _tap_floats(values, digits: int) -> list:
        """Rounded and flattened, with null for NaN or inf (a joint not heard yet): rows are strict JSON."""
        return [x if math.isfinite(x) else None
                for x in np.round(np.asarray(values, dtype=float), digits).ravel().tolist()]

    def _record_tap(self) -> None:
        """This cycle as one JSON datagram for recorder.py (docs/LFD_RECORDING_FORMAT.md, section 2).

        Fire and forget: it only reads the loop's state, on a non-blocking socket, with no
        file and no print. A row it cannot build or send is counted in tap_errors and
        dropped; nothing here may raise into the control loop.
        """
        try:
            seq = self.tap_seq
            self.tap_seq += 1                  # every cycle, so a gap in seq at the recorder is a lost row
            t = time.monotonic()               # the recordings' one clock, not the loop's perf_counter
            # the last pulse ServoLink sent each claw since start-up: a re-arm (a new _Sender) keeps it
            claw_us = self.servos.last_us if self.servos is not None else {}
            f = self._tap_floats
            camera = self.camera
            row = {
                "v": 1, "seq": seq, "t": round(t, 6), "state": self.state, "motors": bool(self.motors),
                "src": "teleop" if self.state == "ARMED" else "cal" if self.state == "CAL" else "idle",
                "q": f(self.driver.q, 5), "qc": f(self.q_cmd, 5),
                "tau": f(self.driver.tau, 3), "tff": f(self.tau_ff, 3),
                "lim": f(self.limits, 3), "zero": f(self.driver.zero_fw, 5),
                "held": [bool(h) for h in self.solver.held],
                # the operator's own hands, before any mirror swap (self.hands, not tick's swapped copy)
                "grip": [bool(g) for g in self.grips],
                "trig": f(self.triggers, 3),
                "hand": [None if h is None else f(h.homogeneous, 5) for h in self.hands],
                "head": None if self.head is None else f(self.head, 5),
                "tgt": [f(target.translation, 5) if held and target is not None else None
                        for held, target in zip(self.solver.held, self.solver.targets)],
                "swap": bool(self.profile.swap_hands), "scale": _num(self.profile.motion_scale, 6),
                "claw": [claw_us.get(3), claw_us.get(7)],
                # without the Nano the camera is not driven: its angle is unknown, not the last one aimed at
                "cam": None if camera is None or not self.servos.ready else f(camera.angle, 5),
                "cam_mode": None if camera is None else camera.mode,
                "intervention": False,
            }
            payload = json.dumps(row, separators=(",", ":"), allow_nan=False)   # a NaN missed above: dropped
            self.tap.sendto(payload.encode(), ("127.0.0.1", self.record_port))
        except Exception:
            self.tap_errors += 1

    # -- live tuning, while you drive: a joint's power and gravity help, the sensitivity,
    # a joint's direction (only while stopped - flipping one that holds makes it jump),
    # and undo. From tools/teleop_tune.py, a Claude session, or "agent, more power left
    # yaw". Each change is saved to the profile at once and written into the teleop log.
    POWER_STEP = 0.5            # N·m, as the calibration's own steps
    GRAVITY_STEP = 0.05

    def _take_tuning(self) -> None:
        while True:
            try:
                data, sender = self.tune_in.recvfrom(4096)
            except (BlockingIOError, OSError):
                return
            ask = {}
            try:
                ask = json.loads(data)
                if not isinstance(ask, dict):
                    raise ValueError("expected a JSON object")
                reply = {"ok": True, "said": self._tune(ask)}
            except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                reply = {"ok": False, "said": str(exc)}
            reply.update(state=self.state, scale=self.profile.motion_scale, joints=self._tuning_table())
            try:
                self.tune_in.sendto(json.dumps(reply).encode(), sender)
            except OSError:
                pass
            if ask.get("what") != "show":
                self._say(("Tuned: " if reply["ok"] else "Not tuned: ") + reply["said"])
                self._log_row({"tune": ask, "ok": reply["ok"], "said": reply["said"]})

    def _tune(self, ask: dict) -> str:
        what = ask.get("what")
        if what == "show":
            return f"{self.state}, sensitivity {self.profile.motion_scale:.2f}"
        if self.state == "CAL":
            raise RuntimeError("the calibration is running: finish or close it first (triple Y)")
        up = float(ask.get("step", 1)) > 0
        if what == "undo":
            if not self.tune_history:
                raise ValueError("nothing to undo")
            key, field, old, new = self.tune_history[-1]
            if field == "axis_flip" and self.motors:
                raise RuntimeError("stop the motors first (triple X) to undo a flip")
            self.tune_history.pop()
            if key == "motion_scale":
                self.profile.motion_scale = old
            elif field == "power":
                power = self.profile.joints[key]
                power.floor, power.headroom, power.max_torque = old
            else:
                setattr(self.profile.joints[key], field, old)
                if field == "axis_flip":
                    self.apply_signs()
            self._save_profile()
            return f"undone: {key.replace('_', ' ')} {field.replace('_', ' ')} back to {self._shown(old)}"
        if what == "scale":
            old = self.profile.motion_scale
            new = float(ask["value"]) if "value" in ask else old * (1.15 if up else 1 / 1.15)
            self.profile.motion_scale = round(float(np.clip(new, 0.15, 1.5)), 2)
            self._keep_tune("motion_scale", "motion_scale", old, self.profile.motion_scale)
            return f"sensitivity {old:.2f} -> {self.profile.motion_scale:.2f}"
        i = joint_named(ask.get("joint", ""))
        joint = ARM_JOINTS[i]
        power = self.profile.joints[joint.key]
        if what == "power":
            ceiling = max(KIND_DEFAULTS[joint.kind][4], power.cal_cap or 0.0)   # the kind's own
            old = (power.floor, power.headroom, power.max_torque)
            if up and power.max_torque >= ceiling - 1e-9:
                raise ValueError(f"{joint.label} is at its ceiling already, {ceiling:.1f} N·m - if it still "
                                 "does not follow, something stops it: try the calibration (triple Y)")
            step = self.POWER_STEP if up else -self.POWER_STEP
            power.max_torque = float(np.clip(power.max_torque + step, 0.5, ceiling))
            power.headroom = float(np.clip(power.headroom + step, 0.0, power.max_torque))
            power.floor = float(np.clip(power.floor + step, 0.2, power.max_torque))
            self._keep_tune(joint.key, "power", old, (power.floor, power.headroom, power.max_torque))
            return f"{joint.label} power {old[2]:.1f} -> {power.max_torque:.1f} N·m (its ceiling {ceiling:.1f})"
        if what == "gravity":
            old = power.gravity_scale
            power.gravity_scale = round(float(np.clip(old + (self.GRAVITY_STEP if up else -self.GRAVITY_STEP),
                                                      0.5, 1.5)), 2)
            self._keep_tune(joint.key, "gravity_scale", old, power.gravity_scale)
            return f"{joint.label} gravity help x{old:.2f} -> x{power.gravity_scale:.2f}"
        if what == "flip":
            if self.motors:
                raise RuntimeError(f"stop the motors first (triple X): flipping {joint.label} while it holds "
                                   "would make it jump")
            old = power.axis_flip
            power.axis_flip = not old
            self.apply_signs()
            self._keep_tune(joint.key, "axis_flip", old, power.axis_flip)
            return f"{joint.label} direction flipped - it now turns the other way for the same hand motion"
        raise ValueError(f"unknown change {what!r}: power, gravity, scale, flip, undo or show")

    @staticmethod
    def _shown(value) -> str:
        if isinstance(value, tuple):
            return f"{value[2]:.1f} N·m"
        return f"{value:.2f}" if isinstance(value, float) else str(value)

    def _keep_tune(self, key: str, field: str, old, new) -> None:
        self.tune_history = (self.tune_history + [(key, field, old, new)])[-20:]
        self._save_profile()

    def _save_profile(self) -> None:
        try:
            self.profile.save()
        except OSError as exc:
            self._say(f"could not save the profile: {exc}")

    def _tuning_table(self) -> list[dict]:
        """Each joint's settings and how it is doing now, for tools/teleop_tune.py."""
        table = []
        for i, joint in enumerate(ARM_JOINTS):
            power = self.profile.joints[joint.key]
            table.append({"joint": joint.label, "key": joint.key, "max": _num(power.max_torque),
                          "headroom": _num(power.headroom), "floor": _num(power.floor),
                          "gravity": _num(power.gravity_scale), "flip": bool(power.axis_flip),
                          "ceiling": _num(max(KIND_DEFAULTS[joint.kind][4], power.cal_cap or 0.0)),
                          "tau": _num(self.driver.tau[i]), "lim": _num(self.limits[i]),
                          "err": _num(self.q_cmd[i] - self.driver.q[i], 3) if self.motors else 0.0,
                          "tag": self.health.tags[i]})
        return table

    def _watch_reach(self, now: float) -> None:
        """How far each held hand's target is from where the arm can put it (10 Hz)."""
        if now - self.reach_at < 0.1:
            return
        self.reach_at = now
        got = self.solver.hand_positions(self.q_cmd)
        for k in (0, 1):
            target = self.solver.targets[k]
            far = self.solver.held[k] and target is not None
            gap = float(np.linalg.norm(target.translation - got[k])) if far else 0.0
            self.reach_cm[k] = 100 * gap
            self.reach_since[k] = (self.reach_since[k] or now) if gap > 0.06 else 0.0

    def _armed_hint(self) -> str:
        """While you drive: what is going wrong right now, and the fix, in one line."""
        for i, joint in enumerate(ARM_JOINTS):
            if not self.solver.held[0 if joint.side == "left" else 1]:
                continue
            side = joint.side
            if self.health.tags[i] == "MAXED" and self.health.maxed_time[i] > 1.0:
                return (f"{joint.label} is at its power limit ({self.limits[i]:.1f} N·m) and cannot keep up - "
                        f"say \"agent, more power {side} {joint.kind}\"")
            if self.health.tags[i] == "LAG" and self.health.lag_time[i] > 1.0:
                return (f"{joint.label} is behind by {abs(self.q_cmd[i] - self.driver.q[i]):.2f} rad - "
                        f"say \"agent, more power {side} {joint.kind}\" if it keeps happening")
        for k, side in enumerate(("left", "right")):
            if self.reach_since[k] and self.now - self.reach_since[k] > 0.5:
                return (f"{side} hand sent {self.reach_cm[k]:.0f} cm past where the arm can reach - "
                        "let go of the grip and grab again closer in")
        return ""

    def _tune_sensitivity(self) -> None:
        """While armed, right stick up/down: robot hand travel per cm of yours, +-15 %."""
        y = self.stick[1]
        if abs(y) < 0.3:
            self.stick_latch = False
            return
        if getattr(self, "stick_latch", False) or abs(y) < 0.7:
            return
        self.stick_latch = True
        scale = self.profile.motion_scale * (1.15 if y > 0 else 1 / 1.15)
        self.profile.motion_scale = round(float(np.clip(scale, 0.15, 1.5)), 2)
        try:
            self.profile.save()
        except OSError:
            pass
        self._say(f"Sensitivity {self.profile.motion_scale:.2f} (stick up = more, down = less)")

    def _arm_teleop(self, now: float, q_meas: np.ndarray) -> None:
        problem = self._arming_problem(q_meas)
        if problem:
            self._say(problem)
            return
        self.solver.reset()
        yaw = self.solver.set_heading(self.head)
        self._motors_on(now, q_meas)
        self.state = "ARMED"
        self._open_log(yaw)
        if self.servo_teleop is not None:
            self.servo_teleop.start(self.head)
        self._say("Motors ARMED: hold a grip to move that arm; squeeze its trigger for the claw")

    def _camera_line(self) -> str:
        mode = self.camera.mode if self.camera is not None else "none"
        return {"track": "Camera TRACKS you (A x2: follow your head)",
                "head": "Camera FOLLOWS your head (A x2: track you)",
                "still": "Camera holds STILL (A x2: track you)"}.get(mode, "No camera servos")

    def _aim_camera(self, now: float) -> None:
        """Every cycle, in every state but the claws-and-camera setup (which owns the servos)."""
        while True:
            try:
                report = json.loads(self.person_in.recv(4096).decode())
            except (BlockingIOError, ValueError, OSError, RecursionError):   # RecursionError: "[[[[..." nested
                break
            if report.get("type") == "person":
                self.person = report
                if report.get("seen") and self.camera is not None:
                    yaw, pitch = self.camera.world(report)
                    face = dict(report, yaw=round(yaw, 1), pitch=round(pitch, 1), world=True)
                    try:
                        self.face_out.sendto(json.dumps(face).encode(), ("127.0.0.1", FACE_PORT))
                    except OSError:
                        pass
        if self.camera is None or isinstance(self.cal, ServoCalibration):
            return
        fresh = self.person if self.person and time.time() - float(self.person.get("t", 0)) < 1.0 else None
        self.camera.update(now, self.head, fresh)

    def _stop(self, reason: str) -> None:
        if self.cal is not None:
            self.cal.abort(reason)
            self._finish_calibration()
        self._motors_off()
        if self.state == "ARMED" and self.servo_teleop is not None:
            self.servo_teleop.stop()
        self._close_log()
        self.state = "STOPPED"
        self.solver.reset()
        self._say(f"Motors STOPPED ({reason})")

    def _tick_calibration(self, now, dt, q_meas, tau_meas, events) -> None:
        cal = self.cal
        cal.step(CalInputs(
            now=now, dt=dt, q=q_meas, tau=tau_meas, g=self.gravity.torque(q_meas),
            miss=self.driver.miss.copy(),
            grip=any(self.grips), a=bool(events.get("a")), b=bool(events.get("b")),
            hands=tuple(None if h is None else h.translation.copy() for h in self.hands),
            stick=self.stick,
        ))
        if isinstance(cal, ReachCheck) and cal.done:
            self.cal = None
            self.state = "STOPPED"
            self._say(f"Reach: {cal.outcome}")
            self._arm_teleop(now, q_meas)
            return
        if isinstance(cal, CalibrationChooser) and cal.done and cal.choice:
            if cal.choice == "power":
                self.cal = PowerCalibration(self.profile, self.gravity, self, caps=self.caps)
                self._say("Power calibration")
            else:
                self.cal = ServoCalibration(self.servos or ServoLink())
                self._say("Claws + camera setup")
            return
        if cal.motors and not self.motors:
            problem = self._arming_problem(cal.q_cmd)
            if problem:
                cal.abort(problem)
                self._say(problem)
            else:
                self._motors_on(now, cal.q_cmd)
        elif not cal.motors and self.motors:
            self._motors_off()
        if self.motors:
            self.q_cmd = cal.q_cmd.copy()
        if cal.done:
            self._finish_calibration()

    def _finish_calibration(self) -> None:
        cal, self.cal = self.cal, None
        self._motors_off()
        self.state = "STOPPED"
        self.apply_signs()
        self.apply_ranges()
        saved = f" Profile: {self.profile.path}" if cal._saved_once else ""
        self._say(f"Calibration closed ({cal.outcome or 'stopped'}).{saved}")
        if cal.report_path:
            self._say(f"Results written to {cal.report_path}")

    def _diagnose(self, now: float) -> None:
        if not self.motors or now - self.last_diag < 1.0:
            return
        lagging = np.flatnonzero(self.health.lag_time > 0.5)
        if lagging.size:
            self.last_diag = now
            i = int(lagging[0])
            self._say(f"{ARM_JOINTS[i].label} is not following: {self.joint_status(i)}")

    def _poll_state(self, now: float) -> None:
        """Read one joint's mode and error bits every 0.2 s (all ten every 2 s).

        The firmware drops a joint to damping on its own (encoder fault,
        watchdog). A limp joint may not lag enough to notice, so ask.
        """
        if now - self.last_poll < 0.2:
            return
        self.last_poll = now
        i = self.poll_index
        self.poll_index = (i + 1) % N_JOINTS
        if self.driver.miss[i] > 5:
            return
        try:
            mode, error = self.driver.read_state(i)
        except Exception:
            return
        if mode is None or error is None:
            return
        label = ARM_JOINTS[i].label
        if error & ENCODER_FAULT and i not in self.faulted:
            self.faulted[i] = ("encoder fault latched (only a power cycle clears it). "
                               "Power-cycle the robot, then restart run_teleop.py")
            reason = f"{label}: {self.faulted[i]}"
            if self.motors:
                self._stop(reason)
            else:
                self._say(reason)
        elif self.motors and now - self.motors_since > 0.3 and mode != Mode.POSITION:
            self._stop(f"{label} dropped out of position control ({describe_state(mode, error)})")

    def _say(self, text: str) -> None:
        self.alert = text
        self.alert_until = self.now + 5.0
        print(f"[{self.state}] {text}", flush=True)

    # -- reporting
    # Poses the headset's robot demonstrates while a menu asks you for one (URDF angles).
    DEMO_POSES = {
        "down": {},
        "forward": {"arm_left_shoulder_pitch_joint": -1.5, "arm_right_shoulder_pitch_joint": 1.5},
    }
    CLAW_WRIST = {"D3": "arm_left_elbow_roll_joint", "D7": "arm_right_elbow_roll_joint"}

    def _model_hint(self) -> dict:
        """What the robot in the headset should light up and show, right now.

        active: URDF joints that are what is happening now (they glow).
        pose:   angles to demonstrate instead of the real ones, when a menu asks you to
                hold a pose (the reach check), else None.
        camera: "follow", "still", or "none" (no camera servos).
        """
        active, pose = [], None
        cal = self.cal
        testing = getattr(cal, "testing", None) if cal is not None else None
        if isinstance(testing, int) and 0 <= testing < len(ARM_JOINTS):
            active = [ARM_JOINTS[testing].urdf]                  # the joint being calibrated
        elif cal is not None and getattr(cal, "demo", None) in self.DEMO_POSES:
            active = [joint.urdf for joint in ARM_JOINTS]        # the reach check: both arms
            pose = self.DEMO_POSES[cal.demo]
        elif cal is not None and getattr(cal, "current", None) in self.CLAW_WRIST:
            active = [self.CLAW_WRIST[cal.current]]              # the claw being set: its hand
        elif self.state == "ARMED":
            grips = list(self.grips)                             # the arm you are driving
            if self.profile.swap_hands:
                grips = [grips[1], grips[0]]
            active = [joint.urdf for joint in ARM_JOINTS if grips[0 if joint.side == "left" else 1]]
        camera = "none" if self.camera is None else self.camera.mode
        return {"active": active, "pose": pose, "camera": camera}

    def status(self) -> dict:
        err = self.q_cmd - self.driver.q
        joints = []
        for i, joint in enumerate(ARM_JOINTS):
            power = self.profile.power(i)
            tag = self.health.tags[i]
            if self.cal is not None and self.cal.tags[i]:
                tag = self.cal.tags[i]
            if i in self.faulted:
                tag = "FAULT"
            joints.append({
                "n": f"{joint.label} {joint.device_id}",
                "q": _num(self.driver.q[i]),
                "e": _num(err[i]) if self.motors else 0.0,
                "tau": _num(self.driver.tau[i]),
                "lim": _num(self.limits[i]),
                "max": _num(power.max_torque),
                "tag": tag,
                "cal": bool(power.calibrated),
                "test": self.cal is not None and self.cal.testing == i,
                # for the 3D robot in the headset: which URDF joint this is, what the
                # simulation asks of it, and the numbers behind its power
                "u": joint.urdf,
                "id": joint.device_id,
                "qc": _num(self.q_cmd[i]),
                "p": {"kp": _num(power.kp), "kd": _num(power.kd), "fl": _num(power.floor),
                      "hr": _num(power.headroom), "g": _num(power.gravity_scale),
                      "flip": bool(power.axis_flip),
                      "lo": _num(power.range_lower) if power.range_lower is not None else None,
                      "hi": _num(power.range_upper) if power.range_upper is not None else None,
                      "when": power.calibrated or "", "last": _cal_brief(power.result)},
            })
        hint = ""
        if self.cal is not None:
            lines = list(self.cal.lines)
            hint = "PAUSED - hold a grip to continue" if self.cal.paused else self.cal.hint
        elif self.state == "ARMED":
            lines = ["ARMED - to move an arm, HOLD its GRIP",
                     "(the side button under your middle finger);",
                     "let go and the arm stays where it is.",
                     "Grip + trigger: close that claw.",
                     f"R stick up/down: sensitivity {self.profile.motion_scale:.2f}", "Triple X = stop"]
            hint = self._armed_hint()
        else:
            lines = ["STOPPED (motors damping)", "Triple X = reach check, then arm",
                     "Triple Y = calibration (arms / claws + camera)",
                     "B = swap hands (now: "
                     + ("MIRROR - facing the robot)" if self.profile.swap_hands else "straight - facing the way it faces)")]
        return {
            "type": "status",
            "state": self.state,
            "motors": self.motors,
            "sim": bool(self.driver.simulated),
            "hz": round(self.hz),
            "calibrated": self.profile.calibrated_count(),
            "scale": self.profile.motion_scale,
            "swap": self.profile.swap_hands,
            "lines": lines,
            "hint": hint,
            "alert": self.alert if self.now < self.alert_until else "",
            "grips": self.grips,
            **self._model_hint(),
            "joints": joints,
            "log": list(TERMINAL_LINES)[-14:],
            **({"tap_errors": self.tap_errors} if self.tap is not None else {}),   # only with --record-tap
        }

    def summary_line(self) -> str:
        tau = " ".join(
            f"{joint.label.split()[1][:2]}{self.driver.tau[i]:+.1f}/{self.limits[i]:.1f}"
            for i, joint in enumerate(ARM_JOINTS)
        )
        link = "link ok" if self.link_age < self.link_timeout else "no headset"
        flags = [f"{ARM_JOINTS[i].label}:{t}" for i, t in enumerate(self.health.tags) if t]
        if self.tap is not None and self.tap_errors:
            flags.append(f"tap errors {self.tap_errors}")       # rows recorder.py never got
        return f"{self.state:7s} {self.hz:4.0f} Hz  {link}  tau/lim[L..R] {tau}  {' '.join(flags)}"


def parse_faults(items: list[str]) -> dict[str, str]:
    faults = {}
    keys = {joint.key for joint in ARM_JOINTS}
    for item in items:
        key, _, kind = item.partition("=")
        kinds = ("stuck", "weak", "sticky", "flipped", "silent", "encfault")
        if key not in keys or kind not in kinds:
            raise SystemExit(f"bad --sim-fault {item!r}; use KEY={'|'.join(kinds)}, KEY in {sorted(keys)}")
        faults[key] = kind
    return faults


def tap_port(text: str) -> int:
    """--record-tap-port: checked here, or every row's send would fail, quietly, all session."""
    port = int(text)
    if not 1024 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"{text}: a UDP port from 1024 to 65535")
    return port


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim", action="store_true", help="simulated arms, no CAN")
    parser.add_argument("--sim-payload", type=float, default=1.25,
                        help="sim: true gravity = this x URDF (claws, cables)")
    parser.add_argument("--sim-fault", action="append", default=[], metavar="KEY=KIND",
                        help="sim: stuck | weak | sticky | flipped | silent | encfault, e.g. left_wrist_yaw=encfault")
    parser.add_argument("--profile", type=Path, default=PROFILE_PATH)
    parser.add_argument("--rate", type=float, default=50.0, help="control loop Hz")
    parser.add_argument("--max-speed", type=float, default=1.5, help="teleop joint speed cap, rad/s")
    parser.add_argument("--orientation-cost", type=float, default=0.1)
    parser.add_argument("--no-gravity-comp", action="store_true")
    parser.add_argument("--swap-hands", action="store_true",
                        help="left controller drives the right arm (also toggled with B in the headset)")
    parser.add_argument("--cal-power-scale", type=float, default=1.0,
                        help=f"scales the per-joint calibration caps; never above {ABSOLUTE_MAX_TORQUE} Nm")
    parser.add_argument("--reply-timeout", type=float, default=0.006, help="CAN reply window per cycle, s")
    parser.add_argument("--no-viz", action="store_true", help="skip the Meshcat view")
    parser.add_argument("--listen-port", type=int, default=11005)
    parser.add_argument("--status-port", type=int, default=11006)
    parser.add_argument("--tune-port", type=int, default=TUNE_PORT,
                        help="live tuning from tools/teleop_tune.py and 'agent, more power ...'")
    parser.add_argument("--link-lost-stop", type=float, default=5.0,
                        help="stop the motors after this many seconds without headset packets")
    parser.add_argument("--record-tap", action="store_true",
                        help="send one JSON row per control cycle to recorder.py (docs/LFD_RECORDING_FORMAT.md); "
                             "off, nothing is sent")
    parser.add_argument("--record-tap-port", type=tap_port, default=None,
                        help=f"UDP port on 127.0.0.1 where recorder.py reads the tap rows (default {RECORD_TAP_PORT}); "
                             "only with --record-tap")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.stdout = LogTee(sys.stdout)
    sys.stderr = LogTee(sys.stderr)
    profile = PowerProfile.load(args.profile)
    if args.swap_hands:
        profile.swap_hands = True
    gravity = GravityModel()
    caps = np.minimum(
        np.array([KIND_DEFAULTS[joint.kind][4] for joint in ARM_JOINTS]) * args.cal_power_scale,
        ABSOLUTE_MAX_TORQUE,
    )
    solver = TeleopIkSolver(orientation_cost=args.orientation_cost, max_speed=args.max_speed)
    if args.sim:
        driver = SimArmDriver(gravity, payload_scale=args.sim_payload, faults=parse_faults(args.sim_fault))
    else:
        driver = ArmDriver(reply_timeout=args.reply_timeout)

    print("Preflight (read-only):", flush=True)
    problems = driver.preflight()
    if problems:
        print("REFUSING TO START:", flush=True)
        for problem in problems:
            print("  -", problem, flush=True)
        driver.close()
        return 1

    source = f"{profile.path} ({profile.calibrated_count()}/{N_JOINTS} joints calibrated)" \
        if profile.loaded else "built-in defaults (not calibrated yet: triple-click Y in the headset)"
    print(f"Power profile: {source}", flush=True)
    for i, joint in enumerate(ARM_JOINTS):
        power = profile.power(i)
        print(f"  {joint.label:8s} kp={power.kp:g} kd={power.kd:g} limit {power.floor:.1f}..{power.max_torque:.1f} Nm "
              f"(+{power.headroom:.1f} over gravity) gravity x{power.gravity_scale:.2f}"
              f"{' FLIPPED' if power.axis_flip else ''}", flush=True)
    print(f"Hands: {'SWAPPED (left controller -> right arm)' if profile.swap_hands else 'straight'}"
          "   (B in the headset while stopped toggles it)", flush=True)
    print("Start-up pose is zero: both arms should hang straight down now.", flush=True)

    viz = None if args.no_viz else MeshcatView()
    servos = ServoLink()
    link = OperatorLink(args.listen_port, args.status_port)
    supervisor = ArmSupervisor(
        driver, profile, gravity, solver,
        gravity_comp=not args.no_gravity_comp, caps=caps,
        link_lost_stop=args.link_lost_stop, viz=viz, servos=servos, tune_port=args.tune_port,
        record_port=(args.record_tap_port or RECORD_TAP_PORT) if args.record_tap else None,
    )
    if args.record_tap_port is not None and not args.record_tap:
        print("--record-tap-port is ignored without --record-tap: no tap rows are sent", flush=True)
    terminated = []

    def on_sigterm(signum, frame):
        terminated.append(signum)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_sigterm)
    signal.signal(signal.SIGINT, signal.default_int_handler)

    rate = RateLimiter(args.rate, warn=False)
    supervisor.start(time.perf_counter())
    print("Motors STOPPED (damping). Headset: triple-click X to arm, triple-click Y to calibrate.", flush=True)

    last_status = last_print = 0.0
    try:
        while True:
            now = time.perf_counter()
            packet, age = link.snapshot()
            supervisor.tick(now, packet, age, link.take_events())
            if now - last_status >= 0.2:
                link.send_status(supervisor.status())
                last_status = now
            if now - last_print >= 1.0:
                local_only(supervisor.summary_line())
                last_print = now
            rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        supervisor.shutdown()
        link.send_status(supervisor.status())

    try:
        if terminated:
            print("Terminated: motors left in damping.", flush=True)
        else:
            print("Motors damping. Press Ctrl+C again to release them to IDLE.", flush=True)
            while True:
                time.sleep(0.2)
    except KeyboardInterrupt:
        if not terminated:
            driver.set_mode(Mode.IDLE)
            print("Released to IDLE.", flush=True)
        else:
            print("Terminated: motors left in damping.", flush=True)
    finally:
        link.close()
        servos.close()
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
