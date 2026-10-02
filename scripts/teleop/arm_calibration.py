"""Headset-guided power calibration for the arms.

Runs inside run_teleop.py; step() advances one control cycle. Sequence:
  zero pose  ->  your reach (motion scale)  ->  motors on  ->  per joint:
    1. direction check: the joint rocks gently until the operator answers
       A (moves as described) or B (flip the axis sign)
    2. power ladder: a slow arc against gravity at rising torque limits until
       the joint tracks. The passing level and the measured torque set the
       joint's max_torque; a fit of measured torque against the URDF gravity
       model sets gravity_scale.
The profile is saved after every joint (previous file kept as .bak).

Safety: motion only while a grip is held (release = hold still), triple-X
stops the motors, no level exceeds the per-joint cap given by the host.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math

import numpy as np

from arm_power import (
    ABSOLUTE_MAX_TORQUE, ARM_JOINTS, KIND_DEFAULTS, N_JOINTS, REPO_ROOT,
    GravityModel, PowerProfile, ceil_to, joint_index,
)

TORQUE_STEP = 0.5             # Nm per thumbstick click
RANGE_STEP = math.radians(5)  # rad per thumbstick click


@dataclasses.dataclass(frozen=True)
class JointTest:
    target: float  # absolute URDF angle the power arc goes to (0 = hanging)
    nudge: float   # signed direction-check amplitude, same direction as the arc
    motion: str    # what the operator should see


TESTS = {
    ("left", "pitch"): JointTest(-1.2, -0.25, "swing FORWARD and up"),
    ("right", "pitch"): JointTest(+1.2, +0.25, "swing FORWARD and up"),
    ("left", "roll"): JointTest(+1.0, +0.25, "swing OUT to the side"),
    ("right", "roll"): JointTest(-1.0, -0.25, "swing OUT to the side"),
    ("left", "elbow"): JointTest(+1.2, +0.25, "BEND (hand comes forward)"),
    ("right", "elbow"): JointTest(-1.2, -0.25, "BEND (hand comes forward)"),
    ("left", "yaw"): JointTest(-0.5, -0.25, "turn the forearm OUTWARD"),
    ("right", "yaw"): JointTest(+0.5, +0.25, "turn the forearm OUTWARD"),
    ("left", "wrist"): JointTest(-0.5, -0.30, "twist: claw top tips OUTWARD"),
    ("right", "wrist"): JointTest(+0.5, +0.30, "twist: claw top tips OUTWARD"),
}
ORDER = ("pitch", "roll", "elbow", "yaw", "wrist")
# Yaw and wrist are tested with the elbow bent so their motion is visible.
ELBOW_BENT = {"left": 1.0, "right": -1.0}
ROBOT_REACH = 0.30  # shoulder to hand frame, m (URDF)

ARC_SPEED = 0.35     # rad/s peak on the power arc
MOVE_SPEED = 0.5     # rad/s peak for setup moves
WIGGLE_PERIOD = 2.0  # s
REVERSE_PROBE = 0.15  # rad, tried the other way when a joint will not move
STALL_ERR = 0.22     # rad: joint is not following (keep short: a blocked joint pushes at this error)
SLIP_ERR = 0.30      # rad: a joint that should hold has moved
PASS_TRACK = 0.12    # rad: max lag on the arc to pass a level
PASS_HOLD = 0.05     # rad: sag allowed after 1 s at the top
PAUSE_LIMIT = 30.0   # s without a grip before a move gives up
LEVEL_STEP = 0.5     # Nm


@dataclasses.dataclass
class CalInputs:
    now: float
    dt: float
    q: np.ndarray       # measured, URDF frame
    tau: np.ndarray     # measured torque, URDF frame
    g: np.ndarray       # URDF gravity torque at q (unscaled)
    miss: np.ndarray    # consecutive cycles without a CAN reply
    grip: bool          # a grip is held and the headset link is alive
    a: bool             # A pressed this cycle
    b: bool             # B pressed this cycle
    hands: tuple        # controller positions (robot frame) or None
    stick: tuple = (0.0, 0.0)  # right thumbstick: tunes power and range


class CalibrationStopped(Exception):
    pass


@dataclasses.dataclass
class MoveResult:
    ok: bool = True
    reason: str = ""
    samples: list = dataclasses.field(default_factory=list)

    def fail(self, reason: str) -> None:
        self.ok = False
        self.reason = reason


def min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


def min_jerk_rate(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return 30.0 * s * s * (1.0 - s) * (1.0 - s)


class PowerCalibration:
    """Host contract: read q_cmd / motors / limit_override / lines each cycle.

    host.capture_zero()  host.apply_signs()  host.joint_status(i) -> str
    """

    def __init__(self, profile: PowerProfile, gravity: GravityModel, host,
                 caps: np.ndarray | None = None):
        self.profile = profile
        self.gravity = gravity
        self.host = host
        if caps is None:
            caps = np.array([KIND_DEFAULTS[j.kind][4] for j in ARM_JOINTS])
        self.caps = np.minimum(np.asarray(caps, dtype=float), ABSOLUTE_MAX_TORQUE)
        self.lower, self.upper = profile.ranges(gravity.lower, gravity.upper)
        self.arcs = []
        for i, joint in enumerate(ARM_JOINTS):
            power = profile.power(i)
            spec = TESTS[(joint.side, joint.kind)]
            self.arcs.append(float(power.cal_arc if power.cal_arc is not None else spec.target))
            if power.cal_cap:
                self.caps[i] = min(float(power.cal_cap), ABSOLUTE_MAX_TORQUE)
        self.stick_latch = {"x": 0, "y": 0}
        self.visual = True   # False = operator cannot see the robot; decide by measurement
        self.only_retry = False   # True = skip joints that already passed
        self.directions_only = False   # True = confirm which way each joint goes, no power ladder
        self.q_cmd = None
        self.motors = False
        self.limit_override: dict[int, float] = {}
        self.lines = ["POWER CALIBRATION"]
        self.hint = ""
        self.paused = False
        self.testing = None
        self.probe_abort = ""
        self.tags = ["" for _ in ARM_JOINTS]
        self.done = False
        self.outcome = ""
        self.report_path = ""
        self.inputs: CalInputs | None = None
        self._saved_once = False
        self._gen = self._script()

    # ------------------------------------------------------------------ host API
    def step(self, inputs: CalInputs) -> None:
        self.inputs = inputs
        if self.done:
            return
        try:
            next(self._gen)
        except StopIteration:
            self.done = True
        except CalibrationStopped as exc:
            self.outcome = str(exc)
            self.lines = ["CALIBRATION STOPPED", str(exc)]
            self.motors = False
            self.done = True

    def abort(self, reason: str) -> None:
        if not self.done:
            self._gen.close()
            self.outcome = reason
            self.motors = False
            self.done = True

    # ------------------------------------------------------------------ script
    def _script(self):
        answer = yield from self._ask([
            "POWER CALIBRATION",
            "Each arm joint gets tested and",
            "given the torque it really needs.",
            "Hold a grip whenever the arm moves;",
            "letting go makes it hold still.",
            "A = start    B = leave",
        ])
        if answer == "B":
            self.outcome = "left before starting"
            return

        answer = yield from self._ask([
            "CAN YOU SEE THE ROBOT?",
            "A = yes, I will check each direction",
            "B = no (headset only): decide directions",
            "    by measurement alone",
        ])
        self.visual = answer == "A"
        self.hint = "" if self.visual else "Directions will be judged by movement, not by eye."

        # Once most joints pass, redoing all ten to reach the two that did not is a
        # long way round, and every extra run is more heat into a joint that is
        # already struggling.
        retry_only = yield from self._ask([
            "WHICH JOINTS?",
            "A = all of them",
            "B = only the ones that have",
            "    not passed before",
        ])
        self.only_retry = retry_only == "B"
        # Teleop moves a joint the way this check decides, so a wrong answer here is
        # felt later as an arm that mirrors the operator. Fixing that should not cost
        # a full power ladder on every joint.
        what = yield from self._ask([
            "WHAT TO TEST?",
            "A = power and direction (full)",
            "B = direction only (quick)",
            "    use this if teleop moves a",
            "    joint the wrong way",
        ])
        self.directions_only = what == "B"
        if self.only_retry:
            waiting = [ARM_JOINTS[i].label for i in range(N_JOINTS) if not self._passed_before(i)]
            self.hint = ("nothing left to retry - every joint has passed" if not waiting
                         else "retrying: " + ", ".join(waiting))

        answer = yield from self._ask([
            "ZERO POSE (motors are off)",
            "Let both arms hang straight down,",
            "elbows straight, claws forward.",
            "A = this is zero",
            "B = keep the start-up zero",
        ])
        if answer == "A":
            # A is pressed a lot on the way through these menus. Zero taken with the arms
            # not hanging (an elbow still bent, still sinking in damping) made teleop
            # refuse to arm, and every angle wrong - so a pose far from the start-up zero
            # is named and asked about again.
            far = [(ARM_JOINTS[i].label, float(self.inputs.q[i])) for i in range(N_JOINTS)
                   if np.isfinite(self.inputs.q[i]) and abs(self.inputs.q[i]) > 0.35]
            if far:
                label, angle = max(far, key=lambda item: abs(item[1]))
                answer = yield from self._ask([
                    "ZERO: ARE THE ARMS HANGING?",
                    f"{label} is {np.degrees(angle):+.0f} deg from the start-up",
                    "zero" + (f" ({len(far) - 1} more joints are off too)" if len(far) > 1 else ""),
                    "A = yes, they hang straight: zero here",
                    "B = no, keep the start-up zero",
                ])
        if answer == "A":
            self.host.capture_zero()
            self.hint = "Zero set."

        yield from self._reach()

        answer = yield from self._ask([
            "MOTORS ON",
            "The arms will hold where they are.",
            "Keep the area around the robot clear.",
            "Hold a grip + A = power on",
            "B = leave",
        ], a_needs_grip=True)
        if answer == "B":
            self.outcome = "left before motors on"
            return
        self.q_cmd = self.inputs.q.copy()
        self.motors = True
        yield from self._idle(1.2)
        holding = np.abs(self.inputs.tau)
        heavy = [(ARM_JOINTS[i].label, holding[i]) for i in range(N_JOINTS)
                 if holding[i] > abs(self.inputs.g[i]) + 1.0]
        if heavy:
            answer = yield from self._ask([
                "THESE JOINTS ARE CARRYING LOAD",
                ", ".join(f"{name} {value:.1f} Nm" for name, value in heavy[:4]),
                "That usually means an arm is not hanging",
                "freely, so zero is wrong. Let them hang,",
                "then: B = back out and set zero again",
                "A = carry on anyway",
            ])
            if answer == "B":
                self.motors = False
                self.outcome = "stopped to re-do the zero pose"
                return

        for side in ("left", "right"):
            bent = False
            for kind in ORDER:
                i = joint_index(side, kind)
                if self.only_retry and self._passed_before(i):
                    self.tags[i] = "KEPT"        # already passed; not touched again
                    continue
                if kind == "yaw":
                    elbow = joint_index(side, "elbow")
                    result = yield from self._move({elbow: ELBOW_BENT[side]}, MOVE_SPEED, None)
                    if not result.ok:
                        yield from self._snap_and_report(elbow, f"bending the elbow: {result.reason}")
                    bent = True
                yield from self._test_joint(i)
            if bent:
                self.hint = f"{side} arm going back down"
            yield from self._home(side)

        self.testing = None
        answer = yield from self._ask(self._summary() + ["A = motors off and finish"])
        self.motors = False
        self.outcome = "finished"

    # ------------------------------------------------------------------ steps
    def _reach(self):
        answer = yield from self._ask([
            "YOUR REACH (optional)",
            "Sets how far the robot hand moves",
            "for each cm your hand moves.",
            "Stand tall, arms hanging straight.",
            "A = record    B = skip",
        ])
        if answer == "B":
            return
        down = yield from self._sample_hands()
        answer = yield from self._ask([
            "Raise both arms straight FORWARD",
            "to shoulder height, elbows straight.",
            "A = record    B = skip",
        ])
        if answer == "B":
            return
        forward = yield from self._sample_hands()
        chords = [float(np.linalg.norm(f - d)) for d, f in zip(down, forward)
                  if d is not None and f is not None]
        if not chords or max(chords) < 0.2:
            self.hint = "Reach not recorded: controllers not tracked or barely moved."
            return
        human_reach = float(np.mean(chords)) / math.sqrt(2.0)
        self.profile.motion_scale = round(float(np.clip(ROBOT_REACH / human_reach, 0.25, 1.5)), 2)
        self.hint = f"Your reach {human_reach:.2f} m -> motion scale {self.profile.motion_scale:.2f}"
        self._save()

    def _test_joint(self, i: int):
        joint = ARM_JOINTS[i]
        test = TESTS[(joint.side, joint.kind)]
        self.testing = i
        aborted = 0
        while True:
            if aborted >= 3:
                self.tags[i] = "DIR!"
                self._record(i, {"verdict": "direction check kept failing; skipped",
                                 "status": self.host.joint_status(i)})
                yield from self._ask([f"{joint.label}: direction check kept failing.",
                                      "Skipping it. Check the joint by hand.",
                                      "A = next joint"])
                return
            self.tags[i] = "NEXT"
            answer = yield from self._ask([
                f"{joint.label}   ({joint.bus} ID {joint.device_id})",
                f"It will {test.motion}.",
                "Hold a grip + A = run",
                "B = skip this joint",
            ], a_needs_grip=True, tune=i)
            if answer == "B":
                self.tags[i] = "SKIP"
                self._record(i, {"verdict": "skipped by operator"})
                return
            self.tags[i] = "DIR?"
            answer = yield from self._wiggle_ask(i, test, [
                f"{joint.label}: WATCH IT MOVE",
                f"It should {test.motion}.",
                "A = yes, that is right",
                "B = no, the other way",
            ] if self.visual else [
                f"{joint.label}: checking which way it moves",
                "(no answer needed)",
            ])
            if answer is None:
                aborted += 1
                continue
            if answer == "SKIP":
                self.tags[i] = "STILL"
                self._record(i, {"verdict": "did not move in the direction check; skipped",
                                 "status": self.host.joint_status(i)})
                return
            if answer in ("B", "REVERSE"):
                why = "you said wrong way" if answer == "B" else "it was blocked that way, free the other"
                self._flip(i)
                if not self.visual:
                    self.hint = f"{joint.label}: direction flipped ({why})"
                    self.tags[i] = "TEST"
                    answer = "A"
                else:
                    answer = yield from self._wiggle_ask(i, test, [
                        f"{joint.label}: direction FLIPPED",
                        f"({why})",
                        f"Does it {test.motion} now?",
                        "A = yes",
                        "B = no (skip, check wiring/ID)",
                    ])
                if answer != "A":
                    self._flip(i)  # direction still unknown: keep the original sign
                if answer in ("B", "SKIP", "REVERSE"):
                    self.tags[i] = "DIR!"
                    self._record(i, {"verdict": "direction unclear", "flipped": False})
                    return
                if answer is None:
                    aborted += 1
                    continue
            if self.directions_only:
                self.tags[i] = "DIR"
                self._record(i, {"verdict": "direction confirmed",
                                 "flipped": self.profile.power(i).axis_flip})
                return
            self.tags[i] = "TEST"
            while True:
                outcome = yield from self._ladder(i, test)
                if not outcome["flip"]:
                    break
                answer = yield from self._ask([
                    f"{joint.label}:",
                    "blocked that way, free the other way.",
                    "A = flip its direction and test again",
                    "B = leave it as it is",
                ])
                if answer != "A":
                    break
                self._flip(i)
                self.tags[i] = "TEST"
            answer = yield from self._ask([
                f"{joint.label}: {outcome['verdict']}",
                "A = next joint    B = run again",
            ], tune=i)
            if answer == "A":
                return

    def _ladder(self, i: int, test: JointTest):
        joint = ARM_JOINTS[i]
        power = self.profile.power(i)
        cap = float(self.caps[i])
        home = float(self.q_cmd[i])
        target = float(np.clip(self.arcs[i], self.lower[i] + 0.05, self.upper[i] - 0.05))
        need_model = self.gravity.peak_along(self.q_cmd, i, target) * max(1.0, power.gravity_scale)
        start = min(cap, max(power.floor, ceil_to(1.15 * need_model + 0.5, LEVEL_STEP)))
        levels = [start]
        while levels[-1] + LEVEL_STEP <= cap + 1e-9:
            levels.append(levels[-1] + LEVEL_STEP)
        if levels[-1] < cap - 1e-9:
            levels.append(cap)

        attempts = []
        passed = None
        blocked = False
        free_other_way = None
        n = 0
        rescaled = False
        while n < len(levels):
            level = levels[n]
            self.limit_override[i] = level
            self.lines = [
                f"{joint.label}: POWER TEST",
                f"Arc to {math.degrees(target):+.0f} deg at {level:.1f} Nm",
                f"(model needs ~{need_model:.1f} Nm)",
                "Keep holding a grip.",
            ]
            out = yield from self._move({i: target}, ARC_SPEED, i, "out")
            top = MoveResult()
            if out.ok:
                top = yield from self._hold(1.0, i, "top")
            if not (out.ok and top.ok):
                self._snap_lagging(i)
            back = yield from self._move({i: home}, ARC_SPEED, i, "back")
            if not back.ok:
                yield from self._recover(i, home)
            yield from self._hold(0.4, i, "rest")
            attempt = self._score(i, level, home, target, out, top, back)
            attempts.append(attempt)
            if attempt["pass"]:
                passed = attempt
                break
            if not attempt["saturated"]:
                # Torque to spare, so the lag is feed-forward error, not power.
                fit = fit_gravity(attempt["samples"])
                scale = fit.get("scale") if fit else None
                if scale is not None and abs(scale - power.gravity_scale) > 0.03 and not rescaled:
                    power.gravity_scale = scale
                    rescaled = True
                    self.hint = f"gravity x{scale:.2f} measured; same {level:.1f} Nm again"
                    continue
                if out.ok and top.ok and attempt["progress"] > 0.9:
                    attempt["pass"] = True
                    attempt["note"] = f"passed with {attempt['why']}"
                    passed = attempt
                    break
            stuck = [a for a in attempts if a["saturated"] and a["progress"] < 0.05]
            if len(stuck) >= 2:
                # It has not moved at all twice at full torque. Rather than keep
                # climbing into what may be an end stop, see if the other way is free.
                self.hint = f"{joint.label}: no movement - checking the other direction"
                probe = yield from self._probe(i, float(self.q_cmd[i]),
                                               -math.copysign(REVERSE_PROBE, target - home))
                if probe is None:
                    break
                free_other_way = probe >= 0.5 * REVERSE_PROBE
                if not free_other_way and levels[n] < cap - 1e-9:
                    # Blocked both ways at this level. Stiction can still let go
                    # higher up, so try the ceiling once before calling it stuck.
                    self.hint = f"{joint.label}: blocked both ways, one try at {cap:.1f} Nm"
                    levels = levels[: n + 1] + [cap]
                    n += 1
                    continue
                blocked = True
                break
            saturated = [a for a in attempts if a["saturated"]]
            if (len(saturated) >= 2 and saturated[-2]["progress"] >= 0.1
                    and saturated[-1]["progress"] <= saturated[-2]["progress"] + 0.03):
                # It moved, then stopped at the same spot with more torque: an end
                # stop or an obstruction. (A joint that has not moved at all keeps
                # climbing: static friction can let go at a higher level.)
                blocked = True
                break
            rescaled = False
            n += 1
            if n < len(levels):
                self.hint = f"{level:.1f} Nm was not enough ({attempt['why']}); trying {levels[n]:.1f} Nm"
        self.limit_override.pop(i, None)
        verdict = self._conclude(i, attempts, passed, cap, need_model, blocked, free_other_way)
        return {"verdict": verdict, "flip": bool(free_other_way)}

    def _conclude(self, i, attempts, passed, cap, need_model, blocked, free_other_way=None) -> str:
        power = self.profile.power(i)
        default_headroom = KIND_DEFAULTS[ARM_JOINTS[i].kind][3]
        rows = [s for a in ([passed] if passed else attempts) for s in a["samples"]]
        fit = fit_gravity(rows)
        summary = {
            "time": dt.datetime.now().isoformat(timespec="seconds"),
            "levels_tried": [a["level"] for a in attempts],
            "attempts": [{k: v for k, v in a.items() if k != "samples"} for a in attempts],
            "model_need_nm": round(need_model, 2),
            "fit": fit,
            "flipped": power.axis_flip,
        }
        if passed:
            need = passed["peak_tau"]
            power.max_torque = float(min(cap, max(passed["level"], ceil_to(1.2 * need + 0.5, 0.25))))
            if fit and fit.get("scale") is not None:
                power.gravity_scale = fit["scale"]
            if fit:
                power.headroom = float(min(power.max_torque, max(default_headroom, ceil_to(fit["friction"] + 1.0, 0.25))))
            power.calibrated = summary["time"]
            verdict = f"OK - peak {need:.1f} Nm, max set {power.max_torque:.2f} Nm"
            if fit and fit.get("scale") is not None:
                verdict += f", gravity x{power.gravity_scale:.2f}"
            self.tags[i] = f"OK {power.max_torque:.1f}"
        else:
            power.max_torque = cap
            power.calibrated = None
            if fit and fit.get("scale") is not None and fit.get("r2", 0.0) >= 0.9 and not blocked:
                power.gravity_scale = fit["scale"]
            best = max(a["progress"] for a in attempts)
            tried = max(a["level"] for a in attempts)
            if free_other_way:
                verdict = (f"BLOCKED going that way at {tried:.1f} Nm, but free the other way: "
                           "its direction is probably flipped")
                self.tags[i] = "FLIP?"
            elif free_other_way is False:
                status = self.host.joint_status(i)
                verdict = f"NOT MOVING either way at {tried:.1f} Nm ({status})"
                self.tags[i] = "STUCK"
            elif blocked and best >= 0.1:
                levels = [a["level"] for a in attempts if a["saturated"]][-2:]
                verdict = (f"BLOCKED at {best * 100:.0f}% of the arc (same spot at "
                           f"{levels[0]:.1f} and {levels[1]:.1f} Nm): wrong direction or obstruction")
                self.tags[i] = "BLOCKED"
            elif best < 0.1:
                status = self.host.joint_status(i)
                verdict = f"NOT MOVING at {tried:.1f} Nm ({status})"
                self.tags[i] = "STUCK"
                summary["status"] = status
            else:
                verdict = f"TOO WEAK at {cap:.1f} Nm (reached {best * 100:.0f}%)"
                self.tags[i] = "WEAK"
        summary["verdict"] = verdict
        summary["visual_check"] = self.visual
        summary["home_rad"] = round(float(self.q_cmd[i]), 3)
        summary["probe_abort"] = self.probe_abort or None
        summary["free_other_way"] = free_other_way
        summary["arc_deg"] = round(math.degrees(self.arcs[i]), 1)
        summary["cap"] = round(float(self.caps[i]), 2)
        summary["max_torque"] = power.max_torque
        summary["gravity_scale"] = power.gravity_scale
        summary["headroom"] = power.headroom
        self._record(i, summary)
        return verdict

    def _score(self, i, level, home, target, out, top, back) -> dict:
        arc = target - home
        moving = out.samples + top.samples
        progress = 0.0
        max_err = 0.0
        peak_tau = 0.0
        saturated = 0
        for phase, q_cmd, q, tau, g, v in moving:
            progress = max(progress, (q - home) / arc if abs(arc) > 1e-6 else 1.0)
            peak_tau = max(peak_tau, abs(tau))
            saturated += abs(tau) >= 0.92 * level
            if phase == "out":
                max_err = max(max_err, abs(q_cmd - q))
        tail = [abs(s[1] - s[2]) for s in top.samples[-15:]]
        top_err = float(np.mean(tail)) if tail else float("inf")
        saturated_frac = saturated / max(1, len(moving))
        why = ""
        if not out.ok:
            why = out.reason
        elif not top.ok:
            why = top.reason
        elif max_err >= PASS_TRACK:
            why = f"lagged {max_err:.2f} rad"
        elif top_err >= PASS_HOLD:
            why = f"sagged {top_err:.2f} rad at the top"
        return {
            "level": level,
            "pass": bool(not why),
            "why": why,
            "progress": round(float(min(progress, 1.0)), 3),
            "max_err": round(max_err, 3),
            "top_err": round(top_err, 3) if math.isfinite(top_err) else None,
            "peak_tau": round(peak_tau, 3),
            "saturated_frac": round(saturated_frac, 3),
            # bool(), because these come from numpy comparisons and numpy's bool is
            # not JSON's: an uncoerced one threw away a whole calibration on 09-20.
            "saturated": bool(saturated_frac >= 0.15 or (not out.ok and peak_tau >= 0.9 * level)),
            "back_ok": bool(back.ok),
            "samples": out.samples + top.samples + back.samples,
        }

    def _home(self, side: str):
        idx = [i for i, j in enumerate(ARM_JOINTS) if j.side == side]
        result = yield from self._move({i: 0.0 for i in idx}, MOVE_SPEED, None)
        if not result.ok:
            self._snap_lagging()
            self.hint = f"{side} arm did not reach home: {result.reason}"

    def _recover(self, i: int, home: float):
        joint = ARM_JOINTS[i]
        for _ in range(3):
            self._snap_lagging(i)
            answer = yield from self._ask([
                f"{joint.label} did not come back.",
                "A = try again at full power",
                "B = stop the motors",
            ])
            if answer == "B":
                raise CalibrationStopped(f"{joint.label} could not return; stopped by operator")
            self.limit_override[i] = float(self.caps[i])
            result = yield from self._move({i: home}, MOVE_SPEED * 0.5, i, "recover")
            if result.ok:
                return
        raise CalibrationStopped(f"{joint.label} could not return to its start angle")

    def _snap_and_report(self, i: int, reason: str):
        self._snap_lagging(i)
        yield from self._ask([f"{ARM_JOINTS[i].label}: {reason}", "A = continue anyway"])

    # ------------------------------------------------------------------ motion helpers
    def _move(self, targets: dict[int, float], speed: float, watch: int | None, phase: str = "move"):
        result = MoveResult()
        start = {i: float(self.q_cmd[i]) for i in targets}
        span = max(abs(targets[i] - start[i]) for i in targets)
        duration = max(0.6, 1.875 * span / speed)
        t = 0.0
        waited = 0.0
        while t < duration:
            if not self.inputs.grip:
                self.paused = True
                waited += self.inputs.dt
                if waited > PAUSE_LIMIT:
                    result.fail("no grip held")
                    break
                yield
                continue
            self.paused = False
            waited = 0.0
            t = min(duration, t + self.inputs.dt)
            s = min_jerk(t / duration)
            for i in targets:
                self.q_cmd[i] = start[i] + (targets[i] - start[i]) * s
            yield
            reason = self._check(watch)
            if watch is not None:
                v_cmd = (targets[watch] - start[watch]) * min_jerk_rate(t / duration) / duration
                result.samples.append(self._sample(watch, phase, v_cmd))
            if reason:
                result.fail(reason)
                break
        self.paused = False
        return result

    def _hold(self, seconds: float, watch: int | None, phase: str):
        result = MoveResult()
        t = 0.0
        waited = 0.0
        while t < seconds:
            if self.inputs.grip:
                t += self.inputs.dt
                waited = 0.0
            else:
                waited += self.inputs.dt
                if waited > PAUSE_LIMIT:
                    result.fail("no grip held")
                    break
            self.paused = not self.inputs.grip
            yield
            reason = self._check(watch)
            if watch is not None:
                result.samples.append(self._sample(watch, phase, 0.0))
            if reason:
                result.fail(reason)
                break
        self.paused = False
        return result

    def _idle(self, seconds: float):
        t = 0.0
        while t < seconds:
            t += self.inputs.dt
            yield

    def _wiggle_ask(self, i: int, test: JointTest, lines: list[str]):
        """Rock joint i out and back until answered.

        Returns "A" / "B" for the direction question, "REVERSE" if it is blocked
        this way but free the other way (a flipped joint pushes into its own end
        stop), "RUN" / "SKIP" if it did not move at all, or None if aborted.
        """
        self.lines = lines
        base = float(self.q_cmd[i])
        amplitude = float(np.clip(base + test.nudge, self.gravity.lower[i] + 0.02,
                                  self.gravity.upper[i] - 0.02)) - base
        phase = 0.0
        excursion = 0.0
        still = False
        answer = None
        stall = abs(amplitude) + STALL_ERR
        while answer is None:
            if self.inputs.grip:
                phase += self.inputs.dt / WIGGLE_PERIOD
                self.paused = False
            else:
                self.paused = True
            self.q_cmd[i] = base + amplitude * 0.5 * (1.0 - math.cos(2.0 * math.pi * phase))
            yield
            reason = self._check(i, stall=stall, others=False)
            if reason:
                self.paused = False
                self._snap_lagging(i)
                yield from self._move({i: base}, MOVE_SPEED, None)
                yield from self._ask([f"{ARM_JOINTS[i].label}: {reason}", "A = back to the joint menu"])
                return None
            excursion = max(excursion, abs(self.inputs.q[i] - base))
            if (self.inputs.a or self.inputs.b) and phase < 1.0 and not still:
                self.hint = "Let it swing once first, then answer."
            if not still and phase >= 1.0 and excursion < 0.3 * abs(amplitude):
                self.q_cmd[i] = base
                reverse = yield from self._probe(i, base, -math.copysign(REVERSE_PROBE, amplitude))
                if reverse is None:
                    return None
                if reverse >= 0.5 * REVERSE_PROBE:
                    self.paused = False
                    return "REVERSE"
                still = True
                if not self.visual:
                    self.hint = f"{ARM_JOINTS[i].label}: did not move; running the power test anyway"
                    return "RUN"
                self.lines = [
                    f"{ARM_JOINTS[i].label}: NOT MOVING",
                    f"moved {math.degrees(excursion):.1f} of {math.degrees(abs(amplitude)):.0f} deg",
                    "A = power test anyway (more torque)",
                    "B = skip this joint",
                ]
            if not self.visual and phase >= 1.0 and answer is None:
                answer = "RUN" if still else "A"
            elif phase >= 1.0 or still:
                if self.inputs.a:
                    answer = "RUN" if still else "A"
                elif self.inputs.b:
                    answer = "SKIP" if still else "B"
        self.paused = False
        yield from self._move({i: base}, MOVE_SPEED, None)
        return answer

    def _probe(self, i: int, base: float, amplitude: float):
        """One slow rock toward base + amplitude and back, ignoring URDF limits
        (they are what is in doubt). Returns the largest excursion, or None."""
        phase = 0.0
        excursion = 0.0
        stall = abs(amplitude) + STALL_ERR
        self.probe_abort = ""
        while phase < 1.0:
            if self.inputs.grip:
                phase = min(1.0, phase + self.inputs.dt / WIGGLE_PERIOD)
            self.paused = not self.inputs.grip
            self.q_cmd[i] = base + amplitude * 0.5 * (1.0 - math.cos(2.0 * math.pi * phase))
            yield
            reason = self._check(i, stall=stall, others=False)
            if reason:
                self.probe_abort = reason
                self._snap_lagging(i)
                yield from self._move({i: base}, MOVE_SPEED, None)
                return None
            excursion = max(excursion, abs(self.inputs.q[i] - base))
        self.paused = False
        return excursion

    def _check(self, watch: int | None, stall: float = STALL_ERR, others: bool = True) -> str:
        inp = self.inputs
        silent = np.flatnonzero(inp.miss > 10)
        if silent.size:
            raise CalibrationStopped(f"{ARM_JOINTS[silent[0]].label}: no CAN reply")
        err = self.q_cmd - inp.q
        if watch is not None and abs(err[watch]) > stall:
            return f"not following ({err[watch]:+.2f} rad behind)"
        if not others:
            return ""
        slipped = [k for k in range(N_JOINTS) if k != watch and abs(err[k]) > SLIP_ERR]
        if slipped:
            k = slipped[0]
            return f"{ARM_JOINTS[k].label} slipped {err[k]:+.2f} rad"
        return ""

    def _snap_lagging(self, also: int | None = None) -> None:
        """After a failed move, hold joints where they are instead of where they were told to be."""
        err = np.abs(self.q_cmd - self.inputs.q)
        for k in range(N_JOINTS):
            if k == also or err[k] > 0.5 * SLIP_ERR:
                self.q_cmd[k] = self.inputs.q[k]

    def _sample(self, i: int, phase: str, v_cmd: float):
        inp = self.inputs
        return (phase, float(self.q_cmd[i]), float(inp.q[i]), float(inp.tau[i]), float(inp.g[i]), float(v_cmd))

    def _sample_hands(self, seconds: float = 0.6):
        points = ([], [])
        t = 0.0
        while t < seconds:
            yield
            t += self.inputs.dt
            for k in (0, 1):
                if self.inputs.hands[k] is not None:
                    points[k].append(np.asarray(self.inputs.hands[k], dtype=float))
        return [np.mean(p, axis=0) if len(p) >= 3 else None for p in points]

    def _ask(self, lines: list[str], a_needs_grip: bool = False, tune: int | None = None):
        self.lines = list(lines)
        if tune is not None:
            self.lines.append(self._tune_line(tune))
        while True:
            yield
            inp = self.inputs
            if tune is not None and self._tune(tune):
                self.lines = list(lines) + [self._tune_line(tune)]
            if inp.b:
                self.hint = ""
                if tune is not None:
                    self._store_tuning(tune)
                return "B"
            if inp.a:
                if a_needs_grip and not inp.grip:
                    self.hint = "Hold a grip while you press A."
                    continue
                self.hint = ""
                if tune is not None:
                    self._store_tuning(tune)
                return "A"

    def _passed_before(self, i: int) -> bool:
        """Did the stored profile already record a passing ladder for this joint?"""
        result = self.profile.power(i).result or {}
        return any(attempt.get("pass") for attempt in (result.get("attempts") or []))

    def _tune_line(self, i: int) -> str:
        return (f"stick up/down: power {self.caps[i]:.1f} Nm   "
                f"left/right: range {math.degrees(abs(self.arcs[i])):.0f} deg")

    def _tune(self, i: int) -> bool:
        """Thumbstick steps: power (up/down) and test range (left/right)."""
        x, y = self.inputs.stick
        changed = False
        step = self._stick_step("y", y)
        if step:
            self.caps[i] = float(np.clip(self.caps[i] + TORQUE_STEP * step, 1.0, ABSOLUTE_MAX_TORQUE))
            changed = True
        step = self._stick_step("x", x)
        if step:
            span = abs(self.arcs[i]) + RANGE_STEP * step
            reach = max(abs(self.lower[i]), abs(self.upper[i]))
            span = float(np.clip(span, RANGE_STEP, max(RANGE_STEP, reach - 0.05)))
            self.arcs[i] = math.copysign(span, self.arcs[i])
            changed = True
        return changed

    def _stick_step(self, axis: str, value: float) -> int:
        """One step per push past 0.6; the stick must return under 0.3 to repeat."""
        latched = self.stick_latch[axis]
        if abs(value) < 0.3:
            self.stick_latch[axis] = 0
            return 0
        if latched or abs(value) < 0.6:
            return 0
        self.stick_latch[axis] = 1
        return 1 if value > 0 else -1

    def _store_tuning(self, i: int) -> None:
        power = self.profile.power(i)
        if power.cal_cap != self.caps[i] or power.cal_arc != self.arcs[i]:
            power.cal_cap = float(self.caps[i])
            power.cal_arc = float(self.arcs[i])
            self._save()

    # ------------------------------------------------------------------ bookkeeping
    def _flip(self, i: int) -> None:
        power = self.profile.power(i)
        power.axis_flip = not power.axis_flip
        if self.q_cmd is not None:
            self.q_cmd[i] = -self.q_cmd[i]
        self.host.apply_signs()

    def _record(self, i: int, summary: dict) -> None:
        self.profile.power(i).result = summary
        self._save()

    def _save(self) -> None:
        self.profile.save(backup=not self._saved_once)
        self._saved_once = True

    def _write_report(self) -> str:
        stamp = dt.datetime.now()
        path = REPO_ROOT / "arm_validation" / f"calibration_{stamp:%Y%m%d_%H%M}.md"
        rows = ["# Arm power calibration", "",
                f"{stamp:%Y-%m-%d %H:%M}. Profile: `{self.profile.path}`",
                f"Hand mapping: {'swapped' if self.profile.swap_hands else 'straight'}, "
                f"motion scale {self.profile.motion_scale:.2f}", "",
                "| joint | bus/ID | verdict | max torque | gravity | flipped | arc | levels tried |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for i, joint in enumerate(ARM_JOINTS):
            power = self.profile.power(i)
            result = power.result or {}
            rows.append(
                f"| {joint.label} | {joint.bus} {joint.device_id} | {result.get('verdict', 'not tested')} | "
                f"{power.max_torque:.2f} Nm | x{power.gravity_scale:.2f} | "
                f"{'yes' if power.axis_flip else 'no'} | {result.get('arc_deg', '-')} deg | "
                f"{result.get('levels_tried', '-')} |")
        rows += ["", "Edit `configs/arm_power_profile.json` to change any of it by hand:",
                 "`max_torque` (power ceiling), `floor` (torque when hanging), `headroom`,",
                 "`gravity_scale`, `axis_flip`, `cal_cap` / `cal_arc` (calibration), and",
                 "`range_lower` / `range_upper` (teleop range, radians)."]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(rows) + "\n")
        return str(path)

    def _summary(self) -> list[str]:
        try:
            self.report_path = self._write_report()
        except OSError as exc:
            self.report_path = f"report not written: {exc}"
        lines = ["CALIBRATION DONE", f"motion scale {self.profile.motion_scale:.2f}"]
        for i, joint in enumerate(ARM_JOINTS):
            power = self.profile.power(i)
            tag = self.tags[i] or "-"
            lines.append(f"{joint.label:8s} {tag:8s} max {power.max_torque:.1f}  g x{power.gravity_scale:.2f}")
        return lines


def fit_gravity(rows: list) -> dict | None:
    """Least squares tau = scale * g + friction * sign(v) + bias.

    Only samples taken near cruise speed while tracking: at rest, static friction
    can hold any torque within its band, which would read as extra gravity.
    """
    usable = [(tau, g, v) for _, q_cmd, q, tau, g, v in rows
              if abs(q_cmd - q) < PASS_TRACK and abs(v) >= 0.5 * ARC_SPEED]
    if len(usable) < 20:
        return None
    tau, g, v = (np.array(col, dtype=float) for col in zip(*usable))
    direction = np.sign(v)
    if np.ptp(g) < 0.4 or len(set(direction)) < 2:
        friction = float(np.mean(np.abs(tau - np.mean(tau)))) if len(set(direction)) == 2 else 0.0
        return {"scale": None, "friction": round(min(friction, 3.0), 3), "n": len(usable),
                "note": "gravity range too small to fit"}
    design = np.column_stack([g, direction, np.ones_like(g)])
    coef, *_ = np.linalg.lstsq(design, tau, rcond=None)
    residual = tau - design @ coef
    r2 = 1.0 - float(np.var(residual)) / max(float(np.var(tau)), 1e-9)
    scale = float(coef[0])
    fit = {"raw_scale": round(scale, 3), "friction": round(float(min(abs(coef[1]), 3.0)), 3),
           "bias": round(float(coef[2]), 3), "r2": round(r2, 3), "n": len(usable)}
    if scale < 0.3 or r2 < 0.5:
        fit["scale"] = None
        fit["note"] = "poor fit; gravity scale unchanged"
    else:
        fit["scale"] = round(float(np.clip(scale, 0.5, 2.0)), 3)
    return fit
