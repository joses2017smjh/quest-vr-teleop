"""Headset setup for the claws and the camera pan/tilt, next to the power calibration.

Triple Y (motors stopped) opens CalibrationChooser: A = arm power calibration,
B = this setup. For each servo the right thumbstick moves it (up/down 20 us per
click, left/right 5 us) and A records the position shown on the panel; B skips
that mark and keeps the old value. The arm motors stay off throughout.

Both classes follow PowerCalibration's host contract (lines, hint, done, motors,
q_cmd, limit_override, tags, testing, paused, outcome, step, abort), so
run_teleop.py drives them the same way.
"""

from __future__ import annotations

from arm_power import N_JOINTS
from servos import MAX_US, MIN_US, ROLES, SERVOS, ServoLink, load_limits, save_limits

COARSE_US = 20
FINE_US = 5


class _MenuBase:
    def __init__(self):
        self.lines: list[str] = []
        self.hint = ""
        self.done = False
        self.outcome = ""
        self.motors = False
        self.q_cmd = None
        self.limit_override: dict[int, float] = {}
        self.tags = ["" for _ in range(N_JOINTS)]
        self.testing = None
        self.paused = False
        self.report_path = ""
        self._saved_once = False
        self.inputs = None
        self._gen = self._script()

    def step(self, inputs) -> None:
        self.inputs = inputs
        if self.done:
            return
        try:
            next(self._gen)
        except StopIteration:
            self.done = True

    def abort(self, reason: str) -> None:
        if not self.done:
            self._gen.close()
            self.outcome = reason
            self.done = True

    def _script(self):
        yield

    def _ask(self, lines: list[str]):
        self.lines = list(lines)
        while True:
            yield
            if self.inputs.b:
                return "B"
            if self.inputs.a:
                return "A"


class CalibrationChooser(_MenuBase):
    """First screen after triple Y; `choice` is "power" or "servos" once answered."""

    def __init__(self, servos: ServoLink | None):
        self.servos = servos
        self.choice = ""
        super().__init__()

    def _script(self):
        nano = "Nano connected" if self.servos is not None and self.servos.ready else "Nano NOT connected"
        answer = yield from self._ask([
            "CALIBRATION MENU",
            "A = arm power calibration",
            f"B = claws + camera setup ({nano})",
            "Triple Y = close",
        ])
        self.choice = "power" if answer == "A" else "servos"
        self.outcome = f"chose {self.choice}"


class ServoCalibration(_MenuBase):
    def __init__(self, servos: ServoLink):
        self.servos = servos
        self.data = load_limits()
        self.latch = {"x": 0, "y": 0}
        super().__init__()

    def _script(self):
        if not self.servos.ready:
            yield from self._ask([
                "CLAWS + CAMERA SETUP",
                "The claw/camera Nano is not connected.",
                self.servos.status[:44],
                "A or B = close",
            ])
            self.outcome = "Nano not connected"
            return
        answer = yield from self._ask([
            "CLAWS + CAMERA SETUP (arm motors stay off)",
            "Each servo is moved with the RIGHT stick:",
            f"  up/down = {COARSE_US} us, left/right = {FINE_US} us",
            "A = record the position asked for",
            "B = skip it (keeps the old value)",
            "A = start    B = leave",
        ])
        if answer == "B":
            self.outcome = "left before starting"
            return
        claws = self.data.setdefault("claws", {})
        for servo in SERVOS:
            self.current = servo.key          # the headset's robot lights that hand
            entry = claws.setdefault(servo.key, {})
            entry["role"] = ROLES[servo.key]
            if servo.key == "D3":
                entry["side"] = "left"
            elif servo.key == "D7":
                entry["side"] = "right"
            us = int(entry.get(servo.marks[0][0]) or 1500)
            self.servos.move(servo.pin, us)
            for n, (field, what) in enumerate(servo.marks, start=1):
                if field in entry:
                    us = int(entry[field])
                    self.servos.move(servo.pin, us)
                recorded, us = yield from self._jog(servo, us, n, what, entry.get(field))
                if recorded:
                    entry[field] = us
                    save_limits(self.data)
                    self._saved_once = True
                    self.hint = f"{servo.label}: {what.split(' (')[0]} = {us} us"
                else:
                    self.hint = f"{servo.label}: kept {entry.get(field, 'no value')}"
            # back to home: a claw goes limp (it rests closed, and only it is attached
            # then); the camera goes to its home mark, the first one asked
            if "claw" in entry["role"]:
                self.servos.off()
            else:
                home = entry.get(servo.marks[0][0])
                if home:
                    self.servos.move(servo.pin, int(home))
        yield from self._ask(self._summary() + ["A = finish"])
        self.outcome = "finished"

    def _jog(self, servo, us: int, n: int, what: str, old):
        total = len(servo.marks)
        entry = self.data["claws"].get(servo.key, {})
        header = [f"{servo.label}  ({n}/{total})", f"Move it to: {what}"]
        held = {"x": None, "y": None}      # when each stick axis was pushed, for auto-repeat
        last = {"x": 0.0, "y": 0.0}
        while True:
            saved = "   ".join(f"{f.split('_')[0]} {entry.get(f, '-')}" for f, _ in servo.marks)
            self.lines = header + [f"now: {us} us", f"saved: {saved}",
                                   "hold stick: up/down fast, left/right fine",
                                   "A = record    B = skip (keep old)"]
            yield
            inp = self.inputs
            if inp.a:
                return True, us
            if inp.b:
                return False, us
            x, y = inp.stick
            step = (self._repeat("y", y, held, last, inp.now, COARSE_US, 50)
                    + self._repeat("x", x, held, last, inp.now, FINE_US, FINE_US))
            if step:
                new = max(servo.min_us, min(servo.max_us, us + step))
                if new != us:
                    us = new
                    if not self.servos.move(servo.pin, us):
                        self.hint = "lost the Nano: " + self.servos.status[:40]

    @staticmethod
    def _repeat(axis, value, held, last, now, step, fast_step) -> int:
        """First push: one step. Held: repeats after 0.35 s, every 0.08 s at full stick
        (slower when pushed lightly), and the step grows to fast_step after 1.2 s."""
        if abs(value) < 0.35:
            held[axis] = None
            return 0
        sign = 1 if value > 0 else -1
        if held[axis] is None:
            held[axis] = now
            last[axis] = now
            return sign * step
        hold = now - held[axis]
        if hold < 0.35:
            return 0
        period = 0.08 / max(0.3, min(1.0, (abs(value) - 0.35) / 0.55))
        if now - last[axis] < period:
            return 0
        last[axis] = now
        return sign * (fast_step if hold > 1.2 else step)

    def _summary(self) -> list[str]:
        lines = ["CLAWS + CAMERA SAVED"]
        for servo in SERVOS:
            entry = self.data["claws"].get(servo.key, {})
            vals = " ".join(f"{f.split('_')[0]} {entry.get(f, '-')}" for f, _ in servo.marks)
            lines.append(f"{servo.label}: {vals}")
        return lines
