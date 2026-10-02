"""Re-measure the operator's reach every time teleop is armed (triple X).

Sitting and standing change how far the hands travel, so the motion scale (robot
cm per operator cm) is taken fresh before each arming: arms hanging, then arms
straight forward; the chord between the two gives the arm length. B skips and
keeps the stored scale. Same host contract as the calibration menus.
"""

from __future__ import annotations

import math

import numpy as np

from arm_calibration import ROBOT_REACH
from servo_calibration import _MenuBase


class ReachCheck(_MenuBase):
    def __init__(self, profile):
        self.profile = profile
        self.measured = False
        self.demo = None
        super().__init__()

    def _script(self):
        scale = self.profile.motion_scale
        self.demo = "down"            # the headset's robot shows the pose being asked for
        answer = yield from self._ask([
            "REACH CHECK 1 of 2 (no buttons to hold)",
            "Let both arms hang straight down",
            "at your sides, controllers in hand,",
            "then press A once.",
            f"B = skip it (keep sensitivity {scale:.2f})",
            "Triple X = cancel",
        ])
        if answer == "B":
            self.outcome = "kept"
            return
        down = yield from self._sample_hands()
        self.demo = "forward"
        answer = yield from self._ask([
            "REACH CHECK 2 of 2",
            "Raise both arms straight out in FRONT",
            "to shoulder height, elbows straight,",
            "hold them there and press A once.",
            f"B = skip it (keep sensitivity {scale:.2f})",
        ])
        if answer == "B":
            self.outcome = "kept"
            return
        forward = yield from self._sample_hands()
        chords = [float(np.linalg.norm(f - d)) for d, f in zip(down, forward)
                  if d is not None and f is not None]
        if not chords or max(chords) < 0.2:
            self.hint = "Reach not measured (controllers not tracked or barely moved); scale kept."
            self.outcome = "kept"
            return
        reach = float(np.mean(chords)) / math.sqrt(2.0)
        self.profile.motion_scale = round(float(np.clip(ROBOT_REACH / reach, 0.25, 1.5)), 2)
        self.profile.save()
        self.measured = True
        self.outcome = f"reach {reach:.2f} m -> scale {self.profile.motion_scale:.2f}"

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
