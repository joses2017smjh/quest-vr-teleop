"""Per-joint power for the as-built arms.

"Power" is joint torque, split in two:
  * gravity feed-forward  tau_ff = gravity_scale * g(q), from the URDF
  * a dynamic torque limit  |tau_ff| + headroom, never below `floor` and never
    above the joint's calibrated `max_torque`

The headset calibration (arm_calibration.py) measures what each joint needs and
writes configs/arm_power_profile.json; run_teleop.py loads it. With no file,
the per-kind defaults below apply.

Frames: ARM_JOINTS order (same as Bimanual.joints and the IK output), URDF
sign convention, zero = arms hanging straight down at start-up.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pinocchio as pin

REPO_ROOT = Path(__file__).resolve().parents[2]
URDF_PATH = REPO_ROOT / (
    "source/berkeley_humanoid_lite_assets/data/robots/berkeley_humanoid/"
    "berkeley_humanoid_lite/urdf/berkeley_humanoid_lite.urdf"
)
PROFILE_PATH = REPO_ROOT / "configs/arm_power_profile.json"

# Nothing in the teleop stack ever lets an arm joint exceed this (Nm).
ABSOLUTE_MAX_TORQUE = 10.0
# How far a profile may push a joint past its URDF range (rad).
RANGE_MARGIN = 0.35


@dataclasses.dataclass(frozen=True)
class ArmJoint:
    key: str
    label: str
    bus: str
    device_id: int
    urdf: str
    axis_sign: int
    kind: str
    side: str


# Axis signs are Bimanual.joint_axis_directions. The right wrist answers as
# ID 12 on this unit (docs/repo: 10).
ARM_JOINTS = (
    ArmJoint("left_shoulder_pitch", "L pitch", "can0", 1, "arm_left_shoulder_pitch_joint", +1, "pitch", "left"),
    ArmJoint("left_shoulder_roll", "L roll", "can0", 3, "arm_left_shoulder_roll_joint", +1, "roll", "left"),
    ArmJoint("left_shoulder_yaw", "L yaw", "can0", 5, "arm_left_shoulder_yaw_joint", -1, "yaw", "left"),
    ArmJoint("left_elbow", "L elbow", "can0", 7, "arm_left_elbow_pitch_joint", -1, "elbow", "left"),
    ArmJoint("left_wrist_yaw", "L wrist", "can0", 9, "arm_left_elbow_roll_joint", -1, "wrist", "left"),
    ArmJoint("right_shoulder_pitch", "R pitch", "can1", 2, "arm_right_shoulder_pitch_joint", -1, "pitch", "right"),
    ArmJoint("right_shoulder_roll", "R roll", "can1", 4, "arm_right_shoulder_roll_joint", +1, "roll", "right"),
    ArmJoint("right_shoulder_yaw", "R yaw", "can1", 6, "arm_right_shoulder_yaw_joint", -1, "yaw", "right"),
    ArmJoint("right_elbow", "R elbow", "can1", 8, "arm_right_elbow_pitch_joint", +1, "elbow", "right"),
    ArmJoint("right_wrist_yaw", "R wrist", "can1", 12, "arm_right_elbow_roll_joint", -1, "wrist", "right"),
)
N_JOINTS = len(ARM_JOINTS)


def _plain(value):
    """numpy scalars on their way to JSON.

    Every measurement here arrives from numpy, and numpy's bool and int are not
    Python's, so json refuses them. This runs only for what json could not handle
    itself, and it matters because the write is the last step of a calibration:
    losing it means the whole arm was done for nothing.
    """
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    raise TypeError(f"cannot store {type(value).__name__} in the power profile")


def joint_index(side: str, kind: str) -> int:
    for i, joint in enumerate(ARM_JOINTS):
        if joint.side == side and joint.kind == kind:
            return i
    raise KeyError((side, kind))


# Model gravity torque peaks (URDF, no claws): ~2.9 Nm for shoulder pitch at
# horizontal and shoulder roll at 75 deg, ~0.8 Nm for elbow and yaw, ~0 wrist.
# (kp, kd, floor, headroom, max) per joint kind
KIND_DEFAULTS = {
    "pitch": (30.0, 2.0, 3.0, 1.5, 6.0),
    "roll": (30.0, 2.0, 3.0, 1.5, 6.0),
    "yaw": (30.0, 2.0, 2.0, 1.0, 4.0),
    "elbow": (30.0, 2.0, 2.0, 1.0, 4.0),
    "wrist": (30.0, 2.0, 1.2, 0.8, 2.5),
}


@dataclasses.dataclass
class JointPower:
    kp: float
    kd: float
    floor: float
    headroom: float
    max_torque: float
    gravity_scale: float = 1.0
    axis_flip: bool = False
    calibrated: str | None = None
    result: dict | None = None
    cal_cap: float | None = None      # torque ceiling the calibration may try
    cal_arc: float | None = None      # angle the calibration sweeps to, rad
    range_lower: float | None = None  # teleop soft limits; None = the URDF value
    range_upper: float | None = None

    @classmethod
    def default(cls, kind: str) -> JointPower:
        kp, kd, floor, headroom, max_torque = KIND_DEFAULTS[kind]
        return cls(kp=kp, kd=kd, floor=floor, headroom=headroom, max_torque=max_torque)

    def sanitized(self) -> JointPower:
        """Clamp values loaded from disk into ranges the driver accepts."""
        out = dataclasses.replace(self)
        out.kp = float(np.clip(out.kp, 0.0, 80.0))
        out.kd = float(np.clip(out.kd, 0.0, 5.0))
        out.max_torque = float(np.clip(out.max_torque, 0.5, ABSOLUTE_MAX_TORQUE))
        out.floor = float(np.clip(out.floor, 0.2, out.max_torque))
        out.headroom = float(np.clip(out.headroom, 0.0, out.max_torque))
        out.gravity_scale = float(np.clip(out.gravity_scale, 0.0, 2.0))
        out.axis_flip = bool(out.axis_flip)
        if out.cal_cap is not None:
            out.cal_cap = float(np.clip(out.cal_cap, 0.5, ABSOLUTE_MAX_TORQUE))
        if out.cal_arc is not None:
            out.cal_arc = float(np.clip(out.cal_arc, -3.2, 3.2))
        return out


class PowerProfile:
    def __init__(self, joints: dict[str, JointPower] | None = None,
                 motion_scale: float = 1.0, path: Path | None = None,
                 loaded: bool = False, swap_hands: bool = False):
        self.joints = {j.key: JointPower.default(j.kind) for j in ARM_JOINTS}
        self.joints.update(joints or {})
        self.motion_scale = motion_scale
        self.swap_hands = swap_hands
        self.path = path
        self.loaded = loaded

    @classmethod
    def load(cls, path: Path = PROFILE_PATH) -> PowerProfile:
        path = Path(path)
        if not path.exists():
            return cls(path=path)
        raw = json.loads(path.read_text())
        fields = {f.name for f in dataclasses.fields(JointPower)}
        joints = {}
        for joint in ARM_JOINTS:
            entry = raw.get("joints", {}).get(joint.key)
            if entry is None:
                continue
            base = dataclasses.asdict(JointPower.default(joint.kind))
            base.update({k: v for k, v in entry.items() if k in fields})
            joints[joint.key] = JointPower(**base).sanitized()
        motion_scale = float(np.clip(raw.get("motion_scale", 1.0), 0.2, 2.0))
        return cls(joints, motion_scale=motion_scale, path=path, loaded=True,
                   swap_hands=bool(raw.get("swap_hands", False)))

    def save(self, path: Path | None = None, backup: bool = False) -> Path:
        path = Path(path or self.path or PROFILE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        if backup and path.exists():
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        data = {
            "version": 1,
            "saved": dt.datetime.now().isoformat(timespec="seconds"),
            "note": "Written by the headset power calibration (scripts/teleop/arm_calibration.py).",
            "motion_scale": self.motion_scale,
            "swap_hands": self.swap_hands,
            "joints": {k: dataclasses.asdict(v) for k, v in self.joints.items()},
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=_plain) + "\n")
        tmp.replace(path)
        self.path = path
        self.loaded = True
        return path

    def power(self, i: int) -> JointPower:
        return self.joints[ARM_JOINTS[i].key]

    def column(self, name: str) -> np.ndarray:
        return np.array([float(getattr(self.power(i), name)) for i in range(N_JOINTS)])

    def axis_signs(self) -> np.ndarray:
        return np.array([
            joint.axis_sign * (-1.0 if self.power(i).axis_flip else 1.0)
            for i, joint in enumerate(ARM_JOINTS)
        ])

    def calibrated_count(self) -> int:
        return sum(1 for i in range(N_JOINTS) if self.power(i).calibrated)

    def ranges(self, urdf_lower: np.ndarray, urdf_upper: np.ndarray,
               margin: float = RANGE_MARGIN) -> tuple[np.ndarray, np.ndarray]:
        """Teleop soft limits: the URDF range unless the profile overrides it,
        and never more than `margin` beyond the URDF in either direction."""
        lower, upper = np.array(urdf_lower, dtype=float), np.array(urdf_upper, dtype=float)
        for i in range(N_JOINTS):
            power = self.power(i)
            if power.range_lower is not None:
                lower[i] = float(np.clip(power.range_lower, urdf_lower[i] - margin, urdf_upper[i]))
            if power.range_upper is not None:
                upper[i] = float(np.clip(power.range_upper, urdf_lower[i], urdf_upper[i] + margin))
        return lower, np.maximum(lower + 0.05, upper)

    def feed_forward(self, gravity_torque: np.ndarray) -> np.ndarray:
        return self.column("gravity_scale") * gravity_torque

    def dynamic_limits(self, tau_ff: np.ndarray) -> np.ndarray:
        """Torque each joint may use right now: gravity load plus PD headroom."""
        want = np.abs(tau_ff) + self.column("headroom")
        return np.clip(want, self.column("floor"), self.column("max_torque"))


class GravityModel:
    """Joint torques that hold the arms still against gravity (robot upright)."""

    def __init__(self, urdf_path: Path = URDF_PATH):
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.q = pin.neutral(self.model)
        ids = [self.model.getJointId(j.urdf) for j in ARM_JOINTS]
        self.q_idx = np.array([self.model.joints[k].idx_q for k in ids])
        self.v_idx = np.array([self.model.joints[k].idx_v for k in ids])
        self.lower = self.model.lowerPositionLimit[self.q_idx].copy()
        self.upper = self.model.upperPositionLimit[self.q_idx].copy()

    def torque(self, q_arm: np.ndarray) -> np.ndarray:
        q_arm = np.asarray(q_arm, dtype=float)
        if not np.all(np.isfinite(q_arm)):
            return np.zeros(N_JOINTS)
        self.q[self.q_idx] = q_arm
        tau = pin.computeGeneralizedGravity(self.model, self.data, self.q)
        return tau[self.v_idx].copy()

    def peak_along(self, q_arm: np.ndarray, i: int, target: float, samples: int = 12) -> float:
        """Largest |g_i| while joint i sweeps from its current value to target."""
        q = np.array(q_arm, dtype=float)
        peak = 0.0
        for value in np.linspace(q[i], target, samples):
            q[i] = value
            peak = max(peak, abs(self.torque(q)[i]))
        return peak


def ceil_to(value: float, step: float) -> float:
    return math.ceil(value / step - 1e-9) * step
