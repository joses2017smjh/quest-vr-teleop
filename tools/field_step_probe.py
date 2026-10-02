"""Open-loop field-step probe for one arm actuator: does the rotor follow the
stator field, and does the encoder agree? Every reading is checked.

Why this exists
  The firmware calibration (motor_controller.c:432) turns the field one motor
  revolution in about 4 s, averages whatever the encoder said, and saves the
  result to flash without checking it. The old "travel" figure came from
  host-side samples about 5 times a second and kept only the widest swing.
  Neither can say *where* the rotor stopped following, or why.

What it does
  Puts the joint in VALPHABETA_OVERRIDE and turns the field itself, in small
  steps: half a motor revolution forward, a full one back, half forward
  (+-12 deg at a 15:1 joint), so every rotor angle is visited once in each
  direction. After each step it waits for the rotor to stop, then records the
  firmware's own 10 kHz-unwrapped ENCODER_POSITION (cross-checked against
  ENCODER_POSITION_RAW) and the three phase currents.

What it reports
  * following: rotor electrical angle against field angle, and the rotor
    angles where it stalled or slipped a pole
  * phase order: slope +1 means the order matches the encoder, -1 reversed
  * the flux offset measured here, against the one stored in the controller
  * friction lag (forward vs backward) and encoder linearity
  * phase health: each phase's current amplitude and offset, and V/I by field
    direction (a bad connection raises V/I along that phase's axis)

Reading discipline
  Every SDO read flushes stale frames and retries on timeout. Each step reads
  DEVICE_ID before and after as a sentinel, and checks the position samples
  against each other and against ENCODER_POSITION_RAW. A step that fails any
  check is kept in the log but left out of the analysis, never patched.
  Writes are read back. Mode and error are checked every step and any change
  aborts. The watchdog is fed with HEARTBEAT every step.

Safety
  Current vector capped (default 1.5 A, the size the firmware calibration
  reaches at max_calibration_current 1.0 A), abort above 3 A; excursion
  capped; always ends with v=0 and IDLE; restores the original phase order;
  never stores flash. Refuses while the teleop runner is up.

  .venv/bin/python tools/field_step_probe.py --bus can0 --id 9
  .venv/bin/python tools/field_step_probe.py --bus can0 --id 9 --phase-order -1
  .venv/bin/python tools/field_step_probe.py --bus can0 --id 9 --compare can1:12
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import struct
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arm_common import (  # noqa: E402
    ARM_CHANNELS, AS_BUILT_ARM_JOINTS, ArmBus, Function, Mode,
    decode_error, joint_name, jsonable, mode_name,
)

TWO_PI = 2.0 * math.pi
FMT = {"u32": "<L", "i32": "<l", "f32": "<f"}

# Offsets straight from motor_controller_conf.h:183-264.
DEVICE_ID, WATCHDOG, MODE, ERROR = 0x000, 0x008, 0x010, 0x014
GEAR_RATIO = 0x01C
I_A, I_B, I_C = 0x080, 0x084, 0x088
I_ALPHA, I_BETA = 0x098, 0x09C
V_ALPHA, V_BETA = 0x0A0, 0x0A4
VBUS, POLE_PAIRS, PHASE_ORDER, CAL_CURRENT = 0x100, 0x104, 0x10C, 0x110
CPR, POS_RAW, POS, FLUX = 0x120, 0x12C, 0x134, 0x13C

# Every word of the MotorController struct. The flag marks what loadConfig
# restores from flash at boot (motor_controller.c:216-283); the rest is live.
STRUCT = [
    (0x000, "device_id", "u32", True),
    (0x004, "firmware_version", "hex", False),
    (0x008, "watchdog_timeout", "u32", True),
    (0x00C, "fast_frame_frequency", "u32", True),
    (0x010, "mode", "u32", False),
    (0x014, "error", "hex", False),
    (0x018, "position_update_counter", "u32", False),
    (0x01C, "gear_ratio", "f32", True),
    (0x020, "position_kp", "f32", True),
    (0x024, "position_ki", "f32", True),
    (0x028, "velocity_kp", "f32", True),
    (0x02C, "velocity_ki", "f32", True),
    (0x030, "torque_limit", "f32", True),
    (0x034, "velocity_limit", "f32", True),
    (0x038, "position_limit_lower", "f32", True),
    (0x03C, "position_limit_upper", "f32", True),
    (0x040, "position_offset", "f32", True),
    (0x044, "torque_target", "f32", False),
    (0x048, "torque_measured", "f32", False),
    (0x04C, "torque_setpoint", "f32", False),
    (0x050, "velocity_target", "f32", False),
    (0x054, "velocity_measured", "f32", False),
    (0x058, "velocity_setpoint", "f32", False),
    (0x05C, "position_target", "f32", False),
    (0x060, "position_measured", "f32", False),
    (0x064, "position_setpoint", "f32", False),
    (0x068, "position_integrator", "f32", False),
    (0x06C, "velocity_integrator", "f32", False),
    (0x070, "torque_filter_alpha", "f32", True),
    (0x074, "i_limit", "f32", True),
    (0x078, "i_kp", "f32", True),
    (0x07C, "i_ki", "f32", True),
    (0x080, "i_a_measured", "f32", False),
    (0x084, "i_b_measured", "f32", False),
    (0x088, "i_c_measured", "f32", False),
    (0x08C, "v_a_setpoint", "f32", False),
    (0x090, "v_b_setpoint", "f32", False),
    (0x094, "v_c_setpoint", "f32", False),
    (0x098, "i_alpha_measured", "f32", False),
    (0x09C, "i_beta_measured", "f32", False),
    (0x0A0, "v_alpha_setpoint", "f32", False),
    (0x0A4, "v_beta_setpoint", "f32", False),
    (0x0A8, "v_q_target", "f32", False),
    (0x0AC, "v_d_target", "f32", False),
    (0x0B0, "v_q_setpoint", "f32", False),
    (0x0B4, "v_d_setpoint", "f32", False),
    (0x0B8, "i_q_target", "f32", False),
    (0x0BC, "i_d_target", "f32", False),
    (0x0C0, "i_q_measured", "f32", False),
    (0x0C4, "i_d_measured", "f32", False),
    (0x0C8, "i_q_setpoint", "f32", False),
    (0x0CC, "i_d_setpoint", "f32", False),
    (0x0D0, "i_q_integrator", "f32", False),
    (0x0D4, "i_d_integrator", "f32", False),
    (0x0E4, "adc_raw_0_1", "u16x2", False),
    (0x0E8, "adc_raw_2", "u16x2", False),
    (0x0EC, "adc_offset_0_1", "u16x2", False),
    (0x0F0, "adc_offset_2", "u16x2", False),
    (0x0F4, "undervoltage_threshold", "f32", True),
    (0x0F8, "overvoltage_threshold", "f32", True),
    (0x0FC, "bus_voltage_filter_alpha", "f32", True),
    (0x100, "bus_voltage_measured", "f32", False),
    (0x104, "pole_pairs", "i32", True),
    (0x108, "torque_constant", "f32", True),
    (0x10C, "phase_order", "i32", True),
    (0x110, "max_calibration_current", "f32", True),
    (0x118, "i2c_buffer", "hex", False),
    (0x120, "encoder_cpr", "i32", True),
    (0x124, "encoder_position_offset", "f32", True),
    (0x128, "encoder_velocity_filter_alpha", "f32", True),
    (0x12C, "encoder_position_raw", "u16x2", False),
    (0x130, "encoder_n_rotations", "i32", False),
    (0x134, "encoder_position", "f32", False),
    (0x138, "encoder_velocity", "f32", False),
    (0x13C, "flux_offset", "f32", True),
]

# Config that is legitimately different from joint to joint.
PER_JOINT = {"device_id", "flux_offset", "position_offset", "gear_ratio",
             "position_limit_lower", "position_limit_upper", "encoder_position_offset"}


class Abort(RuntimeError):
    pass


class ProbeBus(ArmBus):
    """ArmBus plus the writes this probe needs. Reads retry; writes read back."""

    def __post_init__(self):
        super().__post_init__()
        self.reads = 0
        self.retries = 0
        self.read_seconds = 0.0

    def get(self, dev_id: int, param: int, kind: str = "f32", tries: int = 3):
        for _ in range(tries):
            t0 = time.perf_counter()
            d = self.read_raw(dev_id, param)  # flushes stale frames first
            self.read_seconds += time.perf_counter() - t0
            self.reads += 1
            if d is not None:
                if kind == "raw":
                    return bytes(d)
                if kind == "u16":
                    return struct.unpack("<H", d[0:2])[0]
                return struct.unpack(FMT[kind], d)[0]
            self.retries += 1
            time.sleep(0.005)  # a late reply lands now; the next flush drops it
        return None

    def put(self, dev_id: int, param: int, kind: str, value) -> bool:
        data = struct.pack(FMT[kind], value)
        self._send(dev_id, Function.RECEIVE_SDO,
                   struct.pack("<BHB", 0x01 << 5, param, 0) + data)
        return self.get(dev_id, param, "raw") == data

    def heartbeat(self, dev_id: int) -> None:
        self._send(dev_id, Function.HEARTBEAT, b"")

    def set_mode(self, dev_id: int, mode: int, wait: float = 0.3) -> bool:
        self._send(dev_id, Function.NMT, struct.pack("<BB", mode, dev_id))
        t_end = time.time() + wait
        while time.time() < t_end:
            if self.get(dev_id, MODE, "u32") == mode:
                return True
            time.sleep(0.01)
        return False


def read_struct(bus: ProbeBus, dev_id: int) -> dict:
    out = {}
    for off, name, kind, _cfg in STRUCT:
        if kind == "u16x2":
            d = bus.get(dev_id, off, "raw")
            out[name] = None if d is None else list(struct.unpack("<HH", d))
        elif kind == "hex":
            v = bus.get(dev_id, off, "u32")
            out[name] = None if v is None else f"0x{v:08X}"
        else:
            out[name] = bus.get(dev_id, off, kind)
    return out


def teleop_running() -> bool:
    return subprocess.run(["pgrep", "-f", r"run_tele[o]p\.py"],
                          capture_output=True).returncode == 0


def clarke(ia: float, ib: float, ic: float) -> tuple[float, float]:
    """Amplitude-invariant, the same as FOC_clarkTransform (foc_math.c:10)."""
    return (2.0 * ia - ib - ic) / 3.0, (ib - ic) / math.sqrt(3.0)


def idle_current_offsets(bus: ProbeBus, dev_id: int, n: int = 12) -> dict:
    """With PWM off no current can flow, so whatever reads here is sensor offset."""
    got = {"ia": [], "ib": [], "ic": []}
    for _ in range(n):
        for key, param in (("ia", I_A), ("ib", I_B), ("ic", I_C)):
            v = bus.get(dev_id, param)
            if v is not None:
                got[key].append(v)
        time.sleep(0.004)
    return {k: (float(np.mean(v)) if v else None) for k, v in got.items()} | {
        f"{k}_spread": (float(np.ptp(v)) if v else None) for k, v in got.items()}


def set_field(bus: ProbeBus, dev_id: int, theta: float, volts: float) -> bool:
    ok_a = bus.put(dev_id, V_ALPHA, "f32", volts * math.cos(theta))
    ok_b = bus.put(dev_id, V_BETA, "f32", volts * math.sin(theta))
    return ok_a and ok_b


def settle(bus: ProbeBus, dev_id: int, tol: float, t_min: float, t_max: float):
    """Wait until the rotor has stopped: two consecutive calm position reads."""
    time.sleep(t_min)
    t0 = time.time()
    prev = bus.get(dev_id, POS)
    calm = 0
    while True:
        cur = bus.get(dev_id, POS)
        if cur is not None and prev is not None and abs(cur - prev) <= tol:
            calm += 1
            if calm >= 2:
                return True, t_min + time.time() - t0
        else:
            calm = 0
        prev = cur
        if time.time() - t0 > t_max:
            return False, t_min + time.time() - t0
        time.sleep(0.006)


def measure(bus: ProbeBus, dev_id: int, cpr: int, n: int = 3) -> dict:
    s1 = bus.get(dev_id, DEVICE_ID, "u32")
    pos = [bus.get(dev_id, POS) for _ in range(n)]
    raw = [bus.get(dev_id, POS_RAW, "u16") for _ in range(2)]
    ia, ib, ic = [], [], []
    for _ in range(n):
        ia.append(bus.get(dev_id, I_A))
        ib.append(bus.get(dev_id, I_B))
        ic.append(bus.get(dev_id, I_C))
    fw_alpha = bus.get(dev_id, I_ALPHA)
    fw_beta = bus.get(dev_id, I_BETA)
    mode = bus.get(dev_id, MODE, "u32")
    err = bus.get(dev_id, ERROR, "u32")
    s2 = bus.get(dev_id, DEVICE_ID, "u32")

    row = {"mode": mode, "error": err, "pos_samples": pos, "raw_samples": raw,
           "ia_samples": ia, "ib_samples": ib, "ic_samples": ic,
           "fw_i_alpha": fw_alpha, "fw_i_beta": fw_beta, "problems": []}
    probs = row["problems"]
    if s1 != dev_id or s2 != dev_id:
        probs.append(f"sentinel read {s1}/{s2}")
    if any(v is None for v in pos + raw + ia + ib + ic) or mode is None or err is None:
        probs.append("missing reply")
        row["valid"] = False
        return row

    counts_per_rad = abs(cpr) / TWO_PI
    spread = (max(pos) - min(pos)) * counts_per_rad
    if spread > 4:
        probs.append(f"position samples spread {spread:.1f} counts")
    p = float(np.median(pos))
    k = p * cpr / TWO_PI
    for r in raw:
        d = (k - r) % abs(cpr)
        d = min(d, abs(cpr) - d)
        if d > 4:
            probs.append(f"position and raw disagree by {d:.0f} counts")
            break
    if any(r >= abs(cpr) for r in raw):
        probs.append("raw out of range")

    row["pos"] = p
    row["raw"] = int(raw[0])
    row["ia"], row["ib"], row["ic"] = (float(np.mean(ia)), float(np.mean(ib)),
                                       float(np.mean(ic)))
    row["i_sum"] = row["ia"] + row["ib"] + row["ic"]
    al, be = clarke(row["ia"], row["ib"], row["ic"])
    row["i_alpha"], row["i_beta"] = al, be
    row["i_mag"] = math.hypot(al, be)
    row["valid"] = not probs
    return row


def build_path(pp: int, half_rev: float, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Field angles (electrical, cumulative): +H, back to -H, forward to 0."""
    h = half_rev * TWO_PI * pp
    n = int(round(h / step))
    seg1 = np.arange(1, n + 1) * step
    seg2 = (n - np.arange(1, 2 * n + 1)) * step
    seg3 = (-n + np.arange(1, n + 1)) * step
    path = np.concatenate([seg1, seg2, seg3])
    direction = np.concatenate([np.ones(n), -np.ones(2 * n), np.ones(n)])
    return path, direction


def build_play_path(pp: int, step: float, engage: float = 5.0, amp: float = 2.0,
                    cycles: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Forward `engage` electrical turns, then back/forward by `amp` turns `cycles` times,
    one more turn forward, then back to 0. With lost motion wider than `amp` between the
    rotor and the encoder, the encoder does not move at all during the small reversals;
    a rotor that follows the field moves back and forth with them."""
    pts = [0.0, engage]
    for _ in range(cycles):
        pts += [engage - amp, engage]
    pts += [engage + 1.0, 0.0]
    path, direction = [], []
    for a, b in zip(pts, pts[1:]):
        n = int(round(abs(b - a) * TWO_PI / step))
        d = 1 if b > a else -1
        for k in range(1, n + 1):
            path.append((a + d * k * step / TWO_PI) * TWO_PI)
            direction.append(d)
    return np.array(path), np.array(direction, dtype=float)


def ramp(bus: ProbeBus, dev_id: int, target: float, abort_i: float, v_max: float,
         offsets: dict) -> tuple[float, float, list]:
    """Raise the voltage at field angle 0 until the current vector reaches target."""
    v, i_mag, curve = 0.0, 0.0, []
    while True:
        v = round(v + 0.05, 3)
        if v > v_max:
            raise Abort(f"{v_max:.2f} V produced only {i_mag:.2f} A. The circuit through "
                        "the motor is open or very high resistance: check the phase wires")
        if not set_field(bus, dev_id, 0.0, v):
            raise Abort("the field voltage did not read back")
        bus.heartbeat(dev_id)
        time.sleep(0.03)
        vals = [bus.get(dev_id, p) for p in (I_A, I_B, I_C)]
        if any(x is None for x in vals):
            continue
        ia = vals[0] - (offsets.get("ia") or 0.0)
        ib = vals[1] - (offsets.get("ib") or 0.0)
        ic = vals[2] - (offsets.get("ic") or 0.0)
        i_mag = math.hypot(*clarke(ia, ib, ic))
        curve.append({"v": v, "i": i_mag, "ia": ia, "ib": ib, "ic": ic})
        if i_mag > abort_i:
            raise Abort(f"current {i_mag:.2f} A at {v:.2f} V during the ramp")
        if i_mag >= target:
            return v, i_mag, curve


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def wrap(a):
    return (np.asarray(a) + np.pi) % TWO_PI - np.pi


def circmean(a) -> float:
    a = np.asarray(a)
    return float(math.atan2(np.mean(np.sin(a)), np.mean(np.cos(a))))


def fit_sinusoid(x: np.ndarray, y: np.ndarray, harmonics=(1,)) -> dict:
    cols = [np.ones_like(x)]
    for h in harmonics:
        cols += [np.cos(h * x), np.sin(h * x)]
    coef, *_ = np.linalg.lstsq(np.column_stack(cols), y, rcond=None)
    out = {"offset": float(coef[0])}
    for i, h in enumerate(harmonics):
        c, s = coef[1 + 2 * i], coef[2 + 2 * i]
        out[h] = {"amp": float(math.hypot(c, s)), "phase_deg": float(math.degrees(math.atan2(s, c)))}
    return out


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def analyze(rows: list, meta: dict) -> dict:
    pp, cpr, gear = meta["pole_pairs"], meta["cpr"], meta["gear_ratio"]
    volts, order = meta["volts"], meta["phase_order_tested"]
    good = [r for r in rows if r.get("valid")]
    res = {"steps": len(rows), "valid_steps": len(good)}
    if len(good) < 20:
        res["verdict"] = "too few valid steps to analyse"
        return res

    th = np.array([r["theta"] for r in good])
    pos = np.array([r["pos"] for r in good])
    dirs = np.array([r["dir"] for r in good])
    rot_e = pos * pp
    pos0 = meta["pos_start"]

    # ---- following -------------------------------------------------------
    dth, drot = np.diff(th), np.diff(rot_e)
    ratio = drot / dth
    slope = float(np.median(ratio))
    moving = np.abs(ratio) > 0.5
    slope_moving = float(np.median(ratio[moving])) if moving.any() else 0.0
    sign = 1.0 if slope_moving >= 0 else -1.0
    follows = np.abs(ratio - sign) < 0.5
    stalled = np.abs(ratio) < 0.3
    slips = np.abs(drot - sign * dth) > math.pi
    seg_ratio = []
    for d in (1, -1):
        m = dirs[1:] == d
        if m.any():
            seg_ratio.append({"direction": "forward" if d > 0 else "backward",
                              "rotor_over_field": float(np.sum(drot[m]) / np.sum(dth[m]))})
    stall_runs = []
    step_dir = dirs[1:]
    for want in (1, -1):
        for a, b in runs(stalled & (step_dir == want)):
            if b - a + 1 < 2:
                continue
            mech = pos[a + 1:b + 2]
            stall_runs.append({
                "steps": int(b - a + 1),
                "field_deg_e": float(np.degrees(np.sum(np.abs(dth[a:b + 1])))),
                "rotor_deg_from_start": float(np.degrees(np.median(mech) - pos0)),
                "joint_deg_from_start": float(np.degrees((np.median(mech) - pos0) / abs(gear))),
                "rotor_raw": int(good[a + 1]["raw"]),
                "direction": "forward" if want > 0 else "backward",
            })
    stall_runs.sort(key=lambda st: -st["steps"])
    # Where the rotor tracked the field, rotor - sign*field (mod 360) should come out the
    # same in every stretch if the magnet is fixed to the rotor: sticking and breaking
    # free only loses whole electrical turns. Forward and backward differ by 2x lag.
    track = np.abs(ratio - sign) < 0.35
    stretches = []
    for a, b in runs(track):
        if b - a + 1 < 5:
            continue
        rel = wrap(rot_e[a + 1:b + 2] - sign * th[a + 1:b + 2])
        stretches.append({"direction": "forward" if dirs[a + 1] > 0 else "backward",
                          "steps": int(b - a + 1),
                          "relation_deg_e": float(np.degrees(circmean(rel))),
                          "rotor_deg_from_start": float(np.degrees(pos[a + 1] - pos0))})
    groups = {}
    for d in ("forward", "backward"):
        vals = [st["relation_deg_e"] for st in stretches if st["direction"] == d]
        if vals:
            c = circmean(np.radians(vals))
            spread = float(np.degrees(np.ptp(wrap(np.radians(vals) - c)))) if len(vals) > 1 else 0.0
            groups[d] = {"n": len(vals), "mean_deg_e": float(np.degrees(c)), "spread_deg_e": spread}
    rel_flux = None
    if "forward" in groups and "backward" in groups and sign > 0:
        f_, b_ = np.radians(groups["forward"]["mean_deg_e"]), np.radians(groups["backward"]["mean_deg_e"])
        rel_flux = float(wrap(f_ + wrap(b_ - f_) / 2.0))
    res["tracking_stretches"] = {"stretches": stretches, "groups": groups,
                                 "flux_from_stretches_rad": rel_flux}
    segs, a = [], 0
    for i in range(1, len(dirs) + 1):
        if i == len(dirs) or dirs[i] != dirs[a]:
            if i - a >= 2:
                lo = max(a - 1, 0)
                f_tr = float((th[i - 1] - th[lo]) / TWO_PI)
                e_tr = float((rot_e[i - 1] - rot_e[lo]) / TWO_PI)
                segs.append({"direction": "forward" if dirs[a] > 0 else "backward",
                             "field_turns_e": f_tr, "encoder_turns_e": e_tr,
                             "ratio": e_tr / f_tr if f_tr else None})
            a = i
    res["segments"] = segs

    # A rotor held by the field sits within a quarter electrical turn of it, so at the same
    # field angle and direction a fixed magnet reads the same every pass (to within lag
    # changes). Drift beyond that means the encoder is not locked to the rotor.
    visits: dict = {}
    for t, d, e in zip(np.round(th / TWO_PI, 6), dirs, (rot_e - rot_e[0]) / TWO_PI):
        visits.setdefault((float(t), int(d)), []).append(float(e))
    repeat = [{"field_turns_e": k[0], "direction": "forward" if k[1] > 0 else "backward",
               "passes": len(v), "encoder_turns_e": v, "spread_turns_e": max(v) - min(v)}
              for k, v in visits.items() if len(v) >= 2]
    if repeat:
        worst = max(repeat, key=lambda r: r["spread_turns_e"])
        res["repeat_positions"] = {"points": len(repeat),
                                   "median_spread_turns_e": float(np.median([r["spread_turns_e"] for r in repeat])),
                                   "worst": worst}
    res["following"] = {
        "slope_rotor_per_field": slope,
        "slope_while_moving": slope_moving,
        "fraction_following": float(np.mean(follows)),
        "fraction_stalled": float(np.mean(stalled)),
        "pole_slips": int(np.sum(slips)),
        "by_direction": seg_ratio,
        "stalls": stall_runs,
        "rotor_span_rev": float((pos.max() - pos.min()) / TWO_PI),
        "field_span_rev": float((th.max() - th.min()) / TWO_PI / pp),
    }
    healthy_follow = abs(abs(slope) - 1.0) < 0.15 and np.mean(follows) > 0.9

    # ---- flux, lag, linearity (only meaningful when it follows +1) ------
    if healthy_follow and sign > 0:
        # skip the first electrical turn: a rotor that started near the unstable point of
        # the alignment field is still settling there
        skip = min(int(round(TWO_PI / max(float(np.median(np.abs(dth))), 1e-6))), len(th) // 4)
        phi = wrap(rot_e - th)[skip:]
        dk, pk = dirs[skip:], pos[skip:]
        f_fwd, f_bwd = circmean(phi[dk > 0]), circmean(phi[dk < 0])
        lag = float(wrap(f_bwd - f_fwd)) / 2.0
        flux = float(wrap(f_fwd + lag))
        stored = meta["flux_stored"]
        resid = wrap(phi - (flux - dk * lag))
        mech = np.mod(pk, TWO_PI)
        harm = fit_sinusoid(mech, resid, harmonics=(1, 2))
        res["flux"] = {
            "measured_rad": flux,
            "stored_rad": stored,
            "stored_mod_2pi_rad": float(wrap(stored)) if stored is not None else None,
            "stored_minus_measured_deg_e": (float(np.degrees(wrap(stored - flux)))
                                            if stored is not None else None),
            "friction_lag_deg_e": float(np.degrees(lag)),
            "friction_fraction_of_available_torque": float(abs(math.sin(lag))),
            "residual_pp_deg_e": float(np.degrees(np.ptp(resid))),
            "residual_rms_deg_e": float(np.degrees(np.sqrt(np.mean(resid ** 2)))),
            "once_per_rev_deg_e": float(np.degrees(harm[1]["amp"])),
            "twice_per_rev_deg_e": float(np.degrees(harm[2]["amp"])),
        }

    # ---- currents ----------------------------------------------------------
    off = meta["idle_offsets"]
    ia = np.array([r["ia"] for r in good]) - (off.get("ia") or 0.0)
    ib = np.array([r["ib"] for r in good]) - (off.get("ib") or 0.0)
    ic = np.array([r["ic"] for r in good]) - (off.get("ic") or 0.0)
    al = (2 * ia - ib - ic) / 3.0
    be = (ib - ic) / math.sqrt(3.0)
    imag = np.hypot(al, be)
    thm = np.mod(th, TWO_PI)
    fits = {}
    expect = {"a": 0.0, "b": 120.0, "c": -120.0}
    for name, arr in (("a", ia), ("b", ib), ("c", ic)):
        f = fit_sinusoid(thm, arr)
        fits[name] = {"amplitude_A": f[1]["amp"], "phase_deg": f[1]["phase_deg"],
                      "phase_error_deg": float(np.degrees(wrap(np.radians(
                          f[1]["phase_deg"] - expect[name])))),
                      "dc_offset_A": f["offset"]}
    amps = [fits[k]["amplitude_A"] for k in "abc"]
    r_eff = volts / np.maximum(imag, 1e-3)
    rf = fit_sinusoid(2 * thm, r_eff)  # 2nd harmonic in field angle
    peak_deg = (rf[1]["phase_deg"] / 2.0) % 180.0
    axes = {"a": 0.0, "b": 120.0, "c": 60.0}  # each phase axis, mod 180
    nearest = min(axes, key=lambda k: abs(((peak_deg - axes[k] + 90) % 180) - 90))
    phys = {1: {"a": 1, "b": 2, "c": 3}, -1: {"a": 3, "b": 2, "c": 1}}[order]
    bins = []
    for lo in range(0, 360, 30):
        m = (np.degrees(thm) >= lo) & (np.degrees(thm) < lo + 30)
        if m.any():
            bins.append({"field_deg_e": lo + 15, "v_over_i_ohm": float(np.mean(r_eff[m])),
                         "i_A": float(np.mean(imag[m]))})
    res["phases"] = {
        "per_phase": fits,
        "amplitude_min_over_max": float(min(amps) / max(amps)) if max(amps) > 0 else None,
        "current_vector_mean_A": float(np.mean(imag)),
        "current_vector_min_A": float(np.min(imag)),
        "current_vector_max_A": float(np.max(imag)),
        "kirchhoff_sum_mean_A": float(np.mean(ia + ib + ic)),
        "kirchhoff_sum_max_abs_A": float(np.max(np.abs(ia + ib + ic))),
        "v_over_i_mean_ohm": float(np.mean(r_eff)),
        "v_over_i_imbalance_pct": float(100.0 * rf[1]["amp"] / max(rf["offset"], 1e-6)),
        "v_over_i_peak_field_deg_e": float(peak_deg),
        "v_over_i_peak_nearest_phase": nearest,
        "v_over_i_peak_esc_output": phys[nearest],
        "current_angle_error_mean_deg": float(np.degrees(np.mean(np.abs(wrap(np.arctan2(be, al) - th))))),
        "by_field_angle": bins,
    }

    # ---- encoder -------------------------------------------------------------
    spreads = []
    for r in good:
        rs = r["raw_samples"]
        d = abs(rs[0] - rs[1]) % abs(cpr)
        spreads.append(min(d, abs(cpr) - d))
    res["encoder"] = {"raw_pair_spread_median": float(np.median(spreads)),
                      "raw_pair_spread_max": int(max(spreads)),
                      "unsettled_steps": int(sum(1 for r in good if not r.get("settled")))}
    return res


def verdict_lines(res: dict, meta: dict) -> list[str]:
    out = []
    if "following" not in res:
        return [res.get("verdict", "no analysis")]
    f = res["following"]
    s = f["slope_rotor_per_field"]
    out.append(f"Rotor vs field slope {s:+.2f} (ideal +1.00), following on "
               f"{100 * f['fraction_following']:.0f}% of steps, stalled on "
               f"{100 * f['fraction_stalled']:.0f}%, pole slips {f['pole_slips']}.")
    for d in f["by_direction"]:
        out.append(f"  {d['direction']:>8}: rotor moved {d['rotor_over_field']:+.2f}x the field")
    out.append(f"  rotor span {f['rotor_span_rev']:.2f} rev for a field span of "
               f"{f['field_span_rev']:.2f} rev")
    rp = res.get("repeat_positions")
    if rp:
        w = rp["worst"]
        ok = w["spread_turns_e"] < 0.25
        out.append(f"Repeat passes at the same field angle ({rp['points']} angles): encoder spread "
                   f"median {rp['median_spread_turns_e']:.2f}, worst {w['spread_turns_e']:.2f} "
                   f"electrical turns ({w['direction']} at {w['field_turns_e']:+.2f}). "
                   + ("Fixed to the rotor." if ok else
                      "More than the 0.25 a rotor held by a field allows when friction is small: "
                      "either the encoder slips on the rotor, or friction is close to the "
                      "field's torque (compare the forward and backward tracking stretches)."))
    for sg in res.get("segments", []):
        out.append(f"  segment {sg['direction']:>8}: field {sg['field_turns_e']:+6.2f} e-turns, "
                   f"encoder {sg['encoder_turns_e']:+6.2f}  ({sg['ratio']:+.2f}x)")
    for st in f["stalls"][:8]:
        out.append(f"  STALL {st['direction']}: {st['steps']} steps ({st['field_deg_e']:.0f} deg e) "
                   f"at rotor {st['rotor_deg_from_start']:+.1f} deg from start "
                   f"(joint {st['joint_deg_from_start']:+.2f} deg), raw {st['rotor_raw']}")
    sm = f["slope_while_moving"]
    po = meta["phase_order_tested"]
    if abs(abs(sm) - 1) < 0.25:
        if sm > 0:
            out.append(f"PHASE ORDER {po:+d} matches the encoder: when it moves, it moves "
                       f"{sm:+.2f}x the field.")
        else:
            out.append(f"PHASE ORDER {po:+d} is REVERSED relative to the encoder: when it moves, "
                       f"the encoder counts {sm:+.2f}x the field. Closed loop cannot work "
                       f"with this order.")
    elif f["fraction_following"] < 0.1:
        out.append("The encoder does NOT follow the field: it stays put while the field turns.")
    else:
        out.append(f"When it moves, it moves {sm:+.2f}x the field (expected +-1).")
    if f["fraction_stalled"] > 0.1:
        out.append(f"STICKS: the encoder stood still on {100 * f['fraction_stalled']:.0f}% of "
                   f"steps while the field kept turning (the known-good wrist: 0%).")
    tg = res.get("tracking_stretches", {}).get("groups", {})
    for d in ("forward", "backward"):
        if d in tg:
            g = tg[d]
            out.append(f"  {d} tracking stretches: {g['n']}, rotor-minus-field "
                       f"{g['mean_deg_e']:+.0f} deg e, spread {g['spread_deg_e']:.0f} deg "
                       f"(fixed magnet: all alike)")
    rf = res.get("tracking_stretches", {}).get("flux_from_stretches_rad")
    if rf is not None and "flux" not in res and meta.get("flux_stored") is not None:
        e = float(np.degrees(wrap(meta["flux_stored"] - rf)))
        out.append(f"  flux from the tracking stretches {rf:+.2f} rad; stored "
                   f"{meta['flux_stored']:+.2f} rad is {e:+.0f} deg e from it"
                   + (": the stored value pushes the WRONG WAY" if abs(e) > 120 else ""))
    if "flux" in res:
        fx = res["flux"]
        out.append(f"Flux offset measured here {fx['measured_rad']:+.3f} rad (mod 2pi).")
        if fx["stored_minus_measured_deg_e"] is not None:
            e = fx["stored_minus_measured_deg_e"]
            q = "GOOD" if abs(e) < 20 else ("MARGINAL" if abs(e) < 45 else "WRONG")
            out.append(f"  stored {fx['stored_rad']:+.3f} rad is {e:+.0f} deg e from it: {q} "
                       f"(torque falls as cos of this; 90 deg means none)")
        out.append(f"  friction lag {fx['friction_lag_deg_e']:+.1f} deg e "
                   f"({100 * fx['friction_fraction_of_available_torque']:.0f}% of the torque "
                   f"this current makes)")
        out.append(f"  encoder non-linearity {fx['residual_pp_deg_e']:.0f} deg e peak-to-peak "
                   f"(once per rev {fx['once_per_rev_deg_e']:.0f}, twice {fx['twice_per_rev_deg_e']:.0f})")
    p = res.get("phases")
    if p:
        pp_ = p["per_phase"]
        out.append("Phase currents (amplitude, phase error, DC offset):")
        for k in "abc":
            out.append(f"  phase {k}: {pp_[k]['amplitude_A']:.3f} A, "
                       f"{pp_[k]['phase_error_deg']:+.1f} deg, {pp_[k]['dc_offset_A']:+.3f} A")
        out.append(f"  balance min/max {p['amplitude_min_over_max']:.2f} (1.00 ideal); "
                   f"Kirchhoff sum mean {p['kirchhoff_sum_mean_A']:+.3f} A")
        out.append(f"  V/I {p['v_over_i_mean_ohm']:.3f} ohm, varies {p['v_over_i_imbalance_pct']:.1f}% "
                   f"with field angle, highest along phase {p['v_over_i_peak_nearest_phase']} "
                   f"(ESC output {p['v_over_i_peak_esc_output']})")
    e = res.get("encoder")
    if e:
        out.append(f"Encoder noise at rest: median {e['raw_pair_spread_median']:.0f}, max "
                   f"{e['raw_pair_spread_max']} counts; unsettled steps {e['unsettled_steps']}")
    return out


def plot(rows: list, res: dict, meta: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    good = [r for r in rows if r.get("valid")]
    if len(good) < 5:
        return
    pp = meta["pole_pairs"]
    th = np.array([r["theta"] for r in good])
    pos = np.array([r["pos"] for r in good])
    dirs = np.array([r["dir"] for r in good])
    rot_e = pos * pp
    fwd, bwd = dirs > 0, dirs < 0

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    a = ax[0, 0]
    a.plot(th[fwd] / TWO_PI, (rot_e[fwd] - rot_e[0]) / TWO_PI, ".", ms=3, label="field forward")
    a.plot(th[bwd] / TWO_PI, (rot_e[bwd] - rot_e[0]) / TWO_PI, ".", ms=3, label="field backward")
    x = np.array([th.min(), th.max()]) / TWO_PI
    a.plot(x, x - th[0] / TWO_PI, "k--", lw=0.8, label="slope +1")
    a.plot(x, -(x - th[0] / TWO_PI), "k:", lw=0.8, label="slope -1")
    a.set_xlabel("field angle (electrical revolutions)")
    a.set_ylabel("rotor angle from encoder (electrical rev)")
    a.set_title("Does the rotor follow the field?")
    a.legend(fontsize=8)

    a = ax[0, 1]
    if "flux" in res:
        phi = np.degrees(wrap(rot_e - th - res["flux"]["measured_rad"]))
        mech = np.degrees(pos - meta["pos_start"])
        a.plot(mech[fwd], phi[fwd], ".", ms=3, label="forward")
        a.plot(mech[bwd], phi[bwd], ".", ms=3, label="backward")
        st = res["flux"]["stored_minus_measured_deg_e"]
        if st is not None:
            a.axhline(st, color="r", lw=1, label=f"stored flux ({st:+.0f})")
        a.set_xlabel("rotor angle from start (mechanical deg)")
        a.set_ylabel("rotor minus field, from measured flux (deg e)")
        a.set_title("Commutation angle error around one revolution")
    else:
        a.plot(np.arange(len(th)), np.degrees(rot_e - th) , ".", ms=3)
        a.set_xlabel("step")
        a.set_ylabel("rotor minus field (deg e, unwrapped)")
        a.set_title("Following error (not following cleanly)")
    a.legend(fontsize=8)

    off = meta["idle_offsets"]
    ia = np.array([r["ia"] for r in good]) - (off.get("ia") or 0.0)
    ib = np.array([r["ib"] for r in good]) - (off.get("ib") or 0.0)
    ic = np.array([r["ic"] for r in good]) - (off.get("ic") or 0.0)
    al = (2 * ia - ib - ic) / 3.0
    be = (ib - ic) / math.sqrt(3.0)
    thd = np.degrees(np.mod(th, TWO_PI))
    a = ax[1, 0]
    a.plot(thd, meta["volts"] / np.maximum(np.hypot(al, be), 1e-3), ".", ms=3)
    for deg, lab in ((0, "a"), (180, "a"), (120, "b"), (300, "b"), (60, "c"), (240, "c")):
        a.axvline(deg, color="0.8", lw=0.8)
        a.text(deg, a.get_ylim()[1], lab, ha="center", va="bottom", fontsize=8)
    a.set_xlabel("field angle (deg e)")
    a.set_ylabel("V / I (ohm)")
    a.set_title("Resistance seen by field direction (bad phase = peaks on its axis)")

    a = ax[1, 1]
    for arr, lab in ((ia, "i_a"), (ib, "i_b"), (ic, "i_c")):
        a.plot(thd, arr, ".", ms=3, label=lab)
    a.set_xlabel("field angle (deg e)")
    a.set_ylabel("phase current (A, idle offset removed)")
    a.set_title("Phase currents")
    a.legend(fontsize=8)

    fig.suptitle(f"{meta['bus']} ID {meta['id']} {meta['joint']}  phase order "
                 f"{meta['phase_order_tested']:+d}  {meta['volts']:.2f} V  "
                 f"{meta['timestamp']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------
def compare(args) -> int:
    ch, dev = args.bus, args.id
    ref_ch, ref_id = args.compare.split(":")
    ref_id = int(ref_id)
    if ref_ch not in ARM_CHANNELS:
        raise SystemExit(f"REFUSED: {ref_ch} is not an arm bus")
    with ProbeBus(ch) as bus:
        a = read_struct(bus, dev)
        a_idle = idle_current_offsets(bus, dev)
    if ref_ch == ch:
        with ProbeBus(ch) as bus:
            b = read_struct(bus, ref_id)
            b_idle = idle_current_offsets(bus, ref_id)
    else:
        with ProbeBus(ref_ch) as bus:
            b = read_struct(bus, ref_id)
            b_idle = idle_current_offsets(bus, ref_id)

    def same(x, y):
        if isinstance(x, float) and isinstance(y, float):
            return math.isclose(x, y, rel_tol=1e-4, abs_tol=1e-6)
        return x == y

    left = f"{ch}/{dev}"
    right = f"{ref_ch}/{ref_id}"
    print(f"{'field':<32}{left:>16}{right:>16}   note")
    print("-- loaded from flash at boot --")
    for off, name, kind, cfg in STRUCT:
        if not cfg:
            continue
        x, y = a.get(name), b.get(name)
        note = "" if same(x, y) else ("differs (expected per joint)" if name in PER_JOINT
                                      else "DIFFERS")
        print(f"{name:<32}{fmt(x):>16}{fmt(y):>16}   {note}")
    print("-- measured at boot / live --")
    for name in ("firmware_version", "mode", "error", "adc_offset_0_1", "adc_offset_2",
                 "bus_voltage_measured", "encoder_position_raw", "encoder_n_rotations"):
        print(f"{name:<32}{fmt(a.get(name)):>16}{fmt(b.get(name)):>16}")
    for k in ("ia", "ib", "ic"):
        print(f"{'idle current ' + k + ' (A)':<32}{fmt(a_idle[k]):>16}{fmt(b_idle[k]):>16}"
              f"   should be ~0 with PWM off")
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"compare_{stamp}_{ch}_id{dev}_vs_{ref_ch}_id{ref_id}.json")
    with open(path, "w") as fh:
        json.dump(jsonable({left: a, right: b, "idle_currents": {left: a_idle, right: b_idle}}),
                  fh, indent=2)
    print(f"log: {path}")
    return 0


def fmt(v, nd=4):
    if v is None:
        return "NONE"
    if isinstance(v, float):
        return f"{v:.{nd}g}" if abs(v) < 1e5 else f"{v:.3e}"
    return str(v)


def probe(args) -> int:
    ch, dev = args.bus, args.id
    joint = joint_name(ch, dev)
    if teleop_running():
        raise SystemExit("REFUSED: run_teleop.py is running and would fight this test. Stop it first.")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    order_tag = ("" if args.phase_order is None else f"_po{args.phase_order:+d}") + (
        "" if args.path == "sweep" else f"_{args.path}")
    outdir = os.path.join(args.outdir, f"{stamp}_{ch}_id{dev}{order_tag}")
    os.makedirs(outdir, exist_ok=True)

    rows: list = []
    events: list = []
    meta: dict = {"bus": ch, "id": dev, "joint": joint, "timestamp": stamp}
    abort_reason = None

    def event(msg: str) -> None:
        events.append({"t": time.time(), "msg": msg})
        print(f"  {msg}")

    with ProbeBus(ch) as bus:
        pre = read_struct(bus, dev)
        meta["pre"] = pre
        mode, pp, cpr = pre["mode"], pre["pole_pairs"], pre["encoder_cpr"]
        gear, vbus, order0 = pre["gear_ratio"], pre["bus_voltage_measured"], pre["phase_order"]
        print(f"{ch} ID {dev} {joint}: mode {mode_name(mode)} error {pre['error']} "
              f"Vbus {vbus:.2f} pp {pp} cpr {cpr} gear {gear} phase order {order0} "
              f"flux {pre['flux_offset']} cal current {pre['max_calibration_current']}")
        if pre["device_id"] != dev:
            raise SystemExit(f"REFUSED: device answered as {pre['device_id']}")
        if not isinstance(vbus, float) or vbus < 15:
            raise SystemExit(f"REFUSED: bus voltage {vbus}")
        if pp not in range(1, 33) or abs(cpr or 0) != 4096 or not isinstance(gear, float):
            raise SystemExit(f"REFUSED: odd config pp={pp} cpr={cpr} gear={gear}")
        if order0 not in (1, -1):
            raise SystemExit(f"REFUSED: stored phase order is {order0}, which drives no PWM at all")

        order = order0 if args.phase_order is None else args.phase_order
        meta.update(pole_pairs=pp, cpr=cpr, gear_ratio=gear, phase_order_stored=order0,
                    phase_order_tested=order, flux_stored=pre["flux_offset"],
                    current_target=args.current)

        if mode != Mode.IDLE:
            event(f"mode was {mode_name(mode)}: switching to IDLE first")
            if not bus.set_mode(dev, Mode.IDLE):
                raise SystemExit("REFUSED: could not reach IDLE")
        if pre["error"] not in (None, "0x00000000"):
            event(f"clearing old error {pre['error']} so anything new shows up")
            bus.put(dev, ERROR, "u32", 0)

        meta["idle_offsets"] = idle_current_offsets(bus, dev)
        o = meta["idle_offsets"]
        event(f"idle current offsets a {o['ia']:+.3f} b {o['ib']:+.3f} c {o['ic']:+.3f} A")

        changed_order = False
        try:
            if order != order0:
                if not bus.put(dev, PHASE_ORDER, "i32", order):
                    raise Abort("phase order write did not read back")
                changed_order = True
                event(f"phase order {order0:+d} -> {order:+d} (RAM only)")
                # phase order swaps which ADC channel is called a and c, so the offsets
                # read above belong to the old order
                meta["idle_offsets"] = idle_current_offsets(bus, dev)
                o = meta["idle_offsets"]
                event(f"idle offsets in this order: a {o['ia']:+.3f} b {o['ib']:+.3f} "
                      f"c {o['ic']:+.3f} A")

            # Entering the mode zeroes v_alpha/v_beta (CurrentController_reset). In IDLE the
            # current loop rewrites them every cycle, so they can only be checked afterwards.
            if not bus.set_mode(dev, Mode.VALPHABETA_OVERRIDE):
                raise Abort("controller refused VALPHABETA_OVERRIDE")
            bus.heartbeat(dev)
            if not set_field(bus, dev, 0.0, 0.0):
                raise Abort("field voltage would not hold at zero in VALPHABETA_OVERRIDE")
            event("VALPHABETA_OVERRIDE, ramping current at field angle 0")
            volts, i0, curve = ramp(bus, dev, args.current, args.abort_current, args.v_max,
                                    meta["idle_offsets"])
            meta.update(volts=volts, ramp_curve=curve)
            event(f"{volts:.2f} V gives {i0:.2f} A (V/I {volts / max(i0, 1e-3):.3f} ohm)")
            time.sleep(0.3)
            bus.heartbeat(dev)
            tol = 2.0 * TWO_PI / abs(cpr)
            settle(bus, dev, tol, 0.05, 0.5)
            for _ in range(3):
                first = measure(bus, dev, cpr)
                if first.get("valid"):
                    break
                bus.heartbeat(dev)
                time.sleep(0.05)
            else:
                raise Abort(f"first reading failed checks: {first['problems']}")
            meta["pos_start"] = first["pos"]
            first.update(theta=0.0, dir=1, step=0, settled=True, t_settle=0.0)
            rows.append(first)

            if args.path == "play":
                path, dirs = build_play_path(pp, math.radians(args.step_deg))
            elif args.path == "park":
                n = int(round(abs(args.park_turns) * 360.0 / args.step_deg))
                d = 1.0 if args.park_turns > 0 else -1.0
                path = d * np.arange(1, n + 1) * math.radians(args.step_deg)
                dirs = np.full(n, d)
            else:
                path, dirs = build_path(pp, args.half_rev, math.radians(args.step_deg))
            meta["path"] = args.path
            limit = (float(np.max(np.abs(path))) / pp + 0.35 * TWO_PI)
            n_bad_writes = n_missing = 0
            t_start = time.time()
            for k, (th, d) in enumerate(zip(path, dirs), start=1):
                if not set_field(bus, dev, float(th), volts):
                    n_bad_writes += 1
                    if n_bad_writes >= 3:
                        raise Abort("field writes stopped reading back")
                bus.heartbeat(dev)
                settled, t_set = settle(bus, dev, tol, args.settle, 0.4)
                row = measure(bus, dev, cpr)
                row.update(theta=float(th), dir=int(d), step=k, settled=settled, t_settle=t_set,
                           t=time.time() - t_start)
                rows.append(row)
                n_missing = n_missing + 1 if "missing reply" in row["problems"] else 0
                if n_missing >= 5:
                    raise Abort("five steps in a row without complete replies")
                if row["mode"] is not None and row["mode"] != Mode.VALPHABETA_OVERRIDE:
                    raise Abort(f"mode changed to {mode_name(row['mode'])} "
                                f"(error {decode_error(row['error'])})")
                if row["error"]:
                    raise Abort(f"error appeared: {decode_error(row['error'])}")
                if row.get("i_mag", 0.0) > args.abort_current:
                    raise Abort(f"current {row['i_mag']:.2f} A")
                if "pos" in row and abs(row["pos"] - meta["pos_start"]) > limit:
                    raise Abort("rotor went further than the field: stopping")
                if k % 36 == 0 or k == len(path):
                    rel = (row.get("pos", float("nan")) - meta["pos_start"]) * pp
                    print(f"  step {k:4d}/{len(path)}  field {math.degrees(th):+8.0f} deg e  "
                          f"rotor {math.degrees(rel):+8.0f} deg e  I {row.get('i_mag', float('nan')):.2f} A"
                          f"  valid {row['valid']}")
        except Abort as exc:
            abort_reason = str(exc)
            event(f"ABORT: {abort_reason}")
        except KeyboardInterrupt:
            abort_reason = "interrupted"
            event("ABORT: interrupted")
        finally:
            set_field(bus, dev, 0.0, 0.0)
            if not bus.set_mode(dev, Mode.IDLE):
                bus.set_mode(dev, Mode.IDLE, wait=1.0)
            if changed_order and not args.keep_phase_order:
                ok = bus.put(dev, PHASE_ORDER, "i32", order0)
                event(f"phase order restored to {order0:+d}" + ("" if ok else " (READBACK FAILED)"))
            post_mode = bus.get(dev, MODE, "u32")
            post_err = bus.get(dev, ERROR, "u32")
            meta["idle_offsets_after"] = idle_current_offsets(bus, dev)
            event(f"end: mode {mode_name(post_mode)} error {decode_error(post_err)}")
            meta["transport"] = {"reads": bus.reads, "retries": bus.retries,
                                 "timeouts": bus.timeout_count,
                                 "ms_per_read": 1000.0 * bus.read_seconds / max(bus.reads, 1),
                                 "error_frames": len(bus.error_frames)}

    meta["abort"] = abort_reason
    meta["events"] = events
    meta.setdefault("volts", 0.0)
    meta.setdefault("pos_start", rows[0]["pos"] if rows and "pos" in rows[0] else 0.0)
    invalid = [{"step": r.get("step"), "problems": r["problems"]} for r in rows if not r.get("valid")]
    res = analyze(rows, meta) if meta["volts"] else {"verdict": "no sweep"}
    res["invalid_steps"] = invalid
    lines = verdict_lines(res, meta)
    t = meta["transport"]
    lines.insert(0, f"Transport: {t['reads']} reads, {t['retries']} retries, {t['timeouts']} "
                    f"timeouts, {t['ms_per_read']:.2f} ms/read; {len(invalid)} of {len(rows)} "
                    f"steps failed a check and were left out")
    if abort_reason:
        lines.insert(0, f"ABORTED: {abort_reason}")

    with open(os.path.join(outdir, "rows.json"), "w") as fh:
        json.dump(jsonable(rows), fh)
    with open(os.path.join(outdir, "summary.json"), "w") as fh:
        json.dump(jsonable({"meta": meta, "result": res}), fh, indent=2)
    with open(os.path.join(outdir, "report.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    try:
        plot(rows, res, meta, os.path.join(outdir, "probe.png"))
    except Exception as exc:  # the numbers matter more than the picture
        print(f"  plot failed: {exc}")
    print()
    print("\n".join(lines))
    print(f"\nlog: {outdir}")
    return 1 if abort_reason else 0


def reanalyze(run_dir: str) -> int:
    with open(os.path.join(run_dir, "rows.json")) as fh:
        rows = json.load(fh)
    with open(os.path.join(run_dir, "summary.json")) as fh:
        meta = json.load(fh)["meta"]
    res = analyze(rows, meta)
    res["invalid_steps"] = [{"step": r.get("step"), "problems": r["problems"]}
                            for r in rows if not r.get("valid")]
    lines = verdict_lines(res, meta)
    with open(os.path.join(run_dir, "summary.json"), "w") as fh:
        json.dump(jsonable({"meta": meta, "result": res}), fh, indent=2)
    with open(os.path.join(run_dir, "report.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    plot(rows, res, meta, os.path.join(run_dir, "probe.png"))
    print("\n".join(lines))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bus", required=True, choices=ARM_CHANNELS)
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--phase-order", type=int, choices=(1, -1),
                    help="test with this phase order (RAM only, restored afterwards)")
    ap.add_argument("--keep-phase-order", action="store_true",
                    help="leave the tested phase order in RAM (still not stored to flash)")
    ap.add_argument("--current", type=float, default=1.5, help="current vector to hold (A)")
    ap.add_argument("--abort-current", type=float, default=3.0)
    ap.add_argument("--v-max", type=float, default=4.0)
    ap.add_argument("--step-deg", type=float, default=20.0, help="field step, electrical deg")
    ap.add_argument("--half-rev", type=float, default=0.5,
                    help="how far to go each way, in motor revolutions")
    ap.add_argument("--path", choices=("sweep", "play", "park"), default="sweep",
                    help="sweep: +-half-rev; play: small reversals that expose lost motion; "
                         "park: move --park-turns and stay there")
    ap.add_argument("--park-turns", type=float, default=6.0,
                    help="electrical turns for --path park (+ raises the encoder count)")
    ap.add_argument("--settle", type=float, default=0.02, help="minimum wait per step (s)")
    ap.add_argument("--outdir", default="arm_validation/probe")
    ap.add_argument("--compare", metavar="BUS:ID",
                    help="read-only: diff every setting against another joint, no motion")
    ap.add_argument("--reanalyze", metavar="DIR",
                    help="recompute the report and plot from a saved run; no bus access")
    if "--reanalyze" in sys.argv:
        args = ap.parse_args(["--bus", "can0", "--id", "1"] + sys.argv[1:])
        return reanalyze(args.reanalyze)
    args = ap.parse_args()
    if args.id not in AS_BUILT_ARM_JOINTS[args.bus]:
        raise SystemExit(f"REFUSED: {args.bus} ID {args.id} is not an as-built arm joint")
    if args.compare:
        return compare(args)
    return probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
