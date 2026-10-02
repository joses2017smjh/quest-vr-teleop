"""ARM-ONLY config write for Berkeley Humanoid Lite.

Writes RAM (and optionally flash) on can0/can1 as-built IDs only:
  can0: 1 3 5 7 9
  can1: 2 4 6 8 12

Never opens a leg bus. Never uses Humanoid() / write_configurations.py
(those currently bind can0/can1 to the legs). Never changes device_id.
Never writes encoder flux_offset -- that is set by MODE_CALIBRATION later.

Source of values: source/berkeley_humanoid_lite_lowlevel/robot_configuration.backup.json
can1 ID 12 uses the right_wrist_yaw_joint block (backup still labels that ID 10).

Usage:
  .venv/bin/python tools/write_arm_config.py --dry-run
  .venv/bin/python tools/write_arm_config.py --ram
  .venv/bin/python tools/write_arm_config.py --flash
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
import time

import berkeley_humanoid_lite_lowlevel.recoil as recoil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arm_common import (  # noqa: E402
    ARM_CHANNELS, AS_BUILT_ARM_JOINTS, EXPECTED_USB_PORT,
    assert_arm_channel, jsonable, verify_usb_binding,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKUP = os.path.join(
    REPO_ROOT, "source/berkeley_humanoid_lite_lowlevel/robot_configuration.backup.json")

# Config key in backup.json for each (bus, as-built id).
CONFIG_JOINT = {
    ("can0", 1): "left_shoulder_pitch_joint",
    ("can0", 3): "left_shoulder_roll_joint",
    ("can0", 5): "left_shoulder_yaw_joint",
    ("can0", 7): "left_elbow_joint",
    ("can0", 9): "left_wrist_yaw_joint",
    ("can1", 2): "right_shoulder_pitch_joint",
    ("can1", 4): "right_shoulder_roll_joint",
    ("can1", 6): "right_shoulder_yaw_joint",
    ("can1", 8): "right_elbow_joint",
    ("can1", 12): "right_wrist_yaw_joint",
}

DELAY = 0.05


def load_backup(path: str) -> dict:
    with open(path) as f:
        text = f.read()
    # File uses Python Infinity tokens, which std json rejects.
    text = text.replace("-Infinity", "-1e400").replace("Infinity", "1e400")
    return json.loads(text)


def close(a, b, rel=1e-3, abs_tol=1e-4) -> bool:
    if a is None or b is None:
        return False
    if isinstance(a, float) or isinstance(b, float):
        if math.isinf(float(a)) or math.isinf(float(b)):
            return math.isinf(float(a)) and math.isinf(float(b)) and (a > 0) == (b > 0)
        return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_tol)
    return a == b


def arm_bus(channel: str) -> recoil.Bus:
    assert_arm_channel(channel)
    ok, actual = verify_usb_binding(channel)
    if not ok:
        raise SystemExit(
            f"REFUSED: {channel} is on USB {actual}, expected {EXPECTED_USB_PORT[channel]}"
        )
    return recoil.Bus(channel=channel, bitrate=1000000)


TO = 0.08  # SDO timeout; recoil.receive() can spin forever on error frames if this is None


def snapshot(bus: recoil.Bus, dev_id: int) -> dict:
    P = recoil.Parameter
    return {
        "device_id": bus._read_parameter_u32(dev_id, P.DEVICE_ID, timeout=TO),
        "mode": bus._read_parameter_u32(dev_id, P.MODE, timeout=TO),
        "error": bus._read_parameter_u32(dev_id, P.ERROR, timeout=TO),
        "gear_ratio": bus._read_parameter_f32(dev_id, P.POSITION_CONTROLLER_GEAR_RATIO, timeout=TO),
        "position_kp": bus._read_parameter_f32(dev_id, P.POSITION_CONTROLLER_POSITION_KP, timeout=TO),
        "position_ki": bus._read_parameter_f32(dev_id, P.POSITION_CONTROLLER_POSITION_KI, timeout=TO),
        "velocity_kp": bus._read_parameter_f32(dev_id, P.POSITION_CONTROLLER_VELOCITY_KP, timeout=TO),
        "velocity_ki": bus._read_parameter_f32(dev_id, P.POSITION_CONTROLLER_VELOCITY_KI, timeout=TO),
        "torque_limit": bus._read_parameter_f32(dev_id, P.POSITION_CONTROLLER_TORQUE_LIMIT, timeout=TO),
        "velocity_limit": bus._read_parameter_f32(dev_id, P.POSITION_CONTROLLER_VELOCITY_LIMIT, timeout=TO),
        "position_limit_lower": bus._read_parameter_f32(dev_id, P.POSITION_CONTROLLER_POSITION_LIMIT_LOWER, timeout=TO),
        "position_limit_upper": bus._read_parameter_f32(dev_id, P.POSITION_CONTROLLER_POSITION_LIMIT_UPPER, timeout=TO),
        "position_offset": bus._read_parameter_f32(dev_id, P.POSITION_CONTROLLER_POSITION_OFFSET, timeout=TO),
        "torque_filter_alpha": bus._read_parameter_f32(dev_id, P.POSITION_CONTROLLER_TORQUE_FILTER_ALPHA, timeout=TO),
        "i_limit": bus._read_parameter_f32(dev_id, P.CURRENT_CONTROLLER_I_LIMIT, timeout=TO),
        "i_kp": bus._read_parameter_f32(dev_id, P.CURRENT_CONTROLLER_I_KP, timeout=TO),
        "i_ki": bus._read_parameter_f32(dev_id, P.CURRENT_CONTROLLER_I_KI, timeout=TO),
        "pole_pairs": bus._read_parameter_u32(dev_id, P.MOTOR_POLE_PAIRS, timeout=TO),
        "torque_constant": bus._read_parameter_f32(dev_id, P.MOTOR_TORQUE_CONSTANT, timeout=TO),
        "phase_order": bus._read_parameter_i32(dev_id, P.MOTOR_PHASE_ORDER, timeout=TO),
        "max_calibration_current": bus._read_parameter_f32(dev_id, P.MOTOR_MAX_CALIBRATION_CURRENT, timeout=TO),
        "encoder_cpr": bus._read_parameter_u32(dev_id, P.ENCODER_CPR, timeout=TO),
        "encoder_position_offset": bus._read_parameter_f32(dev_id, P.ENCODER_POSITION_OFFSET, timeout=TO),
        "encoder_velocity_filter_alpha": bus._read_parameter_f32(dev_id, P.ENCODER_VELOCITY_FILTER_ALPHA, timeout=TO),
        "encoder_flux_offset": bus._read_parameter_f32(dev_id, P.ENCODER_FLUX_OFFSET, timeout=TO),
        "fast_frame_frequency": bus._read_parameter_u32(dev_id, P.FAST_FRAME_FREQUENCY, timeout=TO),
        "bus_voltage_filter_alpha": bus._read_parameter_f32(dev_id, P.POWERSTAGE_BUS_VOLTAGE_FILTER_ALPHA, timeout=TO),
    }


def expected_from(cfg: dict) -> dict:
    pc, cc, mot, enc, ps = (
        cfg["position_controller"], cfg["current_controller"],
        cfg["motor"], cfg["encoder"], cfg["powerstage"],
    )
    return {
        "gear_ratio": pc["gear_ratio"],
        "position_kp": pc["position_kp"],
        "position_ki": pc["position_ki"],
        "velocity_kp": pc["velocity_kp"],
        "velocity_ki": pc["velocity_ki"],
        "torque_limit": pc["torque_limit"],
        "velocity_limit": pc["velocity_limit"],
        "position_limit_lower": pc["position_limit_lower"],
        "position_limit_upper": pc["position_limit_upper"],
        "position_offset": pc["position_offset"],
        "torque_filter_alpha": pc["torque_filter_alpha"],
        "i_limit": cc["i_limit"],
        "i_kp": cc["i_kp"],
        "i_ki": cc["i_ki"],
        "pole_pairs": mot["pole_pairs"],
        "torque_constant": mot["torque_constant"],
        "phase_order": mot["phase_order"],
        "max_calibration_current": mot["max_calibration_current"],
        "encoder_cpr": enc["cpr"],
        "encoder_position_offset": enc["position_offset"],
        "encoder_velocity_filter_alpha": enc["velocity_filter_alpha"],
        "fast_frame_frequency": cfg["fast_frame_frequency"],
        "bus_voltage_filter_alpha": ps["bus_voltage_filter_alpha"],
    }


def write_ram(bus: recoil.Bus, dev_id: int, exp: dict) -> None:
    bus.write_fast_frame_frequency(dev_id, int(exp["fast_frame_frequency"]))
    time.sleep(DELAY)
    bus.write_gear_ratio(dev_id, float(exp["gear_ratio"]))
    time.sleep(DELAY)
    bus.write_position_kp(dev_id, float(exp["position_kp"]))
    time.sleep(DELAY)
    bus.write_position_ki(dev_id, float(exp["position_ki"]))
    time.sleep(DELAY)
    bus.write_velocity_kp(dev_id, float(exp["velocity_kp"]))
    time.sleep(DELAY)
    bus.write_velocity_ki(dev_id, float(exp["velocity_ki"]))
    time.sleep(DELAY)
    bus.write_torque_limit(dev_id, float(exp["torque_limit"]))
    time.sleep(DELAY)
    bus.write_velocity_limit(dev_id, float(exp["velocity_limit"]))
    time.sleep(DELAY)
    bus.write_position_limit_lower(dev_id, float(exp["position_limit_lower"]))
    time.sleep(DELAY)
    bus.write_position_limit_upper(dev_id, float(exp["position_limit_upper"]))
    time.sleep(DELAY)
    bus.write_position_offset(dev_id, float(exp["position_offset"]))
    time.sleep(DELAY)
    bus.write_torque_filter_alpha(dev_id, float(exp["torque_filter_alpha"]))
    time.sleep(DELAY)
    bus.write_current_limit(dev_id, float(exp["i_limit"]))
    time.sleep(DELAY)
    bus.write_current_kp(dev_id, float(exp["i_kp"]))
    time.sleep(DELAY)
    bus.write_current_ki(dev_id, float(exp["i_ki"]))
    time.sleep(DELAY)
    bus.write_bus_voltage_filter_alpha(dev_id, float(exp["bus_voltage_filter_alpha"]))
    time.sleep(DELAY)
    bus.write_motor_pole_pairs(dev_id, int(exp["pole_pairs"]))
    time.sleep(DELAY)
    bus.write_motor_torque_constant(dev_id, float(exp["torque_constant"]))
    time.sleep(DELAY)
    bus.write_motor_phase_order(dev_id, int(exp["phase_order"]))
    time.sleep(DELAY)
    bus.write_motor_calibration_current(dev_id, float(exp["max_calibration_current"]))
    time.sleep(DELAY)
    bus.write_encoder_cpr(dev_id, int(exp["encoder_cpr"]))
    time.sleep(DELAY)
    bus.write_encoder_position_offset(dev_id, float(exp["encoder_position_offset"]))
    time.sleep(DELAY)
    bus.write_encoder_velocity_filter_alpha(dev_id, float(exp["encoder_velocity_filter_alpha"]))
    time.sleep(DELAY)


def mismatches(got: dict, exp: dict) -> list[dict]:
    out = []
    for k, ev in exp.items():
        av = got.get(k)
        if not close(av, ev):
            out.append({"param": k, "actual": av, "expected": ev})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="read + print; write nothing")
    g.add_argument("--ram", action="store_true", help="write RAM only, no flash")
    g.add_argument("--flash", action="store_true",
                   help="write RAM, verify, then store_settings_to_flash")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--bus", choices=ARM_CHANNELS, default=None,
                    help="one arm bus only (default: both)")
    args = ap.parse_args()

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir or os.path.join("arm_validation", f"config_write_{stamp}")
    os.makedirs(outdir, exist_ok=True)
    backup = load_backup(BACKUP)

    mode = "dry-run" if args.dry_run else ("flash" if args.flash else "ram")
    print(f"ARM-ONLY config write  mode={mode}")
    print(f"  source: {os.path.relpath(BACKUP, REPO_ROOT)}")
    print(f"  skip: device_id, firmware_version, encoder.flux_offset")
    channels = (args.bus,) if args.bus else ARM_CHANNELS
    print(f"  buses: {list(channels)}")
    print(f"  IDs: { {ch: sorted(AS_BUILT_ARM_JOINTS[ch]) for ch in channels} }")

    report = {"created": stamp, "mode": mode, "source": BACKUP, "motors": []}
    n_fail = 0

    for ch in channels:
        bus = arm_bus(ch)
        try:
            ok, actual = verify_usb_binding(ch)
            print(f"\n{ch} USB={actual} ok={ok}")
            print(f"{'id':>3} {'joint':32} {'gear_before':>12} {'gear_after':>12} {'mism':>5} note")
            for dev_id in sorted(AS_BUILT_ARM_JOINTS[ch]):
                joint = AS_BUILT_ARM_JOINTS[ch][dev_id]
                key = CONFIG_JOINT[(ch, dev_id)]
                cfg = backup[key]
                exp = expected_from(cfg)
                if not bus.ping(dev_id):
                    print(f"{dev_id:>3} {joint:32} OFFLINE")
                    report["motors"].append({"bus": ch, "id": dev_id, "joint": joint,
                                             "offline": True})
                    n_fail += 1
                    continue
                before = snapshot(bus, dev_id)
                self_id = before["device_id"]
                note = ""
                if self_id is None:
                    note = "SDO timeout"
                    n_fail += 1
                elif self_id != dev_id:
                    note = f"self-id={self_id}"
                    n_fail += 1
                if args.dry_run:
                    after = before
                    diffs = mismatches(after, exp)
                    action = "dry"
                else:
                    write_ram(bus, dev_id, exp)
                    time.sleep(0.05)
                    after = snapshot(bus, dev_id)
                    diffs = mismatches(after, exp)
                    action = "ram"
                    if args.flash:
                        if diffs:
                            note = (note + " " if note else "") + "SKIP FLASH (verify failed)"
                        else:
                            bus.store_settings_to_flash(dev_id)
                            time.sleep(0.6)
                            bus.ping(dev_id)
                            after = snapshot(bus, dev_id)
                            diffs = mismatches(after, exp)
                            action = "flash"
                            if diffs:
                                note = (note + " " if note else "") + "POST-FLASH MISMATCH"
                if diffs and action != "dry":
                    n_fail += 1
                    if not note:
                        note = "VERIFY FAIL " + ",".join(d["param"] for d in diffs[:6])
                gb = before.get("gear_ratio")
                ga = after.get("gear_ratio")
                print(f"{dev_id:>3} {joint:32} {gb!s:>12} {ga!s:>12} {len(diffs):>5} {action} {note}")
                report["motors"].append({
                    "bus": ch, "id": dev_id, "joint": joint, "config_key": key,
                    "backup_device_id": cfg.get("device_id"),
                    "before": before, "after": after, "expected": exp,
                    "mismatches": diffs, "action": action, "note": note,
                })
        finally:
            bus.stop()

    path = os.path.join(outdir, f"{mode}.json")
    with open(path, "w") as f:
        json.dump(jsonable(report), f, indent=2)
    print(f"\nSaved {path}")
    print(f"failures={n_fail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
