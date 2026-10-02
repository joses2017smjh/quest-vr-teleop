"""PHASE 2 + 3 -- ARM-ONLY configuration and fault audit.

Reads configuration + live encoder/current/voltage from every ID that answers
on can0/can1 (scan 1-16). Prints a table. Compares ACTUAL vs EXPECTED from
source/berkeley_humanoid_lite_lowlevel/robot_configuration.backup.json.

Error bits use the firmware table (motor_controller_conf.h), not recoil/core.py.

Nothing here writes a parameter, sets a mode, or stores to flash.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arm_common import (  # noqa: E402
    ARM_CHANNELS, AS_BUILT_ARM_JOINTS, BUS_LIMB, REPO_ARM_JOINTS, SCAN_RANGE,
    ArmBus, Parameter, decode_error, joint_name, jsonable,
    link_error_summary, link_stats, mode_name,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPECTED_PATH = os.path.join(
    REPO_ROOT, "source/berkeley_humanoid_lite_lowlevel/robot_configuration.backup.json")

CONFIG_PARAMS = [
    ("device_id",                  Parameter.DEVICE_ID,                              "u32"),
    ("firmware_version",           Parameter.FIRMWARE_VERSION,                       "hex"),
    ("watchdog_timeout",           Parameter.WATCHDOG_TIMEOUT,                       "u32"),
    ("fast_frame_frequency",       Parameter.FAST_FRAME_FREQUENCY,                   "u32"),
    ("mode",                       Parameter.MODE,                                   "u32"),
    ("error",                      Parameter.ERROR,                                  "u32"),
    ("gear_ratio",                 Parameter.POSITION_CONTROLLER_GEAR_RATIO,         "f32"),
    ("position_kp",                Parameter.POSITION_CONTROLLER_POSITION_KP,        "f32"),
    ("position_ki",                Parameter.POSITION_CONTROLLER_POSITION_KI,        "f32"),
    ("velocity_kp",                Parameter.POSITION_CONTROLLER_VELOCITY_KP,        "f32"),
    ("velocity_ki",                Parameter.POSITION_CONTROLLER_VELOCITY_KI,        "f32"),
    ("torque_limit",               Parameter.POSITION_CONTROLLER_TORQUE_LIMIT,       "f32"),
    ("velocity_limit",             Parameter.POSITION_CONTROLLER_VELOCITY_LIMIT,     "f32"),
    ("position_limit_lower",       Parameter.POSITION_CONTROLLER_POSITION_LIMIT_LOWER, "f32"),
    ("position_limit_upper",       Parameter.POSITION_CONTROLLER_POSITION_LIMIT_UPPER, "f32"),
    ("position_offset",            Parameter.POSITION_CONTROLLER_POSITION_OFFSET,    "f32"),
    ("torque_filter_alpha",        Parameter.POSITION_CONTROLLER_TORQUE_FILTER_ALPHA, "f32"),
    ("i_limit",                    Parameter.CURRENT_CONTROLLER_I_LIMIT,             "f32"),
    ("i_kp",                       Parameter.CURRENT_CONTROLLER_I_KP,                "f32"),
    ("i_ki",                       Parameter.CURRENT_CONTROLLER_I_KI,                "f32"),
    ("undervoltage_threshold",     Parameter.POWERSTAGE_UNDERVOLTAGE_THRESHOLD,      "f32"),
    ("overvoltage_threshold",      Parameter.POWERSTAGE_OVERVOLTAGE_THRESHOLD,       "f32"),
    ("bus_voltage_filter_alpha",   Parameter.POWERSTAGE_BUS_VOLTAGE_FILTER_ALPHA,    "f32"),
    ("pole_pairs",                 Parameter.MOTOR_POLE_PAIRS,                       "u32"),
    ("torque_constant",            Parameter.MOTOR_TORQUE_CONSTANT,                  "f32"),
    ("phase_order",                Parameter.MOTOR_PHASE_ORDER,                      "i32"),
    ("max_calibration_current",    Parameter.MOTOR_MAX_CALIBRATION_CURRENT,          "f32"),
    ("encoder_cpr",                Parameter.ENCODER_CPR,                            "i32"),
    ("encoder_position_offset",    Parameter.ENCODER_POSITION_OFFSET,                "f32"),
    ("encoder_velocity_filter_alpha", Parameter.ENCODER_VELOCITY_FILTER_ALPHA,       "f32"),
    ("encoder_flux_offset",        Parameter.ENCODER_FLUX_OFFSET,                    "f32"),
]

LIVE_PARAMS = [
    ("bus_voltage_measured",   Parameter.POWERSTAGE_BUS_VOLTAGE_MEASURED,          "f32"),
    ("update_counter",         Parameter.POSITION_CONTROLLER_UPDATE_COUNTER,       "u32"),
    ("encoder_position_raw",   Parameter.ENCODER_POSITION_RAW,                     "u16"),
    ("encoder_n_rotations",    Parameter.ENCODER_N_ROTATIONS,                      "i32"),
    ("encoder_position",       Parameter.ENCODER_POSITION,                         "f32"),
    ("encoder_velocity",       Parameter.ENCODER_VELOCITY,                         "f32"),
    ("position_measured",      Parameter.POSITION_CONTROLLER_POSITION_MEASURED,    "f32"),
    ("velocity_measured",      Parameter.POSITION_CONTROLLER_VELOCITY_MEASURED,    "f32"),
    ("i_a_measured",           Parameter.CURRENT_CONTROLLER_I_A_MEASURED,          "f32"),
    ("i_b_measured",           Parameter.CURRENT_CONTROLLER_I_B_MEASURED,          "f32"),
    ("i_c_measured",           Parameter.CURRENT_CONTROLLER_I_C_MEASURED,          "f32"),
    ("i_q_measured",           Parameter.CURRENT_CONTROLLER_I_Q_MEASURED,          "f32"),
]

EXPECT_MAP = {
    ("position_controller", "gear_ratio"): "gear_ratio",
    ("position_controller", "position_kp"): "position_kp",
    ("position_controller", "position_ki"): "position_ki",
    ("position_controller", "velocity_kp"): "velocity_kp",
    ("position_controller", "velocity_ki"): "velocity_ki",
    ("position_controller", "torque_limit"): "torque_limit",
    ("position_controller", "velocity_limit"): "velocity_limit",
    ("position_controller", "position_limit_lower"): "position_limit_lower",
    ("position_controller", "position_limit_upper"): "position_limit_upper",
    ("position_controller", "position_offset"): "position_offset",
    ("position_controller", "torque_filter_alpha"): "torque_filter_alpha",
    ("current_controller", "i_limit"): "i_limit",
    ("current_controller", "i_kp"): "i_kp",
    ("current_controller", "i_ki"): "i_ki",
    ("powerstage", "undervoltage_threshold"): "undervoltage_threshold",
    ("powerstage", "overvoltage_threshold"): "overvoltage_threshold",
    ("powerstage", "bus_voltage_filter_alpha"): "bus_voltage_filter_alpha",
    ("motor", "pole_pairs"): "pole_pairs",
    ("motor", "torque_constant"): "torque_constant",
    ("motor", "phase_order"): "phase_order",
    ("motor", "max_calibration_current"): "max_calibration_current",
    ("encoder", "cpr"): "encoder_cpr",
    ("encoder", "position_offset"): "encoder_position_offset",
    ("encoder", "velocity_filter_alpha"): "encoder_velocity_filter_alpha",
    ("encoder", "flux_offset"): "encoder_flux_offset",
}

CALIBRATION_KEYS = {"encoder_flux_offset"}


def flatten_expected(joint_cfg: dict) -> dict:
    out = {"device_id": joint_cfg["device_id"],
           "firmware_version": joint_cfg["firmware_version"],
           "watchdog_timeout": joint_cfg["watchdog_timeout"],
           "fast_frame_frequency": joint_cfg["fast_frame_frequency"]}
    for (grp, key), flat in EXPECT_MAP.items():
        out[flat] = joint_cfg[grp][key]
    return out


def close(a, b, rel=1e-4) -> bool:
    if a is None or b is None:
        return False
    if isinstance(a, str) or isinstance(b, str):
        return str(a) == str(b)
    try:
        if math.isinf(b) or math.isinf(a):
            return a == b
        return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=1e-9)
    except (TypeError, ValueError):
        return a == b


def read_all(bus: ArmBus, dev_id: int) -> dict:
    vals = {}
    for name, pid, kind in CONFIG_PARAMS + LIVE_PARAMS:
        if kind == "u32":
            v = bus.read_u32(dev_id, pid)
        elif kind == "u16":
            v = bus.read_u16(dev_id, pid)
        elif kind == "i32":
            v = bus.read_i32(dev_id, pid)
        elif kind == "f32":
            v = bus.read_f32(dev_id, pid)
        else:
            r = bus.read_u32(dev_id, pid)
            v = None if r is None else f"0x{r:08X}"
        vals[name] = v
    raw = bus.read_raw(dev_id, Parameter.ENCODER_I2C_BUFFER)
    vals["encoder_i2c_buffer_raw"] = raw.hex() if raw else None
    if raw:
        vals["encoder_i2c_reading"] = (raw[0] << 8) | raw[1]
    return vals


def fmt(v, nd=3):
    if v is None:
        return "?"
    if isinstance(v, float):
        if not math.isfinite(v):
            return str(v)
        return f"{v:.{nd}f}"
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    cfgdir = os.path.join(args.outdir, "motor_configs")
    os.makedirs(cfgdir, exist_ok=True)

    with open(EXPECTED_PATH) as f:
        expected_all = json.load(f)

    report = {"created": dt.datetime.now().isoformat(timespec="seconds"),
              "phase": "2_3_config_and_fault_audit",
              "expected_source": os.path.relpath(EXPECTED_PATH, REPO_ROOT),
              "motors": []}

    print(f"\n{'bus':<5} {'id':>3} {'joint':32} {'mode':<11} {'err':<8} {'Vbus':>6} "
          f"{'raw':>6} {'i2c':>6} {'pos':>9} {'gear':>6} {'cpr':>5} {'upd':>4} {'mism'}")

    for ch in ARM_CHANNELS:
        before = link_error_summary(link_stats(ch))
        with ArmBus(ch) as bus:
            for dev_id in SCAN_RANGE:
                if bus.read_u32(dev_id, Parameter.DEVICE_ID) is None:
                    # still try ping once so a ping-only node is not invisible
                    if not bus.ping(dev_id):
                        continue
                vals = read_all(bus, dev_id)
                as_built = AS_BUILT_ARM_JOINTS[ch].get(dev_id)
                repo_joint = REPO_ARM_JOINTS[ch].get(dev_id)
                # Prefer as-built name; expected config still keyed by repo joint
                # except ID 12, which we compare against right_wrist_yaw if present.
                joint = as_built or repo_joint or "UNMAPPED"
                exp_key = repo_joint or (
                    "right_wrist_yaw_joint" if (ch == "can1" and dev_id == 12) else None
                )
                exp = flatten_expected(expected_all[exp_key]) if exp_key and exp_key in expected_all else None

                diffs = []
                if exp:
                    for k, ev in exp.items():
                        av = vals.get(k)
                        if not close(av, ev):
                            diffs.append({"param": k, "actual": av, "expected": ev,
                                          "is_calibration": k in CALIBRATION_KEYS})
                entry = {"bus": ch, "limb": BUS_LIMB[ch], "id": dev_id,
                         "joint": joint, "repo_joint": repo_joint,
                         "values": vals, "expected": exp, "mismatches": diffs}
                report["motors"].append(entry)
                with open(os.path.join(cfgdir, f"{ch}_id{dev_id}.json"), "w") as f:
                    json.dump(jsonable(entry), f, indent=2)
                err = decode_error(vals.get("error"))
                print(
                    f"{ch:<5} {dev_id:>3} {joint:32} "
                    f"{mode_name(vals.get('mode')):<11} "
                    f"{(err['hex'] if err else '?'):<8} "
                    f"{fmt(vals.get('bus_voltage_measured'), 1):>6} "
                    f"{fmt(vals.get('encoder_position_raw'), 0):>6} "
                    f"{fmt(vals.get('encoder_i2c_reading'), 0):>6} "
                    f"{fmt(vals.get('encoder_position'), 4):>9} "
                    f"{fmt(vals.get('gear_ratio'), 1):>6} "
                    f"{fmt(vals.get('encoder_cpr'), 0):>5} "
                    f"{fmt(vals.get('update_counter'), 0):>4} "
                    f"{len(diffs) if exp else '-'}"
                )
        report.setdefault("link", {})[ch] = {
            "before": before, "after": link_error_summary(link_stats(ch))}

    path = os.path.join(args.outdir, "phase2_3_audit.json")
    with open(path, "w") as f:
        json.dump(jsonable(report), f, indent=2)
    print(f"\nSaved {len(report['motors'])} motor configs to {cfgdir}")
    print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
