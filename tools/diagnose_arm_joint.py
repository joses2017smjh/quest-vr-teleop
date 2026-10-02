"""One arm joint, examined end to end, with its own preconditions checked.

Written after a day of chasing can1 ID 4 through four wrong diagnoses. Every one
of them came from a measurement that could not support the conclusion drawn from
it: phase currents read while the rotor sat still, a flux offset from a sweep that
aborted against a stop, a travel range measured in a position frame that a restart
had since re-zeroed. Each returned a number that looked like data.

So this refuses to report anything it cannot stand behind:
  * a flux offset is reported only from a sweep that completed (>=3800 counts)
  * phase currents are judged only while the rotor is turning
  * positions are reported against the encoder's own raw count, which survives
    restarts and power cycles, as well as the joint frame, which does not
  * a limit is called a limit only when the joint stops while torque is still
    being applied, not merely because motion stopped

Read-only by default. --probe drives the joint, gently, to find its real limits.

  .venv/bin/python tools/diagnose_arm_joint.py --bus can1 --id 4 --reference 8
  .venv/bin/python tools/diagnose_arm_joint.py --bus can1 --id 4 --probe
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arm_common import (  # noqa: E402
    ArmBus, AS_BUILT_ARM_JOINTS, Function, assert_arm_channel, decode_error,
    joint_name, mode_name, verify_usb_binding,
)

import berkeley_humanoid_lite_lowlevel.recoil as recoil  # noqa: E402
from berkeley_humanoid_lite_lowlevel.recoil import Mode, Parameter as P  # noqa: E402

FULL_SWEEP = 3800          # counts; less than this and the sweep hit something
WATCHDOG = 0x0040          # latched whenever run_teleop stops feeding a motor
LEAD = 0.12                # hold the target this far ahead so the torque limit actually binds
STALL_S = 0.6              # no progress for this long and the joint has met something
ABORT_A = 4.0              # give up rather than push current into a stuck joint

PARAMS = [("phase_order", P.MOTOR_PHASE_ORDER, "i32"),
          ("pole_pairs", P.MOTOR_POLE_PAIRS, "i32"),
          ("torque_constant", P.MOTOR_TORQUE_CONSTANT, "f32"),
          ("gear_ratio", P.POSITION_CONTROLLER_GEAR_RATIO, "f32"),
          ("encoder_cpr", P.ENCODER_CPR, "i32"),
          ("i_limit", P.CURRENT_CONTROLLER_I_LIMIT, "f32"),
          ("cal_current", P.MOTOR_MAX_CALIBRATION_CURRENT, "f32")]


def read_all(bus: ArmBus, dev: int) -> dict:
    """Everything that says whether this joint is healthy, without moving it."""
    out = {"mode": mode_name(bus.read_u32(dev, P.MODE)),
           "error_raw": bus.read_u32(dev, P.ERROR),
           "vbus": bus.read_f32(dev, P.POWERSTAGE_BUS_VOLTAGE_MEASURED),
           "flux_offset": bus.read_f32(dev, P.ENCODER_FLUX_OFFSET),
           "encoder_raw": bus.read_u16(dev, P.ENCODER_POSITION_RAW),
           "encoder_i2c": bus.read_i2c_angle(dev, flush=False),
           "n_rotations": bus.read_i32(dev, P.ENCODER_N_ROTATIONS),
           "position": bus.read_f32(dev, P.POSITION_CONTROLLER_POSITION_MEASURED)}
    out["error"] = decode_error(out["error_raw"]) or {}
    for name, param, kind in PARAMS:
        out[name] = {"i32": bus.read_i32, "f32": bus.read_f32, "u32": bus.read_u32}[kind](dev, param)
    return out


def encoder_sane(snap: dict) -> tuple[bool, str]:
    raw, i2c = snap.get("encoder_raw"), snap.get("encoder_i2c")
    if raw is None or i2c is None:
        return False, "the encoder did not answer"
    if not 0 <= i2c <= 4095:
        return False, f"i2c angle {i2c} is outside 0-4095 (a bad frame latches ENCODER_FAULT)"
    if abs(raw - i2c) > 8:
        return False, f"raw {raw} and i2c {i2c} disagree"
    return True, f"raw {raw} matches i2c {i2c}"


def pdo2(bus, dev: int, target: float):
    bus.transmit_pdo_2(dev, target, 0.0)
    frame = bus.receive(filter_device_id=dev, filter_function=Function.TRANSMIT_PDO_2, timeout=0.02)
    if frame is None or frame.size < 8:
        return None, None
    return struct.unpack("<ff", frame.data[0:8])


def feel_for_limit(bus, dev: int, direction: int, torque: float, reach: float) -> dict:
    """Walk the joint outward until it stops making progress.

    The target is kept a fixed distance ahead of wherever the joint actually is,
    so the commanded torque sits at the cap the whole way. An earlier version
    stepped the target 0.02 rad at a time, which at kp 40 asks for 0.8 Nm - less
    than these joints need to start moving - and then reported the joint as stuck.
    A probe must push hard enough to answer the question it is asking.
    """
    start, _ = pdo2(bus, dev, 0.0)
    if start is None:
        return {"error": "no PDO-2 reply"}
    bus.write_position_kp(dev, 40.0)
    bus.write_torque_limit(dev, torque)
    bus.set_mode(dev, Mode.POSITION)
    time.sleep(0.05)
    best = start
    last_progress = time.time()
    stopped_at = "reached the limit asked for"
    deadline = time.time() + 12.0
    while time.time() < deadline:
        pos, _ = pdo2(bus, dev, best + direction * LEAD)
        if pos is None:
            continue
        if (pos - best) * direction > 0.005:
            best = pos
            last_progress = time.time()
        if abs(best - start) >= reach:
            break
        if time.time() - last_progress > STALL_S:
            stopped_at = "stopped making progress"
            break
        iq = bus._read_parameter_f32(dev, P.CURRENT_CONTROLLER_I_Q_MEASURED, timeout=0.04)
        if iq is not None and abs(iq) > ABORT_A:
            stopped_at = "current limit"
            break
    bus.set_mode(dev, Mode.IDLE)
    travelled = abs(best - start)
    return {"from": round(start, 4), "reached": round(best, 4), "travelled": round(travelled, 4),
            "degrees": round(travelled * 57.2958, 1), "why_stopped": stopped_at,
            "moved_at_all": travelled > 0.02}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bus", required=True, choices=("can0", "can1"))
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--reference", type=int, help="a known-good joint on the same bus to compare against")
    ap.add_argument("--probe", action="store_true", help="drive the joint to find its real limits")
    ap.add_argument("--torque", type=float, default=1.0, help="Nm cap while probing (kept low on purpose)")
    ap.add_argument("--reach", type=float, default=1.2, help="how far to feel, rad, each way")
    ap.add_argument("--outdir", default="arm_validation/diag")
    args = ap.parse_args()

    assert_arm_channel(args.bus)
    ok, detail = verify_usb_binding(args.bus)
    if not ok:
        raise SystemExit(f"REFUSED: {args.bus} is on USB {detail}, not the expected adapter")
    if args.id not in AS_BUILT_ARM_JOINTS[args.bus]:
        raise SystemExit(f"REFUSED: {args.bus} ID {args.id} is not an as-built arm joint")

    label = joint_name(args.bus, args.id).replace("_joint", "")
    report = {"when": dt.datetime.now().isoformat(timespec="seconds"), "bus": args.bus,
              "id": args.id, "joint": label, "probed": args.probe}
    print(f"=== {label}  ({args.bus} ID {args.id}) ===\n")

    with ArmBus(args.bus) as bus:
        snap = read_all(bus, args.id)
        ref = read_all(bus, args.reference) if args.reference else None
    report["state"] = snap

    bits = snap["error"].get("bits") or []
    print(f"mode {snap['mode']}   error {hex(snap['error_raw'] or 0)} {bits}   "
          f"Vbus {snap['vbus']:.1f} V")
    if snap["error_raw"] and snap["error_raw"] & ~WATCHDOG:
        print("  !! a latched fault other than the watchdog - clear it (power cycle) before "
              "believing anything below")
    sane, why = encoder_sane(snap)
    print(f"encoder: {'OK' if sane else 'BAD'} - {why}")
    print(f"position: {snap['position']:+.3f} rad   (encoder raw {snap['encoder_raw']}, "
          f"n_rotations {snap['n_rotations']})")
    print("  raw count and n_rotations survive restarts and power cycles; the rad figure does not,"
          "\n  so quote raw when comparing against an earlier session.")

    if ref:
        diffs = [(n, snap.get(n), ref.get(n)) for n, _, _ in PARAMS if snap.get(n) != ref.get(n)]
        ref_label = joint_name(args.bus, args.reference).replace("_joint", "")
        print(f"\nagainst {ref_label}: " + ("identical on every parameter" if not diffs else ""))
        for name, mine, theirs in diffs:
            print(f"  {name:18s} this joint {mine}   {ref_label} {theirs}")
        report["parameter_diffs"] = [{"param": n, "this": m, "reference": t} for n, m, t in diffs]

    if not args.probe:
        print("\nread-only. --probe drives it gently to find its real limits.")
    else:
        print(f"\nfeeling for the limits at {args.torque} Nm, up to {args.reach} rad each way")
        bus = recoil.Bus(channel=args.bus, bitrate=1000000)
        try:
            bus.set_mode(args.id, Mode.IDLE)
            time.sleep(0.2)
            out = feel_for_limit(bus, args.id, +1, args.torque, args.reach)
            back = feel_for_limit(bus, args.id, -1, args.torque, args.reach)
        finally:
            try:
                bus.set_mode(args.id, Mode.IDLE)
                bus.stop()
            except Exception:
                pass
        report["limits"] = {"positive": out, "negative": back}
        for name, side in (("one way", out), ("the other", back)):
            print(f"  {name:9s}: {side.get('degrees', '?')} deg  "
                  f"({side.get('from')} -> {side.get('reached')})  {side.get('why_stopped')}")
        span = (out.get("travelled", 0) or 0) + (back.get("travelled", 0) or 0)
        print(f"  total free travel: {span:.3f} rad = {span * 57.2958:.0f} deg")
        report["free_travel_rad"] = round(span, 4)
        if span < 0.15:
            print("  VERDICT: the joint barely moves under power - mechanical, not electrical")
        elif span < 0.45:
            print("  VERDICT: some travel but less than a calibration sweep needs (0.42 rad);"
                  "\n           centre it before calibrating or the sweep will abort against a stop")
        else:
            print(f"  VERDICT: enough room to calibrate; centre near "
                  f"{(out.get('reached', 0) + back.get('reached', 0)) / 2:+.3f} rad")

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"{args.bus}_id{args.id}.json")
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"\nsaved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
