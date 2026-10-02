"""Tiny closed-loop POSITION nudge on one as-built arm joint.

Hold current pose 1 s, move +delta rad (joint frame), hold 2 s, return, IDLE.
Low Kp / torque. Watchdog fed by PDO-2. Always drops to IDLE on exit.

  .venv/bin/python tools/tiny_move_arm_joint.py --bus can0 --id 1
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys
import time

import berkeley_humanoid_lite_lowlevel.recoil as recoil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from write_arm_config import arm_bus  # noqa: E402

from arm_common import AS_BUILT_ARM_JOINTS, Function, Parameter, joint_name, mode_name


DT = 0.01
ABORT_IQ = 4.0
KP, KD, TQ = 0.2, 0.005, 0.2


def pdo2(bus, dev_id, pos):
    bus.transmit_pdo_2(dev_id, pos, 0.0)
    frame = bus.receive(filter_device_id=dev_id, filter_function=Function.TRANSMIT_PDO_2,
                        timeout=0.02)
    if frame is None or frame.size < 8:
        return None, None
    return struct.unpack("<ff", frame.data[0:8])


def run_phase(bus, dev_id, target, seconds, label, start_t):
    t_end = time.time() + seconds
    last_print = 0.0
    meas = []
    while time.time() < t_end:
        pos, vel = pdo2(bus, dev_id, target)
        now = time.time() - start_t
        if pos is not None:
            meas.append(pos)
            err = pos - target
            if now - last_print >= 0.2:
                iq = bus._read_parameter_f32(dev_id, Parameter.CURRENT_CONTROLLER_I_Q_MEASURED,
                                             timeout=0.03)
                print(f"  t={now:5.2f}s  {label:8}  tgt={target:+.4f}  meas={pos:+.4f}  "
                      f"err={err:+.4f}  vel={vel:+.3f}  iq={iq}")
                last_print = now
                if iq is not None and abs(iq) > ABORT_IQ:
                    raise RuntimeError(f"i_q {iq:.2f} A over abort")
                mode = bus._read_parameter_u32(dev_id, Parameter.MODE, timeout=0.03)
                errb = bus._read_parameter_u32(dev_id, Parameter.ERROR, timeout=0.03)
                if mode is not None and mode != recoil.Mode.POSITION:
                    raise RuntimeError(f"left POSITION, mode={mode_name(mode)}")
                if errb & ~allow_error:
                    raise RuntimeError(f"error 0x{errb:04X}")
        time.sleep(DT)
    return meas


allow_error = 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bus", required=True, choices=("can0", "can1"))
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--delta", type=float, default=0.10, help="joint-frame radians")
    ap.add_argument("--kp", type=float, default=KP)
    ap.add_argument("--kd", type=float, default=KD)
    ap.add_argument("--torque", type=float, default=TQ)
    # Losing its host latches WATCHDOG_TIMEOUT (0x0040) on every motor, healthy or
    # not, so a diagnostic that refuses to run with it cannot run at all after
    # run_teleop stops. Anything else still aborts.
    ap.add_argument("--allow-error", default="0", metavar="MASK",
                    help="error bits to tolerate, e.g. 0x40 for a stale watchdog latch")
    args = ap.parse_args()
    global allow_error
    allow_error = int(str(args.allow_error), 0)
    ch, dev_id = args.bus, args.id
    kp, kd, tq = args.kp, args.kd, args.torque
    if dev_id not in AS_BUILT_ARM_JOINTS[ch]:
        raise SystemExit(f"REFUSED: {ch} ID {dev_id} is not an as-built arm joint")
    joint = joint_name(ch, dev_id)

    bus = arm_bus(ch)
    engaged = False
    try:
        if not bus.ping(dev_id):
            raise SystemExit("target does not ping")
        mode = bus._read_parameter_u32(dev_id, Parameter.MODE, timeout=0.08)
        vbus = bus._read_parameter_f32(dev_id, Parameter.POWERSTAGE_BUS_VOLTAGE_MEASURED, timeout=0.08)
        gear = bus.read_gear_ratio(dev_id)
        pos0 = bus.read_position_measured(dev_id)
        print(f"{ch} ID {dev_id} {joint}")
        print(f"  pre: mode={mode_name(mode)} Vbus={vbus} gear={gear} pos={pos0}")
        if mode != recoil.Mode.IDLE:
            raise SystemExit(f"REFUSED: need IDLE, got {mode_name(mode)}")
        if vbus is None or vbus < 15:
            raise SystemExit(f"REFUSED: Vbus {vbus}")
        if pos0 is None or not math.isfinite(pos0):
            raise SystemExit("REFUSED: unreadable position")
        if gear is None or abs(gear) < 10:
            raise SystemExit(f"REFUSED: gear {gear}")

        target_hold = pos0
        target_nudge = pos0 + args.delta
        print(f"  kp={kp} kd={kd} torque_limit={tq}  hold {target_hold:.4f} then {target_nudge:.4f} "
              f"(delta {args.delta:.3f} rad / {math.degrees(args.delta):.1f} deg)")

        bus.write_position_kp(dev_id, kp)
        bus.write_position_kd(dev_id, kd)
        bus.write_torque_limit(dev_id, tq)
        bus.set_mode(dev_id, recoil.Mode.POSITION)
        bus.feed(dev_id)
        engaged = True
        t0 = time.time()

        hold = run_phase(bus, dev_id, target_hold, 1.0, "hold", t0)
        nudge = run_phase(bus, dev_id, target_nudge, 2.0, "nudge", t0)
        back = run_phase(bus, dev_id, target_hold, 1.0, "return", t0)

        def last(xs, fallback):
            return xs[-1] if xs else fallback
        print(f"  hold_end={last(hold, None)}  nudge_end={last(nudge, None)}  "
              f"return_end={last(back, None)}")
        if nudge:
            moved = last(nudge, pos0) - pos0
            print(f"  measured_delta={moved:+.4f} rad  commanded={args.delta:+.4f}")
            ok = abs(moved) > abs(args.delta) * 0.4
        else:
            ok = False
        return 0 if ok else 1
    except (RuntimeError, SystemExit) as e:
        print("ABORT:", e)
        return 1
    finally:
        if engaged:
            try:
                bus.set_mode(dev_id, recoil.Mode.IDLE)
                time.sleep(0.05)
                bus.write_position_kp(dev_id, 50.0)
                bus.write_position_kd(dev_id, 2.0)
                bus.write_torque_limit(dev_id, 1.0)
                print("  released IDLE, restored kp=50 kd=2 torque=1 (RAM)")
            except Exception as e:
                print("  release error:", e)
        bus.stop()


if __name__ == "__main__":
    raise SystemExit(main())
