"""ARM-ONLY electrical calibration (flux offset) for one Recoil actuator.

Sends NMT MODE_CALIBRATION. Firmware then:
  * ramps voltage until max_calibration_current
  * turns one motor revolution forward and back (~24 deg at 15:1)
  * stores flash and returns to IDLE

USB-guarded. Never opens a leg bus. One device only.

  .venv/bin/python tools/calibrate_arm_joint.py --bus can0 --id 1
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arm_common import (  # noqa: E402
    ARM_CHANNELS, AS_BUILT_ARM_JOINTS, Function, Parameter, decode_error,
    jsonable, joint_name, mode_name,
    ArmBus,
)

from berkeley_humanoid_lite_lowlevel.recoil import Mode


def unwrapped_span(raws: list[int], cpr: int = 4096) -> int:
    """How far the rotor actually turned, with the 0/4095 wrap taken out.

    max-minus-min counts a single wrap as a whole revolution: a joint that moved
    four counts backwards across the boundary reported a 4095-count sweep and a
    flux offset that looked valid. This follows the steps instead, so a wrap costs
    one count rather than four thousand.
    """
    if len(raws) < 2:
        return 0
    position = 0.0
    seen = [0.0]
    for before, after in zip(raws, raws[1:]):
        step = after - before
        if step > cpr / 2:
            step -= cpr
        elif step < -cpr / 2:
            step += cpr
        position += step
        seen.append(position)
    return int(round(max(seen) - min(seen)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bus", required=True, choices=ARM_CHANNELS)
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--outdir", default="arm_validation/cal")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ch, dev_id = args.bus, args.id
    if dev_id not in AS_BUILT_ARM_JOINTS[ch]:
        raise SystemExit(f"REFUSED: {ch} ID {dev_id} is not an as-built arm joint")
    joint = joint_name(ch, dev_id)

    rows = []
    with ArmBus(ch) as bus:
        pre = bus.snapshot_live(dev_id)
        err = decode_error(pre.get("error_raw"))
        mode = mode_name(pre.get("mode_raw"))
        vbus = pre.get("bus_voltage")
        raw = pre.get("encoder_position_raw")
        i2c = pre.get("encoder_i2c_reading")
        gear = pre.get("gear_ratio")
        ical = pre.get("encoder_flux_offset")
        print(f"{ch} ID {dev_id} {joint}")
        print(f"  pre: mode={mode} err={(err or {}).get('hex')} Vbus={vbus} "
              f"raw={raw} i2c={i2c} gear={gear} flux={ical} "
              f"Ical={pre.get('pole_pairs')}pp")
        if mode != "IDLE":
            raise SystemExit(f"REFUSED: need IDLE, got {mode}")
        if not isinstance(vbus, float) or vbus < 15:
            raise SystemExit(f"REFUSED: Vbus {vbus} < 15 V")
        if not isinstance(raw, int) or not isinstance(i2c, int) or i2c >= 4096 or raw != i2c:
            raise SystemExit(f"REFUSED: encoder raw={raw} i2c={i2c}")
        if not isinstance(gear, float) or abs(gear) < 10:
            raise SystemExit(f"REFUSED: gear_ratio {gear} (want ±15)")

        print("  sending MODE_CALIBRATION — joint will move ~±24 deg")
        bus._send(dev_id, Function.NMT, struct.pack("<BB", Mode.CALIBRATION, dev_id))
        t0 = time.time()
        saw_cal = False
        last = None
        while time.time() - t0 < args.timeout:
            snap = bus.snapshot_live(dev_id, flush=True)
            t = time.time() - t0
            m = mode_name(snap.get("mode_raw"))
            e = decode_error(snap.get("error_raw"))
            row = {"t": t, "mode": m, "error": None if not e else e["hex"],
                   "bits": None if not e else e["bits"],
                   "raw": snap.get("encoder_position_raw"),
                   "i2c": snap.get("encoder_i2c_reading"),
                   "flux": snap.get("encoder_flux_offset"),
                   "vbus": snap.get("bus_voltage")}
            rows.append(row)
            key = (m, row["error"], row["raw"], round(row["flux"] or 0, 3))
            if key != last:
                print(f"  t={t:5.1f}s  mode={m:<11} err={row['error']} "
                      f"raw={row['raw']} i2c={row['i2c']} flux={row['flux']}")
                last = key
            if m == "CALIBRATION":
                saw_cal = True
            if e and "CALIBRATION_ERROR" in (e.get("bits") or []):
                print("  ABORT: CALIBRATION_ERROR")
                break
            if saw_cal and m == "IDLE":
                print("  done: back to IDLE")
                break
            time.sleep(0.15)
        else:
            print("  ABORT: timeout")

        post = bus.snapshot_live(dev_id)
        err2 = decode_error(post.get("error_raw"))
        print(f"  post: mode={mode_name(post.get('mode_raw'))} "
              f"err={(err2 or {}).get('hex')} {(err2 or {}).get('bits')} "
              f"raw={post.get('encoder_position_raw')} "
              f"flux={post.get('encoder_flux_offset')} "
              f"gear={post.get('gear_ratio')}")

    raws = [r["raw"] for r in rows if isinstance(r["raw"], int)]
    travel = unwrapped_span(raws)
    ok = (mode_name(post.get("mode_raw")) == "IDLE"
          and saw_cal
          and not (err2 and err2.get("raw"))
          and isinstance(post.get("encoder_flux_offset"), float)
          and post.get("encoder_flux_offset") != ical)
    path = os.path.join(args.outdir, f"{ch}_id{dev_id}.json")
    with open(path, "w") as f:
        json.dump(jsonable({
            "bus": ch, "id": dev_id, "joint": joint, "ok": ok,
            "pre": pre, "post": post, "travel_counts": travel, "n": len(rows),
        }), f, indent=2)
    print(f"  travel={travel} counts  ok={ok}  log={path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
