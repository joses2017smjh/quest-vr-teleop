"""Closed-loop check for one arm actuator: does POSITION mode move the joint where it is
told, in the right direction, with the gains and torque limit it actually has?

Sequence, in the joint frame, from wherever the joint is now: hold, +step, back, -step,
back. Targets go out by PDO-2 at about 200 Hz, which also feeds the watchdog, and each
reply carries the measured position and velocity. i_q, i_d, mode and error are read by
SDO every 10th cycle.

Aborts to IDLE on a wrong-way move, a runaway, a mode change, a new error bit, or current
above what the torque limit allows. Only settings passed on the command line are changed,
in RAM, and the values that were there are put back afterwards. Never stores flash.

  .venv/bin/python tools/closed_loop_check.py --bus can0 --id 9
  .venv/bin/python tools/closed_loop_check.py --bus can0 --id 9 --step-deg 8 --torque-limit 2
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arm_common import (  # noqa: E402
    ARM_CHANNELS, AS_BUILT_ARM_JOINTS, Function, Mode, decode_error, joint_name, jsonable,
    mode_name,
)
from field_step_probe import (  # noqa: E402
    ERROR, FLUX, GEAR_RATIO, MODE, PHASE_ORDER, VBUS, ProbeBus, teleop_running,
)

POS_KP, POS_KI, VEL_KP, TORQUE_LIMIT = 0x020, 0x024, 0x028, 0x030
TORQUE_TARGET, POS_MEASURED = 0x044, 0x060
I_Q, I_D, TORQUE_CONSTANT = 0x0C0, 0x0C4, 0x108
PERIOD = 0.005


class Abort(RuntimeError):
    pass


def pdo2(bus: ProbeBus, dev_id: int, target: float):
    bus._send(dev_id, Function.RECEIVE_PDO_2, struct.pack("<ff", target, 0.0))
    d = bus._recv(dev_id, Function.TRANSMIT_PDO_2, 0.01)
    if d is None or len(d) < 8:
        return None, None
    return struct.unpack("<ff", d[:8])


def analyse(log: list, moves: list, step: float) -> list:
    out = []
    for m in moves:
        rows = [r for r in log if m["t0"] <= r["t"] < m["t1"] and r["pos"] is not None]
        if len(rows) < 10:
            out.append({**m, "verdict": "no data"})
            continue
        t = np.array([r["t"] for r in rows]) - m["t0"]
        pos = np.array([r["pos"] for r in rows])
        start, target = m["start"], m["target"]
        want = target - start
        final = float(np.median(pos[t > t[-1] - 0.25]))
        prog = (pos - start) * (1 if want >= 0 else -1)       # progress toward target
        wrong = float(max(0.0, -prog.min()))
        reach = np.nonzero(prog >= 0.9 * abs(want))[0]
        iq = [abs(r["iq"]) for r in rows if r.get("iq") is not None]
        idd = [r["id"] for r in rows if r.get("id") is not None]
        res = {
            **m,
            "final": final,
            "error_deg": math.degrees(final - target),
            "travel_fraction": (final - start) / want if want else None,
            "wrong_way_deg": math.degrees(wrong),
            "rise_90_s": float(t[reach[0]]) if len(reach) else None,
            "overshoot_deg": math.degrees(max(0.0, float(prog.max()) - abs(want))) if want else 0.0,
            "peak_iq_A": max(iq) if iq else None,
            "mean_id_A": float(np.mean(idd)) if idd else None,
        }
        ok = (res["wrong_way_deg"] < 0.5 and abs(res["error_deg"]) <= max(1.0, 0.2 * math.degrees(step))
              and (res["travel_fraction"] or 0) >= 0.8)
        # a return from a move that never happened is not a second failure
        res["verdict"] = ("n/a (never left)" if abs(math.degrees(want)) < 0.5 * math.degrees(step)
                          and abs(res["error_deg"]) < 1.0 else ("PASS" if ok else "FAIL"))
        out.append(res)
    return out


def plot(log: list, moves: list, meta: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = [r["t"] for r in log if r["pos"] is not None]
    pos = [math.degrees(r["pos"] - meta["start"]) for r in log if r["pos"] is not None]
    tgt = [math.degrees(r["target"] - meta["start"]) for r in log if r["pos"] is not None]
    ti = [r["t"] for r in log if r.get("iq") is not None]
    iq = [r["iq"] for r in log if r.get("iq") is not None]
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(t, tgt, "k--", lw=1, label="target")
    ax[0].plot(t, pos, lw=1.2, label="measured")
    ax[0].set_ylabel("joint angle from start (deg)")
    ax[0].legend(fontsize=8)
    ax[0].set_title(f"{meta['bus']} ID {meta['id']} {meta['joint']}  kp {meta['kp']:g}  "
                    f"kd {meta['kd']:g}  torque limit {meta['torque_limit']:g} Nm  {meta['timestamp']}")
    ax[1].plot(ti, iq, ".-", ms=3, lw=0.8)
    lim = meta["iq_at_limit"]
    ax[1].axhline(lim, color="r", lw=0.8)
    ax[1].axhline(-lim, color="r", lw=0.8, label="current at the torque limit")
    ax[1].set_ylabel("i_q (A)")
    ax[1].set_xlabel("time (s)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bus", required=True, choices=ARM_CHANNELS)
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--step-deg", type=float, default=5.0, help="joint-frame step size")
    ap.add_argument("--dwell", type=float, default=2.0, help="seconds at each target")
    ap.add_argument("--kp", type=float, help="position kp for this test (RAM, restored)")
    ap.add_argument("--kd", type=float, help="velocity kp for this test (RAM, restored)")
    ap.add_argument("--torque-limit", type=float, help="Nm for this test (RAM, restored)")
    ap.add_argument("--outdir", default="arm_validation/closed_loop")
    args = ap.parse_args()

    ch, dev = args.bus, args.id
    if dev not in AS_BUILT_ARM_JOINTS[ch]:
        raise SystemExit(f"REFUSED: {ch} ID {dev} is not an as-built arm joint")
    if teleop_running():
        raise SystemExit("REFUSED: run_teleop.py is running and would fight this test.")
    joint = joint_name(ch, dev)
    step = math.radians(args.step_deg)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(args.outdir, f"{stamp}_{ch}_id{dev}")
    os.makedirs(outdir, exist_ok=True)

    log, moves, events = [], [], []
    meta = {"bus": ch, "id": dev, "joint": joint, "timestamp": stamp, "step_deg": args.step_deg}
    abort_reason = None
    changed: dict = {}

    with ProbeBus(ch) as bus:
        mode = bus.get(dev, MODE, "u32")
        vbus = bus.get(dev, VBUS)
        gear = bus.get(dev, GEAR_RATIO)
        kt = bus.get(dev, TORQUE_CONSTANT)
        orig = {"kp": bus.get(dev, POS_KP), "ki": bus.get(dev, POS_KI),
                "kd": bus.get(dev, VEL_KP), "torque_limit": bus.get(dev, TORQUE_LIMIT)}
        p0 = bus.get(dev, POS_MEASURED)
        print(f"{ch} ID {dev} {joint}: mode {mode_name(mode)} Vbus {vbus:.2f} gear {gear} "
              f"kt {kt} phase order {bus.get(dev, PHASE_ORDER, 'i32')} flux {bus.get(dev, FLUX)}")
        print(f"  gains in the controller: kp {orig['kp']} ki {orig['ki']} kd {orig['kd']} "
              f"torque limit {orig['torque_limit']} Nm; position {p0}")
        if mode != Mode.IDLE:
            raise SystemExit(f"REFUSED: need IDLE, got {mode_name(mode)}")
        if vbus is None or vbus < 15 or gear is None or abs(gear) < 10 or not kt:
            raise SystemExit(f"REFUSED: Vbus {vbus} gear {gear} kt {kt}")
        if p0 is None or not math.isfinite(p0):
            raise SystemExit("REFUSED: unreadable position")

        want = {"kp": args.kp, "kd": args.kd, "torque_limit": args.torque_limit}
        params = {"kp": POS_KP, "kd": VEL_KP, "torque_limit": TORQUE_LIMIT}
        eff = dict(orig)
        try:
            for k, v in want.items():
                if v is not None and v != orig[k]:
                    if not bus.put(dev, params[k], "f32", v):
                        raise Abort(f"{k} write did not read back")
                    changed[k] = orig[k]
                    eff[k] = v
            if not bus.put(dev, TORQUE_TARGET, "f32", 0.0):
                raise Abort("could not zero the feed-forward torque")
            bus.put(dev, ERROR, "u32", 0)
            iq_limit = eff["torque_limit"] / (kt * abs(gear))
            meta.update(kp=eff["kp"], kd=eff["kd"], ki=eff["ki"], torque_limit=eff["torque_limit"],
                        iq_at_limit=iq_limit, start=p0, gear=gear, kt=kt, original_gains=orig)
            print(f"  test gains: kp {eff['kp']} kd {eff['kd']} torque limit {eff['torque_limit']} Nm "
                  f"(i_q at the limit {iq_limit:.2f} A); steps of {args.step_deg:g} deg")

            # PDO-2 with the current position first, so entering POSITION holds still
            pdo2(bus, dev, p0)
            if not bus.set_mode(dev, Mode.POSITION):
                raise Abort("controller refused POSITION mode")
            t0 = time.time()
            plan = [("hold", 0.0, 1.0), ("+step", +step, args.dwell), ("back", 0.0, args.dwell),
                    ("-step", -step, args.dwell), ("back", 0.0, args.dwell)]
            prev_target = p0
            missing = 0
            last_pos = p0
            for name, offset, dur in plan:
                target = p0 + offset
                m = {"phase": name, "t0": time.time() - t0, "start": last_pos, "target": target}
                t_end = time.time() + dur
                k = 0
                while time.time() < t_end:
                    tc = time.time()
                    pos, vel = pdo2(bus, dev, target)
                    row = {"t": tc - t0, "phase": name, "target": target, "pos": pos, "vel": vel}
                    if k % 10 == 0:
                        row.update(iq=bus.get(dev, I_Q), id=bus.get(dev, I_D),
                                   mode=bus.get(dev, MODE, "u32"), err=bus.get(dev, ERROR, "u32"))
                        if row["mode"] is not None and row["mode"] != Mode.POSITION:
                            raise Abort(f"left POSITION: now {mode_name(row['mode'])} "
                                        f"({decode_error(row['err'])})")
                        if row["err"]:
                            raise Abort(f"error appeared: {decode_error(row['err'])}")
                        if row["iq"] is not None and abs(row["iq"]) > 2 * iq_limit + 0.5:
                            raise Abort(f"i_q {row['iq']:.2f} A, far above the {iq_limit:.2f} A "
                                        f"the torque limit allows")
                    log.append(row)
                    k += 1
                    if pos is None:
                        missing += 1
                        if missing >= 20:
                            raise Abort("20 PDO-2 replies in a row missing")
                    else:
                        missing = 0
                        last_pos = pos
                        if abs(pos - p0) > 2 * step + math.radians(5):
                            raise Abort(f"runaway: {math.degrees(pos - p0):+.1f} deg from start")
                        move = target - prev_target
                        if move and (pos - prev_target) * math.copysign(1, move) < -math.radians(3):
                            raise Abort(f"moving the WRONG WAY: {math.degrees(pos - prev_target):+.1f} "
                                        f"deg while the target is {math.degrees(move):+.1f} deg away")
                    time.sleep(max(0.0, PERIOD - (time.time() - tc)))
                m["t1"] = time.time() - t0
                if name != "hold":
                    moves.append(m)
                prev_target = target
        except Abort as exc:
            abort_reason = str(exc)
            print(f"  ABORT: {abort_reason}")
        except KeyboardInterrupt:
            abort_reason = "interrupted"
            print("  ABORT: interrupted")
        finally:
            bus.set_mode(dev, Mode.IDLE) or bus.set_mode(dev, Mode.IDLE, wait=1.0)
            for k, v in changed.items():
                ok = bus.put(dev, params[k], "f32", v)
                events.append(f"{k} restored to {v}" + ("" if ok else " (READBACK FAILED)"))
            end_mode = bus.get(dev, MODE, "u32")
            print(f"  end: {mode_name(end_mode)}; " + ("; ".join(events) if events else "nothing to restore"))

    results = analyse(log, moves, step)
    meta["abort"] = abort_reason
    lines = []
    if abort_reason:
        lines.append(f"ABORTED: {abort_reason}")
    for r in results:
        if r.get("verdict") == "no data":
            lines.append(f"  {r['phase']:>6}: no data")
            continue
        rise = f"{r['rise_90_s']:.2f} s" if r["rise_90_s"] is not None else "never"
        lines.append(
            f"  {r['phase']:>6} to {math.degrees(r['target'] - p0):+5.1f} deg: ended "
            f"{math.degrees(r['final'] - p0):+6.2f} (error {r['error_deg']:+5.2f}), reached "
            f"{100 * (r['travel_fraction'] or 0):4.0f}%, 90% in {rise}, wrong-way "
            f"{r['wrong_way_deg']:.2f}, overshoot {r['overshoot_deg']:.2f}, peak i_q "
            f"{(r['peak_iq_A'] or 0):.2f} A  {r['verdict']}")
    passed = (results and all(r.get("verdict") in ("PASS", "n/a (never left)") for r in results)
              and any(r.get("verdict") == "PASS" for r in results) and not abort_reason)
    lines.append("CLOSED LOOP: " + ("PASS - it goes where it is told, both ways"
                                    if passed else "FAIL"))
    with open(os.path.join(outdir, "log.json"), "w") as fh:
        json.dump(jsonable({"meta": meta, "moves": results, "log": log}), fh)
    with open(os.path.join(outdir, "report.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    try:
        plot(log, moves, meta, os.path.join(outdir, "closed_loop.png"))
    except Exception as exc:
        print(f"  plot failed: {exc}")
    print("\n".join(lines))
    print(f"log: {outdir}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
