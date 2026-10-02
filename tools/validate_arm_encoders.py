"""ARM encoder validation for Berkeley Humanoid Lite (can0 left, can1 right).

Reads AS5600 ANGLE via the Recoil SDO map. Never writes a parameter, never
sends NMT, never stores flash. Ping (PDO-1) is used for discovery and does
reset the firmware watchdog -- that is intended.

Sensor of record is ENCODER_POSITION_RAW (0x12C), the integer AS5600 ANGLE
the 10 kHz loop latched. Derived ENCODER_POSITION / POSITION_MEASURED are
also captured, but they are divided by gear_ratio (currently 1.0 on most
boards, 0.0 on a dead init) so they are not the verdict source.

ENCODER_I2C_BUFFER (0x118) is the two bytes encoder.c compares against cpr
before raising ERROR_ENCODER_FAULT. It is NOT 12-bit masked first.

Default mode is ``full``: scan 1-16, snapshot every live field, fast raw
stream per actuator, then a round-robin pass on each bus. No prompts, no
mode refusals, no skipping a node that answers ping but not the first SDO.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arm_common import (  # noqa: E402
    ARM_CHANNELS, AS_BUILT_ARM_JOINTS, BUS_LIMB, CPR, REPO_ARM_JOINTS,
    SCAN_RANGE, ArmBus, Parameter, decode_error, joint_name, jsonable,
    link_error_summary, link_stats, mode_name, target_ids,
)

STATIONARY_PASS_P2P = 8
STATIONARY_FAIL_P2P = 40
JUMP_COUNTS = 200


def unwrap(counts: list[int], cpr: int = CPR) -> list[int]:
    if not counts:
        return []
    out = [counts[0]]
    off = 0
    for prev, cur in zip(counts, counts[1:]):
        d = cur - prev
        if d > cpr // 2:
            off -= cpr
        elif d < -cpr // 2:
            off += cpr
        out.append(cur + off)
    return out


def circular_stats(raw: list[int], cpr: int = CPR) -> dict:
    if not raw:
        return {"R": None, "centre_counts": None, "mad": None}
    ang = [2 * math.pi * (v % cpr) / cpr for v in raw]
    cx = sum(math.cos(a) for a in ang) / len(ang)
    cy = sum(math.sin(a) for a in ang) / len(ang)
    R = math.hypot(cx, cy)
    centre = (math.atan2(cy, cx) % (2 * math.pi)) * cpr / (2 * math.pi)
    dev = sorted(min(abs(v - centre), cpr - abs(v - centre)) for v in raw)
    return {
        "R": round(R, 4),
        "centre_counts": round(centre, 2),
        "centre_deg": round(centre * 360.0 / cpr, 2),
        "mad": round(dev[len(dev) // 2], 2),
    }


def histogram(raw: list[int], bins: int = 16, cpr: int = CPR) -> dict:
    if not raw:
        return {}
    width = cpr // bins
    counts = [0] * bins
    for v in raw:
        counts[min((v % cpr) // width, bins - 1)] += 1
    high = collections.Counter((v >> 8) & 0xFF for v in raw)
    return {
        "bins": bins,
        "counts": counts,
        "high_byte": {str(k): high[k] for k in sorted(high)},
        "high_byte_max": max(high) if high else None,
        "high_byte_min": min(high) if high else None,
    }


def analyse(rows: list[dict], duration: float, misses: int, cpr: int = CPR) -> dict:
    n = len(rows)
    if n < 2:
        return {"samples": n, "read_misses": misses, "duration_s": round(duration, 3),
                "verdict": "NO_DATA"}
    raw = [int(r["position_raw"]) for r in rows]
    uw = unwrap(raw, cpr)
    deltas = [b - a for a, b in zip(uw, uw[1:])]
    dts = [b["t"] - a["t"] for a, b in zip(rows, rows[1:])]
    i2c_vals = [r["i2c_reading"] for r in rows if r.get("i2c_reading") is not None]
    bad_range = [v for v in i2c_vals if v >= cpr]
    mismatch = 0
    for r in rows:
        i2c = r.get("i2c_reading")
        if i2c is None:
            continue
        d = abs(i2c - (r["position_raw"] & (cpr - 1)))
        if 4 < d < cpr - 4:
            mismatch += 1
    circ = circular_stats(raw, cpr)
    hist = histogram(raw, 16, cpr)
    extras = [r for r in rows if r.get("update_counter") is not None]
    counters = [r["update_counter"] for r in extras]
    loop_alive = None
    if len(counters) >= 2:
        loop_alive = counters[-1] != counters[0] or len(set(counters)) > 1
    out = {
        "samples": n,
        "duration_s": round(duration, 3),
        "sample_rate_hz": round(n / duration, 1) if duration else 0,
        "read_misses": misses,
        "min": min(raw),
        "max": max(raw),
        "first5": raw[:5],
        "last5": raw[-5:],
        "mean_unwrapped": round(statistics.fmean(uw), 2),
        "stdev_unwrapped": round(statistics.pstdev(uw), 3),
        "p2p_counts": max(uw) - min(uw),
        "p2p_deg_rotor": round((max(uw) - min(uw)) * 360.0 / cpr, 3),
        "repeated_frac": round(sum(1 for d in deltas if d == 0) / max(len(deltas), 1), 3),
        "jumps_gt_%d" % JUMP_COUNTS: sum(1 for d in deltas if abs(d) > JUMP_COUNTS),
        "largest_step": max((abs(d) for d in deltas), default=0),
        "distinct_values": len(set(raw)),
        "i2c_samples": len(i2c_vals),
        "i2c_out_of_range": len(bad_range),
        "i2c_vs_posraw_mismatch": mismatch,
        "i2c_min": min(i2c_vals) if i2c_vals else None,
        "i2c_max": max(i2c_vals) if i2c_vals else None,
        "median_dt_ms": round(1000 * statistics.median(dts), 2) if dts else None,
        "loop_alive": loop_alive,
        "update_counter_first": counters[0] if counters else None,
        "update_counter_last": counters[-1] if counters else None,
        "circular": circ,
        "histogram_16": hist,
    }
    return out


def verdict_from(a: dict, snap: dict) -> tuple[str, str]:
    vbus = snap.get("bus_voltage")
    upd = snap.get("update_counter")
    raw = snap.get("encoder_position_raw")
    if a.get("samples", 0) < 5 and raw is None:
        if vbus == 0.0 and (upd == 0 or upd is None):
            return "BOARD_DEAD", "no encoder samples, Vbus=0, update_counter frozen -- init never finished"
        return "NO_DATA", "node answered poorly; no encoder stream"
    if a.get("i2c_out_of_range"):
        return "ENCODER_OOR", (
            f"{a['i2c_out_of_range']} I2C ANGLE values >= cpr; firmware encoder.c:55 "
            "raises ERROR_ENCODER_FAULT on this exact condition"
        )
    if a.get("loop_alive") is False and (vbus == 0.0 or vbus is None):
        return "BOARD_DEAD", "control loop not running (update_counter frozen, Vbus not live)"
    p2p = a.get("p2p_counts", 0)
    if p2p > STATIONARY_FAIL_P2P:
        R = (a.get("circular") or {}).get("R")
        return "ENCODER_NOISY", (
            f"peak-to-peak {p2p} counts ({a.get('p2p_deg_rotor')} deg rotor) while "
            f"we did not command motion; circular R={R}"
        )
    if a.get("loop_alive") is False:
        return "LOOP_DEAD", "raw encoder present but update_counter never advanced"
    if a.get("repeated_frac") == 1.0 or a.get("distinct_values") == 1:
        return "ENCODER_CONSTANT", (
            f"raw stuck at {a.get('min')} with loop "
            f"{'alive' if a.get('loop_alive') else 'unknown'}; quiet magnet OR frozen sensor -- "
            "hand-rotate to distinguish"
        )
    if p2p > STATIONARY_PASS_P2P:
        return "ENCODER_JITTER", f"p2p {p2p} counts, above {STATIONARY_PASS_P2P}-count quiet budget"
    return "ENCODER_QUIET", f"p2p {p2p} counts, stdev {a.get('stdev_unwrapped')}, loop_alive={a.get('loop_alive')}"


def write_csv(path: str, rows: list[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ["t", "position_raw", "i2c_reading", "n_rotations", "encoder_position",
              "encoder_velocity", "update_counter", "vbus", "error", "mode"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def sample_stream(bus: ArmBus, dev_id: int, duration: float, extra_every: int = 40) -> dict:
    """Max-rate ENCODER_POSITION_RAW. Extra fields every `extra_every` samples."""
    t0 = time.time()
    rows = []
    misses = 0
    i = 0
    bus.flush()
    while time.time() - t0 < duration:
        t = time.time()
        raw = bus.read_u16(dev_id, Parameter.ENCODER_POSITION_RAW, flush=False)
        if raw is None:
            misses += 1
            continue
        row = {"t": t - t0, "position_raw": raw}
        if i % extra_every == 0:
            row["i2c_reading"] = bus.read_i2c_angle(dev_id, flush=False)
            row["n_rotations"] = bus.read_i32(dev_id, Parameter.ENCODER_N_ROTATIONS, flush=False)
            row["encoder_position"] = bus.read_f32(dev_id, Parameter.ENCODER_POSITION, flush=False)
            row["encoder_velocity"] = bus.read_f32(dev_id, Parameter.ENCODER_VELOCITY, flush=False)
            row["update_counter"] = bus.read_u32(
                dev_id, Parameter.POSITION_CONTROLLER_UPDATE_COUNTER, flush=False)
            row["vbus"] = bus.read_f32(
                dev_id, Parameter.POWERSTAGE_BUS_VOLTAGE_MEASURED, flush=False)
            row["error"] = bus.read_u32(dev_id, Parameter.ERROR, flush=False)
            row["mode"] = bus.read_u32(dev_id, Parameter.MODE, flush=False)
        rows.append(row)
        i += 1
    return {"rows": rows, "misses": misses, "duration": time.time() - t0}


def print_snap(ch: str, dev_id: int, snap: dict):
    err = decode_error(snap.get("error_raw"))
    fw = snap.get("firmware_version")
    fw_s = f"0x{fw:08X}" if isinstance(fw, int) else "?"
    raw = snap.get("encoder_position_raw")
    i2c = snap.get("encoder_i2c_reading")
    pos = snap.get("encoder_position")
    pos_s = f"{pos:.4f}" if isinstance(pos, float) and math.isfinite(pos) else str(pos)
    vbus = snap.get("bus_voltage")
    v_s = f"{vbus:.2f}V" if isinstance(vbus, float) and math.isfinite(vbus) else str(vbus)
    print(
        f"  {ch} ID {dev_id:>2} {joint_name(ch, dev_id):32} "
        f"mode={mode_name(snap.get('mode_raw')):<11} "
        f"err={err['hex'] if err else '??':<7} "
        f"fw={fw_s} Vbus={v_s:<8} upd={snap.get('update_counter')}  "
        f"raw={raw} i2c={i2c} nrot={snap.get('encoder_n_rotations')} "
        f"pos={pos_s} rad  cpr={snap.get('encoder_cpr')} gear={snap.get('gear_ratio')}"
    )


def cmd_full(args, outdir) -> int:
    """Non-interactive: inventory + snapshot + per-motor stream + round-robin."""
    report = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "phase": "encoder_full",
        "seconds_per_motor": args.seconds,
        "buses": {},
        "motors": [],
    }
    print("\n=== ARM ENCODER FULL SWEEP (can0 left, can1 right) ===")
    print("No torque/NMT/flash. Ping + SDO only. No motor is skipped.\n")

    for ch in ARM_CHANNELS:
        before = link_error_summary(link_stats(ch))
        print(f"{'=' * 78}\n{ch}  ({BUS_LIMB[ch]})  USB {before.get('state')} "
              f"{before.get('bitrate')} bit/s  oper={before.get('operstate')}\n{'=' * 78}")
        bus_rep = {"limb": BUS_LIMB[ch], "link_before": before, "scan": {}, "online": [],
                   "expected_repo": sorted(REPO_ARM_JOINTS[ch]),
                   "expected_asbuilt": sorted(AS_BUILT_ARM_JOINTS[ch])}
        with ArmBus(ch) as bus:
            print(f"  USB adapter: {bus.usb_port}")
            print(f"  scanning IDs {SCAN_RANGE.start}-{SCAN_RANGE.stop - 1} (ping + SDO DEVICE_ID)")
            found = []
            for dev_id in SCAN_RANGE:
                r = bus.probe(dev_id, attempts=3)
                bus_rep["scan"][str(dev_id)] = r
                mark = "ONLINE" if r["online"] else "----"
                if r["online"] or dev_id in target_ids(ch):
                    print(f"    ID {dev_id:>2}  {mark}  ping={r['ping_ok']}/{r['ping_attempts']}  "
                          f"sdo={r['sdo_reply_rate']}  self_id={r['self_reported_id']}  "
                          f"{joint_name(ch, dev_id)}")
                if r["online"]:
                    found.append(dev_id)
            # Always try as-built + repo IDs even if probe was flaky.
            sample_ids = sorted(set(found) | set(target_ids(ch)))
            bus_rep["online"] = found
            bus_rep["sampled_ids"] = sample_ids
            print(f"  repo expects {sorted(REPO_ARM_JOINTS[ch])}")
            print(f"  as-built     {sorted(AS_BUILT_ARM_JOINTS[ch])}")
            print(f"  online       {found}")
            print(f"  sampling     {sample_ids}")
            print("  missing repo vs online:", sorted(set(REPO_ARM_JOINTS[ch]) - set(found)) or "none")
            print("  unexpected vs as-built:", sorted(set(found) - set(AS_BUILT_ARM_JOINTS[ch])) or "none")

            print("\n  -- live snapshot --")
            snaps = {}
            for dev_id in sample_ids:
                snap = bus.snapshot_live(dev_id)
                snap["online"] = any(v is not None for v in snap.values())
                snaps[dev_id] = snap
                print_snap(ch, dev_id, snap)

            live_ids = [i for i in sample_ids if snaps[i].get("online")]
            missing_ids = [i for i in sample_ids if i not in live_ids]

            print(f"\n  -- fast ENCODER_POSITION_RAW stream ({args.seconds:.1f}s each) --")
            analyses = {}
            for dev_id in sample_ids:
                if dev_id not in live_ids:
                    analyses[dev_id] = {"samples": 0, "read_misses": 0, "verdict": "NO_DATA",
                                        "why": "no snapshot; not burning a stream timeout on a silent ID"}
                    write_csv(os.path.join(outdir, "encoder_csv", f"{ch}_id{dev_id}_stream.csv"), [])
                    print(f"    {ch} ID {dev_id:>2} {joint_name(ch, dev_id):32} skipped stream (silent)")
                    continue
                s = sample_stream(bus, dev_id, args.seconds)
                a = analyse(s["rows"], s["duration"], s["misses"])
                v, why = verdict_from(a, snaps[dev_id])
                a["verdict"] = v
                a["why"] = why
                analyses[dev_id] = a
                write_csv(os.path.join(outdir, "encoder_csv", f"{ch}_id{dev_id}_stream.csv"),
                          s["rows"])
                circ = a.get("circular") or {}
                print(
                    f"    {ch} ID {dev_id:>2} {joint_name(ch, dev_id):32} "
                    f"n={a.get('samples', 0):<5} {a.get('sample_rate_hz', 0):>6}Hz  "
                    f"raw={a.get('min')}..{a.get('max')}  p2p={a.get('p2p_counts')}  "
                    f"sd={a.get('stdev_unwrapped')}  jumps={a.get('jumps_gt_200')}  "
                    f"distinct={a.get('distinct_values')}  "
                    f"R={circ.get('R')}  i2c_oor={a.get('i2c_out_of_range')}  "
                    f"loop={a.get('loop_alive')} -> {v}"
                )
                print(f"         {why}")
                print(f"         first5={a.get('first5')} last5={a.get('last5')}")

            print(f"\n  -- round-robin live IDs on {ch} ({args.seconds:.1f}s) --")
            t0 = time.time()
            series = {i: [] for i in live_ids}
            while time.time() - t0 < args.seconds:
                t = time.time() - t0
                for i in live_ids:
                    vraw = bus.read_u16(i, Parameter.ENCODER_POSITION_RAW, flush=False)
                    if vraw is not None:
                        series[i].append((t, vraw))
            ranked = []
            for i in live_ids + missing_ids:
                vals = [v for _, v in series.get(i, [])]
                uw = unwrap(vals)
                travel = (max(uw) - min(uw)) if uw else 0
                ranked.append({
                    "id": i, "joint": joint_name(ch, i), "samples": len(vals),
                    "travel_counts": travel,
                    "travel_deg_rotor": round(travel * 360.0 / CPR, 2) if uw else 0,
                    "stdev": round(statistics.pstdev(uw), 3) if len(uw) > 1 else 0,
                    "min": min(vals) if vals else None,
                    "max": max(vals) if vals else None,
                })
            ranked.sort(key=lambda r: -(r["travel_counts"] or 0))
            print(f"    {'id':>3} {'joint':32} {'n':>6} {'min':>6} {'max':>6} "
                  f"{'travel':>8} {'deg':>8} {'stdev':>8}")
            for r in ranked:
                print(f"    {r['id']:>3} {r['joint']:32} {r['samples']:>6} "
                      f"{str(r['min']):>6} {str(r['max']):>6} {r['travel_counts']:>8} "
                      f"{r['travel_deg_rotor']:>8} {r['stdev']:>8}")

            print("\n  -- second snapshot (did 'constant' values move?) --")
            snap2 = {}
            for dev_id in sample_ids:
                snap2[dev_id] = bus.snapshot_live(dev_id)
                s1, s2 = snaps[dev_id], snap2[dev_id]
                print(
                    f"    ID {dev_id:>2} raw {s1.get('encoder_position_raw')} -> {s2.get('encoder_position_raw')}   "
                    f"i2c {s1.get('encoder_i2c_reading')} -> {s2.get('encoder_i2c_reading')}   "
                    f"upd {s1.get('update_counter')} -> {s2.get('update_counter')}   "
                    f"V {s1.get('bus_voltage')} -> {s2.get('bus_voltage')}"
                )

            after = link_error_summary(link_stats(ch))
            bus_rep["link_after"] = after
            bus_rep["traffic"] = {
                "tx": bus.tx_count, "rx": bus.rx_count,
                "timeouts": bus.timeout_count, "error_frames": len(bus.error_frames),
            }
            bus_rep["ranked_round_robin"] = ranked
            report["buses"][ch] = bus_rep
            for dev_id in sample_ids:
                report["motors"].append({
                    "bus": ch, "id": dev_id, "joint": joint_name(ch, dev_id),
                    "repo_joint": REPO_ARM_JOINTS[ch].get(dev_id),
                    "as_built": dev_id in AS_BUILT_ARM_JOINTS[ch],
                    "snapshot": snaps[dev_id],
                    "snapshot_after": snap2[dev_id],
                    "stream": analyses.get(dev_id),
                })
        print(f"  traffic tx={bus_rep['traffic']['tx']} rx={bus_rep['traffic']['rx']} "
              f"timeouts={bus_rep['traffic']['timeouts']} "
              f"err_frames={bus_rep['traffic']['error_frames']}")

    path = os.path.join(outdir, "encoder_full.json")
    with open(path, "w") as f:
        json.dump(jsonable(report), f, indent=2)
    print("\n=== SUMMARY ===")
    print(f"{'bus':<5} {'id':>3} {'joint':32} {'mode':<11} {'err':<8} {'Vbus':>6} "
          f"{'raw':>6} {'p2p':>6} {'loop':>5} {'verdict'}")
    for m in report["motors"]:
        s = m["snapshot"]
        a = m.get("stream") or {}
        err = decode_error(s.get("error_raw"))
        vbus = s.get("bus_voltage")
        v_s = f"{vbus:.1f}" if isinstance(vbus, float) and math.isfinite(vbus) else str(vbus)
        print(
            f"{m['bus']:<5} {m['id']:>3} {m['joint']:32} "
            f"{mode_name(s.get('mode_raw')):<11} "
            f"{(err['hex'] if err else '?'):<8} {v_s:>6} "
            f"{str(s.get('encoder_position_raw')):>6} {str(a.get('p2p_counts')):>6} "
            f"{str(a.get('loop_alive')):>5} {a.get('verdict')}"
        )
    print(f"\nSaved {path}")
    return 0


def cmd_liveness(args, outdir):
    """Back-compat name for a per-motor stationary stream (no prompts, no skips)."""
    results = []
    for ch in ARM_CHANNELS:
        with ArmBus(ch) as bus:
            ids = sorted(set(target_ids(ch)) | {
                i for i in SCAN_RANGE if bus.probe(i, attempts=1)["online"]
            })
            for dev_id in ids:
                snap = bus.snapshot_live(dev_id)
                s = sample_stream(bus, dev_id, args.seconds)
                a = analyse(s["rows"], s["duration"], s["misses"])
                v, why = verdict_from(a, snap)
                joint = joint_name(ch, dev_id)
                results.append({"bus": ch, "id": dev_id, "joint": joint,
                                "snapshot": snap, "analysis": a, "verdict": v, "why": why})
                write_csv(os.path.join(outdir, "encoder_csv", f"{ch}_id{dev_id}_stationary.csv"),
                          s["rows"])
                err = decode_error(snap.get("error_raw"))
                print(f"{ch} ID {dev_id:>2} {joint:32} mode={mode_name(snap.get('mode_raw')):<9} "
                      f"err={err['hex'] if err else '??':<7} n={a.get('samples', 0):<5} "
                      f"rate={a.get('sample_rate_hz', 0):>5}Hz min={a.get('min')} max={a.get('max')} "
                      f"p2p={a.get('p2p_counts')} -> {v}")
                print(f"        {why}")
    with open(os.path.join(outdir, "phase4_stationary.json"), "w") as f:
        json.dump(jsonable(results), f, indent=2)
    print(f"\nSaved: {os.path.join(outdir, 'phase4_stationary.json')}")
    return 0


def cmd_rotate(args, outdir):
    ch, dev_id = args.bus, args.id
    joint = joint_name(ch, dev_id)
    with ArmBus(ch) as bus:
        snap = bus.snapshot_live(dev_id)
        print(f"\n{ch} ID {dev_id} -- {joint}")
        print_snap(ch, dev_id, snap)
        if not args.no_prompt:
            input(f"  >>> Rotate ONLY {joint} by hand, then press Enter to START: ")
        s = sample_stream(bus, dev_id, args.seconds, extra_every=20)
        print("  ... sampling done.")
    a = analyse(s["rows"], s["duration"], s["misses"])
    raw = [r["position_raw"] for r in s["rows"]]
    uw = unwrap(raw)
    if uw:
        a["travel_counts"] = max(uw) - min(uw)
        a["travel_deg_rotor"] = round(a["travel_counts"] * 360.0 / CPR, 2)
        a["start"] = uw[0]
        a["end"] = uw[-1]
        a["return_error_counts"] = abs(uw[-1] - uw[0])
        deltas = [b - a_ for a_, b in zip(uw, uw[1:])]
        a["n_positive_steps"] = sum(1 for d in deltas if d > 0)
        a["n_negative_steps"] = sum(1 for d in deltas if d < 0)
        a["direction_reversed"] = a["n_positive_steps"] > 0 and a["n_negative_steps"] > 0
    write_csv(os.path.join(outdir, "encoder_csv", f"{ch}_id{dev_id}_rotate.csv"), s["rows"])
    print(json.dumps(jsonable(a), indent=2))
    with open(os.path.join(outdir, f"phase4_rotate_{ch}_id{dev_id}.json"), "w") as f:
        json.dump(jsonable({"bus": ch, "id": dev_id, "joint": joint,
                            "snapshot": snap, "analysis": a}), f, indent=2)
    return 0


def cmd_watch_arm(args, outdir):
    ch = args.bus
    with ArmBus(ch) as bus:
        ids = sorted({i for i in SCAN_RANGE if bus.probe(i, attempts=1)["online"]}
                     | set(target_ids(ch)))
        print(f"\n{ch} ({BUS_LIMB[ch]}) -- watching encoders on IDs {ids}")
        if not args.no_prompt:
            input("  >>> Rotate ONE joint by hand. Press Enter to START: ")
        t0 = time.time()
        series = {i: [] for i in ids}
        while time.time() - t0 < args.seconds:
            for i in ids:
                v = bus.read_u16(i, Parameter.ENCODER_POSITION_RAW, flush=False)
                if v is not None:
                    series[i].append((time.time() - t0, v))
        print("  ... sampling done.\n")
    summary = []
    for i in ids:
        vals = [v for _, v in series[i]]
        uw = unwrap(vals)
        travel = max(uw) - min(uw) if uw else 0
        summary.append({"id": i, "joint": joint_name(ch, i), "samples": len(vals),
                        "travel_counts": travel,
                        "travel_deg_rotor": round(travel * 360.0 / CPR, 2) if uw else 0,
                        "stdev": round(statistics.pstdev(uw), 2) if len(uw) > 1 else 0,
                        "min": min(vals) if vals else None,
                        "max": max(vals) if vals else None})
    summary.sort(key=lambda r: -r["travel_counts"])
    print(f"{'id':>3} {'joint':32} {'samples':>8} {'min':>6} {'max':>6} "
          f"{'travel(cnt)':>12} {'travel(deg)':>12} {'stdev':>8}")
    for r in summary:
        print(f"{r['id']:>3} {r['joint']:32} {r['samples']:>8} "
              f"{str(r['min']):>6} {str(r['max']):>6} "
              f"{r['travel_counts']:>12} {r['travel_deg_rotor']:>12} {r['stdev']:>8}")
    if summary:
        print(f"\n  Largest movement: ID {summary[0]['id']} ({summary[0]['joint']})")
    with open(os.path.join(outdir, f"phase4_watcharm_{ch}.json"), "w") as f:
        json.dump(jsonable({"bus": ch, "ranked": summary,
                            "series": {str(k): v for k, v in series.items()}}), f, indent=2)
    return 0


def cmd_magnet_track(args, outdir):
    ch, dev_id = args.bus, args.id
    joint = joint_name(ch, dev_id)
    positions = []
    with ArmBus(ch) as bus:
        snap = bus.snapshot_live(dev_id)
        print(f"\n{ch} ID {dev_id} -- {joint}")
        print_snap(ch, dev_id, snap)
        for k in range(args.steps):
            if not args.no_prompt:
                input(f"  >>> Step {k+1}/{args.steps}: set {joint} to a DIFFERENT hand position,\n"
                      f"      hold still, then press Enter to sample {args.seconds:.0f}s: ")
            s = sample_stream(bus, dev_id, args.seconds, extra_every=20)
            raw = [r["position_raw"] for r in s["rows"]]
            circ = circular_stats(raw)
            positions.append({"step": k + 1, "n": len(raw), **circ})
            print(f"      centre={circ['centre_counts']} counts ({circ['centre_deg']} deg)  "
                  f"R={circ['R']}  MAD={circ['mad']}  n={len(raw)}")
            write_csv(os.path.join(outdir, "encoder_csv",
                                   f"{ch}_id{dev_id}_magnettrack_{k+1}.csv"), s["rows"])
    centres = [p["centre_counts"] for p in positions if p.get("centre_counts") is not None]
    spread = (max(centres) - min(centres)) if centres else 0
    print(f"  centres: {centres}  spread={spread:.1f} counts")
    verdict = ("TRACKS" if spread > 100 else "DOES NOT TRACK")
    print(f"  VERDICT: {verdict}")
    with open(os.path.join(outdir, f"phase5_magnettrack_{ch}_id{dev_id}.json"), "w") as f:
        json.dump(jsonable({"bus": ch, "id": dev_id, "joint": joint, "snapshot": snap,
                            "positions": positions, "spread_counts": spread,
                            "verdict": verdict}), f, indent=2)
    return 0


def cmd_live(args, outdir):
    """Poll one actuator until Encoder_init finishes or --seconds elapses.

    ID 5 is typically wedged in Encoder_init()'s I2C retry loop. That loop does
    not need a power cycle: the next successful AS5600 ACK lets init continue
    into IDLE with live Vbus and a real encoder reading.
    """
    ch, dev_id = args.bus, args.id
    joint = joint_name(ch, dev_id)
    path = os.path.join(outdir, f"live_{ch}_id{dev_id}.csv")
    print(f"\nLIVE {ch} ID {dev_id} ({joint}) for {args.seconds:.0f}s")
    print("  Success looks like: mode IDLE, Vbus ~23 V, encoder raw not stuck at 0.")
    print("  Encoder_init retries I2C every ~100 ms -- reseat the AS5600 cable NOW.\n")
    t0 = time.time()
    last_key = None
    rows = []
    alive_at = None
    with ArmBus(ch) as bus:
        while time.time() - t0 < args.seconds:
            snap = bus.snapshot_live(dev_id, flush=True)
            t = time.time() - t0
            mode = mode_name(snap.get("mode_raw"))
            vbus = snap.get("bus_voltage")
            raw = snap.get("encoder_position_raw")
            i2c = snap.get("encoder_i2c_reading")
            upd = snap.get("update_counter")
            gear = snap.get("gear_ratio")
            key = (mode, None if vbus is None else round(vbus, 1), raw, i2c, upd)
            line = (f"  t={t:6.1f}s  mode={mode:<11} Vbus={vbus}  "
                    f"raw={raw} i2c={i2c} upd={upd} gear={gear}")
            if key != last_key:
                print(line, flush=True)
                last_key = key
            v_ok = isinstance(vbus, float) and vbus > 10.0
            enc_ok = isinstance(raw, int) and raw != 0
            if alive_at is None and mode in ("IDLE", "DAMPING") and v_ok and enc_ok:
                alive_at = t
                print(f"  *** BOOT COMPLETE at t={t:.1f}s -- encoder init unblocked ***",
                      flush=True)
            rows.append({"t": t, "mode": mode, "vbus": vbus, "raw": raw,
                         "i2c": i2c, "upd": upd, "gear": gear})
            time.sleep(0.15)
    write_csv(path, [{"t": r["t"], "position_raw": r["raw"], "i2c_reading": r["i2c"],
                      "n_rotations": None, "encoder_position": None,
                      "encoder_velocity": None, "update_counter": r["upd"],
                      "vbus": r["vbus"], "error": None, "mode": r["mode"]}
                     for r in rows])
    print(f"\n  boot_complete={alive_at is not None}  at={alive_at}  log={path}")
    with open(os.path.join(outdir, f"live_{ch}_id{dev_id}.json"), "w") as f:
        json.dump(jsonable({"bus": ch, "id": dev_id, "joint": joint,
                            "boot_at_s": alive_at, "last": rows[-1] if rows else None,
                            "n": len(rows)}), f, indent=2)
    return 0 if alive_at is not None else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="full",
                    choices=["full", "liveness", "stationary", "rotate",
                             "watch-arm", "magnet-track", "stream", "live"])
    ap.add_argument("--bus", choices=ARM_CHANNELS)
    ap.add_argument("--id", type=int)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--no-prompt", action="store_true",
                    help="skip interactive Enter waits (rotate/watch/magnet-track)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "encoder_csv"), exist_ok=True)

    if args.mode in ("full", "stream"):
        return cmd_full(args, args.outdir)
    if args.mode in ("liveness", "stationary"):
        return cmd_liveness(args, args.outdir)
    if args.mode == "rotate":
        if not args.bus or args.id is None:
            ap.error("--bus and --id are required for rotate")
        return cmd_rotate(args, args.outdir)
    if args.mode == "magnet-track":
        if not args.bus or args.id is None:
            ap.error("--bus and --id are required for magnet-track")
        return cmd_magnet_track(args, args.outdir)
    if args.mode == "watch-arm":
        if not args.bus:
            ap.error("--bus is required for watch-arm")
        return cmd_watch_arm(args, args.outdir)
    if args.mode == "live":
        if not args.bus or args.id is None:
            ap.error("--bus and --id are required for live")
        return cmd_live(args, args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
