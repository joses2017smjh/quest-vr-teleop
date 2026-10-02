"""PHASE 1 -- ARM-ONLY CAN and ID inventory.

Discovery uses both PDO-1 ping and an SDO read of DEVICE_ID, then a live
encoder snapshot for every ID that answers. IDs 1-16 are scanned so leftover
foot-board IDs are visible. Nothing is written.

Usage:  python tools/scan_arm_buses.py [--outdir DIR]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arm_common import (  # noqa: E402
    ARM_CHANNELS, AS_BUILT_ARM_JOINTS, BUS_LIMB, LEG_IDS, REPO_ARM_JOINTS,
    SCAN_RANGE, ArmBus, decode_error, joint_name, jsonable,
    link_error_summary, link_stats, mode_name,
)


def classify(channel: str, dev_id: int) -> str:
    if dev_id in AS_BUILT_ARM_JOINTS[channel]:
        note = f"as-built arm joint: {AS_BUILT_ARM_JOINTS[channel][dev_id]}"
        if dev_id not in REPO_ARM_JOINTS[channel]:
            repo = REPO_ARM_JOINTS[channel]
            note += f"  (docs/repo map this limb as {sorted(repo)}; ID {dev_id} is as-built, not docs)"
        return note
    if dev_id in REPO_ARM_JOINTS[channel]:
        return f"docs arm joint: {REPO_ARM_JOINTS[channel][dev_id]} -- NOT in as-built set"
    for limb, ids in LEG_IDS.items():
        if dev_id in ids:
            return f"UNEXPECTED -- matches {limb} ID {dev_id} ({ids[dev_id]}); possible ex-foot board"
    return "UNEXPECTED -- not in any repository joint map"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir or os.path.join("arm_validation", stamp)
    os.makedirs(outdir, exist_ok=True)

    report = {"created": stamp, "phase": "1_can_and_id_inventory", "buses": {}}

    for ch in ARM_CHANNELS:
        print(f"\n{'=' * 72}\n{ch}  ({BUS_LIMB[ch]})\n{'=' * 72}")
        before = link_stats(ch)
        b4 = link_error_summary(before)
        print(f"  link before : state={b4['state']} bitrate={b4['bitrate']} "
              f"oper={b4.get('operstate')} bus_off={b4['bus_off']} "
              f"error_passive={b4['error_passive']} bus_error={b4['bus_error']} "
              f"rx_err={b4['rx_errors']} tx_err={b4['tx_errors']}")

        found, results = {}, {}
        usb_port = None
        with ArmBus(ch) as bus:
            usb_port = bus.usb_port
            print(f"  USB adapter : {usb_port}")
            print(f"  scanning IDs {SCAN_RANGE.start}-{SCAN_RANGE.stop - 1} "
                  f"by ping + SDO DEVICE_ID ...")
            for dev_id in SCAN_RANGE:
                r = bus.probe(dev_id, attempts=3)
                results[dev_id] = r
                if not r["online"]:
                    continue
                st = bus.read_status(dev_id)
                snap = bus.snapshot_live(dev_id, flush=True)
                r["status"] = st
                r["snapshot"] = snap
                found[dev_id] = r
                note = classify(ch, dev_id)
                warn = "" if r["id_consistent"] else \
                    f"  !! self-reports ID {r['self_reported_id']} !!"
                err = st["error"]
                print(f"    ID {dev_id:>2}  ONLINE  ping={r['ping_ok']}/{r['ping_attempts']}  "
                      f"sdo={r['sdo_reply_rate']}  mode={st['mode']:<11} "
                      f"error={err['hex'] if err else '??'}"
                      f"{'[' + ','.join(err['bits']) + ']' if err and err['bits'] else ''}"
                      f"{warn}")
                print(f"           {note}")
                print(f"           raw={snap.get('encoder_position_raw')}  "
                      f"i2c={snap.get('encoder_i2c_reading')}  "
                      f"nrot={snap.get('encoder_n_rotations')}  "
                      f"pos={snap.get('encoder_position')}  "
                      f"vel={snap.get('encoder_velocity')}  "
                      f"Vbus={snap.get('bus_voltage')}  "
                      f"upd={snap.get('update_counter')}  "
                      f"cpr={snap.get('encoder_cpr')}")
            tx, rx, to = bus.tx_count, bus.rx_count, bus.timeout_count
            efr = len(bus.error_frames)

        after = link_stats(ch)
        a4 = link_error_summary(after)
        print(f"  link after  : state={a4['state']} bus_off={a4['bus_off']} "
              f"error_passive={a4['error_passive']} bus_error={a4['bus_error']} "
              f"rx_err={a4['rx_errors']} tx_err={a4['tx_errors']}")
        delta = {k: (a4[k] or 0) - (b4[k] or 0) for k in
                 ("restarts", "bus_error", "arbitration_lost", "error_warning",
                  "error_passive", "bus_off", "rx_errors", "tx_errors")}
        print(f"  error delta : {delta}")
        print(f"  traffic     : tx={tx} rx={rx} timeouts={to} error_frames={efr}")

        expected_repo = set(REPO_ARM_JOINTS[ch])
        expected_hw = set(AS_BUILT_ARM_JOINTS[ch])
        online = set(found)
        print(f"\n  docs/repo IDs     : {sorted(expected_repo)}")
        print(f"  as-built IDs      : {sorted(expected_hw)}")
        print(f"  actually online   : {sorted(online)}")
        print(f"  MISSING vs as-built: {sorted(expected_hw - online) or 'none'}")
        print(f"  MISSING vs docs    : {sorted(expected_repo - online) or 'none'}")
        print(f"  UNEXPECTED         : {sorted(online - expected_hw) or 'none'}")

        report["buses"][ch] = {
            "limb": BUS_LIMB[ch], "usb_port": usb_port,
            "link_before": b4, "link_after": a4, "error_delta": delta,
            "traffic": {"tx": tx, "rx": rx, "timeouts": to, "error_frames": efr},
            "scan": {str(k): v for k, v in results.items()},
            "online": sorted(online),
            "expected_repo": sorted(expected_repo),
            "expected_asbuilt": sorted(expected_hw),
            "missing_asbuilt": sorted(expected_hw - online),
            "missing_repo": sorted(expected_repo - online),
            "unexpected": sorted(online - expected_hw),
        }

    path = os.path.join(outdir, "phase1_inventory.json")
    with open(path, "w") as f:
        json.dump(jsonable(report), f, indent=2)
    print(f"\nSaved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
