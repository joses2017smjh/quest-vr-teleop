"""One leg, read-only: which IDs answer, firmware, mode and errors, bus voltage,
stored gear ratio and calibration, and a stationary encoder sample per motor.

Leg adapters move between hub ports, so the leg is recognised by the IDs that
answer (odd = left, even = right), never by the canN name or USB port.

Only pings and SDO reads go out (arm_common.ArmBus): no mode change, heartbeat,
setpoint or flash write. Keep the leg still while it samples: any motion shows
up as encoder noise.

  .venv/bin/python tools/diagnose_leg.py --bus can2
  .venv/bin/python tools/diagnose_leg.py --bus can2 --seconds 5
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys

import can

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arm_common import (  # noqa: E402
    LEG_IDS, ArmBus, Parameter, decode_error, jsonable, link_error_summary, link_stats, mode_name,
)
from validate_arm_encoders import analyse, sample_stream, verdict_from  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIRMWARE = 0x20250226
# Calibrating against an encoder that never moved stores about +/-14*pi.
FROZEN_FLUX = 14 * math.pi


class LegBus(ArmBus):
    """ArmBus transport (ping + SDO read only) without the arm-channel guard."""

    def __post_init__(self):
        self.usb_port = os.path.basename(os.path.realpath(f"/sys/class/net/{self.channel}/device"))
        self._bus = can.interface.Bus(interface="socketcan", channel=self.channel, bitrate=self.bitrate)


def which_leg(online: list[int]) -> str | None:
    best = max(LEG_IDS, key=lambda leg: len(set(online) & set(LEG_IDS[leg])))
    return best if set(online) & set(LEG_IDS[best]) else None


def calibration(flux: float | None) -> str:
    if flux is None or not math.isfinite(flux):
        return "UNREADABLE"
    if flux == 0.0:
        return "NEVER CALIBRATED (flux 0)"
    if abs(abs(flux) - FROZEN_FLUX) < 1.0:
        return f"INVALID: flux {flux:+.2f} = calibrated while the encoder was frozen"
    return f"stored flux {flux:+.2f}"


def fmt(value, spec: str = ".2f") -> str:
    return format(value, spec) if isinstance(value, float) and math.isfinite(value) else str(value)


def print_motor(dev: int, joint: str, snap: dict, a: dict, verdict: str, why: str) -> None:
    err = decode_error(snap.get("error_raw"))
    fw = snap.get("firmware_version")
    fw_s = f"0x{fw:08X}" if isinstance(fw, int) else "?"
    if isinstance(fw, int) and fw != FIRMWARE:
        fw_s += f" (EXPECTED 0x{FIRMWARE:08X})"
    print(f"\nID {dev:>2}  {joint}")
    print(f"  mode {mode_name(snap.get('mode_raw'))}, error {err['hex'] if err else '??'}"
          f"{' ' + ','.join(err['bits']) if err and err['bits'] else ''}, firmware {fw_s}")
    print(f"  Vbus {fmt(snap.get('bus_voltage'))} V, gear {fmt(snap.get('gear_ratio'), 'g')}, "
          f"Kt {fmt(snap.get('torque_constant'), '.5f')}, pole pairs {snap.get('pole_pairs')}, "
          f"phase order {snap.get('phase_order')}")
    print(f"  calibration: {calibration(snap.get('encoder_flux_offset'))}")
    print(f"  encoder: {verdict} - {why}")
    if a.get("samples", 0) >= 2:
        print(f"    raw {a['min']}..{a['max']} over {a['samples']} samples, "
              f"i2c out of range {a['i2c_out_of_range']}, read misses {a['read_misses']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bus", required=True, help="CAN interface the leg is on, e.g. can2")
    ap.add_argument("--seconds", type=float, default=3.0, help="stationary encoder sample per motor")
    args = ap.parse_args()

    if not os.path.exists(f"/sys/class/net/{args.bus}"):
        raise SystemExit(f"{args.bus} does not exist")
    link_before = link_error_summary(link_stats(args.bus))
    print(f"{args.bus}: {link_before['state']} at {link_before['bitrate']} bit/s")
    if link_before["state"] != "ERROR-ACTIVE":
        print("  warning: the adapter is not ERROR-ACTIVE, so replies may be missing")

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {"created_utc": stamp, "bus": args.bus, "controller_writes": [], "link_before": link_before,
              "motors": {}}
    with LegBus(args.bus) as bus:
        report["usb_port"] = bus.usb_port
        try:
            probes = {dev: bus.probe(dev, attempts=2) for dev in range(1, 17)}
        except can.CanError as exc:
            raise SystemExit(f"{args.bus} cannot transmit ({exc}). Reset it:\n"
                             f"  sudo ip link set {args.bus} down && "
                             f"sudo ip link set {args.bus} up type can bitrate 1000000") from exc
        online = [dev for dev, p in probes.items() if p["online"]]
        leg = which_leg(online)
        joints = LEG_IDS.get(leg, {})
        missing = sorted(set(joints) - set(online))
        report.update(online=online, leg=leg, missing=missing)
        print(f"online: {online} -> {leg or 'no leg IDs'} (USB {bus.usb_port})")
        for dev in missing:
            print(f"  missing: ID {dev} {joints[dev]}")

        for dev in online:
            snap = bus.snapshot_live(dev)
            snap["torque_constant"] = bus.read_f32(dev, Parameter.MOTOR_TORQUE_CONSTANT)
            snap["phase_order"] = bus.read_i32(dev, Parameter.MOTOR_PHASE_ORDER)
            stream = sample_stream(bus, dev, args.seconds)
            a = analyse(stream["rows"], stream["duration"], stream["misses"])
            verdict, why = verdict_from(a, snap)
            joint = joints.get(dev, "UNMAPPED")
            report["motors"][dev] = {"joint": joint, "probe": probes[dev], "snapshot": snap, "analysis": a,
                                     "calibration": calibration(snap.get("encoder_flux_offset")),
                                     "verdict": verdict, "why": why, "samples": stream["rows"]}
            print_motor(dev, joint, snap, a, verdict, why)
        report["error_frames_seen"] = len(bus.error_frames)

    report["link_after"] = link_error_summary(link_stats(args.bus))
    out = os.path.join(REPO, "logs/hardware_audit", f"{stamp}_leg_diagnostic.json")
    with open(out, "w") as f:
        json.dump(jsonable(report), f, indent=2)
    print(f"\n{args.bus} after: {report['link_after']['state']}, "
          f"{report['error_frames_seen']} error frames during the run")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
