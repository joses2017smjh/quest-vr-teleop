"""Read-only dump of each controller's whole MotorController struct over CAN.

Besides live state (phase currents, ADC offsets, encoder buffer, velocity), the struct holds the
128-entry table the firmware stored at the last calibration: the averaged encoder error over one
motor revolution. Its shape shows what the encoder did while calibration turned the motor:
  flat ripple near 0              -> encoder tracked the rotation
  steady ramp of -2*pi*pp/128     -> encoder was frozen
  ramp of twice that slope        -> encoder counted against the motor direction
  anything else                   -> encoder was jumping
Uses the audit script's guarded bus (pings and SDO reads to expected IDs only). Offsets come from
the firmware's DWARF info (sizeof(MotorController) = 0x340).

    .venv/bin/python scripts/hardware/dump_controller_memory.py
    .venv/bin/python scripts/hardware/dump_controller_memory.py --limb "right leg" --tolerate-can-errors --pings 20
"""
import argparse
import datetime
import importlib.util
import json
import math
from pathlib import Path
import statistics
import struct
import time

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("audit", ROOT / "scripts/hardware/stationary_encoder_audit.py")
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)

STRUCT_SIZE = 0x340
POSITION_CONTROLLER = ("update_counter", "gear_ratio", "position_kp", "position_ki", "velocity_kp", "velocity_ki",
                       "torque_limit", "velocity_limit", "position_limit_lower", "position_limit_upper", "position_offset",
                       "torque_target", "torque_measured", "torque_setpoint", "velocity_target", "velocity_measured",
                       "velocity_setpoint", "position_target", "position_measured", "position_setpoint",
                       "position_integrator", "velocity_integrator", "torque_filter_alpha")
CURRENT_CONTROLLER = ("i_limit", "i_kp", "i_ki", "i_a_measured", "i_b_measured", "i_c_measured", "v_a_setpoint",
                      "v_b_setpoint", "v_c_setpoint", "i_alpha_measured", "i_beta_measured", "v_alpha_setpoint",
                      "v_beta_setpoint", "v_q_target", "v_d_target", "v_q_setpoint", "v_d_setpoint", "i_q_target",
                      "i_d_target", "i_q_measured", "i_d_measured", "i_q_setpoint", "i_d_setpoint", "i_q_integrator",
                      "i_d_integrator")
WORDS = {
    0x000: ("device_id", "<I"), 0x004: ("firmware_version", "<I"), 0x008: ("watchdog_timeout", "<I"),
    0x00C: ("fast_frame_frequency", "<I"), 0x010: ("mode", "<B3x"), 0x014: ("error", "<I"),
    0x0D8: ("ps.htim", "<I"), 0x0DC: ("ps.hadc1", "<I"), 0x0E0: ("ps.hadc2", "<I"),
    0x0E4: ("ps.adc_raw[0..1]", "<HH"), 0x0E8: ("ps.adc_raw[2]", "<H2x"),
    0x0EC: ("ps.adc_offset[0..1]", "<HH"), 0x0F0: ("ps.adc_offset[2]", "<H2x"),
    0x0F4: ("ps.undervoltage_threshold", "<f"), 0x0F8: ("ps.overvoltage_threshold", "<f"),
    0x0FC: ("ps.bus_voltage_filter_alpha", "<f"), 0x100: ("ps.bus_voltage_measured", "<f"),
    0x104: ("motor.pole_pairs", "<i"), 0x108: ("motor.torque_constant", "<f"), 0x10C: ("motor.phase_order", "<b3x"),
    0x110: ("motor.max_calibration_current", "<f"),
    0x114: ("enc.hi2c", "<I"), 0x118: ("enc.i2c_buffer", ">H2x"), 0x11C: ("enc.unused", "<I"), 0x120: ("enc.cpr", "<i"),
    0x124: ("enc.position_offset", "<f"), 0x128: ("enc.velocity_filter_alpha", "<f"), 0x12C: ("enc.position_raw", "<H2x"),
    0x130: ("enc.n_rotations", "<i"), 0x134: ("enc.position", "<f"), 0x138: ("enc.velocity", "<f"),
    0x13C: ("enc.flux_offset", "<f"),
}
WORDS.update({0x018 + 4 * i: (f"pc.{n}", "<I" if n == "update_counter" else "<f") for i, n in enumerate(POSITION_CONTROLLER)})
WORDS.update({0x074 + 4 * i: (f"cc.{n}", "<f") for i, n in enumerate(CURRENT_CONTROLLER)})
WORDS.update({0x140 + 4 * i: (f"enc.lut[{i}]", "<f") for i in range(128)})
assert sorted(WORDS) == list(range(0, STRUCT_SIZE, 4))


class TolerantBus(a.ReadOnlyBus):
    """Same transmit guard, but logs CAN error frames instead of stopping."""

    def _error(self, msg):
        try:
            super()._error(msg)
        except a.StopAudit:
            pass


def lut_verdict(lut, pole_pairs):
    if not pole_pairs or any(v is None or not math.isfinite(v) for v in lut):
        return {"verdict": "no calibration table"}
    if max(lut) - min(lut) == 0:
        # Encoder_init zeroes the table at every boot and loadConfig does not restore it.
        return {"verdict": "table empty (not kept across power-up)"}
    steps = [lut[(i + 1) % 128] - lut[i] for i in range(128)]
    median = statistics.median(steps)
    frozen = -2 * math.pi * pole_pairs / 128
    share = lambda target, tol: sum(abs(step - target) < tol for step in steps) / len(steps)
    # Most steps must follow one pattern; the moving average smears ~9 entries around the table's wrap.
    if max(lut) - min(lut) < 3 and share(0, 0.1) >= 0.6:
        verdict = "encoder tracked the rotation"
    elif share(frozen, 0.1) >= 0.6:
        verdict = "encoder frozen during calibration"
    elif share(2 * frozen, 0.15) >= 0.6:
        verdict = "encoder counted against the motor"
    else:
        verdict = "encoder jumping during calibration"
    return {"verdict": verdict, "median_step": round(median, 3), "spread": round(max(lut) - min(lut), 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limb", action="append", help="limb(s) to dump; default: every limb except the right leg")
    parser.add_argument("--tolerate-can-errors", action="store_true", help="log CAN error frames instead of stopping")
    parser.add_argument("--pings", type=int, default=0, help="ping each ID this many times first and report the success rate")
    args = parser.parse_args()

    limbs = args.limb or ["left arm", "right arm", "left leg"]
    buses = {ch: v for ch, v in a.resolve_buses().items() if v[1] in limbs}
    bus_class = TolerantBus if args.tolerate_can_errors else a.ReadOnlyBus
    opened = {ch: bus_class(ch, ids) for ch, (_, _, ids) in buses.items()}
    out = ROOT / "logs/hardware_audit" / (datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_memory_dump.json")
    record = {"limbs": limbs, "controller_writes": [], "pings": {}, "controllers": {}, "error_frames": {}, "stop_reason": None}
    try:
        for ch, (port, limb, ids) in buses.items():
            bus = opened[ch]
            if args.pings:
                for device in ids:
                    before = len(bus.error_frames)
                    ok = sum(a.ping(bus, device) for _ in range(args.pings))
                    record["pings"][f"{ch}:{device}"] = {"ok": ok, "sent": args.pings, "error_frames": len(bus.error_frames) - before}
                    print(f"{ch}:{device:<3} {limb:<9} pings {ok}/{args.pings}  error frames {len(bus.error_frames) - before}")
            for device in ids:
                key = f"{ch}:{device}"
                raw, words = {}, {}
                for offset in range(0, STRUCT_SIZE, 4):
                    frame = None
                    for _ in range(3):
                        bus.drain()
                        frame = bus._read_parameter(device, offset, timeout=a.TIMEOUT)
                        if frame is not None and frame.size == 4:
                            break
                    name, fmt = WORDS[offset]
                    if frame is None or frame.size != 4:
                        raw[offset], words[name] = None, None
                        continue
                    raw[offset] = frame.data.hex()
                    value = struct.unpack(fmt, frame.data)
                    words[name] = value[0] if len(value) == 1 else list(value)
                lut = [words[f"enc.lut[{i}]"] for i in range(128)]
                analysis = lut_verdict(lut, words.get("motor.pole_pairs"))
                missing = sum(v is None for v in raw.values())
                record["controllers"][key] = {"limb": limb, "joint": a.JOINTS[limb].get(device, "NOT IN DOCS"),
                                              "usb_port": port, "missing_words": missing, "lut_analysis": analysis,
                                              "words": words, "raw_hex": {hex(k): v for k, v in raw.items()}}
                w = words
                fmt_f = lambda v, d=2: "-" if v is None else f"{v:.{d}f}"
                print(f"{key:<8} {limb:<9} {a.JOINTS[limb].get(device, '?'):<14} mode={w['mode']} err={w['error']} "
                      f"V={fmt_f(w['ps.bus_voltage_measured'])} "
                      f"I_abc=({fmt_f(w['cc.i_a_measured'])},{fmt_f(w['cc.i_b_measured'])},{fmt_f(w['cc.i_c_measured'])})A "
                      f"adc_off={w['ps.adc_offset[0..1]']}+{w['ps.adc_offset[2]']} i2c=0x{(w['enc.i2c_buffer'] or 0):04x} "
                      f"n_rot={w['enc.n_rotations']} vel={fmt_f(w['enc.velocity'], 1)} flux={fmt_f(w['enc.flux_offset'])} "
                      f"| LUT: {analysis['verdict']} {analysis.get('median_step', '')} spread={analysis.get('spread', '')}"
                      f"{f'  ({missing} words unread)' if missing else ''}")
    except a.StopAudit as e:
        record["stop_reason"] = str(e)
        print("STOPPED:", e)
    finally:
        for ch, bus in opened.items():
            record["error_frames"][ch] = bus.error_frames
            bus.stop()
        record["links_after"] = {ch: a.link_status(ch) for ch in buses}
        out.write_text(json.dumps(record, indent=1))
        print("error frames per bus:", {ch: len(v) for ch, v in record["error_frames"].items()})
        print("links after:", {ch: (s["state"], {k: v for k, v in s["counters"].items() if v}) for ch, s in record["links_after"].items()})
        print("saved", out)


if __name__ == "__main__":
    main()
