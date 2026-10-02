"""Powered encoder test: turn one motor open-loop by a known angle and watch whether its encoder follows.

MOVES THE MOTOR. With the controller in MODE_VALPHABETA_OVERRIDE the firmware applies the voltage vector
we write (v_alpha, v_beta) without using the encoder, so the rotor follows the commanded electrical angle
even when the encoder is broken. The test ramps the vector at angle 0 until the phase current reaches
--current (default 1 A), sweeps --cycles electrical cycles forward and back (default 2 cycles = 2/14 motor
turn, about 3.4 deg at a 15:1 joint), reads the raw encoder at every step, then returns to IDLE.

Guards:
- transmit whitelist: pings, SDO reads, SDO writes to v_alpha/v_beta only (|V| <= --max-volts),
  NMT to IDLE or VALPHABETA_OVERRIDE, and heartbeats; one target device only
- aborts if any phase current exceeds --abort-current, and always writes 0 V and IDLE on exit
- the firmware watchdog drops to DAMPING (0 V) 1 s after heartbeats stop

    .venv/bin/python scripts/hardware/encoder_rotation_test.py --limb "left arm" --id 9 --dry-run
    .venv/bin/python scripts/hardware/encoder_rotation_test.py --limb "left arm" --id 9
"""
import argparse
import datetime
import importlib.util
import json
import math
from pathlib import Path
import struct
import time

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("audit", ROOT / "scripts/hardware/stationary_encoder_audit.py")
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
r = a.r

# Offsets from the firmware DWARF info (normal 0x20250226 layout).
V_ALPHA_SETPOINT = 0x0A0
V_BETA_SETPOINT = 0x0A4
I_ABC = (0x080, 0x084, 0x088)
MODE_VALPHABETA_OVERRIDE = 0x21


class MotionTestBus(a.ReadOnlyBus):
    def __init__(self, channel, device, max_volts):
        super().__init__(channel, [device])
        self.device = device
        self.max_volts = max_volts

    def transmit(self, frame):
        if frame.device_id != self.device:
            raise RuntimeError(f"Prohibited target {frame.device_id}")
        if frame.func_id == r.Function.RECEIVE_SDO and frame.size == 8 and frame.data[0] == 0x20:
            param = struct.unpack("<H", frame.data[1:3])[0]
            value = struct.unpack("<f", frame.data[4:8])[0]
            if param not in (V_ALPHA_SETPOINT, V_BETA_SETPOINT) or not abs(value) <= self.max_volts:
                raise RuntimeError(f"Prohibited write {param:#x}={value}")
        elif frame.func_id == r.Function.NMT:
            if not (frame.size == 2 and frame.data[0] in (r.Mode.IDLE, MODE_VALPHABETA_OVERRIDE) and frame.data[1] == self.device):
                raise RuntimeError(f"Prohibited NMT {frame.data.hex()}")
        elif frame.func_id == r.Function.HEARTBEAT and frame.size == 0:
            pass
        else:
            return super().transmit(frame)
        r.Bus.transmit(self, frame)


def read(bus, param, fmt="<f"):
    value, _ = a.read(bus, bus.device, param, fmt)
    return value


def write_vector(bus, v_alpha, v_beta):
    bus._write_parameter(bus.device, V_ALPHA_SETPOINT, struct.pack("<f", v_alpha))
    bus._write_parameter(bus.device, V_BETA_SETPOINT, struct.pack("<f", v_beta))
    bus.transmit(r.CANFrame(bus.device, r.Function.HEARTBEAT))


def peak_current(bus):
    i_abc = [read(bus, p) for p in I_ABC]
    if any(c is None or not math.isfinite(c) for c in i_abc):
        raise RuntimeError(f"phase current unreadable: {i_abc}")
    return i_abc, max(abs(c) for c in i_abc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limb", required=True, choices=list(a.DOC_CHANNELS))
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--current", type=float, default=1.0, help="peak phase current to hold (A)")
    parser.add_argument("--abort-current", type=float, default=4.0)
    parser.add_argument("--max-volts", type=float, default=1.0)
    parser.add_argument("--no-current-volts", type=float, default=0.3,
                        help="abort if peak current is still <0.1 A at this voltage")
    parser.add_argument("--cycles", type=float, default=2.0, help="electrical cycles to sweep each way")
    parser.add_argument("--steps-per-cycle", type=int, default=48)
    parser.add_argument("--dry-run", action="store_true", help="check preconditions with reads only")
    args = parser.parse_args()

    channel = next(ch for ch, v in a.resolve_buses().items() if v[1] == args.limb)
    if args.id not in dict((v[1], v[2]) for v in a.resolve_buses().values())[args.limb]:
        raise SystemExit(f"ID {args.id} is not on the {args.limb}")
    joint = a.JOINTS[args.limb].get(args.id, "NOT IN DOCS")
    bus = MotionTestBus(channel, args.id, args.max_volts)
    record = {"limb": args.limb, "id": args.id, "joint": joint, "channel": channel, "args": vars(args), "ramp": [], "sweep": [], "events": []}
    out = ROOT / "logs/hardware_audit" / (datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_rotation_{args.limb.replace(' ', '_')}_{args.id}.json")
    engaged = False
    try:
        if not a.ping(bus, args.id):
            raise SystemExit("target does not answer")
        fw, mode, volts, pole_pairs, cpr = (read(bus, 0x004, "<I"), read(bus, 0x010, "<B3x"), read(bus, 0x100),
                                            read(bus, 0x104, "<i"), read(bus, 0x120, "<i"))
        record["pre"] = {"firmware": fw, "mode": mode, "bus_voltage": volts, "pole_pairs": pole_pairs, "cpr": cpr}
        print(f"{channel}:{args.id} {args.limb} {joint}: fw={fw:#x} mode={mode} V={volts:.2f} pole_pairs={pole_pairs} cpr={cpr}")
        if fw != 0x20250226 or mode != r.Mode.IDLE or not volts or volts < 15 or pole_pairs != 14 or cpr != 4096:
            raise SystemExit("preconditions not met (need normal firmware layout, IDLE, bus > 15 V, 14 pole pairs, cpr 4096)")
        i_idle, _ = peak_current(bus)
        print(f"idle currents {['%.2f' % c for c in i_idle]} A, raw encoder {read(bus, 0x12C, '<H2x')}")
        if args.dry_run:
            print("dry run: preconditions OK, nothing written")
            return

        write_vector(bus, 0.0, 0.0)
        bus.transmit(r.CANFrame(args.id, r.Function.NMT, size=2, data=bytes([MODE_VALPHABETA_OVERRIDE, args.id])))
        engaged = True
        time.sleep(0.02)
        if read(bus, 0x010, "<B3x") != MODE_VALPHABETA_OVERRIDE:
            raise RuntimeError("controller refused override mode")

        volts_cmd = 0.05
        while True:
            write_vector(bus, volts_cmd, 0.0)
            time.sleep(0.05)
            i_abc, peak = peak_current(bus)
            record["ramp"].append({"v": volts_cmd, "i_abc": i_abc})
            if volts_cmd >= args.no_current_volts and peak < 0.1:
                raise RuntimeError(f"no current response at {volts_cmd:.2f} V; current sensing or phase wiring suspect")
            if peak > args.abort_current:
                raise RuntimeError(f"phase current {peak:.2f} A over abort limit during ramp")
            if peak >= args.current:
                break
            if volts_cmd + 0.02 > args.max_volts:
                record["events"].append(f"reached {args.max_volts} V with only {peak:.2f} A")
                break
            volts_cmd = round(volts_cmd + 0.02, 3)
        print(f"holding {volts_cmd:.2f} V -> peak {peak:.2f} A; sweeping {args.cycles} electrical cycles each way")
        time.sleep(0.3)

        steps = int(args.cycles * args.steps_per_cycle)
        path = [2 * math.pi * k / args.steps_per_cycle for k in range(steps + 1)]
        path += path[-2::-1]
        for n, theta in enumerate(path):
            write_vector(bus, volts_cmd * math.cos(theta), volts_cmd * math.sin(theta))
            time.sleep(0.012)
            sample = {"theta": theta, "raw": read(bus, 0x12C, "<H2x"), "t": time.time()}
            if n % 12 == 0:
                sample["i_abc"], peak = peak_current(bus)
                if peak > args.abort_current:
                    raise RuntimeError(f"phase current {peak:.2f} A over abort limit during sweep")
            record["sweep"].append(sample)
    except (RuntimeError, a.StopAudit) as e:
        record["events"].append(f"ABORT: {e}")
        print("ABORT:", e)
    finally:
        if engaged:
            for _ in range(3):
                try:
                    write_vector(bus, 0.0, 0.0)
                    bus.transmit(r.CANFrame(args.id, r.Function.NMT, size=2, data=bytes([r.Mode.IDLE, args.id])))
                    time.sleep(0.05)
                    if read(bus, 0x010, "<B3x") == r.Mode.IDLE:
                        break
                except Exception as e:  # keep trying to release the motor
                    record["events"].append(f"release retry: {e}")
            record["post"] = {"mode": read(bus, 0x010, "<B3x"), "error": read(bus, 0x014, "<I"), "i_abc": [read(bus, q) for q in I_ABC]}
            print(f"released: {record['post']}")
        bus.stop()
        if record["sweep"]:
            raws = [s["raw"] for s in record["sweep"] if s["raw"] is not None]
            unwrapped = [raws[0]]
            for prev, cur in zip(raws, raws[1:]):
                unwrapped.append(unwrapped[-1] + ((cur - prev + 2048) % 4096) - 2048)
            half = len(record["sweep"]) // 2
            expected = 4096 * args.cycles / 14
            forward = unwrapped[min(half, len(unwrapped) - 1)] - unwrapped[0]
            record["result"] = {"expected_counts": round(expected), "measured_forward_counts": forward,
                                "returned_to_start_counts": unwrapped[-1] - unwrapped[0],
                                "distinct_values": len(set(raws))}
            ratio = abs(forward) / expected
            verdict = ("TRACKS the rotation" if 0.75 <= ratio <= 1.25 else
                       "FROZEN (no response to rotation)" if abs(forward) < 20 and len(set(raws)) < 5 else
                       "ERRATIC (does not follow the rotation)")
            record["result"]["verdict"] = verdict
            print(f"encoder: expected ±{expected:.0f} counts forward, measured {forward:+d}, back at start {unwrapped[-1] - unwrapped[0]:+d}, "
                  f"{len(set(raws))} distinct values -> {verdict}")
            # A working encoder's change should track the commanded motor angle (sign depends on direction).
            print(f"{'motor angle cmd':>16} {'expected change':>16} {'encoder raw':>12} {'encoder change':>15}")
            every = max(1, len(raws) // 24)
            for k in range(0, len(raws), every):
                motor_deg = math.degrees(record["sweep"][k]["theta"]) / 14
                print(f"{motor_deg:>15.1f}° {motor_deg / 360 * 4096:>+16.0f} {raws[k]:>12} {unwrapped[k] - unwrapped[0]:>+15}")
        out.write_text(json.dumps(record, indent=1))
        print("saved", out)


if __name__ == "__main__":
    main()
