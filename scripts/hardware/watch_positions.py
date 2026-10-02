"""Read-only live position monitor used to physically identify which controller
drives which joint.

You back-drive one joint by hand; the motor whose reading moves is that joint.
The script only issues SDO reads, so the motors stay unpowered and limp: it never
sends a mode change, a heartbeat, a setpoint or a flash write.

    .venv/bin/python scripts/hardware/watch_positions.py --bus can1
    .venv/bin/python scripts/hardware/watch_positions.py --bus can1 --ids 4 6 8
"""
import argparse
import datetime
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
LOW = ROOT / "source/berkeley_humanoid_lite_lowlevel"
spec = importlib.util.spec_from_file_location("watch_recoil", LOW / "berkeley_humanoid_lite_lowlevel/recoil/core.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)

# Readings are motor-side radians while gear_ratio is unconfigured at 1.0, and an
# unpowered limb rattles, so a fixed threshold gives false positives. A motor counts
# as moved only if it travels well past the quiet-baseline band measured at startup.
BASELINE_SECONDS = 4.0
MOVE_MARGIN_RAD = 0.15
MOVE_MULTIPLE = 4.0


class ReadOnlyBus(r.Bus):
    """Refuses to transmit anything except an SDO read or the ping probe."""

    def transmit(self, frame):
        ok = (frame.func_id == r.Function.RECEIVE_SDO and frame.size == 3 and frame.data[0] == 0x40) or (
            frame.func_id == r.Function.RECEIVE_PDO_1 and frame.data == b"\xca"
        )
        if not ok:
            raise RuntimeError("Non-read operation prohibited")
        super().transmit(frame)

    def receive(self, filter_device_id=None, filter_function=None, timeout=0.1):
        deadline = time.monotonic() + (0.1 if timeout is None else timeout)
        while time.monotonic() < deadline:
            msg = self._Bus__bus.recv(timeout=max(0, deadline - time.monotonic()))
            if msg is None:
                return None
            if msg.is_error_frame:
                raise RuntimeError("CAN error frame: stop this bus")
            if msg.is_extended_id or msg.is_remote_frame:
                continue
            if msg.arbitration_id != ((filter_function << 7) | filter_device_id):
                continue
            return r.CANFrame(filter_device_id, filter_function, len(msg.data), msg.data)
        return None


def read_position(bus, device):
    raw = bus._read_parameter_bytes(device, r.Parameter.POSITION_CONTROLLER_POSITION_MEASURED, timeout=0.1)
    return None if raw is None else struct.unpack("<f", raw)[0]


def can_state(channel):
    out = subprocess.check_output(["ip", "-j", "-details", "link", "show", "dev", channel], text=True)
    return json.loads(out)[0]["linkinfo"]["info_data"]["state"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bus", required=True)
    parser.add_argument("--ids", nargs="+", type=int, default=None, help="default: whatever responds to a ping")
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument("--duration", type=float, default=0.0, help="seconds; 0 means run until Ctrl+C")
    args = parser.parse_args()

    if can_state(args.bus) != "ERROR-ACTIVE":
        raise SystemExit(f"{args.bus} is not ERROR-ACTIVE; fix the bus before probing it")

    bus = ReadOnlyBus(args.bus)
    try:
        ids = args.ids
        if ids is None:
            ids = [d for d in range(1, 21) if bus.ping(d, timeout=0.15)]
            print(f"{args.bus}: responding IDs {ids}")
        if not ids:
            raise SystemExit(f"No motors responded on {args.bus}")

        period = 1.0 / args.rate
        print(f"\nMeasuring quiet baseline for {BASELINE_SECONDS:.0f}s -- do NOT touch the robot.")
        base_lo, base_hi = {}, {}
        t_base = time.monotonic() + BASELINE_SECONDS
        while time.monotonic() < t_base:
            for d in ids:
                pos = read_position(bus, d)
                if pos is None:
                    continue
                base_lo[d] = min(base_lo.get(d, pos), pos)
                base_hi[d] = max(base_hi.get(d, pos), pos)
            time.sleep(period)

        noise = {d: base_hi.get(d, 0.0) - base_lo.get(d, 0.0) for d in ids}
        threshold = {d: max(MOVE_MARGIN_RAD, MOVE_MULTIPLE * noise[d]) for d in ids}
        print("Baseline rattle (rad): " + "  ".join(f"{d}={noise[d]:.3f}" for d in ids))
        print("Move threshold  (rad): " + "  ".join(f"{d}={threshold[d]:.3f}" for d in ids))

        low = {d: base_lo.get(d) for d in ids}
        high = {d: base_hi.get(d) for d in ids}
        print("\nNow back-drive ONE joint by hand through a large range.")
        print("The motor marked * is that joint. Ctrl+C when done.\n")
        print("    " + "".join(f"{('id ' + str(d)):>14}" for d in ids))

        t_end = time.monotonic() + args.duration if args.duration > 0 else None
        while t_end is None or time.monotonic() < t_end:
            row = []
            for d in ids:
                pos = read_position(bus, d)
                if pos is None:
                    row.append(f"{'TIMEOUT':>14}")
                    continue
                low[d] = pos if low[d] is None else min(low[d], pos)
                high[d] = pos if high[d] is None else max(high[d], pos)
                mark = "*" if (high[d] - low[d]) > threshold[d] else " "
                row.append(f"{mark}{pos:+8.4f}{'':>5}")
            print("    " + "".join(row), end="\r", flush=True)
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n\nTravel observed per motor (motor-side rad; gear_ratio is 1.0 so this is NOT joint angle):")
        for d in ids:
            if low.get(d) is None or high.get(d) is None:
                print(f"  id {d:<3} no readings")
                continue
            span = high[d] - low[d]
            verdict = "  <-- MOVED" if span > threshold.get(d, MOVE_MARGIN_RAD) else ""
            print(
                f"  id {d:<3} min={low[d]:+9.4f}  max={high[d]:+9.4f}  span={span:8.4f}"
                f"  (baseline {noise.get(d, 0.0):.3f}){verdict}"
            )
        bus.stop()
        print(f"\n{args.bus} state after: {can_state(args.bus)}")
        print(f"Finished {datetime.datetime.now(datetime.timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()
