"""Fast read-only CAN topology scan: ping IDs 1-20 on each bus and read the
few parameters that identify the actuator class. Sends no mode, setpoint,
heartbeat or flash traffic.

Run from the repository root:  .venv/bin/python scripts/hardware/scan_topology.py
"""
import argparse
import datetime
import importlib.util
import json
import math
from pathlib import Path
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
LOW = ROOT / "source/berkeley_humanoid_lite_lowlevel"
spec = importlib.util.spec_from_file_location("scan_recoil", LOW / "berkeley_humanoid_lite_lowlevel/recoil/core.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)

# Parameters that identify which physical actuator is on the other end.
IDENTITY = [
    ("DEVICE_ID", "<I"),
    ("FIRMWARE_VERSION", "<I"),
    ("MODE", "<I"),
    ("ERROR", "<I"),
    ("MOTOR_TORQUE_CONSTANT", "<f"),
    ("MOTOR_POLE_PAIRS", "<I"),
    ("MOTOR_PHASE_ORDER", "<i"),
    ("MOTOR_MAX_CALIBRATION_CURRENT", "<f"),
    ("CURRENT_CONTROLLER_I_KP", "<f"),
    ("CURRENT_CONTROLLER_I_KI", "<f"),
    ("POSITION_CONTROLLER_GEAR_RATIO", "<f"),
    ("ENCODER_FLUX_OFFSET", "<f"),
    ("ENCODER_CPR", "<I"),
    ("POSITION_CONTROLLER_POSITION_MEASURED", "<f"),
]


class ReadOnlyBus(r.Bus):
    """Refuses to transmit anything except an SDO read or the ping probe."""

    def transmit(self, frame):
        ok = (frame.func_id == r.Function.RECEIVE_SDO and frame.size == 3 and frame.data[0] == 0x40) or (
            frame.func_id == r.Function.RECEIVE_PDO_1 and frame.data == b"\xca"
        )
        if not ok:
            raise RuntimeError("Non-read operation prohibited")
        super().transmit(frame)

    def receive(self, filter_device_id=None, filter_function=None, timeout=0.15):
        # Unlike the stock receive, unrelated traffic cannot reset the deadline.
        deadline = time.monotonic() + (0.15 if timeout is None else timeout)
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


def status(bus):
    out = subprocess.check_output(["ip", "-j", "-details", "-statistics", "link", "show", "dev", bus], text=True)
    return json.loads(out)[0]


def healthy(s):
    info = s["linkinfo"]["info_data"]
    return info.get("state") == "ERROR-ACTIVE" and info.get("bittiming", {}).get("bitrate") == 1000000


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buses", nargs="+", default=["can0", "can1", "can2", "can3"])
    parser.add_argument("--max-id", type=int, default=20)
    args = parser.parse_args()

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = ROOT / "logs/hardware_audit" / stamp
    out.mkdir(parents=True)
    report = {"created_utc": stamp, "controller_writes": [], "buses": {}}

    for channel in args.buses:
        before = status(channel)
        rec = report["buses"][channel] = {"before_state": before["linkinfo"]["info_data"]["state"], "motors": {}}
        if not healthy(before):
            rec["blocked"] = "interface not ERROR-ACTIVE at 1 Mbps"
            print(f"{channel}: {rec['blocked']}", flush=True)
            continue
        bus = ReadOnlyBus(channel)
        try:
            online = []
            for device in range(1, args.max_id + 1):
                if bus.ping(device, timeout=0.15):
                    online.append(device)
            print(f"{channel}: online -> {online}", flush=True)
            for device in online:
                entry = {}
                for name, fmt in IDENTITY:
                    raw = bus._read_parameter_bytes(device, getattr(r.Parameter, name), timeout=0.15)
                    if raw is None:
                        entry[name] = None
                        continue
                    entry[name] = struct.unpack(fmt, raw)[0]
                    time.sleep(0.003)
                rec["motors"][str(device)] = entry
                kt = entry.get("MOTOR_TORQUE_CONSTANT")
                cal = entry.get("MOTOR_MAX_CALIBRATION_CURRENT")
                pos = entry.get("POSITION_CONTROLLER_POSITION_MEASURED")
                cls = "UNCONFIGURED" if not kt else ("A/hip-knee" if abs(kt - 0.08958) < 1e-4 else "B/ankle-distal")
                print(
                    f"  {channel} id={device:<3} Kt={kt} calI={cal} class={cls} "
                    f"flux={entry.get('ENCODER_FLUX_OFFSET')} pos={pos} "
                    f"mode={entry.get('MODE')} err={entry.get('ERROR')}",
                    flush=True,
                )
        except Exception as exc:
            rec["blocked"] = str(exc)
            print(f"{channel} STOP: {exc}", flush=True)
        finally:
            bus.stop()
            rec["after_state"] = status(channel)["linkinfo"]["info_data"]["state"]

    (out / "topology.json").write_text(json.dumps(clean(report), indent=2, allow_nan=False) + "\n")
    print(f"\nWrote {out / 'topology.json'}", flush=True)


if __name__ == "__main__":
    main()
