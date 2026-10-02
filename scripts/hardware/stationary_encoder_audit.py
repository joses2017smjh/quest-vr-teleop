"""Read-only stationary encoder audit across all four CAN buses.

With every limb supported and still, sample the raw encoder count of all 22 motors
round-robin, plus each controller's error flags, mode, and measured bus voltage.
This shows whether position noise like can0:7's is isolated, shared by one bus, or
system-wide, and whether any controller sees a sagging battery.

Buses are resolved by USB port, not canN name: the names reorder across reboots.
The script only pings and issues SDO reads, one outstanding request at a time. It
never sends a mode change, heartbeat, setpoint, or flash write, and it stops on any
CAN error, a controller dropping out, or a controller found in an active mode.

    .venv/bin/python scripts/hardware/stationary_encoder_audit.py
    .venv/bin/python scripts/hardware/stationary_encoder_audit.py --duration 30 --rate 10
"""
import argparse
import datetime
import importlib.util
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
LOW = ROOT / "source/berkeley_humanoid_lite_lowlevel"
spec = importlib.util.spec_from_file_location("audit_recoil", LOW / "berkeley_humanoid_lite_lowlevel/recoil/core.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)

# Limb per USB port. IDs verified by ping on 2026-09-12 (logs/hardware_audit/20260912T212240.968358Z);
# names follow the official joint ID mapping (robot's own left/right), which puts odd leg IDs on the left leg.
LIMBS = {
    "3-4.1:1.0": ("left arm", (1, 3, 5, 7, 9)),
    "3-4.2:1.0": ("right arm", (2, 4, 6, 8, 12)),
    "3-4.4:1.0": ("left leg", (1, 3, 5, 7, 11, 13)),
    "3-4.3:1.0": ("right leg", (2, 4, 6, 8, 12, 14)),
}
# Interface names the docs and scripts/hardware/start_can_by_port.sh use.
DOC_CHANNELS = {"left arm": "can0", "right arm": "can1", "left leg": "can2", "right leg": "can3"}
# berkeley-humanoid-lite.gitbook.io/docs/in-depth-contents/joint-id-mapping
JOINTS = {
    "left arm": {1: "shoulder_pitch", 3: "shoulder_roll", 5: "shoulder_yaw", 7: "elbow_pitch", 9: "elbow_yaw"},
    "right arm": {2: "shoulder_pitch", 4: "shoulder_roll", 6: "shoulder_yaw", 8: "elbow_pitch", 10: "elbow_yaw", 12: "wrist_yaw (as-built)"},
    "left leg": {1: "hip_roll", 3: "hip_yaw", 5: "hip_pitch", 7: "knee_pitch", 11: "ankle_pitch", 13: "ankle_roll"},
    "right leg": {2: "hip_roll", 4: "hip_yaw", 6: "hip_pitch", 8: "knee_pitch", 12: "ankle_pitch", 14: "ankle_roll"},
}

# The library's Parameter offsets were checked against the Recoil firmware's DWARF info.
CPR = 4096
DEGREES_PER_COUNT = 360.0 / CPR
SAFE_MODES = (r.Mode.DISABLED, r.Mode.IDLE, r.Mode.DAMPING)
TIMEOUT = 0.05
MAX_CONSECUTIVE_TIMEOUTS = 5
SLOW_READ_SECONDS = 2.0

# Classification thresholds in encoder counts (1 count = 0.088 deg motor-side).
STEADY_RANGE = 2
JITTER_RANGE = 8
JUMP_STEP = 8


# SocketCAN error classes (linux/can/error.h), for logging what an error frame reported.
ERROR_CLASSES = {
    0x001: "tx_timeout", 0x002: "lost_arbitration", 0x004: "controller", 0x008: "protocol",
    0x010: "transceiver", 0x020: "no_ack", 0x040: "bus_off", 0x080: "bus_error", 0x100: "restarted", 0x200: "counters",
}


class StopAudit(Exception):
    pass


class ReadOnlyBus(r.Bus):
    """Refuses to transmit anything except an SDO read or the ping probe to this bus's expected IDs."""

    def __init__(self, channel, allowed_ids):
        super().__init__(channel)
        self.allowed_ids = frozenset(allowed_ids)
        self.unsolicited = []
        self.error_frames = []

    def transmit(self, frame):
        ok = (frame.func_id == r.Function.RECEIVE_SDO and frame.size == 3 and frame.data[0] == 0x40) or (
            frame.func_id == r.Function.RECEIVE_PDO_1 and frame.data == b"\xca"
        )
        if not ok or frame.device_id == 0 or frame.device_id not in self.allowed_ids:
            raise RuntimeError(f"Prohibited frame on {self.channel}: func={frame.func_id} id={frame.device_id}")
        super().transmit(frame)

    def _record(self, msg):
        self.unsolicited.append({"t": time.time(), "id": msg.arbitration_id, "data": bytes(msg.data).hex()})

    def _error(self, msg):
        data = bytes(msg.data)
        self.error_frames.append({
            "t": time.time(),
            "classes": [name for bit, name in ERROR_CLASSES.items() if msg.arbitration_id & bit],
            "id": hex(msg.arbitration_id),
            "data": data.hex(),
            "controller_status": hex(data[1]) if len(data) > 1 else None,
            "protocol_type": hex(data[2]) if len(data) > 2 else None,
            "tx_rx_error_counters": (data[6], data[7]) if len(data) > 7 else None,
        })
        print(f"CAN error frame on {self.channel} (continuing): {self.error_frames[-1]}")

    def drain(self):
        """Log anything already pending so a late reply cannot be mistaken for the next answer."""
        while (msg := self._Bus__bus.recv(timeout=0)) is not None:
            if msg.is_error_frame:
                self._error(msg)
            self._record(msg)

    def receive(self, filter_device_id=None, filter_function=None, timeout=TIMEOUT):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            msg = self._Bus__bus.recv(timeout=max(0, deadline - time.monotonic()))
            if msg is None:
                return None
            if msg.is_error_frame:
                self._error(msg)
            if msg.is_extended_id or msg.is_remote_frame or msg.arbitration_id != ((filter_function << 7) | filter_device_id):
                self._record(msg)
                continue
            return r.CANFrame(filter_device_id, filter_function, len(msg.data), bytes(msg.data))
        return None


def resolve_buses(allow_missing=True):
    found = {}
    for channel in sorted(os.listdir("/sys/class/net")):
        if channel.startswith("can"):
            port = os.path.basename(os.path.realpath(f"/sys/class/net/{channel}/device"))
            if port in LIMBS:
                found[port] = channel
    missing = sorted(set(LIMBS) - set(found))
    if missing:
        msg = f"No CAN interface on USB port(s) {missing}"
        if not allow_missing:
            raise SystemExit(msg + "; do not guess, re-check the adapters")
        print(msg + " -- continuing with the adapters that are present")
    return {found[port]: (port, *LIMBS[port]) for port in LIMBS if port in found}


def link_status(channel):
    s = json.loads(subprocess.check_output(["ip", "-j", "-details", "-statistics", "link", "show", "dev", channel], text=True))[0]
    info = s["linkinfo"]["info_data"]
    counters = {f"{d}_{k}": s["stats64"][d][k] for d in ("rx", "tx") for k in ("errors", "dropped")}
    counters.update(s["linkinfo"].get("info_xstats", {}))
    return {"operstate": s["operstate"], "state": info.get("state"), "bitrate": info.get("bittiming", {}).get("bitrate"), "counters": counters}


def check_links(buses, before):
    now = {channel: link_status(channel) for channel in buses}
    for channel, s in now.items():
        if s["state"] != "ERROR-ACTIVE":
            raise StopAudit(f"{channel} left ERROR-ACTIVE: {s['state']}")
        grew = {k: v - before[channel]["counters"].get(k, 0) for k, v in s["counters"].items() if v != before[channel]["counters"].get(k, 0)}
        if grew:
            print(f"NOTE: {channel} error counters changed: {grew}")
    return now


def ping(bus, device):
    # The library's ping indexes an 8-byte unpack and would crash on a short reply.
    bus.drain()
    bus.transmit(r.CANFrame(device, r.Function.RECEIVE_PDO_1, size=1, data=b"\xca"))
    frame = bus.receive(device, r.Function.TRANSMIT_PDO_1, timeout=TIMEOUT)
    return frame is not None and frame.size >= 1 and frame.data[0] == 0xCA


def read(bus, device, parameter, fmt):
    bus.drain()
    frame = bus._read_parameter(device, parameter, timeout=TIMEOUT)
    if frame is None:
        return None, None
    if frame.size != 4:
        bus.unsolicited.append({"t": time.time(), "note": f"malformed SDO reply from {device}", "data": frame.data.hex()})
        return None, frame.data.hex()
    return struct.unpack(fmt, frame.data)[0], frame.data.hex()


def read_status(bus, device):
    error, _ = read(bus, device, r.Parameter.ERROR, "<I")
    mode, _ = read(bus, device, r.Parameter.MODE, "<B3x")
    voltage, _ = read(bus, device, r.Parameter.POWERSTAGE_BUS_VOLTAGE_MEASURED, "<f")
    if mode is not None and mode not in SAFE_MODES:
        print(f"NOTE: {bus.channel}:{device} is in mode {mode:#x} (not disabled/idle/damping); still sampling")
    return {"t": time.time(), "error": error, "mode": mode, "bus_voltage": voltage}


def analyze(samples):
    raws = [s["raw"] for s in samples if s["raw"] is not None]
    result = {"samples": len(raws), "timeouts": sum(1 for s in samples if s["raw"] is None)}
    if len(raws) < 2:
        return result | {"zeros": raws.count(0), "class": "NO DATA"}
    steps = [((b - a + CPR // 2) % CPR) - CPR // 2 for a, b in zip(raws, raws[1:])]
    # A zero only counts when it arrives as a jump; a motor resting at the 4095/0 seam reads 0 legitimately.
    result["zeros"] = sum(1 for raw, step in zip(raws[1:], steps) if raw == 0 and abs(step) > JUMP_STEP)
    unwrapped = [0]
    for step in steps:
        unwrapped.append(unwrapped[-1] + step)
    span = max(unwrapped) - min(unwrapped)
    max_step = max(abs(step) for step in steps)
    if span <= STEADY_RANGE:
        verdict = "steady"
    elif span <= JITTER_RANGE:
        verdict = "jitter"
    elif max_step <= JUMP_STEP:
        verdict = "drift"
    else:
        verdict = "UNSTABLE"
    return result | {
        "first_raw": raws[0],
        "range_counts": span,
        "range_deg": round(span * DEGREES_PER_COUNT, 2),
        "max_step_counts": max_step,
        "jumps": sum(1 for step in steps if abs(step) > JUMP_STEP),
        "net_drift_counts": unwrapped[-1],
        "stdev_counts": round(statistics.pstdev(unwrapped), 2),
        "distinct_values": len(set(raws)),
        "class": verdict,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=30.0, help="seconds of sampling")
    parser.add_argument("--rate", type=float, default=10.0, help="target rounds per second across all motors")
    parser.add_argument("--skip-limb", action="append", default=[], choices=[limb for limb, _ in LIMBS.values()],
                        help="leave this limb's bus completely untouched (repeatable)")
    parser.add_argument("--arm-only", action="store_true",
                        help="only open left/right arm adapters; never touch legs")
    parser.add_argument("--require-all-buses", action="store_true",
                        help="abort if any of the four USB-CAN adapters is missing")
    args = parser.parse_args()
    if args.arm_only:
        for limb in ("left leg", "right leg"):
            if limb not in args.skip_limb:
                args.skip_limb.append(limb)

    out = ROOT / "logs/hardware_audit" / datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out.mkdir(parents=True)
    buses = {channel: v for channel, v in resolve_buses(allow_missing=not args.require_all_buses).items()
             if v[1] not in args.skip_limb}
    if not buses:
        raise SystemExit("No matching CAN adapters found")
    record = {
        "created_utc": out.name,
        "user_confirmed_stationary_and_supported": True,
        "controller_writes": [],
        "buses": {channel: {"usb_port": port, "limb": limb, "ids": list(ids)} for channel, (port, limb, ids) in buses.items()},
        "fingerprint": {},
        "metadata": {},
        "samples": {},
        "status_reads": {},
        "unsolicited": {},
        "skipped_limbs": args.skip_limb,
        "error_frames": {},
        "stop_reason": None,
    }

    def save():
        (out / "stationary_encoder_audit.json").write_text(json.dumps(record, indent=1))

    print("Bus resolution (by USB port):")
    for channel, (port, limb, ids) in buses.items():
        note = "" if channel == DOC_CHANNELS[limb] else f"  (docs name it {DOC_CHANNELS[limb]}; run start_can_by_port.sh)"
        print(f"  {channel} -> USB {port} = {limb:<9} IDs {list(ids)}{note}")

    before = {channel: link_status(channel) for channel in buses}
    record["links_before"] = before
    for channel, s in before.items():
        if s["operstate"] != "UP" or s["state"] != "ERROR-ACTIVE" or s["bitrate"] != 1000000:
            record["stop_reason"] = f"{channel} not healthy before start: {s}"
            save()
            raise SystemExit(record["stop_reason"])

    opened = {channel: ReadOnlyBus(channel, ids) for channel, (_, _, ids) in buses.items()}
    addresses = [(channel, device) for channel, (_, _, ids) in buses.items() for device in ids]
    try:
        # Passive listen first: anything arriving now was not requested by us.
        t_listen = time.monotonic() + 1.0
        while time.monotonic() < t_listen:
            for bus in opened.values():
                bus.drain()
            time.sleep(0.05)
        passive = {channel: len(bus.unsolicited) for channel, bus in opened.items()}
        print(f"Passive 1 s listen, unsolicited frames per bus: {passive}")

        # Fingerprint: every expected ID must answer on the bus its USB port says it belongs to.
        for channel, device in addresses:
            answered = any(ping(opened[channel], device) for _ in range(3))
            record["fingerprint"][f"{channel}:{device}"] = answered
        silent = [key for key, ok in record["fingerprint"].items() if not ok]
        if silent:
            print(f"WARNING: expected IDs did not answer pings (still sampling the rest): {silent}")
        else:
            print(f"Fingerprint: all {len(addresses)} expected IDs answered on their resolved buses")
        addresses = [(ch, dev) for ch, dev in addresses if record["fingerprint"].get(f"{ch}:{dev}")]
        if not addresses:
            raise StopAudit("No IDs answered ping on the resolved arm buses")

        for channel, device in addresses:
            bus = opened[channel]
            meta = {}
            for name, parameter, fmt in (
                ("device_id", r.Parameter.DEVICE_ID, "<I"),
                ("firmware_version", r.Parameter.FIRMWARE_VERSION, "<I"),
                ("cpr", r.Parameter.ENCODER_CPR, "<i"),
            ):
                meta[name], _ = read(bus, device, parameter, fmt)
            meta.update(read_status(bus, device))
            record["metadata"][f"{channel}:{device}"] = meta
            if meta["device_id"] != device or meta["cpr"] != CPR:
                print(f"WARNING: {channel}:{device} metadata mismatch {meta}; still sampling")
            record["samples"][f"{channel}:{device}"] = []
            record["status_reads"][f"{channel}:{device}"] = [meta | {}]
        save()

        print(f"Sampling {len(addresses)} motors for {args.duration:.0f} s -- keep everything still and supported.")
        period = 1.0 / args.rate
        slow_every = max(1, round(SLOW_READ_SECONDS * args.rate))
        consecutive = {key: 0 for key in record["samples"]}
        t_start = time.monotonic()
        round_index = 0
        while time.monotonic() - t_start < args.duration:
            t_round = time.monotonic()
            for channel, device in addresses:
                key = f"{channel}:{device}"
                bus = opened[channel]
                raw, payload = read(bus, device, r.Parameter.ENCODER_POSITION_RAW, "<H2x")
                record["samples"][key].append({"round": round_index, "t": time.time(), "raw": raw, "payload": payload})
                consecutive[key] = 0 if raw is not None else consecutive[key] + 1
                if consecutive[key] >= MAX_CONSECUTIVE_TIMEOUTS:
                    print(f"WARNING: {key} stopped answering ({MAX_CONSECUTIVE_TIMEOUTS} timeouts in a row); continuing")
                    consecutive[key] = 0
                if raw is not None and raw >= CPR:
                    print(f"WARNING: {key} reported out-of-range raw count {raw}")
                if round_index % slow_every == slow_every - 1:
                    record["status_reads"][key].append(read_status(bus, device))
            if round_index % slow_every == slow_every - 1:
                check_links(buses, before)
                print(f"  t={time.monotonic() - t_start:5.1f} s  round {round_index + 1}")
            round_index += 1
            time.sleep(max(0.0, period - (time.monotonic() - t_round)))
    except StopAudit as e:
        record["stop_reason"] = str(e)
        print(f"\nSTOPPED: {e}")
    except KeyboardInterrupt:
        record["stop_reason"] = "interrupted by user"
    except Exception as e:
        record["stop_reason"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        for channel, bus in opened.items():
            record["unsolicited"][channel] = bus.unsolicited
            record["error_frames"][channel] = bus.error_frames
            bus.stop()
        record["links_after"] = {channel: link_status(channel) for channel in buses}
        record["error_delta"] = {
            channel: {k: v - before[channel]["counters"].get(k, 0) for k, v in s["counters"].items()}
            for channel, s in record["links_after"].items()
        }

        summary = {}
        for key, samples in record["samples"].items():
            statuses = [s for s in record["status_reads"][key] if s.get("bus_voltage") is not None]
            volts = [s["bus_voltage"] for s in statuses]
            summary[key] = analyze(samples) | {
                "voltage_min": round(min(volts), 2) if volts else None,
                "voltage_max": round(max(volts), 2) if volts else None,
                "errors_seen": sorted({s["error"] for s in record["status_reads"][key] if s.get("error") is not None}),
                "modes_seen": sorted({s["mode"] for s in record["status_reads"][key] if s.get("mode") is not None}),
            }

        # Same-round jumps on several motors would point at a shared cause rather than one sensor.
        jumps_by_round = {}
        for key, samples in record["samples"].items():
            valid = [s for s in samples if s["raw"] is not None]
            for a, b in zip(valid, valid[1:]):
                if abs(((b["raw"] - a["raw"] + CPR // 2) % CPR) - CPR // 2) > JUMP_STEP:
                    jumps_by_round.setdefault(b["round"], []).append(key)
        shared = {k: v for k, v in sorted(jumps_by_round.items()) if len(v) > 1}
        record["summary"] = summary
        record["shared_jump_rounds"] = shared
        save()

        print(f"\n{'bus:id':<9} {'limb':<9} {'joint':<14} {'n':>4} {'t/o':>3} {'range':>6} {'(deg)':>6} {'maxstep':>7} {'jumps':>5} "
              f"{'zeros':>5} {'drift':>5} {'class':<8} {'volts':>11} {'error':>6} {'mode':>4}")
        for key, s in summary.items():
            limb = buses[key.split(":")[0]][1]
            joint = JOINTS[limb].get(int(key.split(":")[1]), "NOT IN DOCS")
            volts = f"{s['voltage_min']:.2f}-{s['voltage_max']:.2f}" if s["voltage_min"] is not None else "-"
            print(f"{key:<9} {limb:<9} {joint:<14} {s['samples']:>4} {s['timeouts']:>3} {s.get('range_counts', '-'):>6} "
                  f"{s.get('range_deg', '-'):>6} {s.get('max_step_counts', '-'):>7} {s.get('jumps', '-'):>5} "
                  f"{s['zeros']:>5} {s.get('net_drift_counts', '-'):>5} {s['class']:<8} {volts:>11} "
                  f"{','.join(hex(e) for e in s['errors_seen']) or '-':>6} {','.join(str(m) for m in s['modes_seen']) or '-':>4}")
        print(f"\nRounds where more than one motor jumped (> {JUMP_STEP} counts): {len(shared)}")
        for round_index, keys in list(shared.items())[:10]:
            print(f"  round {round_index}: {keys}")
        print(f"Unsolicited frames per bus: { {c: len(v) for c, v in record['unsolicited'].items()} }")
        print(f"Error counter change per bus: { {c: {k: v for k, v in d.items() if v} for c, d in record['error_delta'].items()} }")
        print(f"Stop reason: {record['stop_reason']}")
        print(f"Saved {out / 'stationary_encoder_audit.json'}")


if __name__ == "__main__":
    main()
