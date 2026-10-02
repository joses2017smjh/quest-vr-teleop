#!/usr/bin/env python3
"""The robot's lidar and IMU, for the headset's developer view ("agent, developer").

  lidar  RPLidar C1 behind a CP2102N (10c4:ea60), 460800 baud, standard scan.
         Each full rotation -> /dev/shm/bhl_lidar.json  {"t", "points": [[deg, m], ...]}
  imu    WitMotion on a CH340 (1a86:7523), factory 9600 baud, 11-byte 0x55 packets.
         Found on the CAN adapters' USB hub; the claw Nano (also a CH340) is never
         touched: ports are opened exclusively, and run_teleop holds the Nano's.
         -> /dev/shm/bhl_imu.json  {"t", "roll", "pitch", "yaw", "acc", "gyro", "mag", "temp"}

The IMU is only read, never configured: the robot's own walking code
(berkeley_humanoid_lite_lowlevel/robot/imu.py) expects it set to 460800 baud with
quaternion output, and that is a separate, deliberate step.

quest_bridge.py serves both as /sensors.json.

  .venv/bin/python scripts/teleop/sensor_share.py
"""

from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import threading
import time
from pathlib import Path

LIDAR_FILE = Path("/dev/shm/bhl_lidar.json")
IMU_FILE = Path("/dev/shm/bhl_imu.json")


def usb_serial_ports(vendor: str) -> list[tuple[str, Path]]:
    """[(/dev/ttyUSBn, its USB device directory)] for one USB vendor id."""
    found = []
    for tty in sorted(Path("/sys/class/tty").glob("ttyUSB*")):
        usb = (tty / "device").resolve().parent.parent
        try:
            if (usb / "idVendor").read_text().strip() == vendor:
                found.append((f"/dev/{tty.name}", usb))
        except OSError:
            continue
    return found


def can_hub() -> Path | None:
    """The USB hub the CAN adapters hang off: where the IMU was plugged in."""
    for net in sorted(Path("/sys/class/net").glob("can*")):
        try:
            return (net / "device").resolve().parent.parent
        except OSError:
            continue
    return None


def put(path: Path, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix="." + path.stem)
    with os.fdopen(fd, "w") as handle:
        json.dump(data, handle, separators=(",", ":"))
    os.replace(tmp, path)


def lidar_loop(stop: threading.Event) -> None:
    import serial
    told = ""
    while not stop.is_set():
        ports = usb_serial_ports("10c4")
        if not ports:
            if told != "missing":
                told = "missing"
                print("Lidar: no CP2102N on USB; looking every 3 s", flush=True)
            stop.wait(3.0)
            continue
        path = ports[0][0]
        try:
            port = serial.Serial(path, 460800, timeout=1, exclusive=True)
        except (OSError, serial.SerialException) as exc:
            if told != "busy":
                told = "busy"
                print(f"Lidar: cannot open {path} ({exc}); retrying", flush=True)
            stop.wait(3.0)
            continue
        try:
            port.dtr = False
            port.write(b"\xa5\x25")                   # stop whatever it was doing
            time.sleep(0.05)
            port.reset_input_buffer()
            port.write(b"\xa5\x20")                   # standard scan
            if port.read(7) != b"\xa5\x5a\x05\x00\x00\x40\x81":
                raise OSError("no scan descriptor")
            print(f"Lidar: scanning on {path}", flush=True)
            told = ""
            buf, current, last = b"", [], 0.0
            heard = time.monotonic()
            while not stop.is_set():
                chunk = port.read(2048)
                if not chunk:
                    # the motor stops with the scan and needs a moment to spin back up
                    if time.monotonic() - heard > 4.0:
                        raise OSError("the lidar stopped sending")
                    continue
                heard = time.monotonic()
                buf += chunk
                i = 0
                while i + 5 <= len(buf):
                    b = buf[i:i + 5]
                    s, ns = b[0] & 1, (b[0] >> 1) & 1
                    if s == ns or not b[1] & 1:        # out of step: slide by one byte
                        i += 1
                        continue
                    if s and current:
                        now = time.time()
                        if now - last > 0.2:           # at most 5 rotations a second on disk
                            last = now
                            put(LIDAR_FILE, {"t": now, "points": current[::max(1, len(current) // 360)]})
                        current = []
                    dist = (b[3] | (b[4] << 8)) / 4000.0          # metres
                    if dist > 0:
                        current.append([round(((b[2] << 7) | (b[1] >> 1)) / 64.0, 1), round(dist, 3)])
                    i += 5
                buf = buf[i:]
        except (OSError, serial.SerialException) as exc:
            print(f"Lidar: {exc}; reopening", flush=True)
        finally:
            try:
                port.write(b"\xa5\x25")
                port.close()
            except Exception:
                pass
        stop.wait(1.0)


def imu_packet(state: dict, kind: int, data: bytes) -> None:
    v = struct.unpack("<hhhh", data)
    if kind == 0x51:
        state["acc"] = [round(x / 32768 * 16, 3) for x in v[:3]]       # g
        state["temp"] = round(v[3] / 100, 1)                            # degrees C
    elif kind == 0x52:
        state["gyro"] = [round(x / 32768 * 2000, 1) for x in v[:3]]    # deg/s
    elif kind == 0x53:
        state["roll"], state["pitch"], state["yaw"] = (round(x / 32768 * 180, 1) for x in v[:3])
    elif kind == 0x54:
        state["mag"] = list(v[:3])


def imu_loop(stop: threading.Event) -> None:
    import serial
    told = ""
    while not stop.is_set():
        hub = can_hub()
        ports = usb_serial_ports("1a86")
        # the CH340 on the CAN hub first; any other only if it speaks WitMotion
        ports.sort(key=lambda item: 0 if hub is not None and item[1].parent == hub else 1)
        port = None
        for path, _usb in ports:
            try:
                candidate = serial.Serial(path, 9600, timeout=0.5, exclusive=True)
            except (OSError, serial.SerialException):
                continue                               # held: that is the claw Nano (run_teleop)
            sample = candidate.read(64)
            if b"\x55\x51" in sample or b"\x55\x53" in sample:
                port = candidate
                break
            candidate.close()
        if port is None:
            if told != "missing":
                told = "missing"
                print("IMU: no WitMotion found on a CH340 at 9600 baud; looking every 5 s", flush=True)
            stop.wait(5.0)
            continue
        print(f"IMU: WitMotion on {port.port}", flush=True)
        told = ""
        state: dict = {}
        buf, last = b"", 0.0
        try:
            while not stop.is_set():
                chunk = port.read(256)
                if not chunk:
                    raise OSError("the IMU stopped sending")
                buf += chunk
                while len(buf) >= 11:
                    if buf[0] != 0x55 or buf[1] < 0x50 or buf[1] > 0x5A:
                        buf = buf[1:]
                        continue
                    packet, buf = buf[:11], buf[11:]
                    if sum(packet[:10]) & 0xFF == packet[10]:
                        imu_packet(state, packet[1], packet[2:10])
                now = time.time()
                if state and now - last > 0.1:
                    last = now
                    put(IMU_FILE, dict(state, t=now))
        except (OSError, serial.SerialException) as exc:
            print(f"IMU: {exc}; reopening", flush=True)
        finally:
            port.close()
        stop.wait(1.0)


def main() -> int:
    stop = threading.Event()
    threads = [threading.Thread(target=lidar_loop, args=(stop,), daemon=True),
               threading.Thread(target=imu_loop, args=(stop,), daemon=True)]
    for thread in threads:
        thread.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        stop.set()
        for thread in threads:
            thread.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
