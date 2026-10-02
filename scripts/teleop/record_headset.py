#!/usr/bin/env python3
"""Record what you see in the Quest 2 - the room and the panels - to a video file.

The headset's own recorder leaves the room black on a Quest 2: Meta lets only the
Quest 3, 3S and Pro record passthrough. scrcpy mirrors the headset's screen over adb
instead, and that has shown Quest 2 passthrough; the first recording tells whether this
firmware still does. The headset encodes the video and this PC only writes it to disk:
no window, so nothing is decoded here (the PC has no CPU to spare while teleop runs).

One-time setup:
  1. Meta Horizon phone app: Menu > Devices > this headset > Headset settings >
     Developer mode on (it has you create a free developer organisation first).
  2. Plug the headset into this PC (USB-C). In the headset, allow USB debugging
     ("Always allow from this computer").

  .venv/bin/python scripts/teleop/record_headset.py wifi          # then unplug: adb over Wi-Fi until it restarts
  .venv/bin/python scripts/teleop/record_headset.py               # record the left eye until Ctrl-C
  .venv/bin/python scripts/teleop/record_headset.py --seconds 10  # a short test clip
  .venv/bin/python scripts/teleop/record_headset.py look          # watch it in a window (decodes: costs CPU)
  .venv/bin/python scripts/teleop/record_headset.py status        # scrcpy found? headset connected?

Recordings go to ~/Videos/bhl/headset-<date>-<time>.mkv. Sound is the headset's app
audio (Claude's voice), copied so it keeps playing in the headset; scrcpy's default would
mute the headset instead. The Quest 2 has no recording light: tell people in the room.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

OUT_DIR = Path.home() / "Videos" / "bhl"
LAST_IP = Path.home() / ".cache" / "bhl" / "quest_adb_ip"   # written by `wifi`, tried when nothing is connected
PORT = 5555
LEFT_EYE = "1832:1920:0:0"              # the Quest 2 screen is 3664x1920: both eyes side by side
SETUP = __doc__.split("One-time setup:")[1].split("\n\n")[0]


def find_scrcpy() -> Path | None:
    """scrcpy from $BHL_SCRCPY, a release unpacked under ~/.local/opt, or PATH."""
    candidates = [os.environ.get("BHL_SCRCPY", "")]
    candidates += sorted(glob.glob(str(Path.home() / ".local/opt/scrcpy-*/scrcpy")), reverse=True)
    candidates.append(shutil.which("scrcpy") or "")
    return next((Path(c) for c in candidates if c and os.access(c, os.X_OK)), None)


def adb_of(scrcpy: Path) -> str:
    """The adb that ships next to scrcpy in its release, else the system's."""
    bundled = scrcpy.parent / "adb"
    return str(bundled) if bundled.exists() else (shutil.which("adb") or "adb")


def adb(adb_bin: str, *args: str, timeout: float = 15) -> str:
    try:
        return subprocess.run([adb_bin, *args], capture_output=True, text=True, timeout=timeout).stdout
    except subprocess.TimeoutExpired:
        return ""


def devices(adb_bin: str) -> list[tuple[str, str]]:
    """[(serial, state)]: "device" is ready, "unauthorized" waits for the OK in the headset."""
    rows = [line.split() for line in adb(adb_bin, "devices").splitlines()[1:]]
    return [(row[0], row[1]) for row in rows if len(row) >= 2]


def pick(adb_bin: str) -> str | None:
    """The headset to record: a cable first (steadier), else Wi-Fi, reconnecting to the
    address `wifi` saved if adb has forgotten it (it does when the headset sleeps)."""
    ready = [serial for serial, state in devices(adb_bin) if state == "device"]
    if not ready and LAST_IP.exists():
        adb(adb_bin, "connect", f"{LAST_IP.read_text().strip()}:{PORT}")
        ready = [serial for serial, state in devices(adb_bin) if state == "device"]
    usb = [serial for serial in ready if ":" not in serial]
    return (usb or ready or [None])[0]


def explain_missing(adb_bin: str) -> int:
    waiting = [serial for serial, state in devices(adb_bin) if state == "unauthorized"]
    if waiting:
        print("The headset is plugged in but has not allowed this PC yet: put it on and accept\n"
              "\"Allow USB debugging\" (tick \"Always allow from this computer\").")
    else:
        print("No headset found. One-time setup:" + SETUP)
    return 2


def wifi(adb_bin: str) -> int:
    """Switch the plugged-in headset's adb to Wi-Fi, so the cable can come out."""
    usb = [serial for serial, state in devices(adb_bin) if state == "device" and ":" not in serial]
    if not usb:
        return explain_missing(adb_bin)
    serial = usb[0]
    found = re.search(r"inet (\d+\.\d+\.\d+\.\d+)",
                      adb(adb_bin, "-s", serial, "shell", "ip", "-f", "inet", "addr", "show", "wlan0"))
    if not found:
        print("The headset did not report a Wi-Fi address: is it on the same Wi-Fi as this PC?")
        return 1
    ip = found.group(1)
    adb(adb_bin, "-s", serial, "tcpip", str(PORT))
    for _ in range(10):                          # adb restarts in the headset, listening on the network
        time.sleep(1)
        if "connected to" in adb(adb_bin, "connect", f"{ip}:{PORT}"):
            LAST_IP.parent.mkdir(parents=True, exist_ok=True)
            LAST_IP.write_text(ip)
            print(f"Headset on Wi-Fi at {ip}:{PORT}. Unplug the cable; recording finds it by itself "
                  "until the headset restarts (then plug in and run `wifi` once more).")
            return 0
    print(f"The headset switched to Wi-Fi but {ip}:{PORT} did not answer. Same Wi-Fi as this PC?")
    return 1


def status(scrcpy: Path | None, adb_bin: str) -> int:
    if scrcpy is None:
        print("scrcpy: not found (unpack the Linux release under ~/.local/opt, or set BHL_SCRCPY)")
        return 1
    version = subprocess.run([str(scrcpy), "--version"], capture_output=True, text=True).stdout.split("\n")[0]
    print(f"scrcpy: {version} at {scrcpy}")
    listed = devices(adb_bin)
    print("headset: " + (", ".join(f"{serial} ({state})" for serial, state in listed) or "none connected"))
    if LAST_IP.exists():
        print(f"last Wi-Fi address: {LAST_IP.read_text().strip()}:{PORT}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="?", default="record", choices=("record", "look", "wifi", "status"))
    ap.add_argument("--seconds", type=int, default=0, help="stop by itself after this long (0: at Ctrl-C)")
    ap.add_argument("--crop", default=LEFT_EYE, help=f"width:height:x:y of the screen to keep (left eye: {LEFT_EYE})")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--bitrate", default="16M")
    ap.add_argument("--no-audio", action="store_true", help="video only")
    args = ap.parse_args()

    scrcpy = find_scrcpy()
    adb_bin = adb_of(scrcpy) if scrcpy else (shutil.which("adb") or "adb")
    if args.what == "status":
        return status(scrcpy, adb_bin)
    if scrcpy is None:
        return status(scrcpy, adb_bin)
    if args.what == "wifi":
        return wifi(adb_bin)
    serial = pick(adb_bin)
    if serial is None:
        return explain_missing(adb_bin)

    cmd = [str(scrcpy), "-s", serial, "--no-control", "--crop", args.crop, "--max-fps", str(args.fps),
           "--video-bit-rate", args.bitrate]
    if args.seconds:
        cmd += ["--time-limit", str(args.seconds)]
    out = None
    if args.what == "look":
        cmd += ["--no-audio", "--window-title", "Quest 2 (left eye)"]
    else:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / time.strftime("headset-%Y%m%d-%H%M%S.mkv")
        cmd += ["--no-window", "--no-playback", "--record", str(out)]
        # "output" (scrcpy's default) would silence the headset while recording
        cmd += ["--no-audio"] if args.no_audio else ["--audio-source", "playback", "--audio-dup"]
        print(f"Recording {serial} -> {out}" + ("" if args.seconds else "  (Ctrl-C stops and saves)"), flush=True)
    proc = subprocess.Popen(cmd)
    try:
        code = proc.wait()
    except KeyboardInterrupt:                    # scrcpy got the Ctrl-C too, and is closing the file
        code = proc.wait(timeout=20)
    if out is not None and out.exists():
        print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return code


if __name__ == "__main__":
    sys.exit(main())
