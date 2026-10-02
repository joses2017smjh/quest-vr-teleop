#!/usr/bin/env python3
"""Stream the robot's stereo USB camera into the headset page.

The "3D USB Camera" (32e4:2b10) sends both eyes side by side in one MJPEG frame.
ffmpeg copies those JPEGs as they are (no re-encoding) into
/dev/shm/bhl_camera.jpg, which quest_bridge.py streams to the headset at
/camera.mjpg; the page shows the left half to the left eye and the right half to
the right eye. Restarts ffmpeg if the camera drops out.

  .venv/bin/python scripts/teleop/camera_share.py                 # 1280x480 (two 640x480)
  .venv/bin/python scripts/teleop/camera_share.py --size 2560x720 # sharper, ~2x the Wi-Fi
"""

from __future__ import annotations

import argparse
import glob
import subprocess
import sys
import time
from pathlib import Path

FRAME_FILE = Path("/dev/shm/bhl_camera.jpg")


def find_camera() -> str | None:
    """The first video node whose card name says 3D USB Camera and that captures."""
    for dev in sorted(glob.glob("/dev/video*")):
        try:
            info = subprocess.run(["v4l2-ctl", "-d", dev, "--info", "--list-formats"],
                                  capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.TimeoutExpired):
            continue
        if "3D USB Camera" in info and "MJPG" in info:
            return dev
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", help="video node (default: find the 3D USB Camera)")
    ap.add_argument("--size", default="1280x480", help="side-by-side frame size the camera offers")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    announced = False
    while True:
        dev = args.device or find_camera()
        if dev is None:
            if not announced:
                print("Camera: no 3D USB Camera found; waiting for it", flush=True)
                announced = True
            time.sleep(3)
            continue
        announced = False
        print(f"Camera: {dev} at {args.size}, {args.fps} fps -> {FRAME_FILE}", flush=True)
        proc = subprocess.run(
            # -y and -nostdin: a frame left over from the last run made ffmpeg stop and
            # ask "Overwrite? [y/N]" on a terminal nobody was watching, and wait forever.
            ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
             "-f", "v4l2", "-input_format", "mjpeg",
             "-video_size", args.size, "-framerate", str(args.fps), "-i", dev,
             "-c:v", "copy", "-f", "image2", "-update", "1", "-atomic_writing", "1", str(FRAME_FILE)],
            stdin=subprocess.DEVNULL)
        print(f"Camera: ffmpeg stopped (exit {proc.returncode}); retrying in 2 s", flush=True)
        time.sleep(2)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass
