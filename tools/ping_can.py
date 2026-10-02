"""Ping motor IDs 1..16 on one CAN bus and say which answer. Read-only.

Sends only PDO-1 echo pings (what the other tools call ping); no mode change,
setpoint or flash write. Works on any bus, arms or legs.

  .venv/bin/python tools/ping_can.py can2
"""

from __future__ import annotations

import subprocess
import sys
import time

import can

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from diagnose_leg import LegBus  # noqa: E402

LEG = {1: "L hip roll", 3: "L hip yaw", 5: "L hip pitch", 7: "L knee", 11: "L ankle pitch", 13: "L ankle roll",
       2: "R hip roll", 4: "R hip yaw", 6: "R hip pitch", 8: "R knee", 12: "R ankle pitch", 14: "R ankle roll"}


def state(channel: str) -> str:
    out = subprocess.run(["ip", "-details", "link", "show", channel], capture_output=True, text=True).stdout
    for word in out.split("can state ")[1:2]:
        return word.split()[0]
    return "missing" if not out else "unknown"


def main() -> int:
    channel = sys.argv[1] if len(sys.argv) > 1 else "can2"
    print(f"{channel}: {state(channel)}")
    try:
        bus = LegBus(channel)
    except Exception as exc:
        print(f"cannot open {channel}: {exc}")
        return 1
    answered = []
    try:
        for dev in range(1, 17):
            try:
                ok = any(bus.ping(dev, 0.03) for _ in range(2))
            except (can.CanError, OSError) as exc:
                print(f"  ID {dev:2d}: send failed ({exc}) - the bus is not taking frames")
                break
            if ok:
                answered.append(dev)
                print(f"  ID {dev:2d}: answers   {LEG.get(dev, '')}")
            time.sleep(0.005)
    finally:
        bus.close()
    print(f"answered: {answered or 'none'}   {channel} now {state(channel)}")
    return 0 if answered else 2


if __name__ == "__main__":
    raise SystemExit(main())
