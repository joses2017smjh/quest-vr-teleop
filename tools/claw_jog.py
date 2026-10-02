"""Jog the claw servos with the keyboard and record their open/closed pulse widths.

Talks to scripts/arduino/claw_controller.ino (text protocol: P 3|7 <us>, STATUS,
OFF) on the CH340 Nano. Run it in a real terminal, since it reads single key presses:

  .venv/bin/python tools/claw_jog.py

Keys:
  1 / 2     select the claw on D3 / D7
  a / d     -10 / +10 us          A / D   -50 / +50 us
  o / c     mark the selected claw's current pulse as OPEN / CLOSED
  space     both claws limp (OFF)
  q         limp, save, quit

Mark CLOSED where the jaws just meet, not squeezing: a servo pushing on its own jaws
stalls and heats. The firmware slews at 250 us/s, so a claw keeps moving for a moment
after the key press. Limits are saved to configs/claw_limits.json.
"""

from __future__ import annotations

import glob
import json
import os
import select
import sys
import termios
import time
import tty

import serial

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIMITS = os.path.join(REPO, "configs", "claw_limits.json")
PINS = (3, 7)
MIN_US, MAX_US = 1000, 2000


def open_claw_port() -> serial.Serial:
    """The CH340 has no unique serial number, so confirm the sketch by its banner."""
    for path in sorted(glob.glob("/dev/serial/by-id/usb-1a86_USB_Serial*")):
        s = serial.Serial(path, 115200, timeout=0.1)
        time.sleep(2.2)                      # opening resets the Nano
        boot = s.read(256).decode(errors="replace")
        s.write(b"STATUS\n")
        time.sleep(0.3)
        reply = boot + s.read(256).decode(errors="replace")
        if "_READY" in reply or "D3 " in reply:
            return s
        s.close()
    raise SystemExit("no claw Nano answering on /dev/serial/by-id/usb-1a86_USB_Serial*")


def load() -> dict:
    try:
        with open(LIMITS) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {"claws": {}}


def save(data: dict) -> None:
    data["saved"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    data["note"] = "Pulse widths for scripts/arduino/claw_controller.ino, from tools/claw_jog.py."
    os.makedirs(os.path.dirname(LIMITS), exist_ok=True)
    with open(LIMITS, "w") as fh:
        json.dump(data, fh, indent=2)


def main() -> int:
    port = open_claw_port()
    data = load()
    claws = data.setdefault("claws", {})
    us = {p: 1500 for p in PINS}
    sel = PINS[0]
    note = "claws start limp; the first key press on a claw moves it to 1500 us first"

    def send(text: str) -> None:
        port.write((text + "\n").encode())

    def show() -> None:
        parts = []
        for p in PINS:
            lim = claws.get(f"D{p}", {})
            mark = ">" if p == sel else " "
            parts.append(f"{mark}D{p} {us[p]:4d}us open={lim.get('open_us', '-')} "
                         f"closed={lim.get('closed_us', '-')}")
        sys.stdout.write("\r\033[K" + "   ".join(parts) + f"   | {note}")
        sys.stdout.flush()

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        print(__doc__.split("Keys:")[1].split("Mark CLOSED")[0].rstrip())
        show()
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            port.read(512)                   # keep the Nano's replies drained
            if not ready:
                continue
            key = sys.stdin.read(1)
            note = ""
            if key in "12":
                sel = PINS[int(key) - 1]
            elif key in "aAdD":
                step = {"a": -10, "d": 10, "A": -50, "D": 50}[key]
                us[sel] = max(MIN_US, min(MAX_US, us[sel] + step))
                send(f"P {sel} {us[sel]}")
            elif key in "oc":
                which = "open_us" if key == "o" else "closed_us"
                claws.setdefault(f"D{sel}", {})[which] = us[sel]
                note = f"D{sel} {which[:-3]} = {us[sel]} us"
            elif key == " ":
                send("OFF")
                note = "both limp"
            elif key == "q":
                break
            show()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        send("OFF")
        time.sleep(0.2)
        port.close()
        save(data)
        print(f"\nboth claws limp; limits saved to {LIMITS}")
        print(json.dumps(data.get("claws", {}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
