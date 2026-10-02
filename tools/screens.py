#!/usr/bin/env python3
"""Lay out the two screens: the PC monitor as the main desktop, the robot's face
display to its right, turned 180 degrees (it is mounted upside down, like the camera).

  python3 tools/screens.py            apply now (no "keep changes?" dialog) and save for the next login
  python3 tools/screens.py --show     print what GNOME has now

Needs the Intel driver (Linux >= 6.9 on this N150; tools/enable_dual_hdmi.sh).
"""
import sys
from pathlib import Path

sys.path.append("/usr/lib/python3/dist-packages")
from gi.repository import Gio, GLib  # noqa: E402

MONITOR, ROBOT = "HDMI-2", "HDMI-1"      # the ASUS VG245, the robot's 1024x600 screen
NORMAL, UPSIDE_DOWN = 0, 2

bus = Gio.bus_get_sync(Gio.BusType.SESSION)


def call(method, args=None):
    return bus.call_sync("org.gnome.Mutter.DisplayConfig", "/org/gnome/Mutter/DisplayConfig",
                         "org.gnome.Mutter.DisplayConfig", method, args, None,
                         Gio.DBusCallFlags.NONE, -1, None).unpack()


def best_mode(monitor):
    (connector, *_), modes, _props = monitor
    preferred = [m for m in modes if m[6].get("is-preferred")]
    return (preferred or modes)[0]


serial, monitors, logical, _ = call("GetCurrentState")
found = {m[0][0]: m for m in monitors}
if "--show" in sys.argv:
    for name, m in found.items():
        print(name, m[0][1], m[0][2], "preferred", best_mode(m)[0])
    for lm in logical:
        print("logical", lm[:5], [c[0] for c in lm[5]])
    sys.exit(0)
missing = [c for c in (MONITOR, ROBOT) if c not in found]
if missing:
    sys.exit(f"not connected: {missing}")
mon, rob = best_mode(found[MONITOR]), best_mode(found[ROBOT])
layout = [
    (0, 0, 1.0, NORMAL, True, [(MONITOR, mon[0], {})]),
    (mon[1], 0, 1.0, UPSIDE_DOWN, False, [(ROBOT, rob[0], {})]),
]
# Method 1 (temporary): method 2 (persistent) pops up GNOME's "Keep these display
# settings?" and quietly reverts after ~20 s unless someone clicks Keep - which is what
# happened on 29 Sep with the operator in the headset. Persistence comes from writing
# GNOME's own layout file instead, which it reads at login and on every hotplug.
call("ApplyMonitorsConfig", GLib.Variant("(uua(iiduba(ssa{sv}))a{sv})", (serial, 1, layout, {})))


def spec(monitor, mode, x, primary, rotation):
    (connector, vendor, product, serial_no), _modes, _props = monitor
    width, height, rate = mode[1], mode[2], mode[3]
    turn = "" if rotation == "normal" else (
        f"      <transform><rotation>{rotation}</rotation><flipped>no</flipped></transform>\n")
    return (f"    <logicalmonitor>\n      <x>{x}</x>\n      <y>0</y>\n      <scale>1</scale>\n"
            + ("      <primary>yes</primary>\n" if primary else "") + turn
            + f"      <monitor>\n        <monitorspec>\n          <connector>{connector}</connector>\n"
            f"          <vendor>{vendor}</vendor>\n          <product>{product}</product>\n"
            f"          <serial>{serial_no}</serial>\n        </monitorspec>\n"
            f"        <mode>\n          <width>{width}</width>\n          <height>{height}</height>\n"
            f"          <rate>{rate}</rate>\n        </mode>\n      </monitor>\n    </logicalmonitor>\n")


xml = ('<monitors version="2">\n  <configuration>\n'
       + spec(found[MONITOR], mon, 0, True, "normal")
       + spec(found[ROBOT], rob, mon[1], False, "upside_down")
       + "  </configuration>\n</monitors>\n")
path = Path.home() / ".config/monitors.xml"
if path.exists():
    path.with_suffix(".xml.before-bhl").write_text(path.read_text())
path.write_text(xml)
print(f"{MONITOR} {mon[0]} main, upright; {ROBOT} {rob[0]} to its right, turned 180 degrees.")
print(f"Applied now, no confirmation dialog; saved in {path} for the next login.")
