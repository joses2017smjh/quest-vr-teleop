#!/usr/bin/env python3
"""The robot's face on its own screen: two eyes that follow you and show how it feels.

person_track.py (and run_teleop, which adds the camera's own turn) sends where the
operator is on UDP 11010; the eyes look there, blink, and drift around when nobody is in
view. They have moods of their own - surprised and then glad when someone turns up,
focused while the arms are armed, worried when the robot raises an alert, sleepy and then
asleep when nobody has been around for a while - and take one on request (`--feel`, or
"agent, look happy" in the headset). The old face, a 3D Claude spark that turns to face
you, is still here: `--face spark`, or "agent, show the spark".

  .venv/bin/python scripts/teleop/face_display.py               # covers the robot's display
  .venv/bin/python scripts/teleop/face_display.py --window      # in a window, to try it
  .venv/bin/python scripts/teleop/face_display.py --snapshot out.png --feel happy --look 20,5
  .venv/bin/python scripts/teleop/face_display.py --feel angry --for 8   # tell the running face
  .venv/bin/python scripts/teleop/face_display.py --feel auto            # its own moods again
  .venv/bin/python scripts/teleop/face_display.py --show spark           # or eyes

The robot's screen is found by its size (1024x600); GNOME turns it 180 degrees
(tools/screens.py), so nothing here is drawn upside down.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import socket
import sys
import threading
import time
import urllib.request

import numpy as np

ORANGE = np.array([217, 119, 87], float)     # Claude's terracotta, #D97757
BACKGROUND = (12, 12, 14)
RAYS = 12
DEPTH = 0.22                                  # extrusion, in units of the spark's radius
MAX_TURN = math.radians(60)                   # never turn edge-on to anyone


def spark_outline(rays: int = RAYS, points_per_ray: int = 9) -> np.ndarray:
    """The outline: rounded rays of slightly different lengths, like the hand-drawn mark."""
    rng = np.random.default_rng(7)
    lengths = 0.78 + 0.22 * rng.random(rays)
    outline = []
    for k in range(rays):
        mid = 2 * math.pi * k / rays
        half = math.pi / rays * 0.66                     # half the width of a ray at its root
        for j in range(points_per_ray):                  # up one side, round the tip, down the other
            u = j / (points_per_ray - 1)
            a = mid - half + 2 * half * u
            tip = math.sin(math.pi * u) ** 0.18           # blunt, rounded end
            r = 0.16 + (lengths[k] - 0.16) * tip
            outline.append((r * math.cos(a), r * math.sin(a)))
        valley = mid + math.pi / rays                     # the notch between two rays
        outline.append((0.15 * math.cos(valley), 0.15 * math.sin(valley)))
    return np.array(outline)


def rotation(yaw: float, pitch: float, spin: float) -> np.ndarray:
    """Spin about the spark's own axis, then tilt up/down, then turn left/right."""
    cz, sz = math.cos(spin), math.sin(spin)
    cx, sx = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return Ry @ Rx @ Rz


class Spark:
    def __init__(self):
        flat = spark_outline()
        n = len(flat)
        self.front = np.column_stack([flat, np.full(n, DEPTH / 2)])
        self.back = np.column_stack([flat, np.full(n, -DEPTH / 2)])
        self.n = n

    def draw(self, surface, pygame, R: np.ndarray, size: float, centre: tuple[float, float]):
        light = np.array([0.3, 0.5, 1.0]) / np.linalg.norm([0.3, 0.5, 1.0])
        eye = 3.2                                         # camera distance, in spark radii

        def project(p):
            q = p @ R.T
            k = eye / (eye - q[:, 2])
            return np.column_stack([centre[0] + q[:, 0] * k * size, centre[1] - q[:, 1] * k * size]), q

        front, qf = project(self.front)
        back, qb = project(self.back)
        faces = []                                        # (depth, colour, polygon)
        for i in range(self.n):                           # the sides, one quad per outline edge
            j = (i + 1) % self.n
            a, b = qf[i], qf[j]
            normal = np.cross(b - a, qb[i] - a)
            norm = np.linalg.norm(normal)
            if norm < 1e-9:
                continue
            normal /= norm
            shade = 0.5 + 0.4 * abs(float(normal @ light))
            faces.append((min(a[2], b[2], qb[i][2], qb[j][2]), ORANGE * shade * 0.78,
                          [front[i], front[j], back[j], back[i]]))
        normal = R @ np.array([0, 0, 1.0])
        cap = (front, qf) if normal[2] >= 0 else (back, qb)
        shade = 0.45 + 0.55 * abs(float(normal @ light))
        faces.append((float(cap[1][:, 2].mean()) + 1.0, ORANGE * shade, list(cap[0])))
        for _, colour, poly in sorted(faces, key=lambda f: f[0]):
            pygame.draw.polygon(surface, tuple(int(c) for c in np.clip(colour, 0, 255)), poly)


# ---------------------------------------------------------------------------------- eyes
# Each eye is a rounded square, glowing, its shape set by lids: the upper one cuts down
# from the top, straight, lower at the inner or outer corner (cross, sad), and the lower
# one pushes up in an arc (glad). Moods are just these numbers, so one eases into the next.
#   w, h      size against the eye at rest
#   top_in    how far the upper lid comes down at the corner nearer the nose, 0..1 of the eye
#   top_out   ... at the outer corner
#   bottom    how far the lower lid pushes up in the middle (its ends a little less)
#   shut      0 open .. 1 closed onto a curved line (asleep; a blink does the same for a moment)
#   colour    what it glows
EXPRESSIONS = {
    "calm":      dict(w=1.0, h=1.0, top_in=0.0, top_out=0.0, bottom=0.0, shut=0.0, colour=(217, 119, 87)),
    "happy":     dict(w=1.06, h=0.96, top_in=0.0, top_out=0.0, bottom=0.44, colour=(240, 146, 82)),
    "surprised": dict(w=1.13, h=1.17, top_in=0.0, top_out=0.0, bottom=0.0, colour=(246, 168, 110)),
    "sad":       dict(w=1.0, h=0.88, top_in=0.0, top_out=0.34, bottom=0.1, colour=(112, 150, 232)),
    "angry":     dict(w=1.0, h=0.92, top_in=0.5, top_out=0.06, bottom=0.12, colour=(236, 72, 58)),
    "worried":   dict(w=1.02, h=1.04, top_in=0.0, top_out=0.3, bottom=0.1, colour=(232, 162, 64)),
    "focused":   dict(w=1.0, h=0.96, top_in=0.24, top_out=0.18, bottom=0.18, colour=(217, 119, 87)),
    "curious":   dict(w=1.0, h=1.0, top_in=0.0, top_out=0.0, bottom=0.0, colour=(217, 119, 87),
                      left=dict(h=1.12), right=dict(h=0.86, top_in=0.26, top_out=0.14)),
    "love":      dict(w=1.06, h=0.96, top_in=0.0, top_out=0.0, bottom=0.44, colour=(242, 104, 150)),
    "sleepy":    dict(w=1.0, h=1.0, top_in=0.54, top_out=0.54, bottom=0.12, colour=(168, 94, 70)),
    "asleep":    dict(w=1.0, h=1.0, top_in=0.0, top_out=0.0, bottom=0.0, shut=1.0, colour=(150, 86, 64)),
}
MOODS = tuple(EXPRESSIONS)
SHAPE_KEYS = ("w", "h", "top_in", "top_out", "bottom", "shut")


def outline(n_points: int = 64, squareness: float = 3.4) -> tuple[np.ndarray, np.ndarray]:
    """A rounded square (superellipse) of half-size 1, as x and y (y down), counter-clockwise."""
    t = np.linspace(0, 2 * math.pi, n_points, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    return (np.sign(c) * np.abs(c) ** (2 / squareness), -np.sign(s) * np.abs(s) ** (2 / squareness))


class Eyes:
    """Where the eyes look, how they feel, when they blink; drawn with anti-aliased polygons
    at the screen's own size, so it costs a fraction of what the spark's supersampling did."""

    def __init__(self, w: int, h: int):
        self.W, self.H = w, h
        self.ux, self.uy = outline()
        self.shape = {side: {k: EXPRESSIONS["calm"].get(k, 0.0) for k in SHAPE_KEYS} for side in ("left", "right")}
        self.colour = np.array(EXPRESSIONS["calm"]["colour"], float)
        self.gaze = np.zeros(2)                    # -1..1 across and down the screen
        self.target = np.zeros(2)
        self.blink_at = time.monotonic() + 2.0
        self.blink = 0.0                           # 0 open .. 1 shut
        self.blinking = -1.0                       # when the current blink started
        self.wander_at = 0.0
        self.zs: list[list[float]] = []            # asleep: (born, x, y) of each floating z
        self.z_at = 0.0
        self.font = None

    def look(self, yaw_deg: float | None, pitch_deg: float | None, now: float, mood: str) -> None:
        """Someone at yaw (+ = the robot's left) and pitch (+ = up), or nobody (None)."""
        if yaw_deg is not None:
            # facing the robot, its left is on your right: the eyes go that way across the screen
            self.target = np.array([max(-1.0, min(1.0, yaw_deg / 35.0)), max(-1.0, min(1.0, -pitch_deg / 25.0))])
        elif mood in ("sleepy", "asleep"):
            self.target = np.array([0.0, 0.35])
        elif now >= self.wander_at:                # nobody: a look around now and then
            self.wander_at = now + random.uniform(1.2, 3.5)
            self.target = np.array([random.uniform(-0.45, 0.45), random.uniform(-0.25, 0.3)])

    def update(self, dt: float, now: float, mood: str) -> None:
        want = EXPRESSIONS.get(mood, EXPRESSIONS["calm"])
        k = 1 - math.exp(-dt / 0.12)                # moods ease into each other
        for side in ("left", "right"):
            goal = dict({key: want.get(key, 0.0) for key in SHAPE_KEYS}, **want.get(side, {}))
            for key in SHAPE_KEYS:
                self.shape[side][key] += (goal[key] - self.shape[side][key]) * k
        self.colour += (np.array(want["colour"], float) - self.colour) * k
        self.gaze += (self.target - self.gaze) * (1 - math.exp(-dt / 0.08))   # quick, like a glance
        # a blink every few seconds (now and then two), slower and rarer when sleepy, none asleep
        if mood != "asleep" and now >= self.blink_at and self.blinking < 0:
            self.blinking = now
        if self.blinking >= 0:
            length = 0.34 if mood == "sleepy" else 0.16
            p = (now - self.blinking) / length
            self.blink = math.sin(math.pi * min(1.0, p)) if p < 1 else 0.0
            if p >= 1:
                self.blinking = -1.0
                again = random.random() < 0.18 and mood != "sleepy"
                self.blink_at = now + (0.12 if again else random.uniform(2.2, 5.5) * (1.8 if mood == "sleepy" else 1))
        else:
            self.blink = 0.0
        if mood == "asleep" and now >= self.z_at:
            self.z_at = now + 1.1
            self.zs.append([now, 0.0])
        self.zs = [z for z in self.zs if now - z[0] < 2.6]

    def eye_points(self, side: str, now: float) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        """The outline of one eye on the screen, with its lids, and (cx, cy, a, b)."""
        W, H, sh = self.W, self.H, self.shape[side]
        gx, gy = float(self.gaze[0]), float(self.gaze[1])
        sign = -1 if side == "left" else 1          # the left eye is on the screen's left
        breathe = 1.0 + 0.012 * math.sin(now * 1.7)
        # turned toward one side, the eyes close up a little and the far one narrows: a head turning
        toward = 1 if sign * gx > 0 else 0
        a = 0.118 * W * sh["w"] * breathe * (1 - (0.12 if toward else 0.04) * abs(gx))
        b = 0.27 * H * sh["h"] * breathe
        cx = W / 2 + sign * 0.205 * W * (1 - 0.1 * abs(gx)) + gx * 0.15 * W
        cy = H / 2 + gy * 0.12 * H
        x, y = self.ux.copy(), self.uy.copy()       # unit outline, y down
        inner = (1 + x) / 2 if side == "left" else (1 - x) / 2    # 1 at the corner nearer the nose
        # looking down, the upper lid follows the eye down a little; looking up, the lower one
        top = -1 + 2 * (sh["top_out"] + (sh["top_in"] - sh["top_out"]) * inner + 0.12 * max(0.0, gy))
        low = 1 - 2 * (sh["bottom"] * (1 - 0.55 * x * x) + 0.08 * max(0.0, -gy))
        mid = 0.16 + 0.16 * (1 - x * x)              # where the lids meet: a curve, lowest mid-eye
        close = max(self.blink, sh["shut"])
        top = top + (mid - top) * close
        low = low + (mid - low) * close
        meet = top > low
        top, low = np.where(meet, (top + low) / 2, top), np.where(meet, (top + low) / 2, low)
        thin = 9.0 / b                               # shut, an eye is still a line you can see
        low = np.maximum(low, top + thin)
        y = np.clip(y, top, low)
        return np.column_stack([cx + x * a, cy + y * b]), (cx, cy, a, b)

    def draw(self, surface, pygame, gfx, now: float, mood: str) -> None:
        surface.fill(BACKGROUND)
        colour = tuple(int(c) for c in np.clip(self.colour, 0, 255))
        for side in ("left", "right"):
            pts, (cx, cy, a, b) = self.eye_points(side, now)
            # the glow: two fainter, slightly bigger copies behind the eye
            for grow, strength in ((1.14, 0.16), (1.06, 0.34)):
                halo = np.column_stack([cx + (pts[:, 0] - cx) * grow, cy + (pts[:, 1] - cy) * grow])
                tint = tuple(int(BACKGROUND[i] + (colour[i] - BACKGROUND[i]) * strength) for i in range(3))
                ring = [(int(round(px)), int(round(py))) for px, py in halo]
                gfx.filled_polygon(surface, ring, tint)
                gfx.aapolygon(surface, ring, tint)
            ring = [(int(round(px)), int(round(py))) for px, py in pts]
            gfx.filled_polygon(surface, ring, colour)
            gfx.aapolygon(surface, ring, colour)
            # a glint, where the eye is open enough to show it
            gx_, gy_ = cx - 0.34 * a, cy - 0.42 * b
            column = np.abs(pts[:, 0] - gx_) < 0.2 * a
            if column.any() and pts[column, 1].min() < gy_ - 0.12 * b and pts[column, 1].max() > gy_ + 0.12 * b:
                shine = tuple(min(255, int(c + (255 - c) * 0.55)) for c in colour)
                r = max(2, int(0.15 * a))
                gfx.filled_circle(surface, int(gx_), int(gy_), r, shine)
                gfx.aacircle(surface, int(gx_), int(gy_), r, shine)
        if self.zs:                                  # asleep: z z z, rising off the right eye
            if self.font is None:
                self.font = pygame.font.SysFont("DejaVu Sans", 54, bold=True)
            _, (cx, cy, a, b) = self.eye_points("right", now)
            for born, _ in self.zs:
                age = (now - born) / 2.6
                glyph = self.font.render("z", True, (236, 178, 150))
                size = 0.6 + 0.9 * age
                glyph = pygame.transform.smoothscale(glyph, (max(1, int(glyph.get_width() * size)),
                                                             max(1, int(glyph.get_height() * size))))
                glyph.set_alpha(int(255 * (1 - age)))
                surface.blit(glyph, (int(cx + a * (0.6 + 1.2 * age)), int(cy - b * (0.4 + 1.6 * age))))


class Moods:
    """How the robot feels, by itself: surprised, then glad, when someone turns up; focused
    while armed; worried while it raises an alert; sleepy after 25 s alone and asleep after
    90 s. A mood asked for (`--feel`, "agent, look happy") wins until it runs out."""

    def __init__(self, now: float):
        self.asked, self.asked_until = "", 0.0
        self.seen_at = now - 5.0                    # awake at start; sleepy only after 25 s alone
        self.surprised_until = self.happy_until = 0.0
        self.smile_at = now + random.uniform(25, 45)
        self.robot: dict = {}

    def saw(self, now: float) -> None:
        if now - self.seen_at > 4.0:                # someone new (or back): a start, then a smile
            self.surprised_until = now + 0.7
            self.happy_until = now + 3.2
        self.seen_at = now

    def ask(self, mood: str, seconds: float, now: float) -> str:
        if mood == "auto":
            self.asked = ""
            return "its own moods again"
        self.asked, self.asked_until = mood, (now + seconds if seconds > 0 else math.inf)
        return f"{mood} for {seconds:g} s" if seconds > 0 else f"{mood} until told otherwise"

    def now(self, now: float) -> str:
        if self.asked and now < self.asked_until:
            return self.asked
        self.asked = ""
        seen = now - self.seen_at < 1.5
        if self.robot.get("alert"):
            return "worried"
        if seen:
            if now < self.surprised_until:
                return "surprised"
            if now < self.happy_until:
                return "happy"
            if now >= self.smile_at:                # a smile now and then, while you are there
                self.happy_until = now + 1.8
                self.smile_at = now + random.uniform(25, 45)
            if self.robot.get("state") == "ARMED":
                return "focused"
            if self.robot.get("state") == "CAL":
                return "curious"
            return "calm"
        alone = now - self.seen_at
        if alone > 90:
            return "asleep"
        if alone > 25:
            return "sleepy"
        return "calm"


def watch_robot(moods: Moods, url: str) -> None:
    """The robot's state and alert, from the bridge, once a second (a thread: never blocks drawing)."""
    while True:
        try:
            with urllib.request.urlopen(url, timeout=0.8) as answer:
                status = json.loads(answer.read().decode()).get("last_status") or {}
            fresh = True
        except Exception:
            status, fresh = {}, False
        moods.robot = {"state": status.get("state", "") if fresh else "", "alert": status.get("alert", "") if fresh else ""}
        time.sleep(1.0)


def robot_screen(connector: str = "HDMI-1") -> tuple[int, int, int, int]:
    """(x, y, width, height) of the robot's screen on the desktop, from GNOME."""
    try:
        sys.path.append("/usr/lib/python3/dist-packages")
        from gi.repository import Gio
        bus = Gio.bus_get_sync(Gio.BusType.SESSION)
        _, monitors, logical, _ = bus.call_sync(
            "org.gnome.Mutter.DisplayConfig", "/org/gnome/Mutter/DisplayConfig",
            "org.gnome.Mutter.DisplayConfig", "GetCurrentState", None, None,
            Gio.DBusCallFlags.NONE, -1, None).unpack()
        for x, y, _scale, transform, _primary, members, _ in logical:
            if any(m[0] == connector for m in members):
                spec = next(m for m in monitors if m[0][0] == connector)
                mode = next(md for md in spec[1] if md[6].get("is-current")) if any(
                    md[6].get("is-current") for md in spec[1]) else spec[1][0]
                w, h = mode[1], mode[2]
                return (x, y, h, w) if transform in (1, 3, 5, 7) else (x, y, w, h)
    except Exception as exc:
        print(f"  (could not ask GNOME where the robot screen is: {exc})", flush=True)
    return 1920, 0, 1024, 600                            # tools/screens.py's layout


def tell_face(port: int, message: dict) -> int:
    """--feel / --show: ask the running face, and say what it answered."""
    link = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    link.settimeout(1.5)
    try:
        link.sendto(json.dumps(dict(message, type="face")).encode(), ("127.0.0.1", port))
        reply = json.loads(link.recvfrom(4096)[0])
        print(reply.get("said", reply))
        return 0 if reply.get("ok") else 1
    except (OSError, ValueError):
        print("the face did not answer: is face_display.py running (bringup window `face`)?")
        return 1
    finally:
        link.close()


def face_command(report: dict, moods: Moods, face: list[str], now: float) -> dict:
    """{"type": "face", "feel": mood|"auto", "for": s} or {"type": "face", "show": "eyes"|"spark"}."""
    if report.get("show") in ("eyes", "spark"):
        face[0] = report["show"]
        return {"ok": True, "said": f"showing the {report['show']}"}
    feel = str(report.get("feel", "")).lower()
    if feel != "auto" and feel not in EXPRESSIONS:
        return {"ok": False, "said": f"no mood called {feel!r}: {', '.join(MOODS)}, or auto"}
    try:
        seconds = float(report.get("for", 10))
    except (TypeError, ValueError):
        seconds = 10.0
    face[0] = "eyes"
    return {"ok": True, "said": moods.ask(feel, seconds, now)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window", action="store_true", help="a window on the main screen instead")
    parser.add_argument("--snapshot", metavar="PNG", help="render one frame to a file and exit")
    parser.add_argument("--port", type=int, default=11010, help="where person_track.py reports")
    parser.add_argument("--spin", type=float, default=18.0, help="the spark: degrees per second about its own axis")
    parser.add_argument("--face", choices=("eyes", "spark"), default="eyes", help="what to show at start")
    parser.add_argument("--feel", help=f"a mood ({', '.join(MOODS)}, auto): for the running face, or the snapshot's")
    parser.add_argument("--for", dest="seconds", type=float, default=10.0, help="how long --feel lasts (0: until told)")
    parser.add_argument("--show", choices=("eyes", "spark"), help="switch the running face")
    parser.add_argument("--look", default="", help="the snapshot's gaze: yaw,pitch in degrees (+ = robot's left, up)")
    parser.add_argument("--bridge", default="http://127.0.0.1:8080/state.json", help="the robot's state, for moods")
    args = parser.parse_args()

    if not args.snapshot and (args.feel or args.show):
        return tell_face(args.port, {"feel": args.feel, "for": args.seconds} if args.feel else {"show": args.show})

    if args.snapshot:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    import pygame.gfxdraw

    pygame.init()
    spark = Spark()
    if args.snapshot:
        if args.face == "spark":
            big = pygame.Surface((2048, 1200))
            big.fill(BACKGROUND)
            spark.draw(big, pygame, rotation(math.radians(35), math.radians(-10), 0.3), 504, (1024, 600))
            pygame.image.save(pygame.transform.smoothscale(big, (1024, 600)), args.snapshot)
        else:
            mood = args.feel if args.feel in EXPRESSIONS else "calm"
            eyes, shot = Eyes(1024, 600), pygame.Surface((1024, 600))
            yaw, pitch = (float(v) for v in args.look.split(",")) if args.look else (None, None)
            for i in range(90):                       # 3 s at 30 Hz: settled into the mood and the look
                now = i / 30.0
                eyes.look(yaw, pitch, now, mood) if yaw is not None else None
                eyes.blink_at = 1e9
                eyes.update(1 / 30.0, now, mood)
            eyes.draw(shot, pygame, pygame.gfxdraw, 3.0, mood)
            pygame.image.save(shot, args.snapshot)
        print("saved", args.snapshot)
        return 0

    if args.window:
        screen = pygame.display.set_mode((1024, 600))
    else:
        # A borderless window laid exactly over the robot's screen - not "full screen".
        # Full screen through XWayland fakes a resolution change by scaling (the screen
        # came out hugely zoomed) and sits on top of every window there.
        x, y, w, h = robot_screen()
        os.environ["SDL_VIDEO_WINDOW_POS"] = f"{x},{y}"
        pygame.display.quit()
        pygame.display.init()
        screen = pygame.display.set_mode((w, h), pygame.NOFRAME)
    pygame.display.set_caption("robot face")
    w, h = screen.get_size()
    inbox = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    inbox.bind(("127.0.0.1", args.port))
    inbox.setblocking(False)
    clock = pygame.time.Clock()
    canvas = None                                         # the spark is drawn 1.5x and scaled down
    target = [0.0, 0.0]                                   # where to face: yaw, pitch (rad)
    shown = [0.0, 0.0]
    spin = 0.0
    last_seen = 0.0
    world_until = 0.0
    started = time.monotonic()
    face = [args.face]
    eyes = Eyes(w, h)
    moods = Moods(started)
    threading.Thread(target=watch_robot, args=(moods, args.bridge), daemon=True).start()
    print(f"Face ({face[0]}) on a {w}x{h} screen; listening for the operator on UDP {args.port}", flush=True)
    while True:
        # A display, not an app: typing never closes it. The borderless window can take
        # the keyboard focus when it opens, and a "q" meant for a terminal ended it.
        for event in pygame.event.get():
            if event.type == pygame.QUIT and args.window:
                return 0
        while True:
            try:
                data, sender = inbox.recvfrom(4096)
                report = json.loads(data.decode())
            except (BlockingIOError, ValueError):
                break
            if not isinstance(report, dict):
                continue
            if report.get("type") == "face":               # --feel, --show, "agent, look happy"
                answer = face_command(report, moods, face, time.monotonic())
                print(f"  {answer['said']}", flush=True)
                inbox.sendto(json.dumps(answer).encode(), sender)
                continue
            if report.get("type") == "person" and report.get("seen"):
                # run_teleop's reports ("world") add the camera's own angle. Once the
                # camera turns to centre you, the tracker's raw ones say "straight ahead",
                # so they only count when run_teleop has not reported lately.
                if report.get("world"):
                    world_until = time.monotonic() + 1.5
                elif time.monotonic() < world_until:
                    continue
                target = [max(-MAX_TURN, min(MAX_TURN, math.radians(report["yaw"]))),
                          max(-MAX_TURN / 2, min(MAX_TURN / 2, math.radians(report["pitch"])))]
                last_seen = time.monotonic()
                moods.saw(last_seen)
        now = time.monotonic()
        if now - last_seen > 3.0:                         # nobody: face straight out again
            target = [0.0, 0.0]
        if face[0] == "eyes":
            dt = clock.tick(25) / 1000.0          # a fifth of a core; the spark took four
            mood = moods.now(now)
            seen = now - last_seen <= 3.0
            eyes.look(math.degrees(target[0]) if seen else None, math.degrees(target[1]) if seen else None, now, mood)
            eyes.update(dt, now, mood)
            eyes.draw(screen, pygame, pygame.gfxdraw, now, mood)
            pygame.display.flip()
            continue
        dt = clock.tick(24) / 1000.0
        k = 1 - math.exp(-dt / 0.35)                      # turn smoothly, never snap
        shown = [shown[0] + (target[0] - shown[0]) * k, shown[1] + (target[1] - shown[1]) * k]
        spin += math.radians(args.spin) * dt
        breathe = 1.0 + 0.03 * math.sin((now - started) * 1.6)
        if canvas is None:
            canvas = pygame.Surface((int(1.5 * w), int(1.5 * h)))
        canvas.fill(BACKGROUND)
        spark.draw(canvas, pygame, rotation(shown[0], shown[1], spin), 0.63 * min(w, h) * breathe, (0.75 * w, 0.75 * h))
        pygame.transform.smoothscale(canvas, (w, h), screen)
        pygame.display.flip()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
