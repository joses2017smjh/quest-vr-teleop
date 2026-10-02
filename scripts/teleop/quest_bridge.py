#!/usr/bin/env python3
"""Quest 2 WebXR bridge for scripts/teleop/run_teleop.py.

SteamVR-Bridge is Windows + Vive. This serves a Quest Browser page that streams
left/right grip pose, squeeze, trigger and button-press counters to run_teleop
(UDP 11005), and relays run_teleop's status (UDP 11006) back to the headset,
where the page draws it as a panel you can read with the headset on.

Passthrough needs a secure page. Plain HTTP on the LAN works once the origin is
whitelisted in Quest Browser (chrome://flags); see the start-up message.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import importlib
import json
import logging
import math
import os
import socket
import ssl
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import tornado.ioloop
import tornado.web
import tornado.websocket
from cc.udp import UDP

HERE = Path(__file__).resolve().parent
HTML_PATH = HERE / "quest_bridge.html"
SAY_FILE = Path("/dev/shm/bhl_say.wav")   # written by scripts/teleop/say.py
CERT_DIR = HERE / ".certs"
EVENTS = ("x3", "y3", "a", "b", "cam", "camtrack", "camhead", "camstill")   # cam: A x2, the camera follows your head or holds still
SCREEN_FILE = Path("/dev/shm/bhl_screen.jpg")  # written by screen_share.py
CAMERA_FILE = Path("/dev/shm/bhl_camera.jpg")  # written by camera_share.py (stereo, side by side)
AUDIO_MAGIC = b"AUD0"  # marks microphone datagrams for voice_typer.py
BEACONS: collections.deque = collections.deque(maxlen=60)  # what the page last reported
BLANK_GIF = bytes.fromhex("47494638396101000100800000000000ffffff21f90401000000002c00000000"
                          "0100010000020144003b")

# WebXR/OpenGL standing space: +X right, +Y up, +Z toward the user.
# SteamVR-Bridge robotics frame used by run_teleop: +X forward, +Y left, +Z up.
_M = np.array(
    [
        [0.0, 0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)
_MINV = np.linalg.inv(_M)


def local_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def identity_pose() -> list[list[float]]:
    return np.eye(4, dtype=float).tolist()


def webxr_to_bhl(pose_list) -> list[list[float]] | None:
    try:
        world = np.array(pose_list, dtype=float)
    except (TypeError, ValueError):
        return None
    if world.shape != (4, 4) or not np.isfinite(world).all():
        return None
    robot = _M @ world @ _MINV
    return np.round(robot, 5).tolist()


def default_hand() -> dict:
    return {
        "pose": identity_pose(),
        "button_pressed": False,
        "trigger": 0.0,
        "tracked": False,
    }


def to_teleop_packet(raw: dict) -> dict:
    packet = {"left": default_hand(), "right": default_hand()}
    for side in ("left", "right"):
        hand = raw.get(side) or {}
        pose = webxr_to_bhl(hand.get("pose"))
        # A pose the headset only estimated (controller out of camera view)
        # must not drive an arm.
        tracked = pose is not None and bool(hand.get("tracked")) and not hand.get("emulated")
        packet[side]["pose"] = pose or identity_pose()
        packet[side]["tracked"] = tracked
        packet[side]["button_pressed"] = tracked and bool(hand.get("button_pressed"))
        try:
            packet[side]["trigger"] = float(hand.get("trigger") or 0.0)
        except (TypeError, ValueError):
            packet[side]["trigger"] = 0.0
    head = webxr_to_bhl(raw.get("head")) if raw.get("head") is not None else None
    if head is not None:
        packet["head"] = head
    stick = raw.get("stick")
    if isinstance(stick, (list, tuple)) and len(stick) == 2:
        try:
            packet["stick"] = [max(-1.0, min(1.0, float(stick[0]))), max(-1.0, min(1.0, float(stick[1])))]
        except (TypeError, ValueError):
            pass
    page = raw.get("page")
    if isinstance(page, str):
        packet["page"] = page[:32]
    counters = raw.get("ev") or {}
    packet["ev"] = {}
    for key in EVENTS:
        try:
            packet["ev"][key] = max(0, int(counters.get(key, 0)))
        except (TypeError, ValueError):
            packet["ev"][key] = 0
    return packet


def ensure_certs(ip: str) -> tuple[str, str]:
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    cert = CERT_DIR / "cert.pem"
    key = CERT_DIR / "key.pem"
    mark = CERT_DIR / "san.txt"
    san = f"DNS:localhost,IP:127.0.0.1,IP:{ip}"
    if cert.exists() and key.exists() and mark.exists() and mark.read_text().strip() == san:
        return str(cert), str(key)
    subprocess.check_call(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "365",
            "-nodes",
            "-subj",
            f"/CN={ip}",
            "-addext",
            f"subjectAltName={san}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    mark.write_text(san)
    return str(cert), str(key)


def maybe_adb_reverse(port: int) -> bool:
    try:
        listing = subprocess.check_output(["adb", "devices"], text=True, timeout=2)
    except (FileNotFoundError, subprocess.SubprocessError):
        return False
    serials = [
        line.split("\t", 1)[0]
        for line in listing.splitlines()[1:]
        if line.endswith("\tdevice")
    ]
    if not serials:
        return False
    try:
        subprocess.check_call(
            ["adb", "reverse", f"tcp:{port}", f"tcp:{port}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
    except subprocess.SubprocessError:
        return False
    return True


class IndexHandler(tornado.web.RequestHandler):
    http_port = 8080

    def get(self):
        self.set_header("Content-Type", "text/html; charset=utf-8")
        self.set_header("Cache-Control", "no-store")
        # The HTTPS copy of the page links to the plain-HTTP one.
        self.write(HTML_PATH.read_bytes().replace(b"__HTTP_PORT__", str(self.http_port).encode()))


SOCKET_STATE = {"0": "connecting", "1": "open", "2": "closing", "3": "closed", "none": "none"}
LAST_BEAT: dict = {}


def mic_level(raw: dict) -> str:
    """How loud the headset microphone is, when the page is reporting it.

    Silence is what a mic reads between sentences, so only a level worth seeing is
    shown; its absence says nothing, its presence says the microphone is alive.
    """
    try:
        level = int(raw.get("level", 0))
    except (TypeError, ValueError):
        return ""
    return f" (level {level}/1000)" if level > 0 else ""


def print_diag(raw: dict, ip: str) -> None:
    """What the headset browser reports: support, session mode, refusals."""
    event = raw.get("event")
    if event == "page":
        if not raw.get("xr"):
            text = (f"page loaded, secure={raw.get('secure')}, WebXR MISSING "
                    "(not a secure page: redo the chrome://flags step for this exact http:// address)")
        else:
            text = (f"page loaded, secure={raw.get('secure')}, "
                    f"passthrough(immersive-ar)={'supported' if raw.get('ar') else 'NOT supported'}, "
                    f"vr={'supported' if raw.get('vr') else 'no'}")
        text += f"  [{raw.get('url')}]"
    elif event == "session":
        mode = raw.get("mode")
        text = f"XR session started: {mode}, blend={raw.get('blend')}"
        text += "  -> passthrough ON" if raw.get("passthrough") else "  -> VR, black background (tap 'Start with passthrough' instead)"
    elif event == "refused":
        text = f"{raw.get('mode')} REFUSED by the browser: {raw.get('error')}"
    elif event == "setup-failed":
        text = f"XR setup failed after {raw.get('mode')} was granted: {raw.get('error')}"
    elif event == "transport":
        text = f"page is talking over {raw.get('how')}" + (f" ({raw.get('why')})" if raw.get("why") else "")
    elif event == "heartbeat":
        link = "connected" if str(raw.get("link")) == "1" else "NO LINK"
        status_age = raw.get("status")
        heard = "no status yet" if status_age in (None, "never") else f"status {status_age} ms ago"
        text = (f"{link} over {raw.get('transport')} ({heard}), "
                f"{'in the headset view' if raw.get('xr') == 'in' else 'on the flat page'}, "
                f"screen {'on' if str(raw.get('screen')) == '1' else 'off'}, "
                f"mic {raw.get('voice', '?')}{mic_level(raw)}"
                f" (audio {raw.get('ctx', '?')}, {raw.get('audio', '?')} chunks sent, "
                f"hands-free {'on' if str(raw.get('wake')) == '1' else 'off'}"
                f"{', Claude speaking' if str(raw.get('speaking')) == '1' else ''})"
                f" [ws {SOCKET_STATE.get(str(raw.get('ws')), raw.get('ws'))},"
                f" sse {SOCKET_STATE.get(str(raw.get('sse')), raw.get('sse'))}]")
        if raw.get("draw"):
            text += f"\n            panel error: {raw.get('draw')}"
    elif event == "fallback":
        text = f"the WebSocket did not work ({raw.get('why')}); trying server-sent events instead"
    elif event == "ws-closed":
        text = (f"WebSocket closed: code {raw.get('code')} {raw.get('reason') or ''}"
                f"{' (clean)' if str(raw.get('clean')) in ('True', 'true') else ''}")
    elif event == "ws-error":
        text = f"WebSocket error (state {SOCKET_STATE.get(str(raw.get('state')), raw.get('state'))})"
    elif event in ("sse-error", "sse-refused"):
        text = f"server-sent events {'refused: ' + str(raw.get('err')) if raw.get('err') else 'dropped; it will retry'}"
    elif event == "draw-error":
        text = f"the {raw.get('where')} panel could not be drawn: {raw.get('err')}"
    elif event == "voice-not-ready":
        text = ("the right trigger was pulled but the microphone is off: leave the headset view "
                "(press the left stick), tap 'Enable voice dictation' on the page and allow it")
    elif event == "screen-stalled":
        text = (f"the PC screen panel froze ({raw.get('quiet')} ms without a frame after "
                f"{raw.get('frames')}); asking for the stream again")
    elif event == "voice-silent":
        text = ("the microphone sent PURE SILENCE (every sample zero) while the trigger was held:\n"
                f"            track={raw.get('track')} muted={raw.get('muted')} secure={raw.get('secure')} "
                f"processing={raw.get('processing')} rate={raw.get('rate')} device={raw.get('label') or '?'}\n"
                "            the browser holds a microphone; the headset is putting no sound into it.\n"
                "              1. Meta button -> Quick Settings -> Microphone: switch it on\n"
                "              2. close anything else using the mic (party chat, casting, recording)\n"
                "              3. press the LEFT STICK to leave the headset view: the page shows a live\n"
                "                 level meter under the voice button - speak and watch it move\n"
                "              4. flat there too? tap 'Enable voice dictation' again and allow it")
    elif event == "voice":
        text = f"microphone {raw.get('state')}" + (f" at {raw.get('rate')} Hz" if raw.get("rate") else "")
        if raw.get("track"):
            text += (f" (track {raw.get('track')}, muted={raw.get('muted')}, "
                     f"processing={raw.get('processing')}, device {raw.get('label') or '?'})")
    elif event == "visibility":
        text = f"XR session visibility: {raw.get('state')}"
    elif event == "ended":
        text = "XR session ended"
    else:
        text = json.dumps(raw)[:300]
    BEACONS.append({"t": time.time(), "ip": ip, "event": event, "text": text,
                    "raw": {k: v for k, v in raw.items() if k not in ("type", "page", "n")}})
    if event == "heartbeat":
        seen, when = LAST_BEAT.get(ip, ("", 0.0))
        if text == seen and time.time() - when < 60.0:
            return                        # nothing has changed; do not fill the terminal
        LAST_BEAT[ip] = (text, time.time())
    print(f"[headset {ip}] {text}", flush=True)
    if event == "page" and raw.get("ua"):
        print(f"[headset {ip}] browser: {str(raw.get('ua'))[:160]}", flush=True)


def to_pointer(payload: bytes) -> None:
    """Where the controller is pointing, on its way to screen_share.py."""
    if PoseSocket.pointer is None:
        return
    try:
        PoseSocket.pointer.sendto(payload, PoseSocket.pointer_addr)
    except OSError:
        pass


def to_voice(payload: bytes) -> None:
    """Microphone data and its start/stop markers, on their way to voice_typer.py."""
    if PoseSocket.voice is None:
        return
    try:
        PoseSocket.voice.sendto(payload, PoseSocket.voice_addr)
    except OSError:
        pass


def handle_client_message(message, remote_ip: str) -> None:
    """One message from the headset page, whichever transport carried it."""
    if isinstance(message, bytes):
        PoseSocket.audio += 1
        to_voice(AUDIO_MAGIC + message)
        return
    try:
        raw = json.loads(message)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return
    if not isinstance(raw, dict):
        return
    if raw.get("type") == "voice":
        to_voice(message.encode() if isinstance(message, str) else message)
        return
    if raw.get("type") == "mouse":
        to_pointer(message.encode() if isinstance(message, str) else message)
        return
    if raw.get("type") == "diag":
        print_diag(raw, remote_ip)
        return
    packet = to_teleop_packet(raw)
    PoseSocket.udp.send_dict(packet)
    PoseSocket.packets += 1
    now = time.time()
    PoseSocket.last_packet = now
    if now - PoseSocket.last_print < 0.5:
        return
    PoseSocket.last_print = now

    def hand(name):
        h = packet[name]
        p = h["pose"]
        where = f"({p[0][3]:6.3f},{p[1][3]:6.3f},{p[2][3]:6.3f})" if h["tracked"] else "(not tracked)       "
        return f"{name[0].upper()} {where} grip={'ON ' if h['button_pressed'] else 'off'} trig={h['trigger']:.2f}"

    ev = packet["ev"]
    print(f"{hand('left')} | {hand('right')} | presses X3={ev['x3']} Y3={ev['y3']} "
          f"A={ev['a']} B={ev['b']}  n={PoseSocket.packets}", flush=True)


def broadcast(text: str) -> None:
    """Status and voice updates to every headset page, on either transport."""
    for client in list(PoseSocket.clients):
        try:
            client.write_message(text)
        except tornado.websocket.WebSocketClosedError:
            PoseSocket.clients.discard(client)
    for client in list(EventsHandler.clients):
        client.push(text)


class EventsHandler(tornado.web.RequestHandler):
    """Server-sent events: the fallback for pages where ws:// is blocked.

    Treating an http:// origin as secure (the chrome://flags allowlist that
    WebXR needs) makes the browser refuse ws:// from that page as insecure
    content, while plain HTTP requests from the same origin still work.
    """

    clients: set = set()

    def check_etag_header(self):
        return False

    async def get(self):
        self.set_header("Content-Type", "text/event-stream")
        self.set_header("Cache-Control", "no-store")
        self.set_header("Connection", "keep-alive")
        self.set_header("X-Accel-Buffering", "no")
        self.stop = False
        EventsHandler.clients.add(self)
        print(f"Headset page connected over HTTP ({self.request.remote_ip})", flush=True)
        self.push(json.dumps({"type": "hello", "diag": True,
                              "screen": screen_fresh(ScreenHandler.frame_file)}))
        try:
            while not self.stop:
                await asyncio.sleep(1.0)
                self.write(": keep-alive\n\n")
                await self.flush()
        except (tornado.iostream.StreamClosedError, tornado.web.Finish):
            pass
        finally:
            EventsHandler.clients.discard(self)
            print(f"Headset page (HTTP) disconnected ({self.request.remote_ip})", flush=True)

    def push(self, text: str) -> None:
        try:
            self.write(f"data: {text}\n\n")
            future = self.flush()
            if future is not None:
                future.add_done_callback(lambda done: done.exception())
        except Exception:
            self.stop = True
            EventsHandler.clients.discard(self)

    def on_connection_close(self):
        self.stop = True


class InputHandler(tornado.web.RequestHandler):
    """Pose packets and voice markers by POST, for the same fallback."""

    def check_etag_header(self):
        return False

    def post(self):
        self.set_header("Cache-Control", "no-store")
        body = self.request.body
        content = self.request.headers.get("Content-Type", "")
        if content.startswith("application/octet-stream"):
            handle_client_message(body, self.request.remote_ip)
        else:
            handle_client_message(body.decode("utf-8", "replace"), self.request.remote_ip)
        self.write(b"")


class DiagHandler(tornado.web.RequestHandler):
    """An image-shaped beacon: the page's last resort for telling us anything.

    When the page cannot open a WebSocket, an EventSource or a fetch, loading an
    image still works, so every report arrives here as a query string.
    """

    def check_etag_header(self):
        return False

    def get(self):
        raw = {"type": "diag"}
        for key, values in self.request.query_arguments.items():
            raw[key] = values[-1].decode("utf-8", "replace")[:400]
        print_diag(raw, self.request.remote_ip)
        self.set_header("Content-Type", "image/gif")
        self.set_header("Cache-Control", "no-store")
        self.write(BLANK_GIF)


class SayHandler(tornado.web.RequestHandler):
    """The last thing Piper synthesised, for the headset to play."""

    def check_etag_header(self):
        return False

    def get(self):
        self.set_header("Cache-Control", "no-store")
        try:
            data = SAY_FILE.read_bytes()
        except OSError:
            self.set_status(404)
            self.write(b"nothing to say")
            return
        self.set_header("Content-Type", "audio/wav")
        self.write(data)


class RobotModelHandler(tornado.web.RequestHandler):
    """The robot's simplified CAD model for the headset (see robot_model.py).

    Built on first use (half a minute) and cached in ~/.cache/bhl; the stamp in
    the JSON changes whenever the URDF or a mesh does, so the page can cache hard.
    """

    files: dict = {}

    def check_etag_header(self):
        return False

    async def get(self, kind: str):
        if not self.files:
            try:
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                import robot_model
                loop = asyncio.get_running_loop()
                json_path, bin_path = await loop.run_in_executor(None, lambda: robot_model.build(quiet=True))
                RobotModelHandler.files = {"json": json_path, "bin": bin_path}
            except Exception as exc:      # no pymeshlab, no meshes: the page falls back to text
                self.set_status(503)
                self.write(f"robot model unavailable: {exc}")
                return
        path = self.files[kind]
        self.set_header("Content-Type", "application/json" if kind == "json" else "application/octet-stream")
        self.set_header("Cache-Control", "no-cache")
        self.write(path.read_bytes())


class SensorsHandler(tornado.web.RequestHandler):
    """The lidar's last rotation and the IMU's last reading (sensor_share.py), for the
    headset's developer view."""

    def check_etag_header(self):
        return False

    def get(self):
        now, out = time.time(), {}
        for key, path in (("lidar", Path("/dev/shm/bhl_lidar.json")), ("imu", Path("/dev/shm/bhl_imu.json"))):
            try:
                data = json.loads(path.read_text())
                data["age"] = round(now - float(data.get("t", 0)), 2)
                out[key] = data
            except (OSError, ValueError):
                out[key] = None
        self.set_header("Content-Type", "application/json")
        self.set_header("Cache-Control", "no-store")
        self.write(json.dumps(out))


class SessionsHandler(tornado.web.RequestHandler):
    """Every Claude session (claude_sessions.panels): what it is about, the conversation
    its terminal shows, whether Claude is working, and what is typed in its prompt."""

    module = None
    loaded = 0.0

    def check_etag_header(self):
        return False

    @classmethod
    def sessions(cls):
        """claude_sessions, read again whenever the file changes: no bridge restart for it."""
        path = Path(__file__).resolve().parent / "claude_sessions.py"
        stamp = path.stat().st_mtime
        if cls.module is None:
            if str(path.parent) not in sys.path:
                sys.path.insert(0, str(path.parent))
            import claude_sessions
            cls.module = claude_sessions
        elif stamp != cls.loaded:
            cls.module = importlib.reload(cls.module)
        cls.loaded = stamp
        return cls.module

    def get(self):
        listing = self.sessions().panels()
        self.set_header("Content-Type", "application/json")
        self.set_header("Cache-Control", "no-store")
        self.write(json.dumps({"sessions": listing}))


class StateHandler(tornado.web.RequestHandler):
    """Everything the bridge knows, for looking at from the PC:

        curl -s localhost:8080/state.json | python3 -m json.tool
    """

    def check_etag_header(self):
        return False

    def get(self):
        now = time.time()
        self.set_header("Content-Type", "application/json")
        self.set_header("Cache-Control", "no-store")
        self.write(json.dumps({
            "now": now,
            "pages_websocket": len(PoseSocket.clients),
            "pages_http": len(EventsHandler.clients),
            "pose_packets": PoseSocket.packets,
            "last_pose_age": (now - PoseSocket.last_packet) if PoseSocket.last_packet else None,
            "last_status_age": (now - StatusRelay.last_status) if StatusRelay.last_status else None,
            "screen_fresh": screen_fresh(ScreenHandler.frame_file),
            "audio_datagrams": PoseSocket.audio,
            "last_status": json.loads(StatusRelay.last_text) if StatusRelay.last_text else None,
            "beacons": list(BEACONS)[-25:],
        }, default=str))


class ScreenHandler(tornado.web.RequestHandler):
    """Stream screen_share.py's frames to the headset as multipart JPEG.

    An MJPEG response never ends by itself, and a browser only allows a handful
    of connections per host, so a forgotten stream from an old tab can block the
    page from loading at all. Only the newest viewer streams.

    A response that ends is invisible to the page: an <img> keeps showing the last
    frame it was given, with no error and no event, so the headset panel freezes on
    a stale terminal. This therefore keeps the stream open as long as the helper is
    running, and repeats the last frame every couple of seconds when the desktop has
    nothing new, which is also how the page can tell a live stream from a dead one.
    """

    frame_file = SCREEN_FILE
    active = None
    MAX_STREAM = 900.0          # a backstop against a leaked response, not a routine cut

    def check_etag_header(self):
        return False

    def on_connection_close(self):
        self.stop = True

    async def get(self):
        if not screen_fresh(self.frame_file):
            self.set_status(503)
            self.set_header("Cache-Control", "no-store")
            self.write(b"no screen share running")
            return
        self.set_header("Content-Type", "multipart/x-mixed-replace; boundary=bhlframe")
        self.set_header("Cache-Control", "no-store")
        self.stop = False
        cls = type(self)
        previous = cls.active
        if previous is not None and previous is not self:
            previous.stop = True  # the newest viewer wins; old tabs let go
            print(f"{cls.__name__}: a newer viewer took over ({self.request.remote_ip})", flush=True)
        cls.active = self
        last = 0.0
        sent_at = 0.0
        deadline = time.monotonic() + self.MAX_STREAM
        while not self.stop and time.monotonic() < deadline:
            try:
                changed = os.stat(self.frame_file).st_mtime
            except OSError:
                return
            now = time.monotonic()
            fresh_frame = changed != last
            if not fresh_frame and now - sent_at <= 2.0:
                await asyncio.sleep(0.05)
                continue
            if not fresh_frame and not screen_fresh(self.frame_file, 10.0):
                return          # the helper stopped; let the page fall back to the log panel
            last = changed
            sent_at = now
            if self.stop:
                return
            try:
                data = self.frame_file.read_bytes()
            except OSError:
                return
            if data[:2] == b"\xff\xd8":
                self.write(b"--bhlframe\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
                try:
                    await self.flush()
                except tornado.iostream.StreamClosedError:
                    return
            await asyncio.sleep(0.05)


class CameraHandler(ScreenHandler):
    """The robot's stereo camera, streamed exactly like the PC screen."""

    frame_file = CAMERA_FILE
    active = None


class Screen2Handler(ScreenHandler):
    """The robot's own screen (screen 2), for "agent, screen 2" in the headset.

    The desktop runs on Xorg, so its region can be captured directly with ffmpeg's
    x11grab - no portal, no Share dialog. The capture runs only while someone watches,
    and stops 20 s after the last viewer leaves.
    """

    frame_file = Path("/dev/shm/bhl_screen2.jpg")
    active = None
    capture = None
    watching = 0
    last_seen = 0.0

    @classmethod
    def ensure_capture(cls) -> None:
        if cls.capture is not None and cls.capture.poll() is None:
            return
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from face_display import robot_screen
            x, y, w, h = robot_screen()
        except Exception:
            x, y, w, h = 1920, 0, 1024, 600
        display = os.environ.get("DISPLAY", ":1")
        cls.capture = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "x11grab", "-framerate", "8", "-video_size", f"{w}x{h}", "-i", f"{display}+{x},{y}",
             "-q:v", "6", "-f", "image2", "-update", "1", "-atomic_writing", "1", str(cls.frame_file)],
            stdin=subprocess.DEVNULL)
        print(f"Screen 2: capturing {w}x{h} at +{x},{y} for the headset", flush=True)

    @classmethod
    def idle_check(cls) -> None:
        if cls.capture is not None and cls.capture.poll() is None and not cls.watching \
                and time.time() - cls.last_seen > 20.0:
            cls.capture.terminate()
            cls.capture = None
            print("Screen 2: nobody watching, capture stopped", flush=True)

    async def get(self):
        cls = type(self)
        cls.ensure_capture()
        for _ in range(30):                       # its first frame, within 3 s
            if screen_fresh(self.frame_file):
                break
            await asyncio.sleep(0.1)
        cls.watching += 1
        cls.last_seen = time.time()
        try:
            await super().get()
        finally:
            cls.watching -= 1
            cls.last_seen = time.time()


class VisionHandler(tornado.web.RequestHandler):
    """Both camera eyes with the person tracker's boxes drawn on them (/vision.jpg).

    One picture per request, not a stream: the page asks about three times a second.
    The headset already holds three never-ending responses open (events, camera, PC
    screen) out of the handful a browser allows per host, and the arm poses travel as
    POSTs through what is left. Each request also tells person_track.py someone is
    watching (a marker file it checks): it only runs the costly pose model then.
    """

    frame_file = Path("/dev/shm/bhl_vision.jpg")
    wanted_file = Path("/dev/shm/bhl_vision.want")
    missing = b"no vision picture yet: is person_track.py running?"

    def check_etag_header(self):
        return False

    def get(self):
        try:
            self.wanted_file.touch()
        except OSError:
            pass
        self.set_header("Cache-Control", "no-store")
        data = self.frame_file.read_bytes() if screen_fresh(self.frame_file) else b""
        if data[:2] != b"\xff\xd8":
            self.set_status(503)                  # the tracker starts drawing within a second
            self.write(self.missing)
            return
        self.set_header("Content-Type", "image/jpeg")
        self.write(data)


class DepthFlowHandler(VisionHandler):
    """The same pair after A x2: depth over optical flow for each eye (/depthflow.jpg).

    depth_flow.py runs its two models only while these requests keep its marker fresh.
    """

    frame_file = Path("/dev/shm/bhl_depthflow.jpg")
    wanted_file = Path("/dev/shm/bhl_depthflow.want")
    missing = b"no depth picture yet: is depth_flow.py running?"


def screen_fresh(path: Path, max_age: float = 3.0) -> bool:
    try:
        return (time.time() - path.stat().st_mtime) < max_age
    except OSError:
        return False


class TlsRefusalHint(logging.Filter):
    """Replace tornado's one-line-per-retry "SSL Error" spam with a hint per headset.

    A browser tab still on the self-signed https:// page (for example restored
    after a browser relaunch, which forgets certificate exceptions) retries its
    WebSocket every second and refuses the certificate each time.
    """

    def __init__(self, http_url: str, https_url: str):
        super().__init__()
        self.http_url = http_url
        self.https_url = https_url
        self.last = -math.inf
        self.tries = 0
        self.host = "a device"

    def filter(self, record: logging.LogRecord) -> bool:
        if not str(record.msg).startswith("SSL Error on"):
            return True
        args = record.args if isinstance(record.args, tuple) else ()
        peer = args[1] if len(args) > 1 else None
        if isinstance(peer, tuple) and peer:
            self.host = str(peer[0])  # otherwise it hung up before we could ask
        reason = str(args[2]) if len(args) > 2 else ""
        self.tries += 1
        now = time.monotonic()
        if now - self.last < 30.0:
            return False
        self.last = now
        if "CERTIFICATE_UNKNOWN" in reason:
            why = "is refusing the self-signed certificate"
        elif "HTTP_REQUEST" in reason:
            why = "sent plain http:// to the HTTPS port"
        else:
            why = f"failed the TLS handshake ({reason})"
        tries = f"{self.tries} {'try' if self.tries == 1 else 'tries'} so far"
        print(f"[https] {self.host} {why} on {self.https_url} ({tries}).\n"
              f"        A tab on the https:// page keeps retrying. In the headset, close that tab and open\n"
              f"        {self.http_url} (type the http:// part). Browser relaunches forget accepted certificates.",
              flush=True)
        return False


class PoseSocket(tornado.websocket.WebSocketHandler):
    udp: UDP
    voice: socket.socket | None = None
    voice_addr = ("127.0.0.1", 11007)
    pointer: socket.socket | None = None
    pointer_addr = ("127.0.0.1", 11008)
    clients: set = set()
    last_print = 0.0
    packets = 0
    last_packet = 0.0
    audio = 0

    def check_origin(self, origin):
        return True

    def open(self):
        PoseSocket.clients.add(self)
        # Tells the page it may send diagnostics; an older bridge would forward them as poses.
        self.write_message(json.dumps({"type": "hello", "diag": True,
                                       "screen": screen_fresh(ScreenHandler.frame_file)}))
        print(f"Headset page connected ({self.request.remote_ip})", flush=True)

    def on_close(self):
        PoseSocket.clients.discard(self)
        print("Headset page disconnected", flush=True)

    def on_message(self, message):
        handle_client_message(message, self.request.remote_ip)


class StatusRelay:
    """run_teleop status (UDP, localhost) -> every connected headset page."""

    last_status = 0.0
    last_text = ""      # the exact payload the pages were last sent

    def __init__(self, port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.sock.setblocking(False)
        tornado.ioloop.IOLoop.current().add_handler(
            self.sock.fileno(), self._on_readable, tornado.ioloop.IOLoop.READ)

    def _on_readable(self, fd, events):
        while True:
            try:
                data = self.sock.recv(65536)
            except (BlockingIOError, InterruptedError):
                return
            except OSError:
                return
            text = data.decode("utf-8", "replace")
            # person_track.py and voice_typer.py share this relay: only run_teleop's own
            # status counts as "the last status" (state.json, bringup status)
            if '"type": "status"' in text[:40]:
                StatusRelay.last_status = time.time()
                StatusRelay.last_text = text
            broadcast(text)


class SpeechRelay:
    """Text to be spoken (UDP, localhost) -> every headset page, which says it aloud.

    The PC has no sound card, so it cannot speak; the headset has speakers and a
    speech synthesiser in its browser. Sending the words rather than audio keeps
    this to one small datagram and costs nothing on the PC.
    """

    def __init__(self, port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.sock.setblocking(False)
        tornado.ioloop.IOLoop.current().add_handler(
            self.sock.fileno(), self._on_readable, tornado.ioloop.IOLoop.READ)

    def _on_readable(self, fd, events):
        while True:
            try:
                data = self.sock.recv(65536)
            except (BlockingIOError, InterruptedError):
                return
            except OSError:
                return
            text = data.decode("utf-8", "replace").strip()
            if not text:
                continue
            # Piper writes the audio next to this; the page fetches it rather than
            # having it pushed, because a datagram cannot carry a few seconds of speech.
            said = {"type": "say", "text": text[:1200]}
            if screen_fresh(SAY_FILE, 30.0):
                said["url"] = f"/say.wav?t={int(SAY_FILE.stat().st_mtime * 1000)}"
            broadcast(json.dumps(said))
            print(f"Speaking to the headset: {text[:70]}"
                  + ("..." if len(text) > 70 else ""), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quest 2 WebXR bridge for BHL teleop")
    parser.add_argument("--port", type=int, default=8443, help="HTTPS port (self-signed)")
    parser.add_argument("--http-port", type=int, default=8080)
    parser.add_argument("--udp-host", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=11005, help="run_teleop listens here")
    parser.add_argument("--status-port", type=int, default=11006, help="run_teleop status arrives here")
    parser.add_argument("--https", action="store_true",
                        help="also serve the self-signed HTTPS port (off by default: "
                             "its certificate is refused and old https bookmarks then hang)")
    parser.add_argument("--http-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--screen-file", type=Path, default=SCREEN_FILE,
                        help="JPEG frames from screen_share.py")
    parser.add_argument("--voice-port", type=int, default=11007,
                        help="microphone audio goes here for voice_typer.py")
    parser.add_argument("--pointer-port", type=int, default=11008,
                        help="where the controller points goes here for screen_share.py")
    parser.add_argument("--say-port", type=int, default=11009,
                        help="text sent here is spoken aloud in the headset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not HTML_PATH.exists():
        sys.exit(f"missing {HTML_PATH}")

    ip = local_ip()
    IndexHandler.http_port = args.http_port
    ScreenHandler.frame_file = args.screen_file
    PoseSocket.udp = UDP(recv_addr=None, send_addr=(args.udp_host, args.udp_port))
    PoseSocket.voice = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    PoseSocket.voice_addr = ("127.0.0.1", args.voice_port)
    PoseSocket.pointer = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    PoseSocket.pointer_addr = ("127.0.0.1", args.pointer_port)
    app = tornado.web.Application(
        [
            (r"/", IndexHandler),
            (r"/ws", PoseSocket),
            (r"/events", EventsHandler),
            (r"/input", InputHandler),
            (r"/screen.mjpg", ScreenHandler),
            (r"/camera.mjpg", CameraHandler),
            (r"/screen2.mjpg", Screen2Handler),
            (r"/vision.jpg", VisionHandler),
            (r"/depthflow.jpg", DepthFlowHandler),
            (r"/sensors.json", SensorsHandler),
            (r"/sessions.json", SessionsHandler),
            (r"/diag", DiagHandler),
            (r"/state.json", StateHandler),
            (r"/robot_model\.(json|bin)", RobotModelHandler),
            (r"/say.wav", SayHandler),
        ],
        websocket_ping_interval=5,
    )

    app.listen(args.http_port, address="0.0.0.0")
    if args.https:
        cert, key = ensure_certs(ip)
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(cert, key)
        app.listen(args.port, address="0.0.0.0", ssl_options=ssl_ctx)
    StatusRelay(args.status_port)
    tornado.ioloop.PeriodicCallback(Screen2Handler.idle_check, 5000).start()
    SpeechRelay(args.say_port)

    http_url = f"http://{ip}:{args.http_port}"
    if args.https:
        logging.getLogger("tornado.general").addFilter(TlsRefusalHint(http_url, f"https://{ip}:{args.port}"))
    print(f"Quest bridge: page on {http_url}"
          + (f" and https://{ip}:{args.port}" if args.https else ""), flush=True)
    print(f"  UDP -> {args.udp_host}:{args.udp_port} (run_teleop), status <- 127.0.0.1:{args.status_port}", flush=True)
    print("", flush=True)
    print("Open the page in Quest Browser. For passthrough (seeing the room) the page must be", flush=True)
    print("a secure origin. Pick one:", flush=True)
    print("  1) LAN, no certificate (recommended, one-time setup in the headset):", flush=True)
    print("       Quest Browser -> new tab -> chrome://flags", flush=True)
    print("       search 'insecure origins', in 'Insecure origins treated as secure'", flush=True)
    print(f"       enter {http_url} , set Enabled, tap Relaunch", flush=True)
    print(f"       then open {http_url}", flush=True)
    if maybe_adb_reverse(args.http_port):
        print(f"  2) USB: adb reverse is set -> open http://localhost:{args.http_port}", flush=True)
    else:
        print(f"  2) USB cable + developer mode: adb reverse tcp:{args.http_port} tcp:{args.http_port}", flush=True)
        print(f"       then http://localhost:{args.http_port}   (adb: sudo apt install adb)", flush=True)
    if args.https:
        print(f"  3) https://{ip}:{args.port} (Advanced -> Proceed). Tracking works;", flush=True)
        print("       passthrough is often refused on a self-signed certificate.", flush=True)
    else:
        print("  Note: the HTTPS port is off. If the headset autocompletes to", flush=True)
        print(f"  https://{ip}:{args.port} it will fail fast instead of hanging on the certificate.", flush=True)
        print(f"  Type the whole address: {http_url}   (--https brings that port back)", flush=True)
    print("", flush=True)
    if screen_fresh(args.screen_file):
        print("PC screen: sharing is live; it appears on the second panel in the headset.", flush=True)
    else:
        print("PC screen in the headset (optional): run in another terminal", flush=True)
        print("  .venv/bin/python scripts/teleop/screen_share.py    then click Share in the dialog", flush=True)
        print("  Without it, that panel shows the last lines of run_teleop's output instead.", flush=True)
    print(f"Dictation (optional): hold the right trigger and speak. Audio goes to "
          f"127.0.0.1:{args.voice_port}; run scripts/teleop/voice_typer.py to type it into tmux.", flush=True)
    print("", flush=True)
    print("Headset: triple X = arm/stop, grip = move that arm, triple Y = power calibration,", flush=True)
    print("A/B = answer prompts. Left stick moves the panels, right stick press re-centres them.", flush=True)
    print("Ctrl+C to stop.", flush=True)
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped.", flush=True)
