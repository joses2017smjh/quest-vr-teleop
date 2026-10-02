#!/usr/bin/env python3
"""Share this PC's screen into the headset page.

Wayland will not let a program grab the screen on its own: x11grab returns
black and GNOME refuses the screenshot API. The desktop portal is the way in,
and it asks you once, in a dialog, which screen or window to share.

  .venv/bin/python scripts/teleop/screen_share.py          # pick a screen, click Share
  .venv/bin/python scripts/teleop/screen_share.py --test   # colour bars, no dialog

Frames land in /dev/shm/bhl_screen.jpg, which quest_bridge.py streams to the
headset at /screen.mjpg. Leave this running next to the bridge.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

try:
    import gi
except ImportError:  # PyGObject and dbus live in the system packages, not the venv
    sys.path.append("/usr/lib/python3/dist-packages")
    try:
        import gi
    except ImportError:
        raise SystemExit(
            "This needs the system GStreamer bindings:\n"
            "  sudo apt install python3-gi python3-dbus gstreamer1.0-pipewire "
            "gstreamer1.0-plugins-good")

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

DEFAULT_FRAME = Path("/dev/shm/bhl_screen.jpg")
BTN_LEFT = 0x110                  # linux/input-event-codes.h, what the portal expects


class PointerRelay:
    """Where the headset is pointing (0..1 across the shared panel) -> the real pointer.

    The portal's calls have to happen on the thread its main loop runs on, so the
    socket thread only parses and hands over.
    """

    def __init__(self, portal: "PortalScreenCast", port: int, size):
        self.portal = portal
        self.size = size
        self.down = False
        self.complained = False
        self.running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.sock.settimeout(0.5)
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        while self.running:
            try:
                data, _ = self.sock.recvfrom(4096)
                message = json.loads(data.decode("utf-8"))
            except socket.timeout:
                continue
            except (OSError, UnicodeDecodeError, ValueError):
                continue
            if isinstance(message, dict):
                GLib.idle_add(self._apply, message)

    def _apply(self, message: dict) -> bool:
        try:
            if message.get("action") == "off":
                if self.down:
                    self.portal.click(False)
                    self.down = False
                return False
            if "u" in message and "v" in message:
                width, height = self.size
                self.portal.move_to(float(message["u"]) * width, float(message["v"]) * height)
            down = bool(message.get("down"))
            if down != self.down:
                self.portal.click(down)
                self.down = down
            steps = int(message.get("wheel") or 0)
            if steps:
                self.portal.scroll(steps)
        except Exception as exc:                 # one complaint, not one per packet
            if not self.complained:
                self.complained = True
                print(f"Pointer: the portal refused it ({str(exc).splitlines()[0][:100]})", flush=True)
        return False

    def close(self) -> None:
        self.running = False
        try:
            self.sock.close()
        except OSError:
            pass


class FrameWriter:
    """Latest JPEG on disk, replaced atomically so a reader never sees half of one."""

    def __init__(self, path: Path, fps: float = 8.0):
        self.path = path
        self.count = 0
        self.skipped = 0
        self.last_report = 0.0
        self.last_frame = 0.0
        self.min_interval = 1.0 / max(1.0, fps)
        self.last_data = b""

    def keepalive(self) -> bool:
        """GNOME only sends a frame when something on screen changes. A still desktop
        then looks exactly like a dead stream to the headset, which dropped it - the
        screen only showed while something was moving. Rewrite the last frame each
        second instead. (A GLib timeout: returns True to keep running.)"""
        if self.last_data and time.monotonic() - self.last_frame > 1.0:
            self._put(self.last_data)
            self.last_frame = time.monotonic()
        return True

    def write(self, data: bytes) -> None:
        now = time.monotonic()
        if now - self.last_frame < self.min_interval:
            self.skipped += 1
            return
        self.last_frame = now
        self.last_data = data
        self._put(data)
        self.count += 1
        if now - self.last_report >= 5.0:
            print(f"  {self.count} frames shared, {len(data) // 1024} kB each", flush=True)
            self.last_report = now

    def _put(self, data: bytes) -> None:
        directory = self.path.parent
        fd, tmp = tempfile.mkstemp(dir=str(directory), prefix=".bhl_screen", suffix=".jpg")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.replace(tmp, self.path)
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


class PortalScreenCast:
    """One portal session: the screen going out, and optionally the pointer coming in.

    Wayland will not let anything synthesise input - no XTEST, and /dev/uinput is
    root-only - but the desktop portal will, through RemoteDesktop, and its absolute
    motion is expressed in the coordinates of a ScreenCast stream. That is exactly
    what a hit on the headset's screen panel gives, so both live on one session and
    one dialog. If remote control is refused, the screen still shares on its own.
    """

    def __init__(self, want_pointer: bool = True):
        import dbus
        from dbus.mainloop.glib import DBusGMainLoop

        DBusGMainLoop(set_as_default=True)
        self.dbus = dbus
        self.bus = dbus.SessionBus()
        portal = self.bus.get_object("org.freedesktop.portal.Desktop", "/org/freedesktop/portal/desktop")
        self.screencast = dbus.Interface(portal, "org.freedesktop.portal.ScreenCast")
        self.remote = dbus.Interface(portal, "org.freedesktop.portal.RemoteDesktop") if want_pointer else None
        self.sender = self.bus.get_unique_name()[1:].replace(".", "_")
        self.loop = GLib.MainLoop()
        self.pointing = False

        owner = self.remote or self.screencast          # whoever owns the session starts it
        try:
            results = self._request(owner.CreateSession, {"session_handle_token": self._token()})
            self.session = results["session_handle"]
            if self.remote is not None:
                self._request(self.remote.SelectDevices,
                              {"types": dbus.UInt32(1 | 2)}, self.session)   # keyboard, pointer
        except Exception as exc:                        # no RemoteDesktop here: share anyway
            if self.remote is None:
                raise
            print(f"  remote control is not available ({str(exc).splitlines()[0][:80]}); "
                  "sharing the screen only.", flush=True)
            self.remote = None
            owner = self.screencast
            results = self._request(owner.CreateSession, {"session_handle_token": self._token()})
            self.session = results["session_handle"]
        print("A dialog is open on the PC screen: choose a screen or window, then click Share.", flush=True)
        if self.remote is not None:
            print("  it also asks to allow remote control - that is the headset pointer.", flush=True)
        self._request(self.screencast.SelectSources, {
            "types": dbus.UInt32(1 | 2),    # monitor or window
            "multiple": dbus.Boolean(False),
            "cursor_mode": dbus.UInt32(2),  # draw the pointer into the picture
        }, self.session)
        results = self._request(owner.Start, {}, self.session, "")
        self.pointing = self.remote is not None
        streams = results.get("streams") or []
        if not streams:
            raise SystemExit("The portal returned no stream.")
        self.node_id = int(streams[0][0])
        properties = dict(streams[0][1]) if len(streams[0]) > 1 else {}
        size = properties.get("size")
        self.source_size = tuple(int(v) for v in size) if size else None

    def _token(self) -> str:
        return f"bhl{random.randint(0, 2**31)}"

    def _request(self, method, options, *args):
        token = self._token()
        path = f"/org/freedesktop/portal/desktop/request/{self.sender}/{token}"
        options["handle_token"] = token
        result: dict = {}

        def on_response(code, results):
            result["code"] = int(code)
            result["results"] = results
            self.loop.quit()

        match = self.bus.add_signal_receiver(
            on_response, "Response", "org.freedesktop.portal.Request", None, path)
        method(*args, options)
        self.loop.run()
        match.remove()
        if result.get("code") != 0:
            raise SystemExit("The screen share was cancelled in the dialog.")
        return result["results"]

    def move_to(self, x: float, y: float) -> None:
        self.remote.NotifyPointerMotionAbsolute(
            self.session, {}, self.dbus.UInt32(self.node_id), float(x), float(y))

    def click(self, down: bool) -> None:
        self.remote.NotifyPointerButton(
            self.session, {}, self.dbus.Int32(BTN_LEFT), self.dbus.UInt32(1 if down else 0))

    def scroll(self, steps: int) -> None:
        self.remote.NotifyPointerAxisDiscrete(
            self.session, {}, self.dbus.UInt32(0), self.dbus.Int32(int(steps)))

    def open_fd(self) -> int:
        fd = self.screencast.OpenPipeWireRemote(
            self.session, {}, dbus_interface="org.freedesktop.portal.ScreenCast")
        return fd.take()


def scaled_caps(args, source_size) -> str:
    """Caps for the JPEG stage.

    videoscale only preserves shape if the height is given too: pinning the width
    alone leaves the height as it was (1920x1080 arrived as 1280x1080, squashed).
    """
    if source_size and source_size[0] > 0 and source_size[1] > 0:
        height = max(2, int(round(args.width * source_size[1] / source_size[0])) // 2 * 2)
        return (f"video/x-raw,format=I420,width={args.width},height={height},"
                "pixel-aspect-ratio=1/1")
    return f"video/x-raw,format=I420,width={args.width},pixel-aspect-ratio=1/1"


def pipeline_variants(source: str, args, source_size=None) -> list[str]:
    """PipeWire screen casts negotiate differently across GNOME/PipeWire versions.

    `! video/x-raw` straight after the source asks for plain system memory: the
    compositor otherwise offers GPU buffers that videoconvert cannot take, which
    shows up as "Internal data stream error". No videorate anywhere: with a
    variable-rate screen source it trips an assertion and aborts the process, so
    the frame rate is limited when frames are written instead.
    """
    tail = f"! jpegenc quality={args.quality} ! appsink name=out emit-signals=true max-buffers=1 drop=true"
    caps = scaled_caps(args, source_size)
    return [
        f"{source} ! video/x-raw ! videoconvert ! videoscale ! {caps} {tail}",
        f"{source} ! videoconvert ! videoscale ! {caps} {tail}",
        f"{source} ! glupload ! glcolorconvert ! gldownload ! videoconvert ! videoscale ! {caps} {tail}",
        f"{source} ! video/x-raw ! videoconvert ! video/x-raw,format=I420,pixel-aspect-ratio=1/1 {tail}",
        f"{source} ! videoconvert ! video/x-raw,format=I420,pixel-aspect-ratio=1/1 {tail}",
    ]


def start_pipeline(description: str, writer: FrameWriter, timeout: float = 6.0):
    """Run one pipeline until it produces a frame. Returns (pipeline, caps) or (None, why)."""
    try:
        pipeline = Gst.parse_launch(description)
    except GLib.Error as exc:
        return None, f"could not build: {exc}"
    sink = pipeline.get_by_name("out")
    first = threading.Event()

    def on_sample(appsink):
        sample = appsink.emit("pull-sample")
        if sample is not None:
            buffer = sample.get_buffer()
            ok, info = buffer.map(Gst.MapFlags.READ)
            if ok:
                try:
                    writer.write(bytes(info.data))
                finally:
                    buffer.unmap(info)
                first.set()
        return Gst.FlowReturn.OK

    sink.connect("new-sample", on_sample)
    bus = pipeline.get_bus()
    pipeline.set_state(Gst.State.PLAYING)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not first.is_set():
        message = bus.timed_pop_filtered(
            100 * Gst.MSECOND, Gst.MessageType.ERROR | Gst.MessageType.EOS)
        if message is None:
            continue
        if message.type == Gst.MessageType.ERROR:
            error, _debug = message.parse_error()
            pipeline.set_state(Gst.State.NULL)
            return None, str(error)
        pipeline.set_state(Gst.State.NULL)
        return None, "stream ended before the first frame"
    if not first.is_set():
        pipeline.set_state(Gst.State.NULL)
        return None, "no frame arrived"
    pad = sink.get_static_pad("sink")
    caps = pad.get_current_caps()
    return pipeline, (caps.to_string() if caps else "unknown caps")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame-file", type=Path, default=DEFAULT_FRAME)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--width", type=int, default=1280, help="frames are scaled to this width")
    parser.add_argument("--quality", type=int, default=55)
    parser.add_argument("--test", action="store_true", help="colour bars instead of the real screen")
    parser.add_argument("--pointer-port", type=int, default=11008,
                        help="where quest_bridge.py sends what the headset is pointing at")
    parser.add_argument("--no-pointer", action="store_true",
                        help="share the screen without asking for remote control")
    args = parser.parse_args()

    Gst.init(None)
    writer = FrameWriter(args.frame_file, fps=args.fps)
    portal = None

    def source_for_attempt() -> str:
        if args.test:
            return "videotestsrc is-live=true ! video/x-raw,width=1920,height=1080 ! queue"
        fd = portal.open_fd()
        return f"pipewiresrc fd={fd} path={portal.node_id}"

    source_size = (1920, 1080) if args.test else None
    if not args.test:
        portal = PortalScreenCast(want_pointer=not args.no_pointer)
        source_size = portal.source_size
        if portal.source_size:
            print(f"Sharing a {portal.source_size[0]}x{portal.source_size[1]} source.", flush=True)

    pipeline = None
    for index, description in enumerate(pipeline_variants("SOURCE", args, source_size), start=1):
        attempt = description.replace("SOURCE", source_for_attempt())
        pipeline, detail = start_pipeline(attempt, writer)
        if pipeline is not None:
            print(f"Pipeline {index} works: {detail}", flush=True)
            break
        print(f"  pipeline {index} did not negotiate ({detail})", flush=True)
    if pipeline is None:
        print("None of the pipelines worked. Send me these lines and run, for more detail:", flush=True)
        print("  GST_DEBUG=pipewiresrc:5,videoconvert:4 .venv/bin/python scripts/teleop/screen_share.py", flush=True)
        return 1

    loop = GLib.MainLoop()

    def on_message(_bus, message):
        if message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            print(f"GStreamer error: {error} ({debug})", flush=True)
            loop.quit()
        elif message.type == Gst.MessageType.EOS:
            print("Screen share ended (the stream stopped).", flush=True)
            loop.quit()

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message)
    GLib.timeout_add(1000, writer.keepalive)
    restart = []
    if portal is not None:
        # Rearranging the monitors ends GNOME's capture without a word: the picture
        # froze for an hour on 29 Sep. Start over instead (the window running this
        # restarts it; GNOME asks for Share once more).
        def monitors_changed(*_):
            print("The monitor layout changed: restarting the screen share.", flush=True)
            restart.append(True)
            loop.quit()

        portal.bus.add_signal_receiver(monitors_changed, "MonitorsChanged",
                                       "org.gnome.Mutter.DisplayConfig", None,
                                       "/org/gnome/Mutter/DisplayConfig")
    print(f"Sharing to {args.frame_file} at up to {args.fps} fps, {args.width}px wide.", flush=True)
    print("The headset page shows it once quest_bridge.py is running. Ctrl+C to stop.", flush=True)

    pointer = None
    if portal is not None and portal.pointing and source_size:
        pointer = PointerRelay(portal, args.pointer_port, source_size)
        print(f"Pointer: on. In the headset, B twice turns it on, the trigger clicks, "
              f"the right stick scrolls. ({source_size[0]}x{source_size[1]})", flush=True)
    elif portal is not None and not args.no_pointer:
        print("Pointer: off - the portal did not grant remote control"
              + ("" if source_size else " (and the stream size is unknown)") + ".", flush=True)

    for sig in (signal.SIGINT, signal.SIGTERM):
        GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, sig, lambda *_: (loop.quit(), True)[1])
    try:
        loop.run()
    finally:
        if pointer is not None:
            pointer.close()
        pipeline.set_state(Gst.State.NULL)
        try:
            args.frame_file.unlink()
        except OSError:
            pass
        print("Stopped sharing.", flush=True)
    return 3 if restart else 0


if __name__ == "__main__":
    sys.exit(main())
