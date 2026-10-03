#!/usr/bin/env python3
"""Record learning-from-demonstration episodes on the robot PC: the arms' rows and the camera's frames.

run_teleop.py --record-tap sends one JSON row per control cycle to UDP 11014, and
camera_share.py keeps the newest stereo frame in /dev/shm/bhl_camera.jpg. Between a
`start` and an `end`, this copies both into an episode folder, on one clock
(time.monotonic()). The JPEG bytes are copied as they are and never decoded, so the N150
pays almost nothing for it. The format is docs/LFD_RECORDING_FORMAT.md; the converter on
the HPC reads it.

  .venv/bin/python scripts/teleop/run_teleop.py --record-tap      # the arms, plus the tap
  .venv/bin/python scripts/teleop/recorder.py --task "Put the foam block in the bowl"
  .venv/bin/python tools/lfd/record_ctl.py start --mode teleop    # or "agent, start episode"
  .venv/bin/python tools/lfd/record_ctl.py end --outcome success  # or "agent, end episode success"
  .venv/bin/python tools/lfd/record_ctl.py status

Turning the tap on means restarting run_teleop.py, and a restart takes zero again. Follow
CLAUDE.md first: restart it only when it is STOPPED with the motors off and the arms hang
still at their zero (curl -s localhost:8080/state.json: every q near 0, unchanged over a
couple of seconds), with Ctrl+C twice and then the explicit command above, never "Up
Enter". Zero taken with an arm bent makes every recorded angle wrong.

Episodes land in ~/bhl_recordings/<session>/ep_0000/, one session per run of this program.
A start is refused unless tap rows and frames are both arriving, unless the camera holds
still ("agent, camera still"), because a moving camera changes the viewpoint between
episodes, and while less than --min-free-gb (2 GB) is free, which keeps room for
run_teleop.py's teleop log on the same disk. An open episode ends by itself, as aborted, if
either stream stops for 2 s, the zero is taken again, or free space falls below that floor
("low disk"). Ctrl+C, a kill or a closed terminal (SIGHUP, unless started under nohup) ends
it the same way.

`record_ctl.py prime --mode policy --trial 17 ...` sets fields for the next start only, so
"agent, start episode" can open an evaluation trial; the start's own fields win, and a prime
not used within 10 minutes expires. An `end` is answered at once ("saving"), then the
episode is synced to disk before any other command is read.

It only listens: it sends nothing to run_teleop.py, the CAN buses or the servos.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import select
import shutil
import signal
import socket
import subprocess
import time
import traceback
from pathlib import Path

from arm_power import ARM_JOINTS, PROFILE_PATH   # the joint order every row's vectors follow
from camera_share import FRAME_FILE

REPO_ROOT = Path(__file__).resolve().parents[2]
TAP_PORT = 11014        # run_teleop.py --record-tap -> here: one row per control cycle
CONTROL_PORT = 11015    # tools/lfd/record_ctl.py and "agent, start episode" -> here
LIVE_S = 0.5            # a start needs a tap row and a frame no older than this
STALL_S = 2.0           # an open episode is aborted when either stream stops for longer
ROW_GAP_S = 0.050       # counted per episode: control-loop stalls (rows come every ~20 ms)...
FRAME_GAP_S = 0.067     # ...and camera hiccups (frames come every ~33 ms)
LOW_DISK_BYTES = 5e9    # about 45 minutes of 1280x480 frames
MIN_FREE_GB = 2.0       # --min-free-gb: below it starts are refused and an open episode ends, before
                        # the disk fills: run_teleop.py's teleop log is on it, and a full one stops teleop
DISK_CHECK_S = 2.0      # free space is checked this often (a statvfs: microseconds)
PRIME_S = 600.0         # a prime unused this long expires: it must not label a later teleop demonstration
PRIME_FIELDS = ("mode", "trial", "arrangement", "policy", "task")
MODES = ("teleop", "guided", "policy")
OUTCOMES = ("success", "failure", "aborted")


def log(text: str) -> None:
    """A line on the terminal. After a hang-up (SIGHUP) there is none, and that must not stop a save."""
    try:
        print(text, flush=True)
    except OSError:
        pass


def write_json(path: Path, data: dict) -> None:
    """Synced to disk before it replaces the old file: a crash leaves one or the other, whole."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    sync_dir(path.parent)


def sync_dir(folder: Path) -> None:
    """A new or renamed file survives a power cut only once its folder is synced too."""
    try:
        fd = os.open(folder, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def read_json(path: Path):
    """The file's JSON, or None when it is missing or unreadable."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def jpeg_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) from a JPEG's frame header. Only the header is read: nothing is decoded."""
    if data[:2] != b"\xff\xd8":
        return None
    i = 2
    while i + 9 <= len(data):
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        if marker == 0xFF:                                   # fill byte before a marker
            i += 1
            continue
        if marker == 0x01 or 0xD0 <= marker <= 0xD9:         # markers without a length
            i += 2
            continue
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):   # a start-of-frame
            return int.from_bytes(data[i + 7:i + 9], "big"), int.from_bytes(data[i + 5:i + 7], "big")
        if marker == 0xDA:                                   # the image data began without one
            return None
        i += 2 + int.from_bytes(data[i + 2:i + 4], "big")
    return None


def calibration_for(size: tuple[int, int] | None) -> str | None:
    """The stereo calibration for this frame size (roadmap 3.2), if one has been made."""
    if size is None:
        return None
    name = f"configs/stereo_{size[0]}x{size[1]}.json"
    return name if (REPO_ROOT / name).exists() else None


def repo_commit() -> str | None:
    try:
        done = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                              timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    commit = done.stdout.strip()
    return commit if done.returncode == 0 and commit else None


def claw_limits(path: Path) -> tuple[dict, list[str]]:
    """The claws' open and closed pulses ({"left": ..., "right": ...}), the converter's 0..1 scale.

    configs/claw_limits.json keeps them per Nano pin: D3 drives the robot's left claw, D7 its right.
    """
    claws = (read_json(path) or {}).get("claws") or {}
    limits, warnings = {}, []
    for key, side in (("D3", "left"), ("D7", "right")):
        entry = claws.get(key) or {}
        if "open_us" in entry and "closed_us" in entry:
            limits[side] = {"open_us": entry["open_us"], "closed_us": entry["closed_us"]}
        else:
            limits[side] = None
            warnings.append(f"no claw limits for {key} ({side})")
    return limits, warnings


def finite(x) -> bool:
    """A JSON number usable as a float: not a bool, NaN, inf, or an integer too big for a float."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return False
    try:
        return math.isfinite(x)
    except OverflowError:
        return False


def usable_row(row) -> bool:
    """A v1 tap row with the fields the recorder itself reads. Anything else is not written."""
    zero = row.get("zero") if isinstance(row, dict) else None
    return (isinstance(row, dict) and row.get("v") == 1 and finite(row.get("t"))
            and isinstance(zero, list) and len(zero) == len(ARM_JOINTS)
            and all(isinstance(z, (int, float)) for z in zero))


def field_problem(fields: dict) -> str:
    """'' when a start's or a prime's fields are of the right kinds, else what is wrong."""
    mode, trial = fields.get("mode"), fields.get("trial")
    if mode is not None and mode not in MODES:
        return f"mode {mode!r}: teleop, guided or policy"
    if trial is not None and (isinstance(trial, bool) or not isinstance(trial, int)):
        return f"trial {trial!r}: a whole number, such as 17 for T017"
    for name in ("arrangement", "policy", "task"):
        if fields.get(name) is not None and not isinstance(fields[name], str):
            return f"{name} {fields[name]!r}: text"
    return ""


def primed_text(fields: dict) -> str:
    return ", ".join(f"{name} {fields[name]}" for name in PRIME_FIELDS if name in fields)


def ended_text(last: dict) -> str:
    """'ep_0003 success, saved': how an episode ended, for status."""
    if last["discarded"]:
        return f"{last['name']} discarded"
    reason = "" if last["end_reason"] == "operator" else f" ({last['end_reason']})"
    return f"{last['name']} {last['outcome']}{reason}, " + ("saved" if last["saved"] else "episode.json NOT written")


def same_zero(a: list, b: list) -> bool:
    """Joint by joint, with NaN equal to NaN, so an unknown joint cannot pass for a new zero."""
    return len(a) == len(b) and all(x == y or (x != x and y != y) for x, y in zip(a, b))


def per_second(times: collections.deque, now: float, window: float = 2.0) -> float:
    """Arrivals per second over the last `window` s, or over less while the stream is younger than that."""
    if not times:
        return 0.0
    span = min(window, now - times[0])
    return sum(1 for t in times if now - t <= window) / max(span, 0.1)


def refused(said: str) -> dict:
    return {"ok": False, "said": said}


class Episode:
    """The open episode: its folder, its three streams, and what episode.json will say."""

    def __init__(self, folder: Path, meta: dict):
        self.folder = folder
        self.name = folder.name
        self.meta = meta
        self.zero = meta["zero_at_start"]
        self.rows = open(folder / "rows.jsonl", "wb")
        self.frames = open(folder / "frames.mjpeg", "wb")
        self.index = open(folder / "frames.jsonl", "w")
        self.n_rows = self.n_frames = 0
        self.row_gaps = self.frame_gaps = 0
        self.lost_rows = 0            # gaps in the tap's seq: rows sent that never arrived
        self.offset = 0
        self.last_t = self.last_cap = self.last_seq = None
        self.dirty = False
        self.next_report = meta["start"]["t"] + 10.0

    def add_row(self, data: bytes, row: dict) -> None:
        self.rows.write(data + b"\n")                   # as received: the converter parses it
        t, seq = float(row["t"]), row.get("seq")
        if self.last_t is not None and t - self.last_t > ROW_GAP_S:
            self.row_gaps += 1
        if isinstance(seq, int) and isinstance(self.last_seq, int) and seq > self.last_seq + 1:
            self.lost_rows += seq - self.last_seq - 1
        self.last_t, self.last_seq = t, seq
        self.n_rows += 1
        self.dirty = True

    def add_frame(self, data: bytes, t_cap: float, t_seen: float, mtime_ns: int) -> None:
        # bytes first, then the line that points at them: after a crash no line points past the data
        self.frames.write(data)
        self.index.write(json.dumps({"i": self.n_frames, "off": self.offset, "len": len(data),
                                     "t_cap": round(t_cap, 6), "t_seen": round(t_seen, 6),
                                     "mtime_ns": mtime_ns}) + "\n")
        self.offset += len(data)
        if self.last_cap is not None and t_cap - self.last_cap > FRAME_GAP_S:
            self.frame_gaps += 1
        self.last_cap = t_cap
        self.n_frames += 1
        self.dirty = True

    def flush(self) -> None:
        """Out of this process at once (a crash loses at most a few ms); synced to disk at the end."""
        if self.dirty:
            for fh in (self.rows, self.frames, self.index):
                fh.flush()
            self.dirty = False

    def close(self) -> None:
        """Flushed, synced and closed, carrying on past any file that fails."""
        for fh in (self.rows, self.frames, self.index):
            try:
                fh.flush()
                os.fsync(fh.fileno())
            except (OSError, ValueError):
                pass
            try:
                fh.close()
            except OSError:
                pass


class Recorder:
    """One process: select() on the tap and control sockets, and the frame file polled on every wake."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = Path(args.root).expanduser()
        self.frame_file = Path(args.frame_file)
        self.poll_s = max(0.001, args.poll_ms / 1000.0)
        self.default_task = (args.task or "").strip()
        self.default_operator = args.operator
        # Rows queue here while episode.json is written and synced: none may be lost to a slow disk.
        self.tap = self._listen(args.tap_port, "tap rows", rcvbuf=4 << 20)
        self.ctl = self._listen(args.control_port, "commands")
        self.row_times: collections.deque = collections.deque(maxlen=256)    # the ring that says the
        self.frame_times: collections.deque = collections.deque(maxlen=256)  # streams are live
        self.last_row: dict | None = None
        self.last_row_at: float | None = None
        self.last_frame_at: float | None = None
        self.bad_rows = 0
        # The frame already in the file may be hours old: a baseline, not an arrival.
        self.frame_key = self._frame_key()
        self.frame_size = self._frame_size()
        self.episode: Episode | None = None
        self.last_closed: Path | None = None       # the newest finished episode, for "discard"
        self.last_ended: dict | None = None        # how the newest one ended and whether it was saved, for status
        self.next_index = 0
        self.stopping = False
        self.next_disk_check = self.next_disk_warning = 0.0
        self.min_free = args.min_free_gb * 1e9
        self.prime: dict | None = None             # fields for the next start only (the prime command)
        self.prime_at = 0.0
        self.after_reply: list = []                # held back until a command's reply is out: an end's save
        self.session_dir, self.session = self._open_session()

    @staticmethod
    def _listen(port: int, what: str, rcvbuf: int = 0) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if rcvbuf:
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
            except OSError:
                pass
        try:
            sock.bind(("127.0.0.1", port))          # this PC only: nothing on the LAN reaches it
        except OSError as exc:
            sock.close()
            raise SystemExit(f"recorder: cannot listen for {what} on UDP 127.0.0.1:{port} ({exc}): "
                             "is another recorder.py running?")
        sock.setblocking(False)
        return sock

    def _frame_key(self) -> tuple[int, int] | None:
        """(inode, mtime) of the frame file. Every atomic write is a new file, so a frame written within
        the same tick of the kernel's coarse file clock (1-4 ms) as the last one still counts as new."""
        try:
            st = os.stat(self.frame_file)
        except OSError:
            return None
        return st.st_ino, st.st_mtime_ns

    def _frame_size(self) -> tuple[int, int] | None:
        try:
            with open(self.frame_file, "rb") as fh:
                return jpeg_size(fh.read(65536))
        except OSError:
            return None

    def _open_session(self) -> tuple[Path, dict]:
        self.root.mkdir(parents=True, exist_ok=True)
        host = socket.gethostname().split(".")[0]
        host = "".join(c if c.isalnum() or c == "-" else "-" for c in host) or "robot"
        base = time.strftime("%Y%m%d_%H%M%S_") + host
        for n in range(1, 100):
            folder = self.root / (base if n == 1 else f"{base}_{n}")
            try:
                folder.mkdir()
                break
            except FileExistsError:
                continue
        else:
            raise SystemExit(f"recorder: could not make a session folder under {self.root}")
        session = {
            "format": "bhl-lfd-recording", "version": 1, "session_id": folder.name,
            "created_wall": round(time.time(), 6), "host": socket.gethostname(), "repo_commit": repo_commit(),
            "joint_names": [joint.key for joint in ARM_JOINTS],
            "urdf_joint_names": [joint.urdf for joint in ARM_JOINTS],
            "camera": {"frame_file": str(self.frame_file), "layout": "side_by_side", "rotated_180": True,
                       "calibration": calibration_for(self.frame_size), "camera_latency_s": None},
            "tap_port": self.args.tap_port, "control_port": self.args.control_port,
        }
        write_json(folder / "session.json", session)
        return folder, session

    def _free_bytes(self) -> int | None:
        try:
            return shutil.disk_usage(self.session_dir).free
        except OSError:
            return None

    # -- the loop
    def run(self) -> None:
        signal.signal(signal.SIGINT, self._stop_asked)
        signal.signal(signal.SIGTERM, self._stop_asked)
        if signal.getsignal(signal.SIGHUP) is not signal.SIG_IGN:   # started with nohup: it keeps on
            signal.signal(signal.SIGHUP, self._stop_asked)      # its terminal or ssh session went away
        print(f"Recorder: session {self.session['session_id']} in {self.session_dir}", flush=True)
        print(f"  tap rows on UDP 127.0.0.1:{self.args.tap_port} (run_teleop.py --record-tap), "
              f"commands on UDP 127.0.0.1:{self.args.control_port}", flush=True)
        print(f"  frames from {self.frame_file}, polled every {self.poll_s * 1000:g} ms", flush=True)
        print(f"  default task: {self.default_task!r}" if self.default_task
              else "  no default task yet: each start needs one (record_ctl.py task --task ...)", flush=True)
        free = self._free_bytes()
        if free is not None and free < self.min_free:
            print(f"  only {free / 1e9:.1f} GB free, below --min-free-gb {self.min_free / 1e9:g}: "
                  "starts are refused until there is more", flush=True)
        reason = "recorder stopped"
        try:
            while not self.stopping:
                try:
                    ready, _, _ = select.select([self.tap, self.ctl], [], [], self.poll_s)
                except InterruptedError:
                    ready = []
                # rows, then the frame, then commands: a start sees the freshest state of both streams
                if self.tap in ready:
                    self._take_rows()
                self._poll_frame()
                if self.ctl in ready:
                    self._take_commands()
                self._watch(time.monotonic())
        except Exception as exc:
            reason = f"recorder error: {exc!r}"[:200]
            raise
        finally:
            # however the loop ends, an open episode is closed properly, as aborted, and an ended
            # one whose save was still held back is saved
            self._after_reply()
            if self.episode is not None:
                self._finish("aborted", reason)
            self.tap.close()
            self.ctl.close()
            log("Recorder stopped.")

    def _stop_asked(self, signum, frame) -> None:
        self.stopping = True              # the loop finishes any open episode, then exits

    def _after_reply(self) -> bool:
        """Run what a command held back until its reply was sent; True when there was any."""
        ran = bool(self.after_reply)
        while self.after_reply:
            job = self.after_reply.pop(0)
            try:
                job()
            except Exception:
                traceback.print_exc()
        return ran

    def _take_rows(self) -> None:
        while True:
            try:
                data = self.tap.recv(65536)
            except (BlockingIOError, InterruptedError):
                return
            except OSError:
                return
            now = time.monotonic()
            try:
                row = json.loads(data)
            except (ValueError, RecursionError):        # RecursionError: "[[[[..." nested too deep
                row = None
            if not usable_row(row):
                self.bad_rows += 1
                continue
            self.row_times.append(now)
            self.last_row, self.last_row_at = row, now
            episode = self.episode
            if episode is None:
                continue
            if not same_zero(row["zero"], episode.zero):
                # Zero was taken again: every angle from here on is in another frame.
                self._finish("aborted", "zero changed", zero_changed=True)
                continue
            try:
                episode.add_row(data, row)
            except OSError as exc:
                self._finish("aborted", f"write failed: {exc}")

    def _poll_frame(self) -> None:
        try:
            st = os.stat(self.frame_file)
        except OSError:
            return
        if (st.st_ino, st.st_mtime_ns) == self.frame_key:
            return
        now = time.monotonic()
        offset_ns = time.time_ns() - time.monotonic_ns()     # wall minus monotonic, as the frame is seen
        episode = self.episode
        data = b""
        if episode is not None:
            try:
                with open(self.frame_file, "rb") as fh:
                    # ffmpeg may have replaced the file since the stat: these bytes are this file's
                    st = os.fstat(fh.fileno())
                    data = fh.read()
            except OSError:
                return                                       # the next poll gets the next frame
            if (st.st_ino, st.st_mtime_ns) == self.frame_key:
                return
        self.frame_key = (st.st_ino, st.st_mtime_ns)
        self.frame_times.append(now)
        self.last_frame_at = now
        if episode is None or not data:
            return
        if episode.n_frames == 0:
            self._note_frame_size(episode, data)
        try:
            episode.add_frame(data, (st.st_mtime_ns - offset_ns) / 1e9, now, st.st_mtime_ns)
        except OSError as exc:
            self._finish("aborted", f"write failed: {exc}")

    def _note_frame_size(self, episode: Episode, data: bytes) -> None:
        """session.json's camera calibration depends on the frame size, known for sure only now."""
        size = jpeg_size(data)
        if size is None or size == self.frame_size:
            return
        if self.frame_size is not None:
            episode.meta["warnings"].append(f"camera frame size changed to {size[0]}x{size[1]}")
        self.frame_size = size
        calibration = calibration_for(size)
        if calibration != self.session["camera"]["calibration"]:
            self.session["camera"]["calibration"] = calibration
            try:
                write_json(self.session_dir / "session.json", self.session)
            except OSError as exc:
                log(f"recorder: could not update session.json: {exc}")

    def _watch(self, now: float) -> None:
        episode = self.episode
        if episode is not None:
            if now - self.last_row_at > STALL_S:
                self._finish("aborted", f"no tap rows for {STALL_S:g} s")
            elif now - self.last_frame_at > STALL_S:
                self._finish("aborted", f"no camera frames for {STALL_S:g} s")
            else:
                try:
                    episode.flush()
                except OSError as exc:
                    self._finish("aborted", f"write failed: {exc}")
                if now >= episode.next_report:
                    episode.next_report = now + 10.0
                    log(f"  {episode.name}: {now - episode.meta['start']['t']:.0f} s, {episode.n_rows} rows, "
                        f"{episode.n_frames} frames")
        self._live_prime(now)                      # an expired prime is dropped, and said so, on time
        if now >= self.next_disk_check:
            self.next_disk_check = now + DISK_CHECK_S
            free = self._free_bytes()
            if free is not None and free < self.min_free and self.episode is not None:
                # Stopped before the disk fills: run_teleop.py's teleop log shares it, and a full disk would
                # stop teleop mid-motion. Closing the episode needs only a few KB more.
                self.episode.meta["warnings"].append(f"low disk: {free / 1e9:.1f} GB free, below the "
                                                     f"{self.min_free / 1e9:g} GB floor (--min-free-gb)")
                self._finish("aborted", "low disk")
            if free is not None and free < LOW_DISK_BYTES and now >= self.next_disk_warning:
                self.next_disk_warning = now + 30.0
                warning = f"low disk: {free / 1e9:.1f} GB free"
                log(f"recorder: {warning} in {self.session_dir}")
                if self.episode is not None and not any(w.startswith("low disk")
                                                        for w in self.episode.meta["warnings"]):
                    self.episode.meta["warnings"].append(warning)

    # -- commands
    def _take_commands(self) -> None:
        while True:
            try:
                data, sender = self.ctl.recvfrom(65536)
            except (BlockingIOError, InterruptedError):
                return
            except OSError:
                return
            try:
                cmd = json.loads(data)
                if not isinstance(cmd, dict):
                    raise ValueError("expected a JSON object")
            except (ValueError, RecursionError) as exc:  # RecursionError: "[[[[..." nested too deep
                cmd, reply = {}, refused(f'send JSON like {{"cmd": "status"}} ({exc})')
            else:
                try:
                    reply = self._command(cmd)
                except Exception as exc:            # a bad command must never end the recorder
                    traceback.print_exc()
                    reply = refused(f"recorder error: {exc}")
            try:
                self.ctl.sendto(json.dumps(reply).encode(), sender)
            except OSError:
                pass
            finally:
                saved = self._after_reply()        # an end's save, now that its answer is out
            if cmd.get("cmd") != "status":
                log(f"[{cmd.get('cmd')}] {'ok' if reply.get('ok') else 'refused'}: {reply.get('said')}")
            if saved:
                return                             # the streams first, then any command still queued

    def _command(self, cmd: dict) -> dict:
        what = cmd.get("cmd")
        if what == "start":
            return self._start(cmd)
        if what == "end":
            return self._end(cmd)
        if what == "discard":
            return self._discard()
        if what == "prime":
            return self._prime(cmd)
        if what == "task":
            return self._set_task(cmd)
        if what == "status":
            return self._status()
        return refused(f"unknown command {what!r}: start, end, discard, prime, task or status")

    def _start(self, cmd: dict) -> dict:
        now = time.monotonic()
        if self.episode is not None:
            return refused(f"{self.episode.name} is still open: end or discard it first")
        if self.last_row_at is None:
            return refused("no tap rows yet: is run_teleop.py running with --record-tap?")
        if now - self.last_row_at > LIVE_S:
            return refused(f"no tap rows for {now - self.last_row_at:.1f} s: "
                           "is run_teleop.py running with --record-tap?")
        if self.last_frame_at is None or now - self.last_frame_at > LIVE_S:
            since = "yet" if self.last_frame_at is None else f"for {now - self.last_frame_at:.1f} s"
            return refused(f"no camera frames {since}: is camera_share.py running?")
        # A prime fills in what this start leaves out (a voice start says nothing but "start").
        prime = self._live_prime(now)
        fields = dict(prime or {})
        fields.update({name: cmd[name] for name in PRIME_FIELDS if cmd.get(name) is not None})
        problem = field_problem(fields)
        if problem:
            return refused(problem)
        task = (fields.get("task") or self.default_task or "").strip()
        if not task:
            return refused("no task: give one, or set a default first (record_ctl.py task --task ...)")
        # A voice start carries no mode: the operator in the headset is teleoperating.
        mode = fields.get("mode") or "teleop"
        row = self.last_row
        warnings = []
        cam, cam_mode = row.get("cam"), row.get("cam_mode")
        if cam_mode not in (None, "still"):
            # also with the Nano gone (cam null): back, it would turn the camera in the middle of the episode
            if not self.args.allow_moving_camera:
                return refused(f'the camera is on {cam_mode}: say "agent, camera still" first - '
                               "a moving camera changes the viewpoint between episodes")
            warnings.append(f"camera on {cam_mode}, not still")
        if cam is None:
            warnings.append("no camera servos: camera angle unknown" if cam_mode is None
                            else "camera servos not connected: camera angle unknown")
        if mode == "teleop" and row.get("state") != "ARMED":
            warnings.append(f"arms {row.get('state')}, not ARMED")
        free = self._free_bytes()
        if free is not None and free < self.min_free:
            return refused(f"only {free / 1e9:.1f} GB free, below the {self.min_free / 1e9:g} GB floor: free some "
                           "space first (run_teleop.py's teleop log is on this disk)")
        if free is not None and free < LOW_DISK_BYTES:
            warnings.append(f"low disk: {free / 1e9:.1f} GB free")
        profile_path = Path(self.args.profile)
        profile = read_json(profile_path)
        profile_sha = file_sha256(profile_path)
        if profile_sha is None:
            warnings.append(f"no power profile at {profile_path}")
        convention = profile.get("zero_convention") if isinstance(profile, dict) else None
        zero_check = read_json(self.args.zero_check)
        if zero_check is None:
            warnings.append("no zero check recorded")
        claws, claw_warnings = claw_limits(self.args.claw_limits)
        warnings += claw_warnings

        index = self.next_index
        folder = self.session_dir / f"ep_{index:04d}"
        folder.mkdir()
        self.next_index += 1
        meta = {
            "format": "bhl-lfd-episode", "version": 1, "episode_index": index,
            "session_id": self.session["session_id"],
            "task": task, "mode": mode, "policy": fields.get("policy"),
            "operator": cmd.get("operator") or self.default_operator,
            "arrangement": fields.get("arrangement"), "trial": fields.get("trial"),
            # end and outcome stay null until the episode is finalized: a crash leaves that mark
            "start": {"t": round(now, 6), "wall": round(time.time(), 6)}, "end": None,
            "outcome": None, "end_reason": None, "discarded": False, "notes": "",
            "counts": {"rows": 0, "frames": 0, "row_gaps_over_50ms": 0, "frame_gaps_over_67ms": 0},
            "cam_mode_at_start": cam_mode, "cam_at_start": cam,
            "zero_at_start": row["zero"], "zero_changed": False,
            "zero_convention": convention if isinstance(convention, str) and convention else "hanging=0",
            "zero_check": zero_check, "profile_sha256": profile_sha, "claw_limits": claws,
            "warnings": warnings,
        }
        write_json(folder / "episode.json", meta)
        self.episode = Episode(folder, meta)
        if prime is not None:
            self.prime = None                    # used: a prime is for one start only
        primed = {name: value for name, value in (prime or {}).items()   # mode and task are said anyway
                  if cmd.get(name) is None and name not in ("mode", "task")}
        said = (f"recording {folder.name} ({mode}): {task}" + (f" ({primed_text(primed)})" if primed else "")
                + (" - " + "; ".join(warnings) if warnings else ""))
        return {"ok": True, "said": said, "episode": folder.name, "warnings": warnings}

    def _live_prime(self, now: float) -> dict | None:
        """The prime, unless it has expired (then it is dropped, and said so)."""
        if self.prime is not None and now - self.prime_at > PRIME_S:
            log(f"recorder: the prime ({primed_text(self.prime)}) expired unused after {PRIME_S / 60:g} min")
            self.prime = None
        return self.prime

    def _prime(self, cmd: dict) -> dict:
        """Fields for the next start only, whoever sends it; without any fields, the prime is cleared."""
        unknown = sorted(set(cmd) - {"cmd", *PRIME_FIELDS})
        if unknown:
            return refused(f"prime takes {', '.join(PRIME_FIELDS)}; not {', '.join(map(str, unknown))}")
        fields = {name: cmd[name].strip() if isinstance(cmd[name], str) else cmd[name]
                  for name in PRIME_FIELDS if cmd.get(name) is not None}
        fields = {name: value for name, value in fields.items() if value != ""}
        problem = field_problem(fields)
        if problem:
            return refused(problem)
        if not fields:
            had, self.prime = self.prime, None
            return {"ok": True, "said": "prime cleared" if had else "nothing was primed", "prime": None}
        self.prime, self.prime_at = fields, time.monotonic()
        return {"ok": True, "said": f"the next start, within {PRIME_S / 60:g} min, gets {primed_text(fields)}",
                "prime": fields}

    def _end(self, cmd: dict) -> dict:
        if self.episode is None:
            return refused("no episode is open")
        outcome = cmd.get("outcome")
        if outcome not in OUTCOMES:
            return refused("end needs an outcome: success, failure or aborted")
        # Answered before the save: syncing a minute of frames can take seconds on the robot PC, and the
        # operator should hear at once that the end was taken. Nothing more is written to the episode,
        # and the save runs as soon as this reply is out, before any other command is read: so the next
        # reply anyone gets comes after it (record_ctl.py end waits for one).
        episode, self.episode = self.episode, None
        now, notes = time.monotonic(), str(cmd.get("notes") or "")
        self.after_reply.append(lambda: self._finish(outcome, "operator", notes=notes, episode=episode, now=now))
        counts = {"rows": episode.n_rows, "frames": episode.n_frames, "row_gaps_over_50ms": episode.row_gaps,
                  "frame_gaps_over_67ms": episode.frame_gaps}
        seconds = now - episode.meta["start"]["t"]
        return {"ok": True, "said": f"{episode.name} {outcome}: {seconds:.1f} s, {counts['rows']} rows, "
                                    f"{counts['frames']} frames - saving", "episode": episode.name, "counts": counts}

    def _finish(self, outcome: str, reason: str, *, zero_changed: bool = False, notes: str = "",
                discarded: bool = False, episode: Episode | None = None, now: float | None = None) -> dict:
        """Close an episode, the open one unless given: streams synced, then episode.json in its final form."""
        if episode is None:
            episode, self.episode = self.episode, None    # nothing more is written to it from here
        now = time.monotonic() if now is None else now
        episode.close()
        meta = episode.meta
        if file_sha256(Path(self.args.profile)) != meta["profile_sha256"]:
            meta["warnings"].append("power profile changed during the episode")
        if episode.lost_rows:
            meta["warnings"].append(f"{episode.lost_rows} tap rows lost (gaps in seq)")
        meta.update(end={"t": round(now, 6), "wall": round(time.time(), 6)}, outcome=outcome, end_reason=reason,
                    discarded=discarded, notes=notes, zero_changed=zero_changed,
                    counts={"rows": episode.n_rows, "frames": episode.n_frames,
                            "row_gaps_over_50ms": episode.row_gaps, "frame_gaps_over_67ms": episode.frame_gaps})
        try:
            write_json(episode.folder / "episode.json", meta)
            saved = True
        except OSError as exc:
            saved = False
            log(f"recorder: could not write {episode.folder / 'episode.json'}: {exc}")
        self.last_closed = episode.folder
        self.last_ended = {"name": episode.name, "outcome": outcome, "end_reason": reason, "saved": saved,
                           "discarded": discarded}
        seconds = now - meta["start"]["t"]
        log(f"{episode.name} {outcome} ({reason}): {seconds:.1f} s, {episode.n_rows} rows, "
            f"{episode.n_frames} frames" + (f"; warnings: {'; '.join(meta['warnings'])}" if meta["warnings"] else ""))
        return meta

    def _discard(self) -> dict:
        if self.episode is not None:
            self._finish("aborted", "discarded", discarded=True)
            folder = self.last_closed
        elif self.last_closed is not None:
            folder = self.last_closed
            meta = read_json(folder / "episode.json")
            if isinstance(meta, dict):
                meta["discarded"] = True
                write_json(folder / "episode.json", meta)
        else:
            return refused("nothing to discard")
        bin_dir = self.session_dir / "_discarded"
        bin_dir.mkdir(exist_ok=True)
        folder.rename(bin_dir / folder.name)
        sync_dir(self.session_dir)
        sync_dir(bin_dir)
        self.last_closed = None                # one discard per episode: a second never reaches further back
        if self.last_ended is not None and self.last_ended["name"] == folder.name:
            self.last_ended["discarded"] = True
        return {"ok": True, "said": f"{folder.name} discarded (moved to _discarded)", "episode": folder.name}

    def _set_task(self, cmd: dict) -> dict:
        task = str(cmd.get("task") or "").strip()
        if not task:
            return refused("task needs the task text")
        self.default_task = task
        return {"ok": True, "said": f"default task: {task}"}

    def _status(self) -> dict:
        now = time.monotonic()
        rows_per_s, frames_per_s = per_second(self.row_times, now), per_second(self.frame_times, now)
        row_age = None if self.last_row_at is None else round(now - self.last_row_at, 3)
        frame_age = None if self.last_frame_at is None else round(now - self.last_frame_at, 3)
        free = self._free_bytes()
        warnings = []
        if row_age is None or row_age > LIVE_S:
            warnings.append("no tap rows")
        if frame_age is None or frame_age > LIVE_S:
            warnings.append("no camera frames")
        row = self.last_row or {}
        cam_mode = row.get("cam_mode")
        if self.last_row is not None and cam_mode not in (None, "still") and not self.args.allow_moving_camera:
            warnings.append(f"camera on {cam_mode}: starts refused until it is still")
        if self.last_row is not None and row.get("cam") is None:
            warnings.append("no camera servos: camera angle unknown" if cam_mode is None
                            else "camera servos not connected: camera angle unknown")
        if free is not None and free < self.min_free:
            warnings.append(f"only {free / 1e9:.1f} GB free, below the {self.min_free / 1e9:g} GB floor: "
                            "starts refused")
        elif free is not None and free < LOW_DISK_BYTES:
            warnings.append(f"low disk: {free / 1e9:.1f} GB free")
        if self.bad_rows:
            warnings.append(f"{self.bad_rows} unreadable tap rows ignored")
        episode = None
        if self.episode is not None:
            ep = self.episode
            seconds = now - ep.meta["start"]["t"]
            episode = {"name": ep.name, "task": ep.meta["task"], "mode": ep.meta["mode"],
                       "seconds": round(seconds, 1), "rows": ep.n_rows, "frames": ep.n_frames,
                       "warnings": ep.meta["warnings"]}
            said = f"recording {ep.name}, {seconds:.0f} s"
        else:
            said = "no episode open" + (f" (last: {ended_text(self.last_ended)})" if self.last_ended else "")
        said += f": {rows_per_s:.0f} rows/s, {frames_per_s:.0f} frames/s"
        if free is not None:
            said += f", {free / 1e9:.0f} GB free"
        prime = self._live_prime(now)
        prime_left = None if prime is None else round(PRIME_S - (now - self.prime_at))
        if prime is not None:
            said += f"; next start primed: {primed_text(prime)} ({prime_left / 60:.0f} min left)"
        if warnings:
            said += " - " + "; ".join(warnings)
        return {"ok": True, "said": said, "episode": episode, "rows_per_s": rows_per_s,
                "frames_per_s": frames_per_s, "last_row_age_s": row_age, "last_frame_age_s": frame_age,
                "warnings": warnings, "free_gb": None if free is None else round(free / 1e9, 1),
                "default_task": self.default_task or None, "session": self.session["session_id"],
                "folder": str(self.session_dir), "prime": prime, "prime_expires_in_s": prime_left,
                "last_episode": self.last_ended}


def gigabytes(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(f"{text}: GB of free space, 0 or more")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="~/bhl_recordings", help="sessions are written under this folder")
    parser.add_argument("--tap-port", type=int, default=TAP_PORT, help="UDP port of run_teleop.py --record-tap")
    parser.add_argument("--control-port", type=int, default=CONTROL_PORT,
                        help="UDP port for record_ctl.py and the voice commands")
    parser.add_argument("--frame-file", type=Path, default=FRAME_FILE, help="camera_share.py's newest frame")
    parser.add_argument("--poll-ms", type=float, default=5.0, help="how often the frame file is checked")
    parser.add_argument("--allow-moving-camera", action="store_true",
                        help="start episodes even while the camera tracks you or follows your head")
    parser.add_argument("--task", default="", help="the default task text, used by voice starts")
    parser.add_argument("--operator", default=None, help="the default operator name")
    parser.add_argument("--zero-check", type=Path, default=REPO_ROOT / "configs/zero_check.json",
                        help="tools/lfd/zero_check.py's result, copied into every episode "
                             "(default: configs/zero_check.json)")
    parser.add_argument("--profile", type=Path, default=PROFILE_PATH,
                        help="the power profile, hashed into every episode (default: configs/arm_power_profile.json)")
    parser.add_argument("--claw-limits", type=Path, default=REPO_ROOT / "configs/claw_limits.json",
                        help="the claws' open and closed pulses (default: configs/claw_limits.json)")
    parser.add_argument("--min-free-gb", type=gigabytes, default=MIN_FREE_GB,
                        help="below this much free space, starts are refused and an open episode ends as aborted "
                             "('low disk'), before the disk fills: run_teleop.py's teleop log is on it (default 2)")
    return parser.parse_args(argv)


def main() -> int:
    Recorder(parse_args()).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
