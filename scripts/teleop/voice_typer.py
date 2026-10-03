#!/usr/bin/env python3
"""Speak into the headset, land the words in a terminal (this Claude session).

The headset page captures its microphone while you hold the right trigger and
sends the audio to quest_bridge.py, which forwards it here. Vosk shows the words
as they are spoken and Whisper reads the finished sentence properly (both offline,
no cloud), and tmux types that text into the pane you name.

tmux is needed because the kernel refuses direct terminal injection
(dev.tty.legacy_tiocsti = 0) and Wayland refuses synthetic keystrokes.

  sudo apt install tmux
  tmux new -s claude            # then, inside it:  claude --continue
  .venv/bin/python scripts/teleop/voice_typer.py     # from a DIFFERENT terminal

Enter is pressed for you once the words are on the screen, so a sentence is sent the
moment you let go of the trigger; --no-submit leaves it sitting in the prompt instead.

Run this in a different terminal from Claude: tmux types into a pane, and if that
pane is this program's own, every word lands in here instead of in Claude.
Everything heard is also appended to /dev/shm/bhl_voice.txt, so nothing is lost
when the text cannot be delivered.

  .venv/bin/python scripts/teleop/voice_typer.py --self-test "hello from voice"
"""

from __future__ import annotations

import argparse
import array
import collections
import json
import os
import queue
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

AUDIO_MAGIC = b"AUD0"
DEFAULT_MODEL = Path.home() / ".cache/vosk/vosk-model-small-en-us-0.15"
DISCARD = {"cancel", "scratch that", "never mind", "nevermind", "forget it"}
TRANSCRIPT = Path("/dev/shm/bhl_voice.txt")     # everything heard, even if it cannot be typed
LAST_AUDIO = Path("/dev/shm/bhl_voice_last.wav")  # the last utterance, to listen to when it goes wrong
QUIET_PEAK = 600                                # out of 32767: below this it is effectively silence
QUIET_RMS = 120                                 # a single click can beat QUIET_PEAK; loudness cannot
BABBLE = 4                                      # one word repeated this often is a hallucination
WHISPER_PROMPT = ("Talking to Claude Code and the Claude sessions about the robot: the servo tilts the "
                  "stereo camera. Servos, claws, Nano, lidar, IMU, scrcpy, tmux, Quest headset, teleop, "
                  "calibration, passthrough, dictation, developer view, bringup, CAN bus, URDF, Vosk, "
                  "Whisper, Berkeley Humanoid Lite. Joints: left yaw, right yaw, pitch, roll, elbow, wrist; "
                  "more power, less sensitive.")
# Sound-alikes Whisper writes for this project's words (from bhl_voice.txt), each put right
# only where the project's word is the one sensible reading: a "cloud session" is Claude's,
# a cloud on its own may be weather.
SOUNDALIKES = (
    (re.compile(r"\b(?i:i?cloud|clawed|clod|claude)\s+(?i:code)\b"), "Claude Code"),
    (re.compile(r"\b(?i:i?cloud|clawed|clod)(?=\s+(?:(?i:sessions?|symbol|icon|logo)\b|[A-HJ-Z]\b|IL\b))"),
     "Claude"),                                          # "cloud session", "iCloud symbol", "cloud C"
    (re.compile(r"\b(?i:plot|cloth)(?=\s+(?i:sessions?)\b)"), "Claude"),
    (re.compile(r"\b((?i:open)\s+(?:(?i:up|a|new|the)\s+)*)(?i:i?cloud|clawed|clod)\b"), r"\1Claude"),
    (re.compile(r"\b(?i:serval)(s?)\b"), r"servo\1"),
    (re.compile(r"\b(?:[Cc][\s.-]*)?[Ss][\s.-]*[Cc][\s.-]*[Rr][\s.-]*[Cc][\s.-]*[Pp][\s.-]*[Yy]\b"), "scrcpy"),
)
MAX_KEPT = 16000 * 2 * 60                       # keep at most a minute of audio for LAST_AUDIO
GAP_END = 4.0          # s without a single packet from the headset: the page is gone, end the sentence.
                       # It was 0.5 s, and every Wi-Fi hiccup that long cut a sentence off mid-word.
SPEECH_RMS = 250       # loudness that counts as speech, unless the room's own noise is higher
WAKE_CONF = 0.6        # how sure Vosk has to be that it heard the wake word, 0..1
WAKE_BUFFER_S = 20.0   # audio kept while waiting, so a sentence said in the same breath as the wake word is kept
SEED_S = 0.3           # a trigger press keeps this much audio from just before it: people start a hair early
MIN_TRIGGER_S = 0.5    # a trigger tap shorter than this is a slip or a double-tap toggle, not a sentence
IGNORE_START_S = 0.4   # the start beep and the network delay, before loudness counts as speech
SILENT_MIC = """\
  Every sample was zero, so this is not a quiet room: the headset granted the
  microphone and then put no sound into it. In the headset:
    1. Meta button -> Quick Settings -> Microphone: switch it on.
    2. Close whatever else is using the microphone (party chat, casting, recording).
    3. Press the LEFT STICK to leave the headset view. Under 'Enable voice dictation'
       the page shows a live level meter - speak and watch it move.
    4. Flat there too? Tap 'Enable voice dictation' again and allow the microphone.
  The page reopens the microphone without echo cancellation after a silent take,
  so holding the trigger once more is worth a try on its own."""
PANE_KEYS = ("id", "session", "window", "pane", "command", "tty", "pid")
PANE_FORMAT = ("#{pane_id}\t#{session_name}\t#{window_index}\t#{pane_index}"
               "\t#{pane_current_command}\t#{pane_tty}\t#{pane_pid}")
CLAUDE_COMMANDS = {"claude", "node", "bun", "deno"}


def tmux_panes() -> list[dict[str, str]]:
    """Every pane tmux owns, on every session."""
    found = subprocess.run(["tmux", "list-panes", "-a", "-F", PANE_FORMAT],
                           capture_output=True, text=True)
    if found.returncode != 0:
        return []
    panes = []
    for row in found.stdout.splitlines():
        fields = row.split("\t")
        if len(fields) == len(PANE_KEYS):
            panes.append(dict(zip(PANE_KEYS, fields)))
    return panes


def process_tree() -> dict[int, list[tuple[int, str]]]:
    """Every process by its parent, so a pane's descendants can be walked."""
    listing = subprocess.run(["ps", "-eo", "pid=,ppid=,comm="], capture_output=True, text=True)
    children: dict[int, list[tuple[int, str]]] = {}
    for row in listing.stdout.splitlines():
        parts = row.split(None, 2)
        if len(parts) != 3:
            continue
        try:
            children.setdefault(int(parts[1]), []).append((int(parts[0]), parts[2].strip()))
        except ValueError:
            continue
    return children


def runs_claude(pane: dict[str, str], children: dict[int, list[tuple[int, str]]]) -> bool:
    """Is Claude Code running in this pane, even under a wrapper?

    `pane_current_command` names the leader of the pane's foreground process
    group. bringup starts Claude from `bash -c ...`, and a non-interactive bash
    does not give its child a group of its own, so the two share one and tmux
    reports `bash`. Claude is then invisible to anything reading that field -
    which is how a whole session of dictation ended up with nowhere to type.
    The process tree does not lie, so ask it too.
    """
    if pane["command"] in CLAUDE_COMMANDS:
        return True
    try:
        stack = [int(pane.get("pid") or 0)]
    except ValueError:
        return False
    seen: set[int] = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        for child, comm in children.get(pid, ()):
            if comm in CLAUDE_COMMANDS:
                return True
            stack.append(child)
    return False


def describe(pane: dict[str, str]) -> str:
    return f"{pane['session']}:{pane['window']}.{pane['pane']} ({pane['command']}, {pane['tty']})"


def letters(text: str) -> str:
    """Only the letters and digits: a TUI wraps lines and draws borders through words."""
    return "".join(ch.lower() for ch in text if ch.isalnum())


def on_screen(session: str) -> str:
    """"@" is what the headset's normal view asks for: the Claude session the PC's own
    screen shows now (claude_sessions.shown). Any other name is itself. "" = the main one."""
    if session != "@":
        return session
    try:
        import claude_sessions
        return claude_sessions.shown() or ""
    except Exception:
        return ""


class Typist:
    """Types finished sentences into the tmux pane that runs Claude.

    tmux can only type into a pane it owns, and it will happily type into this
    program's own pane: run voice_typer inside the session it is aiming at and
    every word lands in voice_typer's stdin instead of in Claude. The target is
    therefore resolved to a real pane, re-checked before every utterance (the
    layout can change while this runs), and refuses to be its own target.
    """

    def __init__(self, target: str, submit: bool = True, settle: float = 0.25):
        self.target = target
        self.submit = submit
        self.settle = settle
        self.pane = ""
        self.session = ""             # the Claude session the headset says you want ("M", "CAM"...)

    def typing_into_myself(self, pane: dict[str, str]) -> bool:
        """Would the keystrokes come back to this process?

        Only if this process is what that terminal currently sends its input to:
        the foreground process group of that tty. TMUX_PANE alone cannot answer
        it, because every child inherits it - a helper started from inside the
        pane's program (Claude, say) is a perfectly good sender.
        """
        for fd in (0, 1, 2):
            try:
                if os.ttyname(fd) != pane["tty"]:
                    continue
                if os.tcgetpgrp(fd) == os.getpgrp():
                    return True
            except OSError:
                continue
        return False

    def resolve(self, session: str | None = None) -> tuple[str, str]:
        """(pane to type into, why not); `session` overrides the one the headset names now."""
        session = on_screen(self.session if session is None else session)
        if shutil.which("tmux") is None:
            return "", "tmux is not installed:  sudo apt install tmux"
        panes = tmux_panes()
        if not panes:
            return "", self._nowhere(panes)

        if self.target == "auto" and session:
            # the headset names a session: the one you look at, else the one used last
            try:
                import claude_sessions
                pane = claude_sessions.pane_of(session)
            except Exception:
                pane = None
            match = next((p for p in panes if p["id"] == pane), None)
            if match is not None and not self.typing_into_myself(match):
                return match["id"], ""

        if self.target == "auto":
            children = process_tree()
            running = [p for p in panes if runs_claude(p, children)
                       and not self.typing_into_myself(p)]
            if len(running) == 1:
                return running[0]["id"], ""
            if not running:
                return "", self._nowhere(panes)
            # several sessions and none named: the main one (window "claude")
            main = next((p for p in running if p.get("session") == "bhl" and p.get("window") == "0"), None)
            if main is not None:
                return main["id"], ""
            names = ", ".join(describe(p) for p in running)
            return "", f"more than one pane is running Claude ({names}); choose one with --target"

        # An unknown target is not an error to tmux: it prints an empty line and
        # exits 0, and `send-keys` would then go to whichever pane is current.
        asked = subprocess.run(["tmux", "display-message", "-p", "-t", self.target, "-F", "#{pane_id}"],
                               capture_output=True, text=True)
        pane_id = asked.stdout.strip() if asked.returncode == 0 else ""
        pane = next((p for p in panes if p["id"] == pane_id), None)
        if pane is None:
            return "", (f"tmux has no pane called {self.target!r}.\n" + self._nowhere(panes))
        if self.typing_into_myself(pane):
            return "", (f"{self.target!r} is the pane I am running in ({describe(pane)}), so every word "
                        "would be typed into me instead of into Claude.\n" + self._nowhere(panes))
        return pane_id, ""      # a pane the user named: their terminal, their choice

    def _nowhere(self, panes: list[dict[str, str]]) -> str:
        listing = "\n".join("      " + describe(p) for p in panes) or "      (none)"
        return ("no tmux pane is running Claude Code, and tmux can only type into a pane it owns.\n"
                "    1. stop Claude in the terminal it runs in now\n"
                "    2. tmux new -s claude\n"
                "    3. claude --continue           <- inside that tmux session\n"
                "    4. start me again from a different terminal:\n"
                "         .venv/bin/python scripts/teleop/voice_typer.py\n"
                "  panes tmux can see right now:\n" + listing)

    def available(self) -> str:
        pane, problem = self.resolve()
        self.pane = pane
        return problem

    def type(self, text: str, session: str | None = None) -> str:
        pane, problem = self.resolve(session)
        self.pane = pane
        if problem:
            return problem
        # A dialog in the prompt's place (a permission question, "Teach auto mode...?")
        # swallows the words, and the Enter after them picks its highlighted answer.
        try:
            import claude_sessions
            dialog = claude_sessions.in_the_way(pane)
        except Exception:
            dialog = ""
        if dialog:
            return (f"not typed: the session is showing a dialog, \"{dialog[:60]}\" - "
                    "answer it on the PC, or say \"agent, escape\" to cancel it")
        sent = subprocess.run(["tmux", "send-keys", "-t", pane, "-l", "--", text],
                              capture_output=True, text=True)
        if sent.returncode != 0:
            return f"tmux refused the text: {sent.stderr.strip()}"
        if self.submit:
            self.wait_until_shown(text, pane)
            subprocess.run(["tmux", "send-keys", "-t", pane, "Enter"], capture_output=True)
        return ""

    def wait_until_shown(self, text: str, pane: str, limit: float = 1.5) -> bool:
        """Hold the Enter back until the words are on the screen.

        Claude Code reads a burst of keystrokes as a paste, and an Enter caught in
        the same burst becomes a newline in the prompt instead of sending it. The
        pane itself says when the TUI has caught up, which no fixed wait can; the
        settle time is the floor under that, and whatever does not echo (a shell
        reading a password, say) simply waits it out and is sent anyway.
        """
        time.sleep(self.settle)
        want = letters(text)[-24:]
        if not want:
            return False
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline:
            shown = subprocess.run(["tmux", "capture-pane", "-p", "-t", pane],
                                   capture_output=True, text=True)
            if want in letters(shown.stdout):
                return True
            time.sleep(0.05)
        return False


def loudness(chunk: bytes) -> tuple[int, int]:
    """(peak, rms) of 16-bit mono audio, for telling silence from bad recognition."""
    try:
        import audioop                                  # gone in 3.13, so fall back
        return audioop.max(chunk, 2), audioop.rms(chunk, 2)
    except Exception:
        samples = array.array("h")
        samples.frombytes(chunk[:len(chunk) // 2 * 2])
        if not samples:
            return 0, 0
        peak = max(abs(v) for v in samples)
        rms = int((sum(v * v for v in samples) / len(samples)) ** 0.5)
        return peak, rms


def save_audio(raw: bytes) -> None:
    """The last utterance as a .wav, so a failure can be listened to afterwards."""
    if not raw:
        return
    try:
        with wave.open(str(LAST_AUDIO), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(16000)
            out.writeframes(raw)
    except (OSError, wave.Error):
        pass


def where(typist: "Typist") -> str:
    pane = next((p for p in tmux_panes() if p["id"] == typist.pane), None)
    return describe(pane) if pane else typist.pane


OFF_WORDS = {"stop", "pause", "disable", "off", "freeze", "no", "don't", "dont", "kill", "end", "quit", "hide"}
ON_WORDS = {"start", "enable", "on", "resume", "begin", "use", "activate", "yes", "show"}
SESSION_OPEN = {"open", "new", "start", "create", "make", "reopen", "resume", "restore", "bring", "continue"}
SESSION_CLOSE = {"close", "remove", "kill", "delete", "shut", "save", "exit", "quit", "end", "stop", "dismiss",
                 "hide", "away"}
# Said alone, these put away whatever the agent has open (the help menu). "stop" is not
# one of them: said to a robot, it should never mean "close a menu".
MENU_CLOSE = {"close", "exit", "done", "dismiss", "hide", "away", "back", "cancel", "finished", "leave", "quit"}
PANELS = {"camera": "camera", "stereo": "camera", "screen": "screen", "pc": "screen", "computer": "screen",
          "desktop": "screen", "robot": "status", "status": "status"}
# "agent, look happy": a word for each of the robot's moods (face_display.py's EXPRESSIONS)
FEELINGS = {"happy": "happy", "glad": "happy", "smile": "happy", "smiling": "happy", "joyful": "happy",
            "sad": "sad", "unhappy": "sad", "upset": "sad", "crying": "sad",
            "angry": "angry", "mad": "angry", "furious": "angry", "grumpy": "angry",
            "surprised": "surprised", "surprise": "surprised", "shocked": "surprised", "amazed": "surprised",
            "worried": "worried", "nervous": "worried", "scared": "worried", "afraid": "worried",
            "focused": "focused", "serious": "focused", "determined": "focused",
            "curious": "curious", "confused": "curious", "suspicious": "curious",
            "love": "love", "loving": "love", "hearts": "love",
            "sleepy": "sleepy", "tired": "sleepy", "bored": "sleepy",
            "asleep": "asleep", "sleep": "asleep",
            "calm": "calm", "neutral": "calm", "relaxed": "calm"}
FACE_PORT = 11010             # face_display.py: the tracker's reports, and these
AGENT_HELP = ("open claude session [called X] · close session [X or its title] · saved sessions · open the <title> session · developer / calibration · vision on/off · repeat · screen 1/2 · stop/start looking to swap · camera track/head/still · details · recenter · reset panel · "
              "show camera/screen/robot · hands-free off · hands on/off · look happy/sad/angry/surprised · your own moods · "
              "show the spark · start/end episode success|fail · discard episode · episode status · "
              "help · close (the menu)")


def parse_command(text: str) -> dict:
    """A spoken command -> {"do": ..., ...}. Deliberately plain: keywords, not a model.

    set:    {"do": "set", "feature": swap|camera|details|handsfree|hands, "on": True/False/None=toggle}
    action: {"do": "recenter"|"reset"|"help"} or {"do": "focus", "panel": camera|screen|status}
    none:   {"do": "unknown"}
    """
    words = re.findall(r"[a-z0-9']+", text.lower().replace("-", " "))
    have = set(words)
    joined = " ".join(words)
    on = False if have & OFF_WORDS else True if have & ON_WORDS else None
    if not words:
        return {"do": "unknown"}
    if have & {"help", "commands"} or "what can you do" in joined:
        return {"do": "help"}
    # demonstrations for recorder.py, before tuning and sessions: a sentence that names an episode is
    # about the recording, never a live change to the arms ("end episode fail, it needs more power" must
    # not raise a joint's power, nor "undo episode" take back the last tuning), and "new episode" is not
    # a Claude session. One it cannot read, "stop episode" among them, does nothing at all.
    if have & EPISODE_WORDS:
        return parse_episode(have) or {"do": "unknown"}
    # the arms, tuned while you drive (run_teleop's tuning port, as tools/teleop_tune.py)
    tune = parse_tune(words, have)
    if tune:
        return tune
    # a dialog stuck on your session (a permission question, "Teach auto mode...?"): Esc
    if have & {"escape"} or (have & {"cancel", "dismiss", "close"} and have & {"dialog", "question", "prompt", "popup"}):
        return {"do": "escape"}
    if have & {"session", "sessions"}:
        # Claude sessions: list the saved ones, close one (it is saved), or open one - a
        # saved one whose title matches the words said, else a new one. claude_sessions
        # decides which, from the words; "session A" or "called CAM" name one.
        called = re.search(r"\b(?:called|named|name)\s+([a-z]+(?:\s+[a-z]+)?)", joined)
        spoken = re.search(r"\bsessions?\s+([a-z]{1,3})\b", joined)
        name = ("".join(w[0] for w in called.group(1).split()) if " " in called.group(1) else called.group(1)) \
            if called else (spoken.group(1) if spoken else "")
        if have & {"list", "which", "what"} or ("saved" in have and not have & (SESSION_OPEN | SESSION_CLOSE)):
            return {"do": "list_sessions"}
        if have & SESSION_CLOSE and not have & SESSION_OPEN:
            return {"do": "close_session", "name": name[:3].upper(), "words": words}
        if have & SESSION_OPEN:
            return {"do": "open_session", "name": name[:3].upper(), "words": words, "new": "new" in have}
    if have & {"developer", "develop", "dev", "sensors", "sensor"}:
        # "exit developer mode" and "developer off" leave it: the word alone went in again
        leaving = have & (OFF_WORDS | {"exit", "leave", "close", "out", "back", "without"})
        return {"do": "view", "view": "normal" if leaving else "developer"}
    if have & {"normal", "regular", "default", "calibration", "calibrate", "calibrating"} and not have & {"camera"}:
        return {"do": "view", "view": "normal"}   # "calibration mode": PC screen, camera, robot panel
    if have & {"vision", "detection", "detections", "detector", "boxes", "bounding", "skeleton", "pose"}:
        return {"do": "set", "feature": "vision", "on": on}   # both eyes, with what the detector sees
    if have & {"repeat", "again"} or "what did you say" in joined:
        return {"do": "repeat"}                   # Claude's last reply, read out once more
    if have & {"recenter", "recentre"} or "re center" in joined or "re centre" in joined:
        return {"do": "recenter"}
    if "reset" in have:
        return {"do": "reset"}
    if "hands free" in joined or "handsfree" in have or "wake word" in joined or "dictation" in have:
        return {"do": "set", "feature": "handsfree", "on": on}
    # the robot's face (face_display.py): a mood for its eyes, its own moods again, or the spark.
    # Before "look": "look happy" is a face, not the looking swap.
    if "spark" in have or ("face" in have and have & {"eyes", "eye"}):
        return {"do": "feel", "show": "spark" if "spark" in have else "eyes"}
    if have & {"mood", "moods", "emotion", "emotions", "feeling", "feelings"} \
            and have & {"own", "auto", "automatic", "yourself", "free", "back"}:
        return {"do": "feel", "emotion": "auto"}
    felt = next((FEELINGS[word] for word in words if word in FEELINGS), "")
    if felt:
        return {"do": "feel", "emotion": felt, "for": 0 if have & {"stay", "keep", "always", "forever"} else 15}
    if have & {"look", "looking", "gaze", "stare", "staring", "swap", "swapping", "auto", "middle", "center", "centre"}:
        return {"do": "set", "feature": "swap", "on": on}
    if have & {"hand", "hands", "tracking"}:
        return {"do": "set", "feature": "hands", "on": on}
    if "camera" in have:                          # track me / follow my head / hold still
        if have & {"still", "hold", "freeze", "stop", "pause"}:
            return {"do": "camera", "mode": "still"}
        if have & {"head", "headset", "look", "looking"}:
            return {"do": "camera", "mode": "head"}
        if have & {"track", "tracking", "me", "center", "centre", "follow", "following", "find"}:
            return {"do": "camera", "mode": "track"}
    if have & {"detail", "details", "numbers", "info", "information"}:
        return {"do": "set", "feature": "details", "on": on}
    if have & {"screen", "screens", "monitor", "display"}:   # which PC screen the panel shows
        if have & {"two", "2", "to", "too", "second", "small", "smaller", "little", "robot"}:
            return {"do": "screen", "n": 2}
        if have & {"one", "1", "won", "first", "big", "bigger", "main", "large", "computer", "pc"}:
            return {"do": "screen", "n": 1}
    for word in words:
        if word in PANELS and have & {"show", "main", "focus", "open", "bring", "switch", "go", "put"}:
            return {"do": "focus", "panel": PANELS[word]}
    if have & MENU_CLOSE or "put away" in joined or "put it away" in joined:
        return {"do": "close"}                    # "agent, close" / "done" / "exit": the menu goes away
    return {"do": "unknown"}


TUNE_PORT = 11012             # run_teleop.py takes live changes here (tools/teleop_tune.py)
JOINT_KINDS = {"pitch": "pitch", "roll": "roll", "yaw": "yaw", "jaw": "yaw", "yah": "yaw", "elbow": "elbow",
               "wrist": "wrist"}   # Whisper writes "yaw" as "jaw" now and then
MORE = {"more", "increase", "raise", "up", "stronger", "higher", "add", "boost"}
LESS = {"less", "decrease", "lower", "down", "weaker", "reduce", "drop"}


def parse_tune(words: list[str], have: set[str]) -> dict | None:
    """"more power left yaw", "less gravity right pitch", "flip left wrist", "less
    sensitive", "undo that" -> a change for run_teleop; None when it is none of those."""
    side = "left" if "left" in have else "right" if "right" in have else ""
    kind = next((JOINT_KINDS[w] for w in words if w in JOINT_KINDS), "")
    joint = f"{side} {kind}".strip()
    step = 1 if have & MORE else -1 if have & LESS else 0
    if have & {"power", "strength", "torque"} and step:
        return {"do": "tune", "what": "power", "joint": joint, "step": step}
    if "gravity" in have and step:
        return {"do": "tune", "what": "gravity", "joint": joint, "step": step}
    if have & {"sensitive", "sensitivity"} and step:
        return {"do": "tune", "what": "scale", "step": step}
    if have & {"flip", "reverse", "invert"} and kind:
        return {"do": "tune", "what": "flip", "joint": joint}
    if "undo" in have:
        return {"do": "tune", "what": "undo"}
    return None


def send_tune(command: dict) -> dict:
    """A change to the running run_teleop.py, and what it said back."""
    link = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    link.settimeout(1.5)
    try:
        link.sendto(json.dumps({k: v for k, v in command.items() if k != "do"}).encode(), ("127.0.0.1", TUNE_PORT))
        reply = json.loads(link.recvfrom(65536)[0])
        return {"ok": bool(reply.get("ok")), "said": str(reply.get("said", ""))[:160]}
    except (OSError, ValueError):
        return {"ok": False, "said": "run_teleop did not answer (is it running, and new enough?)"}
    finally:
        link.close()


RECORDER_PORT = 11015         # recorder.py takes episode commands here (tools/lfd/record_ctl.py)
RECORDER_TIMEOUT = 4.0        # s; an end is answered before its save, but a start still syncs episode.json
EPISODE_WORDS = {"episode", "episodes", "recording", "recordings"}   # a sentence with one never tunes the arms
EPISODE_GOOD = {"success", "successful", "succeeded", "good", "pass", "passed"}
EPISODE_BAD = {"fail", "failed", "failure", "bad"}
EPISODE_ABORT = {"abort", "aborted", "cancel", "cancelled", "canceled"}


def parse_episode(have: set[str]) -> dict | None:
    """"start episode", "end episode success", "abort episode", "discard episode", "episode
    status" -> a command for recorder.py; None when it is none of those.

    "stop" never is one: said to this robot, stop means the motors."""
    if "stop" in have:
        return None
    if not have & {"episode", "episodes"}:
        if {"start", "recording"} <= have:                    # "start recording"
            return {"do": "episode", "cmd": "start"}
        return None
    if have & {"discard", "delete"}:
        return {"do": "episode", "cmd": "discard"}
    if have & EPISODE_ABORT:
        return {"do": "episode", "cmd": "end", "outcome": "aborted"}
    if "end" in have:
        # The outcome labels the demonstration: both, neither or "not good" is asked again, not guessed.
        good, bad = bool(have & EPISODE_GOOD), bool(have & EPISODE_BAD)
        sure = not have & {"not", "no"}
        outcome = "success" if good and not bad and sure else "failure" if bad and not good and sure else None
        return {"do": "episode", "cmd": "end", "outcome": outcome}
    if have & {"start", "new"}:
        return {"do": "episode", "cmd": "start"}
    if "status" in have:
        return {"do": "episode", "cmd": "status"}
    return None


def send_episode(command: dict) -> dict:
    """An episode command for recorder.py, and what it said back."""
    link = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    link.settimeout(RECORDER_TIMEOUT)
    try:
        # connected, so a recorder that is not running is refused at once: a timeout means a busy one
        link.connect(("127.0.0.1", RECORDER_PORT))
        link.send(json.dumps({k: v for k, v in command.items() if k != "do"}).encode())
        reply = json.loads(link.recv(65536))
        return {"ok": bool(reply.get("ok")), "said": str(reply.get("said", ""))[:160]}
    except socket.timeout:                  # before OSError, which it is
        return {"ok": False, "said": f"no answer from the recorder in {RECORDER_TIMEOUT:g} s - it may be busy: "
                                     "say \"agent, episode status\" before saying it again"}
    except ConnectionRefusedError:
        return {"ok": False, "said": "the recorder is not running"}
    except (OSError, ValueError, AttributeError):
        return {"ok": False, "said": "the recorder did not answer properly"}
    finally:
        link.close()


def run_episode(command: dict) -> dict:
    """"agent, start episode" and the rest -> what to show and say: recorder.py's answer.

    An end without an outcome is not sent: a guessed outcome would mislabel the demonstration."""
    if command.get("cmd") == "end" and command.get("outcome") is None:
        return {"do": "recorder", "ok": False, "said": "say end episode success or end episode fail"}
    reply = send_episode(command)
    return {"do": "recorder", "ok": reply["ok"], "said": reply["said"]}


def send_face(command: dict) -> dict:
    """A mood (or the spark) for the robot's face, face_display.py, and what it said back."""
    message = {"type": "face", "show": command["show"]} if command.get("show") \
        else {"type": "face", "feel": command.get("emotion", "auto"), "for": command.get("for", 15)}
    link = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    link.settimeout(1.5)
    try:
        link.sendto(json.dumps(message).encode(), ("127.0.0.1", FACE_PORT))
        reply = json.loads(link.recvfrom(4096)[0])
        return {"ok": bool(reply.get("ok")), "said": str(reply.get("said", ""))[:120]}
    except (OSError, ValueError):
        return {"ok": False, "said": "the robot's face did not answer (bringup window `face`)"}
    finally:
        link.close()


def strip_wake(text: str, wake: str) -> str:
    """'Voice, open the camera' -> 'Open the camera'."""
    if not wake:
        return text
    rest = re.sub(rf"^\W*{re.escape(wake)}\b[\s,.!?:;-]*", "", text, flags=re.IGNORECASE).strip()
    return rest[:1].upper() + rest[1:] if rest else rest


def fix_heard(text: str) -> str:
    """'open up cloud with this session' -> 'open up Claude with this session' (SOUNDALIKES)."""
    for pattern, word in SOUNDALIKES:
        text = pattern.sub(word, text)
    return text


def track_floor(floor: float, level: float) -> float:
    """The room's noise: falls fast to anything quieter, creeps up slowly, so talking
    while nobody is dictating does not raise it much."""
    if floor <= 0:
        return level
    return floor * 0.9 + level * 0.1 if level < floor else floor * 0.995 + level * 0.005


def keep(text: str) -> None:
    """Every finished sentence, whether or not tmux could deliver it."""
    try:
        with TRANSCRIPT.open("a") as handle:
            handle.write(f"{time.strftime('%H:%M:%S')} {text}\n")
    except OSError:
        pass


class Reporter:
    """Tells the headset page what the microphone is doing, through the bridge relay."""

    def __init__(self, port: int, wake: str = ""):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = ("127.0.0.1", port)
        self.wake = wake                    # every message carries it, so the page can show it

    def say(self, state: str, text: str = "", **extra) -> None:
        payload = {"type": "voice", "state": state, "text": text, "wake": self.wake, "t": time.time()}
        payload.update(extra)
        try:
            self.sock.sendto(json.dumps(payload).encode(), self.addr)
        except OSError:
            pass


class Whisper:
    """Whisper on the finished utterance, while Vosk keeps showing the live partials.

    Vosk is fast enough to put words on the headset panel as they are spoken, but it
    mishears whatever it was not trained on - "Claude Code" arrives as "the clock
    code". Whisper reads the whole sentence at once, which is better and slower, so
    it runs at the end on the audio that is being kept for the wav anyway. Whatever
    it cannot do falls back to Vosk rather than losing the sentence.
    """

    def __init__(self, name: str, beam: int, prompt: str):
        self.name = name
        self.beam = beam
        self.prompt = prompt
        self.model = None

    def load(self) -> str:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            return ("faster-whisper is missing:  "
                    "uv pip install --python .venv/bin/python faster-whisper")
        try:
            self.model = WhisperModel(self.name, device="cpu", compute_type="int8",
                                      cpu_threads=os.cpu_count() or 4)
        except Exception as exc:                    # a bad name, no disk, no network on first use
            return f"{self.name} could not be loaded: {str(exc).splitlines()[0]}"
        return ""

    def hear(self, raw: bytes) -> str:
        if self.model is None or not raw:
            return ""
        try:
            import numpy as np
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            segments, _ = self.model.transcribe(audio, language="en", beam_size=self.beam,
                                                initial_prompt=self.prompt or None,
                                                condition_on_previous_text=False, vad_filter=True)
            return " ".join(segment.text.strip() for segment in segments).strip()
        except Exception:
            return ""                               # Vosk's version is the fallback


def load_model(path: Path):
    try:
        from vosk import KaldiRecognizer, Model, SetLogLevel
    except ImportError:
        raise SystemExit("Vosk is missing:  uv pip install --python .venv/bin/python vosk")
    if not path.exists():
        raise SystemExit(
            f"No speech model at {path}. Fetch the small English one (68 MB):\n"
            "  mkdir -p ~/.cache/vosk && cd ~/.cache/vosk\n"
            "  curl -LO https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip\n"
            "  unzip vosk-model-small-en-us-0.15.zip")
    SetLogLevel(-1)
    return Model(str(path)), KaldiRecognizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", default="auto",
                        help="tmux pane to type into; 'auto' finds the pane running Claude")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--whisper", default="base.en", metavar="MODEL",
                        help="model that reads the finished sentence: base.en, small.en, "
                             "distil-small.en, or 'none' to leave it to Vosk")
    parser.add_argument("--whisper-beam", type=int, default=1, metavar="N",
                        help="wider is a little better and a lot slower")
    parser.add_argument("--whisper-prompt", default=WHISPER_PROMPT,
                        help="words to expect, so the names in this project are not misheard")
    parser.add_argument("--listen-port", type=int, default=11007, help="audio from quest_bridge.py")
    parser.add_argument("--status-port", type=int, default=11006, help="where the headset page listens")
    parser.add_argument("--no-submit", action="store_true", help="type the text but do not press Enter")
    parser.add_argument("--submit-delay", type=float, default=0.25, metavar="SECONDS",
                        help="settle time between the words and the Enter (the text has to reach the "
                             "screen first, or Claude Code takes the Enter for a newline)")
    parser.add_argument("--wake", default="voice", metavar="WORD",
                        help="say this, then speak, to dictate hands-free; 'off' for the trigger only")
    parser.add_argument("--agent", default="agent", metavar="WORD",
                        help="say this, then a command, to drive the rig itself (never typed to Claude); "
                             "'off' to disable")
    parser.add_argument("--agent-silence", type=float, default=2.0, metavar="SECONDS",
                        help="a spoken command ends after this long without speech")
    parser.add_argument("--silence", type=float, default=5.0, metavar="SECONDS",
                        help="a hands-free sentence ends after this long without speech")
    parser.add_argument("--max-seconds", type=float, default=90.0, metavar="SECONDS",
                        help="a hands-free sentence never runs longer than this")
    parser.add_argument("--self-test", metavar="TEXT", help="type TEXT and exit, without any audio")
    args = parser.parse_args()

    typist = Typist(args.target, submit=not args.no_submit, settle=args.submit_delay)
    if args.self_test:
        problem = typist.type(args.self_test)
        print(problem or f"typed into tmux {args.target!r}: {args.self_test!r}", flush=True)
        return 1 if problem else 0

    problem = typist.available()
    if problem:
        print("Nothing to type into yet:", flush=True)
        print("  " + problem, flush=True)
        print(f"\nCarrying on: every sentence is printed here and appended to {TRANSCRIPT},", flush=True)
        print("and typing starts by itself as soon as a Claude pane appears.", flush=True)
    else:
        print("Typing into " + where(typist), flush=True)
    announced = problem

    model, recognizer_class = load_model(args.model)
    recognizer = recognizer_class(model, 16000)
    recognizer.SetWords(False)
    wake = args.wake.strip().lower()
    if wake in ("off", "none", ""):
        wake = ""
    # A second recogniser that can only hear the wake word or "something else": cheap
    # enough to run on everything the headset hears, and hard to fool with other words.
    agent = args.agent.strip().lower()
    if agent in ("off", "none", ""):
        agent = ""
    words_heard = [w for w in (wake, agent) if w]
    wake_rec = None
    if words_heard:
        wake_rec = recognizer_class(model, 16000, json.dumps(words_heard + ["[unk]"]))
        wake_rec.SetWords(True)
    reporter = Reporter(args.status_port, wake)

    whisper = None
    if args.whisper.strip().lower() not in ("none", "off", ""):
        whisper = Whisper(args.whisper, args.whisper_beam, args.whisper_prompt)
        print(f"Loading {args.whisper} (the first time also downloads it)...", flush=True)
        trouble = whisper.load()
        if trouble:
            whisper = None
            print(f"  {trouble}", flush=True)
            print("  carrying on with Vosk alone; expect names to come out wrong.", flush=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:  # speech arrives faster than it is recognised; give it room
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
    except OSError:
        pass
    sock.bind(("127.0.0.1", args.listen_port))
    sock.settimeout(0.25)
    print(f"Dictation ready. Live words from {args.model.name}"
          + (f", sentences from {whisper.name}." if whisper else " alone."), flush=True)
    print("Hold the right trigger in the headset and speak; release to send.", flush=True)
    if wake:
        print(f"Or say {wake!r}, wait for the beep, and speak: it sends after "
              f"{args.silence:g} s of silence.", flush=True)
    if agent:
        print(f"Say {agent!r} and a command to drive the rig (never typed to Claude): {AGENT_HELP}", flush=True)

    def announce(typist: Typist) -> None:
        """Claude is usually started after this is; notice when it turns up."""
        nonlocal announced
        problem = typist.available()
        if problem == announced:
            return
        announced = problem
        if problem:
            print(f"  lost the target: {problem.splitlines()[0]}", flush=True)
            reporter.say("error", problem.splitlines()[0])
        else:
            print(f"  now typing into {where(typist)}", flush=True)
            reporter.say("ready", f"typing into {where(typist)}")

    last_look = time.monotonic()
    rms = 0
    told_silent = False
    listening = False
    spoken = 0
    last_partial = ""
    last_audio = 0.0
    heard: list[str] = []      # segments Vosk closed while the trigger was still held
    raw = bytearray()          # the utterance itself, kept for LAST_AUDIO
    peak = 0
    mode = ""                  # what started this sentence: "trigger" or "wake"
    addressed = ""             # the session this sentence is for: the target when it began
    started_at = 0.0
    last_speech = 0.0          # when the sentence last had speech in it; 0 = none yet
    speech_end = 0             # bytes of raw up to the last speech
    floor_rms = 0.0            # the room's own noise, learned while nobody is dictating
    recent: collections.deque = collections.deque()   # (offset, audio) heard while waiting
    wake_fed = 0               # bytes the wake recogniser has had since it was reset

    jobs: queue.Queue = queue.Queue()

    def babbling(text: str) -> bool:
        """Whisper fills silence with a repeated word; a held trigger is not a sentence."""
        words = [w.strip(".,!?").lower() for w in text.split()]
        return len(words) >= BABBLE and len(set(words)) == 1

    def deliver(text: str, kept: bytes, peak: int, seconds: float, rms: int, how: str, session: str) -> None:
        """Whisper, then tmux. Off the socket loop: this takes about as long as the
        sentence did, and speech arriving meanwhile would be lost."""
        nonlocal told_silent
        source = args.model.name
        if rms < QUIET_RMS:
            # Loud enough somewhere, quiet throughout: a click, a bump, a held trigger.
            why = (f"{seconds:.1f}s at peak {peak} but loudness only {rms} - nothing was said "
                   "(is the trigger stuck?)")
            reporter.say("nothing", why)
            print(f"  ignored: {why}", flush=True)
            return
        if whisper is not None and peak >= QUIET_PEAK:
            reporter.say("thinking", "reading it back")
            better = whisper.hear(kept)
            if better:                      # Vosk keeps the sentence when Whisper has nothing
                text, source = better, whisper.name
        fixed = fix_heard(text)
        if fixed != text:                   # the words as heard, in case a fix was wrong
            print(f"  (heard as: {text})", flush=True)
            text = fixed
        if how == "wake":                   # said in one breath, the wake word is in the audio
            text = strip_wake(text, wake)
        if how == "trigger" and agent and re.match(rf"^\W*{re.escape(agent)}\b", text, re.IGNORECASE):
            how = "agent"                   # held the trigger and said "agent, ...": a command too
        if how == "agent":                  # the rig's own commands: nothing reaches Claude
            text = strip_wake(text, agent)
            command = parse_command(text)
            if command.get("do") in ("open_session", "close_session", "list_sessions"):
                try:
                    import claude_sessions
                    command = claude_sessions.by_voice(command, current=on_screen(typist.session))
                    if command["do"] in ("session_opened", "session_reopened"):
                        if typist.session == "@":              # calibration mode: talk to what the
                            claude_sessions.show(command["name"])  # PC shows, so show it there
                        else:
                            typist.session = command["name"]   # developer view: dictation follows it
                    elif command["do"] == "session_closed" and typist.session == command["name"]:
                        typist.session = ""                    # back to the main one
                except Exception as exc:
                    command = {"do": "session_failed", "why": str(exc)[:80]}
            elif command.get("do") == "tune":
                reply = send_tune(command)
                command = {"do": "tuned", "ok": reply["ok"], "said": reply["said"]}
            elif command.get("do") == "feel":               # the robot's face
                reply = send_face(command)
                command = {"do": "felt", "ok": reply["ok"], "said": reply["said"]}
            elif command.get("do") == "episode":            # recorder.py: start, end, discard, status
                command = run_episode(command)
            elif command.get("do") == "escape":             # on the session dictation goes to
                pane, problem = typist.resolve(on_screen(typist.session))
                try:
                    import claude_sessions
                    cancelled = claude_sessions.escape(pane) if pane else ""
                except Exception:
                    cancelled = ""
                command = {"do": "escaped", "what": cancelled[:80]} if cancelled \
                    else {"do": "escaped", "what": "", "why": problem.splitlines()[0][:80] if problem else ""}
            reporter.say("command", text, command=command)
            print(f"  agent: {text!r} -> {command}", flush=True)
            return
        if not text:
            if peak == 0:
                why = (f"{seconds:.1f}s of digital silence (every sample zero): the headset is sending an "
                       "empty microphone, not a quiet room")
            elif peak < QUIET_PEAK:
                why = (f"{seconds:.1f}s of near-silence (peak {peak}/32767): the microphone is barely "
                       "picking anything up")
            else:
                why = f"{seconds:.1f}s of audio, peak {peak}/32767, but no words came out"
            reporter.say("nothing", why)
            print(f"  nothing recognised: {why}", flush=True)
            if peak == 0 and not told_silent:
                told_silent = True
                print(SILENT_MIC, flush=True)
            print(f"  the audio is in {LAST_AUDIO} if you want to hear it", flush=True)
            return
        if babbling(text):
            reporter.say("nothing", "one word over and over: silence, not speech")
            print(f"  ignored (one word repeated): {text[:60]}", flush=True)
            return
        if text.lower().strip(" .!?,") in DISCARD:
            reporter.say("discarded", text)
            print(f"  discarded: {text}", flush=True)
            return
        keep(text)
        trouble = typist.type(text, session)
        if trouble:
            reporter.say("error", trouble.splitlines()[0])
            print(f"  {trouble}\n  text was: {text}", flush=True)
        else:
            reporter.say("sent", text)
            print(f"  typed [{source}] to {session or 'the main session'}: {text}", flush=True)

    def worker() -> None:
        while True:
            job = jobs.get()
            try:
                deliver(*job)
            finally:
                jobs.task_done()

    threading.Thread(target=worker, daemon=True).start()

    def reset_wake() -> None:
        """Back to waiting for the wake word, with a clean slate."""
        nonlocal wake_fed
        recent.clear()
        wake_fed = 0
        if wake_rec is not None:
            wake_rec.Reset()

    def speech_level() -> float:
        return max(SPEECH_RMS, 3.0 * floor_rms)

    def take(chunk: bytes, now: float, seeded: bool = False) -> None:
        """One piece of the sentence being dictated."""
        nonlocal spoken, last_audio, peak, rms, last_speech, speech_end, last_partial
        spoken += len(chunk)
        last_audio = now
        raw.extend(chunk[:MAX_KEPT - len(raw)])
        chunk_peak, chunk_rms = loudness(chunk)
        peak = max(peak, chunk_peak)
        rms = max(rms, chunk_rms)
        # The start beep comes back through the microphone; it is not you talking.
        if chunk_rms >= speech_level() and (seeded or now - started_at >= IGNORE_START_S):
            last_speech = now
            speech_end = len(raw)
        if recognizer.AcceptWaveform(chunk):
            piece = json.loads(recognizer.Result()).get("text", "").strip()
            if piece:                     # thrown away before, which lost whole sentences
                heard.append(piece)
                reporter.say("partial", " ".join(heard))
        else:
            partial = json.loads(recognizer.PartialResult()).get("partial", "")
            if partial and partial != last_partial:
                last_partial = partial
                reporter.say("partial", partial)

    def begin(how: str, seed: bytes = b"") -> None:
        nonlocal listening, mode, spoken, peak, rms, last_audio, last_partial
        nonlocal started_at, last_speech, speech_end, addressed
        listening, mode = True, how
        # Where this sentence goes is settled now. Glancing at another panel during the
        # silence that ends it, or while Whisper reads it, sent it there instead.
        addressed = on_screen(typist.session)
        spoken = peak = rms = speech_end = 0
        last_audio = last_speech = 0.0
        last_partial = ""
        started_at = time.monotonic()
        heard.clear()
        raw.clear()
        recognizer.Reset()
        reporter.say("listening", how)    # the page beeps for a hands-free start
        print(f"  listening ({'heard ' + repr(agent if how == 'agent' else wake) if how != 'trigger' else 'trigger'})...",
              flush=True)
        if seed:
            take(seed, started_at, seeded=True)

    def finish(why: str) -> None:
        """End the sentence: keep what Vosk heard, hand the rest to Whisper."""
        nonlocal listening, last_partial
        listening = False
        last_partial = ""
        seconds = spoken / 32000.0
        # Vosk closes a segment at every pause and hands it over there and then;
        # FinalResult() holds only what came after the last one, which is usually
        # nothing, because the pause that closed it was the end of the sentence.
        tail = json.loads(recognizer.FinalResult()).get("text", "").strip()
        text = " ".join(heard + ([tail] if tail else [])).strip()
        heard.clear()
        kept = bytes(raw)
        raw.clear()
        if mode in ("wake", "agent") and speech_end:   # the silence that ended it is not worth reading
            kept = kept[:min(len(kept), speech_end + 2 * int(0.6 * 16000))]
        save_audio(kept)
        print(f"  ended: {why} ({seconds:.1f}s)", flush=True)
        reset_wake()
        if mode == "trigger" and seconds < MIN_TRIGGER_S and not text:
            reporter.say("cancelled", why)    # a tap: a slip, or the double-tap toggle
            return
        if mode in ("wake", "agent") and not last_speech:
            reporter.say("nothing", f"nothing said within {args.silence:g} s of the beep")
            print(f"  nothing said within {args.silence:g} s of the beep", flush=True)
            return
        reporter.say("heard", why)            # the end beep now, not after Whisper
        jobs.put((text, kept, peak, seconds, rms, mode, addressed))

    def check_end(now: float) -> None:
        if not listening:
            return
        if last_audio and now - last_audio > GAP_END:
            finish(f"no audio from the headset for {GAP_END:g} s")
        elif not last_audio and now - started_at > GAP_END:
            finish(f"no audio arrived in {GAP_END:g} s")     # waited forever before
        elif mode in ("wake", "agent"):
            quiet = args.agent_silence if mode == "agent" else args.silence
            if now - started_at > args.max_seconds:
                # Still talking: send this piece and carry straight on with the next, to
                # the same session - a long report was cut at the limit, the rest lost.
                talking = mode == "wake" and last_speech and now - last_speech < quiet
                finish(f"{args.max_seconds:g} s limit" + ("; still talking, carrying on" if talking else ""))
                if talking:
                    begin("wake")
            elif last_speech and now - last_speech >= quiet:
                finish(f"{quiet:g} s of silence")
            elif not last_speech and now - started_at >= args.silence + IGNORE_START_S:
                finish("nothing said after the beep")

    def woke(result: dict):
        """(which word, the audio after it) if this segment starts with the wake or agent
        word; else None."""
        words = result.get("result") or []
        if not words or words[0].get("word") not in words_heard or words[0].get("conf", 0) < WAKE_CONF:
            return None
        cut = 2 * int(words[0].get("end", 0) * 16000)   # where the word ends, in bytes
        seed = b"".join(chunk[max(0, cut - at):] for at, chunk in recent if at + len(chunk) > cut)
        return ("agent" if words[0]["word"] == agent else "wake"), seed

    # Stopping (Ctrl+C, a restart, `kill`) finishes the sentence first: a restart once
    # dropped a whole minute of speech that Whisper was still reading, and the native
    # libraries then hung the process instead of letting it exit. A second Ctrl+C goes
    # at once, whatever is pending.
    stopping = threading.Event()

    def stop_asked(_signum, _frame) -> None:
        if stopping.is_set():
            os._exit(1)
        stopping.set()

    signal.signal(signal.SIGINT, stop_asked)
    signal.signal(signal.SIGTERM, stop_asked)

    while not stopping.is_set():
        now = time.monotonic()
        check_end(now)
        if now - last_look > 5.0:
            last_look = now
            announce(typist)
            if not listening:
                reporter.say("idle")          # alive, and which wake word to show
        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            continue
        except KeyboardInterrupt:
            break
        except OSError:
            continue
        now = time.monotonic()

        if data.startswith(AUDIO_MAGIC):
            chunk = data[len(AUDIO_MAGIC):]
            if listening:
                take(chunk, now)
                continue
            # Waiting: learn the room's noise, and listen for the wake word. The page
            # only streams while waiting when hands-free mode is on.
            floor_rms = track_floor(floor_rms, loudness(chunk)[1])
            recent.append((wake_fed, chunk))
            wake_fed += len(chunk)
            while recent and wake_fed - recent[0][0] > WAKE_BUFFER_S * 32000:
                recent.popleft()
            if wake_rec is not None and wake_rec.AcceptWaveform(chunk):
                heard_word = woke(json.loads(wake_rec.Result()))
                if heard_word is not None:
                    begin(*heard_word)
            continue

        try:
            message = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            continue
        if message.get("type") != "voice":
            continue
        action = message.get("action")
        if action == "target":
            wanted = str(message.get("session") or "")[:3].upper()
            if wanted != typist.session:
                typist.session = wanted
                print("  dictation now goes to " + ("whichever session the PC's screen shows" if wanted == "@"
                                                     else f"session {wanted or '(the main one)'}"), flush=True)
            continue
        if action == "start":
            if listening and mode in ("wake", "agent"):
                mode = "trigger"              # pressed mid-sentence: it now ends on release
                print("  trigger pressed: this sentence now ends when you let go", flush=True)
            elif listening:
                # A second start with no stop between used to wipe the first part.
                print("  pressed again while recording (a stop went missing): kept the first part", flush=True)
            else:
                begin("trigger", b"".join(chunk for _, chunk in recent)[-2 * int(SEED_S * 16000):])
        elif action == "stop" and listening and mode == "trigger":
            why = str(message.get("why") or "released")
            finish("trigger " + why if why != "released" else "trigger released")
    if listening:
        finish("stopping")                    # what was said so far is kept, not dropped
    if jobs.unfinished_tasks:
        print("  stopping: finishing the sentence Whisper is reading first (Ctrl+C again to drop it)",
              flush=True)
        jobs.join()
    print("Stopped.", flush=True)
    os._exit(0)                               # past the native libraries' teardown, which hung


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
        sys.exit(0)
