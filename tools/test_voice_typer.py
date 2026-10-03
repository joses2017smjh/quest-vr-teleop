"""Drive voice_typer.py the way the headset does, without a headset or a voice.

A stand-in Vosk module (written to a temp dir) behaves like the real one where it
matters: the wake-word recogniser closes a segment at each pause and reports word
timings, and the dictation recogniser returns a sentence for any loud audio. Audio
is sent in real time, because the silence and gap timers run on the clock.

  .venv/bin/python tools/test_voice_typer.py

Needs tmux. Uses its own ports and its own tmux session; it never types into the
real Claude pane.
"""

from __future__ import annotations

import json
import math
import os
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENV = REPO / ".venv/bin/python"
AUDIO_PORT, STATUS_PORT = 11127, 11126
SESSION = "bhl-voicetest"
RATE, CHUNK = 16000, 1365            # what the page sends: 4096 frames at 48 kHz, as 16 kHz
SENTENCE = "open the stereo camera"

FAKE_VOSK = r'''
"""A stand-in for Vosk. Loud is speech; quiet is silence."""
import json, os, struct


def SetLogLevel(level):
    pass


class Model:
    def __init__(self, path):
        self.path = path


def _loud(chunk):
    n = len(chunk) // 2
    if not n:
        return False
    samples = struct.unpack("<%dh" % n, chunk[:2 * n])
    return (sum(v * v for v in samples) / n) ** 0.5 > 1000


class KaldiRecognizer:
    def __init__(self, model, rate, grammar=None):
        self.grammar = json.loads(grammar) if grammar else None
        self.Reset()

    def SetWords(self, on):
        pass

    def Reset(self):
        self.fed = 0             # bytes since the last reset: the clock word times use
        self.loud_from = None    # where the current stretch of speech began
        self.loud_to = None
        self.quiet = 0
        self.any_loud = False
        self.result = {"text": ""}

    def AcceptWaveform(self, chunk):
        start = self.fed
        self.fed += len(chunk)
        if _loud(chunk):
            self.any_loud = True
            if self.loud_from is None:
                self.loud_from = start
            self.loud_to = self.fed
            self.quiet = 0
            return False
        if self.loud_from is None:
            return False
        self.quiet += len(chunk)
        if self.grammar is None or self.quiet < 2 * 16000 * 0.3:
            return False
        # a pause after speech: the wake recogniser closes the segment here
        t0, t1 = self.loud_from / 32000, self.loud_to / 32000
        word = os.environ.get("FAKE_WAKE") or self.grammar[0]
        if t1 - t0 < 0.8:        # a short burst: the wake word alone
            self.result = {"text": word, "result": [{"word": word, "conf": 0.95, "start": t0, "end": t1}]}
        else:                    # the wake word and a sentence in one breath
            self.result = {"text": word + " [unk]",
                           "result": [{"word": word, "conf": 0.95, "start": t0, "end": t0 + 0.5},
                                      {"word": "[unk]", "conf": 1.0, "start": t0 + 0.5, "end": t1}]}
        self.loud_from = None
        self.quiet = 0
        return True

    def Result(self):
        return json.dumps(self.result)

    def PartialResult(self):
        return json.dumps({"partial": ""})

    def FinalResult(self):
        text = os.environ.get("FAKE_SENTENCE", "") if self.any_loud and self.grammar is None else ""
        return json.dumps({"text": text})
'''


def tone(seconds: float, loud: bool) -> list[bytes]:
    amplitude = 8000 if loud else 3
    frames = int(RATE * seconds)
    audio = b"".join(struct.pack("<h", int(amplitude * math.sin(i * 0.3))) for i in range(frames))
    return [audio[i:i + 2 * CHUNK] for i in range(0, len(audio), 2 * CHUNK)]


class Rig:
    """A voice_typer, a fake Claude pane for it to type into, and the headset's side."""

    def __init__(self, workdir: Path, silence: float = 1.0, wake_word: str = "", sentence: str = SENTENCE,
                 max_seconds: float | None = None):
        subprocess.run(["tmux", "kill-session", "-t", SESSION], capture_output=True)
        fake_claude = workdir / "claude"
        if not fake_claude.exists():
            fake_claude.symlink_to("/bin/sleep")
        subprocess.run(["tmux", "new-session", "-d", "-s", SESSION, f"{fake_claude} 600"], check=True)
        self.status = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.status.bind(("127.0.0.1", STATUS_PORT))
        self.status.settimeout(0.05)
        env = dict(os.environ, PYTHONPATH=str(workdir), FAKE_SENTENCE=sentence, FAKE_WAKE=wake_word)
        self.log = open(workdir / "voice_typer.log", "w+")
        self.proc = subprocess.Popen(
            [str(VENV), str(REPO / "scripts/teleop/voice_typer.py"), "--target", SESSION,
             "--listen-port", str(AUDIO_PORT), "--status-port", str(STATUS_PORT),
             "--whisper", "none", "--silence", str(silence), "--agent-silence", str(silence),
             *(["--max-seconds", str(max_seconds)] if max_seconds else []),
             "--submit-delay", "0.05"],
            cwd=REPO, env=env, stdout=self.log, stderr=subprocess.STDOUT, text=True)
        self.out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.notes: list[dict] = []
        time.sleep(2.0)

    def send(self, chunks: list[bytes]) -> None:
        for chunk in chunks:           # real time: 1365 frames at 16 kHz is 85 ms
            self.out.sendto(b"AUD0" + chunk, ("127.0.0.1", AUDIO_PORT))
            time.sleep(CHUNK / RATE)
            self.drain()

    def voice(self, action: str, why: str = "") -> None:
        message = {"type": "voice", "action": action}
        if why:
            message["why"] = why
        self.out.sendto(json.dumps(message).encode(), ("127.0.0.1", AUDIO_PORT))

    def drain(self) -> None:
        while True:
            try:
                self.notes.append(json.loads(self.status.recvfrom(65535)[0].decode()))
            except (socket.timeout, BlockingIOError):
                return

    def wait(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.drain()
            time.sleep(0.05)

    def pane(self) -> str:
        return subprocess.run(["tmux", "capture-pane", "-p", "-t", SESSION],
                              capture_output=True, text=True).stdout

    def said(self, state: str) -> list[dict]:
        return [n for n in self.notes if n.get("state") == state]

    def output(self) -> str:
        self.log.flush()
        self.log.seek(0)
        return self.log.read()

    def close(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        subprocess.run(["tmux", "kill-session", "-t", SESSION], capture_output=True)
        self.status.close()
        self.log.close()


def main() -> int:
    if subprocess.run(["which", "tmux"], capture_output=True).returncode:
        print("tmux is missing")
        return 2
    failures: list[str] = []

    # 0. Spoken rig commands: what each phrase does.
    sys.path.insert(0, str(REPO / "scripts/teleop"))
    from voice_typer import parse_command
    for said, wanted in (("developer", {"do": "view", "view": "developer"}),
                         ("exit developer mode", {"do": "view", "view": "normal"}),
                         ("developer mode off", {"do": "view", "view": "normal"}),
                         ("normal", {"do": "view", "view": "normal"}),
                         ("calibration mode", {"do": "view", "view": "normal"}),
                         ("vision off", {"do": "set", "feature": "vision", "on": False}),
                         ("show the detections", {"do": "set", "feature": "vision", "on": True}),
                         ("close session a", {"do": "close_session", "name": "A"}),
                         ("close the demo video session", {"do": "close_session", "name": ""}),
                         ("save this session", {"do": "close_session"}),
                         ("open the demo video session", {"do": "open_session", "new": False}),
                         ("open a new claude session", {"do": "open_session", "new": True}),
                         ("open claude session called cam", {"do": "open_session", "name": "CAM"}),
                         ("which sessions are saved", {"do": "list_sessions"}),
                         ("what sessions are open", {"do": "list_sessions"}),
                         ("escape", {"do": "escape"}),
                         ("more power left yaw", {"do": "tune", "what": "power", "joint": "left yaw", "step": 1}),
                         ("more power left jaw", {"do": "tune", "what": "power", "joint": "left yaw", "step": 1}),
                         ("less sensitive", {"do": "tune", "what": "scale", "step": -1}),
                         ("flip the right wrist", {"do": "tune", "what": "flip", "joint": "right wrist"}),
                         ("cancel the dialog", {"do": "escape"}),
                         ("Close.", {"do": "close"}),
                         ("done", {"do": "close"}),
                         ("put it away", {"do": "close"}),
                         ("exit", {"do": "close"}),
                         ("stop", {"do": "unknown"}),              # "stop" never just closes a menu
                         # the robot's face: a mood, its own moods, the spark - and "look" alone
                         # still the looking swap, "focus" still a panel
                         ("look happy", {"do": "feel", "emotion": "happy", "for": 15}),
                         ("be angry", {"do": "feel", "emotion": "angry"}),
                         ("act surprised", {"do": "feel", "emotion": "surprised"}),
                         ("stay sleepy", {"do": "feel", "emotion": "sleepy", "for": 0}),
                         ("go to sleep", {"do": "feel", "emotion": "asleep"}),
                         ("your own moods", {"do": "feel", "emotion": "auto"}),
                         ("show the spark", {"do": "feel", "show": "spark"}),
                         ("face eyes", {"do": "feel", "show": "eyes"}),
                         ("i love the developer view", {"do": "view", "view": "developer"}),
                         ("stop looking swap", {"do": "set", "feature": "swap", "on": False}),
                         ("focus the camera", {"do": "focus", "panel": "camera"})):
        got = parse_command(said)
        if any(got.get(key) != value for key, value in wanted.items()):
            failures.append(f"'agent, {said}' does {got}, not {wanted}")
    if not failures:
        print("ok    spoken commands: developer in and out, vision on and off, sessions closed, listed, reopened, "
              "the menu closed, the robot's moods")

    # 0b. Whisper's sound-alikes of this project's words (real ones, from bhl_voice.txt),
    # and look-alikes that are ordinary words and must be left alone.
    from voice_typer import fix_heard
    before = len(failures)
    for heard, wanted in (("In my other cloud session, I think it's cloud session A",
                           "In my other Claude session, I think it's Claude session A"),
                          ("all the plot sessions on the other", "all the Claude sessions on the other"),
                          ("both the cloth sessions and the sensors", "both the Claude sessions and the sensors"),
                          ("a miniature iCloud symbol rotating", "a miniature Claude symbol rotating"),
                          ("agent open up cloud with this session", "agent open up Claude with this session"),
                          ("oh, cloud C is waiting for you", "oh, Claude C is waiting for you"),
                          ("the cloud code window", "the Claude Code window"),
                          ("moving the serval like up and down", "moving the servo like up and down"),
                          ("the servals twitch", "the servos twitch"),
                          ("address the C-S-C-R-C-P-Y question", "address the scrcpy question"),
                          ("the cloud is dark today", None), ("plot a graph of the torque", None),
                          ("the plot code draws it", None), ("a cloud I saw", None),
                          ("the cord is unplugged", None), ("the cat clawed the couch", None),
                          ("open the stereo camera", None)):
        got = fix_heard(heard)
        if got != (wanted or heard):
            failures.append(f"'{heard}' is put right as '{got}', not '{wanted or heard}'")
    if len(failures) == before:
        print("ok    sound-alikes: cloud/plot/cloth session -> Claude session, serval -> servo, "
              "C-S-C-R-C-P-Y -> scrcpy; the weather's clouds left alone")

    # 0c. Demonstration episodes for recorder.py: every phrase exactly, "stop" never one of
    # them (to this operator it means the motors), and the other commands what they were.
    before = len(failures)
    start, discard, status = ({"do": "episode", "cmd": "start"}, {"do": "episode", "cmd": "discard"},
                              {"do": "episode", "cmd": "status"})
    success, failure, aborted, unsure = ({"do": "episode", "cmd": "end", "outcome": outcome}
                                         for outcome in ("success", "failure", "aborted", None))
    for said, wanted in (("start episode", start), ("new episode", start), ("start recording", start),
                         ("Start a new episode.", start),
                         ("end episode success", success), ("end episode succeeded", success),
                         ("end episode good", success), ("end episode pass", success),
                         ("End episode, success.", success),
                         ("end episode fail", failure), ("end episode failed", failure),
                         ("end episode failure", failure), ("end episode bad", failure),
                         ("abort episode", aborted), ("cancel episode", aborted),
                         ("end episode aborted", aborted), ("episode cancelled", aborted),
                         ("Cancelled the episode.", aborted), ("canceled episode", aborted),
                         ("end episode", unsure), ("end episode not good", unsure),
                         ("end episode success no fail", unsure),
                         ("discard episode", discard), ("delete episode", discard), ("Delete the episode.", discard),
                         ("episode status", status)):
        got = parse_command(said)
        if got != wanted:
            failures.append(f"'agent, {said}' does {got}, not {wanted}")
    for said in ("stop", "stop episode", "stop the episode", "stop recording", "stop episode success",
                 "stop episode more power left yaw", "start", "status", "recording", "end", "cancel",
                 "new session"):
        got = parse_command(said)
        if got.get("do") == "episode":
            failures.append(f"'agent, {said}' became an episode command: {got}")
    # A sentence that names an episode never tunes the arms, whatever else is in it: the episode
    # command, or nothing ("undo episode" must not take back the last tuning change).
    for said, wanted in (("end episode success more power left yaw", success),
                         ("end episode fail, it needs more power", failure),
                         ("End episode success, the left elbow needed more power.", success),
                         ("start episode less sensitive", start), ("discard episode undo", discard),
                         ("episode status flip the right wrist", status),
                         ("start recording more gravity left pitch", start),
                         ("undo episode", {"do": "unknown"}), ("undo the last episode", {"do": "unknown"}),
                         ("stop episode more power left yaw", {"do": "unknown"}),
                         ("stop recording, less power right elbow", {"do": "unknown"})):
        got = parse_command(said)
        if got != wanted:
            failures.append(f"'agent, {said}' does {got}, not {wanted}")
    tunes = ("more power left yaw", "less power right elbow", "more gravity left pitch", "less sensitive",
             "more sensitivity", "flip the right wrist", "reverse left pitch", "undo that", "undo")
    for phrase in ("start episode", "new episode", "start recording", "end episode success", "end episode fail",
                   "end episode", "abort episode", "cancel episode", "discard episode", "episode status",
                   "stop episode", "stop recording", "undo episode", "episode", "recordings"):
        for tune in tunes:
            for said in (f"{phrase} {tune}", f"{tune} {phrase}", f"{phrase}, it needs {tune}"):
                got = parse_command(said)
                if got.get("do") == "tune":
                    failures.append(f"'agent, {said}' tunes the arms: {got}")
    for said, wanted in (("help", {"do": "help"}), ("what can you do", {"do": "help"}),
                         ("open a new claude session", {"do": "open_session", "new": True}),
                         ("close session a", {"do": "close_session", "name": "A"}),
                         ("which sessions are saved", {"do": "list_sessions"}),
                         ("more power left yaw", {"do": "tune", "what": "power", "joint": "left yaw", "step": 1}),
                         ("undo that", {"do": "tune", "what": "undo"}),
                         ("camera still", {"do": "camera", "mode": "still"}),
                         ("camera follow my head", {"do": "camera", "mode": "head"}),
                         ("camera track me", {"do": "camera", "mode": "track"}),
                         ("stop", {"do": "unknown"})):
        got = parse_command(said)
        if any(got.get(key) != value for key, value in wanted.items()):
            failures.append(f"'agent, {said}' now does {got}, not {wanted}")
    if len(failures) == before:
        print("ok    episodes: start/new episode, start recording, end episode success|fail, abort/cancel(led)/"
              "aborted, discard/delete, episode status; 'stop' never one; a sentence naming an episode never "
              "tunes the arms (15 x 9 mixes, both orders); sessions, tuning, camera, help unchanged")

    # 0d. What the headset hears back: recorder.py's reply, an end without an outcome asked
    # again without sending anything, and a recorder that is not there.
    import voice_typer
    before = len(failures)
    port = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    port.bind(("127.0.0.1", 0))
    port.settimeout(0.3)
    saved_port, voice_typer.RECORDER_PORT = voice_typer.RECORDER_PORT, port.getsockname()[1]
    heard: list[dict] = []
    try:
        unsure_reply = voice_typer.run_episode(parse_command("end episode"))
        try:
            heard.append(json.loads(port.recvfrom(4096)[0]))
        except socket.timeout:
            pass
        if heard or unsure_reply.get("said") != "say end episode success or end episode fail":
            failures.append(f"'end episode' without an outcome was sent ({heard}) or not asked again: {unsure_reply}")

        def recorder() -> None:
            try:
                data, sender = port.recvfrom(4096)
            except socket.timeout:
                return
            heard.append(json.loads(data))
            port.sendto(json.dumps({"ok": True, "said": "ep_0000 success: 61.0 s"}).encode(), sender)

        answering = threading.Thread(target=recorder)
        answering.start()
        reply = voice_typer.run_episode(parse_command("end episode success"))
        answering.join(timeout=3)
        if heard != [{"cmd": "end", "outcome": "success"}]:
            failures.append(f"the recorder was sent {heard}, not {{'cmd': 'end', 'outcome': 'success'}}")
        elif reply != {"do": "recorder", "ok": True, "said": "ep_0000 success: 61.0 s"}:
            failures.append(f"the recorder's answer is not passed on as it is: {reply}")
        # a recorder that is there but busy (an end's save, a start's sync) is not "not running"
        saved_timeout, voice_typer.RECORDER_TIMEOUT = voice_typer.RECORDER_TIMEOUT, 0.3
        try:
            busy = voice_typer.run_episode(parse_command("episode status"))
        finally:
            voice_typer.RECORDER_TIMEOUT = saved_timeout
        if busy.get("ok") is not False or not busy.get("said", "").startswith("no answer from the recorder") \
                or "episode status" not in busy["said"]:
            failures.append(f"a recorder slow to answer is not reported as such: {busy}")
        port.close()                                    # nobody listens there now
        asked = time.monotonic()
        gone = voice_typer.run_episode(parse_command("episode status"))
        if gone != {"do": "recorder", "ok": False, "said": "the recorder is not running"} \
                or time.monotonic() - asked > 1.0:
            failures.append(f"a recorder that is not running is not reported as such, at once: {gone} after "
                            f"{time.monotonic() - asked:.1f} s")
    finally:
        voice_typer.RECORDER_PORT = saved_port
        port.close()
    page = (REPO / "scripts/teleop/quest_bridge.html").read_text()
    if 'c.do === "recorder"' not in page:
        failures.append("quest_bridge.html has no runCommand case for {do: 'recorder'}: the headset never shows "
                        "the recorder's answer")
    if len(failures) == before:
        print("ok    voice_typer passes the recorder's answer on as {do: 'recorder'}, and quest_bridge.html shows "
              "it; 'end episode' alone is asked again, nothing sent; a busy recorder -> 'no answer ... episode "
              "status'; none -> 'the recorder is not running', at once")

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "vosk.py").write_text(FAKE_VOSK)

        # 1. Say the wake word, wait for the beep, speak, stop talking.
        rig = Rig(work)
        try:
            rig.send(tone(1.0, False) + tone(0.4, True) + tone(0.5, False))   # "voice", pause
            started = rig.said("listening")
            rig.send(tone(1.0, True) + tone(1.6, False))                     # the sentence, then silence
            rig.wait(1.0)
            if not started or started[-1].get("text") != "wake":
                failures.append("saying the wake word does not start a hands-free sentence")
            elif SENTENCE not in rig.pane().lower():
                failures.append(f"the hands-free sentence never reached Claude: {rig.output()[-300:]}")
            elif "1 s of silence" not in rig.output():
                failures.append("the hands-free sentence did not end on the silence timer")
            elif not rig.said("heard"):
                failures.append("the page is not told when listening stops (no end beep)")
            else:
                print("ok    wake word, a pause, a sentence, silence: typed, ended by 1 s of silence")
        finally:
            rig.close()

        # 2. The wake word and the sentence in one breath: nothing lost, no wake word typed.
        rig = Rig(work)
        try:
            rig.send(tone(1.0, False) + tone(1.6, True) + tone(1.8, False))
            rig.wait(1.0)
            typed = rig.pane().lower()
            if SENTENCE not in typed:
                failures.append(f"a sentence said straight after the wake word was lost: {rig.output()[-300:]}")
            elif "voice" in typed:
                failures.append(f"the wake word was typed along with the sentence: {typed.strip()!r}")
            else:
                print("ok    wake word and sentence in one breath: the sentence is kept, the wake word is not typed")
        finally:
            rig.close()

        # 3. A press that never delivers audio must end by itself, not wait forever for
        #    the next press to wipe it (the "listening... listening..." pair).
        rig = Rig(work, silence=5.0)
        try:
            rig.voice("start")
            rig.wait(5.0)
            if "no audio arrived" not in rig.output():
                failures.append("a trigger press with no audio waits forever")
            else:
                print("ok    a press that sends no audio ends by itself after 4 s")
        finally:
            rig.close()

        # 4. A second press with no stop between keeps the first part; a one-second Wi-Fi
        #    stall in the middle does not end the sentence (it used to after 0.5 s).
        rig = Rig(work, silence=5.0)
        try:
            rig.voice("start")
            rig.send(tone(0.6, True))
            rig.voice("start")                      # the stop went missing
            rig.wait(1.0)                           # nothing arrives for a second
            rig.send(tone(0.6, True))
            rig.voice("stop")
            rig.wait(1.5)
            out = rig.output()
            if "kept the first part" not in out:
                failures.append("a second start without a stop wiped the first part")
            elif "no audio from the headset" in out:
                failures.append("a one-second gap in the audio ended the sentence")
            elif SENTENCE not in rig.pane().lower():
                failures.append(f"the sentence spoken around the gap was not typed: {out[-300:]}")
            else:
                print("ok    a lost stop keeps the first part, and a 1 s Wi-Fi stall does not cut the sentence")
        finally:
            rig.close()

        # 6. "agent, stop the looking swap": a command for the rig, and nothing typed to Claude.
        rig = Rig(work, wake_word="agent", sentence="stop the looking swap")
        try:
            rig.send(tone(1.0, False) + tone(0.4, True) + tone(0.5, False))
            rig.send(tone(1.0, True) + tone(1.6, False))
            rig.wait(1.0)
            commands = rig.said("command")
            if not commands:
                failures.append(f"saying 'agent' and a command produced no command: {rig.output()[-300:]}")
            elif commands[-1].get("command") != {"do": "set", "feature": "swap", "on": False}:
                failures.append(f"the command was misread: {commands[-1]}")
            elif "looking swap" in rig.pane().lower():
                failures.append("an agent command was typed into Claude's pane")
            else:
                print("ok    'agent, stop the looking swap' becomes a command, and nothing is typed to Claude")
        finally:
            rig.close()

        # 5. A tap (the double-tap toggle, or a slip) types nothing and makes no end beep.
        rig = Rig(work, silence=5.0)
        try:
            rig.voice("start")
            rig.send(tone(0.2, False))
            rig.voice("stop")
            rig.wait(0.8)
            if not rig.said("cancelled") or rig.said("heard"):
                failures.append("a short tap is treated as a sentence")
            else:
                print("ok    a tap on the trigger is cancelled quietly")
        finally:
            rig.close()

        # 8. Talking past the length limit: that piece is sent, and the next carries on
        #    (a 90 s report was cut off at the limit and the rest of it lost).
        rig = Rig(work, silence=1.0, max_seconds=2.0)
        try:
            rig.send(tone(1.0, False) + tone(0.4, True) + tone(0.5, False))   # "voice", pause
            rig.send(tone(3.5, True) + tone(1.8, False))                     # talking on past 2 s
            rig.wait(1.5)
            pieces = rig.said("sent")
            if "carrying on" not in rig.output() or len(pieces) < 2:
                failures.append(f"talking past the limit lost the rest ({len(pieces)} piece(s) sent): {rig.output()[-300:]}")
            else:
                print("ok    talking past the length limit sends that piece and carries on with the next")
        finally:
            rig.close()

        # 7. Stopped mid-sentence (Ctrl+C, a restart): the sentence is still typed, and the
        #    process then exits - a restart once dropped a minute of speech and hung.
        rig = Rig(work, silence=5.0)
        try:
            rig.voice("start")
            rig.send(tone(1.2, True))
            rig.proc.send_signal(signal.SIGINT)     # still talking, trigger still held
            try:
                code = rig.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                code = None
            if code != 0:
                failures.append(f"Ctrl+C mid-sentence did not exit cleanly (exit {code}): {rig.output()[-300:]}")
            elif SENTENCE not in rig.pane().lower():
                failures.append(f"Ctrl+C mid-sentence dropped the sentence: {rig.output()[-300:]}")
            else:
                print("ok    stopped mid-sentence, it types what was said and then exits")
        finally:
            rig.close()

    print()
    for failure in failures:
        print("FAIL  " + failure)
    print("dictation is sound" if not failures else f"{len(failures)} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
