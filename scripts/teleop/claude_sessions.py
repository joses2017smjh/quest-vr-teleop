#!/usr/bin/env python3
"""Claude Code sessions in the rig's tmux session, each with a 1-3 letter name.

"agent, open claude session" (or "... called CAM") starts a new Claude in its own
tmux window, `c-<NAME>`, in this repo, on Opus 5.5 at max effort. The session
Claude was already running in (window "claude") is "M". In the headset's developer
view every session has its own panel, and dictation goes to the one you look at - or,
when you look at none, the one you used last. In the normal view there are no session
panels: dictation goes to the session the PC's own screen shows (`shown`).

Each panel is headed with a few words on what that session is about: Claude Code's
own title for it, cut short, or a label given by hand (a session can name its own
panel with `label here ...`; `label NAME` with no words clears it).

"agent, close session A" (or "... the demo video session", or "... this session")
saves a session and closes its window; its title is how it comes back: "agent, open
the demo video session" resumes that conversation where it stopped. "agent, saved
sessions" lists them. The main session, M, is never closed.

  .venv/bin/python scripts/teleop/claude_sessions.py list
  .venv/bin/python scripts/teleop/claude_sessions.py open [NAME]
  .venv/bin/python scripts/teleop/claude_sessions.py show NAME
  .venv/bin/python scripts/teleop/claude_sessions.py label NAME|here [WORDS...]
  .venv/bin/python scripts/teleop/claude_sessions.py shown        # which one the PC's screen shows
  .venv/bin/python scripts/teleop/claude_sessions.py close NAME   # save it, then close it
  .venv/bin/python scripts/teleop/claude_sessions.py saved        # the ones closed that way
  .venv/bin/python scripts/teleop/claude_sessions.py reopen WORDS...   # by words of its title
"""

from __future__ import annotations

import json
import os
import re
import string
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SESSION = "bhl"
MAIN = "M"                    # the window called "claude", made by bringup.py
MODEL = "claude-opus-5-5"
EFFORT = "max"
# Claude Code keeps each conversation in ~/.claude/projects/<this repo, dashed>/<id>.jsonl
TRANSCRIPTS = Path.home() / ".claude/projects" / re.sub(r"[^A-Za-z0-9]", "-", str(REPO))
SAVED = Path.home() / ".config/bhl/saved_sessions.json"   # sessions closed by voice, to reopen


def tmux(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["tmux", *args], capture_output=True, text=True)


def sessions() -> list[dict]:
    """[{"name", "window", "pane", "title", "label", "sid"}], the main one first, then in
    the order opened. The title is the one Claude Code gives its terminal; the label, one
    given with `label` (a tmux window option, so it goes when the window does); sid, the
    conversation's id when this module started it (another window option)."""
    listing = tmux("list-windows", "-t", SESSION, "-F",
                   "#{window_index}\t#{window_name}\t#{pane_id}\t#{pane_title}\t#{@label}\t#{@session_id}")
    found = []
    for row in listing.stdout.splitlines():
        index, window, pane, title, label, sid = (row.split("\t", 5) + ["", "", ""])[:6]
        if window == "claude" or window.startswith("c-"):
            found.append({"name": MAIN if window == "claude" else window[2:], "window": f"{SESSION}:{index}",
                          "pane": pane, "index": int(index), "title": title, "label": label, "sid": sid})
    return sorted(found, key=lambda s: (s["name"] != MAIN, s["index"]))


# A few words on what each session is about, for the top of its panel. Claude Code
# names every session itself and puts the name in the terminal's title ("✳ Demo video
# scripts and recording setup"), changing it as the work does; that is cut down to
# its first part, without the project's own name. A label given by hand wins.
PROJECT = {"berkeley", "humanoid", "lite", "bhl"}
JOINS = {"and", "with", "for", "in", "to", "on", "via", "of", "from", "using", "after", "before", "vs", "-", "–", "—"}


def topic(title: str) -> str:
    """'✳ Berkeley Humanoid Lite arm bring-up and Quest teleop' -> 'Arm bring-up'."""
    title = re.sub(r"^\W+", "", title or "").strip()
    if title.lower() in ("", "claude code", "claude"):     # a new session has no subject yet
        return ""
    words: list[str] = []
    for word in title.split():
        bare = word.lower().strip(",.:;")
        if bare in ("the", "a", "an"):
            continue
        if bare in JOINS or (bare in PROJECT and not words):
            if words and bare in JOINS:
                break
            continue
        words.append(word.strip(",.:;"))
        if len(words) == 3 or word.endswith((",", ":", ";")):
            break
    text = " ".join(words)
    return text[:1].upper() + text[1:]


def set_label(name_or_pane: str, words: str) -> str:
    """Name a session's panel by hand (a few words); no words gives it back its own title."""
    pane = name_or_pane if name_or_pane.startswith("%") else pane_of(name_or_pane)
    if not pane:
        raise RuntimeError(f"no Claude session called {name_or_pane!r}")
    words = " ".join(words.split())[:30]
    done = (tmux("set-option", "-w", "-t", pane, "@label", words) if words
            else tmux("set-option", "-w", "-u", "-t", pane, "@label"))
    if done.returncode != 0:
        raise RuntimeError(done.stderr.strip() or "tmux refused")
    return words


def clean_name(said: str | None) -> str:
    """'cam' -> 'CAM'; letters only, at most three."""
    return re.sub(r"[^A-Z]", "", (said or "").upper())[:3]


def open_session(name: str | None = None, resume: str | None = None) -> dict:
    """A Claude session: new, or `resume` (a conversation id) where it stopped. Named as
    asked for, or the next free letter. Its conversation id is kept on the window, so
    closing it later knows exactly what to save."""
    taken = {s["name"] for s in sessions()}
    name = clean_name(name)
    if not name or name in taken:
        name = next((c for c in string.ascii_uppercase if c not in taken and c != MAIN), "")
    if not name:
        raise RuntimeError("no free session name left")
    sid = resume or str(uuid.uuid4())
    program = os.environ.get("BHL_CLAUDE_CMD") or "claude"                 # tests swap the program
    start = f"--resume {sid}" if resume else f"--session-id {sid}"
    command = f"cd {REPO} && {program} {start} --model {MODEL} --effort {EFFORT}; exec bash"
    made = tmux("new-window", "-d", "-t", SESSION, "-n", f"c-{name}", "-P", "-F", "#{pane_id}", command)
    if made.returncode != 0:
        raise RuntimeError(made.stderr.strip() or "tmux refused")
    pane = made.stdout.strip()
    tmux("set-option", "-w", "-t", pane, "@session_id", sid)
    return {"name": name, "pane": pane, "sid": sid}


# ---- closing a session, keeping it, and bringing it back
def title_of(s: dict) -> str:
    """A session's whole title: a label given by hand, else the one Claude Code keeps."""
    return s.get("label") or re.sub(r"^\W+", "", s.get("title") or "").strip()


def last_title(path: Path) -> str:
    """The newest title a transcript records ("ai-title" lines, written as the work goes)."""
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - 262144))
            tail = handle.read().decode("utf-8", "replace")
    except OSError:
        return ""
    for line in reversed(tail.splitlines()):
        if "-title" in line:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            return entry.get("customTitle") or entry.get("aiTitle") or entry.get("title") or ""
    return ""


def conversation_of(s: dict) -> str | None:
    """The id of the conversation running in a session's window: the one it was started
    with, else - for one started by hand - the transcript whose newest title is the
    title Claude Code shows on that window."""
    if s.get("sid"):
        return s["sid"]
    title = re.sub(r"^\W+", "", s.get("title") or "").strip()
    if not title or title.lower() in ("claude code", "claude"):
        return None
    for path in sorted(TRANSCRIPTS.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)[:40]:
        if last_title(path) == title:
            return path.stem
    return None


def saved() -> list[dict]:
    """Sessions closed by voice, newest first: [{"name", "title", "id", "saved"}]."""
    try:
        return json.loads(SAVED.read_text())
    except (OSError, ValueError):
        return []


def _keep(entries: list[dict]) -> None:
    SAVED.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(SAVED.parent), prefix=".saved")
    with os.fdopen(fd, "w") as handle:
        json.dump(entries, handle, indent=1)
    os.replace(tmp, SAVED)


def close(name: str) -> dict:
    """Save a session (its title and conversation) and close its window. Refused for the
    main one, and for one whose conversation cannot be found - it would be lost."""
    s = next((s for s in sessions() if s["name"] == name.upper()), None)
    if s is None:
        raise RuntimeError(f"there is no session {name.upper()}")
    if s["name"] == MAIN:
        raise RuntimeError("M is the main session: it stays open")
    sid = conversation_of(s)
    if not (sid and (TRANSCRIPTS / f"{sid}.jsonl").exists()):
        # No transcript: nothing has been said in it - started here and not used yet, or
        # a new one opened by hand (Claude Code titles those "Claude Code").
        fresh = re.sub(r"^\W+", "", s.get("title") or "").strip().lower() in ("", "claude code", "claude")
        if s.get("sid") or fresh:
            tmux("kill-window", "-t", s["window"])
            return {"name": s["name"], "title": "", "id": None, "empty": True}
        raise RuntimeError(f"cannot find {s['name']}'s conversation to save, so it stays open")
    entry = {"name": s["name"], "title": title_of(s) or f"session {s['name']}", "id": sid, "saved": time.time()}
    _keep([entry] + [e for e in saved() if e.get("id") != sid])
    tmux("kill-window", "-t", s["window"])          # its transcript is already on disk
    return entry


def reopen(entry: dict) -> dict:
    """A saved session back in its own window, under its old name if that is free."""
    made = open_session(entry.get("name"), resume=entry["id"])
    _keep([e for e in saved() if e.get("id") != entry["id"]])
    return dict(made, title=entry.get("title", ""))


FILLER = {"agent", "open", "up", "reopen", "resume", "bring", "back", "restore", "continue", "start",
          "claude", "cloud", "clod", "claud", "code", "session", "sessions", "conversation", "the", "a",
          "an", "with", "this", "that", "called", "named", "name", "for", "please", "my", "me", "on",
          "of", "to", "and", "in", "again", "one", "close", "remove", "save", "kill", "delete", "shut",
          "away", "exit", "quit", "end", "dismiss", "hide", "put", "new", "can", "you", "i", "want"}


def match(words: list[str], candidates: list[dict], text_of) -> dict | None:
    """The candidate whose text shares the most words with what was said (first wins a tie)."""
    said = {w for w in words if len(w) >= 3 and w not in FILLER}
    best, best_score = None, 0
    for candidate in candidates:
        have = set(re.findall(r"[a-z0-9]+", text_of(candidate).lower()))
        score = len(said & have)
        if score > best_score:
            best, best_score = candidate, score
    return best


def by_voice(command: dict, current: str = "") -> dict:
    """An "agent, ... session" command, carried out; what the headset is told back.

    open   a saved session whose title matches the words said (or its old name), else
           a new one ("new" always makes a new one)
    close  the session named ("session A"), or whose title matches, or the current one
    list   the saved ones
    """
    words = [w.lower() for w in command.get("words") or []]
    named = clean_name(command.get("name"))
    if command.get("do") == "list_sessions":
        return {"do": "sessions_saved", "saved": [{"name": e["name"], "title": e["title"]} for e in saved()[:8]]}
    if command.get("do") == "close_session":
        running = sessions()
        s = (next((s for s in running if s["name"] == named), None)
             or match(words, running, title_of)
             or next((s for s in running if s["name"] == (current or MAIN)), None))
        entry = close(s["name"] if s else (named or current or MAIN))
        return {"do": "session_closed", "name": entry["name"], "title": entry["title"],
                "empty": bool(entry.get("empty"))}
    if not command.get("new"):
        keep = saved()
        entry = next((e for e in keep if e.get("name") == named), None) if named else None
        entry = entry or match(words, keep, lambda e: e.get("title", ""))
        if entry:
            made = reopen(entry)
            return {"do": "session_reopened", "name": made["name"], "title": made["title"]}
    made = open_session(command.get("name"))
    return {"do": "session_opened", "name": made["name"]}


def pane_of(name: str) -> str | None:
    return next((s["pane"] for s in sessions() if s["name"] == name.upper()), None)


def shown() -> str | None:
    """The session on the PC's own screen: the Claude window this tmux session shows now
    (every terminal attached to it shows the same one), else the one it showed just
    before - a quick look at the teleop window keeps it. None if neither is Claude."""
    listing = tmux("list-windows", "-t", SESSION, "-F", "#{window_active}#{window_last_flag}\t#{window_name}")
    before = None
    for row in listing.stdout.splitlines():
        flags, window = row.split("\t", 1)
        name = MAIN if window == "claude" else window[2:] if window.startswith("c-") else None
        if name and flags[:1] == "1":
            return name
        if name and flags[1:2] == "1":
            before = name
    return before


def show(name: str) -> bool:
    """Put a session on the PC's screen (the window every attached terminal shows)."""
    target = next((s["window"] for s in sessions() if s["name"] == name.upper()), None)
    return target is not None and tmux("select-window", "-t", target).returncode == 0


ESCAPE = re.compile(r"\x1b\[[0-9;:?]*[A-Za-z]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")   # colours and the like
DIM = re.compile(r"\x1b\[2m.*?(?:\x1b\[(?:0|22)?m|$)")


def plain(line: str) -> str:
    return ESCAPE.sub("", line).rstrip()


def screen(name_or_pane: str) -> list[str]:
    """What the session's terminal shows now, top to bottom, with its colour codes
    (tidy() needs them to tell a greyed-out hint in the prompt from typed words)."""
    pane = name_or_pane if name_or_pane.startswith("%") else pane_of(name_or_pane)
    if not pane:
        return []
    shot = tmux("capture-pane", "-p", "-e", "-J", "-t", pane)
    text = [line.rstrip() for line in shot.stdout.splitlines()]
    while text and not plain(text[-1]):
        text.pop()
    return text


# Claude Code keeps its prompt at the bottom of the terminal: a rule, the prompt, a
# rule, the mode line. While it works a spinner sits above the prompt ("✢ Gallivanting…
# (6m 54s · ↓ 30.2k tokens)"), with a tip under it, and between the conversation and
# the spinner is empty screen. The headset used to show the last 24 lines as they
# were: while Claude worked that was mostly the empty screen, and the conversation
# above it - the messages, the files it was reading and writing - never showed.
RULE = re.compile(r"^─{20,}$")
SPINNER = re.compile(r"^[^\w\s●❯⎿]\s\w[\w'’-]*…(?:\s.*)?$")
DONE = re.compile(r"^[^\w\s●❯⎿]\s\w+ for \d+[hms]")           # "✻ Worked for 8m 34s · done 7:01 PM"
TIP = re.compile(r"^\s*⎿\s+Tip:")
CHOICE = re.compile(r"^[\s│]*❯\s*\d+\.\s")                     # a question with numbered answers


def tidy(text: list[str]) -> dict:
    """A Claude Code screen as the headset shows it:

    lines   the conversation, without the empty screen, the prompt box or the spinner
    state   "working", "asking" (a permission or a question is up) or "idle"
    status  the spinner while working ("Gallivanting… (6m 54s · ↓ 30.2k tokens)"),
            the question when asking, the last "done" line when idle
    typed   whatever is in its prompt, not sent yet (not the greyed-out 'Try "..."' hint)
    """
    lines = [plain(line) for line in text]
    rules = [i for i, line in enumerate(lines) if RULE.match(line)]
    typed, mode = "", ""
    # The prompt box: two rules close together, with only the mode line under them.
    # A permission dialog replaces it, and then the whole screen is conversation.
    if len(rules) >= 2 and rules[-1] - rules[-2] <= 8 and len(lines) - rules[-1] <= 4:
        top, bottom = rules[-2], rules[-1]
        prompt = [plain(DIM.sub("", line)).strip() for line in text[top + 1:bottom]]
        typed = re.sub(r"^❯\s*", "", " ".join(line for line in prompt if line))
        mode = " ".join(line.strip() for line in lines[bottom + 1:] if line.strip())
        lines = lines[:top]
    status = ""
    for i in range(len(lines) - 1, max(-1, len(lines) - 30), -1):
        if SPINNER.match(lines[i]):
            status = lines[i][2:].strip()
            del lines[i]
            break
    lines = [line for line in lines if not TIP.match(line)]
    body: list[str] = []
    for line in lines:                                  # empty screen: one blank line at most
        if len(line) - len(line.lstrip()) > 40:         # a notice pushed to the right edge
            line = line.strip()
        if line or (body and body[-1]):
            body.append(line)
    while body and not body[-1]:
        body.pop()
    tail = body[-25:]
    choice = next((i for i, line in enumerate(tail) if CHOICE.match(line)), None)
    if choice is not None:
        state = "asking"
        status = next((line.strip(" │") for line in reversed(tail[:choice]) if line.strip(" │")), "")
    elif status or "esc to interrupt" in mode:
        state = "working"
    else:
        state = "idle"                                  # (a "new task? /clear" hint may follow it)
        done = next((line for line in reversed(tail[-4:]) if DONE.match(line)), "")
        status = done[2:].strip()
    return {"lines": body, "state": state, "status": status, "typed": typed}


# Claude Code's dialogs - a permission question, a question with answers, a settings
# prompt like "Teach auto mode about your environment?" - take the prompt box's place,
# and say which keys they want. Dictation typed into one is lost, and the Enter after it
# picks whatever is highlighted: on a permission question, usually "Yes".
DIALOG = re.compile(r"Esc to (?:cancel|exit|go back|close)|Enter to (?:continue|select|confirm|submit)"
                    r"|Do you want to (?:proceed|make|create|allow)")


def blocked_by(text: list[str]) -> str:
    """What stands in the prompt box's place on a Claude Code screen, in a few words (the
    dialog's question, if it asks one); "" when typing would land in the prompt."""
    lines = [plain(line) for line in text]
    rules = [i for i, line in enumerate(lines) if RULE.match(line)]
    if len(rules) >= 2 and rules[-1] - rules[-2] <= 8 and len(lines) - rules[-1] <= 4:
        return ""                                        # the prompt box, at the bottom
    tail = [line.strip(" │╭╮╰╯▔▁") for line in lines[-20:]]
    if not any(CHOICE.match(line) or DIALOG.search(line) for line in lines[-20:]):
        return ""                                        # not Claude Code's dialog: type as ever
    return next((line for line in reversed(tail) if line.endswith("?")), "") or "a dialog"


def in_the_way(name_or_pane: str) -> str:
    return blocked_by(screen(name_or_pane))


def escape(name_or_pane: str) -> str:
    """Cancel the dialog in a session's way (Esc), and only a dialog: Esc also stops
    Claude's work, so with no dialog up nothing is sent. What was cancelled, or ""."""
    pane = name_or_pane if name_or_pane.startswith("%") else pane_of(name_or_pane)
    what = in_the_way(pane) if pane else ""
    if what:
        tmux("send-keys", "-t", pane, "Escape")
    return what


def panels() -> list[dict]:
    """What the headset shows of each session (the bridge's /sessions.json)."""
    return [{"name": s["name"], "topic": s["label"] or topic(s["title"]), **tidy(screen(s["pane"]))}
            for s in sessions()]


def main() -> int:
    what = sys.argv[1] if len(sys.argv) > 1 else "list"
    if what == "open":
        print(json.dumps(open_session(sys.argv[2] if len(sys.argv) > 2 else None)))
    elif what == "show" and len(sys.argv) > 2:
        seen = tidy(screen(sys.argv[2]))
        print("\n".join(seen["lines"]))
        print(f"\n[{seen['state']}] {seen['status']}" + (f"\n[typed] {seen['typed']}" if seen["typed"] else ""))
    elif what == "label" and len(sys.argv) > 2:
        # "here" is the session this runs in: a Claude session can name its own panel
        who = os.environ.get("TMUX_PANE", "") if sys.argv[2].lower() == "here" else sys.argv[2]
        try:
            words = set_label(who, " ".join(sys.argv[3:]))
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(f"labelled {sys.argv[2]}: {words!r}" if words else f"{sys.argv[2]} is back to its own title")
    elif what == "shown":
        print(shown() or "(no Claude session on the PC's screen)")
    elif what in ("close", "reopen") and len(sys.argv) > 2:
        try:
            done = (close(sys.argv[2]) if what == "close"
                    else by_voice({"do": "open_session", "words": sys.argv[2:], "name": sys.argv[2]}))
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(json.dumps(done))
    elif what == "saved":
        for e in saved():
            print(f"{e['name']:4s} {time.strftime('%b %d %H:%M', time.localtime(e['saved']))}  {e['title']}  ({e['id']})")
    else:
        for s in sessions():
            print(f"{s['name']:4s} {s['window']:8s} {s['pane']:5s} {s['label'] or topic(s['title']) or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
