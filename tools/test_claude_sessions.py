"""Close, save and reopen Claude sessions the way "agent, ..." does, without touching the rig.

Runs claude_sessions against its own tmux session, a scratch transcript folder and a
fake `claude` that writes down how it was started, so nothing real is opened, closed
or resumed.

  .venv/bin/python tools/test_claude_sessions.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/teleop"))
import claude_sessions as cs  # noqa: E402

TEST_SESSION = "bhl-sessions-test"


RULE = "─" * 80
SCREENS = {
    # session B, 30 Sep: stuck here for an hour, eating every dictated sentence
    "teach": ["✻ Sautéed for 33m 34s · done 1:49 AM", "▔" * 80,
              "   Teach auto mode about your environment?",
              "   Claude Code reads this project, your recent Claude sessions, and optionally your shell history",
              "     How you use Claude here     Mixed", "   ❯ Also scan shell history     false",
              "     Also scan your other repos  false", "     Continue",
              "   ←/→ to change usage · Enter to continue · Esc to cancel"],
    "permission": ["● Bash(rm -rf build)", " Bash command", "   rm -rf build", " Do you want to proceed?",
                   " ❯ 1. Yes", "   2. Yes, and don't ask again for rm commands", "   3. No (esc)"],
    # the prompt box at the bottom wins, even under a reply that shows a menu
    "prompt": ["● The dialog looked like this:", "   ❯ 1. Yes", "   Esc to cancel", "", RULE, "❯ ", RULE,
               "  ⏵⏵ auto mode on (shift+tab to cycle)"],
    "shell": ["joses@joses-EQ:~/Berkeley-Humanoid-Lite$ "],
}


def main() -> int:
    failures: list[str] = []
    seen = {name: cs.blocked_by(text) for name, text in SCREENS.items()}
    if seen["teach"] != "Teach auto mode about your environment?" or seen["permission"] != "Do you want to proceed?":
        failures.append(f"a dialog in the prompt's place is not recognised: {seen}")
    elif seen["prompt"] or seen["shell"]:
        failures.append(f"a prompt ready for typing is taken for a dialog: {seen}")
    else:
        print("ok    dialogs in the prompt's place are recognised (so dictation holds back); a ready prompt is not")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        started = work / "started"
        fake = work / "fake-claude"
        fake.write_text(f'#!/bin/sh\necho "$@" >> {started}\nexec sleep 600\n')
        fake.chmod(0o755)
        os.environ["BHL_CLAUDE_CMD"] = str(fake)
        cs.SESSION = TEST_SESSION
        cs.SAVED = work / "saved.json"
        cs.TRANSCRIPTS = work / "transcripts"
        cs.TRANSCRIPTS.mkdir()
        subprocess.run(["tmux", "kill-session", "-t", TEST_SESSION], capture_output=True)
        subprocess.run(["tmux", "new-session", "-d", "-s", TEST_SESSION, "-n", "claude", "sleep 600"], check=True)
        try:
            # a session that has talked: its transcript exists, and Claude Code titled it
            made = cs.open_session()
            (cs.TRANSCRIPTS / f"{made['sid']}.jsonl").write_text(
                json.dumps({"type": "ai-title", "aiTitle": "Lidar mapping tests"}) + "\n")
            cs.tmux("select-pane", "-t", made["pane"], "-T", "✳ Lidar mapping tests")
            empty = cs.open_session()                          # and one nothing was said in
            time.sleep(0.3)
            closed = cs.by_voice({"do": "close_session", "words": "close the lidar session".split()})
            names = [s["name"] for s in cs.sessions()]
            kept = cs.saved()
            listed = cs.by_voice({"do": "list_sessions"})
            try:
                cs.by_voice({"do": "close_session", "words": ["close", "session"]}, current="M")
                main_closed = True
            except RuntimeError:
                main_closed = False
            gone = cs.by_voice({"do": "close_session", "name": empty["name"], "words": []})
            after_empty = [s["name"] for s in cs.sessions()]
            back = cs.by_voice({"do": "open_session", "words": "open cloud with the lidar mapping session".split()})
            time.sleep(0.5)
            how = started.read_text().splitlines()
            fresh = cs.by_voice({"do": "open_session", "words": "open a new claude session".split(), "new": True})

            if closed != {"do": "session_closed", "name": made["name"], "title": "Lidar mapping tests", "empty": False}:
                failures.append(f"closing by title did not close and save the lidar session: {closed}")
            elif made["name"] in names or not kept or kept[0]["id"] != made["sid"]:
                failures.append(f"the closed session is still open, or was not saved: {names} {kept}")
            elif listed["saved"] != [{"name": made["name"], "title": "Lidar mapping tests"}]:
                failures.append(f"'saved sessions' does not list it: {listed}")
            elif main_closed or "M" not in [s["name"] for s in cs.sessions()]:
                failures.append("the main session M was closed")
            elif not gone.get("empty") or empty["name"] in after_empty:
                failures.append(f"an empty session was not simply closed: {gone}")
            elif back.get("do") != "session_reopened" or back.get("title") != "Lidar mapping tests":
                failures.append(f"'open the lidar mapping session' did not bring it back: {back}")
            elif not any(f"--resume {made['sid']}" in line for line in how):
                failures.append(f"it was not resumed from its own conversation: {how}")
            elif cs.saved():
                failures.append(f"a session that is open again is still listed as saved: {cs.saved()}")
            elif fresh.get("do") != "session_opened" or "--session-id" not in started.read_text().splitlines()[-1]:
                failures.append(f"'open a new claude session' did not start a new one with its own id: {fresh}")
            else:
                print("ok    'close the lidar session' saves it by its title and closes it; M never closes")
                print("ok    an empty session just closes; 'saved sessions' lists what was saved")
                print("ok    'open the lidar mapping session' resumes that conversation; 'new' starts afresh")
        finally:
            subprocess.run(["tmux", "kill-session", "-t", TEST_SESSION], capture_output=True)
    print()
    for failure in failures:
        print("FAIL  " + failure)
    print("sessions are sound" if not failures else f"{len(failures)} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
