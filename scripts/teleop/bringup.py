#!/usr/bin/env python3
"""Bring the whole teleop rig up in one tmux session.

Five programs have to run at once, in the right order, in separate terminals,
and dictation only works if Claude Code is itself inside tmux. That is a lot of
windows to open by hand every time the PC reboots, so this does it:

    ./scripts/teleop/bringup.py            # start everything and attach
    ./scripts/teleop/bringup.py status     # what is up, without touching it
    ./scripts/teleop/bringup.py stop       # shut the rig down
    ./scripts/teleop/bringup.py attach     # come back to a running session

Windows in the `bhl` session (Ctrl-B then the number; Ctrl-B d to leave):

    0 claude    claude --continue         - dictation types in here
    1 bridge    quest_bridge.py           - the headset page, on :8080
    2 teleop    run_teleop.py             - the arms
    3 screen    screen_share.py           - your desktop, in the headset
    4 voice     voice_typer.py            - speech -> the claude window
    5 camera    camera_share.py           - the robot's stereo camera, in the headset
      depth     depth_flow.py             - depth + optical flow per eye (A x2), idle until shown

Nothing is started twice: a program already running keeps its terminal, and the
window says so instead. CAN comes up first (sudo asks for your password) unless
you pass --sim.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VENV = REPO / ".venv/bin/python"
SESSION = "bhl"
CAN_BUSES = ("can0", "can1")
BITRATE = 1_000_000
HTTP_PORT = 8080


def tmux(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["tmux", *args], capture_output=True, text=True, check=check)


def session_exists() -> bool:
    return tmux("has-session", "-t", SESSION).returncode == 0


def running(script: str) -> int:
    """PID of a teleop helper already running, or 0."""
    found = subprocess.run(["pgrep", "-f", f"scripts/teleop/{script}"],
                           capture_output=True, text=True)
    for line in found.stdout.split():
        pid = int(line)
        if pid != os.getpid():
            return pid
    return 0


def can_up(name: str) -> bool:
    state = subprocess.run(["ip", "-br", "link", "show", name], capture_output=True, text=True)
    return state.returncode == 0 and " UP " in f" {state.stdout.split()[1] if len(state.stdout.split()) > 1 else ''} "


def bring_can_up() -> bool:
    """Both arm buses up at 1 Mbit. A reboot always leaves these down."""
    missing = [name for name in CAN_BUSES if not can_up(name)]
    if not missing:
        print("CAN: can0 and can1 are already up.")
        return True
    print(f"CAN: bringing up {', '.join(missing)} (sudo will ask for your password)")
    for name in missing:
        up = subprocess.run(["sudo", "ip", "link", "set", name, "up", "type", "can",
                             "bitrate", str(BITRATE)])
        if up.returncode != 0:
            print(f"  {name} did not come up. The arms cannot run; use --sim to rehearse without them.")
            return False
        subprocess.run(["sudo", "ip", "link", "set", name, "txqueuelen", "1000"])
    print(f"CAN: {', '.join(missing)} up at {BITRATE // 1000} kbit.")
    return True


def local_ip() -> str:
    """The address the headset should open, not 127.0.0.1."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 53))
        return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()


def claude_outside_session() -> int:
    """A Claude Code running somewhere else would fight this one for the conversation."""
    found = subprocess.run(["pgrep", "-x", "claude"], capture_output=True, text=True)
    pids = [int(p) for p in found.stdout.split()]
    if not pids:
        return 0
    panes = tmux("list-panes", "-a", "-F", "#{pane_pid}").stdout.split()
    inside = set()
    for pane_pid in panes:
        children = subprocess.run(["pgrep", "-P", pane_pid], capture_output=True, text=True)
        inside.update(int(c) for c in children.stdout.split())
        inside.add(int(pane_pid))
    outside = [pid for pid in pids if pid not in inside]
    return outside[0] if outside else 0


def claude_pane() -> str:
    """The pane running Claude Code, in any session tmux can see, or "".""" 
    listing = tmux("list-panes", "-a", "-F",
                   "#{session_name}:#{window_index}.#{pane_index} #{pane_current_command}").stdout
    for line in listing.splitlines():
        where, _, command = line.rpartition(" ")
        if command.strip() == "claude":
            return where
    return ""


def heal_claude_window(args: argparse.Namespace) -> None:
    """Window 0 can be closed, or Claude can exit inside it, long after startup.

    The window keeps the last screen Claude painted, so it still looks alive while
    dictation has nowhere to type. Attaching is the moment to notice and say so.
    """
    if args.no_claude:
        return
    where = claude_pane()
    if where:
        print(f"claude    {where}  (dictation types here)")
        return
    busy = claude_outside_session()
    if busy:
        print(f"\nClaude Code is running outside tmux (pid {busy}), where nothing can type into it.")
        print("Dictation only reaches a tmux pane. Quit that one, then bring it in here with:")
        print(f"  tmux new-window -t {SESSION}:0 -n claude 'claude --continue'\n")
        return
    print(f"No Claude window (dictation had nowhere to type); starting one in {SESSION}:0.")
    shell = (f"cd {REPO} && claude {args.claude_args}; "
             f"printf '\\n[%s] stopped. Up-arrow to run it again.\\n' claude; exec bash")
    tmux("new-window", "-t", f"{SESSION}:0", "-n", "claude", shell)


def window(name: str, command: str, first: bool = False) -> None:
    """One window per program, dropping to a shell when the program stops.

    The shell keeps the output on screen after a crash and leaves the command in
    the history, so restarting one piece is an up-arrow away.
    """
    shell = f"cd {REPO} && {command}; printf '\\n[%s] stopped. Up-arrow to run it again.\\n' {name}; exec bash"
    if first:
        tmux("new-session", "-d", "-s", SESSION, "-n", name, shell, check=True)
    else:
        tmux("new-window", "-t", SESSION, "-n", name, shell, check=True)


def note(name: str, text: str, first: bool = False) -> None:
    """A window for something that is already running elsewhere."""
    shell = f"printf '%s\\n' {shlex.quote(text)}; exec bash"
    if first:
        tmux("new-session", "-d", "-s", SESSION, "-n", name, shell, check=True)
    else:
        tmux("new-window", "-t", SESSION, "-n", name, shell, check=True)


def start(args: argparse.Namespace) -> int:
    if shutil.which("tmux") is None:
        print("tmux is missing:  sudo apt install tmux")
        return 1
    if not VENV.exists():
        print(f"no virtualenv at {VENV}")
        return 1
    if session_exists():
        print(f"The {SESSION} session is already up. Attaching; 'stop' first if you want it fresh.")
        heal_claude_window(args)
        return attach()

    if args.sim:
        print("SIM: no CAN, no motors. The headset flow is the same.")
    elif not bring_can_up():
        return 1

    busy = claude_outside_session()
    want_claude = not args.no_claude
    if want_claude and busy:
        print(f"\nClaude Code is already running outside tmux (pid {busy}).")
        print("Two of them would resume the same conversation, so this one is left alone.")
        print("Window 0 says how to move it in here; dictation picks it up by itself.\n")
        want_claude = False

    first = True
    if want_claude:
        window("claude", f"claude {args.claude_args}".strip(), first=True)
        first = False
    elif busy:
        note("claude",
             f"Claude Code is already running outside tmux (pid {busy}), where nothing can type into it.\n"
             "To dictate to it, move it in here:\n"
             "  1. in the terminal it is running in, leave it (/exit, or Ctrl-C twice)\n"
             "  2. then, in this window:  claude --continue\n"
             "Dictation notices within a few seconds; nothing else needs restarting.",
             first=True)
        first = False

    teleop = f"{VENV} scripts/teleop/run_teleop.py"
    if args.sim:
        teleop += " --sim"
    if args.no_viz:
        teleop += " --no-viz"
    pieces = [
        ("bridge", "quest_bridge.py", f"{VENV} scripts/teleop/quest_bridge.py", True),
        ("teleop", "run_teleop.py", teleop, True),
        # restarts itself when the monitor layout changes (exit 3); GNOME asks for Share again
        ("screen", "screen_share.py",
         f"while true; do {VENV} scripts/teleop/screen_share.py; [ $? -eq 3 ] || break; sleep 1; done",
         not args.no_screen),
        ("voice", "voice_typer.py", f"{VENV} scripts/teleop/voice_typer.py", not args.no_voice),
        ("camera", "camera_share.py", f"{VENV} scripts/teleop/camera_share.py", True),
        # where the operator is (the camera keeps them centred) and the robot's face
        ("track", "person_track.py", f"{VENV} scripts/teleop/person_track.py", True),
        # the lidar and the IMU, for the headset's developer view ("agent, developer")
        ("sensors", "sensor_share.py", f"{VENV} scripts/teleop/sensor_share.py", True),
        # depth + optical flow per camera eye (A x2 in the headset); idle until shown
        ("depth", "depth_flow.py", f"{VENV} scripts/teleop/depth_flow.py", True),
        ("face", "face_display.py",
         f"while true; do {VENV} scripts/teleop/face_display.py; sleep 3; done", True),
    ]
    # voice_typer needs no ordering: it watches for a Claude pane and adopts one
    # whenever it turns up, so it is started like everything else.
    for name, script, command, wanted in pieces:
        if not wanted:
            continue
        pid = running(script)
        if pid:
            note(name, f"{script} is already running (pid {pid}) in another terminal; left alone.",
                 first=first)
        else:
            window(name, command, first=first)
        first = False

    tmux("select-window", "-t", f"{SESSION}:0")
    address = f"http://{local_ip()}:{HTTP_PORT}"
    print(f"\nThe {SESSION} session is up.")
    print(f"  headset page   {address}")
    print("  switch window  Ctrl-B then 0..4        leave it running  Ctrl-B d")
    print("  come back      ./scripts/teleop/bringup.py attach")
    if args.no_screen:
        print("  (no screen share: start it in a window with screen_share.py when you want it)")
    return attach()


def attach() -> int:
    if not session_exists():
        print("Nothing is running. Start it with:  ./scripts/teleop/bringup.py")
        return 1
    if os.environ.get("TMUX"):
        print(f"You are already inside tmux. Switch with:  tmux switch-client -t {SESSION}")
        return 0
    if not sys.stdout.isatty():          # run from a script, or from Claude
        print(f"Not a terminal, so not attaching. Run:  tmux attach -t {SESSION}")
        return 0
    os.execvp("tmux", ["tmux", "attach", "-t", SESSION])


def stop(args: argparse.Namespace) -> int:
    scripts = ["camera_share.py", "voice_typer.py", "screen_share.py", "run_teleop.py", "face_display.py",
               "person_track.py", "sensor_share.py", "depth_flow.py", "quest_bridge.py"]
    for script in scripts:      # run_teleop parks the motors as it exits, so it is given time below
        pid = running(script)
        if pid:
            print(f"  stopping {script} (pid {pid})")
            subprocess.run(["kill", str(pid)])
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline and any(running(s) for s in scripts):
        time.sleep(0.3)
    if session_exists():
        tmux("kill-session", "-t", SESSION)
        print(f"  {SESSION} session closed")
    left = [s for s in scripts if running(s)]
    print("Stopped." if not left else f"Still running: {', '.join(left)}")
    return 0


def claude_under(pane_pid: str) -> bool:
    """Is Claude Code somewhere below this pane's process?

    `pane_current_command` names the foreground process group's leader, and this
    script starts Claude from `bash -c ...`, which leaves the two sharing a group:
    tmux then says `bash`, and everything that trusts that field decides Claude is
    not running - which silently leaves dictation with nowhere to type.
    """
    listing = subprocess.run(["ps", "-eo", "pid=,ppid=,comm="], capture_output=True, text=True)
    children: dict[str, list[tuple[str, str]]] = {}
    for row in listing.stdout.splitlines():
        parts = row.split(None, 2)
        if len(parts) == 3:
            children.setdefault(parts[1], []).append((parts[0], parts[2].strip()))
    stack, seen = [pane_pid], set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        for child, comm in children.get(pid, ()):
            if comm == "claude":
                return True
            stack.append(child)
    return False


def status(args: argparse.Namespace) -> int:
    print("CAN     " + "  ".join(f"{name} {'up' if can_up(name) else 'DOWN'}" for name in CAN_BUSES))
    for script in ("quest_bridge.py", "run_teleop.py", "screen_share.py", "voice_typer.py", "camera_share.py",
                   "person_track.py", "face_display.py", "sensor_share.py", "depth_flow.py"):
        pid = running(script)
        print(f"{script:18s} {'running, pid ' + str(pid) if pid else 'not running'}")
    print(f"tmux    {SESSION} session " + ("up" if session_exists() else "not running"))
    if session_exists():
        for line in tmux("list-windows", "-t", SESSION,
                         "-F", "          #{window_index} #{window_name}  (#{pane_current_command})"
                         ).stdout.splitlines():
            print(line)
    # Dictation types into whichever pane runs Claude Code, wherever that is. A
    # window still called "claude" that has fallen back to a shell is the usual
    # surprise: it keeps the last screen Claude painted, so it looks alive.
    claude_panes = [where for where, _, pane_pid in (
        line.split() for line in tmux(
            "list-panes", "-a",
            "-F", "#{session_name}:#{window_index}.#{pane_index} #{pane_current_command} #{pane_pid}"
        ).stdout.splitlines() if len(line.split()) == 3)
        if _ == "claude" or claude_under(pane_pid)]
    if not claude_panes:
        print("claude  not running in any tmux pane - dictation has nowhere to type")
    elif claude_panes == [f"{SESSION}:0.0"]:
        print(f"claude  {claude_panes[0]}  (dictation types here)")
    else:
        print(f"claude  {', '.join(claude_panes)}  (dictation types here)")
        if f"{SESSION}:0.0" not in claude_panes:
            has_zero = any(line.startswith("0 ") for line in tmux(
                "list-windows", "-t", SESSION, "-F", "#{window_index} #{window_name}").stdout.splitlines())
            print(f"        note: {SESSION} window 0 " + ("is a shell, not Claude - whatever it shows "
                  "is leftover paint" if has_zero else "does not exist, so dictation types elsewhere"))
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{HTTP_PORT}/state.json", timeout=3) as answer:
            state = json.loads(answer.read().decode())
    except Exception:
        return 0
    pages = state["pages_websocket"] + state["pages_http"]
    print(f"headset {pages} page(s) connected", end="")
    if state.get("last_pose_age") is not None:
        print(f", last hand pose {state['last_pose_age']:.1f}s ago", end="")
    print()
    if state.get("beacons"):
        print(f"        last report: {state['beacons'][-1]['text'][:100]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", nargs="?", default="start",
                        choices=("start", "stop", "status", "attach", "restart"))
    parser.add_argument("--sim", action="store_true", help="simulated arms: no CAN, no motors")
    parser.add_argument("--no-viz", action="store_true", help="skip the Meshcat view (starts faster)")
    parser.add_argument("--no-screen", action="store_true", help="no desktop sharing (it asks for a click)")
    parser.add_argument("--no-voice", action="store_true", help="no dictation")
    parser.add_argument("--no-claude", action="store_true", help="do not start Claude Code in window 0")
    parser.add_argument("--claude-args", default="--continue", help="arguments for claude")
    args = parser.parse_args()

    if args.action == "attach":
        return attach()
    if args.action == "status":
        return status(args)
    if args.action == "stop":
        return stop(args)
    if args.action == "restart":
        stop(args)
        time.sleep(0.5)
    return start(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        sys.exit(130)
