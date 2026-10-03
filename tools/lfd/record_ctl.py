#!/usr/bin/env python3
"""Start, end and discard learning-from-demonstration episodes from a terminal.

It sends a command to scripts/teleop/recorder.py on UDP 127.0.0.1:11015 and prints the
reply (docs/LFD_RECORDING_FORMAT.md, section 3); an end is followed by a status, below.
"agent, start episode" in the headset does the same through voice_typer.py. Standard
library only.

  .venv/bin/python tools/lfd/record_ctl.py task --task "Put the foam block in the bowl"
  .venv/bin/python tools/lfd/record_ctl.py start --mode teleop --operator jose --arrangement A --trial 3
  .venv/bin/python tools/lfd/record_ctl.py start --mode guided --task "Fold the washcloth once"
  .venv/bin/python tools/lfd/record_ctl.py end --outcome success --notes "slipped once, regrasped"
  .venv/bin/python tools/lfd/record_ctl.py discard        # the open episode, else the last one
  .venv/bin/python tools/lfd/record_ctl.py status
  .venv/bin/python tools/lfd/record_ctl.py prime --mode policy --trial 17 --arrangement A07 --policy pi05
  .venv/bin/python tools/lfd/record_ctl.py prime          # no fields: clears the prime

--mode is required, because it decides which action labels are valid: a hand-guided episode
recorded as teleop would be trained on the wrong one. Exits 0 when the recorder says ok, and
1 otherwise, including when it does not answer.

prime sets fields for the next start only, whoever sends it: "agent, start episode" in the
headset then opens trial 17 in policy mode. A start's own fields win, the prime is used up
by the first start that opens an episode, and one not used within 10 minutes expires, so it
cannot label a later teleop demonstration. status shows it.

end returns once the episode is on disk: the recorder answers an end at once ("saving") and
reads no other command until the save is done, so a status sent next is answered after it.

The rows come from run_teleop.py --record-tap. Turning the tap on means restarting
run_teleop.py, and a restart takes zero again: restart it only when it is STOPPED with the
motors off and the arms hang still at their zero (CLAUDE.md: curl -s localhost:8080/state.json,
every q near 0 and unchanged over a couple of seconds), with Ctrl+C twice and then the explicit
command, never "Up Enter". Zero taken with an arm bent makes every recorded angle wrong.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys

CONTROL_PORT = 11015    # recorder.py listens here
SAVE_TIMEOUT = 120.0    # s an end may take to reach the disk: a long episode on a slow one


def send(cmd: dict, port: int = CONTROL_PORT, timeout: float = 2.0) -> dict:
    """One command to recorder.py, and its reply: {"ok": ..., "said": ..., ...}."""
    link = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    link.settimeout(timeout)
    try:
        link.connect(("127.0.0.1", port))     # connected: with nothing listening, the send is refused at once
        link.send(json.dumps(cmd).encode())
        reply = json.loads(link.recv(65536))
        if not isinstance(reply, dict):
            raise ValueError("the reply is not a JSON object")
        return reply
    except ConnectionRefusedError:
        return {"ok": False, "said": f"recorder.py is not running: nothing listens on UDP 127.0.0.1:{port}"}
    except (OSError, ValueError) as exc:
        return {"ok": False, "said": f"recorder.py did not answer on UDP 127.0.0.1:{port} within {timeout:g} s "
                                     f"({exc.__class__.__name__}): busy, or stuck?"}
    finally:
        link.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=CONTROL_PORT, help="recorder.py's control port")
    commands = parser.add_subparsers(dest="cmd", required=True)
    start = commands.add_parser("start", help="open a new episode")
    start.add_argument("--task", help="what is demonstrated; optional once a default is set (the task command)")
    start.add_argument("--mode", required=True, choices=("teleop", "guided", "policy"))
    start.add_argument("--operator")
    start.add_argument("--arrangement", help="which object layout, for the evaluation protocol")
    start.add_argument("--trial", type=int, help="trial number, for the evaluation protocol")
    start.add_argument("--policy", help="the policy that drove the arms, in policy mode")
    end = commands.add_parser("end", help="close the open episode")
    end.add_argument("--outcome", required=True, choices=("success", "failure", "aborted"))
    end.add_argument("--notes", default="")
    commands.add_parser("discard", help="move the open episode, or else the last one, to _discarded/")
    task = commands.add_parser("task", help="set the default task text that voice starts use")
    task.add_argument("--task", required=True)
    commands.add_parser("status", help="the open episode, the streams' rates and ages, warnings, free disk")
    prime = commands.add_parser("prime", help="fields for the next start only (a voice start too); expires "
                                              "after 10 min unused; with none, clears the prime")
    prime.add_argument("--mode", choices=("teleop", "guided", "policy"))
    prime.add_argument("--trial", type=int, help="trial number, for the evaluation protocol")
    prime.add_argument("--arrangement", help="which object layout, for the evaluation protocol")
    prime.add_argument("--policy", help="the policy that will drive the arms, in policy mode")
    prime.add_argument("--task", help="what is demonstrated")
    return parser.parse_args(argv)


def show(reply: dict) -> None:
    print(("ok    " if reply.get("ok") else "NO    ") + str(reply.get("said", "")))
    extra = {k: v for k, v in reply.items() if k not in ("ok", "said")}
    if extra:
        print(json.dumps(extra, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    fields = ("task", "mode", "operator", "arrangement", "trial", "policy", "outcome", "notes")
    cmd = {"cmd": args.cmd}
    cmd.update({name: getattr(args, name) for name in fields if getattr(args, name, None) is not None})
    reply = send(cmd, args.port)
    show(reply)
    if args.cmd == "end" and reply.get("ok"):
        # answered before the save: the next reply comes only once the episode is on disk
        status = send({"cmd": "status"}, args.port, timeout=SAVE_TIMEOUT)
        last = status.get("last_episode") if isinstance(status.get("last_episode"), dict) else {}
        if not status.get("ok") or last.get("name") != reply.get("episode") or not last.get("saved"):
            print(f"NO    {reply.get('episode')} was not confirmed saved: {status.get('said', '')}")
            return 1
        print(f"ok    {last['name']} saved")
    return 0 if reply.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
