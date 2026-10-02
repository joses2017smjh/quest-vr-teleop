"""Tune the arms while someone teleoperates: see what went wrong, change it live.

For the operator in the headset, and for a Claude session they talk to ("that last reach
was wrong - the left arm did not follow"). run_teleop.py takes each change at once and
saves it to configs/arm_power_profile.json; every change also goes into the teleop log.

  .venv/bin/python tools/teleop_tune.py audit [--seconds 30] [LOG]   what went wrong, and what to try
  .venv/bin/python tools/teleop_tune.py show                         each joint's settings, live
  .venv/bin/python tools/teleop_tune.py power "left yaw" up|down     +-0.5 N·m, never past its ceiling
  .venv/bin/python tools/teleop_tune.py gravity "left pitch" up|down +-0.05 of its gravity help (sags / floats)
  .venv/bin/python tools/teleop_tune.py sensitivity up|down|0.6      hand travel per cm of yours
  .venv/bin/python tools/teleop_tune.py flip "left yaw"              its direction; motors stopped only
  .venv/bin/python tools/teleop_tune.py undo                         the last change

Joints: left|right + pitch|roll|yaw|elbow|wrist ("L yaw" works too). What the audit's
words mean:
  followed   of the motion commanded, the share the joint made (100% = all of it)
  at limit   the share of the time it pushed with all the torque it is allowed
  wrong way  its motion ran against its command: a direction to flip (while stopped)
  out of reach  the hand was sent where the arm cannot go: lower the sensitivity, or
             let go and grab again closer in
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
LOGS = REPO / "arm_validation" / "teleop_logs"
TUNE_PORT = 11012
NAMES = ["L pitch", "L roll", "L yaw", "L elbow", "L wrist", "R pitch", "R roll", "R yaw", "R elbow", "R wrist"]
SPOKEN = ["left pitch", "left roll", "left yaw", "left elbow", "left wrist",
          "right pitch", "right roll", "right yaw", "right elbow", "right wrist"]


def ask(message: dict, port: int = TUNE_PORT, timeout: float = 1.5) -> dict:
    """One change (or "show") to the running run_teleop.py, and its answer."""
    link = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    link.settimeout(timeout)
    try:
        link.sendto(json.dumps(message).encode(), ("127.0.0.1", port))
        return json.loads(link.recvfrom(65536)[0])
    except socket.timeout:
        return {"ok": False, "said": "run_teleop.py did not answer: is it running (bringup window 2), "
                                     "and new enough to take live tuning?"}
    finally:
        link.close()


def show(reply: dict) -> None:
    print(f"{reply.get('state', '?')}   sensitivity {reply.get('scale', '?')}")
    print("  joint      power now/max (ceiling)  headroom floor  gravity  flip   torque  error  flag")
    for j in reply.get("joints") or []:
        print(f"  {j['joint']:9s}  {j['lim'] or 0:4.1f} / {j['max']:4.1f} ({j['ceiling']:4.1f})      "
              f"{j['headroom']:4.2f}  {j['floor']:4.2f}   x{j['gravity']:.2f}  {'yes' if j['flip'] else ' - '}  "
              f"{j['tau'] or 0:+5.2f}  {j['err'] or 0:+.3f}  {j['tag']}")


def audit(path: Path, seconds: float | None) -> list[str]:
    """What went wrong while driving, one line per problem, with what to try."""
    rows = [json.loads(line) for line in path.open() if line.strip()]
    rows = [r for r in rows if "q_cmd" in r]
    if seconds is not None and rows:
        rows = [r for r in rows if r["t"] >= rows[-1]["t"] - seconds]
    if len(rows) < 20:
        return [f"{path.name}: only {len(rows)} samples of driving (a grip must be held) - nothing to judge"]
    qc = np.array([r["q_cmd"] for r in rows], float)
    qm = np.array([r["q_meas"] for r in rows], float)
    held = np.array([r["held"] for r in rows], bool)
    tau = np.array([r["tau"] for r in rows], float) if all("tau" in r for r in rows) else None
    lim = np.array([r["lim"] for r in rows], float) if all("lim" in r for r in rows) else None
    span = rows[-1]["t"] - rows[0]["t"]
    found = [f"{path.name}: {span:.0f} s of driving, left arm held {held[:, 0].mean():.0%}, right {held[:, 1].mean():.0%}"]
    fine = []
    for j, name in enumerate(NAMES):
        use = held[:, 0 if j < 5 else 1]
        if use.sum() < 15:
            continue
        # over 0.4 s windows: what it was told to do against what it did
        w = 10
        starts = [i for i in range(0, len(rows) - w, 3) if use[i] and use[i + w]]
        told = np.array([qc[i + w, j] - qc[i, j] for i in starts])
        did = np.array([qm[i + w, j] - qm[i, j] for i in starts])
        moving = np.abs(told) > 0.03
        if moving.sum() < 4:
            continue
        followed = float(np.sum(told[moving] * did[moving]) / np.sum(told[moving] ** 2))
        agree = float(np.corrcoef(told[moving], did[moving])[0, 1])
        maxed = float(np.mean(np.abs(tau[use, j]) >= 0.92 * lim[use, j])) if tau is not None else None
        behind = float(np.percentile(np.abs(qc[use, j] - qm[use, j]), 90))
        facts = f"followed {followed:.0%}" + (f", at its limit {maxed:.0%} of the time" if maxed is not None else "")
        # A joint set the wrong way round moves about as far as it is told, the other way.
        # One that cannot lift sags a little while told to rise: that reads negative too,
        # and flipping it would be exactly wrong - so a flip needs real motion against.
        if agree < -0.5 and followed <= -0.4:
            found.append(f"  {name:8s} WRONG WAY ({facts}): stop (triple X), then: teleop_tune.py flip \"{SPOKEN[j]}\"")
        elif followed < 0.6 and maxed is not None and maxed > 0.2:
            found.append(f"  {name:8s} NOT ENOUGH POWER ({facts}): teleop_tune.py power \"{SPOKEN[j]}\" up")
        elif followed < 0.6 and maxed is not None:
            found.append(f"  {name:8s} FALLS SHORT ({facts}, not at its limit): friction or an end stop - "
                         "watch it; the calibration (triple Y) tests it properly")
        elif followed < 0.6:
            found.append(f"  {name:8s} FALLS SHORT ({facts}; this log has no torque, so the cause is not "
                         f"known - most often power): try teleop_tune.py power \"{SPOKEN[j]}\" up")
        elif behind > 0.15:
            found.append(f"  {name:8s} LAGS ({facts}, 90% of the time within {behind:.2f} rad): "
                         f"teleop_tune.py power \"{SPOKEN[j]}\" up if it bothers you")
        else:
            fine.append(f"{name} {followed:.0%}")
    for k, side in enumerate(("left", "right")):
        key = "LR"[k]
        sent = [r[key] for r in rows if key in r and "commanded" in r[key]]
        if not sent:
            sent = [r[key] for r in rows if key in r]
            gap = [np.linalg.norm(np.subtract(s["target"], s["reached"])) * 100 for s in sent]
            if gap and np.percentile(gap, 90) > 6:
                found.append(f"  {side} hand ended up {np.median(gap):.0f} cm from its target (90%: "
                             f"{np.percentile(gap, 90):.0f} cm) - out of reach, or the arm not following")
            continue
        reach = np.array([np.linalg.norm(np.subtract(s["target"], s["commanded"])) for s in sent]) * 100
        follow = np.array([np.linalg.norm(np.subtract(s["commanded"], s["reached"])) for s in sent]) * 100
        if np.mean(reach > 6) > 0.1:
            found.append(f"  {side} hand OUT OF REACH {np.mean(reach > 6):.0%} of the time (worst {reach.max():.0f} cm): "
                         "teleop_tune.py sensitivity down, or let go and grab again closer in")
        if np.percentile(follow, 90) > 5:
            found.append(f"  {side} hand trails where it was sent by {np.median(follow):.0f} cm "
                         f"(90%: {np.percentile(follow, 90):.0f} cm) - see the joints above")
    tunes = [r for r in (json.loads(line) for line in path.open() if line.strip()) if "tune" in r]
    if tunes:
        found.append("  changes made during it: " + "; ".join(t["said"] for t in tunes[-5:]))
    if fine:
        found.append("  followed well: " + ", ".join(fine))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["audit", "show", "power", "gravity", "sensitivity", "flip", "undo"])
    parser.add_argument("args", nargs="*")
    parser.add_argument("--seconds", type=float, help="audit: only the last this many seconds of driving")
    parser.add_argument("--port", type=int, default=TUNE_PORT)
    args = parser.parse_args()
    if args.what == "audit":
        logs = [Path(a) for a in args.args] or sorted(LOGS.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        driven = [p for p in reversed(logs) if sum(1 for _ in p.open()) > 20]
        if not driven:
            print("no teleop log with any driving in it yet (arm, then hold a grip)")
            return 1
        print("\n".join(audit(driven[0], args.seconds)))
        return 0
    if args.what in ("power", "gravity"):
        if len(args.args) < 2 or args.args[-1] not in ("up", "down"):
            parser.error(f'{args.what} "JOINT" up|down')
        message = {"what": args.what, "joint": " ".join(args.args[:-1]), "step": 1 if args.args[-1] == "up" else -1}
    elif args.what == "sensitivity":
        if not args.args:
            parser.error("sensitivity up|down|VALUE")
        word = args.args[0]
        message = ({"what": "scale", "step": 1 if word == "up" else -1} if word in ("up", "down")
                   else {"what": "scale", "value": float(word)})
    elif args.what == "flip":
        message = {"what": "flip", "joint": " ".join(args.args)}
    else:
        message = {"what": args.what}
    reply = ask(message, args.port)
    if args.what != "show" or not reply.get("ok"):
        print(("done: " if reply.get("ok") else "NOT DONE: ") + reply.get("said", ""))
    if "joints" in reply:
        show(reply)
    return 0 if reply.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
