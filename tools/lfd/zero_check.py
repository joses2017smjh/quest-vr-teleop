#!/usr/bin/env python3
"""Record the shoulder-roll zero test (roadmap section 3.1) in configs/zero_check.json.

The question: is today's zero about 15 deg off on shoulder roll? run_teleop.py takes zero with
the arms hanging and calls that pose URDF zero, but URDF zero is an A-pose: the upper arm 15 deg
out from vertical. If the hanging upper arm is in fact vertical, every roll angle, and every
recording made with it, is off by about 15 deg.

The test takes 5 minutes, with the motors off and no camera:
  1. Stand the robot so the torso is vertical.
  2. Hang the arms as when taking zero.
  3. Read a phone inclinometer on the torso, then on each upper arm, in the frontal plane
     (the plane you see when you face the robot).

  python3 tools/lfd/zero_check.py --torso-deg 0.5 --left-upper-arm-deg 1.0 --right-upper-arm-deg -0.5
  python3 tools/lfd/zero_check.py ... --method camera --notes "stereo snapshot after A0"
  python3 tools/lfd/zero_check.py ... --zero-fix-applied     # measured with the q_hang zero in use
  python3 tools/lfd/zero_check.py ... --out /tmp/zero_check.json

Signs, in degrees from true vertical:
  --torso-deg            + when the top of the torso leans toward the robot's own LEFT (your right
                         as you face it); 0 when it stands straight
  --left-upper-arm-deg   + when the left upper arm hangs OUTWARD: the elbow further from the body
                         than the shoulder; - when it hangs in toward the torso
  --right-upper-arm-deg  the same for the right arm: + outward, - inward

Each arm is judged on its angle from the torso's own vertical (left + torso, right - torso), so
a torso that leans a little does not decide it. The rule was fixed in the roadmap beforehand:
  within +-3 deg             -> holds: the arm hangs vertical, so the zero is about 15 deg off
  12 to 19 deg outward       -> rejected: the arm hangs where URDF zero says, so the zero is right
  anything else              -> report the readings and stop
Overall, "holds" or "rejected" only when both arms agree; otherwise "report".

recorder.py copies the result into every episode.json, and the converter on the HPC refuses
episodes without one unless told otherwise. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / "configs/zero_check.json"
PROFILE_PATH = REPO_ROOT / "configs/arm_power_profile.json"
HOLDS_DEG = 3.0                  # the hanging upper arm is vertical, within this
REJECTED_DEG = (12.0, 19.0)      # the hanging upper arm sits where URDF zero puts it
RULE = ("Per arm, on the upper arm's angle from the torso's own vertical in the frontal plane (outward +; "
        "left = left reading + torso, right = right reading - torso): within +-3 deg -> holds (the hanging "
        "arm is vertical, so zero is ~15 deg off on shoulder roll); 12 to 19 deg outward -> rejected (zero "
        "is right); anything else -> report the readings and stop. Overall holds or rejected only when both "
        "arms agree, else report. Preregistered in docs/ROADMAP_2026-10-02.md, section 3.1.")
SIGNS = ("degrees from true vertical in the frontal plane; torso + when its top leans toward the robot's own "
         "left; each upper arm + when it hangs outward (elbow further from the body than the shoulder)")


def relative_to_torso(torso: float, left: float, right: float) -> tuple[float, float]:
    """Each upper arm's angle from the torso's own vertical, outward +.

    A torso whose top leans toward the robot's left points its lower end toward the robot's
    right: outward for the right arm, inward for the left. So an arm hanging along the torso
    reads left = -torso and right = +torso, and both come out 0 here.
    """
    return left + torso, right - torso


def decide(angle: float) -> str:
    if abs(angle) <= HOLDS_DEG:
        return "holds"
    if REJECTED_DEG[0] <= angle <= REJECTED_DEG[1]:
        return "rejected"
    return "report"


def overall(left: str, right: str) -> str:
    return left if left == right and left in ("holds", "rejected") else "report"


def check(torso: float, left: float, right: float) -> dict:
    """The readings -> per-arm angles and decisions, and the overall decision.

    Decided on the angle rounded to 0.01 deg, the value written and shown: in floats -4.9 + 1.9 is
    -3.0000000000000004, which is not within 3 deg, though an inclinometer reading it says -3.0.
    """
    rel_left, rel_right = (round(angle, 2) for angle in relative_to_torso(torso, left, right))
    decisions = {"left": decide(rel_left), "right": decide(rel_right)}
    return {"relative_to_torso_deg": {"left": rel_left, "right": rel_right},
            "decisions": decisions, "decision": overall(decisions["left"], decisions["right"])}


NEXT = {
    "holds": ("HOLDS: the hanging upper arms are vertical, but URDF zero is a 15 deg A-pose, so today's zero "
              "(hanging = 0) puts shoulder roll about 15 deg off.\n"
              "Next (roadmap 3.1): keep the interim clamp (range_lower 0.0 on left_shoulder_roll, range_upper "
              "0.0 on right_shoulder_roll; edit the profile only while run_teleop.py is STOPPED). Then land the "
              "fix: q_hang = -0.2618 rad on left roll and +0.2618 on right roll, the profile's zero_convention "
              "set to \"hanging=q_hang\", and the clamp removed in the same change. Then re-run the power "
              "calibration (B4). Episodes recorded before the fix say \"hanging=0\" and are never mixed with "
              "later ones."),
    "rejected": ("REJECTED: the hanging upper arms already sit about 15 deg out, where URDF zero puts them, so "
                 "the zero is right.\n"
                 "Next (roadmap 3.1): drop the interim roll clamp (edit the profile only while run_teleop.py is "
                 "STOPPED), and look elsewhere for the roll gravity bias: cable drag, a current-sensor offset or "
                 "friction asymmetry."),
    "report": ("REPORT: the readings fit neither case, or the arms disagree.\n"
               "Next (roadmap 3.1): report these readings and stop; change neither the zero nor the limits. "
               "Check that the torso stands vertical and that the phone lies in the frontal plane, then "
               "measure again."),
}


def explain(result: dict) -> str:
    rel, decisions = result["relative_to_torso_deg"], result["decisions"]
    lines = [f"  {side} upper arm: {rel[side]:+.2f} deg from the torso's vertical (outward +) -> {decisions[side]}"
             for side in ("left", "right")]
    lines.append(NEXT[result["decision"]])
    if result["zero_fix_applied"] and result["decision"] == "holds":
        lines.append("The q_hang zero is already in use: this confirms it.")
    elif result["zero_fix_applied"] and result["decision"] == "rejected":
        lines.append("The q_hang zero is in use, and this says it is wrong: take it out (zero_convention back "
                     "to \"hanging=0\").")
    return "\n".join(lines)


def file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def degrees(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or abs(value) > 90.0:
        raise argparse.ArgumentTypeError(f"{text}: expected degrees from vertical, between -90 and 90")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--torso-deg", type=degrees, required=True,
                        help="torso from true vertical, + when its top leans toward the robot's own left")
    parser.add_argument("--left-upper-arm-deg", type=degrees, required=True,
                        help="left upper arm from true vertical, + outward (away from the torso)")
    parser.add_argument("--right-upper-arm-deg", type=degrees, required=True,
                        help="right upper arm from true vertical, + outward (away from the torso)")
    parser.add_argument("--method", choices=("inclinometer", "camera"), default="inclinometer")
    parser.add_argument("--zero-fix-applied", action="store_true",
                        help="the corrected zero (q_hang, roadmap 3.1) was in use when this was measured")
    parser.add_argument("--notes", default="")
    parser.add_argument("--profile", type=Path, default=PROFILE_PATH,
                        help="the power profile in use, hashed into the result "
                             "(default: configs/arm_power_profile.json)")
    parser.add_argument("--out", type=Path, default=OUT_PATH, help="where to write (default: configs/zero_check.json)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = {
        "format": "bhl-zero-check", "version": 1,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": args.method,
        "readings_deg": {"torso": args.torso_deg, "left_upper_arm": args.left_upper_arm_deg,
                         "right_upper_arm": args.right_upper_arm_deg},
        "sign_convention": SIGNS,
        **check(args.torso_deg, args.left_upper_arm_deg, args.right_upper_arm_deg),
        "rule": RULE,
        "profile_sha256": file_sha256(args.profile),
        "zero_fix_applied": bool(args.zero_fix_applied),
        "notes": args.notes,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(tmp, out)
    print(f"Zero check: {result['decision'].upper()} -> {out}")
    print(explain(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
