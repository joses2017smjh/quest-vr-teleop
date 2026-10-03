"""Whole-episode train/validation splits for a converted LfD dataset (docs/LFD_RECORDING_FORMAT.md, section 4).

Validation is done on whole episodes, so neighbouring frames of one demonstration never land on both sides. The
rule, applied by make_splits:
  1. Each mode (teleop, guided, policy) is split on its own. Of its n episodes, round(val_fraction * n) go to
     validation. round() is Python's: to the nearest integer, and a tie goes to the even one (0.5 -> 0, 1.5 -> 2,
     2.5 -> 2).
  2. A mode with 5 or more episodes gives at least one to validation, and every mode keeps at least one for
     training: at most n - 1 go to validation.
  3. Once the dataset has 2 or more episodes there is always at least one validation episode overall. If steps 1-2
     chose none, the largest mode that can still keep a training episode gives one; if every mode has a single
     episode, the largest mode gives its only one (ties: the first mode in alphabetical order).
  4. With val_fraction 0 nothing goes to validation (steps 2 and 3 do not apply).
Which episodes go is decided by a seeded hash, sha256(f"{seed}:{source}") with source = session_id/ep_NNNN, so the
split is reproducible and does not depend on the order the episodes were converted in.

convert_to_lerobot.py writes splits.json itself. Run this to re-split an existing dataset, for example with another
seed; it reads the dataset's conversion.json and rewrites its splits.json:

  python tools/lfd/splits.py /nfs/hpc/share/$USER/bhl-data/lerobot/local/fold_v1
  python tools/lfd/splits.py DATASET_ROOT --seed 1 --val-fraction 0.25
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SPLITS_FORMAT = "bhl-lfd-splits"
MIN_MODE_SIZE = 5
RULE = ("per mode: round(val_fraction * n) episodes to val (Python round, ties to even), at least 1 once n >= 5, "
        "at most n - 1; at least 1 val episode overall once there are >= 2 episodes (taken from the largest mode "
        "that keeps a train episode, else the largest mode); none when val_fraction is 0; chosen in "
        "sha256(f'{seed}:{source}') order")


def _order(group: list[dict], seed: int) -> list[dict]:
    return sorted(group, key=lambda e: hashlib.sha256(f"{seed}:{e['source']}".encode()).hexdigest())


def make_splits(episodes: list[dict], val_fraction: float = 0.2, seed: int = 0,
                min_mode_size: int = MIN_MODE_SIZE) -> dict:
    """episodes: [{"index": dataset episode index, "mode": ..., "source": "session_id/ep_NNNN"}, ...]."""
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    indices = [int(e["index"]) for e in episodes]
    if len(set(indices)) != len(indices):
        raise ValueError("episode indices repeat")
    by_mode: dict[str, list[dict]] = {}
    for episode in episodes:
        by_mode.setdefault(episode["mode"], []).append(episode)
    groups = {mode: _order(by_mode[mode], seed) for mode in sorted(by_mode)}
    n_val: dict[str, int] = {}
    for mode, group in groups.items():
        n = len(group)
        k = round(val_fraction * n)
        if n >= min_mode_size and val_fraction > 0:
            k = max(k, 1)
        n_val[mode] = min(k, max(n - 1, 0))
    if val_fraction > 0 and len(episodes) >= 2 and not any(n_val.values()):
        can_keep_train = [m for m in groups if len(groups[m]) >= 2]
        pool = can_keep_train or list(groups)
        mode = max(pool, key=lambda m: len(groups[m]))    # the first of equal ones: groups is sorted by name
        n_val[mode] = 1
    result_modes: dict[str, dict] = {}
    assigned: dict[int, str] = {}
    for mode, group in groups.items():
        val = sorted(int(e["index"]) for e in group[:n_val[mode]])
        train = sorted(int(e["index"]) for e in group[n_val[mode]:])
        result_modes[mode] = {"train": train, "val": val}
        assigned.update({i: "val" for i in val})
        assigned.update({i: "train" for i in train})
    return {
        "format": SPLITS_FORMAT, "version": 1, "unit": "episode", "seed": seed, "val_fraction": val_fraction,
        "rule": RULE,
        "train": sorted(i for i, s in assigned.items() if s == "train"),
        "val": sorted(i for i, s in assigned.items() if s == "val"),
        "by_mode": result_modes,
        "episodes": {str(int(e["index"])): {"source": e["source"], "mode": e["mode"],
                                            "split": assigned[int(e["index"])]}
                     for e in sorted(episodes, key=lambda e: int(e["index"]))},
    }


def write_splits(path: Path, splits: dict) -> None:
    Path(path).write_text(json.dumps(splits, indent=1) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", type=Path, help="a dataset root that convert_to_lerobot.py wrote")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    conversion = json.loads((args.dataset / "conversion.json").read_text())
    episodes = [{"index": e["index"], "mode": e["mode"], "source": e["source"]} for e in conversion["episodes"]]
    result = make_splits(episodes, args.val_fraction, args.seed)
    write_splits(args.dataset / "splits.json", result)
    for mode, part in result["by_mode"].items():
        print(f"{mode}: {len(part['train'])} train, {len(part['val'])} val {part['val']}")
    print(f"wrote {args.dataset / 'splits.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
