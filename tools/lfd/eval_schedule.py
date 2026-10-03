"""The trial schedule for a live policy evaluation (docs/LFD_EVAL_PROTOCOL.md).

A protocol JSON names the policies, the numbered start arrangements, how many times
each policy runs on each arrangement, and a seed. This tool turns it into schedule.json:
one fixed, counterbalanced order of trials, made before the first trial and committed.

How the order is built:
1. A block is one arrangement, set up once. Every policy runs on it once.
2. The policy order inside a block is one row of a Latin square. Successive blocks take
   successive rows, so every P blocks (P = number of policies) form a whole square:
   each policy runs once in each position.
3. The squares form a Williams design. For an even P it is one square; for an odd P it
   is two squares, the second the mirror image of the first. Over a whole design, each
   policy also follows every other policy equally often.
4. Each repeat visits every arrangement once, in a fresh random order.

When the number of blocks is not a multiple of P (2P for carry-over), the counts can
only be as even as possible: they then differ by at most 1 (2 for carry-over).

Commands:
  template  write an example protocol JSON, filled with the first task's defaults
  make      write schedule.json from a protocol JSON
  next      print the next pending trial, given the trial log, and the recorder `prime` command for it
  show      print the whole schedule as a run sheet

A trial log written under another schedule is refused here, as in eval_log.py and analyze_eval.py.

  python tools/lfd/eval_schedule.py template --out protocol.json
  python tools/lfd/eval_schedule.py make --protocol protocol.json --out schedule.json
  python tools/lfd/eval_schedule.py next --schedule schedule.json --log trials.jsonl

Standard library only. The trial log itself is written by tools/lfd/eval_log.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shlex
import sys
from pathlib import Path

PROTOCOL_FORMAT = "bhl-lfd-eval-protocol"
SCHEDULE_FORMAT = "bhl-lfd-eval-schedule"
VERSION = 1
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
PREFIX_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,15}$")
ROW_TRIES = 200     # random row assignments tried per square, to avoid repeating an arrangement's order
S3_VOIDS = 3        # protocol S3: a third void of one trial, or three in one session, stops the session
PRIME_MINUTES = 10  # the recorder forgets an unused prime after this long (docs/LFD_RECORDING_FORMAT.md)


# ---------------------------------------------------------------- JSON helpers

def canonical(obj) -> bytes:
    """The bytes every hash is taken over: sorted keys, no spaces, UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_of(obj) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def load_json(path: str | Path) -> dict:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"{path}: no such file") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON ({exc})") from None
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def write_json(path: str | Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value \
        and value not in (float("inf"), float("-inf"))


# ---------------------------------------------------------------- the protocol

def default_protocol() -> dict:
    """The first task's defaults (docs/LFD_EVAL_PROTOCOL.md). Fill the nulls before `make`."""
    marks = ["B1", "B2", "B3", "B4", "B5"]
    arrangements = [{"id": f"A{i + 1:02d}", "block_mark": marks[i % 5], "bowl_mark": "W1" if i < 5 else "W2"}
                    for i in range(10)]
    return {
        "format": PROTOCOL_FORMAT,
        "version": VERSION,
        "study_id": "block_bowl_v1",
        "title": "Foam block into a bowl: ACT, pi0.5 and GR00T N1.7 trained on one dataset",
        "task": "Move the foam block into the bowl within 30 seconds without human assistance",
        "arm": "right",
        "timeout_s": 30.0,
        "trial_prefix": "T",
        "policies": [
            {"id": "act", "name": "ACT", "checkpoint": None, "checkpoint_sha256": None, "train_steps": None},
            {"id": "pi05", "name": "pi0.5", "checkpoint": None, "checkpoint_sha256": None, "train_steps": None},
            {"id": "groot_n17", "name": "GR00T N1.7", "checkpoint": None, "checkpoint_sha256": None,
             "train_steps": None},
        ],
        "arrangements": arrangements,
        "repeats": 2,
        "seed": 20261003,
        "comparisons": [["pi05", "act"], ["groot_n17", "act"], ["pi05", "groot_n17"], ["groot_n17", "pi05"]],
        "alpha": 0.05,
        "confidence": 0.95,
        "bootstrap": {"resamples": 10000, "seed": 20261003, "min_n": 5},
        "dataset": {"path": None, "label": "meas_future", "lookahead_s": 0.10, "conversion_sha256": None,
                    "splits_sha256": None, "zero_convention": None},
        "camera": {"mode": "still", "preset_deg": None, "tolerance_deg": 1.0},
        "execution": {"max_speed_rad_s": 0.5, "policy_max_lead_rad": 0.15, "policy_headroom_scale": 0.6,
                      "fps": 30},
        "preregistered": {"date": None, "commit": None, "by": None},
    }


def policy_ids(protocol: dict) -> list[str]:
    return [p.get("id") if isinstance(p, dict) else p for p in protocol.get("policies") or []]


def arrangement_ids(protocol: dict) -> list[str]:
    return [a.get("id") if isinstance(a, dict) else a for a in protocol.get("arrangements") or []]


def arrangement_info(protocol: dict, arrangement: str) -> dict:
    for item in protocol.get("arrangements") or []:
        if isinstance(item, dict) and item.get("id") == arrangement:
            return item
    return {"id": arrangement}


def check_protocol(protocol: dict) -> list[str]:
    """Every problem with a protocol, in plain words. An empty list means it is usable."""
    errors = []
    if protocol.get("format") != PROTOCOL_FORMAT:
        errors.append(f"format must be {PROTOCOL_FORMAT!r}")
    if protocol.get("version") != VERSION:
        errors.append(f"version must be {VERSION}")
    if not isinstance(protocol.get("study_id"), str) or not ID_RE.match(protocol.get("study_id", "")):
        errors.append("study_id must be a short id: letters, digits, '_', '-', '.'")
    if not isinstance(protocol.get("task"), str) or not protocol["task"].strip():
        errors.append("task must be a non-empty string")
    if not is_number(protocol.get("timeout_s")) or protocol["timeout_s"] <= 0:
        errors.append("timeout_s must be a number > 0")
    if not isinstance(protocol.get("trial_prefix", "T"), str) or not PREFIX_RE.match(protocol.get("trial_prefix", "T")):
        errors.append("trial_prefix must start with a letter (letters, digits, '_', '-'; at most 16)")
    for key, least in (("policies", 2), ("arrangements", 1)):
        items = protocol.get(key)
        if not isinstance(items, list) or len(items) < least:
            errors.append(f"{key} must be a list of at least {least}")
            continue
        ids = []
        for item in items:
            ident = item.get("id") if isinstance(item, dict) else item
            if not isinstance(ident, str) or not ID_RE.match(ident):
                errors.append(f"{key}: {ident!r} is not a valid id")
            ids.append(ident)
        if len(set(map(str, ids))) != len(ids):
            errors.append(f"{key}: ids must be unique")
    if not is_int(protocol.get("repeats")) or protocol["repeats"] < 1:
        errors.append("repeats must be an integer >= 1")
    if not is_int(protocol.get("seed")):
        errors.append("seed must be an integer")
    known = set(policy_ids(protocol)) if isinstance(protocol.get("policies"), list) else set()
    pairs = protocol.get("comparisons")
    if not isinstance(pairs, list) or not pairs:
        errors.append("comparisons must list at least one [better, worse] pair")
    else:
        seen = set()
        for pair in pairs:
            if not (isinstance(pair, list) and len(pair) == 2 and all(isinstance(x, str) for x in pair)):
                errors.append(f"comparisons: {pair!r} must be [better, worse]")
                continue
            if pair[0] == pair[1] or not set(pair) <= known:
                errors.append(f"comparisons: {pair!r} must name two different policies of this protocol")
            if tuple(pair) in seen:
                errors.append(f"comparisons: {pair!r} is listed twice")
            seen.add(tuple(pair))
    for key in ("alpha", "confidence"):
        if not is_number(protocol.get(key)) or not 0.0 < protocol[key] < 1.0:
            errors.append(f"{key} must be a number between 0 and 1")
    boot = protocol.get("bootstrap")
    if not isinstance(boot, dict) or not is_int(boot.get("resamples")) or boot["resamples"] < 100 \
            or not is_int(boot.get("seed")) or not is_int(boot.get("min_n")) or boot["min_n"] < 2:
        errors.append("bootstrap must be {resamples: int >= 100, seed: int, min_n: int >= 2}")
    camera = protocol.get("camera", {})
    preset = camera.get("preset_deg") if isinstance(camera, dict) else None
    if not isinstance(camera, dict) or (preset is not None and not (
            isinstance(preset, list) and len(preset) == 2 and all(is_number(x) for x in preset))):
        errors.append("camera.preset_deg must be null or [pan, tilt] in degrees")
    elif not is_number(camera.get("tolerance_deg", 1.0)) or camera.get("tolerance_deg", 1.0) <= 0:
        errors.append("camera.tolerance_deg must be a number > 0")
    return errors


# ---------------------------------------------------------------- the design

def williams_squares(n: int) -> list[list[list[int]]]:
    """Latin squares of order n whose rows, taken together, balance carry-over.

    For an even n: one square, from the first row 0, 1, n-1, 2, n-2, ... and its shifts.
    For an odd n: that square plus its mirror image (each row reversed).
    """
    if n < 1:
        raise ValueError("need at least one policy")
    first = [0]
    for j in range(1, n):
        first.append((j + 1) // 2 if j % 2 else n - j // 2)
    square = [[(x + i) % n for x in first] for i in range(n)]
    if n % 2 == 0 or n == 1:
        return [square]
    return [square, [row[::-1] for row in square]]


def build_schedule(protocol: dict) -> dict:
    """The schedule for a protocol. The same protocol always gives the same schedule."""
    errors = check_protocol(protocol)
    if errors:
        raise ValueError("the protocol is not usable:\n  " + "\n  ".join(errors))
    policies = policy_ids(protocol)
    arrangements = arrangement_ids(protocol)
    repeats = protocol["repeats"]
    rng = random.Random(protocol["seed"])
    symbols = policies[:]
    rng.shuffle(symbols)                                  # which policy plays row symbol 0, 1, ...
    squares = williams_squares(len(policies))
    blocks = []                                           # (repeat, arrangement), in run order
    for rep in range(1, repeats + 1):
        order = arrangements[:]
        rng.shuffle(order)
        blocks += [(rep, arrangement) for arrangement in order]

    used: dict[str, set] = {a: set() for a in arrangements}
    rows_in_order = []
    unit = 0
    for start in range(0, len(blocks), len(policies)):
        chunk = blocks[start:start + len(policies)]
        rows = [tuple(row) for row in squares[unit % len(squares)]]
        best = None
        for _ in range(ROW_TRIES):                        # best effort: no arrangement gets one order twice
            rng.shuffle(rows)
            clash = sum(row in used[arrangement] for (_, arrangement), row in zip(chunk, rows))
            if best is None or clash < best[0]:
                best = (clash, rows[:])
            if clash == 0:
                break
        for (_, arrangement), row in zip(chunk, best[1]):
            used[arrangement].add(row)
            rows_in_order.append(row)
        unit += 1

    prefix = protocol.get("trial_prefix", "T")
    width = max(3, len(str(len(blocks) * len(policies))))
    trials = []
    for number, ((rep, arrangement), row) in enumerate(zip(blocks, rows_in_order), start=1):
        for position, symbol in enumerate(row, start=1):
            index = len(trials) + 1
            trials.append({"trial": f"{prefix}{index:0{width}d}", "index": index, "block": number,
                           "repeat": rep, "arrangement": arrangement, "position": position,
                           "policy": symbols[symbol]})
    return {
        "format": SCHEDULE_FORMAT,
        "version": VERSION,
        "study_id": protocol["study_id"],
        "generated_by": "tools/lfd/eval_schedule.py",
        "design": "williams" if len(squares) == 1 else "williams (two mirrored Latin squares)",
        "protocol_sha256": sha256_of(protocol),
        "protocol": protocol,
        "n_trials": len(trials),
        "n_blocks": len(blocks),
        "n_per_policy": len(arrangements) * repeats,
        "trials": trials,
        "balance": balance(trials, policies, arrangements),
    }


def balance(trials: list[dict], policies: list[str], arrangements: list[str]) -> dict:
    """How even the schedule is: per arrangement, per position, and who follows whom."""
    per_arrangement = {a: {p: 0 for p in policies} for a in arrangements}
    positions = {p: [0] * len(policies) for p in policies}
    carry = {f"{a}>{b}": 0 for a in policies for b in policies if a != b}
    blocks: dict[int, list[dict]] = {}
    for t in trials:
        per_arrangement[t["arrangement"]][t["policy"]] += 1
        positions[t["policy"]][t["position"] - 1] += 1
        blocks.setdefault(t["block"], []).append(t)
    orders: dict[str, list[tuple]] = {a: [] for a in arrangements}
    for block in blocks.values():
        block.sort(key=lambda t: t["position"])
        for before, after in zip(block, block[1:]):
            carry[f"{before['policy']}>{after['policy']}"] += 1
        orders[block[0]["arrangement"]].append(tuple(t["policy"] for t in block))
    position_values = [c for counts in positions.values() for c in counts]
    carry_values = list(carry.values()) or [0]
    return {
        "per_arrangement": per_arrangement,
        "position_counts": positions,
        "position_spread": max(position_values) - min(position_values),
        "carryover_counts": carry,
        "carryover_spread": max(carry_values) - min(carry_values),
        "arrangements_with_a_repeated_order": sum(len(set(o)) < len(o) for o in orders.values()),
    }


def load_schedule(path: str | Path) -> tuple[dict, list[str]]:
    """Read schedule.json. Raises on anything that breaks it; returns warnings beside it."""
    schedule = load_json(path)
    if schedule.get("format") != SCHEDULE_FORMAT or schedule.get("version") != VERSION:
        raise ValueError(f"{path}: not a {SCHEDULE_FORMAT} v{VERSION} file")
    protocol = schedule.get("protocol")
    if not isinstance(protocol, dict) or sha256_of(protocol) != schedule.get("protocol_sha256"):
        raise ValueError(f"{path}: the protocol copy inside it was changed after the schedule was made")
    trials = schedule.get("trials")
    if not isinstance(trials, list) or not trials:
        raise ValueError(f"{path}: no trials")
    ids = [t.get("trial") for t in trials]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: trial ids repeat")
    warnings = []
    try:
        if build_schedule(protocol)["trials"] != trials:
            warnings.append("the trial list differs from what its protocol and seed give: edited by hand?")
    except ValueError as exc:
        warnings.append(f"its protocol no longer passes the checks: {exc}")
    return schedule, warnings


# ---------------------------------------------------------------- the trial log (read side)

def read_log(path: str | Path) -> list[dict]:
    """The trial log's records in order (JSON lines). A missing file is an empty log."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: not a JSON line ({exc})") from None
        if not isinstance(record, dict) or not isinstance(record.get("kind"), str):
            raise ValueError(f"{path}:{number}: a record must be a JSON object with a 'kind'")
        records.append(record)
    return records


def foreign_hashes(records: list[dict], schedule_hash: str) -> list[str]:
    """The schedule hashes in a log that are not this schedule's: a log made under another schedule."""
    return sorted({str(r.get("schedule_sha256")) for r in records if r.get("schedule_sha256") != schedule_hash})


def log_mismatch(records: list[dict], schedule: dict) -> str | None:
    """Why this log cannot be read against this schedule, or None. Every tool refuses such a log."""
    schedule_hash = sha256_of(schedule)
    others = foreign_hashes(records, schedule_hash)
    if not others:
        return None
    return (f"this log was written under another schedule ({', '.join(o[:12] for o in others)}); "
            f"the current one is {schedule_hash[:12]}. Use the schedule the log was made with.")


def s3_stop(schedule: dict, records: list[dict]) -> str | None:
    """Why stopping rule S3 holds the session now, or None (docs/LFD_EVAL_PROTOCOL.md, section 10).

    Voids are counted per trial and per repeat: a session runs one repeat (section 6), so the tools
    take a session to be a repeat. A `deviation` logged while the stop holds records the cause and
    its fix; it ends the stop, and the counts start again from there.
    """
    by_id = {t["trial"]: t for t in schedule["trials"]}
    per_trial: dict[str, int] = {}
    per_repeat: dict[int, int] = {}
    stop = None
    for record in records:
        if record.get("kind") == "deviation" and stop:
            stop, per_trial, per_repeat = None, {}, {}
        elif record.get("kind") == "trial" and record.get("void") and record.get("trial") in by_id:
            t = by_id[record["trial"]]
            per_trial[t["trial"]] = per_trial.get(t["trial"], 0) + 1
            per_repeat[t["repeat"]] = per_repeat.get(t["repeat"], 0) + 1
            if stop is None and per_trial[t["trial"]] >= S3_VOIDS:
                stop = f"the third void of {t['trial']}"
            elif stop is None and per_repeat[t["repeat"]] >= S3_VOIDS:
                stop = f"the third void in this session (repeat {t['repeat']}; the last was {t['trial']})"
    return stop


def log_state(schedule: dict, records: list[dict]) -> dict:
    """What the log says has happened: done, voided, withdrawn and pending trials, and an S3 stop."""
    by_id = {t["trial"]: t for t in schedule["trials"]}
    done: dict[str, dict] = {}
    voids: dict[str, list[dict]] = {}
    withdrawn: dict[str, dict] = {}
    unknown, duplicates = [], []
    for record in records:
        if record["kind"] == "trial":
            trial = record.get("trial")
            if trial not in by_id:
                unknown.append(record)
            elif record.get("void"):
                voids.setdefault(trial, []).append(record)
            elif trial in done:
                duplicates.append(record)
            else:
                done[trial] = record
        elif record["kind"] == "withdraw":
            withdrawn.setdefault(record.get("policy"), record)
    pending = [t for t in schedule["trials"] if t["trial"] not in done and t["policy"] not in withdrawn]
    withdrawn_trials = [t for t in schedule["trials"] if t["trial"] not in done and t["policy"] in withdrawn]
    return {"done": done, "voids": voids, "withdrawn": withdrawn, "withdrawn_trials": withdrawn_trials,
            "pending": pending, "unknown": unknown, "duplicates": duplicates, "s3": s3_stop(schedule, records)}


def s3_line(stop: str) -> str:
    return (f"STOP (S3): {stop}. Stop the session until the cause is fixed. Then log a `deviation` with the "
            "cause and the fix (eval_log.py deviation --note ...); until then no trial can be logged.")


def next_pending(schedule: dict, records: list[dict]) -> dict | None:
    pending = log_state(schedule, records)["pending"]
    return pending[0] if pending else None


def progress_line(schedule: dict, state: dict) -> str:
    policies = policy_ids(schedule["protocol"])
    per = {p: [0, 0] for p in policies}
    for t in schedule["trials"]:
        per[t["policy"]][1] += 1
        per[t["policy"]][0] += t["trial"] in state["done"]
    parts = ", ".join(f"{p} {done}/{total}" for p, (done, total) in per.items())
    voids = sum(len(v) for v in state["voids"].values())
    return (f"{len(state['done'])} of {len(schedule['trials'])} trials logged ({parts}); "
            f"{voids} void, {len(state['withdrawn'])} polic{'y' if len(state['withdrawn']) == 1 else 'ies'} "
            f"withdrawn, {len(state['pending'])} pending")


def start_fields(schedule: dict, trial: dict) -> dict:
    """The recorder's episode fields for this trial (docs/LFD_RECORDING_FORMAT.md, section 3).

    The same fields go into the one-shot `prime` command, which the next start uses, whether it
    comes from the terminal or from "agent, start episode". `trial` is the trial's number (T017 -> 17):
    the recorder takes it as an integer. The episode check accepts the number or the id.
    """
    return {"task": schedule["protocol"]["task"], "mode": "policy", "trial": trial["index"],
            "arrangement": trial["arrangement"], "policy": trial["policy"]}


def shell_text(text: str) -> str:
    """Text for a shell command line: in double quotes when the shell leaves it alone there."""
    if text and not any(c in text for c in '"$`\\!'):
        return f'"{text}"'
    return shlex.quote(text)


def record_ctl_line(schedule: dict, trial: dict) -> str:
    """The recorder `prime` command for this trial, from any folder ("python": the rig's .venv one).

    After it, "agent, start episode" in the headset opens the episode in policy mode with these
    fields. The recorder forgets an unused prime after PRIME_MINUTES.
    """
    fields = start_fields(schedule, trial)
    script = Path(__file__).resolve().parent / "record_ctl.py"
    return (f"python {shlex.quote(str(script))} prime --mode policy "
            f"--trial {fields['trial']} --arrangement {shlex.quote(fields['arrangement'])} "
            f"--policy {shlex.quote(fields['policy'])} --task {shell_text(fields['task'])}")


def describe_arrangement(protocol: dict, arrangement: str) -> str:
    info = arrangement_info(protocol, arrangement)
    extra = ", ".join(f"{k} {v}" for k, v in info.items() if k != "id" and v is not None)
    return f"{arrangement} ({extra})" if extra else arrangement


# ---------------------------------------------------------------- commands

def cmd_template(args) -> int:
    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"REFUSED: {out} exists (use --force to overwrite)", file=sys.stderr)
        return 2
    write_json(out, default_protocol())
    print(f"wrote {out}: fill in the nulls (checkpoints, dataset hashes, camera preset, preregistration),")
    print("check the arrangements against the printed template, then run `make`.")
    return 0


def cmd_make(args) -> int:
    try:
        schedule = build_schedule(load_json(args.protocol))
    except ValueError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    out = Path(args.out)
    if out.exists() and not args.force:
        try:
            old = load_json(out)
        except ValueError:
            old = None
        if old == schedule:
            print(f"{out} is already this schedule (sha256 {sha256_of(schedule)[:12]}); left as it is")
            return 0
        print(f"REFUSED: {out} exists and differs. A preregistered schedule is not regenerated: "
              "write a new file, or use --force and log a deviation.", file=sys.stderr)
        return 2
    write_json(out, schedule)
    b = schedule["balance"]
    print(f"wrote {out}: {schedule['n_trials']} trials in {schedule['n_blocks']} blocks, "
          f"{schedule['n_per_policy']} per policy; sha256 {sha256_of(schedule)[:12]}")
    print(f"balance: positions spread {b['position_spread']}, carry-over spread {b['carryover_spread']}, "
          f"{b['arrangements_with_a_repeated_order']} arrangements see one order twice")
    return 0


def cmd_next(args) -> int:
    try:
        schedule, warnings = load_schedule(args.schedule)
        records = read_log(args.log)
    except ValueError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    mismatch = log_mismatch(records, schedule)
    if mismatch:
        print(f"REFUSED: {mismatch}", file=sys.stderr)
        return 2
    state = log_state(schedule, records)
    trial = state["pending"][0] if state["pending"] else None
    if args.json:
        out = {"next": trial, "start_fields": start_fields(schedule, trial) if trial else None,
               "prime_command": record_ctl_line(schedule, trial) if trial else None, "s3_stop": state["s3"],
               "progress": progress_line(schedule, state), "warnings": warnings}
        print(json.dumps(out, indent=1))
        return 0
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"Study {schedule['study_id']}: {progress_line(schedule, state)}.")
    if state["s3"]:
        print(s3_line(state["s3"]))
    if trial is None:
        print("No trial is pending. Run tools/lfd/analyze_eval.py report.")
        return 0
    n_pol = len(policy_ids(schedule["protocol"]))
    retry = trial["trial"] in state["voids"]
    print(f"Next: {trial['trial']}  (block {trial['block']}, position {trial['position']} of {n_pol}, "
          f"repeat {trial['repeat']}{', a re-run after a void' if retry else ''})")
    print(f"  policy:       {trial['policy']}")
    print(f"  arrangement:  {describe_arrangement(schedule['protocol'], trial['arrangement'])}")
    if trial["position"] == 1 and not retry:
        print("  set-up:       a new block: place the block and bowl on this arrangement's marks")
    else:
        print("  set-up:       reset to the same arrangement's marks")
    print(f"  start fields: {json.dumps(start_fields(schedule, trial))}")
    print(f"  recorder:     {record_ctl_line(schedule, trial)}")
    print(f'                then say "agent, start episode" in the headset; the prime serves one start, '
          f"within {PRIME_MINUTES} minutes")
    return 0


def cmd_show(args) -> int:
    try:
        schedule, warnings = load_schedule(args.schedule)
        records = read_log(args.log) if args.log else []
    except ValueError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    mismatch = log_mismatch(records, schedule)
    if mismatch:
        print(f"REFUSED: {mismatch}", file=sys.stderr)
        return 2
    for warning in warnings:
        print(f"WARNING: {warning}")
    state = log_state(schedule, records)
    print(f"{'trial':<8} {'block':>5} {'rep':>3} {'pos':>3}  {'arrangement':<12} {'policy':<14} status")
    for t in schedule["trials"]:
        if t["trial"] in state["done"]:
            status = "logged"
        elif t["policy"] in state["withdrawn"]:
            status = "withdrawn"
        else:
            status = "pending" + (" (void before)" if t["trial"] in state["voids"] else "")
        print(f"{t['trial']:<8} {t['block']:>5} {t['repeat']:>3} {t['position']:>3}  {t['arrangement']:<12} "
              f"{t['policy']:<14} {status}")
    print(progress_line(schedule, state))
    if state["s3"]:
        print(s3_line(state["s3"]))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("template", help="write an example protocol JSON with the first task's defaults")
    p.add_argument("--out", default="protocol.json")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_template)
    p = sub.add_parser("make", help="write schedule.json from a protocol JSON")
    p.add_argument("--protocol", default="protocol.json")
    p.add_argument("--out", default="schedule.json")
    p.add_argument("--force", action="store_true", help="overwrite a different schedule (log a deviation)")
    p.set_defaults(func=cmd_make)
    p = sub.add_parser("next", help="print the next pending trial and the recorder prime command for it")
    p.add_argument("--schedule", default="schedule.json")
    p.add_argument("--log", default="trials.jsonl")
    p.add_argument("--json", action="store_true", help="print JSON, for scripts")
    p.set_defaults(func=cmd_next)
    p = sub.add_parser("show", help="print the whole schedule as a run sheet")
    p.add_argument("--schedule", default="schedule.json")
    p.add_argument("--log", default="", help="mark logged trials from this trial log")
    p.set_defaults(func=cmd_show)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
