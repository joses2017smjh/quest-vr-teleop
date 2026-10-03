"""Log live-evaluation trials, one JSON line each, checked against the schedule.

Every trial of docs/LFD_EVAL_PROTOCOL.md ends with one `trial` command. It appends a line
to the trial log (default trials.jsonl) after these checks:
1. The trial id is in schedule.json, and it has no result yet.
2. It is the next pending trial, or it carries --out-of-order REASON, which the analysis
   lists as a deviation.
3. The policy and arrangement, when given, match the schedule. So does the recorded
   episode (episode.json: mode "policy", the same trial, arrangement and policy).
4. The facts come first. A success with an intervention, or one the operator logs as having
   reached the time limit (--timed-out yes), is refused: by the protocol's rules it is a
   failure. A failure without an intervention must say whether it reached the limit.
5. A success needs its completion time (--time-s): from t = 0, the hand-over (the tap's first
   row with src "policy"), to the release, by the stopwatch or the video. Only that time is
   checked against the limit, never the episode's length, which includes ending it.
6. A success on a recording that ended by itself (aborted) needs --accept-aborted REASON,
   and is then also logged as a deviation.
7. Every record carries the schedule's hash; a log made under another schedule is refused.
8. Stopping rule S3: after a third void of one trial, or a third in one session (one repeat),
   no trial is logged until a `deviation` records the cause and its fix.

The log is append-only. Fix a mistake with `amend` (with a reason), never by editing it.

Commands:
  trial      record one trial's result, or a void attempt (--void REASON)
  amend      correct a logged trial, with a reason
  withdraw   withdraw a policy for safety, with the S2 evidence: its remaining trials count as failures
  deviation  note any other departure from the protocol
  pack       copy each trial's photos and frames for a blind scorer, under random codes
  score      record a blind scorer's call in a separate scorer file, keyed by code

  python tools/lfd/eval_log.py trial --trial T001 --success yes --time-s 21.4 --episode ~/bhl_recordings/<s>/ep_0003
  python tools/lfd/eval_log.py trial --trial T002 --success no --timed-out yes
  python tools/lfd/eval_log.py trial --trial T003 --success no --interventions 1 --collision --notes "torso"
  python tools/lfd/eval_log.py pack --out scorer_pack
  python tools/lfd/eval_log.py score --sheet scorer_pack/sheet.jsonl --scorer ana --code S3F9A1C --success yes

The operator's name comes from --operator, or else from the BHL_EVAL_OPERATOR variable.
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_schedule as es  # noqa: E402

YES = {"yes", "y", "true", "1", "success", "s"}
NO = {"no", "n", "false", "0", "failure", "fail", "f"}
SCORE_KIND = "score"
PACK_KEY_FORMAT = "bhl-lfd-eval-pack-key"
ENDED_NORMALLY = ("success", "failure")      # recorder outcomes of an episode the operator ended
JPEG_DROP = {0xE1, 0xED, 0xFE}              # APP1 (Exif, XMP), APP13 (IPTC), COM: dates, devices, notes
PACK_MTIME = 946684800                      # every packed file gets this mtime (2000-01-01): no run order


class Refused(Exception):
    """A check failed: nothing was written."""


# ---------------------------------------------------------------- small helpers

def stamp() -> dict:
    now = time.time()
    return {"logged_wall": now, "logged_iso": datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds")}


def parse_bool(text: str) -> bool:
    word = str(text).strip().lower()
    if word in YES:
        return True
    if word in NO:
        return False
    raise Refused(f"{text!r} is not yes or no")


def finite_nonneg(value, what: str) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise Refused(f"{what} must be a finite number >= 0, not {value}")
    return value


def append_jsonl(path: str | Path, record: dict) -> None:
    """Append one line and flush it to disk. Refuses if the file's last line is cut off."""
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                raise Refused(f"{path} does not end with a newline: its last line may be cut off. Fix that first.")
    line = json.dumps(record, ensure_ascii=False, allow_nan=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise Refused(f"{path}:{number}: not a JSON line ({exc})") from None
            if not isinstance(item, dict):
                raise Refused(f"{path}:{number}: expected a JSON object")
            out.append(item)
    return out


def rule_problems(success: bool | None, time_s: float | None, interventions: int, timeout_s: float,
                  timed_out: bool | None = None) -> list[str]:
    """Why a claimed success is a failure under the protocol's rules (empty: no reason).

    The facts the operator records decide first: an intervention, or the trial reaching the time
    limit. Then a completion time past the limit.
    """
    problems = []
    if success and interventions:
        problems.append(f"{interventions} intervention(s): any intervention makes the trial a failure")
    if success and timed_out is True:
        problems.append("it was logged as reaching the time limit: a trial that reaches the limit is a failure")
    if success and time_s is not None and time_s > timeout_s:
        problems.append(f"{time_s:g} s is past the {timeout_s:g} s limit: a late success is a failure")
    return problems


def time_unknown(success: bool | None, time_s: float | None, timed_out: bool | None) -> bool:
    """A claimed success with no evidence that it came within the limit: no completion time, and the
    trial was not logged as ending before the limit (--timed-out no). It counts as a failure."""
    return bool(success) and time_s is None and timed_out is not False


def parse_optional_bool(text: str | None) -> bool | None:
    return None if text is None else parse_bool(text)


# ---------------------------------------------------------------- latency input

def parse_latency_list(text: str) -> list[float]:
    try:
        values = [float(x) for x in text.replace(",", " ").split()]
    except ValueError:
        raise Refused(f"--latency-ms must be numbers in ms, comma-separated: {text!r}") from None
    return [finite_nonneg(v, "a latency") for v in values]


def read_latency_file(path: str | Path) -> list[float]:
    """Per-request latencies in ms from a file:
    a JSON list; a JSON object with "latency_ms" (a list); JSON lines each with "latency_ms";
    or plain text, one number per line ('#' starts a comment)."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise Refused(f"cannot read the latency file {path}: {exc}") from None
    values: list = []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            values = data
        elif isinstance(data, dict) and isinstance(data.get("latency_ms"), list):
            values = data["latency_ms"]
        elif isinstance(data, dict) and es.is_number(data.get("latency_ms")):
            values = [data["latency_ms"]]
        elif es.is_number(data):
            values = [data]
        else:
            raise Refused(f"{path}: a JSON latency file must be a list or have a 'latency_ms' list")
    except json.JSONDecodeError:
        lines = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
        try:
            if lines and lines[0].startswith("{"):
                for line in lines:
                    item = json.loads(line).get("latency_ms")
                    values += item if isinstance(item, list) else [item] if item is not None else []
            else:
                values = [float(line.split()[0]) for line in lines]
        except (json.JSONDecodeError, ValueError, AttributeError):
            raise Refused(f"{path}: not a latency file (see `trial --help`)") from None
    for value in values:
        if not es.is_number(value):
            raise Refused(f"{path}: latencies must be finite numbers, not {value!r}")
    return [finite_nonneg(v, f"a latency in {path}") for v in values]


# ---------------------------------------------------------------- recorded episodes

def resolve_episode(path: str | None, root: str | None = None) -> Path | None:
    """Where an episode directory is here. With `root`, <root>/<session_id>/<ep_dir> (after rsync)."""
    if not path:
        return None
    path = Path(path).expanduser()
    if root:
        return Path(root).expanduser() / path.parent.name / path.name
    return path


def policy_start(episode: Path) -> float | None:
    """t of the first tap row with src "policy" in rows.jsonl: the hand-over, t = 0 of the protocol.

    None when the episode has no such row (no POLICY state yet, or no rows.jsonl). Rows without the
    word are skipped unparsed, so this reads only up to the hand-over.
    """
    try:
        with open(episode / "rows.jsonl", "rb") as f:
            for line in f:
                if b"policy" not in line:
                    continue
                try:
                    row = json.loads(line)
                except (ValueError, RecursionError):
                    continue
                if isinstance(row, dict) and row.get("src") == "policy" and es.is_number(row.get("t")):
                    return float(row["t"])
    except OSError:
        return None
    return None


def same_trial(recorded, trial: dict) -> bool:
    """The recorder's `trial` field names this trial: its id ("T017") or its number (17 or "17")."""
    if isinstance(recorded, str):
        return recorded == trial["trial"] or (recorded.isdigit() and int(recorded) == trial["index"])
    return es.is_int(recorded) and recorded == trial["index"]


def check_episode(path: Path | None, trial: dict) -> dict:
    """Compare a recorded episode (docs/LFD_RECORDING_FORMAT.md, episode.json) with a scheduled trial."""
    if path is None:
        return {"found": False, "problems": []}
    meta = path / "episode.json"
    if not meta.is_file():
        return {"found": False, "problems": [], "path": str(path)}
    try:
        episode = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"found": True, "problems": [f"episode.json unreadable: {exc}"], "path": str(path)}
    problems = []
    if episode.get("format") != "bhl-lfd-episode":
        problems.append(f"format is {episode.get('format')!r}, not 'bhl-lfd-episode'")
    if episode.get("mode") != "policy":
        problems.append(f"mode is {episode.get('mode')!r}, not 'policy'")
    if not same_trial(episode.get("trial"), trial):
        problems.append(f"trial is {episode.get('trial')!r}, the schedule says {trial['trial']} "
                        f"(number {trial['index']})")
    for key in ("arrangement", "policy"):
        if episode.get(key) is None or str(episode.get(key)) != trial[key]:
            problems.append(f"{key} is {episode.get(key)!r}, the schedule says {trial[key]!r}")
    if episode.get("discarded"):
        problems.append("the episode was discarded")
    start, end = episode.get("start") or {}, episode.get("end") or {}

    def span(key):
        a, b = start.get(key), end.get(key)
        return float(b) - float(a) if es.is_number(a) and es.is_number(b) and b >= a else None

    t0 = policy_start(path)
    start_t, end_t = start.get("t"), end.get("t")
    zero_check = episode.get("zero_check")
    return {"found": True, "problems": problems, "path": str(path),
            "session_id": episode.get("session_id"), "episode_index": episode.get("episode_index"),
            "outcome": episode.get("outcome"), "end_reason": episode.get("end_reason"),
            "duration_s": span("t"), "wall_s": span("wall"),
            # t = 0 of the protocol: the tap's first policy row, as seconds after the recording's start,
            # and how long the recording ran after it (null without a policy row)
            "policy_t0": t0,
            "handover_s": t0 - float(start_t) if t0 is not None and es.is_number(start_t) else None,
            "after_handover_s": float(end_t) - t0 if t0 is not None and es.is_number(end_t) else None,
            "task": episode.get("task"),
            "zero_check": isinstance(zero_check, dict),
            "zero_check_decision": zero_check.get("decision") if isinstance(zero_check, dict) else None,
            "zero_check_date": zero_check.get("date") if isinstance(zero_check, dict) else None,
            "zero_convention": episode.get("zero_convention"), "zero_changed": episode.get("zero_changed"),
            "cam_mode_at_start": episode.get("cam_mode_at_start"), "cam_at_start": episode.get("cam_at_start")}


def ended_normally(check: dict) -> bool:
    """The recorder's outcome says the operator ended the episode (not aborted, not left unfinished)."""
    return check.get("outcome") in ENDED_NORMALLY


# ---------------------------------------------------------------- the log, read back

def check_log_hash(records: list[dict], schedule_hash: str) -> None:
    """Refuse a log written under another schedule (every tool does: eval_schedule, eval_log, analyze_eval)."""
    others = es.foreign_hashes(records, schedule_hash)
    if others:
        raise Refused(f"this log was written under another schedule ({', '.join(o[:12] for o in others)}); "
                      f"the current one is {schedule_hash[:12]}. Use the schedule the log was made with.")


def effective_trials(state: dict, records: list[dict]) -> dict[str, dict]:
    """Each logged trial's record with its amendments applied, in log order."""
    out = {trial: dict(record, amends=[]) for trial, record in state["done"].items()}
    for record in records:
        if record["kind"] == "amend" and record.get("trial") in out:
            current = out[record["trial"]]
            current.update(record.get("changes") or {})
            current["amends"].append({"changes": record.get("changes"), "reason": record.get("reason"),
                                      "logged_iso": record.get("logged_iso")})
    return out


def load(args) -> tuple[dict, list[dict], str]:
    schedule, warnings = es.load_schedule(args.schedule)
    for warning in warnings:
        print(f"WARNING: {args.schedule}: {warning}", file=sys.stderr)
    records = es.read_log(args.log)
    schedule_hash = es.sha256_of(schedule)
    check_log_hash(records, schedule_hash)
    return schedule, records, schedule_hash


def operator_of(args) -> str:
    name = args.operator or os.environ.get("BHL_EVAL_OPERATOR")
    if not name:
        raise Refused("who ran it? Give --operator NAME, or set BHL_EVAL_OPERATOR")
    return name


def common(schedule: dict, schedule_hash: str, operator: str) -> dict:
    return {"study_id": schedule["study_id"], "schedule_sha256": schedule_hash, "operator": operator, **stamp()}


# ---------------------------------------------------------------- commands

def cmd_trial(args) -> int:
    schedule, records, schedule_hash = load(args)
    protocol = schedule["protocol"]
    by_id = {t["trial"]: t for t in schedule["trials"]}
    trial = by_id.get(args.trial)
    if trial is None:
        ids = list(by_id)
        raise Refused(f"unknown trial id {args.trial!r}: this schedule runs {ids[0]} to {ids[-1]}")
    for key in ("policy", "arrangement"):
        given = getattr(args, key)
        if given is not None and given != trial[key]:
            raise Refused(f"{trial['trial']} is {key} {trial[key]!r} in the schedule, not {given!r}. If the wrong "
                          "one ran, do not log it as this trial: log a `deviation`, then run the trial as scheduled.")
    state = es.log_state(schedule, records)
    if state["s3"]:
        raise Refused(es.s3_line(state["s3"]))
    if trial["policy"] in state["withdrawn"]:
        raise Refused(f"{trial['policy']} was withdrawn; its remaining trials already count as failures")
    if trial["trial"] in state["done"]:
        raise Refused(f"{trial['trial']} is already logged ({state['done'][trial['trial']].get('logged_iso')}). "
                      "To correct it, use `amend` with a reason.")
    expected = state["pending"][0] if state["pending"] else None
    out_of_order = (args.out_of_order or "").strip() or None
    if expected is not None and expected["trial"] == trial["trial"]:
        if out_of_order:
            raise Refused(f"{trial['trial']} is the next trial, so it is not out of order: drop --out-of-order")
    elif not out_of_order:
        raise Refused(f"the next trial is {expected['trial'] if expected else '(none)'}, not {trial['trial']}. "
                      "Run that one, or give --out-of-order REASON: it is logged as a deviation.")
    operator = operator_of(args)
    void = (args.void or "").strip() or None
    accept_aborted = (args.accept_aborted or "").strip() or None
    timeout = float(protocol["timeout_s"])
    if void and (args.success is not None or args.timed_out is not None or args.collision or accept_aborted):
        raise Refused("a void attempt has no result: drop --success, --timed-out, --collision and --accept-aborted")
    if not void and args.success is None:
        raise Refused("give --success yes|no (or --void REASON for an attempt that does not count)")
    success = None if void else parse_bool(args.success)
    timed_out = parse_optional_bool(args.timed_out)
    if args.interventions < 0:
        raise Refused("--interventions must be >= 0")
    if args.collision and not args.interventions:
        raise Refused("--collision describes an intervention: give --interventions N too")
    if success and timed_out:
        raise Refused(f"a trial that reached the {timeout:g} s limit is a failure: log it with --success no "
                      "--timed-out yes")
    if success is False and not args.interventions and timed_out is None:
        raise Refused(f"say whether this failure reached the {timeout:g} s limit: --timed-out yes, or "
                      "--timed-out no if it failed before (the block dropped outside, the bowl tipped, ...)")
    time_s = finite_nonneg(args.time_s, "--time-s")
    wall_s = finite_nonneg(args.wall_s, "--wall-s")

    latency = []
    if args.latency_ms:
        latency += parse_latency_list(args.latency_ms)
    if args.latency_file:
        latency += read_latency_file(args.latency_file)
    p50 = finite_nonneg(args.latency_p50, "--latency-p50")
    p95 = finite_nonneg(args.latency_p95, "--latency-p95")

    episode_path = resolve_episode(args.episode)
    check = check_episode(episode_path, trial)
    if check["found"] and check["problems"]:
        raise Refused(f"the episode at {episode_path} does not match {trial['trial']}: " + "; ".join(check["problems"])
                      + ". Give the right --episode, or log a `deviation` if the recording is wrong.")
    if episode_path is not None and not check["found"]:
        print(f"WARNING: no episode.json at {episode_path}: the path is kept and checked again at analysis",
              file=sys.stderr)
    # The completion time runs from t = 0, the hand-over, to the release. It is never taken from the
    # episode's length: the recording also holds the hand-over and the operator ending it.
    if success and time_s is None:
        raise Refused("a success needs its completion time: give --time-s, the seconds from the hand-over "
                      "(t = 0) to the release, by the stopwatch or the video")
    time_source = "operator" if time_s is not None else None
    wall_source = "operator" if wall_s is not None else None
    if wall_s is None and check.get("wall_s") is not None:
        wall_s, wall_source = check["wall_s"], "episode"
    problems = rule_problems(success, time_s, args.interventions, timeout, timed_out)
    if problems:
        raise Refused("; ".join(problems) + ". Log it with --success no"
                      + (" --timed-out yes." if time_s is not None and time_s > timeout else "."))
    why = None
    if check["found"] and not ended_normally(check):
        why = f"the recording {'ended by itself' if check.get('outcome') == 'aborted' else 'was never closed'} " \
              f"({check.get('outcome')}: {check.get('end_reason')})"
        if success and not accept_aborted:
            raise Refused(f"{why}, so it cannot show this success. If the success stands anyway, give "
                          "--accept-aborted REASON: it is logged as a deviation too.")
        print(f"WARNING: {why}; the analysis flags it", file=sys.stderr)
    if accept_aborted and not (success and check["found"] and not ended_normally(check)):
        raise Refused("--accept-aborted is only for a success on a recording that ended by itself: drop it")
    if check["found"] and not void:
        warn_episode(check, protocol, success, time_s)

    photos = {}
    for which in ("before", "after"):
        given = getattr(args, which)
        if given:
            photo = Path(given).expanduser().resolve()
            if not photo.is_file():
                raise Refused(f"--{which} {given}: no such file")
            photos[which] = str(photo)
    record = {
        "kind": "trial", "trial": trial["trial"], "policy": trial["policy"], "arrangement": trial["arrangement"],
        "block": trial["block"], "position": trial["position"],
        "void": void, "success": success, "time_s": time_s, "time_source": time_source,
        "timed_out": False if success else timed_out, "interventions": int(args.interventions),
        "collision": bool(args.collision), "wall_s": wall_s, "wall_source": wall_source,
        "latency_ms": latency, "latency_file": str(Path(args.latency_file).resolve()) if args.latency_file else None,
        "latency_p50_ms": p50, "latency_p95_ms": p95,
        "episode": str(episode_path) if episode_path else None, "episode_check": check if check["found"] else None,
        "accept_aborted": accept_aborted,
        "before": photos.get("before"), "after": photos.get("after"),
        "out_of_order": out_of_order, "notes": args.notes or "",
        **common(schedule, schedule_hash, operator),
    }
    append_jsonl(args.log, record)
    written = [record]
    if accept_aborted:
        note = f"{trial['trial']}: a success logged although {why}: {accept_aborted}"
        written.append({"kind": "deviation", "trial": trial["trial"], "note": note,
                        **common(schedule, schedule_hash, operator)})
        append_jsonl(args.log, written[-1])
    if void:
        said = f"VOID ({void}); it stays pending and is run again"
    else:
        said = "SUCCESS" if success else "failure"
        said += f" in {time_s:.1f} s" if success else ""
        said += " (reached the time limit)" if timed_out else ""
        said += f", {args.interventions} intervention(s)" if args.interventions else ""
        said += " (collision)" if args.collision else ""
    print(f"logged {trial['trial']}: {trial['policy']} on {trial['arrangement']}: {said}"
          + (f" [out of order: {out_of_order}]" if out_of_order else "")
          + (" [also logged as a deviation]" if accept_aborted else ""))
    state = es.log_state(schedule, records + written)
    if state["s3"]:
        print(es.s3_line(state["s3"]), file=sys.stderr)
        return 0
    following = state["pending"][0] if state["pending"] else None
    print(f"next: {following['trial']} ({following['policy']} on {following['arrangement']})" if following
          else "no trial is pending: run tools/lfd/analyze_eval.py report")
    return 0


def warn_episode(check: dict, protocol: dict, success: bool | None, time_s: float | None) -> None:
    """Warnings about a matching episode that do not stop the logging; the analysis flags them again."""
    warnings = []
    if not check.get("zero_check"):
        warnings.append("the episode carries no zero check (protocol 7.1: one per session)")
    if check.get("zero_changed"):
        warnings.append("the zero changed during the episode")
    if str(check.get("task") or "").strip() != str(protocol.get("task") or "").strip():
        warnings.append(f"the episode's task is {check.get('task')!r}, not the protocol's")
    if check.get("outcome") in ENDED_NORMALLY and (check["outcome"] == "success") != bool(success):
        warnings.append(f"the recording was ended as {check['outcome']}")
    after = check.get("after_handover_s")
    if success and time_s is not None and after is not None and time_s > after + 0.5:
        warnings.append(f"--time-s {time_s:g} is later than the recording's end, {after:.1f} s after the hand-over")
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)


def cmd_amend(args) -> int:
    schedule, records, schedule_hash = load(args)
    state = es.log_state(schedule, records)
    current = effective_trials(state, records).get(args.trial)
    if current is None:
        raise Refused(f"{args.trial} has no logged result to amend")
    changes = {}
    if args.success is not None:
        changes["success"] = parse_bool(args.success)
    if args.time_s is not None:
        changes["time_s"] = finite_nonneg(args.time_s, "--time-s")
        changes["time_source"] = "operator"
    if args.interventions is not None:
        if args.interventions < 0:
            raise Refused("--interventions must be >= 0")
        changes["interventions"] = int(args.interventions)
    if args.timed_out is not None:
        changes["timed_out"] = parse_bool(args.timed_out)
    if args.collision is not None:
        changes["collision"] = parse_bool(args.collision)
    if args.wall_s is not None:
        changes["wall_s"] = finite_nonneg(args.wall_s, "--wall-s")
        changes["wall_source"] = "operator"
    if args.notes is not None:
        changes["notes"] = args.notes
    if not changes:
        raise Refused("nothing to change: give --success, --time-s, --interventions, --timed-out, --collision, "
                      "--wall-s or --notes")
    merged = {**current, **changes}
    if merged.get("success") and merged.get("timed_out") is None:
        merged["timed_out"] = changes["timed_out"] = False       # a success came within the limit
    if merged.get("success") and merged.get("time_s") is None:
        raise Refused("a success needs its completion time: give --time-s too (from the hand-over to the release)")
    if merged.get("collision") and not merged.get("interventions"):
        raise Refused("--collision yes describes an intervention: give --interventions N too")
    problems = rule_problems(merged.get("success"), merged.get("time_s"), merged.get("interventions") or 0,
                             float(schedule["protocol"]["timeout_s"]), merged.get("timed_out"))
    if problems:
        raise Refused("; ".join(problems))
    append_jsonl(args.log, {"kind": "amend", "trial": args.trial, "changes": changes, "reason": args.reason,
                            **common(schedule, schedule_hash, operator_of(args))})
    print(f"amended {args.trial}: {json.dumps(changes)} ({args.reason})")
    return 0


S2_RUN = 3     # S2: a collision intervention in this many of a policy's own consecutive trials


def withdrawal_evidence(state: dict, records: list[dict], policy: str, s1_contact: str | None) -> dict:
    """The evidence stopping rule S2 asks for, or Refused (docs/LFD_EVAL_PROTOCOL.md, section 10):
    a collision intervention in each of the policy's last S2_RUN logged trials, or one logged trial
    of the policy in which it caused a contact listed under S1."""
    mine = [r for r in effective_trials(state, records).values() if r.get("policy") == policy]   # log order
    if s1_contact:
        record = next((r for r in mine if r.get("trial") == s1_contact), None)
        if record is None:
            raise Refused(f"--s1-contact {s1_contact}: not a logged trial of {policy}. Log that trial first "
                          "(a failure, with --interventions and a note on the contact).")
        return {"rule": "S1 contact", "trial": s1_contact, "notes": record.get("notes", "")}
    last = mine[-S2_RUN:]
    if len(last) == S2_RUN and all(r.get("collision") is True and (r.get("interventions") or 0) > 0 for r in last):
        return {"rule": f"S2: a collision intervention in {S2_RUN} consecutive trials",
                "trials": [r["trial"] for r in last]}
    shown = ", ".join(f"{r['trial']} {'collision' if r.get('collision') else 'no collision'}" for r in last)
    raise Refused(f"S2 withdraws a policy only after a collision intervention (--collision) in each of its last "
                  f"{S2_RUN} trials, or a contact under S1 (--s1-contact TRIAL). {policy}'s last logged trials: "
                  f"{shown or 'none'}.")


def cmd_withdraw(args) -> int:
    schedule, records, schedule_hash = load(args)
    if args.policy not in es.policy_ids(schedule["protocol"]):
        raise Refused(f"unknown policy {args.policy!r}")
    state = es.log_state(schedule, records)
    if args.policy in state["withdrawn"]:
        raise Refused(f"{args.policy} is already withdrawn")
    evidence = withdrawal_evidence(state, records, args.policy, (args.s1_contact or "").strip() or None)
    left = [t["trial"] for t in state["pending"] if t["policy"] == args.policy]
    append_jsonl(args.log, {"kind": "withdraw", "policy": args.policy, "reason": args.reason, "trials": left,
                            "evidence": evidence, **common(schedule, schedule_hash, operator_of(args))})
    shown = evidence.get("trial") or ", ".join(evidence.get("trials", []))
    print(f"withdrew {args.policy} ({evidence['rule']}: {shown}): its {len(left)} remaining trial(s) count as "
          f"failures ({args.reason})")
    return 0


def cmd_deviation(args) -> int:
    schedule, records, schedule_hash = load(args)
    if args.trial and args.trial not in {t["trial"] for t in schedule["trials"]}:
        raise Refused(f"unknown trial id {args.trial!r}")
    stop = es.log_state(schedule, records)["s3"]
    append_jsonl(args.log, {"kind": "deviation", "trial": args.trial, "note": args.note,
                            **common(schedule, schedule_hash, operator_of(args))})
    print(f"deviation logged: {args.note}"
          + (f". It ends the S3 stop ({stop}): the session may resume" if stop else ""))
    return 0


def strip_jpeg(data: bytes) -> bytes | None:
    """The JPEG without its metadata segments (APP1 Exif and XMP, APP13 IPTC, COM), and without anything
    after its end marker, where phones append previews and depth maps with metadata of their own.

    The image data are copied as they are: nothing is decoded. None when the bytes do not parse as
    a JPEG; the pack then leaves the file out.
    """
    n = len(data)
    if n < 4 or data[:2] != b"\xff\xd8":
        return None
    out = bytearray(b"\xff\xd8")
    i = 2
    while i < n:
        if data[i] != 0xFF:
            return None
        while i < n and data[i] == 0xFF:                     # fill bytes before a marker
            i += 1
        if i >= n or data[i] == 0x00:
            return None
        marker = data[i]
        i += 1
        if marker == 0xD9:                                   # EOI: the end of the image
            return bytes(out + b"\xff\xd9")
        if 0xD0 <= marker <= 0xD7 or marker == 0x01:         # markers without a length
            out += bytes((0xFF, marker))
            continue
        if i + 2 > n:
            return None
        length = int.from_bytes(data[i:i + 2], "big")
        if length < 2 or i + length > n:
            return None
        if marker not in JPEG_DROP:
            out += bytes((0xFF, marker)) + data[i:i + length]
        i += length
        if marker == 0xDA:                                   # SOS: entropy-coded data, up to the next real marker
            start = i
            while True:
                j = data.find(b"\xff", i, n - 1)
                if j < 0:
                    return None
                if data[j + 1] == 0x00 or 0xD0 <= data[j + 1] <= 0xD7:      # a stuffed 0xFF, or a restart
                    i = j + 2
                    continue
                break
            out += data[start:j]
            i = j
    return None


def _frame_entries(episode: Path) -> list[dict]:
    lines = (episode / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _read_frame(f, entry: dict) -> bytes | None:
    """One camera frame's JPEG bytes, by offset in frames.mjpeg; nothing is decoded."""
    f.seek(int(entry["off"]))
    data = f.read(int(entry["len"]))
    return data if len(data) == int(entry["len"]) else None


def _pack_frames(episode: Path, out: Path, code: str, timeout: float, video: bool, left_out: list[str]) -> list[str]:
    """The first frame and the end frame, metadata stripped, and with `video` every frame in a folder.

    t = 0 is the hand-over, the tap's first policy row. When the recording has one, the end frame is
    the last one at or before t = 0 + the time limit, and the video frames are named by their time
    from t = 0. Without one, the end frame is the recording's last, and the frames carry no time.
    """
    entries = _frame_entries(episode)
    if not entries:
        return []
    t0 = policy_start(episode)
    end = len(entries) - 1
    if t0 is not None:
        inside = [k for k, e in enumerate(entries) if es.is_number(e.get("t_cap")) and e["t_cap"] <= t0 + timeout]
        end = inside[-1] if inside else 0
    media = []
    with open(episode / "frames.mjpeg", "rb") as f:
        for which, k in (("first", 0), ("end", end)):
            data = _read_frame(f, entries[k])
            clean = strip_jpeg(data) if data else None
            if clean is None:
                left_out.append(f"{code} {which} frame: not a JPEG that parses")
                continue
            (out / f"{code}_{which}_frame.jpg").write_bytes(clean)
            media.append(f"{code}_{which}_frame.jpg")
        if video:
            folder = out / f"{code}_frames"
            folder.mkdir()
            written = 0
            for e in entries:
                data = _read_frame(f, e)
                clean = strip_jpeg(data) if data else None
                if clean is None:
                    continue
                timed = t0 is not None and es.is_number(e.get("t_cap"))
                name = f"{written:05d}_{e['t_cap'] - t0:+08.2f}s.jpg" if timed else f"{written:05d}.jpg"
                (folder / name).write_bytes(clean)
                written += 1
            if written < len(entries):
                left_out.append(f"{code}: {len(entries) - written} video frame(s) did not parse")
            media.append(folder.name + "/")
    return media


SCORER_README = """Blind scoring sheet for: {task}

You score each trial from its pictures only. Each trial carries a random code, and nothing
else: please do not look up which trial or policy it was (schedule.json, the trial log,
episode.json, the pack key).

For each code in sheet.jsonl, decide from CODE_end_frame.jpg and the after photo, if any:
  success = the block lies inside the bowl, released (not held by the claw).
The end frame shows the scene at the time limit, {timeout:g} s after the hand-over (or at the
recording's end, if that came first or the recording does not mark the hand-over). You judge
only the end state: the time limit and any intervention are decided from the operator's log.
Leave out a code you cannot judge, and tell the owner which one and why.
The camera frames are side-by-side stereo and upside down (the camera is mounted rotated).

Record each call (one line per call; it refuses a second call unless you give --amend):
  python {script} score --sheet {sheet} --scorer YOUR_NAME --code CODE --success yes
(If this folder or the repo moved, adjust both paths.)

Optional, for a success: --time-s, the release time. If the pack has a CODE_frames/ folder
whose file names carry a time (00412_+0012.34s.jpg: frame 412, 12.34 s after t = 0, the
hand-over), find the first frame where the claw has let go of the block in the bowl, and give
the time in its name. Without such names there is no time reference: do not give a time.
"""


def _new_codes(trials: list[str]) -> dict[str, str]:
    """A random code per trial, unrelated to the trial id or the run order."""
    codes: dict[str, str] = {}
    for trial in trials:
        code = "S" + secrets.token_hex(3).upper()
        while code in codes.values():
            code = "S" + secrets.token_hex(3).upper()
        codes[trial] = code
    return codes


def cmd_pack(args) -> int:
    schedule, records, schedule_hash = load(args)
    timeout = float(schedule["protocol"]["timeout_s"])
    state = es.log_state(schedule, records)
    trials = effective_trials(state, records)
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise Refused(f"{out} is not empty: pack into a new directory")
    key_path = (Path(args.key) if args.key else Path(args.log).resolve().parent / f"pack_key_{out.name}.json").resolve()
    if key_path == out or out in key_path.parents:
        raise Refused(f"the key {key_path} would sit inside the scorer's folder: give --key outside {out}")
    if key_path.exists():
        raise Refused(f"{key_path} exists: it is an earlier pack's key. Give --key with a new path")
    out.mkdir(parents=True, exist_ok=True)
    codes = _new_codes(list(trials))
    order = list(trials)
    secrets.SystemRandom().shuffle(order)                 # an order nobody can reproduce from a seed
    pack_id = secrets.token_hex(4)
    sheet, bare, left_out = [], 0, []
    for trial in order:
        record, code, media = trials[trial], codes[trial], []
        for which in ("before", "after"):
            if record.get(which) and Path(record[which]).is_file():
                clean = strip_jpeg(Path(record[which]).read_bytes())
                if clean is None:
                    left_out.append(f"{trial} {which} photo {Path(record[which]).name}: not a JPEG that parses "
                                    "(its metadata cannot be stripped; convert it to JPEG to include it)")
                    continue
                (out / f"{code}_{which}_photo.jpg").write_bytes(clean)
                media.append(f"{code}_{which}_photo.jpg")
        episode = resolve_episode(record.get("episode"), args.episode_root)
        if episode and (episode / "frames.jsonl").is_file() and (episode / "frames.mjpeg").is_file():
            media += _pack_frames(episode, out, code, timeout, args.video, left_out)
        bare += not media
        sheet.append({"pack_id": pack_id, "code": code, "media": media})
    sheet_path = out / "sheet.jsonl"
    sheet_path.write_text("".join(json.dumps(item) + "\n" for item in sheet), encoding="utf-8")
    (out / "README.txt").write_text(SCORER_README.format(task=schedule["protocol"]["task"], sheet=sheet_path,
                                                         script=Path(__file__).resolve(), timeout=timeout),
                                    encoding="utf-8")
    for path in [out, *out.rglob("*")]:                   # file times would show the order of writing
        os.utime(path, (PACK_MTIME, PACK_MTIME))
    key = {"format": PACK_KEY_FORMAT, "version": 1, "pack_id": pack_id, "study_id": schedule["study_id"],
           "schedule_sha256": schedule_hash, "pack": str(out), **stamp(),
           "codes": {codes[trial]: trial for trial in sorted(trials)}}
    key_path.write_text(json.dumps(key, indent=1) + "\n", encoding="utf-8")
    for line in left_out:
        print(f"WARNING: left out: {line}", file=sys.stderr)
    print(f"packed {len(sheet)} trial(s) into {out} under random codes, in a random order; {bare} without "
          f"pictures. The key is {key_path}: keep it, and never give it to the scorer.")
    return 0


def cmd_score(args) -> int:
    sheet_path = Path(args.sheet)
    by_code = {str(item.get("code")): item for item in read_jsonl(sheet_path) if item.get("code")}
    if not by_code:
        raise Refused(f"{sheet_path}: no codes on this sheet")
    code = args.code.strip()
    if code not in by_code:
        raise Refused(f"{code!r} is not on the sheet {sheet_path}")
    scorer = args.scorer.strip()
    if not scorer:
        raise Refused("give --scorer NAME")
    scores_path = Path(args.scores) if args.scores else sheet_path.parent / "scores.jsonl"
    earlier = [r for r in read_jsonl(scores_path)
               if r.get("kind") == SCORE_KIND and r.get("code") == code and r.get("scorer") == scorer]
    amend = (args.amend or "").strip() or None
    if earlier and not amend:
        raise Refused(f"{scorer} already scored {code}: give --amend REASON to replace that call")
    if amend and not earlier:
        raise Refused(f"{scorer} has not scored {code} yet: drop --amend")
    record = {"kind": SCORE_KIND, "pack_id": by_code[code].get("pack_id"), "code": code, "scorer": scorer,
              "success": parse_bool(args.success), "time_s": finite_nonneg(args.time_s, "--time-s"),
              "notes": args.notes or "", "amend": amend, **stamp()}
    append_jsonl(scores_path, record)
    scored = {r.get("code") for r in read_jsonl(scores_path)
              if r.get("kind") == SCORE_KIND and r.get("scorer") == scorer}
    print(f"scored {code}: {'success' if record['success'] else 'failure'}; "
          f"{len(set(by_code) - scored)} left on the sheet for {scorer}")
    return 0


# ---------------------------------------------------------------- command line

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, func, help_text: str, files: bool = True):
        p = sub.add_parser(name, help=help_text, description=help_text)
        if files:
            p.add_argument("--schedule", default="schedule.json")
            p.add_argument("--log", default="trials.jsonl")
            p.add_argument("--operator", help="default: $BHL_EVAL_OPERATOR")
        p.set_defaults(func=func)
        return p

    p = add("trial", cmd_trial, "record one trial's result")
    p.add_argument("--trial", required=True, help="trial id from the schedule, e.g. T017")
    p.add_argument("--success", help="yes or no: the block rests in the bowl, released, within the limit")
    p.add_argument("--void", help="REASON: this attempt does not count (the policy never got control, or a "
                                  "fault unrelated to the policy); the trial stays pending")
    p.add_argument("--time-s", type=float, help="completion time of a success, required: seconds from the hand-over "
                                                "(t = 0) to the release, by the stopwatch or the video")
    p.add_argument("--timed-out", help="yes or no: the trial reached the time limit. Required for a failure "
                                       "without an intervention")
    p.add_argument("--interventions", type=int, default=0, help="operator takeovers or grip releases (default 0)")
    p.add_argument("--collision", action="store_true", help="the intervention was for a collision (S2 counts these)")
    p.add_argument("--accept-aborted", help="REASON a success stands although its recording ended by itself; "
                                            "also logged as a deviation")
    p.add_argument("--latency-ms", help="per-request latencies, comma-separated")
    p.add_argument("--latency-file", help="a file of per-request latencies in ms (JSON list, JSON lines with "
                                          "'latency_ms', or one number per line)")
    p.add_argument("--latency-p50", type=float, help="ms, when only a summary exists")
    p.add_argument("--latency-p95", type=float, help="ms, when only a summary exists")
    p.add_argument("--wall-s", type=float, help="the trial's wall time; default: the episode's wall-clock length")
    p.add_argument("--episode", help="the recorded episode directory (<root>/<session_id>/ep_NNNN)")
    p.add_argument("--before", help="a photo of the scene before the trial")
    p.add_argument("--after", help="a photo of the scene after the trial")
    p.add_argument("--policy", help="what ran; checked against the schedule")
    p.add_argument("--arrangement", help="what was set up; checked against the schedule")
    p.add_argument("--out-of-order", help="REASON for running a trial other than the next one")
    p.add_argument("--notes", default="")

    p = add("amend", cmd_amend, "correct a logged trial (appends an amendment; the original stays)")
    p.add_argument("--trial", required=True)
    p.add_argument("--reason", required=True)
    p.add_argument("--success")
    p.add_argument("--time-s", type=float)
    p.add_argument("--interventions", type=int)
    p.add_argument("--timed-out", help="yes or no")
    p.add_argument("--collision", help="yes or no")
    p.add_argument("--wall-s", type=float)
    p.add_argument("--notes")

    p = add("withdraw", cmd_withdraw, "withdraw a policy for safety (S2): its remaining trials count as failures. "
                                      "Needs a collision intervention in each of its last 3 trials, or --s1-contact")
    p.add_argument("--policy", required=True)
    p.add_argument("--reason", required=True)
    p.add_argument("--s1-contact", help="TRIAL: the logged trial in which the policy caused a contact listed under S1")

    p = add("deviation", cmd_deviation, "note a departure from the protocol (it also ends an S3 stop)")
    p.add_argument("--note", required=True)
    p.add_argument("--trial", help="the trial it concerns, if any")

    p = add("pack", cmd_pack, "copy pictures for a blind scorer, under random codes; the key stays outside")
    p.add_argument("--out", required=True, help="a new, empty directory to hand to the scorer")
    p.add_argument("--key", help="where to write the code key (default: pack_key_<out>.json beside the log); "
                                 "never inside --out")
    p.add_argument("--episode-root", help="where the episodes are now: <root>/<session_id>/ep_NNNN")
    p.add_argument("--video", action="store_true", help="also every frame, named by its time from the hand-over")

    p = add("score", cmd_score, "record a blind scorer's call, keyed by the pack's code", files=False)
    p.add_argument("--sheet", required=True, help="sheet.jsonl from `pack`")
    p.add_argument("--scores", help="the scorer file (default: scores.jsonl beside the sheet)")
    p.add_argument("--scorer", required=True)
    p.add_argument("--code", required=True, help="the trial's code on the sheet, e.g. S3F9A1C")
    p.add_argument("--success", required=True, help="yes or no")
    p.add_argument("--time-s", type=float, help="for a success: the time in the release frame's name (README.txt)")
    p.add_argument("--notes", default="")
    p.add_argument("--amend", help="REASON for replacing your earlier call on this trial")

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (Refused, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
