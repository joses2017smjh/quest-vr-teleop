"""Preregistered statistics for a live policy evaluation (docs/LFD_EVAL_PROTOCOL.md, section 11).

It reads schedule.json (with the protocol copy inside it), the trial log and any blind
scorer files, and writes report.json and report.md. Every setting comes from that protocol
copy: no flag here changes a statistic.

What it computes:
1. Primary endpoint per policy: successes out of the trials run, with a Wilson interval.
   Each trial is decided in a fixed order (protocol section 9): first the facts the operator
   logged (an intervention, or the time limit reached) make it a failure, whatever anyone
   called it; then the blind scorer's call on the end state, or else the operator's; then a
   success needs a completion time within the limit, or the operator's word that the trial
   ended before it (--timed-out no). A success with neither counts as a failure, flagged.
2. The preregistered one-sided comparisons: Fisher's exact test, Holm-corrected over the
   protocol's list. Other directions are shown as exploratory and unadjusted.
3. Two sensitivity analyses: missing trials counted as failures, and the operator's calls
   instead of the blind scorer's, under the same rules.
4. Completion time of successes (median, percentile bootstrap with a fixed seed),
   interventions per trial, inference latency p50/p95 (pooled over requests, and the
   per-trial median), and wall time per trial.
5. Deviations: missing, void, out-of-order and withdrawn trials, amendments, scorer
   disagreements, rule overrides, malformed records, episode mismatches, a moved camera,
   zero checks and conventions per session, and notes from the log.
Failures stay in every table. A trial log written under another schedule is refused.

  python tools/lfd/analyze_eval.py report --schedule schedule.json --log trials.jsonl \\
      --scores scorer_pack/scores.jsonl --pack-key pack_key_scorer_pack.json --out eval_report
  python tools/lfd/analyze_eval.py power --n 20 --alpha 0.05 --m 4

Pure Python and numpy: math.comb for Fisher, statistics.NormalDist for z.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_log as el  # noqa: E402
import eval_schedule as es  # noqa: E402

REPORT_FORMAT = "bhl-lfd-eval-report"
VERSION = 1


# ---------------------------------------------------------------- statistics

def z_for(confidence: float) -> float:
    """The two-sided normal quantile: 1.95996... for 95%."""
    return statistics.NormalDist().inv_cdf(0.5 + confidence / 2.0)


def wilson(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float] | None:
    """Wilson score interval for successes out of n, without continuity correction. None when n = 0."""
    successes, n = int(successes), int(n)
    if n == 0:
        return None
    if not 0 <= successes <= n:
        raise ValueError(f"{successes} successes out of {n}")
    z = z_for(confidence)
    z2 = z * z
    centre = (successes + z2 / 2.0) / (n + z2)
    half = z / (n + z2) * math.sqrt(successes * (n - successes) / n + z2 / 4.0)
    low = 0.0 if successes == 0 else max(0.0, centre - half)
    high = 1.0 if successes == n else min(1.0, centre + half)
    return low, high


def fisher_greater(x_a: int, n_a: int, x_b: int, n_b: int) -> float:
    """One-sided Fisher exact p for "A succeeds more often than B".

    With both margins fixed, A's success count is hypergeometric. The p-value is
    P(A's successes >= x_a) = sum over k >= x_a of C(n_a, k) C(n_b, s - k) / C(n_a + n_b, s),
    where s = x_a + x_b. Exact integers until the one division.
    """
    x_a, n_a, x_b, n_b = int(x_a), int(n_a), int(x_b), int(n_b)
    for x, n in ((x_a, n_a), (x_b, n_b)):
        if not 0 <= x <= n:
            raise ValueError(f"need 0 <= successes <= n, got {x}/{n}")
    s = x_a + x_b
    tail = sum(math.comb(n_a, k) * math.comb(n_b, s - k) for k in range(x_a, min(s, n_a) + 1))
    return tail / math.comb(n_a + n_b, s)


def holm(pvalues: list[float]) -> list[float]:
    """Holm's step-down adjusted p-values, in the input order.

    Sort ascending; the i-th smallest (i from 1) becomes max over j <= i of min(1, (m - j + 1) p_(j)).
    A hypothesis is rejected when its adjusted p is <= alpha.
    """
    m = len(pvalues)
    adjusted = [1.0] * m
    running = 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda j: pvalues[j])):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def percentile(values, q: float) -> float:
    """numpy's default ("linear") percentile, as a plain float."""
    return float(np.percentile(np.asarray(values, dtype=float), q, method="linear"))


def bootstrap_median_ci(values, resamples: int, seed: int, confidence: float) -> tuple[float, float]:
    """Percentile bootstrap interval for the median.

    numpy's legacy RandomState is used on purpose: its stream is frozen across numpy versions,
    so the same seed gives the same interval on the robot PC and on the HPC.
    """
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        raise ValueError("no values")
    draws = np.random.RandomState(seed).randint(0, x.size, size=(resamples, x.size))
    medians = np.median(x[draws], axis=1)
    tail = (1.0 - confidence) / 2.0 * 100.0
    low, high = np.percentile(medians, [tail, 100.0 - tail], method="linear")
    return float(low), float(high)


def fisher_power(p_a: float, p_b: float, n_a: int, n_b: int, alpha: float) -> float:
    """Exact power of the one-sided Fisher test at level alpha, when the true rates are p_a and p_b."""
    def pmf(n: int, p: float) -> list[float]:
        return [math.comb(n, k) * p ** k * (1.0 - p) ** (n - k) for k in range(n + 1)]

    pa, pb = pmf(n_a, p_a), pmf(n_b, p_b)
    return float(sum(pa[i] * pb[j] for i in range(n_a + 1) for j in range(n_b + 1)
                     if fisher_greater(i, n_a, j, n_b) <= alpha))


# ---------------------------------------------------------------- inputs

def file_sha256(path: str | Path) -> str | None:
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def read_pack_keys(paths: list[str]) -> dict[str, dict[str, str]]:
    """{pack_id: {code: trial}} from the keys `eval_log.py pack` wrote beside the log."""
    keys: dict[str, dict[str, str]] = {}
    for path in paths:
        data = es.load_json(path)
        if data.get("format") != el.PACK_KEY_FORMAT or not isinstance(data.get("codes"), dict):
            raise el.Refused(f"{path}: not a pack key written by eval_log.py pack")
        keys[str(data.get("pack_id"))] = {str(code): str(trial) for code, trial in data["codes"].items()}
    return keys


def read_scores(paths: list[str], keys: dict[str, dict[str, str]] | None = None
                ) -> tuple[dict[str, dict[str, dict]], list[str], list[str]]:
    """{trial: {scorer: their last call}}, the scorers in order of first appearance, and why any
    call was left out. Calls made on a pack's codes are mapped back to trials with its key."""
    keys = keys or {}
    by_trial: dict[str, dict[str, dict]] = {}
    order: list[str] = []
    left_out: list[str] = []
    for path in paths:
        if not Path(path).is_file():
            raise el.Refused(f"no scorer file at {path}")
        for number, record in enumerate(el.read_jsonl(path), start=1):
            if record.get("kind") != el.SCORE_KIND:
                continue
            where = f"{Path(path).name}:{number}"
            if "code" in record:
                pack = str(record.get("pack_id"))
                if pack not in keys:
                    raise el.Refused(f"{path} holds calls on pack {pack}'s codes: give that pack's key with "
                                     "--pack-key (pack_key_<pack>.json beside the trial log)")
                trial = keys[pack].get(str(record.get("code")))
                if trial is None:
                    left_out.append(f"{where}: code {record.get('code')!r} is not in pack {pack}'s key")
                    continue
            else:
                trial = str(record.get("trial"))
            if not isinstance(record.get("success"), bool):
                left_out.append(f"{where}: success is {record.get('success')!r}, not true or false")
                continue
            scorer = str(record.get("scorer") or "?")
            if scorer not in order:
                order.append(scorer)
            by_trial.setdefault(trial, {})[scorer] = dict(record, trial=trial)
    return by_trial, order, left_out


# ---------------------------------------------------------------- the analysis

def yes_no(value) -> str:
    return "unknown" if value is None else ("success" if value else "failure")


def tally(rows: list[dict], policies: list[str], call, confidence: float) -> dict:
    """Successes per policy under one way of calling each trial (call(row) -> bool, or None to leave it out)."""
    out = {}
    for policy in policies:
        mine = [r for r in rows if r["policy"] == policy]
        counted = [bool(c) for c in (call(r) for r in mine) if c is not None]
        x, n = sum(counted), len(counted)
        out[policy] = {"n_scheduled": len(mine), "n": n, "successes": x, "failures": n - x,
                       "rate": x / n if n else None, "wilson": wilson(x, n, confidence),
                       "n_run": sum(r["status"] == "done" for r in mine),
                       "missing": sum(r["status"] == "missing" for r in mine),
                       "withdrawn": sum(r["status"] == "withdrawn" for r in mine),
                       "void_attempts": sum(r["voids"] for r in mine),
                       "scored_blind": sum(r.get("source") == "scorer" for r in mine)}
    return out


def compare(counts: dict, pairs: list[list[str]], alpha: float) -> list[dict]:
    """The preregistered one-sided Fisher tests, Holm-corrected over exactly this list."""
    out = []
    for better, worse in pairs:
        a, b = counts[better], counts[worse]
        p = fisher_greater(a["successes"], a["n"], b["successes"], b["n"]) if a["n"] and b["n"] else None
        out.append({"hypothesis": f"{better} > {worse}", "better": better, "worse": worse,
                    "x_better": a["successes"], "n_better": a["n"], "x_worse": b["successes"], "n_worse": b["n"],
                    "p_one_sided": p})
    adjusted = holm([c["p_one_sided"] if c["p_one_sided"] is not None else 1.0 for c in out])
    for c, adj in zip(out, adjusted):
        c["p_holm"] = adj if c["p_one_sided"] is not None else None
        c["supported"] = c["p_one_sided"] is not None and adj <= alpha
    return out


def exploratory(counts: dict, policies: list[str], pairs: list[list[str]]) -> list[dict]:
    """Every other direction: unadjusted, not preregistered, for generating hypotheses only."""
    listed = {tuple(p) for p in pairs}
    out = []
    for a in policies:
        for b in policies:
            if a != b and (a, b) not in listed and counts[a]["n"] and counts[b]["n"]:
                out.append({"hypothesis": f"{a} > {b}", "x_better": counts[a]["successes"], "n_better": counts[a]["n"],
                            "x_worse": counts[b]["successes"], "n_worse": counts[b]["n"],
                            "p_one_sided_unadjusted": fisher_greater(counts[a]["successes"], counts[a]["n"],
                                                                     counts[b]["successes"], counts[b]["n"])})
    return out


def decide(call: bool, time_s: float | None, interventions: int, timed_out: bool | None,
           timeout: float) -> tuple[bool, list[str], bool]:
    """One trial under the preregistered rules: (success, rule problems, no in-time evidence).

    The facts decide first (an intervention, the limit reached), then the call on the end state,
    then a success needs its time within the limit, or the word that it ended before the limit.
    """
    problems = el.rule_problems(call, time_s, interventions, timeout, timed_out)
    unknown = not problems and el.time_unknown(call, time_s, timed_out)
    return bool(call) and not problems and not unknown, problems, unknown


def analyze(schedule: dict, records: list[dict], scores: dict[str, dict[str, dict]], scorers: list[str],
            episode_root: str | None = None, inputs: dict | None = None,
            score_problems: list[str] | None = None) -> dict:
    protocol = schedule["protocol"]
    policies = es.policy_ids(protocol)
    pairs = protocol["comparisons"]
    timeout = float(protocol["timeout_s"])
    alpha, confidence = float(protocol["alpha"]), float(protocol["confidence"])
    boot = protocol["bootstrap"]
    camera = protocol.get("camera") or {}
    convention = (protocol.get("dataset") or {}).get("zero_convention")
    schedule_hash = es.sha256_of(schedule)
    scheduled = {t["trial"]: t for t in schedule["trials"]}
    state = es.log_state(schedule, records)
    trials = el.effective_trials(state, records)
    deviations: list[dict] = []

    def flag(kind: str, detail: str, trial: str | None = None, policy: str | None = None) -> None:
        deviations.append({"kind": kind, "trial": trial, "policy": policy, "detail": detail})

    # 1. the log as a whole
    for record in records:
        if record.get("schedule_sha256") != schedule_hash:      # report refuses such a log; kept for callers
            flag("schedule_mismatch", f"a {record['kind']} record was written under schedule "
                                      f"{str(record.get('schedule_sha256'))[:12]}, not {schedule_hash[:12]}",
                 record.get("trial"))
        kind = record["kind"]
        if kind == "deviation":
            flag("logged", str(record.get("note", "")), record.get("trial"))
        elif kind == "withdraw":
            flag("withdrawn", f"{record.get('policy')} withdrawn: {record.get('reason')}", policy=record.get("policy"))
            if not isinstance(record.get("evidence"), dict):
                flag("withdrawal_without_evidence", "the withdrawal names no S2 evidence (3 consecutive collision "
                                                    "interventions, or an S1 contact)", policy=record.get("policy"))
        elif kind == "amend":
            flag("amended", f"{json.dumps(record.get('changes'))}: {record.get('reason')}", record.get("trial"))
        elif kind == "trial" and record.get("void") and record.get("trial") in scheduled:
            flag("void", f"void attempt: {record['void']}", record["trial"], record.get("policy"))
    for record in state["unknown"]:
        flag("unknown_trial", "a logged trial id that is not in the schedule", record.get("trial"))
    for record in state["duplicates"]:
        flag("duplicate", "a second result for this trial; the first one is used", record.get("trial"))
    if not scorers:
        flag("no_blind_scoring", "no scorer file was given: every outcome is the operator's own call")
    for problem in score_problems or []:
        flag("malformed_score", f"a scorer's call was left out: {problem}")
    for trial, calls in scores.items():
        if trial not in scheduled:
            flag("unknown_scored_trial", f"scored by {', '.join(calls)} but not in the schedule", trial)
        elif trial not in trials:
            flag("scored_but_not_logged", f"scored by {', '.join(calls)} but has no logged result", trial)

    # 2. one row per scheduled trial
    rows = []
    sessions: dict[str, dict] = {}                  # zero checks and conventions, per recorder session
    preset, tolerance = camera.get("preset_deg"), float(camera.get("tolerance_deg", 1.0))
    for t in schedule["trials"]:
        trial, policy = t["trial"], t["policy"]
        row = {k: t[k] for k in ("trial", "block", "repeat", "position", "arrangement", "policy")}
        row["voids"] = len(state["voids"].get(trial, []))
        record = trials.get(trial)
        if record is None:
            withdrawn = policy in state["withdrawn"]
            row.update(status="withdrawn" if withdrawn else "missing", success=False if withdrawn else None,
                       success_operator_rules=False if withdrawn else None,
                       source="withdrawn" if withdrawn else None, latency=[])
            if not withdrawn:
                flag("missing", "scheduled but never logged", trial, policy)
            rows.append(row)
            continue

        # what the operator logged, read strictly: anything but true is not a success
        if not isinstance(record.get("success"), bool):
            flag("malformed_record", f"success is {record.get('success')!r}, not true or false: read as not a "
                                     "success", trial, policy)
        operator_call = record.get("success") is True
        timed_out = record.get("timed_out")
        if timed_out is not None and not isinstance(timed_out, bool):
            flag("malformed_record", f"timed_out is {timed_out!r}, not true, false or null: read as not logged",
                 trial, policy)
            timed_out = None
        interventions = record.get("interventions") or 0
        if not es.is_int(interventions) or interventions < 0:
            flag("malformed_record", f"interventions is {interventions!r}, not a count: read as 1", trial, policy)
            interventions = 1
        operator_time = float(record["time_s"]) if es.is_number(record.get("time_s")) else None

        episode = el.resolve_episode(record.get("episode"), episode_root)
        check = el.check_episode(episode, t) if episode else {"found": False}
        if not record.get("episode"):
            flag("no_episode", "no recorded episode was given for this trial", trial, policy)
        elif not check["found"]:
            flag("episode_not_found", f"no episode.json at {episode}; the check made at logging time is used",
                 trial, policy)
            check = record.get("episode_check") or check
        elif check["problems"]:
            flag("episode_mismatch", "; ".join(check["problems"]), trial, policy)
        # Did the trial reach the limit? The operator's fact (--timed-out); without it, a recording that
        # ended within the limit (counted from the hand-over, or from its own start) shows it did not.
        ended = [check.get(k) for k in ("after_handover_s", "duration_s") if es.is_number(check.get(k))]
        limit_reached = timed_out if timed_out is not None else (False if any(s <= timeout for s in ended) else None)
        facts = interventions > 0 or limit_reached is True

        # the call on the end state: the first blind scorer's, else the operator's; then the rules
        calls = scores.get(trial, {})
        primary = next((s for s in scorers if s in calls), None)
        blind = calls[primary]["success"] if primary else None
        if primary and es.is_number(calls[primary].get("time_s")):
            time_s, time_source = float(calls[primary]["time_s"]), "scorer"
        elif operator_time is not None:
            time_s, time_source = operator_time, record.get("time_source") or "operator"
        else:
            time_s, time_source = None, None
        raw = blind if primary else operator_call
        caller = f"scorer {primary}" if primary else "the operator"
        success, problems, unknown = decide(raw, time_s, interventions, limit_reached, timeout)
        if problems:
            flag("rule_override", f"{caller} called it a success, but " + "; ".join(problems), trial, policy)
        elif unknown:
            flag("success_time_unknown", f"{caller} called it a success, but nothing shows it came within "
                                         f"{timeout:g} s: no completion time, the trial was not logged as ending "
                                         "before the limit, and the recording does not show it. Counted as a "
                                         "failure.", trial, policy)
        if primary and blind != operator_call:
            flag("scorer_disagreement", f"operator: {yes_no(operator_call)}; scorer {primary}: {yes_no(blind)}. "
                 + ("The logged facts decide it." if facts else "The scorer's call is used."), trial, policy)
        others = {s: c["success"] for s, c in calls.items() if s != primary}
        if any(v != blind for v in others.values()):
            flag("inter_scorer_disagreement", f"{primary}: {yes_no(blind)}; "
                 + "; ".join(f"{s}: {yes_no(v)}" for s, v in others.items()), trial, policy)
        if scorers and not primary:
            flag("unblinded", "no blind score for this trial: the operator's call is used", trial, policy)
        if record.get("out_of_order"):
            flag("out_of_order", str(record["out_of_order"]), trial, policy)

        if check.get("found"):
            mode = check.get("cam_mode_at_start")
            if mode != camera.get("mode", "still"):
                flag("camera_mode", f"the camera was in {mode!r} mode at the start", trial, policy)
            cam = check.get("cam_at_start")
            if preset and isinstance(cam, list) and len(cam) == 2 and all(es.is_number(c) for c in cam):
                off = max(abs(cam[0] - preset[0]), abs(cam[1] - preset[1]))
                if off > tolerance:
                    flag("camera_moved", f"pan/tilt {cam} is {off:.1f} deg from the preset {preset}", trial, policy)
            outcome = check.get("outcome")
            if outcome == "aborted":
                flag("episode_aborted", f"the recording ended by itself ({check.get('end_reason')})", trial, policy)
            elif outcome not in ("success", "failure"):
                flag("episode_unfinished", f"the recording was never closed (outcome {outcome!r})", trial, policy)
            elif (outcome == "success") != success:
                flag("episode_outcome_mismatch", f"the recording says {outcome}; the trial is scored "
                                                 f"{yes_no(success)}", trial, policy)
            if "task" in check and str(check.get("task") or "").strip() != str(protocol["task"]).strip():
                flag("task_mismatch", f"the recording's task is {check.get('task')!r}", trial, policy)
            if check.get("zero_changed"):
                flag("zero_changed", "the zero changed during the recording", trial, policy)
            after = check.get("after_handover_s")
            if success and time_s is not None and es.is_number(after) and time_s > after + 0.5:
                flag("time_after_recording", f"the completion time {time_s:g} s ({time_source}) is later than the "
                                             f"recording's end, {after:.1f} s after the hand-over", trial, policy)
            if "zero_check" in check:
                session = sessions.setdefault(str(check.get("session_id")), {"trials": [], "missing": [],
                                                                              "decisions": set(), "conventions": {}})
                session["trials"].append(trial)
                if not check.get("zero_check"):
                    session["missing"].append(trial)
                else:
                    session["decisions"].add(str(check.get("zero_check_decision")))
                session["conventions"].setdefault(str(check.get("zero_convention")), []).append(trial)

        latency = [float(x) for x in record.get("latency_ms") or []]
        op_success, _, _ = decide(operator_call, operator_time, interventions, limit_reached, timeout)
        by_rules = facts or bool(problems) or unknown
        row.update(
            status="done", success=success, source="rules" if by_rules else ("scorer" if primary else "operator"),
            scorer=primary, success_operator=operator_call, success_blind=blind, success_operator_rules=op_success,
            time_s=time_s, time_source=time_source, timed_out=timed_out, interventions=interventions,
            wall_s=record.get("wall_s"), latency_n=len(latency),
            latency_p50_ms=percentile(latency, 50) if latency else record.get("latency_p50_ms"),
            latency_p95_ms=percentile(latency, 95) if latency else record.get("latency_p95_ms"),
            episode=record.get("episode"), notes=record.get("notes", ""), latency=latency)
        rows.append(row)

    # zero checks and conventions, per recorder session (protocol 7.1: one zero check per session)
    seen_conventions: dict[str, list[str]] = {}
    for session_id, session in sessions.items():
        if session["missing"]:
            flag("no_zero_check", f"session {session_id}: no zero check in the recordings of "
                                  f"{', '.join(session['missing'])}")
        unsure = session["decisions"] - {"holds", "rejected"}
        if unsure:
            flag("zero_check_inconclusive", f"session {session_id}: zero check decision {', '.join(sorted(unsure))}")
        for value, members in session["conventions"].items():
            seen_conventions.setdefault(value, []).extend(members)
            if convention and value != convention:
                flag("zero_convention", f"session {session_id}: zero_convention {value!r}, the dataset's is "
                                        f"{convention!r} ({', '.join(members)})")
    if len(seen_conventions) > 1:
        flag("zero_convention_mixed", "trials recorded under more than one zero convention: "
             + "; ".join(f"{value} ({len(members)} trials)" for value, members in seen_conventions.items()))

    # 3. primary endpoint, sensitivity analyses
    primary_counts = tally(rows, policies, lambda r: r["success"], confidence)
    all_counts = tally(rows, policies, lambda r: bool(r["success"]), confidence)
    result = {
        "per_policy": primary_counts,
        "comparisons": compare(primary_counts, pairs, alpha),
        "exploratory": exploratory(primary_counts, policies, pairs),
    }
    sensitivity = {"missing_as_failure": {"per_policy": all_counts, "comparisons": compare(all_counts, pairs, alpha)}}
    if scorers:
        op_counts = tally(rows, policies, lambda r: r["success_operator_rules"], confidence)
        sensitivity["operator_calls"] = {"per_policy": op_counts, "comparisons": compare(op_counts, pairs, alpha)}
    else:
        sensitivity["operator_calls"] = None
    for policy in state["withdrawn"]:
        if policy in primary_counts and primary_counts[policy]["n_run"] == 0:
            flag("withdrawn_before_running", "withdrawn before any of its trials was logged: every one of its "
                                             "trials is an imputed failure", policy=policy)

    # 4. secondary endpoints
    secondary = {}
    for index, policy in enumerate(policies):
        done = [r for r in rows if r["policy"] == policy and r["status"] == "done"]
        wins = [r for r in done if r["success"]]
        times = [float(r["time_s"]) for r in wins if es.is_number(r.get("time_s"))]
        ci, why = None, None
        if len(times) >= boot["min_n"]:
            ci = bootstrap_median_ci(times, boot["resamples"], boot["seed"] + index, confidence)
        else:
            why = f"fewer than {boot['min_n']} timed successes"
        pooled = [x for r in done for x in r["latency"]]          # each request once: long trials weigh more
        per_trial_p50 = [percentile(r["latency"], 50) for r in done if r["latency"]]
        summary_only = [r for r in done if not r["latency"] and es.is_number(r.get("latency_p50_ms"))]
        summary_p95 = [float(r["latency_p95_ms"]) for r in summary_only if es.is_number(r.get("latency_p95_ms"))]
        walls = [float(r["wall_s"]) for r in done if es.is_number(r.get("wall_s"))]
        secondary[policy] = {
            "completion_time_s": {
                "n_trials": len(done), "n_successes": len(wins), "n_timed": len(times),
                "median": float(np.median(times)) if times else None, "bootstrap_ci": ci, "ci_note": why,
                "sources": dict(Counter(str(r.get("time_source")) for r in wins))},
            "interventions": {
                "n_trials": len(done), "total": sum(r["interventions"] for r in done),
                "per_trial": sum(r["interventions"] for r in done) / len(done) if done else None,
                "trials_with_any": sum(r["interventions"] > 0 for r in done)},
            "latency_ms": {
                "requests": len(pooled), "trials_with_samples": sum(bool(r["latency"]) for r in done),
                "p50": percentile(pooled, 50) if pooled else None,
                "p95": percentile(pooled, 95) if pooled else None,
                "per_trial_median_p50": float(np.median(per_trial_p50)) if per_trial_p50 else None,
                "trials_with_summary_only": len(summary_only),
                "summary_only_median_p50": float(np.median([float(r["latency_p50_ms"]) for r in summary_only]))
                if summary_only else None,
                "summary_only_median_p95": float(np.median(summary_p95)) if summary_p95 else None},
            "wall_s": {"n": len(walls), "median": float(np.median(walls)) if walls else None,
                       "mean": float(np.mean(walls)) if walls else None},
        }

    for row in rows:
        row.pop("latency", None)
    statuses = Counter(r["status"] for r in rows)
    return {
        "format": REPORT_FORMAT, "version": VERSION,
        "study_id": schedule["study_id"], "task": protocol["task"],
        "generated_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": "tools/lfd/analyze_eval.py",
        "inputs": inputs or {}, "schedule_sha256": schedule_hash, "protocol_sha256": schedule["protocol_sha256"],
        "settings": {
            "alpha_familywise_one_sided": alpha, "confidence": confidence, "timeout_s": timeout,
            "comparisons": pairs, "bootstrap": boot, "scorers_in_order": scorers,
            "methods": {"interval": "Wilson score, no continuity correction",
                        "test": "Fisher exact, one-sided, conditional on both margins",
                        "correction": "Holm step-down over the comparisons list",
                        "median_ci": "percentile bootstrap, numpy RandomState(seed + policy index)",
                        "percentiles": "numpy linear",
                        "outcome": "facts first (an intervention, or the time limit reached, is a failure); then "
                                   "the first blind scorer's call on the end state, else the operator's; a success "
                                   "needs its completion time within the limit, or evidence it ended before it",
                        "latency": "pooled over every request of the policy's trials (each request once), and "
                                   "the median of the per-trial p50s"}},
        "complete": statuses["missing"] == 0,
        "n_scheduled": len(rows), "n_done": statuses["done"], "n_missing": statuses["missing"],
        "n_withdrawn": statuses["withdrawn"], "n_void_attempts": sum(r["voids"] for r in rows),
        "primary": result, "sensitivity": sensitivity, "secondary": secondary,
        "deviation_counts": dict(Counter(d["kind"] for d in deviations)), "deviations": deviations,
        "trials": rows,
    }


# ---------------------------------------------------------------- the markdown report

def _p(value) -> str:
    if value is None:
        return "no data"
    return "<0.0001" if value < 1e-4 else f"{value:.4f}"


def _ci(ci, digits: int = 2) -> str:
    return "—" if ci is None else f"{ci[0]:.{digits}f}–{ci[1]:.{digits}f}"


def _num(value, digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _cell(text) -> str:
    return str(text).replace("|", "/").replace("\n", " ")


def _counts_table(counts: dict, confidence: float, blind_column: bool = True) -> list[str]:
    lines = [f"| policy | successes / n | failures | rate | {confidence:.0%} Wilson CI | missing | withdrawn "
             "| void attempts |" + (" blind-scored |" if blind_column else ""),
             "|---|---|---|---|---|---|---|---|" + ("---|" if blind_column else "")]
    for policy, c in counts.items():
        lines.append(f"| {policy} | {c['successes']} / {c['n']} | {c['failures']} | {_num(c['rate'], 2)} | "
                     f"{_ci(c['wilson'])} | {c['missing']} | {c['withdrawn']} | {c['void_attempts']} |"
                     + (f" {c['scored_blind']} |" if blind_column else ""))
    return lines


def _deviation_lines(deviations: list[dict]) -> list[str]:
    """Grouped by kind; a kind whose entries all say the same thing takes one line."""
    lines = []
    by_kind: dict[str, list[dict]] = {}
    for d in deviations:
        by_kind.setdefault(d["kind"], []).append(d)
    for kind, items in by_kind.items():
        details = {d["detail"] for d in items}
        if len(items) > 1 and len(details) == 1:
            trials = ", ".join(d["trial"] for d in items if d["trial"])
            detail = _cell(items[0]["detail"]).rstrip(".")
            lines.append(f"- **{kind}** ({len(items)}): {detail}." + (f" Trials: {trials}." if trials else ""))
            continue
        for d in items:
            where = " ".join(x for x in (d["trial"], f"({d['policy']})" if d["policy"] else None) if x)
            lines.append(f"- **{kind}**{' ' + where if where else ''}: {_cell(d['detail'])}")
    return lines


def _tests_table(comparisons: list[dict]) -> list[str]:
    lines = ["| hypothesis | successes | p, one-sided | p, Holm | verdict |", "|---|---|---|---|---|"]
    for c in comparisons:
        verdict = "no data" if c["p_one_sided"] is None else ("supported" if c["supported"] else "not supported")
        lines.append(f"| {c['hypothesis']} | {c['x_better']}/{c['n_better']} vs {c['x_worse']}/{c['n_worse']} | "
                     f"{_p(c['p_one_sided'])} | {_p(c['p_holm'])} | {verdict} |")
    return lines


def markdown(report: dict) -> str:
    s = report["settings"]
    out = [f"# Evaluation report: {report['study_id']}", "",
           f"Task: {report['task']}.",
           f"Made {report['generated_iso']} by `{report['tool']}`, from schedule `{report['schedule_sha256'][:12]}` "
           f"(protocol `{report['protocol_sha256'][:12]}`).", ""]
    if report["complete"]:
        out.append(f"**Complete:** all {report['n_scheduled']} scheduled trials are logged or withdrawn.")
    else:
        out.append(f"**INCOMPLETE:** {report['n_missing']} of {report['n_scheduled']} scheduled trials are missing. "
                   "If the study is still running, this is an interim look: log it as a deviation.")
    out += ["", f"## Primary endpoint: success within {s['timeout_s']:g} s, with no assistance", "",
            "Trials run, failures included. Withdrawn trials are imputed failures; missing ones are left out here "
            "and counted as failures in the first sensitivity analysis.",
            "",
            "How each trial was decided: the logged facts first (any intervention, or the time limit reached, is a "
            "failure, whatever anyone called it); then the first blind scorer's call on the end state, or else the "
            f"operator's; a success also needs its completion time within {s['timeout_s']:g} s, or evidence that "
            "it ended before the limit. Every override and disagreement is listed under Deviations.", ""]
    out += _counts_table(report["primary"]["per_policy"], s["confidence"])
    m = len(s["comparisons"])
    out += ["", f"### Preregistered comparisons: one-sided Fisher exact, Holm over m = {m}, "
                f"family-wise α = {s['alpha_familywise_one_sided']:g}", ""]
    out += _tests_table(report["primary"]["comparisons"])
    if report["primary"]["exploratory"]:
        out += ["", "Exploratory, not preregistered and not corrected (for new hypotheses only):", ""]
        out += ["| direction | successes | p, one-sided, unadjusted |", "|---|---|---|"]
        out += [f"| {e['hypothesis']} | {e['x_better']}/{e['n_better']} vs {e['x_worse']}/{e['n_worse']} | "
                f"{_p(e['p_one_sided_unadjusted'])} |" for e in report["primary"]["exploratory"]]
    out += ["", "## Sensitivity analyses", "", "### Every scheduled trial, missing ones counted as failures", ""]
    sens = report["sensitivity"]["missing_as_failure"]
    out += _counts_table(sens["per_policy"], s["confidence"]) + [""] + _tests_table(sens["comparisons"])
    if report["sensitivity"]["operator_calls"]:
        op = report["sensitivity"]["operator_calls"]
        out += ["", "### The operator's own calls instead of the blind scorer's (same rules)", ""]
        out += _counts_table(op["per_policy"], s["confidence"], blind_column=False) + [""]
        out += _tests_table(op["comparisons"])
    out += ["", "## Secondary endpoints", "",
            "| policy | completion time of successes, median s [95% bootstrap CI] | timed / successes / trials "
            "| interventions per trial (trials with any) | latency p50 / p95 ms (requests) | wall time per trial, "
            "median s (n) |",
            "|---|---|---|---|---|---|"]
    for policy, sec in report["secondary"].items():
        ct, iv, lat, wall = sec["completion_time_s"], sec["interventions"], sec["latency_ms"], sec["wall_s"]
        ci = _ci(ct["bootstrap_ci"], 1) if ct["bootstrap_ci"] else (ct["ci_note"] or "—")
        if lat["requests"]:
            latency = (f"{lat['p50']:.0f} / {lat['p95']:.0f} ({lat['requests']}); per-trial median p50 "
                       f"{lat['per_trial_median_p50']:.0f}")
        elif lat["summary_only_median_p50"] is not None:
            latency = (f"{lat['summary_only_median_p50']:.0f} / {_num(lat['summary_only_median_p95'], 0)} "
                       f"(medians of {lat['trials_with_summary_only']} per-trial summaries)")
        else:
            latency = "—"
        out.append(f"| {policy} | {_num(ct['median'])} [{ci}] | {ct['n_timed']} / {ct['n_successes']} / "
                   f"{ct['n_trials']} | {_num(iv['per_trial'], 2)} ({iv['trials_with_any']}) | {latency} | "
                   f"{_num(wall['median'])} ({wall['n']}) |")
    out += ["", f"## Deviations ({len(report['deviations'])})", ""]
    if report["deviations"]:
        counts = ", ".join(f"{k} {v}" for k, v in sorted(report["deviation_counts"].items()))
        out += [f"By kind: {counts}.", ""] + _deviation_lines(report["deviations"])
    else:
        out.append("None.")
    out += ["", "## Every trial", "",
            "| trial | block | policy | arrangement | status | result | called by | time s | interventions "
            "| voids | notes |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in report["trials"]:
        result = yes_no(r.get("success")) if r["status"] != "missing" else "—"
        out.append(f"| {r['trial']} | {r['block']} | {r['policy']} | {r['arrangement']} | {r['status']} | {result} | "
                   f"{r.get('source') or '—'} | {_num(r.get('time_s'))} | {r.get('interventions', '—')} | "
                   f"{r['voids']} | {_cell(r.get('notes', ''))} |")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- commands

def cmd_report(args) -> int:
    schedule, warnings = es.load_schedule(args.schedule)
    if not Path(args.log).is_file():
        raise el.Refused(f"no trial log at {args.log}")
    records = es.read_log(args.log)
    el.check_log_hash(records, es.sha256_of(schedule))        # refused here as in eval_log and eval_schedule
    scores, scorers, score_problems = read_scores(args.scores, read_pack_keys(args.pack_key))
    inputs = {"schedule": str(args.schedule), "log": str(args.log), "log_sha256": file_sha256(args.log),
              "scores": [{"path": str(p), "sha256": file_sha256(p)} for p in args.scores],
              "pack_keys": [{"path": str(p), "sha256": file_sha256(p)} for p in args.pack_key],
              "episode_root": args.episode_root}
    report = analyze(schedule, records, scores, scorers, args.episode_root, inputs, score_problems)
    for warning in warnings:
        report["deviations"].insert(0, {"kind": "schedule_warning", "trial": None, "policy": None, "detail": warning})
        report["deviation_counts"]["schedule_warning"] = report["deviation_counts"].get("schedule_warning", 0) + 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    (out / "report.md").write_text(markdown(report), encoding="utf-8")
    state = "complete" if report["complete"] else f"INCOMPLETE ({report['n_missing']} missing)"
    print(f"wrote {out / 'report.json'} and {out / 'report.md'}: {state}, "
          f"{len(report['deviations'])} deviation(s)")
    for policy, c in report["primary"]["per_policy"].items():
        print(f"  {policy:<14} {c['successes']:>3}/{c['n']:<3} Wilson {_ci(c['wilson'])}")
    for c in report["primary"]["comparisons"]:
        print(f"  {c['hypothesis']:<24} p {_p(c['p_one_sided'])}  Holm {_p(c['p_holm'])}  "
              f"{'supported' if c['supported'] else 'not supported'}")
    return 0


def cmd_power(args) -> int:
    n_b = args.n_b or args.n
    pairs = []
    for item in args.rates.split(","):
        low, high = (float(x) for x in item.split(":"))
        pairs.append((high, low))
    adjusted = args.alpha / args.m
    print(f"Exact power of the one-sided Fisher test, n = {args.n} against {n_b}.")
    print(f"'first Holm step' tests at alpha/m = {adjusted:.4g} (the worst case); 'alone' at alpha = {args.alpha:g}.")
    print("")
    print("| true rate, better | true rate, worse | power, first Holm step | power, alone |")
    print("|---|---|---|---|")
    for high, low in pairs:
        print(f"| {high:.2f} | {low:.2f} | {fisher_power(high, low, args.n, n_b, adjusted):.2f} | "
              f"{fisher_power(high, low, args.n, n_b, args.alpha):.2f} |")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("report", help="compute the preregistered statistics; write report.json and report.md")
    p.add_argument("--schedule", default="schedule.json")
    p.add_argument("--log", default="trials.jsonl")
    p.add_argument("--scores", action="append", default=[],
                   help="a blind scorer file (repeatable; the first scorer to call a trial decides it)")
    p.add_argument("--pack-key", action="append", default=[],
                   help="the key of the pack the scorer was given (pack_key_<pack>.json; repeatable)")
    p.add_argument("--episode-root", help="where the episodes are now: <root>/<session_id>/ep_NNNN")
    p.add_argument("--out", default=".", help="directory for report.json and report.md")
    p.set_defaults(func=cmd_report)
    p = sub.add_parser("power", help="exact power table for the one-sided Fisher test")
    p.add_argument("--n", type=int, default=20, help="trials per policy")
    p.add_argument("--n-b", type=int, default=0, help="trials for the worse policy (default: --n)")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--m", type=int, default=4, help="Holm family size (the protocol's number of comparisons)")
    p.add_argument("--rates", default="0.10:0.50,0.20:0.60,0.30:0.70,0.40:0.80,0.50:0.90,0.20:0.80,0.30:0.90,"
                                      "0.10:0.70", help="worse:better true-rate pairs")
    p.set_defaults(func=cmd_power)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (el.Refused, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
