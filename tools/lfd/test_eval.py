"""Tests for the live-evaluation tools: statistics, schedule balance, log checks, blind scoring, report.

Standalone: one PASS or FAIL line per check, exit status 1 if any failed. Nothing here
touches the robot, the recorder or the network. It writes only under --tmp (default: a
fresh temporary directory) and removes its own files when every check passed.

Besides the statistics, it replays the review's sandboxes (sb2, sb3): a logged timeout that a
scorer calls a success, a success in a recording that ran past the limit, a success on an
aborted recording, a withdrawal with no evidence, and repeated voids (S3).

  python tools/lfd/test_eval.py [--tmp DIR] [--keep]

Reference values:
- Fisher, one-sided: 9/16 vs 2/16 gives 0.0117, the value the owner's LeHome campaign
  reports (lehome-fold-repro, campaigns/20260926-anchor-diagnostic-v8/REPORT.md); exactly
  1,505,296 / 129,024,480. Fisher's tea-tasting table, 3/4 vs 1/4, gives 17/70.
- Wilson 95%: 10/20 gives 0.2993-0.7007; Newcombe (1998, Stat Med 17:857, method 3)
  gives 81/263: 0.2553-0.3662, 15/148: 0.0624-0.1605, 0/20: 0-0.1611, 1/29: 0.0061-0.1718.
- Holm: R's p.adjust(c(0.01, 0.04, 0.03), "holm") gives 0.03, 0.06, 0.06.
When scipy is installed, every 20-against-20 table is also checked against scipy.stats.fisher_exact.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze_eval as ae  # noqa: E402
import eval_log as el  # noqa: E402
import eval_schedule as es  # noqa: E402

RESULTS: list[bool] = []


def check(ok, what: str, detail: str = "") -> None:
    RESULTS.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {what}" + (f"   [{detail}]" if detail and not ok else ""), flush=True)


def close(a, b, tol: float) -> bool:
    return a is not None and b is not None and abs(a - b) <= tol


def run(main, argv: list[str]) -> tuple[int, str, str]:
    """A tool's main() in this process, with its output captured."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = main([str(a) for a in argv])
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
    return code, out.getvalue(), err.getvalue()


def counts_from(trials: list[dict], key: str) -> dict:
    out: dict = {}
    for t in trials:
        out[t[key]] = out.get(t[key], 0) + 1
    return out


def carry_and_positions(trials: list[dict]) -> tuple[dict, dict]:
    """Recomputed from the trial list itself, not from the schedule's own balance summary."""
    positions: dict = {}
    carry: dict = {}
    blocks: dict = {}
    for t in trials:
        positions[(t["policy"], t["position"])] = positions.get((t["policy"], t["position"]), 0) + 1
        blocks.setdefault(t["block"], []).append(t)
    for block in blocks.values():
        block.sort(key=lambda t: t["position"])
        for a, b in zip(block, block[1:]):
            carry[(a["policy"], b["policy"])] = carry.get((a["policy"], b["policy"]), 0) + 1
    return positions, carry


# ---------------------------------------------------------------- 1. statistics

def test_statistics() -> None:
    p = ae.fisher_greater(9, 16, 2, 16)
    check(close(p, 0.0117, 5e-5) and close(p, 1505296 / 129024480, 1e-15),
          "Fisher one-sided 9/16 vs 2/16 = 0.0117 (LeHome campaign value)", f"{p}")
    check(close(ae.fisher_greater(3, 4, 1, 4), 17 / 70, 1e-15), "Fisher tea-tasting 3/4 vs 1/4 = 17/70")
    check(ae.fisher_greater(0, 20, 0, 20) == 1.0 and ae.fisher_greater(20, 20, 20, 20) == 1.0,
          "Fisher gives p = 1 when nobody (or everybody) succeeds")
    check(close(ae.fisher_greater(20, 20, 0, 20), 1 / math.comb(40, 20), 1e-25), "Fisher 20/20 vs 0/20 = 1/C(40,20)")
    worst = 0.0
    for xa, xb in itertools.product(range(9), range(7)):
        s = xa + xb
        point = math.comb(8, xa) * math.comb(6, s - xa) / math.comb(14, s)
        worst = max(worst, abs(ae.fisher_greater(xa, 8, xb, 6) + ae.fisher_greater(xb, 6, xa, 8) - 1 - point))
    check(worst < 1e-12, "Fisher tails: P(X >= x) + P(X <= x) = 1 + P(X = x) on every 8-vs-6 table", f"{worst}")
    try:
        from scipy.stats import fisher_exact
    except ImportError:
        print("SKIP  scipy is not installed: no cross-check against scipy.stats.fisher_exact")
    else:
        worst = max(abs(ae.fisher_greater(xa, 20, xb, 20)
                        - fisher_exact([[xa, 20 - xa], [xb, 20 - xb]], alternative="greater").pvalue)
                    for xa, xb in itertools.product(range(21), range(21)))
        check(worst < 1e-12, "Fisher matches scipy.stats.fisher_exact(greater) on all 441 tables of 20 vs 20",
              f"{worst}")
    try:
        ae.fisher_greater(5, 4, 0, 4)
        check(False, "Fisher refuses more successes than trials")
    except ValueError:
        check(True, "Fisher refuses more successes than trials")

    known = {(10, 20): (0.2993, 0.7007), (81, 263): (0.2553, 0.3662), (15, 148): (0.0624, 0.1605),
             (0, 20): (0.0, 0.1611), (1, 29): (0.0061, 0.1718)}
    misses = []
    for (x, n), (lo, hi) in known.items():
        got = ae.wilson(x, n, 0.95)
        if not (close(got[0], lo, 5.1e-5) and close(got[1], hi, 5.1e-5)):
            misses.append(f"{x}/{n}: {got}")
    check(not misses, "Wilson 95% matches 10/20 and Newcombe's four examples to 4 decimals", "; ".join(misses))
    check(ae.wilson(0, 20)[0] == 0.0 and ae.wilson(20, 20)[1] == 1.0 and ae.wilson(0, 0) is None,
          "Wilson ends at exactly 0 and 1; no interval for n = 0")
    check(close(ae.z_for(0.95), 1.959963984540054, 1e-9), "z for 95% is 1.959964")

    check(all(close(a, b, 1e-12) for a, b in zip(ae.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])),
          "Holm [0.01, 0.04, 0.03] -> [0.03, 0.06, 0.06] (R p.adjust)")
    check(all(close(a, b, 1e-12) for a, b in zip(ae.holm([0.04, 0.005, 0.03, 0.02]), [0.06, 0.02, 0.06, 0.06])),
          "Holm [0.04, 0.005, 0.03, 0.02] -> [0.06, 0.02, 0.06, 0.06]")
    check(ae.holm([0.6, 0.7]) == [1.0, 1.0] and ae.holm([]) == [], "Holm caps at 1 and accepts an empty family")

    times = [8.2, 9.1, 11.5, 12.0, 14.8, 15.2, 21.0, 26.4]
    first = ae.bootstrap_median_ci(times, 10000, 7, 0.95)
    again = ae.bootstrap_median_ci(times, 10000, 7, 0.95)
    draws = np.random.RandomState(7).randint(0, len(times), size=(10000, len(times)))
    medians = np.median(np.asarray(times)[draws], axis=1)
    by_hand = (float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5)))
    check(first == again == by_hand and min(times) <= first[0] <= float(np.median(times)) <= first[1] <= max(times),
          "bootstrap median CI: fixed RandomState seed, same interval every run, contains the median", f"{first}")
    check(close(ae.percentile([1, 2, 3, 4], 50), 2.5, 1e-12) and close(ae.percentile([1, 2, 3, 4], 95), 3.85, 1e-12),
          "percentiles are numpy's linear method")
    null = ae.fisher_power(0.5, 0.5, 20, 20, 0.05)
    check(null <= 0.05 and ae.fisher_power(0.9, 0.5, 20, 20, 0.05) > ae.fisher_power(0.7, 0.5, 20, 20, 0.05),
          "Fisher power: at most alpha when the rates are equal, and grows with the gap", f"{null}")


# ---------------------------------------------------------------- 2. schedules

def protocol_with(**changes) -> dict:
    protocol = es.default_protocol()
    protocol.update(changes)
    return protocol


def test_schedule(tmp: Path) -> None:
    protocol = es.default_protocol()
    problems = es.check_protocol(protocol)
    check(problems == [], "the template protocol passes its own checks", str(problems))
    schedule = es.build_schedule(protocol)
    trials = schedule["trials"]
    policies = es.policy_ids(protocol)
    per_policy = counts_from(trials, "policy")
    check(len(trials) == 60 and all(per_policy[p] == 20 for p in policies),
          "default schedule: 60 trials, 20 per policy (10 arrangements x 2)")
    per_cell: dict = {}
    for t in trials:
        per_cell[(t["arrangement"], t["policy"])] = per_cell.get((t["arrangement"], t["policy"]), 0) + 1
    check(len(per_cell) == 30 and set(per_cell.values()) == {2}, "each policy runs exactly twice on every arrangement")
    blocks: dict = {}
    for t in trials:
        blocks.setdefault(t["block"], []).append(t)
    check(all(sorted(x["policy"] for x in b) == sorted(policies) and len({x["arrangement"] for x in b}) == 1
              for b in blocks.values()) and len(blocks) == 20,
          "every block is one arrangement with every policy once")
    positions, carry = carry_and_positions(trials)
    check(max(positions.values()) - min(positions.values()) <= 1 and len(positions) == 9,
          "20 blocks of 3 cannot balance exactly: positions differ by at most 1", str(positions))
    check(max(carry.values()) - min(carry.values()) <= 2 and len(carry) == 6,
          "carry-over (who follows whom) differs by at most 2", str(carry))
    check(schedule["balance"]["position_spread"] == max(positions.values()) - min(positions.values()),
          "the schedule's own balance summary agrees with a recount")
    check(all(not any(p in t["trial"] for p in policies) for t in trials) and len({t["trial"] for t in trials}) == 60,
          "trial ids are unique and do not name the policy (blinding)")
    check(es.build_schedule(es.default_protocol()) == schedule, "the same protocol gives the same schedule")
    check(es.build_schedule(protocol_with(seed=1))["trials"] != trials, "another seed gives another order")
    check(schedule["balance"]["arrangements_with_a_repeated_order"] == 0,
          "no arrangement sees the same policy order twice")

    # exact balance when the blocks fill whole designs
    twelve = protocol_with(arrangements=[f"A{i:02d}" for i in range(1, 13)])
    positions, carry = carry_and_positions(es.build_schedule(twelve)["trials"])
    check(set(positions.values()) == {8} and set(carry.values()) == {8} and len(carry) == 6,
          "12 arrangements x 2 (24 blocks of 3): every position 8 times, every ordered pair 8 times",
          f"{positions} {carry}")
    four = protocol_with(policies=["a", "b", "c", "d"], arrangements=["A1", "A2", "A3", "A4"],
                         comparisons=[["b", "a"]])
    positions, carry = carry_and_positions(es.build_schedule(four)["trials"])
    check(set(positions.values()) == {2} and set(carry.values()) == {2} and len(carry) == 12,
          "4 policies, 8 blocks: every position twice, every ordered pair twice", f"{positions} {carry}")
    five = protocol_with(policies=["a", "b", "c", "d", "e"], arrangements=["A1", "A2"], repeats=5,
                         comparisons=[["b", "a"]])
    positions, carry = carry_and_positions(es.build_schedule(five)["trials"])
    check(set(positions.values()) == {2} and set(carry.values()) == {2} and len(carry) == 20,
          "5 policies, 10 blocks (one full Williams design): positions and ordered pairs twice each")

    bad = []
    for n in range(1, 8):
        squares = es.williams_squares(n)
        rows = [row for square in squares for row in square]
        for square in squares:
            latin = all(sorted(row) == list(range(n)) for row in square) and \
                all(sorted(col) == list(range(n)) for col in zip(*square))
            if not latin:
                bad.append(f"n={n} not Latin")
        pairs: dict = {}
        for row in rows:
            for a, b in zip(row, row[1:]):
                pairs[(a, b)] = pairs.get((a, b), 0) + 1
        if n > 1 and (len(pairs) != n * (n - 1) or len(set(pairs.values())) != 1):
            bad.append(f"n={n} carry-over {pairs}")
    check(not bad, "Williams design for 1-7 policies: Latin squares with every ordered pair equally often",
          "; ".join(bad))

    errors = es.check_protocol(protocol_with(policies=["act", "act"], repeats=0, seed=True,
                                             comparisons=[["act", "ghost"]]))
    check(len(errors) >= 4 and any("unique" in e for e in errors) and any("repeats" in e for e in errors)
          and any("seed" in e for e in errors) and any("comparisons" in e for e in errors),
          "protocol checks catch repeated ids, zero repeats, a boolean seed and an unknown comparison", str(errors))

    # the command line: template, make, refusing to overwrite a different schedule
    study = tmp / "make"
    study.mkdir()
    code, _, _ = run(es.main, ["template", "--out", study / "protocol.json"])
    code2, _, _ = run(es.main, ["make", "--protocol", study / "protocol.json", "--out", study / "schedule.json"])
    code3, out3, _ = run(es.main, ["make", "--protocol", study / "protocol.json", "--out", study / "schedule.json"])
    changed = es.load_json(study / "protocol.json")
    changed["seed"] = 99
    es.write_json(study / "protocol2.json", changed)
    code4, _, err4 = run(es.main, ["make", "--protocol", study / "protocol2.json", "--out", study / "schedule.json"])
    check(code == 0 and code2 == 0 and code3 == 0 and "already" in out3 and code4 == 2 and "differs" in err4,
          "make: writes once, accepts the identical schedule, refuses to overwrite a different one")
    code5, _, _ = run(es.main, ["template", "--out", study / "protocol.json"])
    check(code5 == 2, "template refuses to overwrite a protocol without --force")

    tampered = es.load_json(study / "schedule.json")
    tampered["protocol"]["timeout_s"] = 45.0
    es.write_json(study / "tampered.json", tampered)
    try:
        es.load_schedule(study / "tampered.json")
        check(False, "a schedule whose protocol copy was edited is refused")
    except ValueError:
        check(True, "a schedule whose protocol copy was edited is refused")
    swapped = es.load_json(study / "schedule.json")
    swapped["trials"][0]["policy"], swapped["trials"][1]["policy"] = \
        swapped["trials"][1]["policy"], swapped["trials"][0]["policy"]
    es.write_json(study / "swapped.json", swapped)
    _, warnings = es.load_schedule(study / "swapped.json")
    check(any("edited by hand" in w for w in warnings), "a hand-edited trial list is flagged")


# ---------------------------------------------------------------- 3. the trial log

TASK = es.default_protocol()["task"]
EXIF = b"Exif\x00\x00DateTimeOriginal 2026:11:01 10:00:00"


def jpeg(payload: bytes, exif: bytes = EXIF, trailer: bytes = b"") -> bytes:
    """A small JPEG-shaped file: SOI, APP0, APP1 (metadata), SOS with `payload` as its scan, EOI, then `trailer`."""
    app0 = b"\xff\xe0" + (2 + 5).to_bytes(2, "big") + b"JFIF\x00"
    app1 = b"\xff\xe1" + (2 + len(exif)).to_bytes(2, "big") + exif if exif else b""
    sos = b"\xff\xda" + (2 + 6).to_bytes(2, "big") + b"\x01\x01\x00\x00\x3f\x00"
    return b"\xff\xd8" + app0 + app1 + sos + payload + b"\xff\xd9" + trailer


def clean_jpeg(payload: bytes) -> bytes:
    """What strip_jpeg must make of jpeg(payload): the same file without APP1 and without a trailer."""
    return jpeg(payload, exif=b"")


def fake_episode(root: Path, session: str, index: int, trial, arrangement: str, policy: str,
                 mode: str = "policy", duration: float = 12.0, cam=(0.0, -20.0), handover: float | None = 1.5,
                 frame_step: float | None = None, outcome: str | None = "success", end_reason: str = "operator",
                 task: str = TASK, zero_check: bool = True, convention: str = "hanging=0") -> Path:
    """A recorded episode as docs/LFD_RECORDING_FORMAT.md lays it out: episode.json, tap rows whose src
    turns to "policy" at the hand-over (none if handover is None), and small JPEG frames carrying metadata."""
    ep = root / session / f"ep_{index:04d}"
    ep.mkdir(parents=True)
    t0, w0 = 5000.0 + 100 * index, 1.79e9 + 100 * index
    meta = {"format": "bhl-lfd-episode", "version": 1, "episode_index": index, "session_id": session,
            "task": task, "mode": mode, "policy": policy, "operator": "test",
            "arrangement": arrangement, "trial": trial, "start": {"t": t0, "wall": w0},
            "end": {"t": t0 + duration, "wall": w0 + duration + 0.25} if outcome else None, "outcome": outcome,
            "end_reason": end_reason if outcome else None, "discarded": False, "notes": "",
            "counts": {"rows": 10, "frames": 2, "row_gaps_over_50ms": 0, "frame_gaps_over_67ms": 0},
            "cam_mode_at_start": "still", "cam_at_start": list(cam), "zero_changed": False,
            "zero_convention": convention,
            "zero_check": {"format": "bhl-zero-check", "decision": "holds", "date": "2026-11-01"} if zero_check
            else None, "warnings": []}
    (ep / "episode.json").write_text(json.dumps(meta))
    rows = []
    for k in range(int(duration / 0.25) + 1):
        t = t0 + 0.25 * k
        src = "policy" if handover is not None and t >= t0 + handover else "teleop"
        rows.append(json.dumps({"v": 1, "seq": k, "t": t, "state": "ARMED", "src": src}))
    (ep / "rows.jsonl").write_text("\n".join(rows) + "\n")
    times = [t0, t0 + duration / 2, t0 + duration] if frame_step is None else \
        [t0 + frame_step * k for k in range(int(duration / frame_step) + 1)]
    offset, lines, data = 0, [], []
    for i, t in enumerate(times):
        frame = jpeg(f"frame-{i}-{trial}".encode())
        lines.append(json.dumps({"i": i, "off": offset, "len": len(frame), "t_cap": t, "t_seen": t, "mtime_ns": 0}))
        data.append(frame)
        offset += len(frame)
    (ep / "frames.mjpeg").write_bytes(b"".join(data))
    (ep / "frames.jsonl").write_text("\n".join(lines) + "\n")
    return ep


def new_study(tmp: Path, name: str) -> tuple[Path, Path, dict, list]:
    study = tmp / name
    study.mkdir()
    sched, log = study / "schedule.json", study / "trials.jsonl"
    es.write_json(sched, es.build_schedule(es.default_protocol()))
    schedule, _ = es.load_schedule(sched)
    return sched, log, schedule, ["--schedule", sched, "--log", log, "--operator", "tester"]


def test_jpeg() -> None:
    payload = b"scan\xff\x00data\xff\xd3more"          # a stuffed 0xFF and a restart marker stay in the scan
    raw = jpeg(payload, trailer=jpeg(b"preview", exif=b"Exif\x00\x00GPS 44.56 N"))
    clean = el.strip_jpeg(raw)
    check(clean == clean_jpeg(payload), "strip_jpeg drops APP1 and whatever follows the end marker, scan untouched")
    check(clean is not None and b"2026:11:01" not in clean and b"GPS" not in clean and clean.endswith(b"\xff\xd9"),
          "the capture date and an appended preview's metadata are gone")
    commented = b"\xff\xd8" + b"\xff\xfe" + (2 + 9).to_bytes(2, "big") + b"camera 7a" + jpeg(b"x")[2:]
    check(el.strip_jpeg(commented) == clean_jpeg(b"x"), "a COM segment is dropped too")
    check(el.strip_jpeg(b"\x89PNG\r\n\x1a\n...") is None and el.strip_jpeg(raw[:30]) is None
          and el.strip_jpeg(jpeg(b"x")[:-2]) is None, "a PNG, a cut header or a missing end marker do not parse")


def test_log(tmp: Path) -> None:
    sched, log, schedule, files = new_study(tmp, "log")
    study = sched.parent
    order = schedule["trials"]

    def log_trial(*extra):
        return run(el.main, ["trial", *files, *extra])

    first = order[0]
    code, out, err = log_trial("--trial", first["trial"], "--success", "yes", "--time-s", "12.5",
                               "--policy", first["policy"], "--arrangement", first["arrangement"])
    record = es.read_log(log)[-1] if log.exists() else {}
    check(code == 0 and record.get("success") is True and record.get("timed_out") is False
          and record.get("schedule_sha256") == es.sha256_of(schedule), "logs the next trial, with the schedule's hash",
          err)
    code, _, err = log_trial("--trial", first["trial"], "--success", "no", "--timed-out", "no")
    check(code == 2 and "already logged" in err, "refuses a second result for the same trial (no duplicates)", err)
    code, _, err = log_trial("--trial", "T999", "--success", "no", "--timed-out", "no")
    check(code == 2 and "unknown trial id" in err, "refuses an unknown trial id", err)
    third = order[2]
    code, _, err = log_trial("--trial", third["trial"], "--success", "no", "--timed-out", "no")
    check(code == 2 and f"the next trial is {order[1]['trial']}" in err,
          "refuses a trial out of order without a reason",
          err)
    code, _, err = log_trial("--trial", third["trial"], "--success", "no", "--timed-out", "no",
                             "--out-of-order", "pi server restarting")
    check(code == 0 and es.read_log(log)[-1]["out_of_order"] == "pi server restarting",
          "accepts it with --out-of-order REASON, kept for the deviations list", err)
    second = order[1]
    other = next(p for p in es.policy_ids(schedule["protocol"]) if p != second["policy"])
    code, _, err = log_trial("--trial", second["trial"], "--success", "no", "--timed-out", "no", "--policy", other)
    check(code == 2 and "in the schedule, not" in err, "refuses a policy that does not match the schedule", err)
    code, _, err = log_trial("--trial", second["trial"], "--success", "yes", "--time-s", "9", "--interventions", "1")
    check(code == 2 and "intervention" in err, "refuses a success with an intervention (it is a failure)", err)
    code, _, err = log_trial("--trial", second["trial"], "--success", "yes", "--time-s", "31")
    check(code == 2 and "limit" in err and "--timed-out yes" in err, "refuses a success past the 30 s limit", err)
    code, _, err = log_trial("--trial", second["trial"], "--success", "yes")
    check(code == 2 and "completion time" in err and "hand-over" in err,
          "refuses a success with no completion time (it is never taken from the episode)", err)
    code, _, err = log_trial("--trial", second["trial"], "--success", "yes", "--time-s", "9", "--timed-out", "yes")
    check(code == 2 and "limit is a failure" in err, "refuses a success that the operator says reached the limit", err)
    code, _, err = log_trial("--trial", second["trial"], "--success", "no")
    check(code == 2 and "--timed-out" in err, "a failure without an intervention must say whether it reached the limit",
          err)
    code, _, err = log_trial("--trial", second["trial"], "--success", "no", "--timed-out", "maybe")
    check(code == 2 and "not yes or no" in err, "--timed-out takes yes or no", err)
    code, _, err = log_trial("--trial", second["trial"], "--success", "no", "--timed-out", "no", "--collision")
    check(code == 2 and "--interventions" in err, "--collision without an intervention is refused", err)
    code, _, err = log_trial("--trial", second["trial"], "--void", "x", "--timed-out", "yes")
    check(code == 2 and "void attempt has no result" in err, "a void attempt takes no --timed-out", err)
    code, _, err = log_trial("--trial", second["trial"], "--void", "the serving job died before the first action")
    state = es.log_state(schedule, es.read_log(log))
    check(code == 0 and state["pending"][0]["trial"] == second["trial"] and len(state["voids"][second["trial"]]) == 1,
          "a void attempt is logged and the trial stays next", err)
    code, _, err = log_trial("--trial", second["trial"], "--success", "no", "--interventions", "1", "--collision",
                             "--latency-ms", "110, 125,140", "--notes", "headed for the torso")
    record = es.read_log(log)[-1]
    check(code == 0 and record["latency_ms"] == [110.0, 125.0, 140.0] and record["collision"] is True
          and record["timed_out"] is None, "a failure with a collision intervention and latency samples is logged", err)

    # latency files in their three shapes
    (study / "lat.json").write_text("[100, 101.5, 130]")
    (study / "lat.jsonl").write_text('{"latency_ms": 90}\n{"latency_ms": 95.5}\n')
    (study / "lat.txt").write_text("# per request, ms\n80\n85.5\n")
    (study / "bad.txt").write_text("fast\n")
    shapes = [el.read_latency_file(study / name) for name in ("lat.json", "lat.jsonl", "lat.txt")]
    check(shapes == [[100.0, 101.5, 130.0], [90.0, 95.5], [80.0, 85.5]],
          "latency files: JSON list, JSON lines with latency_ms, one number per line", str(shapes))
    try:
        el.read_latency_file(study / "bad.txt")
        check(False, "a latency file that is not numbers is refused")
    except el.Refused:
        check(True, "a latency file that is not numbers is refused")

    # recorded episodes are checked against the schedule
    nxt = es.next_pending(schedule, es.read_log(log))
    wrong = fake_episode(study / "rec", "20261101_100000_bhl", 1, nxt["index"], nxt["arrangement"], nxt["policy"],
                         mode="teleop")
    code, _, err = log_trial("--trial", nxt["trial"], "--success", "yes", "--time-s", "9", "--episode", wrong)
    check(code == 2 and "mode" in err, "refuses an episode recorded in teleop mode (a voice start without a prime)",
          err)
    stray = fake_episode(study / "rec", "20261101_100000_bhl", 3, nxt["index"] + 1, nxt["arrangement"],
                         nxt["policy"])
    code, _, err = log_trial("--trial", nxt["trial"], "--success", "yes", "--time-s", "9", "--episode", stray)
    check(code == 2 and "trial is" in err, "refuses an episode recorded under another trial number", err)
    good = fake_episode(study / "rec", "20261101_100000_bhl", 2, nxt["index"], nxt["arrangement"], nxt["policy"],
                        duration=14.25)
    code, _, err = log_trial("--trial", nxt["trial"], "--success", "yes", "--episode", good)
    check(code == 2 and "completion time" in err, "an episode does not stand in for the completion time", err)
    code, _, err = log_trial("--trial", nxt["trial"], "--success", "yes", "--time-s", "11.0", "--episode", good)
    record = es.read_log(log)[-1]
    found = record.get("episode_check") or {}
    check(code == 0 and close(record["time_s"], 11.0, 1e-9) and record["time_source"] == "operator"
          and close(record["wall_s"], 14.5, 1e-9) and found.get("found") and close(found.get("handover_s"), 1.5, 1e-9)
          and close(found.get("after_handover_s"), 12.75, 1e-9) and found.get("zero_check") is True,
          "a matching episode (trial number, as the recorder stores it) gives the wall time and the hand-over", err)
    probe = {"trial": "T017", "index": 17}
    check(el.same_trial(17, probe) and el.same_trial("17", probe) and el.same_trial("T017", probe)
          and not el.same_trial(18, probe) and not el.same_trial(True, probe) and not el.same_trial(None, probe),
          "the recorder's trial field may be the number or the id, nothing else")

    code, out, _ = run(es.main, ["next", "--schedule", sched, "--log", log, "--json"])
    info = json.loads(out) if code == 0 else {}
    want = es.next_pending(schedule, es.read_log(log))
    check(info.get("next", {}).get("trial") == want["trial"] and info["start_fields"]["mode"] == "policy"
          and info["start_fields"]["trial"] == want["index"] and info["start_fields"]["policy"] == want["policy"]
          and info.get("s3_stop") is None, "next --json names the next trial and the recorder's fields")
    code, out, _ = run(es.main, ["next", "--schedule", sched, "--log", log])
    prime = (f"tools/lfd/record_ctl.py prime --mode policy --trial {want['index']} --arrangement "
             f"{want['arrangement']} --policy {want['policy']} --task \"{TASK}\"")
    check(code == 0 and prime in out and "agent, start episode" in out,
          "next prints the recorder prime command, then the voice start", out)

    # amendments, deviations
    code, _, err = run(el.main, ["amend", *files, "--trial", first["trial"], "--success", "no", "--timed-out", "no",
                                 "--reason", "the block bounced out after the photo"])
    check(code == 0 and es.read_log(log)[-1]["kind"] == "amend", "amend appends a correction with its reason", err)
    code, _, err = run(el.main, ["amend", *files, "--trial", first["trial"], "--success", "yes", "--interventions", "2",
                                 "--reason", "x"])
    check(code == 2 and "intervention" in err, "amend keeps the intervention rule", err)
    code, _, err = run(el.main, ["amend", *files, "--trial", third["trial"], "--timed-out", "yes", "--reason", "x"])
    code2, _, err2 = run(el.main, ["amend", *files, "--trial", third["trial"], "--success", "yes", "--time-s", "20",
                                   "--reason", "x"])
    check(code == 0 and code2 == 2 and "time limit" in err2, "amend keeps the time-limit fact: no success over it",
          err + err2)
    code, _, err = run(el.main, ["amend", *files, "--trial", order[-1]["trial"], "--success", "no", "--reason", "x"])
    check(code == 2 and "no logged result" in err, "amend refuses a trial with no result", err)
    code, _, _ = run(el.main, ["deviation", *files, "--note", "tilt servo slipped; preset re-set"])
    check(code == 0 and es.read_log(log)[-1]["kind"] == "deviation", "deviation notes go into the same log")

    # integrity of the file itself, and a log made under another schedule (refused by every tool)
    cut = study / "cut.jsonl"
    cut.write_text(log.read_text() + '{"kind": "trial", "tri')
    code, _, err = run(el.main, ["deviation", "--schedule", sched, "--log", cut, "--operator", "t", "--note", "x"])
    check(code == 2, "refuses to append to a log whose last line is cut off", err)
    other_schedule = study / "other.json"
    es.write_json(other_schedule, es.build_schedule(protocol_with(seed=5)))
    code, _, err = run(el.main, ["deviation", "--schedule", other_schedule, "--log", log, "--operator", "t",
                                 "--note", "x"])
    code2, _, err2 = run(es.main, ["next", "--schedule", other_schedule, "--log", log])
    code3, _, err3 = run(es.main, ["show", "--schedule", other_schedule, "--log", log])
    check(code == 2 and "another schedule" in err and code2 == 2 and "another schedule" in err2
          and code3 == 2 and "another schedule" in err3,
          "a log made under another schedule is refused by eval_log, next and show", err + err2 + err3)
    env_before = os.environ.pop("BHL_EVAL_OPERATOR", None)
    code, _, err = run(el.main, ["deviation", "--schedule", sched, "--log", log, "--note", "x"])
    if env_before is not None:
        os.environ["BHL_EVAL_OPERATOR"] = env_before
    check(code == 2 and "operator" in err, "every record needs an operator name", err)

    # the scripts run as scripts (sibling imports from their own directory)
    proc = subprocess.run([sys.executable, str(HERE / "eval_schedule.py"), "next", "--schedule", str(sched),
                           "--log", str(log)], capture_output=True, text=True, timeout=60)
    check(proc.returncode == 0 and "Next:" in proc.stdout and "record_ctl.py prime" in proc.stdout,
          "eval_schedule.py next runs as a script", proc.stderr)


def test_withdraw_and_s3(tmp: Path) -> None:
    sched, log, schedule, files = new_study(tmp, "withdraw")
    order = schedule["trials"]
    act = "act"
    code, _, err = run(el.main, ["withdraw", *files, "--policy", act, "--reason", "harm"])
    check(code == 2 and "S2" in err and not log.exists(),
          "withdraw is refused before the policy has run (S2 needs its evidence)", err)
    mine = []
    for t in order:
        if len(mine) == 3:
            break
        if t["policy"] == act:
            extra = ["--success", "no", "--interventions", "1", "--collision", "--notes", "headed for the torso"]
            mine.append(t["trial"])
        else:
            extra = ["--success", "no", "--timed-out", "no"]
        code, _, err = run(el.main, ["trial", *files, "--trial", t["trial"], *extra])
        if code:
            check(False, "set-up for withdrawal", err)
            return
    gone = next(p for p in es.policy_ids(schedule["protocol"]) if p != act)
    code, _, err = run(el.main, ["withdraw", *files, "--policy", gone, "--reason", "harm"])
    check(code == 2 and "collision" in err, "withdraw is refused without 3 consecutive collision interventions", err)
    code, _, err = run(el.main, ["withdraw", *files, "--policy", gone, "--s1-contact", mine[0], "--reason", "x"])
    check(code == 2 and "not a logged trial of" in err, "--s1-contact must name a logged trial of that policy", err)
    code, out, err = run(el.main, ["withdraw", *files, "--policy", act, "--reason", "3 collisions in a row"])
    record = es.read_log(log)[-1]
    state = es.log_state(schedule, es.read_log(log))
    check(code == 0 and record["evidence"]["trials"] == mine and all(t["policy"] != act for t in state["pending"])
          and state["withdrawn_trials"], "withdraw with the S2 evidence: recorded, remaining trials leave the list",
          err)
    code, _, err = run(el.main, ["trial", *files, "--trial", next(t["trial"] for t in state["withdrawn_trials"]),
                                 "--success", "no", "--timed-out", "no", "--out-of-order", "x"])
    check(code == 2 and "withdrawn" in err, "refuses a trial of a withdrawn policy", err)
    done_gone = next(r["trial"] for r in es.read_log(log) if r.get("kind") == "trial" and r.get("policy") == gone)
    code, _, err = run(el.main, ["withdraw", *files, "--policy", gone, "--s1-contact", done_gone,
                                 "--reason", "hit the camera"])
    check(code == 0 and es.read_log(log)[-1]["evidence"] == {"rule": "S1 contact", "trial": done_gone, "notes": ""},
          "withdraw on an S1 contact names that trial as its evidence", err)

    # S3: a third void of one trial stops the session; a deviation resumes it
    sched, log, schedule, files = new_study(tmp, "s3")
    first, second, third = schedule["trials"][:3]
    codes = [run(el.main, ["trial", *files, "--trial", first["trial"], "--void", f"link down {k}"]) for k in range(3)]
    check([c for c, _, _ in codes] == [0, 0, 0] and "STOP (S3)" in codes[2][2] and "STOP" not in codes[1][2],
          "the third void of one trial is logged, with the S3 stop", codes[2][2])
    code, _, err = run(el.main, ["trial", *files, "--trial", first["trial"], "--void", "again"])
    code2, _, err2 = run(el.main, ["trial", *files, "--trial", first["trial"], "--success", "no", "--timed-out", "no"])
    out = run(es.main, ["next", "--schedule", sched, "--log", log])[1]
    check(code == 2 and code2 == 2 and "S3" in err and "S3" in err2 and "STOP (S3)" in out,
          "while the S3 stop holds, no trial is logged and next says STOP", err + err2)
    code, out, _ = run(el.main, ["deviation", *files, "--note", "S3: the serving job ran out of memory; restarted"])
    code2, _, err2 = run(el.main, ["trial", *files, "--trial", first["trial"], "--success", "no", "--timed-out", "no"])
    check(code == 0 and "ends the S3 stop" in out and code2 == 0, "a deviation with the cause and fix resumes it", err2)
    fourth = schedule["trials"][3]
    for t in (second, third, fourth):                     # all in repeat 1: the third void in the session
        run(el.main, ["trial", *files, "--trial", t["trial"], "--void", "link down"])
        if t is not fourth:
            run(el.main, ["trial", *files, "--trial", t["trial"], "--success", "no", "--timed-out", "no"])
    state = es.log_state(schedule, es.read_log(log))
    check(state["s3"] is not None and "session" in state["s3"],
          "three voids in one session (repeat) also stop it, counted from the last resumption", str(state["s3"]))


# ---------------------------------------------------------------- 4. blind scoring

def test_blind(tmp: Path) -> None:
    sched, log, schedule, files = new_study(tmp, "blind")
    study = sched.parent
    trials = schedule["trials"][:4]
    plan = [  # (episode kwargs, trial arguments)
        ({"duration": 34.0, "frame_step": 1.0}, ["--success", "no", "--timed-out", "yes"]),
        ({"duration": 12.0}, ["--success", "yes", "--time-s", "10"]),
        ({"duration": 9.0}, ["--success", "no", "--timed-out", "no"]),
        ({"duration": 12.0, "handover": None}, ["--success", "yes", "--time-s", "8"]),
    ]
    before = study / "before.jpg"
    before.write_bytes(jpeg(b"before-photo"))
    after_png = study / "after.png"
    after_png.write_bytes(b"\x89PNG\r\n\x1a\nnot a jpeg")
    episodes = {}
    for i, (t, (kwargs, extra)) in enumerate(zip(trials, plan)):
        episodes[t["trial"]] = fake_episode(study / "rec", "20261102_090000_bhl", i, t["index"], t["arrangement"],
                                            t["policy"], **kwargs)
        photos = ["--before", before, "--after", after_png] if i == 0 else []
        code, _, err = run(el.main, ["trial", *files, "--trial", t["trial"], *extra, "--episode",
                                     episodes[t["trial"]], *photos])
        if code:
            check(False, "set-up for blind scoring", err)
            return
    pack = study / "pack"
    code, out, err = run(el.main, ["pack", *files, "--out", pack, "--video"])
    key_path = study / "pack_key_pack.json"
    key = json.loads(key_path.read_text()) if key_path.exists() else {}
    sheet = [json.loads(line) for line in (pack / "sheet.jsonl").read_text().splitlines()] if code == 0 else []
    names = [str(p.relative_to(pack)) for p in pack.rglob("*")] if pack.exists() else []
    listed = (pack / "sheet.jsonl").read_text() + " ".join(names) if code == 0 else ""
    text = listed + (pack / "README.txt").read_text() if code == 0 else ""
    check(code == 0 and len(sheet) == 4 and "README.txt" in names and key.get("pack_id") == sheet[0]["pack_id"],
          "pack writes a sheet, instructions, and a key beside the log (not in the pack)", err)
    codes = {trial: c for c, trial in key.get("codes", {}).items()}
    check(sorted(codes) == sorted(episodes) and len(set(codes.values())) == 4
          and not any(t["trial"] in text for t in schedule["trials"])
          and not any(p in listed for p in es.policy_ids(schedule["protocol"])) and "episode.json" not in names,
          "the scorer's files carry random codes: no trial id and no policy anywhere in the pack")
    check(all(os.stat(pack / name).st_mtime == el.PACK_MTIME for name in names),
          "every packed file has the same mtime, so file times show no run order")
    t0, t1 = trials[0]["trial"], trials[3]["trial"]
    c0, c1 = codes.get(t0, "?"), codes.get(t1, "?")
    check((pack / f"{c0}_first_frame.jpg").read_bytes() == clean_jpeg(f"frame-0-{trials[0]['index']}".encode()),
          "the first frame is the recording's own bytes, its metadata stripped")
    check((pack / f"{c0}_end_frame.jpg").read_bytes() == clean_jpeg(f"frame-31-{trials[0]['index']}".encode()),
          "the end frame is the last one at or before the hand-over + 30 s (here frame 31 of 34)")
    check((pack / f"{c1}_end_frame.jpg").read_bytes() == clean_jpeg(f"frame-2-{trials[3]['index']}".encode()),
          "without a hand-over in the tap, the end frame is the recording's last")
    frames0 = sorted(p.name for p in (pack / f"{c0}_frames").iterdir()) if (pack / f"{c0}_frames").is_dir() else []
    frames1 = sorted(p.name for p in (pack / f"{c1}_frames").iterdir()) if (pack / f"{c1}_frames").is_dir() else []
    check(len(frames0) == 35 and frames0[0] == "00000_-0001.50s.jpg" and frames0[31] == "00031_+0029.50s.jpg"
          and frames1 == ["00000.jpg", "00001.jpg", "00002.jpg"],
          "with --video, every frame named by its time from the hand-over (no time without one)", str(frames0[:2]))
    packed = b"".join(p.read_bytes() for p in pack.rglob("*.jpg"))
    check(b"2026:11:01" not in packed and (pack / f"{c0}_before_photo.jpg").read_bytes() == clean_jpeg(b"before-photo")
          and not any("after_photo" in n for n in names) and "left out" in err,
          "photo and frame metadata are stripped; a photo that is not a JPEG is left out, with a warning", err)
    code, _, _ = run(el.main, ["pack", *files, "--out", pack])
    code2, _, err2 = run(el.main, ["pack", *files, "--out", study / "pack2", "--key", key_path])
    code3, _, err3 = run(el.main, ["pack", *files, "--out", study / "pack3", "--key", study / "pack3" / "k.json"])
    check(code == 2 and code2 == 2 and "earlier pack" in err2 and code3 == 2 and "inside" in err3,
          "pack refuses a used folder, an existing key, and a key inside the scorer's folder", err2 + err3)

    score = ["score", "--sheet", pack / "sheet.jsonl", "--scorer", "ana"]
    code, out, err = run(el.main, [*score, "--code", c0, "--success", "yes"])
    check(code == 0 and (pack / "scores.jsonl").exists(), "a scorer's call goes to scores.jsonl beside the sheet", err)
    code, _, err = run(el.main, [*score, "--code", c0, "--success", "no"])
    check(code == 2 and "already scored" in err, "a second call on the same code is refused", err)
    code, _, err = run(el.main, [*score, "--code", c0, "--success", "yes", "--amend", "looked again"])
    check(code == 0, "it can be replaced with --amend REASON", err)
    code, _, err = run(el.main, [*score, "--code", t0, "--success", "no"])
    check(code == 2 and "not on the sheet" in err, "a trial id is not a code on the sheet", err)
    try:
        ae.read_scores([pack / "scores.jsonl"])
        check(False, "calls on codes cannot be read without the pack's key")
    except el.Refused as exc:
        check("--pack-key" in str(exc), "calls on codes cannot be read without the pack's key")
    scores, scorers, left_out = ae.read_scores([pack / "scores.jsonl"], ae.read_pack_keys([key_path]))
    check(scorers == ["ana"] and scores[t0]["ana"]["success"] is True and not left_out,
          "the key maps each code back to its trial; the later, amended call is the one read")

    # the analysis: facts first. T(0) reached the limit, so the scorer's "yes" does not make it a success.
    code, out, err = run(ae.main, ["report", "--schedule", sched, "--log", log, "--scores", pack / "scores.jsonl",
                                   "--pack-key", key_path, "--out", study / "report"])
    report = json.loads((study / "report" / "report.json").read_text()) if code == 0 else {"trials": []}
    row = next((r for r in report["trials"] if r["trial"] == t0), {})
    kinds = {(d["kind"], d["trial"]) for d in report.get("deviations", [])}
    check(code == 0 and row.get("success") is False and row.get("source") == "rules" and ("rule_override", t0) in kinds
          and ("scorer_disagreement", t0) in kinds, "a scorer's success never overturns a logged timeout", err)


def test_sandboxes(tmp: Path) -> None:
    """The reviewer's sandboxes (sb2): a timeout scored 'yes', a late-ended success, an aborted recording."""
    sched, log, schedule, files = new_study(tmp, "sb2")
    study = sched.parent
    t1, t2, t3, t4, t5, t6 = schedule["trials"][:6]
    rec = study / "rec"
    ep1 = fake_episode(rec, "S1", 0, t1["index"], t1["arrangement"], t1["policy"], duration=34.0, outcome="failure")
    code, _, err = run(el.main, ["trial", *files, "--trial", t1["trial"], "--success", "no", "--timed-out", "yes",
                                 "--episode", ep1, "--notes", "released at 31 s"])
    check(code == 0 and es.read_log(log)[-1]["timed_out"] is True, "sb2 T001: a timeout is logged as a fact", err)
    ep2 = fake_episode(rec, "S1", 1, t2["index"], t2["arrangement"], t2["policy"], duration=31.5)
    code, _, err = run(el.main, ["trial", *files, "--trial", t2["trial"], "--success", "yes", "--episode", ep2])
    check(code == 2 and "--success no" not in err, "sb2 T002: without --time-s, a long episode is not called a failure",
          err)
    code, out, err = run(el.main, ["trial", *files, "--trial", t2["trial"], "--success", "yes", "--time-s", "27",
                                   "--episode", ep2])
    check(code == 0 and "SUCCESS in 27.0 s" in out, "sb2 T002: a 27 s success in a 31.5 s recording is accepted", err)
    ep3 = fake_episode(rec, "S1", 2, t3["index"], t3["arrangement"], t3["policy"], duration=4.0, outcome="aborted",
                       end_reason="tap rows stopped")
    code, _, err = run(el.main, ["trial", *files, "--trial", t3["trial"], "--success", "yes", "--time-s", "3.5",
                                 "--episode", ep3])
    check(code == 2 and "--accept-aborted" in err, "sb2 T003: a success on an aborted recording is refused", err)
    code, out, err = run(el.main, ["trial", *files, "--trial", t3["trial"], "--success", "yes", "--time-s", "3.5",
                                   "--episode", ep3, "--accept-aborted", "seen live: released in the bowl at 3.5 s"])
    tail = es.read_log(log)[-2:]
    check(code == 0 and tail[0]["accept_aborted"] and tail[1]["kind"] == "deviation" and t3["trial"] in tail[1]["note"],
          "with --accept-aborted REASON it is logged, and logged as a deviation too", err)
    # T004 failed before the limit; the scorer will call it a success: the blind call on the end state counts
    code, _, err = run(el.main, ["trial", *files, "--trial", t4["trial"], "--success", "no", "--timed-out", "no"])
    # T005: a scorer's time past the limit; T006: a legacy failure line with no timed_out, scored 'yes', untimed
    run(el.main, ["trial", *files, "--trial", t5["trial"], "--success", "yes", "--time-s", "25"])
    legacy = dict(es.read_log(log)[-1], trial=t6["trial"], policy=t6["policy"], arrangement=t6["arrangement"],
                  block=t6["block"], position=t6["position"], success=False, time_s=None, time_source=None)
    legacy.pop("timed_out")
    el.append_jsonl(log, legacy)
    bad = dict(legacy, trial=schedule["trials"][6]["trial"], policy=schedule["trials"][6]["policy"], success="no")
    el.append_jsonl(log, bad)
    scores = study / "scores.jsonl"
    for t, ok, time_s in ((t1, True, None), (t2, True, None), (t4, True, None), (t5, True, 31.0), (t6, True, None)):
        el.append_jsonl(scores, {"kind": "score", "trial": t["trial"], "scorer": "ana", "success": ok,
                                 "time_s": time_s, "notes": "", "amend": None, "logged_wall": 0.0})
    el.append_jsonl(scores, {"kind": "score", "trial": t3["trial"], "scorer": "ana", "success": "yes"})
    records = es.read_log(log)
    by_trial, scorers, left_out = ae.read_scores([scores])
    report = ae.analyze(schedule, records, by_trial, scorers, score_problems=left_out)
    rows = {r["trial"]: r for r in report["trials"]}
    kinds = {(d["kind"], d["trial"]) for d in report["deviations"]}
    check(rows[t1["trial"]]["success"] is False and rows[t1["trial"]]["source"] == "rules"
          and ("rule_override", t1["trial"]) in kinds, "T001: the logged timeout decides, whatever the scorer says")
    check(rows[t2["trial"]]["success"] is True and rows[t2["trial"]]["time_s"] == 27.0
          and rows[t2["trial"]]["source"] == "scorer", "T002: the scorer's success, timed by the operator's 27 s")
    check(rows[t3["trial"]]["success"] is True and ("episode_aborted", t3["trial"]) in kinds
          and ("logged", t3["trial"]) in kinds and len(left_out) == 1,
          "T003: the accepted success stands, flagged as aborted and as a logged deviation; a bad call is left out")
    check(rows[t4["trial"]]["success"] is True and ("scorer_disagreement", t4["trial"]) in kinds,
          "T004: ended before the limit, the blind call on the end state wins, and the disagreement is listed")
    check(rows[t5["trial"]]["success"] is False and ("rule_override", t5["trial"]) in kinds,
          "T005: the scorer's own time past the limit makes it a failure")
    check(rows[t6["trial"]]["success"] is False and ("success_time_unknown", t6["trial"]) in kinds,
          "T006: a success with no evidence it came within the limit counts as a failure, flagged")
    seventh = schedule["trials"][6]["trial"]
    check(rows[seventh]["success"] is False and ("malformed_record", seventh) in kinds,
          "a success value that is not true/false (\"no\") is read as not a success, and flagged")
    check(any(k == "malformed_score" for k, _ in kinds), "a scorer's malformed call is listed")
    check(rows[t1["trial"]]["success_operator_rules"] is False and rows[t2["trial"]]["success_operator_rules"] is True,
          "the operator-calls sensitivity analysis applies the same rules")

    # zero checks per session, zero conventions, the task text
    sched, log, schedule, files = new_study(tmp, "zero")
    study = sched.parent
    plan = [("A", {}), ("B", {"zero_check": False}), ("B", {"convention": "hanging=q_hang", "task": "Fold the cloth"})]
    for i, (t, (session, kwargs)) in enumerate(zip(schedule["trials"], plan)):
        ep = fake_episode(study / "rec", session, i, t["index"], t["arrangement"], t["policy"], **kwargs)
        code, _, err = run(el.main, ["trial", *files, "--trial", t["trial"], "--success", "no", "--timed-out", "no",
                                     "--episode", ep])
        if code:
            check(False, "set-up for the zero checks", err)
            return
    report = ae.analyze(schedule, es.read_log(log), {}, [])
    kinds = [d["kind"] for d in report["deviations"]]
    details = " ".join(d["detail"] for d in report["deviations"] if d["kind"] == "no_zero_check")
    check(kinds.count("no_zero_check") == 1 and "session B" in details and "zero_convention_mixed" in kinds
          and kinds.count("task_mismatch") == 1, "zero checks per session, mixed conventions and the task are flagged",
          str(kinds))
    protocol = dict(schedule["protocol"], dataset=dict(schedule["protocol"]["dataset"], zero_convention="hanging=0"))
    report = ae.analyze(dict(schedule, protocol=protocol), es.read_log(log), {}, [])
    check(sum(d["kind"] == "zero_convention" for d in report["deviations"]) == 1,
          "a session under another convention than the dataset's is flagged")


# ---------------------------------------------------------------- 5. the analysis of a toy log

SUCCESS_PLAN = {"act": {0, 5, 10, 15}, "pi05": {0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17},
                "groot_n17": {0, 2, 4, 6, 8, 10, 12, 14, 16}}


def fisher_reference(a_success: int, a_n: int, b_success: int, b_n: int) -> float:
    """The owner's LeHome formula, written differently on purpose (denominator C(total, a_n))."""
    total, successes = a_n + b_n, a_success + b_success
    return sum(math.comb(successes, k) * math.comb(total - successes, a_n - k)
               for k in range(a_success, min(successes, a_n) + 1)) / math.comb(total, a_n)


def test_analysis(tmp: Path) -> None:
    study = tmp / "analysis"
    study.mkdir()
    schedule = es.build_schedule(es.default_protocol())
    es.write_json(study / "schedule.json", schedule)
    schedule, _ = es.load_schedule(study / "schedule.json")
    log, scores_path = study / "trials.jsonl", study / "scores.jsonl"
    base = {"study_id": schedule["study_id"], "schedule_sha256": es.sha256_of(schedule), "operator": "tester",
            "logged_wall": 0.0, "logged_iso": "2026-11-01T00:00:00+00:00"}
    trials = schedule["trials"]
    missing = trials[-1]["trial"]
    swapped = (trials[9]["trial"], trials[10]["trial"])
    seen: dict = {}
    plan = []                      # (trial, operator success, interventions, time_s, latency, wall)
    for t in trials:
        k = seen.get(t["policy"], 0)
        seen[t["policy"]] = k + 1
        ok = k in SUCCESS_PLAN[t["policy"]]
        offset = {"act": 40.0, "pi05": 120.0, "groot_n17": 200.0}[t["policy"]]
        plan.append((t, ok, 0, 6.0 + k if ok else None, [offset + k, offset + 2 * k, offset + 3 * k],
                     (6.0 + k) if ok else 30.0))
    # one pi05 failure had an intervention; the scorer will (wrongly) call it a success
    rescued = next(t for t, ok, *_ in plan if t["policy"] == "pi05" and not ok)
    plan = [(t, ok, 1 if t is rescued else iv, ts, lat, wall) for t, ok, iv, ts, lat, wall in plan]

    def trial_record(t, ok, iv, ts, lat, wall, **extra):
        return {"kind": "trial", "trial": t["trial"], "policy": t["policy"], "arrangement": t["arrangement"],
                "block": t["block"], "position": t["position"], "void": None, "success": ok, "time_s": ts,
                "time_source": "operator" if ts is not None else None, "interventions": iv,
                "timed_out": False if ok else (True if not iv else None), "collision": False, "wall_s": wall,
                "wall_source": "operator", "latency_ms": lat, "latency_file": None, "latency_p50_ms": None,
                "latency_p95_ms": None, "episode": None, "episode_check": None, "before": None, "after": None,
                "out_of_order": None, "notes": "", **base, **extra}

    by_id = {t["trial"]: (t, ok, iv, ts, lat, wall) for t, ok, iv, ts, lat, wall in plan}
    void_trial = trials[4]["trial"]
    for t in trials:
        if t["trial"] == missing or t["trial"] == swapped[0]:
            continue
        if t["trial"] == swapped[1]:
            el.append_jsonl(log, trial_record(*by_id[swapped[1]], out_of_order="pi server restarting"))
            el.append_jsonl(log, trial_record(*by_id[swapped[0]]))
            continue
        if t["trial"] == void_trial:
            el.append_jsonl(log, dict(trial_record(*by_id[void_trial]), void="link down before the first action",
                                      success=None))
        el.append_jsonl(log, trial_record(*by_id[t["trial"]]))
    # the blind scorer agrees everywhere except: one disagreement, the rescued trial, and one trial left unscored
    disagree = next(t for t, ok, *_ in plan if t["policy"] == "groot_n17" and ok)
    unscored = next(t for t, ok, *_ in plan if t["policy"] == "act" and not ok)
    for t, ok, *_ in plan:
        if t["trial"] in (missing, unscored["trial"]):
            continue
        call = (not ok) if t is disagree else (True if t is rescued else ok)
        el.append_jsonl(scores_path, {"kind": "score", "trial": t["trial"], "scorer": "ana", "success": call,
                                      "time_s": None, "notes": "", "amend": None, "logged_wall": 0.0})

    records = es.read_log(log)
    scores, scorers, left_out = ae.read_scores([scores_path])
    report = ae.analyze(schedule, records, scores, scorers, score_problems=left_out)

    expected: dict = {p: [0, 0] for p in SUCCESS_PLAN}
    for t, ok, iv, ts, *_ in plan:
        if t["trial"] == missing:
            continue
        call = (not ok) if t is disagree else (True if t is rescued else ok)
        final = call and iv == 0 and (ts is None or ts <= 30.0)
        expected[t["policy"]][0] += final
        expected[t["policy"]][1] += 1
    got = {p: [c["successes"], c["n"]] for p, c in report["primary"]["per_policy"].items()}
    check(got == expected, "per-policy successes and n follow the blind calls and the rules, failures kept",
          f"{got} vs {expected}")
    check(got["pi05"][0] == 12 and got["groot_n17"][0] == 8 and got["act"][0] == 4,
          "the intervention rule overrides the scorer's success, and the scorer's failure call wins")
    wilson_ok = all(report["primary"]["per_policy"][p]["wilson"] == list(ae.wilson(x, n)) or
                    tuple(report["primary"]["per_policy"][p]["wilson"]) == ae.wilson(x, n)
                    for p, (x, n) in got.items())
    check(wilson_ok, "each policy's Wilson interval is the Wilson function's")
    comps = report["primary"]["comparisons"]
    ref = [fisher_reference(c["x_better"], c["n_better"], c["x_worse"], c["n_worse"]) for c in comps]
    check(len(comps) == 4 and all(close(c["p_one_sided"], r, 1e-12) for c, r in zip(comps, ref)),
          "the four preregistered p-values match an independently written Fisher formula")
    check(all(close(c["p_holm"], h, 1e-12) for c, h in zip(comps, ae.holm(ref)))
          and all(c["supported"] == (c["p_holm"] <= 0.05) for c in comps),
          "Holm-adjusted p-values and verdicts over the family of 4")
    check(len(report["primary"]["exploratory"]) == 2 and
          {e["hypothesis"] for e in report["primary"]["exploratory"]} == {"act > pi05", "act > groot_n17"},
          "the two directions not preregistered are listed as exploratory")

    kinds: dict = {}
    for d in report["deviations"]:
        kinds.setdefault(d["kind"], []).append(d["trial"])
    check(kinds.get("missing") == [missing] and report["complete"] is False and report["n_missing"] == 1,
          "the missing trial is flagged and the report says it is incomplete")
    check(kinds.get("out_of_order") == [swapped[1]], "the out-of-order trial is flagged with its reason")
    check(kinds.get("void") == [void_trial], "the void attempt is flagged")
    check(rescued["trial"] in kinds.get("rule_override", [])
          and disagree["trial"] in kinds.get("scorer_disagreement", []),
          "a rule override and a scorer disagreement are flagged")
    rows = {r["trial"]: r for r in report["trials"]}
    check(rows[rescued["trial"]]["source"] == "rules" and rows[disagree["trial"]]["source"] == "scorer",
          "the report says which trials the rules decided, and which the blind call")
    check(kinds.get("unblinded") == [unscored["trial"]], "a trial the scorer did not call is flagged as unblinded")
    sens = report["sensitivity"]["missing_as_failure"]["per_policy"]
    check(all(c["n"] == 20 for c in sens.values()),
          "sensitivity: every scheduled trial counts, missing ones as failures")
    op = report["sensitivity"]["operator_calls"]["per_policy"]
    check(op["groot_n17"]["successes"] == 9 and op["pi05"]["successes"] == 12,
          "sensitivity: the operator's own calls give the planned counts")

    sec = report["secondary"]
    wins = [float(r["time_s"]) for r in report["trials"] if r["policy"] == "pi05" and r["success"]]
    check(close(sec["pi05"]["completion_time_s"]["median"], float(np.median(wins)), 1e-12)
          and sec["pi05"]["completion_time_s"]["bootstrap_ci"] is not None,
          "completion time: the median of successes, with a bootstrap interval")
    again = ae.analyze(schedule, records, scores, scorers)
    check(again["secondary"]["pi05"]["completion_time_s"]["bootstrap_ci"]
          == sec["pi05"]["completion_time_s"]["bootstrap_ci"], "the bootstrap interval is the same on a re-run")
    act_lat = [lat for t, ok, iv, ts, lat, wall in plan if t["policy"] == "act" and t["trial"] != missing]
    pooled = [x for lat in act_lat for x in lat]
    check(close(sec["act"]["latency_ms"]["p50"], float(np.percentile(pooled, 50)), 1e-9)
          and close(sec["act"]["latency_ms"]["p95"], float(np.percentile(pooled, 95)), 1e-9)
          and sec["act"]["latency_ms"]["requests"] == len(pooled), "latency p50/p95 over every request of a policy")
    check(close(sec["act"]["latency_ms"]["per_trial_median_p50"],
                float(np.median([np.percentile(lat, 50) for lat in act_lat])), 1e-9),
          "latency: the median of the per-trial p50s is reported beside the pooled one")
    check(sec["pi05"]["interventions"]["total"] == 1 and sec["pi05"]["interventions"]["trials_with_any"] == 1,
          "interventions per trial are counted")
    check(sec["act"]["wall_s"]["n"] == got["act"][1], "wall time per trial covers every trial run")

    out = study / "report"
    code, printed, err = run(ae.main, ["report", "--schedule", study / "schedule.json", "--log", log,
                                       "--scores", scores_path, "--out", out])
    md = (out / "report.md").read_text() if code == 0 else ""
    loaded = json.loads((out / "report.json").read_text()) if code == 0 else {}
    lines = [line for line in md.splitlines() if line.startswith("| T0")]
    check(code == 0 and loaded.get("format") == "bhl-lfd-eval-report" and "INCOMPLETE" in md,
          "report writes report.json and report.md, and says it is incomplete", err)
    check(len(lines) == 60 and sum("| failure |" in r for r in lines) == sum(n - x for x, n in got.values()),
          "the markdown lists every trial, failures included")
    try:
        json.dumps(report, allow_nan=False)
        check(True, "the report is plain JSON (no numpy types, no NaN)")
    except (TypeError, ValueError) as exc:
        check(False, "the report is plain JSON (no numpy types, no NaN)", str(exc))
    foreign = study / "foreign.jsonl"
    foreign.write_text(log.read_text() + json.dumps(dict(records[0], schedule_sha256="0" * 64)) + "\n")
    code, _, err = run(ae.main, ["report", "--schedule", study / "schedule.json", "--log", foreign,
                                 "--out", study / "foreign_report"])
    check(code == 2 and "another schedule" in err and not (study / "foreign_report").exists(),
          "report refuses a log with a line written under another schedule, as eval_log does", err)
    withdraw = {"kind": "withdraw", "policy": "act", "reason": "typed by hand", "trials": [], **base}
    alone = ae.analyze(schedule, [withdraw], {}, [])
    flagged = {d["kind"] for d in alone["deviations"] if d["policy"] == "act"}
    check({"withdrawal_without_evidence", "withdrawn_before_running"} <= flagged
          and alone["primary"]["per_policy"]["act"]["n"] == 20 and alone["primary"]["per_policy"]["act"]["n_run"] == 0,
          "a withdrawal with no S2 evidence, before any trial ran, is flagged; its 20 trials are imputed failures")
    code, printed, _ = run(ae.main, ["power", "--n", "20", "--m", "4"])
    check(code == 0 and printed.count("\n| 0.") == 8, "the power table prints one row per rate pair")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tmp", help="where to write test files (default: a new temporary directory)")
    parser.add_argument("--keep", action="store_true", help="keep the test files")
    args = parser.parse_args()
    if args.tmp:
        Path(args.tmp).mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix="test_eval_", dir=args.tmp))
    else:
        tmp = Path(tempfile.mkdtemp(prefix="test_eval_"))
    os.environ.pop("BHL_EVAL_OPERATOR", None)       # every command below names its operator itself
    test_statistics()
    test_schedule(tmp)
    test_jpeg()
    test_log(tmp)
    test_withdraw_and_s3(tmp)
    test_blind(tmp)
    test_sandboxes(tmp)
    test_analysis(tmp)
    failed = RESULTS.count(False)
    print(f"\n{len(RESULTS) - failed} passed, {failed} failed; files in {tmp}")
    if not failed and not args.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
