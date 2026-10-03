"""Check recorded LfD sessions against docs/LFD_RECORDING_FORMAT.md before they are converted or trained on.

For every episode of every session given, it checks the recording against the spec (sections 3-4):
  * session.json and episode.json: format, version, mode, outcome, joint names, claw limits, counts, and the types
    of trial (an int), arrangement, policy and operator (strings);
  * tap rows: t and seq increase, the row rate and its gaps, the zero never changes, src agrees with the mode (a
    teleop episode with a policy or guide row, or a guided one with a teleop or policy row, is refused), and nulls:
    a null q or qc in a row the steps read is an ERROR, nulls elsewhere a WARN;
  * frames: the index i increases, t_cap never steps back by more than 5 ms (equal times are normal: one coarse
    mtime tick), the frame rate and its gaps, every byte range is a whole JPEG (SOI ... EOI) of one even-width
    size, and a sample of them decodes (with cv2 or PIL, if either is installed);
  * the two streams overlap, and the converter's time grid (labels.frame_grid): how many steps have no frame within
    +-1/fps (missing; > 2 % rejects) and how many frames are reused or skipped (> 5 % of the steps rejects);
  * cam_mode_at_start is "still" (or a warning: no servos, or a moving camera), zero_changed is false, zero_check
    is there and what it decided, the claw limits are usable;
  * the measured-behind-commanded lag of each joint, from the cross-correlation of qc and q while it moves;
  * a preview of both labels (cmd and meas_future) with their statistics.

ERROR means the converter will refuse the episode (or, for a session.json fault, every episode of the session);
WARN lines are worth a look but convert. Episodes the converter skips by design (aborted, discarded) are notes. An
incomplete episode (outcome null: the recorder stopped without an end) is one WARN, 'incomplete: the converter
skips it', and anything else found in it is a WARN too. Episodes that differ in joint names, claw limits, frame size
or zero convention are a WARN: the converter refuses to mix them in one run, but each can be converted alone.
Where the converter's verdict depends on the label (the rows and frames its steps read), an ERROR means at least
one label (cmd for teleop episodes, meas_future with --lookahead) refuses it.

It prints a readable report, writes all of it as JSON with --out, and exits 1 if any check found an ERROR. Only
numpy is required.

  python tools/lfd/validate_recording.py /nfs/hpc/share/$USER/bhl-data/incoming/20261003_101500_robot
  python tools/lfd/validate_recording.py SESSION [SESSION ...] --out report.json --fps 30 --lookahead 0.10
  python tools/lfd/validate_recording.py /nfs/hpc/share/$USER/bhl-data/incoming      # every session inside
  python tools/lfd/validate_recording.py SESSION --max-missing 0.02 --max-reused-skipped 0.05  # the converter's limits
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels  # noqa: E402

ROW_KEYS = ("v", "seq", "t", "state", "motors", "src", "q", "qc", "tau", "tff", "lim", "zero", "held", "grip",
            "trig", "hand", "head", "tgt", "swap", "scale", "claw", "cam", "cam_mode", "intervention")
FRAME_KEYS = ("i", "off", "len", "t_cap", "t_seen", "mtime_ns")
OUTCOMES = ("success", "failure", "aborted")
ROW_GAP_S = 0.050
FRAME_GAP_S = 0.067
# the src values most rows of each mode carry (a policy episode's teleop rows are the operator's interventions)
MODE_SRC = {"teleop": ("teleop",), "guided": ("guide",), "policy": ("policy", "teleop")}
WARN_ONLY_NULLS = ("tau", "tff", "lim")          # vectors the converter never reads: a null there is a WARN
POSE_FIELDS = (("hand", 2, 16), ("head", None, 9), ("tgt", 2, 3))   # (key, entries or None, floats per entry)


class Findings:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.notes: list[str] = []

    def error(self, text: str) -> None:
        self.errors.append(text)

    def warn(self, text: str) -> None:
        self.warnings.append(text)

    def note(self, text: str) -> None:
        self.notes.append(text)

    def as_dict(self) -> dict:
        return {"errors": self.errors, "warnings": self.warnings, "notes": self.notes}


def _ms(seconds) -> str:
    return "-" if seconds is None or not np.isfinite(seconds) else f"{1000 * seconds:.0f} ms"


def _clean(value):
    """JSON-safe: numpy scalars to Python, NaN and infinities to null."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def interval_stats(t: np.ndarray, gap: float) -> dict:
    if len(t) < 2:
        return {"count": int(len(t)), "rate_hz": None}
    d = np.diff(t)
    span = float(t[-1] - t[0])
    return {"count": int(len(t)), "span_s": span, "rate_hz": (len(t) - 1) / span if span > 0 else None,
            "interval_median_s": float(np.median(d)), "interval_p90_s": float(np.percentile(d, 90)),
            "interval_max_s": float(d.max()), f"gaps_over_{round(gap * 1000)}ms": int((d > gap).sum())}


# ---------------------------------------------------------------------------------------------------- the lag


def estimate_lag(t: np.ndarray, qc: np.ndarray, q: np.ndarray, rate: float = 200.0, min_lag: float = -0.2,
                 max_lag: float = 0.5, min_speed: float = 0.05, min_moving: float = 0.1,
                 max_window: float = 120.0) -> dict:
    """Per joint: how long q (measured) trails qc (commanded), in seconds.

    Both are resampled at `rate` and differentiated; the lag is the peak of the normalised cross-correlation of the
    two velocities, using only the samples where the commanded velocity exceeds min_speed rad/s (the joint is being
    moved), refined between samples by a parabola. Velocities ignore the steady offset a sagging arm holds."""
    ok = np.isfinite(t) & np.all(np.isfinite(qc), axis=1) & np.all(np.isfinite(q), axis=1)
    t, qc, q = t[ok], qc[ok], q[ok]
    out: dict = {"rate_hz": rate, "search_s": [min_lag, max_lag], "min_speed_rad_s": min_speed, "joints": []}
    n_joints = qc.shape[1] if qc.ndim == 2 else 0
    if len(t) < 10 or t[-1] - t[0] < 1.0 or np.any(np.diff(t) <= 0):
        out["note"] = "too short, or times not increasing"
        out["joints"] = [{"lag_s": None, "corr": None, "moving_fraction": None} for _ in range(n_joints)]
        out["median_lag_s"] = None
        return out
    end = min(t[-1], t[0] + max_window)
    if end < t[-1]:
        out["note"] = f"first {max_window:.0f} s only"
    grid = np.arange(t[0], end, 1.0 / rate)
    n = len(grid)
    lags = np.arange(int(np.floor(min_lag * rate)), int(np.ceil(max_lag * rate)) + 1)
    for j in range(n_joints):
        dc = np.gradient(np.interp(grid, t, qc[:, j])) * rate
        dm = np.gradient(np.interp(grid, t, q[:, j])) * rate
        moving = np.abs(dc) > min_speed
        if moving.mean() < min_moving:
            out["joints"].append({"lag_s": None, "corr": None, "moving_fraction": float(moving.mean())})
            continue
        scores = np.full(len(lags), np.nan)
        for k, lag in enumerate(lags):
            if lag >= 0:
                x, y, w = dc[:n - lag], dm[lag:], moving[:n - lag]
            else:
                x, y, w = dc[-lag:], dm[:n + lag], moving[-lag:]
            x, y = x[w] - x[w].mean(), y[w] - y[w].mean()
            denom = math.sqrt(float((x * x).sum() * (y * y).sum()))
            if denom > 0:
                scores[k] = float((x * y).sum()) / denom
        if not np.any(np.isfinite(scores)):
            out["joints"].append({"lag_s": None, "corr": None, "moving_fraction": float(moving.mean())})
            continue
        k = int(np.nanargmax(scores))
        shift = 0.0
        if 0 < k < len(lags) - 1 and np.all(np.isfinite(scores[k - 1:k + 2])):
            a, b, c = scores[k - 1], scores[k], scores[k + 1]
            if a - 2 * b + c < 0:
                shift = 0.5 * (a - c) / (a - 2 * b + c)
        entry = {"lag_s": float((lags[k] + shift) / rate), "corr": float(scores[k]),
                 "moving_fraction": float(moving.mean())}
        if k in (0, len(lags) - 1):
            entry["note"] = "at the edge of the search range"
        out["joints"].append(entry)
    good = [j["lag_s"] for j in out["joints"] if j["lag_s"] is not None and (j["corr"] or 0) > 0.5]
    out["median_lag_s"] = float(np.median(good)) if good else None
    return out


# --------------------------------------------------------------------------------------------------- the checks


def check_meta(meta: dict, session: dict, ep_dir: Path, f: Findings) -> None:
    if meta.get("format") != labels.EPISODE_FORMAT or meta.get("version") != labels.FORMAT_VERSION:
        f.error(f"episode.json is format {meta.get('format')!r} version {meta.get('version')!r}, expected "
                f"{labels.EPISODE_FORMAT!r} version {labels.FORMAT_VERSION}")
    if meta.get("mode") not in labels.MODES:
        f.error(f"mode {meta.get('mode')!r} is not one of {', '.join(labels.MODES)}")
    outcome = meta.get("outcome")
    if outcome is None:
        f.warn("incomplete: the converter skips it (outcome null: the recorder stopped without ending the episode, "
               "or is still recording it)")
    elif outcome not in OUTCOMES:
        f.error(f"outcome {outcome!r} is not one of {', '.join(OUTCOMES)}")
    elif outcome == "aborted":
        f.note(f"outcome aborted (end_reason {meta.get('end_reason')!r}): the converter skips it")
    if meta.get("discarded") or ep_dir.parent.name == "_discarded":
        f.note("discarded: the converter skips it")
    task = meta.get("task")
    if not isinstance(task, str) or not task.strip():
        f.error(f"no task text ({task!r}): the converter refuses it")
    if meta.get("session_id") != session.get("session_id"):
        f.warn(f"episode.json session_id {meta.get('session_id')!r} differs from session.json's "
               f"{session.get('session_id')!r}")
    expected = f"ep_{meta.get('episode_index', -1):04d}" if isinstance(meta.get("episode_index"), int) else None
    if expected != ep_dir.name:
        f.warn(f"episode_index {meta.get('episode_index')!r} does not match the directory name {ep_dir.name}")
    trial = meta.get("trial")
    if trial is not None and (isinstance(trial, bool) or not isinstance(trial, int)):
        f.warn(f"trial {trial!r} is not an integer (the evaluation tools key on it); conversion.json copies it as is")
    for name in ("arrangement", "policy", "operator"):
        if meta.get(name) is not None and not isinstance(meta[name], str):
            f.warn(f"{name} {meta[name]!r} is not a string; conversion.json copies it as is")
    cam_mode = meta.get("cam_mode_at_start")
    if cam_mode is None:
        f.warn("cam_mode_at_start is null: no camera servos, so the camera angle is not recorded")
    elif cam_mode != "still":
        f.warn(f"cam_mode_at_start is {cam_mode!r}, not 'still': the viewpoint can move during and between "
               f"episodes")
    if meta.get("zero_changed") is not False:
        f.error(f"zero_changed is {meta.get('zero_changed')!r}: joint angles before and after the change disagree")
    verified, message = labels.zero_check_verdict(meta.get("zero_check"))
    if not verified:
        f.warn(f"{message}; the converter needs --allow-unverified-zero")
    elif message:
        f.warn(message)
    else:
        f.note(f"zero_check decision: {meta['zero_check']['decision']!r}")
    if not meta.get("zero_convention"):
        f.warn("zero_convention is missing")
    reason = labels.check_claw_limits(meta.get("claw_limits"))
    if reason:
        f.error(reason)


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def pose_nulls(records: list[dict], key: str, entries: int | None, width: int) -> tuple[int, int]:
    """(rows where a whole entry of key is null, rows with a null or non-number inside an entry). For hand and tgt
    a whole null entry is the spec's 'not tracked' / 'not held'; head is one entry."""
    whole = inside = 0
    for record in records:
        value = record.get(key)
        if entries is None:
            items = [value]
        else:
            items = value if isinstance(value, list) and len(value) == entries else [None]
        if any(item is None for item in items):
            whole += 1
        if any(isinstance(item, list) and (len(item) != width or not all(map(_is_number, item))) for item in items):
            inside += 1
    return whole, inside


def check_rows(records: list[dict], ep: labels.Episode, f: Findings) -> dict:
    rows, meta = ep.rows, ep.meta
    n = len(rows["t"])
    stats: dict = {"count": n}
    if n == 0:
        f.error("no tap rows")
        return stats
    missing = {}
    for record in records:
        for key in ROW_KEYS:
            if key not in record:
                missing[key] = missing.get(key, 0) + 1
    read = {k: c for k, c in missing.items() if k in labels.ROW_KEYS_READ}
    unread = {k: c for k, c in missing.items() if k not in labels.ROW_KEYS_READ}
    if read:
        f.error("rows lack fields the converter reads: " + ", ".join(f"{k} ({c} rows)" for k, c in sorted(read.items()))
                + "; the converter refuses it")
    if unread:
        f.warn("rows lack spec fields: " + ", ".join(f"{k} ({c} rows)" for k, c in sorted(unread.items())))
    versions = set(rows["v"])
    if versions != {labels.TAP_VERSION}:
        f.error(f"tap row versions {sorted(map(str, versions))}, expected {labels.TAP_VERSION}")
    t, seq = rows["t"], rows["seq"]
    if not np.all(np.isfinite(t)):
        f.error(f"{int((~np.isfinite(t)).sum())} rows have no valid t")
    elif np.any(np.diff(t) <= 0):
        f.error(f"t is not strictly increasing ({int((np.diff(t) <= 0).sum())} times)")
    if np.any(seq < 0):                               # the converter does not read seq
        f.warn(f"{int((seq < 0).sum())} rows have no integer seq")
    elif np.any(np.diff(seq) <= 0):
        f.warn(f"seq is not strictly increasing ({int((np.diff(seq) <= 0).sum())} times): did run_teleop restart?")
    else:
        skipped = int((np.diff(seq) - 1).sum())
        stats["seq_skipped"] = skipped
        if skipped > 0.01 * n:
            f.warn(f"{skipped} control cycles missing from the rows (seq gaps): dropped tap datagrams")
    stats.update(interval_stats(t[np.isfinite(t)], ROW_GAP_S))
    rate, gaps = stats.get("rate_hz"), stats.get("gaps_over_50ms", 0)
    if rate is not None and not 40.0 <= rate <= 56.0:
        f.warn(f"row rate {rate:.1f} Hz, expected about 48")
    if n > 1 and gaps > 0.10 * (n - 1):
        f.warn(f"{gaps} of {n - 1} row intervals exceed 50 ms (loop stalls)")
    if (stats.get("interval_max_s") or 0) > 0.5:
        f.warn(f"a {_ms(stats['interval_max_s'])} hole in the tap rows")
    counted = (meta.get("counts") or {})
    if meta.get("end") is not None:                   # a provisional episode.json has no counts yet
        if counted.get("rows") != n:
            f.warn(f"episode.json counts {counted.get('rows')} rows, rows.jsonl has {n}")
        if "row_gaps_over_50ms" in counted and counted["row_gaps_over_50ms"] != gaps:
            f.warn(f"episode.json counts {counted['row_gaps_over_50ms']} row gaps over 50 ms, the rows show {gaps}")
    # nulls (decision G): q and qc are judged in check_grid, on the rows the steps read; zero must be finite
    stats["null_rows"] = {key: int((~np.isfinite(rows[key])).any(axis=1).sum()) for key in labels.ROW_VECTORS}
    for key in WARN_ONLY_NULLS:
        if stats["null_rows"][key]:
            f.warn(f"{stats['null_rows'][key]} rows have a null or non-finite {key} (the converter does not read it)")
    if stats["null_rows"]["zero"]:
        f.error(f"{stats['null_rows']['zero']} rows have a null or non-finite zero: the converter refuses the episode")
    for key, entries, width in POSE_FIELDS:
        whole, inside = pose_nulls(records, key, entries, width)
        if inside:
            f.warn(f"{inside} rows have a null or non-finite value inside {key} (the converter does not read it)")
        if whole:
            why = "not available" if key == "head" else "an entry not tracked or not held"
            f.note(f"{key} is null in {whole} rows ({why})")
    zero = rows["zero"]
    finite = np.all(np.isfinite(zero), axis=1)
    if finite.any():
        z = zero[finite]
        changed = np.any(np.abs(z - z[0]) > labels.ZERO_TOL, axis=1)
        if changed.any():
            k = int(np.flatnonzero(finite)[np.argmax(changed)])
            f.error(f"the zero changes at row {k} (t = +{t[k] - t[0]:.2f} s)")
    if finite[0] and labels.zero_start_mismatch(meta, zero[0]):
        f.error("the rows' zero differs from episode.json zero_at_start: the converter refuses it")
    states = {s: rows["state"].count(s) for s in set(rows["state"])}
    sources = {s: rows["src"].count(s) for s in set(rows["src"])}
    stats.update({"states": states, "src": sources, "motors_off_rows": int((~rows["motors"]).sum()),
                  "intervention_rows": int(rows["intervention"].sum())})
    if any(s != "ARMED" for s in states):
        f.warn("rows outside ARMED: " + ", ".join(f"{s} {c}" for s, c in sorted(states.items(), key=str)))
    if stats["motors_off_rows"]:
        f.warn(f"motors off in {stats['motors_off_rows']} rows")
    mode = meta.get("mode")
    conflict = labels.mode_src_conflict(mode, rows["src"])
    if conflict:
        f.error(f"{conflict}: the converter refuses it")
    elif mode in MODE_SRC and sum(sources.get(src, 0) for src in MODE_SRC[mode]) < 0.9 * n:
        share = 100 * sum(sources.get(src, 0) for src in MODE_SRC[mode]) / n
        f.warn(f"mode {mode!r}, but only {share:.0f} % of rows have src {' or '.join(MODE_SRC[mode])} ("
               + ", ".join(f"{s} {c}" for s, c in sorted(sources.items(), key=str)) + ")")
    first_claw = []
    for side, name in enumerate(labels.CLAW_SIDES):
        commanded = np.isfinite(rows["claw"][:, side])
        first_claw.append(float(t[np.argmax(commanded)] - t[0]) if commanded.any() else None)
        if not commanded.any():
            f.warn(f"{name} claw never commanded in this episode (counts as open)")
    stats["claw_first_command_s"] = first_claw
    cam_modes = {m: rows["cam_mode"].count(m) for m in set(rows["cam_mode"])}
    stats["cam_modes"] = {str(k): v for k, v in cam_modes.items()}
    cam = rows["cam"]
    known = np.all(np.isfinite(cam), axis=1)
    stats["cam_unknown_rows"] = int((~known).sum())
    if known.any():
        moved = float(np.abs(cam[known] - cam[known][0]).max())
        stats["cam_moved_deg"] = moved
        if moved > 0.5:
            f.warn(f"the camera was commanded {moved:.1f} deg away from its start during the episode")
    if 0 < stats["cam_unknown_rows"] < n:
        f.warn(f"no camera angle in {stats['cam_unknown_rows']} of {n} rows (the servo link dropped): "
               f"cam_pan_tilt is 0 and cam_known false there")
    if set(cam_modes) - {"still"} and set(cam_modes) != {None}:
        f.warn("cam_mode was not 'still' in some rows: " + ", ".join(f"{m} {c}" for m, c in cam_modes.items()))
    return stats


def check_frames(ep: labels.Episode, f: Findings, decode_sample: int) -> tuple[dict, dict]:
    """The frame index and the frame bytes. Returns (stats, detail): detail holds, per frame, what is wrong with it,
    so classify_frames can tell frames the steps show (an ERROR: the converter refuses them) from the rest."""
    frames, meta = ep.frames, ep.meta
    n = len(frames["t_cap"])
    stats: dict = {"count": n, "order_ok": False}
    detail: dict = {"bad": {}, "dims": [None] * n, "undecodable": {}}
    if n == 0:
        f.error("no frames")
        return stats, detail
    for key in ("t_seen", "mtime_ns"):                # the converter reads neither
        invalid = frames[key] < 0 if key == "mtime_ns" else ~np.isfinite(frames[key])
        if invalid.any():
            f.warn(f"frames.jsonl: {key} is missing or invalid in {int(invalid.sum())} lines")
    if not np.any(np.diff(frames["i"]) <= 0) and (frames["i"][0] != 0 or np.any(np.diff(frames["i"]) != 1)):
        f.note("frame index i is not 0, 1, 2, ...")
    t_cap = frames["t_cap"]
    try:
        labels.order_frames(t_cap, frames["i"])
        stats["order_ok"] = True
    except labels.LabelError as exc:
        f.error(f"{exc.reason}: the converter refuses it")
    step = np.diff(t_cap)
    same, back = int((step == 0).sum()), int((step < 0).sum())
    stats.update({"same_t_cap": same, "back_in_time": back})
    if same or back:
        text = (f"{same} frames share the previous frame's t_cap and {back} sit up to "
                f"{1000 * labels.FRAME_BACKSTEP_S:.0f} ms before it (one coarse mtime tick); the converter orders "
                f"frames by (t_cap, i)")
        (f.warn if same + back > max(2, 0.01 * n) else f.note)(text)
    good_t = np.sort(t_cap[np.isfinite(t_cap)])
    stats.update(interval_stats(good_t, FRAME_GAP_S))
    rate, gaps = stats.get("rate_hz"), stats.get("gaps_over_67ms", 0)
    if rate is not None and not 25.0 <= rate <= 35.0:
        f.warn(f"frame rate {rate:.1f} Hz, expected about 30")
    if n > 1 and gaps:
        f.warn(f"{gaps} frame intervals exceed 67 ms (dropped camera frames)")
    counted = meta.get("counts") or {}
    if meta.get("end") is not None:                   # a provisional episode.json has no counts yet
        if counted.get("frames") != n:
            f.warn(f"episode.json counts {counted.get('frames')} frames, frames.jsonl has {n}")
        if "frame_gaps_over_67ms" in counted and counted["frame_gaps_over_67ms"] != gaps:
            f.warn(f"episode.json counts {counted['frame_gaps_over_67ms']} frame gaps over 67 ms, the frames show "
                   f"{gaps}")
    seen = frames["t_seen"] - t_cap
    if np.any(np.isfinite(seen)):
        seen = seen[np.isfinite(seen)]
        stats["seen_after_cap_s"] = {"median": float(np.median(seen)), "max": float(seen.max()),
                                     "min": float(seen.min())}
        if seen.min() < -0.002:
            f.warn(f"a frame was seen {_ms(-seen.min())} before its t_cap: the mtime-to-monotonic conversion is off")
        if np.median(seen) > 0.050:
            f.warn(f"frames reach the recorder {_ms(np.median(seen))} after capture (median)")
    # t_cap = mtime - (time.time() - time.monotonic()): that offset drifts slowly under NTP, but a jump between two
    # frames means the wall clock was stepped and t_cap is off by the jump from there on
    offset = t_cap - frames["mtime_ns"] * 1e-9
    if n > 1 and np.all(np.isfinite(offset)):
        stats["wall_offset_spread_s"] = float(offset.max() - offset.min())
        jump = float(np.abs(np.diff(offset)).max())
        if jump > 0.005:
            f.warn(f"t_cap - mtime jumps by {_ms(jump)} between two frames: the wall clock was stepped")
    # byte ranges: inside frames.mjpeg, back to back, each a whole JPEG with a frame header
    path = ep.path / "frames.mjpeg" if ep.path else None
    if path is None or not path.exists():
        return stats, detail
    size = path.stat().st_size
    off, length = frames["off"], frames["len"]
    outside = (off < 0) | (length <= 0) | (off + length > size)
    if not outside.any() and (off[0] != 0 or np.any(off[1:] != off[:-1] + length[:-1])
                              or off[-1] + length[-1] != size):
        f.warn("frame byte ranges are not back to back in frames.mjpeg")
    for k in range(n):
        if outside[k]:
            detail["bad"][k] = "its byte range falls outside frames.mjpeg"
            continue
        data = ep.frame_bytes(k)
        if not labels.is_jpeg(data):
            detail["bad"][k] = "not a whole JPEG (no SOI/EOI)"
            continue
        detail["dims"][k] = labels.jpeg_size(data)
        if detail["dims"][k] is None:
            detail["bad"][k] = "no JPEG frame header"
    stats["jpeg_ok"] = n - len(detail["bad"])
    sizes: dict = {}
    for dims in detail["dims"]:
        if dims is not None:
            sizes[dims] = sizes.get(dims, 0) + 1
    stats["sizes"] = {f"{d[0]}x{d[1]}": c for d, c in sizes.items()}
    if decode_sample > 0:
        good = [k for k in range(n) if k not in detail["bad"]]
        picks = sorted({good[int(round(x))] for x in np.linspace(0, len(good) - 1, min(decode_sample, len(good)))}
                       if good else set())
        decoded = 0
        for k in picks:
            try:
                image = labels.decode_jpeg(ep.frame_bytes(k))
            except ImportError:
                f.warn("neither cv2 nor PIL is installed: frames were checked by their markers only")
                break
            except Exception as exc:  # noqa: BLE001 - any decoder failure is the finding
                detail["undecodable"][k] = str(exc) or type(exc).__name__
                continue
            decoded += 1
            if image.shape[1::-1] != tuple(detail["dims"][k]):
                detail["undecodable"][k] = f"decoded {image.shape[1]}x{image.shape[0]}, header {detail['dims'][k]}"
        stats["decoded"] = decoded
    return stats, detail


def classify_frames(detail: dict, used: np.ndarray | None, stats: dict, f: Findings) -> None:
    """Frame defects are an ERROR when a step of either label shows the frame (the whole grid: cmd keeps every step,
    meas_future all but its last H seconds), so the converter refuses the episode with at least one label; a WARN
    otherwise. used is None when no grid could be laid: then every defect counts."""
    n = len(detail["dims"])
    shown = np.ones(n, dtype=bool) if used is None else np.isin(np.arange(n), used)

    def report(problems: dict, what: str) -> None:
        hit = sorted(k for k in problems if shown[k])
        rest = sorted(k for k in problems if not shown[k])
        if hit:
            f.error(f"{len(hit)} frames a step shows {what}, first at index {hit[0]} ({problems[hit[0]]}): the "
                    f"converter refuses it")
        if rest:
            f.warn(f"{len(rest)} frames no step shows {what}, first at index {rest[0]} ({problems[rest[0]]})")

    report(detail["bad"], "are unreadable")
    report(detail["undecodable"], "do not decode")
    dims_shown = {detail["dims"][k] for k in range(n) if shown[k] and detail["dims"][k] is not None}
    dims_all = {d for d in detail["dims"] if d is not None}
    if len(dims_shown) > 1:
        f.error(f"the frames a step shows differ in size: {sorted(map(str, dims_shown))}: the converter refuses it")
    elif len(dims_all) > 1:
        f.warn(f"frames differ in size ({sorted(map(str, dims_all))}), though not among those the steps show")
    if len(dims_shown) == 1:
        w, h = next(iter(dims_shown))
        stats["width"], stats["height"] = w, h
        if w % 2:
            f.error(f"frame width {w} is odd: the side-by-side frame cannot split into two eyes; the converter "
                    f"refuses it")


def check_grid(ep: labels.Episode, fps: float, lookahead: float, max_missing: float, max_reused_skipped: float,
               order_ok: bool, f: Findings) -> tuple[dict, np.ndarray | None]:
    """The converter's time grid over this episode (labels.frame_grid), its frame counts, and the nulls in the tap
    rows the steps read. Returns (report, the frames.jsonl positions the steps show, or None without a grid)."""
    row_t, frame_t = ep.rows["t"], ep.frame_times()
    if len(row_t) < 2 or len(frame_t) < 1 or not np.all(np.isfinite(row_t)) or not np.all(np.isfinite(frame_t)):
        return {}, None
    both = min(row_t[-1], frame_t.max()) - max(row_t[0], frame_t.min())
    span = max(row_t[-1], frame_t.max()) - min(row_t[0], frame_t.min())
    out: dict = {"overlap_s": float(both), "span_s": float(span)}
    if both <= 0:
        f.error("the tap rows and the frames do not overlap in time")
        return out, None
    if both < 0.9 * span:
        f.warn(f"the streams overlap for {both:.2f} s of {span:.2f} s")
    start, end = (ep.meta.get("start") or {}).get("t"), (ep.meta.get("end") or {}).get("t")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        early = min(row_t[0], ep.frames["t_cap"].min()) < start - 0.1
        late = max(row_t[-1], ep.frames["t_cap"].max()) > end + 0.1
        if early or late:
            f.warn("data lies outside episode.json's start.t .. end.t")
    if not order_ok or not np.all(np.diff(row_t) > 0):
        return out, None                              # already an ERROR above; no grid to lay
    try:
        grid = labels.frame_grid(frame_t, ep.frames["i"], row_t, fps)
    except labels.LabelError as exc:
        f.error(exc.reason)
        return out, None
    c = grid.counts
    out.update({"grid": c, "grid_steps": c["steps"], "missing_steps": c["missing"],
                "missing_fraction": c["missing"] / c["steps"], "reused_frames": c["reused"],
                "skipped_frames": c["skipped"], "frame_skew_s": labels.abs_stats(grid.skew)})
    reason = labels.grid_rejection(c, max_missing, max_reused_skipped)
    if reason:
        f.error(f"{reason}: the converter will reject this episode")
    elif c["missing"] or c["reused"] or c["skipped"]:
        f.warn(f"{c['missing']} of {c['steps']} grid steps have no frame within +-{1000 * c['window_s']:.1f} ms; "
               f"{c['reused']} frames reused, {c['skipped']} skipped")
    # nulls in q and qc (decision G): an ERROR in a row some label's steps read, a WARN anywhere else
    mode = ep.meta.get("mode")
    applicable = [label for label in labels.LABELS if mode in labels.MODES and (label != "cmd" or mode == "teleop")]
    read = {"q": np.zeros(0, dtype=np.int64), "qc": np.zeros(0, dtype=np.int64)}
    refused = []
    for label in applicable:
        q_rows, qc_rows = labels.rows_used(row_t, grid, label, lookahead)
        read["q"], read["qc"] = np.union1d(read["q"], q_rows), np.union1d(read["qc"], qc_rows)
        why = labels.null_rows_reason(ep.rows, q_rows, qc_rows)
        if why:
            refused.append(f"{label}: {why}")
    if refused:
        f.error("; ".join(refused) + " (the converter refuses it)")
    for key in ("q", "qc"):
        null = np.flatnonzero(~np.all(np.isfinite(ep.rows[key]), axis=1))
        unread = np.setdiff1d(null, read[key])
        if len(unread):
            f.warn(f"{len(unread)} rows have a null or non-finite {key} that no step reads (the converter ignores "
                   f"them)")
    return out, np.unique(grid.frame)


def label_preview(ep: labels.Episode, fps: float, lookahead: float, max_missing: float,
                  max_reused_skipped: float) -> dict:
    preview: dict = {}
    for label in labels.LABELS:
        try:
            steps = labels.build_steps(ep, fps, label, lookahead, allow_policy=True, max_missing=max_missing,
                                       max_reused_skipped=max_reused_skipped)
        except labels.LabelError as exc:
            preview[label] = {"refused": exc.reason}
            continue
        diff = steps.action - steps.state
        preview[label] = {
            "steps": int(len(steps.t)),
            "lookahead_s": steps.report["lookahead_s"],
            "action_minus_state_rms": {"joints_rad": float(np.sqrt(np.mean(diff[:, :-2] ** 2))),
                                       "claws": float(np.sqrt(np.mean(diff[:, -2:] ** 2)))},
            "action": {"names": steps.names, "mean": steps.action.mean(0), "std": steps.action.std(0),
                       "min": steps.action.min(0), "max": steps.action.max(0)},
            "warnings": steps.report["warnings"],
        }
    return preview


def validate_episode(ep_dir: Path, session: dict, fps: float, lookahead: float, decode_sample: int,
                     max_missing: float = labels.MAX_MISSING_FRACTION,
                     max_reused_skipped: float = labels.MAX_REUSED_SKIPPED_FRACTION) -> dict:
    f = Findings()
    result: dict = {"path": str(ep_dir), "episode": ep_dir.name}
    try:
        meta = json.loads((ep_dir / "episode.json").read_text())
    except (OSError, ValueError) as exc:
        f.error(f"episode.json unreadable: {exc}")
        result.update(f.as_dict())
        return result
    problems: list[str] = []
    records: dict[str, list] = {}
    for name in ("rows.jsonl", "frames.jsonl"):
        if (ep_dir / name).exists():
            records[name], bad = labels.read_jsonl(ep_dir / name)
            problems += bad
        else:
            records[name] = []
            problems.append(f"{name} is missing")
    if not (ep_dir / "frames.mjpeg").exists():
        problems.append("frames.mjpeg is missing")
    for problem in problems:
        f.error(problem)
    ep = labels.Episode.from_records(meta, session, records["rows.jsonl"], records["frames.jsonl"], path=ep_dir,
                                     problems=problems)
    result.update({"source": ep.source, "mode": meta.get("mode"), "outcome": meta.get("outcome"),
                   "task": meta.get("task"), "zero_convention": meta.get("zero_convention"),
                   "zero_check": meta.get("zero_check"), "claw_limits": meta.get("claw_limits"),
                   "discarded": bool(meta.get("discarded")) or ep_dir.parent.name == "_discarded"})
    with ep:
        check_meta(meta, session, ep_dir, f)
        result["rows"] = check_rows(records["rows.jsonl"], ep, f)
        del records
        result["frames"], detail = check_frames(ep, f, decode_sample)
        result["overlap"], used = check_grid(ep, fps, lookahead, max_missing, max_reused_skipped,
                                             result["frames"].get("order_ok", False), f)
        classify_frames(detail, used, result["frames"], f)
        result["lag"] = estimate_lag(ep.rows["t"], ep.rows["qc"], ep.rows["q"])
        result["labels"] = (label_preview(ep, fps, lookahead, max_missing, max_reused_skipped)
                            if not problems else {})
    if meta.get("outcome") is None:
        # the converter skips an incomplete episode whatever is in it, so nothing in it can fail a conversion
        f.warnings = [w for w in f.warnings if w.startswith("incomplete")] + [
            f"(incomplete episode) {line}" for line in f.errors] + [w for w in f.warnings
                                                                     if not w.startswith("incomplete")]
        f.errors = []
    result.update(f.as_dict())
    return result


def validate_session(session_dir: Path, fps: float, lookahead: float, decode_sample: int,
                     max_missing: float = labels.MAX_MISSING_FRACTION,
                     max_reused_skipped: float = labels.MAX_REUSED_SKIPPED_FRACTION) -> dict:
    f = Findings()
    session_dir = Path(session_dir)
    result: dict = {"path": str(session_dir)}
    try:
        session = labels.load_session(session_dir)
    except ValueError as exc:
        session = {}
        f.error(f"session.json unreadable: {exc}")
    if not session and not f.errors:
        f.error("session.json is missing")
    if session:
        problem = labels.session_problem(session)
        if problem:
            f.error(f"{problem}: the converter refuses every episode of this session")
        names = session.get("joint_names")
        if len(session.get("urdf_joint_names") or []) != len(names or []):
            f.warn("urdf_joint_names does not match joint_names in length")
        camera = session.get("camera") or {}
        if camera.get("rotated_180") is not True:
            f.warn(f"camera rotated_180 is {camera.get('rotated_180')!r}: the converter will not turn the frames")
        if camera.get("camera_latency_s") is None:
            f.note("camera_latency_s is null (not measured yet, roadmap A4): frames are timed at t_cap")
    result.update({"session_id": session.get("session_id"), "joint_names": session.get("joint_names"),
                   "camera": session.get("camera")})
    episodes = labels.list_episodes(session_dir)
    discarded = labels.list_episodes(session_dir, include_discarded=True)[len(episodes):]
    if not episodes:
        f.warn("no episodes")
    if discarded:
        f.note(f"{len(discarded)} discarded episodes in _discarded/ (not validated)")
    result["episodes"] = [validate_episode(ep, session, fps, lookahead, decode_sample, max_missing,
                                           max_reused_skipped) for ep in episodes]
    result.update(f.as_dict())
    return result


def mixing_preview(sessions: list[dict]) -> dict:
    """What the converter would refuse to put into one dataset, among episodes it would otherwise take."""
    seen: dict[str, dict] = {"joint_names": {}, "claw_limits": {}, "frame_size": {}, "zero_convention": {}}
    for s in sessions:
        if s["errors"]:                               # the converter takes nothing from a broken session.json
            continue
        for ep in s["episodes"]:
            if ep.get("outcome") in ("aborted", None) or ep.get("discarded") or ep.get("errors"):
                continue
            frames = ep.get("frames") or {}
            values = {"joint_names": s.get("joint_names"), "claw_limits": ep.get("claw_limits"),
                      "frame_size": [frames.get("width"), frames.get("height")],
                      "zero_convention": ep.get("zero_convention")}
            for key, value in values.items():
                seen[key].setdefault(json.dumps(value, sort_keys=True), []).append(ep["source"])
    return {key: groups for key, groups in seen.items() if len(groups) > 1}


def find_sessions(paths: list[Path]) -> tuple[list[Path], list[str]]:
    """Session directories among paths (a folder of sessions counts for each inside), and what was not found."""
    found, problems = [], []
    for path in map(Path, paths):
        if not path.is_dir():
            problems.append(f"{path} is not a directory")
        elif (path / "session.json").exists() or any(path.glob("ep_*")):
            found.append(path)
        else:
            inside = sorted(p for p in path.iterdir() if p.is_dir() and (p / "session.json").exists())
            found += inside
            if not inside:
                problems.append(f"{path} holds no session")
    return found, problems


def validate(paths: list[Path], fps: float = labels.FPS, lookahead: float = labels.LOOKAHEAD_S,
             decode_sample: int = 8, max_missing: float = labels.MAX_MISSING_FRACTION,
             max_reused_skipped: float = labels.MAX_REUSED_SKIPPED_FRACTION) -> dict:
    found, problems = find_sessions(paths)
    sessions = [validate_session(p, fps, lookahead, decode_sample, max_missing, max_reused_skipped) for p in found]
    mixing = mixing_preview(sessions)
    n_errors = len(problems) + sum(len(s["errors"]) + sum(len(e["errors"]) for e in s["episodes"])
                                   for s in sessions)
    n_warnings = len(mixing) + sum(len(s["warnings"]) + sum(len(e["warnings"]) for e in s["episodes"])
                                   for s in sessions)
    if not sessions and not problems:
        problems.append("no sessions found")
        n_errors += 1
    return {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "spec": "docs/LFD_RECORDING_FORMAT.md v1",
            "fps": fps, "lookahead_s": lookahead, "max_missing": max_missing,
            "max_reused_skipped": max_reused_skipped, "errors": problems, "sessions": sessions, "mixing": mixing,
            "n_errors": n_errors, "n_warnings": n_warnings, "ok": n_errors == 0}


# ------------------------------------------------------------------------------------------------- the printout


def print_report(report: dict) -> None:
    for line in report["errors"]:
        print(f"ERROR   {line}")
    for s in report["sessions"]:
        print(f"SESSION {s.get('session_id') or s['path']}  ({len(s['episodes'])} episodes)  {s['path']}")
        for line in s["errors"]:
            print(f"  ERROR   {line}")
        for line in s["warnings"]:
            print(f"  WARN    {line}")
        for line in s["notes"]:
            print(f"  note    {line}")
        for ep in s["episodes"]:
            rows, frames, overlap = ep.get("rows") or {}, ep.get("frames") or {}, ep.get("overlap") or {}
            span = overlap.get("span_s")
            print(f"  {ep['episode']}  {ep.get('mode')}  {ep.get('outcome')}  {ep.get('task')!r}"
                  + (f"  {span:.1f} s" if span else ""))
            if rows.get("rate_hz"):
                print(f"    rows    {rows['count']} at {rows['rate_hz']:.1f} Hz; intervals median "
                      f"{_ms(rows['interval_median_s'])}, max {_ms(rows['interval_max_s'])}; "
                      f"{rows.get('gaps_over_50ms', 0)} over 50 ms; seq skipped {rows.get('seq_skipped', '-')}")
            if frames.get("rate_hz"):
                print(f"    frames  {frames['count']} at {frames['rate_hz']:.1f} Hz; "
                      f"{frames.get('gaps_over_67ms', 0)} gaps over 67 ms; JPEG {frames.get('jpeg_ok', '-')}/"
                      f"{frames['count']} whole, size {', '.join(frames.get('sizes', {})) or '-'}, "
                      f"{frames.get('decoded', 0)} decoded")
            if "grid_steps" in overlap:
                skew, steps = overlap["frame_skew_s"], overlap["grid_steps"]
                print(f"    grid    {steps} steps: {overlap['missing_steps']} missing (no frame within +-1/fps, "
                      f"{100 * overlap['missing_fraction']:.1f} %), {overlap['reused_frames']} frames reused, "
                      f"{overlap['skipped_frames']} skipped "
                      f"({100 * (overlap['reused_frames'] + overlap['skipped_frames']) / steps:.1f} %); frame skew "
                      f"median {_ms(skew['median_abs'])}, max {_ms(skew['max_abs'])}; streams overlap "
                      f"{overlap['overlap_s']:.2f} s")
            lag = ep.get("lag") or {}
            joints = lag.get("joints") or []
            if joints:
                moving = [j for j in joints if j.get("lag_s") is not None]
                each = " ".join("-" if j.get("lag_s") is None else f"{1000 * j['lag_s']:.0f}" for j in joints)
                print(f"    lag     q behind qc: median {_ms(lag.get('median_lag_s'))} over {len(moving)}/"
                      f"{len(joints)} moving joints [ms: {each}]")
            for label, p in (ep.get("labels") or {}).items():
                if "refused" in p:
                    print(f"    {label:<12}refused: {p['refused']}")
                else:
                    h = f" (H = {p['lookahead_s']:.2f} s)" if p.get("lookahead_s") else ""
                    rms = p["action_minus_state_rms"]
                    print(f"    {label:<12}{p['steps']} steps{h}; action - state rms {rms['joints_rad']:.3f} rad "
                          f"(joints), {rms['claws']:.3f} (claws)")
                    for line in p["warnings"]:
                        print(f"                {line}")
            for line in ep["errors"]:
                print(f"    ERROR   {line}")
            for line in ep["warnings"]:
                print(f"    WARN    {line}")
            for line in ep["notes"]:
                print(f"    note    {line}")
    for key, groups in report["mixing"].items():
        print(f"WARN    these episodes differ in {key} and cannot be converted into one dataset:")
        for value, sources in groups.items():
            print(f"          {value}: {', '.join(sources)}")
    print(f"\n{report['n_errors']} error(s), {report['n_warnings']} warning(s)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sessions", nargs="+", type=Path, help="session directories, or folders of sessions")
    parser.add_argument("--out", type=Path, default=None, help="write the full report as JSON here")
    parser.add_argument("--fps", type=float, default=labels.FPS)
    parser.add_argument("--lookahead", type=float, default=labels.LOOKAHEAD_S, help="H for the meas_future preview")
    parser.add_argument("--decode-sample", type=int, default=8, help="frames per episode to decode (0: none)")
    parser.add_argument("--max-missing", type=float, default=labels.MAX_MISSING_FRACTION,
                        help="the converter's limit on steps without a frame within +-1/fps (default 0.02)")
    parser.add_argument("--max-reused-skipped", type=float, default=labels.MAX_REUSED_SKIPPED_FRACTION,
                        help="the converter's limit on reused + skipped frames, a fraction of the steps (default 0.05)")
    args = parser.parse_args()
    report = validate(args.sessions, args.fps, args.lookahead, args.decode_sample, args.max_missing,
                      args.max_reused_skipped)
    print_report(report)
    if args.out:
        args.out.write_text(json.dumps(_clean(report), indent=1) + "\n")
        print(f"wrote {args.out}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
