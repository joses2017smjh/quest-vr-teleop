"""Turn one recorded episode into time-aligned training steps (docs/LFD_RECORDING_FORMAT.md, sections 3-4).

The recorder (scripts/teleop/recorder.py) writes tap rows at about 48 Hz and camera frames at about 30 Hz, all on
the robot PC's monotonic clock. This module puts both on one grid and builds the spec's two action labels:

  cmd          action = [qc(t_k), claw(t_k)]           teleop episodes only
  meas_future  action = [q(t_k + H), claw(t_k + H)]    teleop or hand-guided; the last H seconds are trimmed

The state is always [q(t_k), claw(t_k)]: q is interpolated linearly between tap rows, qc and the claw pulses are
held from the last row at or before the time asked for, and each claw pulse becomes 0 (open) .. 1 (closed) with the
episode's own claw limits.

The time grid (frame_grid):
  * frames are put in time order by (t_cap - camera_latency_s, i). Equal or nearly equal times are normal: the
    recorder keys frames by (inode, mtime) and the kernel's coarse mtime clock can give two frames one t_cap. Only a
    step back of more than 5 ms (a clock step) or a frame index i out of order refuses the episode;
  * t_k = t_0 + k/fps from the first frame at or after the first tap row to min(last frame, last tap row);
  * each step shows the nearest frame. A step with no frame within +-1/fps is "missing"; a frame used by more than
    one step is "reused"; a frame inside the grid's span used by no step is "skipped". A camera slightly off 30 Hz
    reuses or skips the odd frame. More than 2 % missing steps, or reused + skipped frames above 5 % of the steps,
    rejects the episode (both limits are options). The counts are taken on the whole grid, before meas_future
    trims its tail, so the validator and both labels report the same numbers.

Nulls: a null q or qc in a tap row the steps read (the two rows around each interpolation time, the held row for
qc) refuses the episode; nulls elsewhere are ignored. A null zero anywhere refuses it too.

It is plain numpy on Python >= 3.10 and never imports LeRobot, so the validator, the converter and the tests share
it in both the rig's environment and the LeRobot one. Only the image helpers touch an image library, imported when
they are called: decode_jpeg() needs cv2 or PIL, and split_eyes() uses cv2 for speed when it is there.

Print one episode's label report:

  python tools/lfd/labels.py ~/bhl_recordings/20261003_101500_robot/ep_0000
  python tools/lfd/labels.py SESSION/ep_0001 --label cmd --fps 30
  python tools/lfd/labels.py SESSION/ep_0002 --label meas_future --lookahead 0.15 --allow-policy
  python tools/lfd/labels.py SESSION/ep_0003 --max-missing 0.02 --max-reused-skipped 0.05
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import sys
from pathlib import Path

import numpy as np

SESSION_FORMAT = "bhl-lfd-recording"
EPISODE_FORMAT = "bhl-lfd-episode"
FORMAT_VERSION = 1
TAP_VERSION = 1

MODES = ("teleop", "guided", "policy")
MODE_ID = {"teleop": 0, "guided": 1, "policy": 2}
LABELS = ("cmd", "meas_future")
FPS = 30
LOOKAHEAD_S = 0.10
MAX_MISSING_FRACTION = 0.02            # more missing grid steps than this rejects the episode (spec section 4)
MAX_REUSED_SKIPPED_FRACTION = 0.05     # reused + skipped frames above this fraction of the steps rejects it too
MISSING_WINDOW_PERIODS = 1.0           # a step is missing when no frame lies within this many 1/fps of it
FRAME_BACKSTEP_S = 0.005               # t_cap may step back this far (coarse mtime ticks); more is a clock step
MIN_STEPS = 2                          # fewer labelled steps than this is not an episode worth keeping
CLAW_SIDES = ("left", "right")         # the tap row's `claw` order: robot-left claw (D3), robot-right claw (D7)
CLAW_NAMES = ("left_claw", "right_claw")
ROW_VECTORS = ("q", "qc", "tau", "tff", "lim", "zero")
ROW_KEYS_READ = ("t", "src", "q", "qc", "zero", "claw", "cam", "intervention")   # what build_steps reads
N_ARM_JOINTS = 10
EPS = 1e-9                             # float slack for grid arithmetic on times near 1e5 s
ZERO_TOL = 1e-6                        # rad: a firmware zero that moves more than this was taken again
# Decision F: the row sources each episode mode must not contain. A policy episode may hold teleop rows (the
# operator's interventions); idle and cal rows are allowed in every mode.
SRC_FORBIDDEN = {"teleop": ("policy", "guide"), "guided": ("teleop", "policy"), "policy": ("guide",)}
ZERO_DECISIONS = ("holds", "rejected", "report")


class LabelError(ValueError):
    """An episode the spec's rules refuse to label. str(error) is the reason that goes into reports."""

    def __init__(self, reason: str, report: dict | None = None):
        super().__init__(reason)
        self.reason = reason
        self.report = report or {}


# ----------------------------------------------------------------------------------------------- reading files


def read_jsonl(path: Path) -> tuple[list[dict], list[str]]:
    """Every line of a JSON-lines file as a dict, plus one problem string for each line that is not one."""
    records: list[dict] = []
    problems: list[str] = []
    lines = Path(path).read_bytes().split(b"\n")
    if lines and not lines[-1].strip():
        lines.pop()                                      # the newline after the last record
    for n, raw in enumerate(lines, 1):
        if not raw.strip():
            problems.append(f"{Path(path).name} line {n} is empty")
            continue
        try:
            record = json.loads(raw)
        except ValueError as exc:
            torn = " (torn last line: the writer stopped mid-line)" if n == len(lines) else ""
            problems.append(f"{Path(path).name} line {n} is not JSON{torn}: {exc}")
            continue
        if not isinstance(record, dict):
            problems.append(f"{Path(path).name} line {n} is not a JSON object")
            continue
        records.append(record)
    return records, problems


def _number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _vectors(rows: list[dict], key: str, width: int) -> np.ndarray:
    """(N, width) float64 from a list-valued row field; NaN where it is null, missing or the wrong length."""
    try:                                                 # fast path: every row well formed
        out = np.array([row[key] for row in rows], dtype=float)
        if out.shape == (len(rows), width):
            return out
    except (KeyError, TypeError, ValueError):
        pass
    out = np.full((len(rows), width), np.nan)
    for i, row in enumerate(rows):
        value = row.get(key)
        if isinstance(value, (list, tuple)) and len(value) == width:
            out[i] = [_number(x) for x in value]
    return out


def _flags(rows: list[dict], key: str, width: int) -> np.ndarray:
    out = np.zeros((len(rows), width), dtype=bool)
    for i, row in enumerate(rows):
        value = row.get(key)
        if isinstance(value, (list, tuple)) and len(value) == width:
            out[i] = [bool(x) for x in value]
    return out


def rows_to_arrays(rows: list[dict], n_joints: int) -> dict:
    """The tap rows as numpy arrays. Joint vectors are (N, n_joints) in ARM_JOINTS order; nulls become NaN."""
    arrays = {
        "t": np.array([_number(row.get("t")) for row in rows], dtype=float),
        "seq": np.array([int(row["seq"]) if isinstance(row.get("seq"), int) else -1 for row in rows],
                        dtype=np.int64),
        "v": [row.get("v") for row in rows],
        "state": [row.get("state") for row in rows],
        "src": [row.get("src") for row in rows],
        "cam_mode": [row.get("cam_mode") for row in rows],
        "motors": np.array([bool(row.get("motors", False)) for row in rows], dtype=bool),
        "intervention": np.array([bool(row.get("intervention", False)) for row in rows], dtype=bool),
        "claw": _vectors(rows, "claw", 2),
        "cam": _vectors(rows, "cam", 2),
        "held": _flags(rows, "held", 2),
    }
    for key in ROW_VECTORS:
        arrays[key] = _vectors(rows, key, n_joints)
    arrays["missing_keys"] = {key: n for key in ROW_KEYS_READ if (n := sum(key not in row for row in rows))}
    return arrays


def frames_to_arrays(frames: list[dict]) -> dict:
    def ints(key):
        return np.array([int(f[key]) if isinstance(f.get(key), int) else -1 for f in frames], dtype=np.int64)

    return {
        "i": ints("i"),
        "off": ints("off"),
        "len": ints("len"),
        "t_cap": np.array([_number(f.get("t_cap")) for f in frames], dtype=float),
        "t_seen": np.array([_number(f.get("t_seen")) for f in frames], dtype=float),
        "mtime_ns": ints("mtime_ns"),
    }


@dataclasses.dataclass
class Episode:
    """One episode directory: its episode.json, its session's session.json, the tap rows and the frame index.

    Frame bytes stay in frames.mjpeg until frame_bytes() asks for one."""

    meta: dict
    session: dict
    rows: dict
    frames: dict
    problems: list[str] = dataclasses.field(default_factory=list)
    path: Path | None = None
    _fh: object = dataclasses.field(default=None, repr=False)

    @classmethod
    def from_records(cls, meta: dict, session: dict, rows: list[dict], frames: list[dict],
                     path: Path | None = None, problems: list[str] | None = None) -> Episode:
        names = session.get("joint_names") or []
        n_joints = len(names) or (len(rows[0].get("q") or []) if rows else 0) or 10
        return cls(meta=meta, session=session, rows=rows_to_arrays(rows, n_joints), frames=frames_to_arrays(frames),
                   problems=list(problems or []), path=path)

    # --- what the spec keeps in episode.json and session.json
    @property
    def mode(self) -> str | None:
        return self.meta.get("mode")

    @property
    def task(self) -> str:
        return self.meta.get("task") or ""

    @property
    def joint_names(self) -> list[str]:
        return list(self.session.get("joint_names") or [])

    @property
    def claw_limits(self) -> dict | None:
        return self.meta.get("claw_limits")

    @property
    def zero_convention(self) -> str | None:
        return self.meta.get("zero_convention")

    @property
    def camera(self) -> dict:
        return self.session.get("camera") or {}

    @property
    def camera_latency_s(self) -> float:
        latency = self.camera.get("camera_latency_s")
        return float(latency) if latency is not None else 0.0

    @property
    def rotated_180(self) -> bool:
        return bool(self.camera.get("rotated_180", True))  # the rig's camera hangs upside down

    @property
    def source(self) -> str:
        """session_id/ep_NNNN (or session_id/_discarded/ep_NNNN): where the episode came from."""
        sid = self.session.get("session_id") or self.meta.get("session_id") or "?"
        if self.path is None:
            return f"{sid}/ep_{int(self.meta.get('episode_index', 0)):04d}"
        name = self.path.name
        if self.path.parent.name == "_discarded":
            name = f"_discarded/{name}"
        return f"{sid}/{name}"

    def frame_times(self) -> np.ndarray:
        """When each frame was exposed, as well as we know: t_cap minus the session's camera latency, if measured."""
        return self.frames["t_cap"] - self.camera_latency_s

    def frame_bytes(self, k: int) -> bytes:
        """The JPEG bytes of the k-th frame of frames.jsonl, read from frames.mjpeg."""
        if self.path is None:
            raise FileNotFoundError("this episode was built in memory and has no frames.mjpeg")
        if self._fh is None:
            self._fh = open(self.path / "frames.mjpeg", "rb")
        self._fh.seek(int(self.frames["off"][k]))
        return self._fh.read(int(self.frames["len"][k]))

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> Episode:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def session_dir_of(episode_dir: Path) -> Path:
    episode_dir = Path(episode_dir)
    parent = episode_dir.parent
    return parent.parent if parent.name == "_discarded" else parent


def load_session(session_dir: Path) -> dict:
    """session.json as a dict; {} when it is missing (the episode then carries the problem)."""
    path = Path(session_dir) / "session.json"
    return json.loads(path.read_text()) if path.exists() else {}


def list_episodes(session_dir: Path, include_discarded: bool = False) -> list[Path]:
    """The ep_NNNN directories of a session, in order; with include_discarded, then those under _discarded/."""
    session_dir = Path(session_dir)
    found = sorted(p for p in session_dir.glob("ep_*") if p.is_dir())
    if include_discarded:
        found += sorted(p for p in (session_dir / "_discarded").glob("ep_*") if p.is_dir())
    return found


def load_episode(episode_dir: Path, session: dict | None = None) -> Episode:
    """Read an episode directory. Raises only when episode.json itself is unreadable; every other defect (a missing
    file, a line that is not JSON) is listed in Episode.problems for the validator and build_steps to judge."""
    episode_dir = Path(episode_dir)
    meta = json.loads((episode_dir / "episode.json").read_text())
    problems: list[str] = []
    if session is None:
        session = load_session(session_dir_of(episode_dir))
        if not session:
            problems.append("session.json is missing")
    rows: list[dict] = []
    frames: list[dict] = []
    for name, out in (("rows.jsonl", rows), ("frames.jsonl", frames)):
        path = episode_dir / name
        if path.exists():
            records, bad = read_jsonl(path)
            out.extend(records)
            problems.extend(bad)
        else:
            problems.append(f"{name} is missing")
    if not (episode_dir / "frames.mjpeg").exists():
        problems.append("frames.mjpeg is missing")
    return Episode.from_records(meta, session, rows, frames, path=episode_dir, problems=problems)


# ------------------------------------------------------------------------------------------------- the time grid


def order_frames(frame_t: np.ndarray, frame_i: np.ndarray) -> np.ndarray:
    """Positions in frames.jsonl, sorted by (time, i): the order the grid reads the frames in.

    Two frames may share a time, or the later one may sit a little earlier: the recorder converts each file mtime
    from the kernel's coarse clock with an offset sampled per frame. Raises LabelError for a frame time that is
    missing, a frame index i that does not increase down the file, or a time more than FRAME_BACKSTEP_S below an
    earlier frame's (the wall clock was stepped, so t_cap is wrong from there on)."""
    frame_t, frame_i = np.asarray(frame_t, dtype=float), np.asarray(frame_i)
    if len(frame_t) == 0:
        return np.zeros(0, dtype=np.int64)
    if not np.all(np.isfinite(frame_t)):
        raise LabelError(f"{int((~np.isfinite(frame_t)).sum())} frames have no time (t_cap)")
    if np.any(frame_i < 0) or np.any(np.diff(frame_i) <= 0):
        raise LabelError("the frame index i does not increase down frames.jsonl: frames out of order")
    back = np.maximum.accumulate(frame_t)[:-1] - frame_t[1:]
    if len(back) and back.max() > FRAME_BACKSTEP_S + EPS:
        k = int(np.argmax(back > FRAME_BACKSTEP_S + EPS)) + 1
        raise LabelError(f"frame times (t_cap) step back by {1000 * back[k - 1]:.1f} ms at frame {k}, more than "
                         f"{1000 * FRAME_BACKSTEP_S:.0f} ms: the clock was stepped")
    return np.lexsort((frame_i, frame_t)).astype(np.int64)


def time_grid(frame_t: np.ndarray, row_t: np.ndarray, fps: float) -> np.ndarray:
    """Spec section 4: t_k = t_0 + k/fps, from the first frame after both streams are present to the last frame.

    frame_t must be sorted (order_frames). "Both present" means t_0 is the first frame at or after the first tap
    row. The grid also stops at the last tap row, because q is interpolated, never extrapolated; the recorder ends
    both streams together, so this trims at most a step."""
    if len(frame_t) == 0 or len(row_t) == 0:
        raise LabelError("an empty stream: no tap rows or no frames")
    first = int(np.searchsorted(frame_t, row_t[0] - EPS, side="left"))
    if first >= len(frame_t):
        raise LabelError("the streams never overlap: no frame at or after the first tap row")
    t0 = float(frame_t[first])
    t_end = float(min(frame_t[-1], row_t[-1]))
    if t_end < t0:
        raise LabelError("the streams never overlap: the tap rows end before the first frame")
    n = int(np.floor((t_end - t0) * fps + EPS)) + 1
    return t0 + np.arange(n) / fps


def nearest_frames(grid_t: np.ndarray, frame_t: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each grid time: the nearest frame of the sorted frame_t, the skew t_k - t_frame, and whether no frame lies
    within +-1/fps (a missing step). A missing step still gets its nearest frame, so its image is stale by the
    skew shown."""
    right = np.clip(np.searchsorted(frame_t, grid_t, side="left"), 0, len(frame_t) - 1)
    left = np.clip(right - 1, 0, len(frame_t) - 1)
    use_left = np.abs(grid_t - frame_t[left]) <= np.abs(frame_t[right] - grid_t)   # ties go to the earlier frame
    index = np.where(use_left, left, right)
    skew = grid_t - frame_t[index]
    missing = np.abs(skew) > MISSING_WINDOW_PERIODS / fps + EPS
    return index, skew, missing


@dataclasses.dataclass
class FrameGrid:
    """The spec's time grid for one episode, with the frame each step shows."""

    t: np.ndarray            # (K,) float64 step times t_k, monotonic seconds
    frame: np.ndarray        # (K,) int64: the position in frames.jsonl of the frame each step shows
    skew: np.ndarray         # (K,) float64: t_k minus that frame's t_cap - camera_latency_s
    missing: np.ndarray      # (K,) bool: no frame within +-1/fps of t_k
    t_end: float             # min(last frame, last tap row)
    counts: dict             # steps, missing, reused, skipped (and how many frames shared or reversed a time)


def frame_grid(frame_t: np.ndarray, frame_i: np.ndarray, row_t: np.ndarray, fps: float) -> FrameGrid:
    """Order the frames, lay the grid over both streams and pick each step's frame (module docstring)."""
    order = order_frames(frame_t, frame_i)
    sorted_t = np.asarray(frame_t, dtype=float)[order]
    t = time_grid(sorted_t, row_t, fps)
    index, skew, missing = nearest_frames(t, sorted_t, fps)
    uses = np.bincount(index, minlength=len(sorted_t))
    inside = (sorted_t >= t[0] - EPS) & (sorted_t <= t[-1] + EPS)
    step = np.diff(np.asarray(frame_t, dtype=float))
    counts = {
        "steps": int(len(t)), "missing": int(missing.sum()), "reused": int((uses > 1).sum()),
        "skipped": int(((uses == 0) & inside).sum()),
        "frames_in_span": int(inside.sum()), "frames_same_time": int((step == 0).sum()),
        "frames_back_in_time": int((step < 0).sum()), "window_s": MISSING_WINDOW_PERIODS / fps,
    }
    return FrameGrid(t=t, frame=order[index], skew=skew, missing=missing,
                     t_end=float(min(sorted_t[-1], row_t[-1])), counts=counts)


def grid_rejection(counts: dict, max_missing: float = MAX_MISSING_FRACTION,
                   max_reused_skipped: float = MAX_REUSED_SKIPPED_FRACTION) -> str | None:
    """Why the grid's frame counts reject the episode, or None (decision A of the spec)."""
    k = max(int(counts["steps"]), 1)
    missing, resampled = int(counts["missing"]), int(counts["reused"]) + int(counts["skipped"])
    if missing / k > max_missing + EPS:
        return (f"{missing} of {k} steps ({100 * missing / k:.1f} %) have no frame within "
                f"+-{1000 * counts['window_s']:.1f} ms; the limit is {100 * max_missing:g} %")
    if resampled / k > max_reused_skipped + EPS:
        return (f"{counts['reused']} frames reused and {counts['skipped']} skipped over {k} steps "
                f"({100 * resampled / k:.1f} %): the camera ran off the grid's rate or stuttered; the limit is "
                f"{100 * max_reused_skipped:g} %")
    return None


def bracketing_rows(row_t: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """The tap rows linear interpolation reads at t_query: for each time the last row at or before it and the next
    one (both, even when the time falls exactly on a row). Sorted, unique."""
    n = len(row_t)
    if n == 0 or len(t_query) == 0:
        return np.zeros(0, dtype=np.int64)
    i = np.clip(np.searchsorted(row_t, t_query, side="right") - 1, 0, max(n - 2, 0))
    return np.unique(np.concatenate([i, np.minimum(i + 1, n - 1)])).astype(np.int64)


def held_rows(row_t: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """The tap rows a zero-order hold reads at t_query (hold_last): the last row at or before each time."""
    index = np.searchsorted(row_t, np.asarray(t_query) + EPS, side="right") - 1
    return np.unique(index[index >= 0]).astype(np.int64)


def rows_used(row_t: np.ndarray, grid: FrameGrid, label: str, lookahead: float) -> tuple[np.ndarray, np.ndarray]:
    """(rows whose q the label reads, rows whose qc it reads): q at t_k (state), and at t_k + H for meas_future;
    qc held at t_k (the action under cmd, and the q_cmd column under both labels)."""
    t = grid.t if label == "cmd" else grid.t[grid.t + lookahead <= grid.t_end + EPS]
    q_rows = bracketing_rows(row_t, t)
    if label == "meas_future":
        q_rows = np.union1d(q_rows, bracketing_rows(row_t, t + lookahead))
    return q_rows, held_rows(row_t, t)


def null_rows_reason(rows: dict, q_rows: np.ndarray, qc_rows: np.ndarray) -> str | None:
    """Why the rows the steps read refuse the episode (a null q or qc in one of them), or None."""
    for key, used in (("q", q_rows), ("qc", qc_rows)):
        values = rows[key][used]
        bad = used[~np.all(np.isfinite(values), axis=1)] if len(used) else used
        if len(bad):
            t = rows["t"]
            return (f"{len(bad)} tap rows the steps read have a null or non-finite {key} (first: row {int(bad[0])}, "
                    f"t = +{t[bad[0]] - t[0]:.2f} s)")
    return None


def mode_src_conflict(mode, sources: list) -> str | None:
    """Decision F: a teleop episode may hold no policy or guide rows, a guided one no teleop or policy rows. A policy
    episode may hold teleop rows (interventions) but no guide rows. Returns why the rows contradict the mode, or
    None."""
    found = {src: sources.count(src) for src in SRC_FORBIDDEN.get(mode, ()) if src in sources}
    if not found:
        return None
    return (f"a {mode} episode with rows whose src is " + ", ".join(f"{s} ({n} rows)" for s, n in found.items())
            + ": the rows contradict the episode's mode")


def zero_check_verdict(check) -> tuple[bool, str | None]:
    """(verified, message) for episode.json's zero_check (spec section 4).

    Not verified (the converter refuses it unless --allow-unverified-zero): null, not an object, an unknown decision,
    or "report" (inconclusive). Verified: "rejected" (the hanging zero is URDF zero), and "holds", which carries a
    warning when zero_fix_applied is not true: the data are consistent within their zero_convention, but the gravity
    model was off while recording."""
    if check is None:
        return False, "zero_check is null: the zero is unverified"
    if not isinstance(check, dict):
        return False, f"zero_check is a {type(check).__name__}, not an object: the zero is unverified"
    decision = check.get("decision")
    if decision not in ZERO_DECISIONS:
        return False, f"zero_check decision {decision!r} is not one of {', '.join(ZERO_DECISIONS)}: the zero is " \
                      f"unverified"
    if decision == "report":
        return False, "zero_check decided 'report' (inconclusive): the zero is unverified"
    if decision == "holds" and check.get("zero_fix_applied") is not True:
        return True, ("zero_check holds with zero_fix_applied false: the data are consistent within their "
                      "zero_convention, but the gravity model was off while recording")
    return True, None


def interpolate(t_query: np.ndarray, t: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Linear interpolation of each column of values (N, D) at t_query. Outside [t[0], t[-1]] it clamps."""
    out = np.empty((len(t_query), values.shape[1]))
    for d in range(values.shape[1]):
        out[:, d] = np.interp(t_query, t, values[:, d])
    return out


def hold_last(t_query: np.ndarray, t: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Zero-order hold: the value of the last row at or before each query time (NaN before the first row)."""
    index = np.searchsorted(t, t_query + EPS, side="right") - 1
    out = values[np.clip(index, 0, len(t) - 1)].astype(float)
    out[index < 0] = np.nan
    return out


def claw_fraction(pulses: np.ndarray, limits: dict) -> tuple[np.ndarray, np.ndarray]:
    """Claw pulses (K, 2), in microseconds and NaN where never commanded, to 0 (open) .. 1 (closed).

    Each side uses (pulse - open_us) / (closed_us - open_us), clipped to 0..1. A claw never commanded counts as
    open; the second array marks those steps so the caller can warn."""
    out = np.zeros(pulses.shape, dtype=float)
    for side, name in enumerate(CLAW_SIDES):
        lim = limits[name]
        open_us, closed_us = float(lim["open_us"]), float(lim["closed_us"])
        if closed_us == open_us:
            raise LabelError(f"claw_limits.{name}: open_us and closed_us are both {open_us:g}")
        out[:, side] = np.clip((pulses[:, side] - open_us) / (closed_us - open_us), 0.0, 1.0)
    never = ~np.isfinite(pulses)
    out[never] = 0.0
    return out, never


def session_problem(session: dict) -> str | None:
    """Why session.json makes every episode of its session unusable, or None."""
    if not session:
        return "session.json is missing"
    if session.get("format") != SESSION_FORMAT or session.get("version") != FORMAT_VERSION:
        return f"session.json is format {session.get('format')!r} version {session.get('version')!r}"
    names = session.get("joint_names")
    if (not isinstance(names, list) or len(names) != N_ARM_JOINTS or len(set(map(str, names))) != N_ARM_JOINTS
            or not all(isinstance(n, str) for n in names)):
        return f"session.json joint_names should be {N_ARM_JOINTS} distinct names, got {names!r}"
    layout = (session.get("camera") or {}).get("layout")
    if layout != "side_by_side":
        return f"camera layout {layout!r} is not side_by_side"
    return None


def zero_start_mismatch(meta: dict, zero_row0: np.ndarray) -> bool:
    """True when episode.json's zero_at_start disagrees with the first tap row's zero (when both are there)."""
    start = meta.get("zero_at_start")
    if not isinstance(start, list) or len(start) != len(zero_row0):
        return False
    start = np.array([_number(x) for x in start])
    return bool(np.any(~(np.abs(start - zero_row0) <= ZERO_TOL)))


def check_claw_limits(limits) -> str | None:
    """Why these claw limits are unusable, or None."""
    if not isinstance(limits, dict):
        return "claw_limits is missing"
    for name in CLAW_SIDES:
        side = limits.get(name)
        if not isinstance(side, dict) or not all(isinstance(side.get(k), (int, float)) for k in ("open_us",
                                                                                                "closed_us")):
            return f"claw_limits.{name} needs open_us and closed_us"
        if side["open_us"] == side["closed_us"]:
            return f"claw_limits.{name}: open_us equals closed_us"
    return None


# --------------------------------------------------------------------------------------------------- the labels


@dataclasses.dataclass
class Steps:
    """One episode on the grid, ready for a dataset. K steps; every array's first axis is K."""

    t: np.ndarray                # float64, monotonic seconds on the robot PC; stored as t_rel, with t0 beside it
    frame: np.ndarray            # int, which frame of frames.jsonl each step shows
    frame_skew: np.ndarray       # float64, t_k minus that frame's t_cap - camera_latency_s, s
    state: np.ndarray            # float32 (K, n+2): q(t_k), then the two claws
    action: np.ndarray           # float32 (K, n+2): the label
    q_cmd: np.ndarray            # float32 (K, n): qc(t_k), held
    cam_pan_tilt: np.ndarray     # float32 (K, 2): commanded camera degrees; 0.0 where unknown (see cam_known)
    cam_known: np.ndarray        # bool (K,): the held tap row had a camera angle (servos connected)
    intervention: np.ndarray     # bool (K,)
    mode_id: int
    names: list[str]             # the n+2 names of the state and action entries
    report: dict

    @property
    def t0(self) -> float:
        """The first step's monotonic time, s (conversion.json keeps it: float32 cannot hold the PC's uptime)."""
        return float(self.t[0])

    @property
    def t_rel(self) -> np.ndarray:
        """float32 (K,): seconds since the episode's first step, the dataset's t_rel column."""
        return (self.t - self.t[0]).astype(np.float32)


def abs_stats(x: np.ndarray) -> dict:
    if len(x) == 0:
        return {"median_abs": None, "p95_abs": None, "max_abs": None}
    a = np.abs(x)
    return {"median_abs": float(np.median(a)), "p95_abs": float(np.percentile(a, 95)), "max_abs": float(a.max())}


def check_mode(mode, label: str, allow_policy: bool = False) -> None:
    """The spec's mode rules. Raises LabelError for a combination it refuses."""
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}, not {label!r}")
    if mode not in MODES:
        raise LabelError(f"unknown mode {mode!r} (expected one of {', '.join(MODES)})")
    if mode == "policy" and not allow_policy:
        raise LabelError("policy episodes are excluded unless asked for")
    if label == "cmd" and mode != "teleop":
        raise LabelError(f"the cmd label is teleop-only, and this is a {mode} episode: its qc is not a "
                         f"demonstration; use meas_future")


def build_steps(episode: Episode, fps: float = FPS, label: str = "meas_future", lookahead: float = LOOKAHEAD_S,
                allow_policy: bool = False, max_missing: float = MAX_MISSING_FRACTION,
                max_reused_skipped: float = MAX_REUSED_SKIPPED_FRACTION) -> Steps:
    """Put one episode on the spec's time grid and build its state and its action label.

    Raises LabelError, with the reason and the report so far, when the spec refuses the episode: a mode rule, rows
    whose src contradicts the mode, more than max_missing of the grid without a frame or more than
    max_reused_skipped of it showing a reused or skipped frame, frames out of order, a null q or qc in a row the
    steps read, a null or changed zero, unreadable data, or too few steps."""
    check_mode(episode.mode, label, allow_policy)
    if fps <= 0:
        raise ValueError("fps must be positive")
    if label == "meas_future" and not lookahead > 0:
        raise ValueError("meas_future needs a positive lookahead (seconds)")
    rows, frames = episode.rows, episode.frames
    names = episode.joint_names
    n_joints = rows["q"].shape[1]
    report: dict = {
        "source": episode.source, "mode": episode.mode, "label": label, "fps": fps,
        "lookahead_s": lookahead if label == "meas_future" else None,
        "rows": int(len(rows["t"])), "frames": int(len(frames["t_cap"])),
        "camera_latency_s": episode.camera_latency_s, "warnings": [],
    }

    def refuse(reason: str):
        raise LabelError(reason, report)

    if episode.problems:
        refuse(f"unreadable recording: {'; '.join(episode.problems[:3])}"
               + (f" (+{len(episode.problems) - 3} more)" if len(episode.problems) > 3 else ""))
    if names and len(names) != n_joints:
        refuse(f"session.json names {len(names)} joints but the rows carry {n_joints}")
    if not names:
        refuse("session.json has no joint_names")
    reason = check_claw_limits(episode.claw_limits)
    if reason:
        refuse(reason)
    row_t, frame_t = rows["t"], episode.frame_times()
    if len(row_t) < 2 or len(frame_t) < 1:
        refuse(f"too little data: {len(row_t)} tap rows and {len(frame_t)} frames")
    versions = set(rows["v"])
    if versions != {TAP_VERSION}:
        refuse(f"tap rows of version {sorted(map(str, versions))}, not {TAP_VERSION}")
    if rows.get("missing_keys"):
        refuse("tap rows lack " + ", ".join(f"{k} ({n} rows)" for k, n in sorted(rows["missing_keys"].items())))
    if not np.all(np.isfinite(row_t)) or not np.all(np.diff(row_t) > 0):
        refuse("tap row times are not strictly increasing")
    reason = mode_src_conflict(episode.mode, rows["src"])
    if reason:
        refuse(reason)
    zero = rows["zero"]
    changed = ~np.all(np.isfinite(zero), axis=1) | np.any(np.abs(zero - zero[0]) > ZERO_TOL, axis=1)
    if changed.any():
        first = int(np.argmax(changed))
        what = "is null" if not np.all(np.isfinite(zero[first])) else "changed mid-episode"
        refuse(f"the zero {what} (row {first}, t = +{row_t[first] - row_t[0]:.2f} s)")
    if zero_start_mismatch(episode.meta, zero[0]):
        refuse("the rows' zero differs from episode.json zero_at_start")

    try:
        grid = frame_grid(frame_t, frames["i"], row_t, fps)
    except LabelError as exc:
        refuse(exc.reason)
    report.update({"t0": float(grid.t[0]), "t_end": grid.t_end, "duration_s": float(grid.t[-1] - grid.t[0]),
                   "grid": grid.counts, "grid_steps": grid.counts["steps"], "missing_steps": grid.counts["missing"],
                   "missing_fraction": grid.counts["missing"] / grid.counts["steps"],
                   "reused_frames": grid.counts["reused"], "skipped_frames": grid.counts["skipped"],
                   "reused_skipped_fraction": (grid.counts["reused"] + grid.counts["skipped"]) / grid.counts["steps"]})
    reason = grid_rejection(grid.counts, max_missing, max_reused_skipped)
    if reason:
        refuse(reason)

    keep = np.ones(len(grid.t), dtype=bool)
    if label == "meas_future":
        keep = grid.t + lookahead <= grid.t_end + EPS      # the last H seconds have no label
    report["kept_steps"] = int(keep.sum())
    report["trimmed_steps"] = int((~keep).sum())
    if keep.sum() < MIN_STEPS:
        refuse(f"too short: {int(keep.sum())} labelled steps")
    reason = null_rows_reason(rows, *rows_used(row_t, grid, label, lookahead))
    if reason:
        refuse(reason)
    t, index, skew, missing = grid.t[keep], grid.frame[keep], grid.skew[keep], grid.missing[keep]

    limits = episode.claw_limits
    claw_now, never_now = claw_fraction(hold_last(t, row_t, rows["claw"]), limits)
    q_now = interpolate(t, row_t, rows["q"])
    qc_now = hold_last(t, row_t, rows["qc"])
    state = np.concatenate([q_now, claw_now], axis=1)
    if label == "cmd":
        action = np.concatenate([qc_now, claw_now], axis=1)
    else:
        claw_next, _ = claw_fraction(hold_last(t + lookahead, row_t, rows["claw"]), limits)
        action = np.concatenate([interpolate(t + lookahead, row_t, rows["q"]), claw_next], axis=1)
    if not (np.all(np.isfinite(state)) and np.all(np.isfinite(action)) and np.all(np.isfinite(qc_now))):
        refuse("a state or action value came out non-finite")      # cannot happen after the null check above
    cam = hold_last(t, row_t, rows["cam"])
    cam_known = np.all(np.isfinite(cam), axis=1)
    intervention = hold_last(t, row_t, rows["intervention"].astype(float)[:, None])[:, 0] > 0.5

    for side, name in enumerate(CLAW_SIDES):
        n_never = int(never_now[:, side].sum())
        if n_never:
            report["warnings"].append(f"{name} claw never commanded during {n_never} of {len(t)} steps; "
                                      f"they count as open (0)")
    if missing.any():
        report["warnings"].append(f"{int(missing.sum())} steps have no frame within "
                                  f"+-{1000 * grid.counts['window_s']:.1f} ms and show the nearest one (skew up to "
                                  f"{1000 * np.abs(skew[missing]).max():.0f} ms)")
    if grid.counts["reused"] or grid.counts["skipped"]:
        report["warnings"].append(f"{grid.counts['reused']} frames reused and {grid.counts['skipped']} skipped over "
                                  f"{grid.counts['steps']} steps (a camera slightly off {fps:g} Hz does this)")
    if not cam_known.all():
        report["warnings"].append(f"no camera angle in {int((~cam_known).sum())} of {len(t)} steps (no servos, or "
                                  f"the servo link not ready): cam_pan_tilt is 0 and cam_known false there")
    report.update({
        "frame_skew_s": abs_stats(skew),
        "frames_used": int(len(np.unique(index))),
        "claw_uncommanded_steps": [int(never_now[:, s].sum()) for s in range(2)],
        "cam_unknown_steps": int((~cam_known).sum()),
        "interventions": int(intervention.sum()),
    })
    return Steps(
        t=t, frame=index.astype(np.int64), frame_skew=skew, state=state.astype(np.float32),
        action=action.astype(np.float32), q_cmd=qc_now.astype(np.float32),
        cam_pan_tilt=np.where(cam_known[:, None], cam, 0.0).astype(np.float32), cam_known=cam_known,
        intervention=intervention, mode_id=MODE_ID[episode.mode], names=list(names) + list(CLAW_NAMES),
        report=report)


# ------------------------------------------------------------------------------------------------------- images


def is_jpeg(data: bytes) -> bool:
    """Starts with the JPEG SOI marker and ends with EOI: the whole frame was copied."""
    return len(data) >= 4 and data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9"


def jpeg_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) from the JPEG's frame header, without decoding; None if there is none."""
    i, n = 2, len(data)
    while i + 4 <= n:
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        if marker == 0xFF:                               # fill byte
            i += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        length = int.from_bytes(data[i + 2:i + 4], "big")
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            if i + 9 > n:
                return None
            height = int.from_bytes(data[i + 5:i + 7], "big")
            width = int.from_bytes(data[i + 7:i + 9], "big")
            return width, height
        if marker in (0xD9, 0xDA):                       # end of image or start of scan before any frame header
            return None
        i += 2 + length
    return None


def decode_jpeg(data: bytes) -> np.ndarray:
    """RGB uint8 (H, W, 3). Uses cv2 when it is installed, else PIL; imported here so the module stays numpy-only."""
    try:
        import cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:
        bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("the JPEG did not decode")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)     # 10x faster than copying bgr[:, :, ::-1]
    from PIL import Image
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image.convert("RGB"))


def split_eyes(frame: np.ndarray, rotated_180: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(left eye, right eye) of a side-by-side frame. The rig's camera hangs upside down, so the whole frame is
    turned 180 degrees first, as depth_flow.grab and person_track do; the left half is then the left eye."""
    if rotated_180:
        try:
            import cv2
            frame = cv2.rotate(frame, cv2.ROTATE_180)   # the same pixels as frame[::-1, ::-1], 20x faster
        except ImportError:
            frame = frame[::-1, ::-1]
    half = frame.shape[1] // 2
    return np.ascontiguousarray(frame[:, :half]), np.ascontiguousarray(frame[:, half:])


# ---------------------------------------------------------------------------------------------------------- CLI


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("episode", type=Path, help="an ep_NNNN directory")
    parser.add_argument("--label", choices=LABELS, default="meas_future")
    parser.add_argument("--fps", type=float, default=FPS)
    parser.add_argument("--lookahead", type=float, default=LOOKAHEAD_S, help="H for meas_future, seconds")
    parser.add_argument("--allow-policy", action="store_true", help="label policy-mode episodes too")
    parser.add_argument("--max-missing", type=float, default=MAX_MISSING_FRACTION,
                        help="reject above this fraction of steps without a frame within +-1/fps (default 0.02)")
    parser.add_argument("--max-reused-skipped", type=float, default=MAX_REUSED_SKIPPED_FRACTION,
                        help="reject when reused + skipped frames exceed this fraction of the steps (default 0.05)")
    args = parser.parse_args()
    with load_episode(args.episode) as episode:
        try:
            steps = build_steps(episode, args.fps, args.label, args.lookahead, allow_policy=args.allow_policy,
                                max_missing=args.max_missing, max_reused_skipped=args.max_reused_skipped)
        except LabelError as exc:
            print(json.dumps({"refused": exc.reason, **exc.report}, indent=2))
            return 1
    print(json.dumps(steps.report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
