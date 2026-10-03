"""Label construction (tools/lfd/labels.py) against synthetic episodes whose answers are known.

The episodes come from synth_session.make_episode(): q lags qc by a known delay, the claws change at known times,
and frames can be dropped, duplicated or retimed where we choose. Everything is checked against those formulas or
against a brute-force recomputation, never against labels.py itself. Needs only numpy; when cv2 or PIL is there it
also writes sessions to disk and checks the round trip, the images, and that the validator and the converter's
episode selection (convert_to_lerobot.collect, which needs no LeRobot) agree on every refusal. Temporary files go
to $TMPDIR.

  /nfs/hpc/share/sanchej7/envs/bhl-rig-py310/bin/python tools/lfd/test_labels.py
  /nfs/hpc/share/sanchej7/envs/lerobot-py312-cpu/bin/python tools/lfd/test_labels.py
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import convert_to_lerobot as conv  # noqa: E402  (numpy-only until convert() runs)
import labels  # noqa: E402
import splits  # noqa: E402
import synth_session as synth  # noqa: E402
import validate_recording  # noqa: E402

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"PASS  {name}" + (f"  [{detail}]" if detail and len(detail) < 160 else ""))
    else:
        print(f"FAIL  {name}" + (f": {detail}" if detail else ""))
        failures.append(name)


def raises(fn, *args, **kwargs) -> str | None:
    """The LabelError's reason, or None if fn did not raise one."""
    try:
        fn(*args, **kwargs)
    except labels.LabelError as exc:
        return exc.reason
    return None


def last_row(row_t: np.ndarray, t: float) -> int:
    """Brute force: the index of the last row at or before t."""
    return max(i for i, rt in enumerate(row_t) if rt <= t + 1e-9)


def expected_claw(pulse, lim: dict) -> float:
    if pulse is None or not np.isfinite(pulse):
        return 0.0
    return float(np.clip((pulse - lim["open_us"]) / (lim["closed_us"] - lim["open_us"]), 0.0, 1.0))


def retimed(ep: labels.Episode, times: np.ndarray, keep: np.ndarray | None = None) -> labels.Episode:
    """A copy of ep whose frames sit exactly at times (float64, not rounded to the microsecond like a recording),
    keeping only the frames where keep is true."""
    frames = {k: np.array(v, copy=True) for k, v in ep.frames.items()}
    n = min(len(times), len(frames["t_cap"]))
    frames = {k: v[:n] for k, v in frames.items()}
    frames["t_cap"] = np.asarray(times[:n], dtype=float)
    if keep is not None:
        frames = {k: v[keep[:n]] for k, v in frames.items()}
    return dataclasses.replace(ep, frames=frames)


def copy_episode(ep: labels.Episode) -> labels.Episode:
    rows = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else dict(v) if isinstance(v, dict) else list(v))
            for k, v in ep.rows.items()}
    frames = {k: np.array(v, copy=True) for k, v in ep.frames.items()}
    return dataclasses.replace(ep, rows=rows, frames=frames)


def main() -> int:
    fps, H = 30, 0.10
    spec = synth.EpisodeSpec(mode="teleop", seconds=4.0, delay=0.08)
    ep = synth.make_episode(spec, index=0, seed=1)
    t_start = ep.meta["start"]["t"]                      # the synthetic trajectories run from the first row
    row_t, frame_t = ep.rows["t"], ep.frame_times()

    # --- the grid
    cmd = labels.build_steps(ep, fps, "cmd")
    first = frame_t[frame_t >= row_t[0]][0]
    t_end = min(frame_t[-1], row_t[-1])
    check("grid starts at the first frame at or after the first tap row", cmd.t[0] == first,
          f"{cmd.t[0]} vs {first}")
    check("grid steps are exactly 1/fps apart", np.allclose(np.diff(cmd.t), 1 / fps, rtol=0, atol=1e-9))
    check("grid stops at the last frame (and the last row)",
          cmd.t[-1] <= t_end + 1e-9 and cmd.t[-1] + 1 / fps > t_end, f"{cmd.t[-1]} vs end {t_end}")

    # --- t_rel and t0 (decision H): float32 seconds since the first step, and the first step's monotonic time
    k = np.arange(len(cmd.t))
    check("t0 is the first step's monotonic time (float64, near 1e5 s)",
          cmd.t0 == float(first) and isinstance(cmd.t0, float) and cmd.t0 > 1e5)
    check("t_rel is float32 seconds since the first step: k/fps",
          cmd.t_rel.dtype == np.float32 and cmd.t_rel[0] == 0.0
          and np.allclose(cmd.t_rel, k / fps, rtol=0, atol=2e-6)
          and np.allclose(cmd.t0 + cmd.t_rel.astype(np.float64), cmd.t, rtol=0, atol=1e-5),
          f"max |t_rel - k/fps| {np.abs(cmd.t_rel - k / fps).max():.1e}")
    mf0 = labels.build_steps(ep, fps, "meas_future", lookahead=H)
    check("meas_future keeps the same t0 (only the tail is trimmed)", mf0.t0 == cmd.t0)

    # --- cmd: qc held from the last row at or before t_k
    want = np.array([ep.rows["qc"][last_row(row_t, t)] for t in cmd.t], dtype=np.float32)
    check("cmd action = qc of the last row at or before t_k (zero-order hold)",
          np.array_equal(cmd.action[:, :10], want), f"max diff {np.abs(cmd.action[:, :10] - want).max()}")
    check("cmd q_cmd column = the same held qc", np.array_equal(cmd.q_cmd, want))
    check("cmd action claws = state claws (both claw(t_k))", np.array_equal(cmd.action[:, 10:], cmd.state[:, 10:]))
    q_true = synth.measured(cmd.t - t_start, spec.delay)
    err = np.abs(cmd.state[:, :10] - q_true).max()
    check("state q = q(t_k), linearly interpolated (within 1e-3 rad of the true trajectory)", err < 1e-3,
          f"max error {err:.2e}")

    # --- meas_future: q(t_k + H), trailing H trimmed
    mf = mf0
    err = np.abs(mf.action[:, :10] - synth.measured(mf.t + H - t_start, spec.delay)).max()
    check("meas_future action = q(t_k + H) within interpolation tolerance (1e-3 rad)", err < 1e-3,
          f"max error {err:.2e}")
    check("meas_future and cmd share the grid's start and spacing",
          mf.t[0] == cmd.t[0] and np.array_equal(mf.t, cmd.t[:len(mf.t)]))
    n_trim = int((cmd.t + H > t_end + 1e-9).sum())
    check("the trailing H seconds are trimmed, and only those",
          len(mf.t) == len(cmd.t) - n_trim and mf.report["trimmed_steps"] == n_trim
          and mf.t[-1] + H <= t_end + 1e-9 and n_trim in (3, 4),
          f"kept {len(mf.t)} of {len(cmd.t)}, trimmed {n_trim}")
    claw_next = np.array([[expected_claw(ep.rows["claw"][last_row(row_t, t + H)][s],
                                         ep.claw_limits[labels.CLAW_SIDES[s]]) for s in range(2)] for t in mf.t])
    check("meas_future action claws = claw(t_k + H)", np.allclose(mf.action[:, 10:], claw_next, atol=1e-7))
    same = synth.make_episode(synth.EpisodeSpec(seconds=4.0, delay=H), index=0, seed=1)
    a = labels.build_steps(same, fps, "meas_future", lookahead=H)
    err = np.abs(a.action[:, :10] - synth.command(a.t - same.meta["start"]["t"])).max()
    check("when H equals the arm's lag, meas_future recovers the command: q(t + H) = qc(t) within 1e-3 rad",
          err < 1e-3, f"max error {err:.2e}")
    check("the grid counts are the same for both labels (taken before the meas_future trim)",
          mf.report["grid"] == cmd.report["grid"], str(mf.report["grid"]))

    # --- claw normalisation
    claws = np.array([[expected_claw(ep.rows["claw"][last_row(row_t, t)][s], ep.claw_limits[labels.CLAW_SIDES[s]])
                       for s in range(2)] for t in cmd.t])
    check("claw = (pulse - open_us)/(closed_us - open_us), clipped, held", np.allclose(cmd.state[:, 10:], claws,
                                                                                       atol=1e-7))
    t_rel = cmd.t - t_start
    left, right = cmd.state[:, 10], cmd.state[:, 11]
    check("closed pulse 1640 us on the left claw gives 1.0", np.all(left[(t_rel > 1.6) & (t_rel < 2.9)] == 1.0))
    check("1320 us on the left claw gives 0.5", np.allclose(left[t_rel > 3.1], 0.5))
    check("1700 us (past closed) clips to 1.0 and 900 us (below open) clips to 0.0",
          np.all(right[(t_rel > 2.1) & (t_rel < 3.4)] == 1.0) and np.all(right[t_rel > 3.6] == 0.0))
    never = np.array([not np.isfinite(ep.rows["claw"][last_row(row_t, t)][0]) for t in cmd.t])
    check("a claw never commanded counts as 0 (open), with a warning",
          never.sum() > 10 and np.all(left[never] == 0.0)
          and cmd.report["claw_uncommanded_steps"][0] == int(never.sum())
          and any("never commanded" in w for w in cmd.report["warnings"]), str(cmd.report["warnings"]))
    check("claw_fraction refuses equal open and closed limits",
          raises(labels.claw_fraction, np.zeros((1, 2)), {"left": {"open_us": 1000, "closed_us": 1000},
                                                          "right": {"open_us": 1000, "closed_us": 1600}}) is not None)

    # --- frame skew and the nearest frame
    brute = np.array([int(np.argmin(np.abs(frame_t - t))) for t in cmd.t])
    check("each step uses the nearest frame", np.array_equal(cmd.frame, brute))
    check("frame_skew = t_k minus the used frame's t_cap - camera_latency_s (a clean 30 Hz camera: within "
          "+-0.5/fps plus jitter)",
          np.allclose(cmd.frame_skew, cmd.t - frame_t[cmd.frame], atol=1e-12)
          and np.abs(cmd.frame_skew).max() <= 0.5 / fps + 2 * spec.frame_jitter)

    # --- decision A, the missing rule: no frame within +-1/fps. Frames sit exactly on the grid (float64, so the
    # neighbours of a gap are exactly 1/fps away); each 3-frame gap leaves one step 2/fps from both neighbours.
    base = synth.make_episode(synth.EpisodeSpec(seconds=20.0, frame_jitter=0.0), seed=2)
    t0 = base.frames["t_cap"][0]
    exact = t0 + np.arange(len(base.frames["t_cap"])) / fps
    n_steps = labels.build_steps(retimed(base, exact), fps, "cmd").report["grid_steps"]
    allowed = int(np.floor(0.02 * n_steps + 1e-9))
    starts = 40 + 6 * np.arange(allowed + 1)                     # gaps 6 frames apart, through the middle

    def gapped(n_gaps: int) -> labels.Episode:
        keep = np.ones(len(exact), dtype=bool)
        for j in starts[:n_gaps]:
            keep[j:j + 3] = False
        return retimed(base, exact, keep)

    ok = labels.build_steps(gapped(allowed), fps, "cmd")
    reason = raises(labels.build_steps, gapped(allowed + 1), fps, "cmd")
    grid_ok = ok.report["grid"]
    check(f"{allowed} missing of {n_steps} steps (<= 2 %) is kept: each 3-frame gap is 1 missing step and 2 reused "
          f"frames, and the missing step shows the nearest frame",
          grid_ok["missing"] == allowed and grid_ok["reused"] == 2 * allowed and grid_ok["skipped"] == 0
          and np.abs(ok.frame_skew).max() > 1 / fps, str(grid_ok))
    check(f"{allowed + 1} missing of {n_steps} steps (> 2 %) rejects the episode, with the reason",
          reason is not None and "%" in reason and "no frame within" in reason, str(reason))
    single = np.ones(len(exact), dtype=bool)
    single[100] = False
    one = labels.build_steps(retimed(base, exact, single), fps, "cmd").report["grid"]
    check("one dropped frame is a reused frame, not a missing step (its neighbours are 1/fps away)",
          one["missing"] == 0 and one["reused"] == 1 and one["skipped"] == 0, str(one))
    lenient = labels.build_steps(gapped(allowed + 1), fps, "cmd", max_missing=0.05)
    check("--max-missing raises the limit", lenient.report["grid"]["missing"] == allowed + 1)

    # --- decision A, the reused + skipped rule (> 5 % of the steps), with cameras off 30 Hz
    long = synth.make_episode(synth.EpisodeSpec(seconds=60.0, frame_jitter=0.0), seed=3)
    t0 = long.frames["t_cap"][0]
    slow28 = retimed(long, t0 + np.arange(2000) / 28.0)
    reason = raises(labels.build_steps, slow28, fps, "cmd")
    check("a 28 Hz camera (no step missing, 7 % of frames reused) is rejected by the reused + skipped rule",
          reason is not None and "reused" in reason and "5 %" in reason, str(reason))
    fast32 = retimed(long, t0 + np.arange(2000) / 32.0)
    reason = raises(labels.build_steps, fast32, fps, "cmd")
    check("a 32 Hz camera (6 % of frames skipped) is rejected too", reason is not None and "skipped" in reason,
          str(reason))
    g29 = labels.build_steps(retimed(long, t0 + np.arange(2000) / 29.0), fps, "cmd").report["grid"]
    check("a 29 Hz camera (3 % reused) is kept, and the counts match the rate difference",
          g29["missing"] == 0 and abs(g29["reused"] - (g29["steps"] - g29["steps"] * 29 / 30)) <= 2
          and g29["skipped"] == 0, str(g29))
    check("--max-reused-skipped raises the limit",
          raises(labels.build_steps, slow28, fps, "cmd", max_reused_skipped=0.10) is None)

    # a real camera runs a hair off 30 Hz, and its file times jitter by a few ms
    for hz in (29.97, 30.03):
        counts = []
        for seed in range(3):
            e = synth.make_episode(synth.EpisodeSpec(seconds=60.0, frame_hz=hz, frame_jitter=0.002), seed=seed)
            try:
                counts.append(labels.build_steps(e, fps, "meas_future").report["grid"])
            except labels.LabelError as exc:
                counts.append({"refused": exc.reason})
        kept = all("refused" not in c for c in counts)
        share = [round(100 * (c["reused"] + c["skipped"]) / c["steps"], 1) for c in counts if "refused" not in c]
        check(f"a {hz} Hz camera with +-2 ms jitter for 60 s is accepted, with small reused/skipped counts",
              kept and all(c["missing"] == 0 for c in counts) and max(share) < 3.5,
              f"(reused, skipped) {[(c.get('reused'), c.get('skipped')) for c in counts]}, % of steps {share}")
    gap = synth.make_episode(synth.EpisodeSpec(seconds=60.0, frame_hz=29.97, frame_jitter=0.002,
                                               frame_gaps=((20.0, 1.5),)), seed=0)
    reason = raises(labels.build_steps, gap, fps, "meas_future")
    check("a true gap (1.5 s without frames in 60 s at 29.97 Hz) is still rejected as missing steps",
          reason is not None and "no frame within" in reason, str(reason))

    # --- decision A: frame order. Equal times (one coarse mtime tick) are fine; a step back > 5 ms is not
    sixty = synth.make_episode(synth.EpisodeSpec(seconds=60.0), seed=4)
    pair = copy_episode(sixty)
    pair.frames["t_cap"][100] = pair.frames["t_cap"][99]
    pair.frames["mtime_ns"][100] = pair.frames["mtime_ns"][99]
    s = labels.build_steps(pair, fps, "cmd")
    check("two frames with an equal t_cap (one mtime tick) are accepted",
          s.report["grid"]["frames_same_time"] == 1 and len(s.t) > 1700, str(s.report["grid"]))
    burst = synth.make_episode(synth.EpisodeSpec(seconds=60.0, same_mtime=(100, 900, 1500)), seed=4)
    s = labels.build_steps(burst, fps, "meas_future")
    check("synthetic same-mtime frames (3 in 60 s) are accepted",
          s.report["grid"]["frames_same_time"] == 3, str(s.report["grid"]))
    micro = copy_episode(sixty)
    micro.frames["t_cap"][100] = micro.frames["t_cap"][99] - 1e-6
    s = labels.build_steps(micro, fps, "cmd")
    times = micro.frame_times()
    brute = np.array([int(np.argmin(np.abs(times - t))) for t in s.t])
    check("a frame 1 us before the previous one is accepted, and each step still shows the nearest frame",
          np.array_equal(s.frame, brute) and s.report["grid"]["frames_back_in_time"] == 1)
    back = copy_episode(sixty)
    back.frames["t_cap"][100] = back.frames["t_cap"][99] - 0.004
    check("a step back of 4 ms is accepted", raises(labels.build_steps, back, fps, "cmd") is None)
    clock = copy_episode(sixty)
    clock.frames["t_cap"][100:] -= 0.045
    reason = raises(labels.build_steps, clock, fps, "cmd")
    check("a step back of more than 5 ms (the clock was stepped) is refused",
          reason is not None and "clock was stepped" in reason, str(reason))
    swapped = copy_episode(sixty)
    swapped.frames["i"][[100, 101]] = swapped.frames["i"][[101, 100]]
    reason = raises(labels.build_steps, swapped, fps, "cmd")
    check("a frame index out of order is refused", reason is not None and "out of order" in reason, str(reason))

    # --- mode rules, and decision F: rows whose src contradicts the mode
    guided = synth.make_episode(synth.EpisodeSpec(mode="guided", seconds=3.0), seed=3)
    reason = raises(labels.build_steps, guided, fps, "cmd")
    check("a guided episode refuses the cmd label", reason is not None and "teleop-only" in reason, str(reason))
    g = labels.build_steps(guided, fps, "meas_future", lookahead=H)
    err = np.abs(g.action[:, :10] - synth.measured(g.t + H - guided.meta["start"]["t"], 0.08)).max()
    check("a guided episode takes meas_future (where the arm went)", err < 1e-3, f"max error {err:.2e}")
    policy = synth.make_episode(synth.EpisodeSpec(mode="policy", seconds=3.0), seed=4)
    check("policy episodes are excluded unless asked for",
          raises(labels.build_steps, policy, fps, "meas_future") is not None
          and labels.build_steps(policy, fps, "meas_future", allow_policy=True).mode_id == 2)
    check("policy episodes refuse cmd even when asked for",
          raises(labels.build_steps, policy, fps, "cmd", allow_policy=True) is not None)
    check("mode_id is 0 teleop, 1 guided", cmd.mode_id == 0 and g.mode_id == 1)
    for mode, src, refused in (("teleop", "policy", True), ("teleop", "guide", True), ("guided", "teleop", True),
                               ("guided", "policy", True), ("teleop", "idle", False), ("guided", "idle", False)):
        e = copy_episode(synth.make_episode(synth.EpisodeSpec(mode=mode, seconds=3.0), seed=5))
        e.rows["src"][40] = src
        reason = raises(labels.build_steps, e, fps, "meas_future")
        check(f"a {mode} episode with one src {src!r} row is {'refused' if refused else 'accepted'}",
              (reason is not None and "contradict" in reason) if refused else reason is None, str(reason))
    e = copy_episode(synth.make_episode(synth.EpisodeSpec(mode="policy", seconds=3.0), seed=5))
    e.rows["src"][40] = "guide"
    reason = raises(labels.build_steps, e, fps, "meas_future", allow_policy=True)
    check("a policy episode with one src 'guide' row is refused", reason is not None and "contradict" in reason,
          str(reason))
    pol = synth.make_episode(synth.EpisodeSpec(mode="policy", seconds=4.0, interventions=((1.0, 2.0),)), seed=6)
    p = labels.build_steps(pol, fps, "meas_future", allow_policy=True)
    srcs = np.array(pol.rows["src"])
    want = np.array([pol.rows["intervention"][last_row(pol.rows["t"], t)] for t in p.t])
    check("a policy episode may hold teleop rows (the operator's interventions)",
          (srcs == "teleop").sum() > 30 and (srcs == "policy").sum() > 100 and p.mode_id == 2)
    check("intervention is bool, held from the last row at or before t_k",
          p.intervention.dtype == np.bool_ and np.array_equal(p.intervention, want) and 25 <= p.intervention.sum()
          <= 35, f"{int(p.intervention.sum())} steps")

    # --- decision G in labels: a null q or qc refuses only when a step reads that row; a null zero always
    def with_null(key, row, col=3, **kw):
        e = copy_episode(synth.make_episode(synth.EpisodeSpec(seconds=3.0, **kw), seed=7))
        e.rows[key][row, col] = np.nan
        return e

    probe = synth.make_episode(synth.EpisodeSpec(seconds=3.0), seed=7)
    grid_t = labels.build_steps(probe, fps, "cmd").t
    held_set = {last_row(probe.rows["t"], t) for t in grid_t}          # brute force: the rows qc is held from
    held = last_row(probe.rows["t"], grid_t[45])
    unheld = next((r for r in range(held, held + 20) if r not in held_set), None)   # 48 Hz rows, 30 Hz steps
    check("a null q in a row the steps read (around step 45) is refused",
          "null or non-finite q" in (raises(labels.build_steps, with_null("q", held), fps, "cmd") or ""))
    check("a null qc in the row a step holds is refused",
          "null or non-finite qc" in (raises(labels.build_steps, with_null("qc", held), fps, "meas_future") or ""))
    check("a null qc in a row no step holds (48 Hz rows, 30 Hz steps) is ignored",
          unheld is not None and raises(labels.build_steps, with_null("qc", unheld), fps, "cmd") is None,
          f"row {unheld}")
    late = synth.EpisodeSpec(seconds=3.0, frame_start=0.1)          # the first frame 0.1 s after the first row
    early = with_null("q", 0, frame_start=0.1)
    s = labels.build_steps(early, fps, "meas_future")
    check("a null q in a row no step reads (before the first frame) is ignored",
          np.all(np.isfinite(s.state)) and s.t[0] > early.rows["t"][1] + 0.05)
    check("a null tau is ignored by the labels", raises(labels.build_steps, with_null("tau", 60), fps, "cmd") is None)
    reason = raises(labels.build_steps, with_null("zero", 60), fps, "cmd")
    check("a null zero refuses the episode (the zero check cannot be skipped)",
          reason is not None and "zero is null" in reason, str(reason))
    zc = synth.make_episode(synth.EpisodeSpec(seconds=3.0, zero_change_at=1.5), seed=5)
    check("a zero change mid-episode is refused (and the synthetic recorder aborted it)",
          "zero changed" in (raises(labels.build_steps, zc, fps, "meas_future") or "")
          and zc.meta["outcome"] == "aborted" and zc.meta["zero_changed"] is True)
    bad = copy_episode(synth.make_episode(spec, seed=1))
    bad.rows["t"][30] = bad.rows["t"][29]
    check("tap rows whose times do not increase are refused",
          "increasing" in (raises(labels.build_steps, bad, fps, "cmd") or ""))

    # --- decision C: unknown camera angle is 0 with cam_known false
    blind = synth.make_episode(synth.EpisodeSpec(seconds=2.0, servos=False), seed=8)
    b = labels.build_steps(blind, fps, "cmd")
    check("without servos cam_pan_tilt is 0.0 and cam_known false; with them the commanded angle and true",
          np.all(b.cam_pan_tilt == 0.0) and not b.cam_known.any() and b.cam_known.dtype == np.bool_
          and np.all(cmd.cam_pan_tilt == np.float32(synth.CAM_ANGLE)) and cmd.cam_known.all())
    drop = copy_episode(ep)
    drop.rows["cam"][50:70] = np.nan
    d = labels.build_steps(drop, fps, "cmd")
    unknown = np.array([not np.isfinite(drop.rows["cam"][last_row(row_t, t)][0]) for t in d.t])
    check("a servo link that drops mid-episode: cam_known false exactly where the held row has no angle",
          unknown.any() and np.array_equal(~d.cam_known, unknown) and np.all(d.cam_pan_tilt[unknown] == 0.0)
          and np.all(np.isfinite(d.cam_pan_tilt)))

    # --- camera latency: frames are matched at t_cap - camera_latency_s
    lat = synth.make_episode(spec, seed=1, camera_latency_s=0.03)
    s = labels.build_steps(lat, fps, "cmd")
    shifted = lat.frames["t_cap"] - 0.03
    check("camera_latency_s is subtracted from t_cap before matching (frame_skew = t_k - (t_cap - latency))",
          np.allclose(s.frame_skew, s.t - shifted[s.frame], atol=1e-12) and s.t[0] in shifted)

    # --- the zero check (spec section 4): what the converter takes
    verdicts = {
        "null": labels.zero_check_verdict(None), "not an object": labels.zero_check_verdict("holds"),
        "report": labels.zero_check_verdict({"decision": "report"}),
        "unknown": labels.zero_check_verdict({"decision": "maybe"}),
        "holds, no fix": labels.zero_check_verdict({"decision": "holds", "zero_fix_applied": False}),
        "holds, fixed": labels.zero_check_verdict({"decision": "holds", "zero_fix_applied": True}),
        "rejected": labels.zero_check_verdict({"decision": "rejected", "zero_fix_applied": False}),
    }
    check("zero_check null, malformed, unknown or 'report' is unverified; 'holds' without the fix is kept with a "
          "warning; 'holds' with it and 'rejected' are kept",
          [v[0] for v in verdicts.values()] == [False, False, False, False, True, True, True]
          and "gravity model" in verdicts["holds, no fix"][1] and verdicts["holds, fixed"][1] is None
          and verdicts["rejected"][1] is None, str(verdicts))

    # --- the validator's lag estimate (cross-correlation of qc and q while moving) finds the known delay
    for delay in (0.08, 0.15):
        e = synth.make_episode(synth.EpisodeSpec(seconds=8.0, delay=delay), seed=6)
        lag = validate_recording.estimate_lag(e.rows["t"], e.rows["qc"], e.rows["q"])
        found = [j["lag_s"] for j in lag["joints"] if j["lag_s"] is not None]
        check(f"validator lag estimate finds a {1000 * delay:.0f} ms delay on all 10 joints (within 5 ms)",
              len(found) == 10 and max(abs(x - delay) for x in found) < 0.005,
              f"{[round(x, 4) for x in found]}")

    # --- splits (decision D): whole episodes, per mode, round(), at least one validation episode overall
    eps = [{"index": i, "mode": "teleop" if i < 12 else "guided", "source": f"s/ep_{i:04d}"} for i in range(17)]
    sp = splits.make_splits(eps, val_fraction=0.2, seed=0)
    train, val = set(sp["train"]), set(sp["val"])
    check("splits are disjoint and cover every episode once",
          not train & val and train | val == set(range(17)) and len(sp["train"]) + len(sp["val"]) == 17)
    check("round(0.2 * n) per mode goes to validation: 12 teleop -> 2, 5 guided -> 1",
          len(sp["by_mode"]["teleop"]["val"]) == 2 and len(sp["by_mode"]["guided"]["val"]) == 1)
    check("splits are reproducible with a seed and change with another",
          splits.make_splits(eps, 0.2, 0) == sp and splits.make_splits(eps, 0.2, 1)["val"] != sp["val"])
    for n in (2, 3, 4):
        small = [{"index": i, "mode": "teleop", "source": f"s/ep_{i}"} for i in range(n)]
        part = splits.make_splits(small, 0.2, 0)
        check(f"{n} episodes of one mode: one validation episode, the rest train",
              len(part["val"]) == 1 and len(part["train"]) == n - 1, json.dumps(part["by_mode"]))
    mixed = [{"index": 0, "mode": "teleop", "source": "s/ep_0"}, {"index": 1, "mode": "guided", "source": "s/ep_1"}]
    part = splits.make_splits(mixed, 0.2, 0)
    check("1 teleop + 1 guided: one goes to validation (a mode gives up its only episode)",
          len(part["val"]) == 1 and len(part["train"]) == 1, json.dumps(part["by_mode"]))
    three = mixed + [{"index": 2, "mode": "teleop", "source": "s/ep_2"}]
    part = splits.make_splits(three, 0.2, 0)
    check("2 teleop + 1 guided: the validation episode comes from the mode that keeps a training one",
          len(part["val"]) == 1 and part["by_mode"]["guided"]["val"] == [] and part["by_mode"]["teleop"]["train"],
          json.dumps(part["by_mode"]))
    nine = [{"index": i, "mode": "teleop", "source": f"s/ep_{i}"} for i in range(9)]
    check("round, not floor: 9 episodes at 0.2 give 2 validation episodes (floor gave 1)",
          len(splits.make_splits(nine, 0.2, 0)["val"]) == 2)
    check("one episode, or val_fraction 0: no validation episode",
          splits.make_splits(mixed[:1], 0.2, 0)["val"] == [] and splits.make_splits(three, 0.0, 0)["val"] == [])
    check("the rule is written into splits.json", "round" in sp["rule"] and ">= 2 episodes" in sp["rule"])

    # --- the synthetic joint names are the rig's
    source = (REPO / "scripts/teleop/arm_power.py").read_text()
    found = re.findall(r'ArmJoint\("(\w+)", "[^"]*", "can\d", \d+, "(\w+)"', source)
    check("synthetic joint names = arm_power.ARM_JOINTS keys and URDF names",
          [k for k, _ in found] == list(synth.DEFAULT_JOINT_NAMES)
          and [u for _, u in found] == list(synth.DEFAULT_URDF_JOINT_NAMES), str(found))

    # --- on disk: the same labels after a write and a read, valid JPEGs, and the upside-down eyes
    has_codec = importlib.util.find_spec("cv2") or importlib.util.find_spec("PIL")
    if not has_codec:
        print("SKIP  file round trip and validator/converter agreement: neither cv2 nor PIL is installed")
    else:
        with tempfile.TemporaryDirectory(prefix="lfd_test_labels_") as tmp:
            disk_checks(Path(tmp), spec, fps, H, late)

    print()
    for name in failures:
        print("FAIL  " + name)
    print("labels are sound" if not failures else f"{len(failures)} problem(s)")
    return 1 if failures else 0


def mutate(ep_dir: Path, key: str, row: int, value) -> None:
    """Damage one recorded episode on disk the way a test case asks."""
    lines = (ep_dir / "rows.jsonl").read_text().splitlines()
    meta = json.loads((ep_dir / "episode.json").read_text())
    frames = [json.loads(line) for line in (ep_dir / "frames.jsonl").read_text().splitlines()]
    if key == "provisional":
        meta.update(outcome=None, end=None, end_reason=None, counts={"rows": 0, "frames": 0})
        lines.append('{"v": 1, "seq": 77')
    elif key == "zero_at_start":
        meta["zero_at_start"][0] += 0.1
    elif key in ("tear", "tear_early"):
        if key == "tear_early":                       # move the first frame before the first tap row
            frames[0]["t_cap"] = round(json.loads(lines[0])["t"] - 0.040, 6)
            frames[0]["mtime_ns"] -= 52_000_000
        with open(ep_dir / "frames.mjpeg", "r+b") as fh:
            fh.seek(frames[row]["off"] + frames[row]["len"] - 2)
            fh.write(b"\x00\x00")
    else:
        record = json.loads(lines[row])
        if key == "src":
            record["src"] = value
        elif key == "pose":
            record["hand"][0][3] = None
            record["head"][4] = None
            record["tgt"][1][2] = None
        elif key == "tff":
            record["tff"][value] = None
            record["lim"][value] = None
        elif key == "drop":
            del record[value]
        elif key == "seq":
            record["seq"] = json.loads(lines[row - 1])["seq"]
        elif key == "v":
            record["v"] = value
        else:
            record[key][value] = None
        lines[row] = json.dumps(record)
    (ep_dir / "rows.jsonl").write_text("\n".join(lines) + "\n")
    (ep_dir / "episode.json").write_text(json.dumps(meta))
    (ep_dir / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in frames))


def disk_checks(tmp: Path, spec: synth.EpisodeSpec, fps: int, H: float, late: synth.EpisodeSpec) -> None:
    session = synth.write_session(tmp / "a", [spec, synth.EpisodeSpec(mode="guided", seconds=2.0)], seed=1)
    episodes = labels.list_episodes(session)
    with labels.load_episode(episodes[0]) as disk:
        d = labels.build_steps(disk, fps, "cmd")
        mem = synth.make_episode(spec, index=0, seed=1)
        m = labels.build_steps(mem, fps, "cmd")
        check("an episode read from disk labels exactly like the in-memory one",
              np.array_equal(d.state, m.state) and np.array_equal(d.action, m.action) and np.array_equal(d.t, m.t))
        blobs = [disk.frame_bytes(i) for i in range(len(disk.frames["t_cap"]))]
        check("every frame's byte range is a whole JPEG (SOI ... EOI)", all(labels.is_jpeg(b) for b in blobs))
        check("the JPEG header gives the 1280x480 side-by-side size",
              {labels.jpeg_size(b) for b in blobs} == {(1280, 480)})
        k = len(blobs) // 2
        lefty, righty = labels.split_eyes(labels.decode_jpeg(blobs[k]), rotated_180=True)
        box = synth.marker_boxes(640, 480)
        x0, y0, x1, y1 = box["left_marker"]
        lc = lefty[y0 + 5:y1 - 5, x0 + 5:x1 - 5].reshape(-1, 3).mean(0)
        x0, y0, x1, y1 = box["right_marker"]
        rc = righty[y0 + 5:y1 - 5, x0 + 5:x1 - 5].reshape(-1, 3).mean(0)
        check("after the 180-degree turn, the left eye shows red at its upper left and the right eye blue "
              "at its lower right", lc[0] > 200 and lc[2] < 60 and rc[2] > 200 and rc[0] < 60,
              f"left {lc.round()}, right {rc.round()}")
        check("the frame's drawn index reads back from the left eye", synth.read_code(lefty) == k,
              f"{synth.read_code(lefty)} vs {k}")
        unturned_left, _ = labels.split_eyes(labels.decode_jpeg(blobs[k]), rotated_180=False)
        x0, y0, x1, y1 = box["left_marker"]
        check("without the turn the left half shows no red marker there (the test can tell)",
              unturned_left[y0 + 5:y1 - 5, x0 + 5:x1 - 5].reshape(-1, 3).mean(0)[0] < 150)
    report = validate_recording.validate([session], fps=fps, lookahead=H, decode_sample=4)
    ep_reports = report["sessions"][0]["episodes"]
    check("the validator passes a clean synthetic session (no errors)", report["n_errors"] == 0,
          str([e["errors"] for e in ep_reports]))
    lags = [j["lag_s"] for j in ep_reports[0]["lag"]["joints"] if j["lag_s"] is not None]
    check("the validator reports the teleop episode's 80 ms lag", len(lags) == 10
          and max(abs(x - 0.08) for x in lags) < 0.006, str(lags))
    grid = ep_reports[0]["overlap"]["grid"]
    check("the validator reports the grid's missing, reused and skipped counts, equal to labels.py's",
          {k: grid[k] for k in ("steps", "missing", "reused", "skipped")}
          == {k: d.report["grid"][k] for k in ("steps", "missing", "reused", "skipped")}, str(grid))

    # --- decisions F and G on disk: the validator flags an ERROR exactly when the converter refuses the episode
    # for a defect; deliberate exclusions (incomplete, aborted, discarded) are never an ERROR
    cases = [  # (name, spec, mutation, converter refuses, validator ERROR)
        ("clean teleop", synth.EpisodeSpec(seconds=2.0), None, False, False),
        ("null q in a row the steps read", synth.EpisodeSpec(seconds=2.0), ("q", 40, 3), True, True),
        ("null qc in a row the steps read", synth.EpisodeSpec(seconds=2.0), ("qc", 40, 0), True, True),
        ("null q in a row no step reads", late, ("q", 0, 3), False, False),
        ("null tau", synth.EpisodeSpec(seconds=2.0), ("tau", 40, 2), False, False),
        ("null tff and lim", synth.EpisodeSpec(seconds=2.0), ("tff", 41, 1), False, False),
        ("null inside hand, head and tgt", synth.EpisodeSpec(seconds=2.0), ("pose", 42, 0), False, False),
        ("null zero", synth.EpisodeSpec(seconds=2.0), ("zero", 40, 1), True, True),
        ("teleop with a policy row", synth.EpisodeSpec(seconds=2.0), ("src", 40, "policy"), True, True),
        ("guided with a teleop row", synth.EpisodeSpec(mode="guided", seconds=2.0), ("src", 40, "teleop"), True,
         True),
        ("policy with teleop rows", synth.EpisodeSpec(mode="policy", seconds=2.0, interventions=((0.5, 1.0),)),
         None, False, False),
        ("incomplete (outcome null), torn last line", synth.EpisodeSpec(seconds=2.0), ("provisional", 0, 0), True,
         False),
        ("a torn JPEG a step shows", synth.EpisodeSpec(seconds=2.0), ("tear", 20, 0), True, True),
        ("a torn JPEG no step shows (before the first row)", synth.EpisodeSpec(seconds=2.0), ("tear_early", 0, 0),
         False, False),
        ("rows without claw (a field the converter reads)", synth.EpisodeSpec(seconds=2.0), ("drop", 40, "claw"),
         True, True),
        ("rows without trig (a field it does not read)", synth.EpisodeSpec(seconds=2.0), ("drop", 40, "trig"), False,
         False),
        ("a repeated seq", synth.EpisodeSpec(seconds=2.0), ("seq", 40, 0), False, False),
        ("a row of tap version 2", synth.EpisodeSpec(seconds=2.0), ("v", 40, 2), True, True),
        ("zero_at_start differs from the rows' zero", synth.EpisodeSpec(seconds=2.0), ("zero_at_start", 0, 0), True,
         True),
    ]
    session = synth.write_session(tmp / "b", [c[1] for c in cases], seed=2)
    for (name, _, mutation, _, _), ep_dir in zip(cases, labels.list_episodes(session)):
        if mutation is not None:
            mutate(ep_dir, *mutation)
    report = validate_recording.validate([session], fps=fps, lookahead=H, decode_sample=0)
    by_name = dict(zip([c[0] for c in cases], report["sessions"][0]["episodes"]))
    accepted, rejected, _ = conv.collect([session], "meas_future", fps, H, ("teleop", "guided", "policy"), False,
                                         log=lambda *a: None)
    taken = {a["source"].split("/")[-1] for a in accepted}
    reasons = {r["source"].split("/")[-1]: r["reason"] for r in rejected}
    for (name, _, _, refuses, errors), ep_dir in zip(cases, labels.list_episodes(session)):
        found = by_name[name]
        did_refuse = ep_dir.name not in taken
        has_error = bool(found["errors"])
        check(f"agreement, {name}: converter {'refuses' if refuses else 'takes'} it, validator "
              f"{'ERROR' if errors else 'no ERROR'}",
              did_refuse == refuses and has_error == errors,
              f"converter {reasons.get(ep_dir.name, 'took it')!r}; validator errors {found['errors']}, "
              f"warnings {found['warnings'][:3]}")
    soft = by_name["null tau"]["warnings"] + by_name["null tff and lim"]["warnings"]
    check("a null tau, tff or lim is a validator WARN",
          sum("null or non-finite tau" in w or "null or non-finite tff" in w or "null or non-finite lim" in w
              for w in soft) == 3, str(soft))
    pose = by_name["null inside hand, head and tgt"]["warnings"]
    check("a null inside hand, head or tgt is a validator WARN", sum("inside" in w for w in pose) == 3, str(pose))
    unread = by_name["null q in a row no step reads"]["warnings"]
    check("a null q in a row no step reads is a validator WARN", any("no step reads" in w for w in unread),
          str(unread))
    incomplete = by_name["incomplete (outcome null), torn last line"]
    check("an incomplete episode is a WARN ('incomplete: the converter skips it'), its torn line too, and the "
          "converter skips it as not finalized",
          incomplete["warnings"][0].startswith("incomplete: the converter skips it")
          and any(w.startswith("(incomplete episode)") and "torn" in w for w in incomplete["warnings"])
          and "not finalized" in reasons.get("ep_0011", ""), str(incomplete["warnings"]))
    early = by_name["a torn JPEG no step shows (before the first row)"]["warnings"]
    check("a torn JPEG no step shows is a validator WARN", any("no step shows" in w for w in early), str(early))
    check("validating that session exits 1 only for the defects (the incomplete episode alone is no error)",
          report["n_errors"] == sum(1 for c in cases if c[4]))


if __name__ == "__main__":
    sys.exit(main())
