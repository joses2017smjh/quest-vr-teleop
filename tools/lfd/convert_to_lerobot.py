"""Convert recorded LfD sessions into one LeRobotDataset v3.0 (docs/LFD_RECORDING_FORMAT.md, section 4).

Runs on the HPC in the Python 3.12 LeRobot 0.6.1 environment, which needs LeRobot's dataset extra (datasets, av,
pandas, pyarrow). For each episode it:
  1. applies the spec's refusals: discarded, an outcome other than success or failure (null is an incomplete
     episode), no task text, policy mode (unless --modes asks for it), a mode not in --modes, cmd on anything but
     teleop, a zero_check that is null, malformed or 'report' (unless --allow-unverified-zero; 'holds' with
     zero_fix_applied false is taken with a warning), and a session.json of another format, without 10 distinct
     joint names or a side_by_side camera;
  2. puts the tap rows and frames on a 30 fps grid and builds the state and the chosen action label (labels.py).
     That refuses rows whose src contradicts the mode (a teleop episode with a policy or guide row, a guided one
     with a teleop or policy row), tap rows of another version or without a field it reads, frames out of order
     or stepping back more than 5 ms, a null q or qc in a row the steps read, a null or changed zero (or one that
     differs from zero_at_start), more than --max-missing (2 %) of the steps without a frame within +-1/fps, and
     reused + skipped frames above --max-reused-skipped (5 %) of the steps;
  3. checks that every frame the steps show is a whole JPEG of one even-width size, then decodes each one, turns
     the side-by-side frame 180 degrees, splits it into the two eyes and (with --image-size) shrinks each eye. A
     frame that does not decode refuses its episode.
It refuses the whole run, writing nothing, if the accepted episodes differ in joint names, claw limits, camera frame
size or zero convention. It then writes the dataset with observation.state[12], action[12] (both named),
observation.images.left / .right as AV1 video, the spec's extra columns (q_cmd, cam_pan_tilt with cam_known,
mode_id, intervention (bool), t_rel (float32 seconds since the episode's first step), frame_skew: deliberately not
under "observation.", so no policy takes them as inputs) and each episode's task, and calls finalize().

After finalize() it checks every episode's video holds exactly its number of steps, and replaces the quantiles of
meta/stats.json (q01, q10, q50, q90, q99) for observation.state and action with numpy.quantile over every frame of
the dataset: LeRobot 0.6.1 writes the count-weighted mean of per-episode quantiles there, which is not a quantile,
and pi0.5 normalizes with q01/q99 without clipping. LeRobot's count, mean, std, min and max are kept, after a check
against numpy. stats.json is written as strict JSON (no NaN or Infinity). Last, beside the data in the dataset root,
it writes splits.json (whole episodes, per mode, seeded; see splits.py) and conversion.json: the arguments, the
sha256 of every input episode.json and rows.jsonl, the code's hashes and git state, the label, the episode-to-source
map with each episode's trial, arrangement, operator, policy, outcome, end_reason, notes, t0 (monotonic s of its
first step) and grid counts (missing, reused, skipped), every rejected episode with its reason, the video settings,
the stats recomputation, and the LeRobot version. A run that fails after writing has no conversion.json.

The dataset goes to OUT_ROOT/REPO_ID, which LeRobotDataset(REPO_ID, root=OUT_ROOT/REPO_ID) loads. Export HF_HOME to
the share first: LeRobot's readers cache under it, and the home directory has no room.

  export HF_HOME=/nfs/hpc/share/$USER/.cache/huggingface
  python tools/lfd/convert_to_lerobot.py --sessions /nfs/hpc/share/$USER/bhl-data/incoming/2026100* \\
      --out-root /nfs/hpc/share/$USER/bhl-data/lerobot --repo-id local/fold_v1 --label meas_future
  python tools/lfd/convert_to_lerobot.py --sessions S1 S2 --out-root OUT --repo-id local/teleop_cmd --label cmd \\
      --modes teleop --image-size 320x240 --overwrite

Encoding. AV1 (libsvtav1) on the CPU is the cost. The options, all LeRobot 0.6.1's own:
  --streaming-encoding        feed frames straight to one encoder thread per eye instead of writing PNGs and
                              encoding them at the end of each episode. LeRobot drops a frame when its queue stays
                              full, so the converter waits for room before each frame, checks LeRobot's drop count
                              after each episode, and checks every video's length at the end.
  --image-writer-threads N    without streaming: write the PNGs in N background threads.
  --parallel-encoding / --no-parallel-encoding
                              without streaming: encode the two eyes in two processes. The default is on when the
                              job has 4 or more CPUs (counted with os.sched_getaffinity, which respects a Slurm
                              allocation; os.cpu_count() gives the node's).
  --encoder-threads N         SVT-AV1 threads (lp) per encoder; default half the job's CPUs, at least 1 (0: the
                              encoder decides, and SVT-AV1 then counts the node's cores, not the job's).
Measured on a 2-CPU job, one synthetic 15 s episode (447 steps, two 640x480 eyes), time spent writing the episode:
PNG path 23.4 s, with 2 PNG writer threads 22.5 s, plus parallel encoding 23.2 s, streaming 16.0 s, streaming with
1 encoder thread per eye 14.8 s (no frame dropped). Hence the defaults. conversion.json's "encoding" records what a
run used and how long it took.

Running it on the cluster. Run real conversions as a CPU batch job on the share partition, not on an interactive
2-CPU job: before these settings a 60 s episode at 640x480 per eye took 2 min 17 s on 2 CPUs; with them the
synthetic episode above wrote at 30 steps/s, but real camera images hold more detail and encode slower, so budget
generously and read the "encoding" seconds of the first run. With 8 CPUs each eye's encoder gets 4 threads. Logs and
data stay on the share; nothing goes to the home directory. From inside an interactive job, first drop the SLURM_*
variables it set (they would leak into the batch job). For example (this is not submitted by any tool):

  cd /nfs/hpc/share/$USER/quest-vr-teleop
  env $(env | sed -n 's/^\\(SLURM_[A-Z0-9_]*\\)=.*/-u \\1/p') \\
  sbatch --account=eecs --partition=share --cpus-per-task=8 --mem=16G --time=12:00:00 --job-name=lfd-convert \\
      --output=/nfs/hpc/share/%u/bhl-data/logs/%x-%j.out \\
      --wrap "export HF_HOME=/nfs/hpc/share/$USER/.cache/huggingface; \\
              /nfs/hpc/share/$USER/envs/lerobot-py312-cpu/bin/python tools/lfd/convert_to_lerobot.py \\
              --sessions /nfs/hpc/share/$USER/bhl-data/incoming/2026100* \\
              --out-root /nfs/hpc/share/$USER/bhl-data/lerobot --repo-id local/fold_v1 --label meas_future"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import labels  # noqa: E402
import splits  # noqa: E402

CONVERSION_FORMAT = "bhl-lfd-conversion"
ROBOT_TYPE = "bhl_arms"
STATE, ACTION = "observation.state", "action"
LEFT, RIGHT = "observation.images.left", "observation.images.right"
# The spec's extra columns and their dtypes. None of these names starts with "observation" or "action", which is
# what LeRobot's dataset_to_policy_features keys on, so they never become policy inputs or outputs.
EXTRAS = {"q_cmd": "float32", "cam_pan_tilt": "float32", "cam_known": "bool", "mode_id": "int64",
          "intervention": "bool", "t_rel": "float32", "frame_skew": "float32"}
QUANTILE_KEY = re.compile(r"q(\d+)")          # LeRobot's quantile keys: q01, q10, q50, q90, q99 (percent)
DEFAULT_STREAMING = True                      # measured fastest on 2 CPUs; see the module docstring
DEFAULT_IMAGE_WRITER_THREADS = 2              # used only without streaming
DEFAULT_ENCODER_QUEUE = 30                    # frames per eye buffered for the streaming encoder (LeRobot's default)
ENCODER_WAIT_S = 600.0                        # give up when a streaming encoder takes no frame for this long


class ConversionRefused(RuntimeError):
    """The run was refused before anything was written; str() says why."""


class FrameDecodeError(ValueError):
    """A frame the steps show did not decode; its episode is refused."""


def usable_cpus() -> int:
    """CPUs this process may run on: a Slurm job's allocation, not the node's count (os.cpu_count)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:                    # not Linux
        return os.cpu_count() or 1


def sha256_file(path: Path) -> str | None:
    if not Path(path).exists():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def code_record() -> dict:
    """Which code did the conversion: the commit, whether the tree (and tools/lfd) differ from it, and the hashes of
    the files that decide the dataset's contents."""
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    tools = _git("status", "--porcelain", "--", "tools/lfd")
    return {"repo_commit": commit.strip() if commit else None,
            "repo_dirty": bool(status.strip()) if status is not None else None,
            "tools_lfd_status": tools.splitlines()[:50] if tools is not None else None,
            "convert_to_lerobot_sha256": sha256_file(Path(__file__)),
            "labels_sha256": sha256_file(HERE / "labels.py"), "splits_sha256": sha256_file(HERE / "splits.py")}


def parse_size(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    try:
        width, height = (int(v) for v in text.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--image-size wants WxH, not {text!r}") from exc
    if width < 16 or height < 16 or width % 2 or height % 2:
        raise argparse.ArgumentTypeError("--image-size needs even sizes of at least 16 (the video is 4:2:0)")
    return width, height


# ----------------------------------------------------------------------------------------- choosing episodes


def gate(meta: dict, session: dict, ep_dir: Path, label: str, modes: tuple[str, ...],
         allow_unverified_zero: bool) -> str | None:
    """Why the spec refuses this episode before its data is even read, or None."""
    if meta.get("format") != labels.EPISODE_FORMAT or meta.get("version") != labels.FORMAT_VERSION:
        return f"episode.json is format {meta.get('format')!r} version {meta.get('version')!r}"
    if meta.get("discarded") or ep_dir.parent.name == "_discarded":
        return "discarded"
    outcome = meta.get("outcome")
    if outcome == "aborted":
        return f"outcome aborted (end_reason {meta.get('end_reason')!r})"
    if outcome not in ("success", "failure"):
        return f"not finalized: outcome {outcome!r}"
    mode = meta.get("mode")
    if mode not in labels.MODES:
        return f"unknown mode {mode!r}"
    if mode == "policy" and "policy" not in modes:
        return "policy episodes are excluded unless asked for (--modes ...,policy)"
    if mode not in modes:
        return f"mode {mode} not requested (--modes {','.join(modes)})"
    try:
        labels.check_mode(mode, label, allow_policy=True)
    except labels.LabelError as exc:
        return exc.reason
    if meta.get("zero_changed") is not False:
        return "the zero changed mid-episode"
    verified, message = labels.zero_check_verdict(meta.get("zero_check"))
    if not verified and not allow_unverified_zero:
        return f"{message} (--allow-unverified-zero keeps it)"
    task = meta.get("task")
    if not isinstance(task, str) or not task.strip():
        return "no task text"
    reason = labels.check_claw_limits(meta.get("claw_limits"))
    if reason:
        return reason
    return labels.session_problem(session)


def used_frame_size(episode: labels.Episode, steps: labels.Steps) -> tuple[int, int]:
    """The (width, height) shared by every frame the steps show; LabelError if one is not a whole JPEG, the size
    changes or the width is odd, which would otherwise stop the writer halfway through the dataset."""
    sizes = set()
    for k in np.unique(steps.frame):
        data = episode.frame_bytes(int(k))
        if not labels.is_jpeg(data):
            raise labels.LabelError(f"frame {int(k)} is not a whole JPEG (no SOI/EOI marker)")
        sizes.add(labels.jpeg_size(data))
    if len(sizes) != 1 or None in sizes:
        raise labels.LabelError(f"frames without a JPEG header, or of several sizes: {sorted(map(str, sizes))}")
    width, height = sizes.pop()
    if width % 2:
        raise labels.LabelError(f"frame width {width} is odd: the side-by-side frame cannot split into two eyes")
    return width, height


def mixing_conflicts(chosen: list[dict]) -> list[str]:
    """The spec's mixing rules: what differs among the accepted episodes, one line per rule broken."""
    lines = []
    for key in ("joint_names", "claw_limits", "frame_size", "zero_convention"):
        groups: dict[str, list[str]] = {}
        for c in chosen:
            groups.setdefault(json.dumps(c[key], sort_keys=True), []).append(c["source"])
        if len(groups) > 1:
            parts = "; ".join(f"{value} in {', '.join(sources)}" for value, sources in groups.items())
            lines.append(f"different {key}: {parts}")
    return lines


def collect(sessions: list[Path], label: str, fps: float, lookahead: float, modes: tuple[str, ...],
            allow_unverified_zero: bool, log=print, max_missing: float = labels.MAX_MISSING_FRACTION,
            max_reused_skipped: float = labels.MAX_REUSED_SKIPPED_FRACTION
            ) -> tuple[list[dict], list[dict], list[dict]]:
    """(accepted episodes with their steps, rejected episodes with reasons, input records with hashes)."""
    accepted, rejected, inputs = [], [], []
    for session_dir in sessions:
        session_dir = Path(session_dir).resolve()
        try:
            session = labels.load_session(session_dir)
        except ValueError as exc:
            raise ConversionRefused(f"{session_dir}/session.json is unreadable: {exc}") from exc
        record = {"session_id": session.get("session_id"), "path": str(session_dir),
                  "session_json_sha256": sha256_file(session_dir / "session.json"), "episodes": []}
        inputs.append(record)
        for ep_dir in labels.list_episodes(session_dir, include_discarded=True):
            entry = {"episode_dir": str(ep_dir), "episode_json_sha256": sha256_file(ep_dir / "episode.json"),
                     "rows_jsonl_sha256": sha256_file(ep_dir / "rows.jsonl"),
                     "frames_jsonl_sha256": sha256_file(ep_dir / "frames.jsonl"),
                     "frames_mjpeg_bytes": (ep_dir / "frames.mjpeg").stat().st_size
                     if (ep_dir / "frames.mjpeg").exists() else None}
            record["episodes"].append(entry)
            source = f"{session.get('session_id') or session_dir.name}/" + (
                f"_discarded/{ep_dir.name}" if ep_dir.parent.name == "_discarded" else ep_dir.name)
            entry["source"] = source
            try:
                meta = json.loads((ep_dir / "episode.json").read_text())
            except (OSError, ValueError) as exc:
                rejected.append({"source": source, "episode_dir": str(ep_dir), "reason": f"episode.json: {exc}"})
                continue
            reason = gate(meta, session, ep_dir, label, modes, allow_unverified_zero)
            if reason:
                rejected.append({"source": source, "episode_dir": str(ep_dir), "reason": reason})
                continue
            episode = labels.load_episode(ep_dir, session=session)
            try:
                steps = labels.build_steps(episode, fps, label, lookahead, allow_policy="policy" in modes,
                                           max_missing=max_missing, max_reused_skipped=max_reused_skipped)
                size = used_frame_size(episode, steps)
            except labels.LabelError as exc:
                rejected.append({"source": source, "episode_dir": str(ep_dir), "reason": exc.reason,
                                 "label_report": exc.report})
                continue
            finally:
                episode.close()
            episode.rows = {}                                # the writer needs only the frame index from here on
            verified, message = labels.zero_check_verdict(meta.get("zero_check"))
            notes = [message] if message else []
            if not verified:
                notes[-1] += " (kept: --allow-unverified-zero)"
            accepted.append({"source": source, "session": session, "session_dir": str(session_dir),
                             "episode_dir": str(ep_dir), "meta": meta, "steps": steps, "episode": episode,
                             "joint_names": episode.joint_names, "claw_limits": meta.get("claw_limits"),
                             "frame_size": list(size), "zero_convention": meta.get("zero_convention"),
                             "rotated_180": episode.rotated_180, "input": entry, "warnings": notes})
            log(f"  take {source}: {meta.get('mode')}, {len(steps.t)} steps"
                + "".join(f"; warning: {n}" for n in notes))
    for r in rejected:
        log(f"  skip {r['source']}: {r['reason']}")
    return accepted, rejected, inputs


# ------------------------------------------------------------------------------------------------ the writer


def eye_pair(jpeg: bytes, rotated_180: bool, image_size: tuple[int, int] | None) -> tuple[np.ndarray, np.ndarray]:
    """The two eyes of one frame as RGB uint8 (H, W, 3), turned upright and shrunk to image_size (W, H)."""
    try:
        frame = labels.decode_jpeg(jpeg)
    except ImportError:
        raise
    except Exception as exc:  # noqa: BLE001 - a decoder failure of any kind refuses the episode
        raise FrameDecodeError(str(exc) or type(exc).__name__) from exc
    left, right = labels.split_eyes(frame, rotated_180)
    if image_size is None or (left.shape[1], left.shape[0]) == tuple(image_size):
        return left, right
    try:
        import cv2
        return tuple(cv2.resize(eye, image_size, interpolation=cv2.INTER_AREA) for eye in (left, right))
    except ImportError:
        from PIL import Image
        return tuple(np.asarray(Image.fromarray(eye).resize(image_size, Image.BOX)) for eye in (left, right))


def features_for(names: list[str], joint_names: list[str], eye_hw: tuple[int, int]) -> dict:
    h, w = eye_hw
    video = {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
    return {
        STATE: {"dtype": "float32", "shape": (len(names),), "names": list(names)},
        ACTION: {"dtype": "float32", "shape": (len(names),), "names": list(names)},
        LEFT: dict(video),
        RIGHT: dict(video),
        "q_cmd": {"dtype": "float32", "shape": (len(joint_names),), "names": list(joint_names)},
        "cam_pan_tilt": {"dtype": "float32", "shape": (2,), "names": ["pan_deg", "tilt_deg"]},
        "cam_known": {"dtype": "bool", "shape": (1,), "names": None},
        "mode_id": {"dtype": "int64", "shape": (1,), "names": None},
        "intervention": {"dtype": "bool", "shape": (1,), "names": None},
        "t_rel": {"dtype": "float32", "shape": (1,), "names": None},
        "frame_skew": {"dtype": "float32", "shape": (1,), "names": None},
    }


def prepare_root(root: Path, overwrite: bool) -> None:
    if not root.exists():
        return
    if not overwrite:
        raise ConversionRefused(f"{root} exists; pass --overwrite to replace it")
    looks_like_ours = (root / "meta/info.json").exists() or (root / "conversion.json").exists() or not any(
        root.iterdir())
    if not looks_like_ours:
        raise ConversionRefused(f"{root} exists and is not a dataset this tool wrote; not deleting it")
    shutil.rmtree(root)


def _streaming_encoder(dataset):
    return getattr(getattr(dataset, "writer", None), "_streaming_encoder", None)


def wait_for_encoder_room(dataset) -> None:
    """Block while a streaming encoder's queue is full. LeRobot 0.6.1 drops a frame, with only a log line, when its
    queue stays full for 0.1 s; one dropped frame would shift that eye's video against every later step. This
    process is the only producer, so a queue with room now still has room when add_frame puts the frame."""
    encoder = _streaming_encoder(dataset)
    queues = list(getattr(encoder, "_frame_queues", {}).values()) if encoder is not None else []
    deadline = time.monotonic() + ENCODER_WAIT_S
    while any(q.full() for q in queues):
        if time.monotonic() > deadline:
            raise RuntimeError(f"the streaming video encoder took no frame for {ENCODER_WAIT_S:.0f} s")
        time.sleep(0.002)


def dropped_frames(dataset) -> int:
    """Frames LeRobot's streaming encoder dropped in the episode just saved (it resets at the next episode)."""
    encoder = _streaming_encoder(dataset)
    return int(sum(getattr(encoder, "_dropped_frames", {}).values())) if encoder is not None else 0


def read_episodes_meta(root: Path):
    import pandas as pd
    files = sorted((root / "meta/episodes").rglob("*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def video_length_problems(root: Path, fps: int, video_keys: tuple[str, ...]) -> list[str]:
    """Each episode's video span (to_timestamp - from_timestamp in meta/episodes) must hold exactly its steps."""
    problems = []
    for record in read_episodes_meta(root).to_dict("records"):
        for key in video_keys:
            span = float(record[f"videos/{key}/to_timestamp"]) - float(record[f"videos/{key}/from_timestamp"])
            if abs(span * fps - int(record["length"])) > 0.5:
                problems.append(f"episode {int(record['episode_index'])} {key}: {span * fps:.1f} frames of video "
                                f"for {int(record['length'])} steps")
    return problems


def _strict(value):
    raise ValueError(f"{value} is not valid in strict JSON")


def recompute_quantiles(root: Path, keys: tuple[str, ...] = (STATE, ACTION)) -> dict:
    """Replace the quantile entries LeRobot wrote into meta/stats.json for keys with numpy.quantile (its default
    'linear' method) over every frame of the dataset's parquet files; keep LeRobot's count, mean, std, min and max
    after checking them against numpy. Writes strict JSON. Returns the record conversion.json keeps."""
    import pandas as pd
    path = root / "meta/stats.json"
    stats = json.loads(path.read_text())
    files = sorted((root / "data").rglob("*.parquet"))
    table = pd.concat([pd.read_parquet(p, columns=list(keys)) for p in files], ignore_index=True)
    record: dict = {"method": "numpy.quantile, method 'linear', over every frame (validation episodes included)",
                    "frames": int(len(table)), "keys": {}}
    for key in keys:
        values = np.stack(table[key].to_numpy()).astype(np.float64)        # float32 in parquet, exact in float64
        entry = stats[key]
        replaced = []
        for name in sorted(entry):
            match = QUANTILE_KEY.fullmatch(name)
            if match and 0 <= int(match.group(1)) <= 100:
                entry[name] = np.quantile(values, int(match.group(1)) / 100.0, axis=0).tolist()
                replaced.append(name)
        count = int(np.asarray(entry["count"]).ravel()[0])
        if count != len(values):
            raise RuntimeError(f"meta/stats.json counts {count} frames of {key}, the parquet files hold "
                               f"{len(values)}")
        for name, want in (("min", values.min(axis=0)), ("max", values.max(axis=0))):
            if not np.array_equal(np.asarray(entry[name], dtype=np.float64), want):
                raise RuntimeError(f"meta/stats.json {key} {name} differs from numpy's: {entry[name]} vs "
                                   f"{want.tolist()}")
        record["keys"][key] = {
            "quantiles_replaced": replaced,
            "mean_max_abs_diff": float(np.abs(np.asarray(entry["mean"]) - values.mean(axis=0)).max()),
            "std_max_abs_diff": float(np.abs(np.asarray(entry["std"]) - values.std(axis=0)).max()),
            "count_min_max": "equal to numpy's",
        }
    bad = [f"{feature}.{name}" for feature, entry in stats.items() for name, value in entry.items()
           if not np.all(np.isfinite(np.asarray(value, dtype=float)))]
    if bad:
        raise RuntimeError("meta/stats.json would hold NaN or Infinity in " + ", ".join(bad))
    text = json.dumps(stats, indent=4, allow_nan=False)
    json.loads(text, parse_constant=_strict)
    path.write_text(text)
    return record


def convert(sessions: list[Path], out_root: Path, repo_id: str, label: str, fps: int = labels.FPS,
            lookahead: float = labels.LOOKAHEAD_S, modes: tuple[str, ...] = ("teleop", "guided"),
            allow_unverified_zero: bool = False, val_fraction: float = 0.2, seed: int = 0,
            image_size: tuple[int, int] | None = None, overwrite: bool = False,
            parallel_encoding: bool | None = None, log=print,
            max_missing: float = labels.MAX_MISSING_FRACTION,
            max_reused_skipped: float = labels.MAX_REUSED_SKIPPED_FRACTION,
            streaming_encoding: bool = DEFAULT_STREAMING, image_writer_threads: int = DEFAULT_IMAGE_WRITER_THREADS,
            encoder_threads: int | None = None, encoder_queue: int = DEFAULT_ENCODER_QUEUE) -> dict:
    """Convert sessions into OUT_ROOT/REPO_ID. Returns the conversion record; raises ConversionRefused.

    parallel_encoding None means: on when this job may use 4 or more CPUs. encoder_threads None means half the
    job's CPUs (at least 1); 0 leaves it to the encoder."""
    if label not in labels.LABELS:
        raise ValueError(f"label must be one of {labels.LABELS}")
    if "/" not in repo_id or repo_id.startswith("/") or ".." in repo_id.split("/"):
        raise ValueError(f"--repo-id wants a name like local/NAME, not {repo_id!r}")
    unknown = set(modes) - set(labels.MODES)
    if unknown:
        raise ValueError(f"unknown modes {sorted(unknown)}")
    for name, value in (("--max-missing", max_missing), ("--max-reused-skipped", max_reused_skipped)):
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name} is a fraction in [0, 1), not {value}")
    if image_writer_threads < 0 or (encoder_threads is not None and encoder_threads < 0) or encoder_queue < 1:
        raise ValueError("thread and queue counts must be positive")
    cpus = usable_cpus()
    if parallel_encoding is None:
        parallel_encoding = cpus >= 4
    if encoder_threads is None:
        encoder_threads = max(1, cpus // 2)                # two encoders, one per eye
    encoder_threads = encoder_threads or None             # 0: the encoder decides
    root = Path(out_root).resolve() / repo_id
    started = time.time()
    log(f"Reading {len(sessions)} session(s) for label {label}"
        + (f" (H = {lookahead:.3f} s)" if label == "meas_future" else ""))
    accepted, rejected, inputs = collect(sessions, label, fps, lookahead, tuple(modes), allow_unverified_zero, log,
                                         max_missing, max_reused_skipped)
    conflicts = mixing_conflicts(accepted)
    if conflicts:
        raise ConversionRefused("these episodes cannot share one dataset:\n  " + "\n  ".join(conflicts))
    if not accepted:
        raise ConversionRefused("no episode passed the spec's rules; see the reasons above")
    prepare_root(root, overwrite)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")       # a local conversion never needs the Hub
    os.environ.setdefault("SVT_LOG", "2")              # SVT-AV1: warnings and errors, not its config banner
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")   # one line per episode is enough in a job log
    warnings.filterwarnings("ignore", message="Cannot enable progress bars")   # LeRobot tries to, after loading
    import lerobot
    from lerobot.datasets import LeRobotDataset

    width, height = accepted[0]["frame_size"]
    eye_wh = tuple(image_size) if image_size else (width // 2, height)
    first = accepted[0]
    features = features_for(first["steps"].names, first["joint_names"], (eye_wh[1], eye_wh[0]))
    encoding = {"streaming_encoding": bool(streaming_encoding),
                "image_writer_threads": 0 if streaming_encoding else int(image_writer_threads),
                "parallel_encoding": bool(parallel_encoding) and not streaming_encoding,
                "encoder_threads": encoder_threads, "encoder_queue": encoder_queue if streaming_encoding else None,
                "usable_cpus": cpus}
    log("Encoding: " + ", ".join(f"{k} {v}" for k, v in encoding.items()))
    dataset = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=features, root=root, robot_type=ROBOT_TYPE,
                                    use_videos=True, image_writer_threads=encoding["image_writer_threads"],
                                    streaming_encoding=encoding["streaming_encoding"],
                                    encoder_queue_maxsize=encoder_queue, encoder_threads=encoder_threads)
    if streaming_encoding and not hasattr(_streaming_encoder(dataset), "_frame_queues"):
        dataset.finalize()
        raise RuntimeError(f"LeRobot {lerobot.__version__}'s streaming encoder has no _frame_queues to throttle: "
                           f"run with --no-streaming-encoding")
    episodes_out = []
    encode_s = 0.0
    try:
        for item in accepted:
            steps, meta = item["steps"], item["meta"]
            task, index = meta["task"].strip(), len(episodes_out)
            log(f"  episode {index}: {item['source']} ({len(steps.t)} steps) ...")
            tick = time.time()
            t_rel = steps.t_rel
            try:
                with item["episode"] as episode:
                    cached, eyes = -1, None
                    for k in range(len(steps.t)):
                        if steps.frame[k] != cached:              # consecutive steps can share a frame
                            cached = int(steps.frame[k])
                            eyes = eye_pair(episode.frame_bytes(cached), item["rotated_180"], eye_wh)
                        if streaming_encoding:
                            wait_for_encoder_room(dataset)
                        dataset.add_frame({
                            STATE: steps.state[k], ACTION: steps.action[k], LEFT: eyes[0], RIGHT: eyes[1],
                            "q_cmd": steps.q_cmd[k], "cam_pan_tilt": steps.cam_pan_tilt[k],
                            "cam_known": np.array([steps.cam_known[k]], dtype=bool),
                            "mode_id": np.array([steps.mode_id], dtype=np.int64),
                            "intervention": np.array([steps.intervention[k]], dtype=bool),
                            "t_rel": np.array([t_rel[k]], dtype=np.float32),
                            "frame_skew": np.array([steps.frame_skew[k]], dtype=np.float32),
                            "task": task,
                        })
            except FrameDecodeError as exc:
                dataset.clear_episode_buffer()
                reason = f"frame {cached} does not decode: {exc}"
                rejected.append({"source": item["source"], "episode_dir": item["episode_dir"], "reason": reason})
                log(f"  skip {item['source']}: {reason}")
                continue
            dataset.save_episode(parallel_encoding=encoding["parallel_encoding"])
            lost = dropped_frames(dataset)
            if lost:
                raise RuntimeError(f"the streaming encoder dropped {lost} frames of {item['source']}: the video no "
                                   f"longer matches the steps; rerun with --no-streaming-encoding")
            encode_s += time.time() - tick
            item["input"]["frames_mjpeg_sha256"] = sha256_file(Path(item["episode_dir"]) / "frames.mjpeg")
            report = steps.report
            episodes_out.append({
                "index": index, "source": item["source"], "session_dir": item["session_dir"],
                "episode_dir": item["episode_dir"], "mode": meta["mode"], "task": task,
                "trial": meta.get("trial"), "arrangement": meta.get("arrangement"),
                "operator": meta.get("operator"), "policy": meta.get("policy"), "outcome": meta["outcome"],
                "end_reason": meta.get("end_reason"), "notes": meta.get("notes"), "t0": steps.t0,
                "steps": len(steps.t), "rotated_180": item["rotated_180"],
                "grid": {"steps": report["grid"]["steps"], "missing": report["grid"]["missing"],
                         "reused": report["grid"]["reused"], "skipped": report["grid"]["skipped"]},
                "warnings": item["warnings"] + report["warnings"], "label_report": report,
            })
    except ValueError as exc:            # LeRobot refusing a frame: a fault here, not a refusal of the data
        raise RuntimeError(f"writing the dataset failed: {exc}") from exc
    finally:
        dataset.finalize()
    if not episodes_out:
        raise RuntimeError(f"every accepted episode failed while writing; {root} holds no episode")

    problems = video_length_problems(root, fps, (LEFT, RIGHT))
    if problems:
        raise RuntimeError("the videos do not match the steps: " + "; ".join(problems[:5]))
    stats_record = recompute_quantiles(root)
    split = splits.make_splits([{"index": e["index"], "mode": e["mode"], "source": e["source"]}
                                for e in episodes_out], val_fraction, seed)
    splits.write_splits(root / "splits.json", split)
    info = json.loads((root / "meta/info.json").read_text())
    record = {
        "format": CONVERSION_FORMAT, "version": 1, "spec": "docs/LFD_RECORDING_FORMAT.md v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "host": socket.gethostname(),
        "seconds": round(time.time() - started, 1),
        "args": {"sessions": [str(s) for s in sessions], "out_root": str(out_root), "repo_id": repo_id,
                 "fps": fps, "label": label, "lookahead": lookahead, "modes": list(modes),
                 "allow_unverified_zero": allow_unverified_zero, "val_fraction": val_fraction, "seed": seed,
                 "image_size": list(image_size) if image_size else None, "overwrite": overwrite,
                 "max_missing": max_missing, "max_reused_skipped": max_reused_skipped},
        "label": label, "lookahead_s": lookahead if label == "meas_future" else None, "fps": fps,
        "lerobot_version": lerobot.__version__,
        "code": code_record(),
        "encoding": {**encoding, "seconds": round(encode_s, 1),
                     "video": {key: (info["features"][key].get("info") or {}) for key in (LEFT, RIGHT)}},
        "dataset": {"repo_id": repo_id, "root": str(root), "episodes": len(episodes_out),
                    "frames": sum(e["steps"] for e in episodes_out), "robot_type": ROBOT_TYPE,
                    "source_frame_size": [width, height], "image_size": list(eye_wh),
                    "camera_latency_s": {a["session"].get("session_id"): (a["session"].get("camera") or {})
                                         .get("camera_latency_s") for a in accepted},
                    "extra_columns": dict(EXTRAS),
                    "time": "t_rel (float32) is seconds since the episode's first step; that step's monotonic time "
                            "is the episode's t0 below"},
        "stats": {"recomputed": True, "quantiles": stats_record,
                  "note": "LeRobot 0.6.1 aggregates per-episode quantiles by a count-weighted mean; the quantiles "
                          "of observation.state and action were replaced with numpy's over every frame. The "
                          "per-episode stats in meta/episodes keep LeRobot's values."},
        "inputs": inputs,
        "episodes": episodes_out,
        "rejected": rejected,
        "splits": {"file": "splits.json", "val_fraction": val_fraction, "seed": seed,
                   "train": len(split["train"]), "val": len(split["val"]), "rule": split["rule"]},
    }
    (root / "conversion.json").write_text(json.dumps(_json_safe(record), indent=1, allow_nan=False) + "\n")
    log(f"Wrote {len(episodes_out)} episodes, {record['dataset']['frames']} frames to {root} "
        f"({len(rejected)} rejected, {record['seconds']} s)")
    return record


def _plain(value):
    """numpy values on their way to JSON."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot store {type(value).__name__} in conversion.json")


def _json_safe(value):
    """conversion.json is strict JSON: numpy values to Python, NaN and infinities to null."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.generic, np.ndarray, Path)):
        return _json_safe(_plain(value))
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sessions", nargs="+", type=Path, required=True, help="session directories")
    parser.add_argument("--out-root", type=Path, required=True, help="the dataset goes to OUT_ROOT/REPO_ID")
    parser.add_argument("--repo-id", required=True, help="local/NAME")
    parser.add_argument("--fps", type=int, default=labels.FPS)
    parser.add_argument("--label", choices=labels.LABELS, required=True,
                        help="cmd (teleop only) or meas_future (teleop and hand-guided)")
    parser.add_argument("--lookahead", type=float, default=labels.LOOKAHEAD_S, help="H for meas_future, seconds")
    parser.add_argument("--modes", default="teleop,guided", help="episode modes to take; add policy to take those")
    parser.add_argument("--allow-unverified-zero", action="store_true",
                        help="keep episodes whose zero_check is null, malformed or 'report'")
    parser.add_argument("--max-missing", type=float, default=labels.MAX_MISSING_FRACTION,
                        help="reject an episode with more than this fraction of steps without a frame within "
                             "+-1/fps (default 0.02)")
    parser.add_argument("--max-reused-skipped", type=float, default=labels.MAX_REUSED_SKIPPED_FRACTION,
                        help="reject an episode whose reused + skipped frames exceed this fraction of its steps "
                             "(default 0.05)")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=parse_size, default=None, metavar="WxH",
                        help="shrink each eye to this size (default: half the frame width by its height)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing dataset at OUT_ROOT/REPO_ID")
    parser.add_argument("--streaming-encoding", action=argparse.BooleanOptionalAction, default=DEFAULT_STREAMING,
                        help="encode while frames are added, one thread per eye, instead of via PNG files "
                             f"(default {'on' if DEFAULT_STREAMING else 'off'})")
    parser.add_argument("--image-writer-threads", type=int, default=DEFAULT_IMAGE_WRITER_THREADS,
                        help=f"without streaming: threads writing the PNGs (default {DEFAULT_IMAGE_WRITER_THREADS})")
    parser.add_argument("--parallel-encoding", action=argparse.BooleanOptionalAction, default=None,
                        help="without streaming: encode the two eyes in two processes (default: on with 4 or more "
                             "CPUs in this job)")
    parser.add_argument("--encoder-threads", type=int, default=None,
                        help="SVT-AV1 threads per encoder (default: half this job's CPUs, at least 1; 0: the "
                             "encoder decides)")
    parser.add_argument("--encoder-queue", type=int, default=DEFAULT_ENCODER_QUEUE,
                        help=f"with streaming: frames buffered per eye (default {DEFAULT_ENCODER_QUEUE})")
    args = parser.parse_args(argv)
    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    try:
        convert(args.sessions, args.out_root, args.repo_id, args.label, args.fps, args.lookahead, modes,
                args.allow_unverified_zero, args.val_fraction, args.seed, args.image_size, args.overwrite,
                args.parallel_encoding, max_missing=args.max_missing, max_reused_skipped=args.max_reused_skipped,
                streaming_encoding=args.streaming_encoding, image_writer_threads=args.image_writer_threads,
                encoder_threads=args.encoder_threads, encoder_queue=args.encoder_queue)
    except (ConversionRefused, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"FAILED: {exc}\n(the dataset folder, if any, is incomplete: it has no conversion.json)", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
