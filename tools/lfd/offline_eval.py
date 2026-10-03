"""Offline (open-loop) evaluation of one trained LeRobot policy on the validation episodes.

For every sampled frame of every validation episode named in splits.json, the policy sees the recorded
observation (both camera eyes, the 12-D state, the task text) and predicts a chunk of actions. The chunk is
compared with the actions the demonstrator actually produced next. Nothing is executed and nothing
compounds: every prediction starts again from a recorded observation.

THIS MEASURES IMITATION ERROR, NOT TASK SUCCESS. A policy can score well here and still fail on the robot
(drift, timing, contact), or score worse and succeed. Ranking policies needs the matched live evaluation
in docs/LFD_EVAL_PROTOCOL.md.

It reports, in the dataset's units (rad for the 10 joints, 0..1 for the two claws):
  - L1 against horizon (1, 5, 10, 20, 30, 40 frames), for the joints and the claws apart. These are the
    numbers to compare across policies: every policy here predicts at least 40 steps, and every horizon is
    scored on the same (frame, step) pairs;
  - per-joint L1 over the common horizon (the first 40 steps), also comparable across policies;
  - per-joint L1 of the first predicted action;
  - L1 over the policy's own whole chunk (100, 50 or 40 steps): per-policy only, NOT comparable across
    policies, because each averages over a different length;
  - the same numbers for a "hold the current state" baseline, so a policy that learned nothing shows up;
  - inference time per call (preprocess + model + postprocess, batch 1) on the current device.

Before scoring, it checks the checkpoint was trained on THIS dataset: the conversion.json sha256 recorded
when the run started (the run's inputs.json, written by tools/lfd/train/common.sh) must equal this
dataset's. A re-conversion or a copy can put other demonstrations behind the same episode indices, so a
different or unknown dataset is refused unless --allow-different-dataset is given, and the report then says
"different dataset", never "ok". It also refuses a validation episode that is in the training list.

It works for the three policies of tools/lfd/train/ (act, pi05 with or without LoRA, groot), and for any
LeRobot 0.6.1 checkpoint directory holding config.json, train_config.json and the weights.

  source tools/lfd/train/common.sh     # every cache on the share: an evaluation refuses to fill home's
  RUN=/nfs/hpc/share/$USER/bhl-data/train/washcloth_v1/act/seed1000
  "$LEROBOT_PY" tools/lfd/offline_eval.py --checkpoint $RUN/train/checkpoints/last/pretrained_model \\
      --dataset-root /nfs/hpc/share/$USER/bhl-data/lerobot/local/washcloth_v1 --out $RUN/offline_eval/last.json

  # the episode lists, as tools/lfd/train/common.sh passes them to --dataset.episodes:
  python tools/lfd/offline_eval.py --dataset-root /path/to/dataset --print-episodes train
  # the dataset's identity by content, as common.sh records it in <run>/inputs.json:
  python tools/lfd/offline_eval.py --dataset-root /path/to/dataset --print-identity

Needs Python 3.12 with LeRobot 0.6.1 and its dataset extra (the GPU env of tools/lfd/train/README.md,
or the CPU env for tests). --print-episodes and --print-identity need only the standard library.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import platform
import random
import socket
import sys
import time
from pathlib import Path

FORMAT = "bhl-lfd-offline-eval"
VERSION = 1
NOT_TASK_SUCCESS = (
    "This measures imitation error (open-loop action prediction on held-out demonstrations), not task "
    "success. Ranking policies needs the matched live evaluation in docs/LFD_EVAL_PROTOCOL.md."
)
# LeRobot 0.6.1 prints these instead of raising when PI0.5 weights fail to load; the policy then keeps its
# random init and every number below would be meaningless, so loading output is checked for them.
WEIGHT_LOAD_FAILURES = ("Returning model without loading pretrained weights", "Could not load state dict")
# Frames at 30 fps. 40 is GR00T N1.7's native chunk, the shortest of the three policies.
SUMMARY_HORIZONS = (1, 5, 10, 20, 30, 40)
# The common horizon: every policy here predicts at least this many steps (ACT 100, pi05 50, GR00T 40).
COMMON_HORIZON = 40
# <run>/inputs.json, written by tools/lfd/train/common.sh when a run starts: what it was trained on.
RUN_INPUTS_FORMAT = "bhl-lfd-run-inputs"
# What a resumed run must still match (common.sh refuses a resume on any change). conversion.json holds the
# label and lookahead too; they are listed on their own so a refusal names them.
GATED_IDENTITY_KEYS = ("conversion_sha256", "stats_sha256", "splits_sha256", "label", "lookahead_s")


class SplitsError(ValueError):
    """splits.json is missing, or does not hold disjoint whole-episode train/val lists."""


def find_splits(dataset_root: Path, explicit: str | None) -> Path:
    """The splits file: --splits if given, else inside the dataset root, else next to it."""
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise SplitsError(f"--splits {path} does not exist")
        return path
    inside, beside = dataset_root / "splits.json", dataset_root.parent / "splits.json"
    if inside.is_file():
        return inside
    if beside.is_file():
        # docs/LFD_RECORDING_FORMAT.md says only "beside the dataset"; a sibling file could belong to
        # another dataset in the same folder, so say which one was taken.
        print(f"note: using {beside} (no splits.json inside {dataset_root})", file=sys.stderr)
        return beside
    raise SplitsError(f"no splits.json in {dataset_root} or beside it; the converter writes one (format v1 §4)")


def _episode_list(value: object, key: str, path: Path) -> list[int]:
    if not isinstance(value, list):
        raise SplitsError(f"{path}: '{key}' must be a list of episode indices, got {type(value).__name__}")
    out: list[int] = []
    for item in value:
        if isinstance(item, dict):
            item = item.get("episode_index")
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise SplitsError(f"{path}: '{key}' entries must be LeRobot episode_index values (ints >= 0)")
        out.append(item)
    if len(set(out)) != len(out):
        raise SplitsError(f"{path}: '{key}' lists an episode twice")
    return sorted(out)


def load_splits(path: Path, total_episodes: int | None = None) -> dict:
    """Read {"train": [...], "val": [...]} (or "validation"); check the lists are disjoint and in range."""
    raw = path.read_bytes()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SplitsError(f"{path}: not JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise SplitsError(f"{path}: expected a JSON object with 'train' and 'val' lists")
    val_key = next((k for k in ("val", "validation") if k in data), None)
    if "train" not in data or val_key is None:
        raise SplitsError(f"{path}: needs top-level 'train' and 'val' lists of episode indices, has {sorted(data)}")
    train = _episode_list(data["train"], "train", path)
    val = _episode_list(data[val_key], val_key, path)
    both = sorted(set(train) & set(val))
    if both:
        raise SplitsError(f"{path}: episodes {both} are in both train and {val_key}")
    if not train:
        raise SplitsError(f"{path}: the train list is empty")
    if total_episodes is not None:
        outside = [e for e in train + val if e >= total_episodes]
        if outside:
            raise SplitsError(f"{path}: episodes {outside} do not exist (the dataset has {total_episodes})")
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "train": train, "val": val}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode_sources(splits_path: Path) -> dict[int, str]:
    """episode_index -> source (session_id/ep_NNNN) from splits.json's "episodes", when it has them."""
    try:
        data = json.loads(splits_path.read_bytes())
    except (OSError, json.JSONDecodeError):
        return {}
    entries = data.get("episodes") if isinstance(data, dict) else None
    if isinstance(entries, dict):  # {"<index>": {"source": ...}}, as tools/lfd/splits.py writes it
        items = [(k, v) for k, v in entries.items()]
    elif isinstance(entries, list):  # [{"episode_index" or "index": i, "source": ...}]
        items = [(e.get("episode_index", e.get("index")), e) for e in entries if isinstance(e, dict)]
    else:
        return {}
    out = {}
    for key, value in items:
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict) and isinstance(value.get("source"), str):
            out[index] = value["source"]
    return out


def dataset_identity(dataset_root: Path, splits_file: str | None = None) -> dict:
    """The dataset a run is given, by content. Standard library only: common.sh records it in <run>/inputs.json
    when a run starts and compares it on every resume, and evaluate() compares it before scoring."""
    info_path = dataset_root / "meta" / "info.json"
    total = json.loads(info_path.read_text()).get("total_episodes") if info_path.is_file() else None
    splits_path = find_splits(dataset_root, splits_file)
    splits = load_splits(splits_path, total)
    conversion, stats = dataset_root / "conversion.json", dataset_root / "meta" / "stats.json"
    record: dict = {}
    if conversion.is_file():
        with contextlib.suppress(json.JSONDecodeError):
            loaded = json.loads(conversion.read_bytes())
            record = loaded if isinstance(loaded, dict) else {}
    sources = _episode_sources(splits_path)
    return {
        "dataset_root": str(dataset_root),
        "conversion_sha256": file_sha256(conversion) if conversion.is_file() else None,
        "stats_sha256": file_sha256(stats) if stats.is_file() else None,
        "info_sha256": file_sha256(info_path) if info_path.is_file() else None,
        "splits_file": splits["path"],
        "splits_sha256": splits["sha256"],
        "label": record.get("label"),
        "lookahead_s": record.get("lookahead_s"),
        "train_episodes": splits["train"],
        "val_episodes": splits["val"],
        "train_sources": [sources[e] for e in splits["train"]] if sources and all(e in sources for e in
                                                                                    splits["train"]) else None,
    }


def identity_changes(recorded: dict, now: dict, keys: tuple[str, ...] = GATED_IDENTITY_KEYS) -> list[str]:
    """The gated keys whose value differs, as 'key: old -> new'."""
    return [f"{k}: {recorded.get(k)!r} -> {now.get(k)!r}" for k in keys if recorded.get(k) != now.get(k)]


def find_train_identity(ckpt: Path, explicit: str | None) -> tuple[dict | None, str | None]:
    """The identity of the dataset the checkpoint was trained on: --train-identity, else the run's inputs.json
    (<run>/train/checkpoints/<step>/pretrained_model -> <run>/inputs.json), else a job record holding it."""
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise SystemExit(f"error: --train-identity {path} does not exist")
        data = json.loads(path.read_text())
        return (data.get("inputs") if data.get("format") != RUN_INPUTS_FORMAT else data), str(path)
    for parent in list(ckpt.parents)[:4]:
        candidate = parent / "inputs.json"
        if candidate.is_file():
            with contextlib.suppress(json.JSONDecodeError):
                data = json.loads(candidate.read_text())
                if isinstance(data, dict) and data.get("format") == RUN_INPUTS_FORMAT:
                    return data, str(candidate)
        for record in sorted((parent / "jobs").glob("*.json"), reverse=True) if (parent / "jobs").is_dir() else []:
            with contextlib.suppress(json.JSONDecodeError, OSError):
                data = json.loads(record.read_text())
                if isinstance(data, dict) and isinstance(data.get("inputs"), dict):
                    return data["inputs"], str(record)
    return None, None


def dataset_problem(trained: dict | None, scored: dict) -> str | None:
    """Why this dataset may not be the one the checkpoint was trained on, or None when the content matches."""
    if trained is None:
        return ("unknown training dataset: no inputs.json was found for this checkpoint's run (pass "
                "--train-identity), so its training data cannot be compared with this dataset")
    was, now = trained.get("conversion_sha256"), scored.get("conversion_sha256")
    if was is None or now is None:
        return (f"unknown training dataset: conversion.json sha256 recorded at training {was!r}, here {now!r}; "
                "without both, the two datasets cannot be compared")
    if was != now:
        return (f"different dataset: the checkpoint was trained on conversion.json sha256 {was[:12]} "
                f"({trained.get('dataset_root')}), this dataset has {now[:12]} ({scored.get('dataset_root')})")
    return None


def read_info(dataset_root: Path) -> dict:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        # Checked before LeRobotDataset is built: given a root without meta/, it creates the folder and
        # tries to download the repo id from the Hub.
        raise SystemExit(f"error: {info_path} not found; --dataset-root must be a LeRobot v3 dataset root")
    return json.loads(info_path.read_text())


def home_cache_problem() -> str | None:
    """Where the caches an evaluation writes would go, if into the home directory (on this HPC home is 25 GB and
    nearly full: tools/lfd/train/common.sh points them at the share). Loading a dataset writes the datasets
    cache; building ACT downloads its ImageNet backbone into TORCH_HOME before the checkpoint replaces it."""
    home = os.path.realpath(os.path.expanduser("~"))
    xdg = os.environ.get("XDG_CACHE_HOME") or os.path.join(home, ".cache")
    hf_home = os.environ.get("HF_HOME") or os.path.join(xdg, "huggingface")
    caches = {
        "HF_HOME": hf_home,
        "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE") or os.path.join(hf_home, "datasets"),
        "TORCH_HOME": os.environ.get("TORCH_HOME") or os.path.join(xdg, "torch"),
    }
    for name, path in caches.items():
        real = os.path.realpath(path)
        if real == home or real.startswith(home + os.sep):
            return f"{name} would be {path}, in the home directory"
    return None


def finite(value: float) -> float | None:
    """JSON has no NaN: a horizon no validation frame reaches is reported as null."""
    return float(value) if math.isfinite(value) else None


def describe(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)

    def pick(q: float) -> float:
        return ordered[min(n - 1, int(round(q * (n - 1))))]

    return {"n": n, "mean": sum(ordered) / n, "median": pick(0.5), "p95": pick(0.95),
            "min": ordered[0], "max": ordered[-1]}


def action_names(features: dict, dim: int) -> list[str]:
    names = (features.get("action") or {}).get("names")
    if isinstance(names, dict):  # some LeRobot datasets nest them, e.g. {"motors": [...]}
        names = next(iter(names.values()), None)
    if not isinstance(names, list) or len(names) != dim:
        names = [f"action_{i}" for i in range(dim)]
    return [str(n) for n in names]


def action_units(dim: int) -> list[str]:
    # Format v1 §4: action = 10 joint angles (rad, ARM_JOINTS order), then the two claw commands (0..1).
    return ["rad"] * 10 + ["claw_0_to_1"] * 2 if dim == 12 else ["unknown"] * dim


def resolve_checkpoint(path: str) -> Path:
    """Accept the pretrained_model dir, its step dir, or checkpoints/last."""
    ckpt = Path(path).resolve()
    if not (ckpt / "config.json").is_file() and (ckpt / "pretrained_model" / "config.json").is_file():
        ckpt = ckpt / "pretrained_model"
    if not (ckpt / "config.json").is_file():
        raise SystemExit(f"error: no config.json in {ckpt}; pass .../checkpoints/<step>/pretrained_model")
    return ckpt


def leakage_problem(train_cfg: dict | None, val: list[int]) -> str | None:
    """Why this checkpoint must not be scored on these episodes, or None."""
    if train_cfg is None:
        return "the checkpoint has no train_config.json, so its training episodes cannot be checked"
    episodes = (train_cfg.get("dataset") or {}).get("episodes")
    if episodes is None:
        return "it was trained on every episode (dataset.episodes is null), validation included"
    overlap = sorted(set(episodes) & set(val))
    if overlap:
        return f"validation episodes {overlap} were in its training list"
    return None


@contextlib.contextmanager
def watch_output(markers: tuple[str, ...]):
    """Echo what the block prints, and keep a copy so it can be searched for failure messages."""
    seen = io.StringIO()

    class Tee(io.TextIOBase):
        def write(self, text: str) -> int:
            seen.write(text)
            return sys.__stdout__.write(text)

        def flush(self) -> None:
            sys.__stdout__.flush()

    with contextlib.redirect_stdout(Tee()):
        yield seen
    found = [m for m in markers if m in seen.getvalue()]
    if found:
        raise SystemExit(f"error: the policy weights did not load ({found[0]!r}); refusing to score a random init")


def load_policy(cfg, ckpt: Path):
    """The trained policy, LoRA adapters included (mirrors lerobot.rollout.context._load_pretrained_policy)."""
    from lerobot.policies import get_policy_class

    policy_cls = get_policy_class(cfg.type)
    with watch_output(WEIGHT_LOAD_FAILURES):
        if not getattr(cfg, "use_peft", False):
            # strict: a checkpoint that does not match its own config is an error, not a warning
            return policy_cls.from_pretrained(str(ckpt), config=cfg, strict=True)
        from peft import PeftConfig, PeftModel

        peft_cfg = PeftConfig.from_pretrained(str(ckpt))
        base = policy_cls.from_pretrained(peft_cfg.base_model_name_or_path, config=cfg, revision=peft_cfg.revision)
        return PeftModel.from_pretrained(base, str(ckpt), config=peft_cfg)


def pick_device(name: str) -> str:
    import torch

    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("error: --device cuda but no GPU is visible")
    return name


def sample_indices(episode_col: list[int], frame_col: list[int], stride: int, per_episode: int) -> list[int]:
    """Row positions to score: every stride-th frame of each episode; with a cap, that many spread evenly
    over the episode, so a capped run still sees the start, the grasp and the end."""
    rows_by_episode: dict[int, list[int]] = {}
    for row, (ep, frame) in enumerate(zip(episode_col, frame_col)):
        if frame % stride == 0:
            rows_by_episode.setdefault(ep, []).append(row)
    chosen = []
    for rows in rows_by_episode.values():
        if per_episode and len(rows) > per_episode:
            last = len(rows) - 1
            rows = [rows[round(i * last / (per_episode - 1))] for i in range(per_episode)] if per_episode > 1 \
                else [rows[0]]
        chosen.extend(rows)
    return sorted(set(chosen))


def evaluate(args: argparse.Namespace) -> dict:
    # Every refusal below comes before torch and LeRobot are imported: they need only the standard library.
    dataset_root = Path(args.dataset_root).resolve()
    info = read_info(dataset_root)
    splits = load_splits(find_splits(dataset_root, args.splits), info.get("total_episodes"))
    val = splits["val"]
    if not val:
        raise SystemExit("error: splits.json has no validation episodes to score")

    ckpt = resolve_checkpoint(args.checkpoint)
    train_cfg_path = ckpt / "train_config.json"
    train_cfg = json.loads(train_cfg_path.read_text()) if train_cfg_path.is_file() else None
    # 1. The same dataset, by content: the episode indices below only mean something within one conversion.
    trained, identity_file = find_train_identity(ckpt, args.train_identity)
    scored = dataset_identity(dataset_root, args.splits)
    other_dataset = dataset_problem(trained, scored)
    if other_dataset and not args.allow_different_dataset:
        raise SystemExit(f"error: refusing to score this checkpoint on this dataset: {other_dataset}. A re-conversion "
                         "or a copy can put other demonstrations behind the same episode indices. Use "
                         "--allow-different-dataset only to debug; the report then says 'different dataset'.")
    # 2. No validation episode in the training list: by index, and by source where both splits files name them.
    problem = leakage_problem(train_cfg, val)
    val_sources = _episode_sources(Path(splits["path"]))
    trained_sources = set((trained or {}).get("train_sources") or [])
    shared = sorted(trained_sources & {val_sources[e] for e in val if e in val_sources})
    if shared:
        problem = "; ".join(p for p in (problem, f"validation sources {shared} were in its training list") if p)
    if problem and not args.allow_overlap:
        raise SystemExit(f"error: refusing to score this checkpoint on these episodes: {problem}. "
                         "Use --allow-overlap only to debug, never for a reported number.")
    trained_root = ((train_cfg or {}).get("dataset") or {}).get("root")
    if trained_root and Path(trained_root).resolve() != dataset_root and not other_dataset:
        print(f"note: the checkpoint was trained on {trained_root}; {dataset_root} has the same conversion.json, "
              "so it is the same dataset", file=sys.stderr)
    if other_dataset:
        leakage_check = f"{other_dataset}; episode check: {problem or 'no validation index in the training list'}"
    elif problem:
        leakage_check = problem
    else:
        leakage_check = ("ok: same dataset (conversion.json sha256 matches the run's record) and no validation "
                         "episode was in the training list")
    repo_id = args.repo_id or ((train_cfg or {}).get("dataset") or {}).get("repo_id") or f"local/{dataset_root.name}"

    import numpy as np
    import torch

    import lerobot
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata, resolve_delta_timestamps
    from lerobot.policies import make_pre_post_processors

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # pi05 and groot sample noise for every chunk; ACT is deterministic
    device = pick_device(args.device)

    cfg = PreTrainedConfig.from_pretrained(str(ckpt))
    cfg.device = device  # config.json names the training device
    cfg.pretrained_path = ckpt
    policy = load_policy(cfg, ckpt)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=str(ckpt), preprocessor_overrides={"device_processor": {"device": device}}
    )

    meta = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    delta_timestamps = resolve_delta_timestamps(cfg, meta)
    image_transforms = None
    if args.resize:
        from torchvision.transforms import v2

        # Smoke tests only: must equal the resize the policy was trained with.
        image_transforms = v2.Resize(tuple(args.resize))
    dataset = LeRobotDataset(repo_id, root=dataset_root, episodes=val, delta_timestamps=delta_timestamps,
                             image_transforms=image_transforms, video_backend=args.video_backend)
    columns = dataset.hf_dataset.select_columns(["episode_index", "frame_index"])
    rows = sample_indices([int(v) for v in columns["episode_index"]], [int(v) for v in columns["frame_index"]],
                          args.stride, args.max_samples_per_episode)
    if not rows:
        raise SystemExit("error: no frames selected; check --stride and the validation episodes")
    # spawn, as lerobot-train does: PyAV and torchcodec are not fork-safe
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, rows), batch_size=1, shuffle=False,
                                         num_workers=args.num_workers,
                                         multiprocessing_context="spawn" if args.num_workers else None)

    action_dim = int(meta.features["action"]["shape"][0])
    names = action_names(meta.features, action_dim)
    units = action_units(action_dim)
    state_dim = int((meta.features.get("observation.state") or {}).get("shape", [0])[0])
    hold_ok = state_dim == action_dim  # format v1: state and action are both [10 joints, 2 claws]
    std = meta.stats.get("action", {}).get("std") if meta.stats else None
    std = np.asarray(std, dtype=np.float64).reshape(-1) if std is not None else None

    horizon = len(cfg.action_delta_indices or [0])
    common = min(COMMON_HORIZON, horizon)  # 40 for all three policies here; less only for a tiny test policy
    sums = {k: np.zeros((horizon, action_dim)) for k in ("policy", "hold")}
    counts = np.zeros(horizon)
    per_episode: dict[int, dict] = {}
    total_times, model_times = [], []
    pred_len = None
    sync = torch.cuda.synchronize if device.startswith("cuda") else (lambda: None)

    for n, batch in enumerate(loader):
        obs = {k: v for k, v in batch.items() if k.startswith("observation.")}
        obs["task"] = list(batch["task"])  # a list of str, as lerobot-rollout passes it
        truth = batch["action"][0].double().numpy()  # (horizon, action_dim)
        pad = batch["action_is_pad"][0].numpy().astype(bool)
        state = batch["observation.state"][0].double().numpy()
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            processed = preprocessor(obs)
            sync()
            t1 = time.perf_counter()
            chunk = policy.predict_action_chunk(processed)
            sync()
            t2 = time.perf_counter()
            chunk = postprocessor(chunk)
        sync()
        t3 = time.perf_counter()
        if n >= args.warmup:  # the first calls include CUDA init and autotuning
            total_times.append(t3 - t0)
            model_times.append(t2 - t1)
        pred = chunk[0].detach().double().cpu().numpy()
        pred_len = pred.shape[0]
        steps = min(pred_len, truth.shape[0])
        valid = ~pad[:steps]
        diff = np.abs(pred[:steps] - truth[:steps])
        err = diff[valid]
        err_common = diff[:common][valid[:common]]
        sums["policy"][:steps][valid] += err
        if hold_ok:
            sums["hold"][:steps][valid] += np.abs(state[None, :] - truth[:steps])[valid]
        counts[:steps][valid] += 1
        ep = int(batch["episode_index"][0])
        rec = per_episode.setdefault(ep, {"episode_index": ep, "n_samples": 0, "_first": 0.0, "_common": 0.0,
                                          "_chunk": 0.0})
        rec["n_samples"] += 1
        rec["_first"] += float(err[0].mean())
        rec["_common"] += float(err_common.mean()) if err_common.size else 0.0
        rec["_chunk"] += float(err.mean())

    joints, claws = (slice(0, 10), slice(10, 12)) if action_dim == 12 else (slice(0, action_dim), slice(0, 0))

    def table(which: str) -> dict:
        s = sums[which]
        with np.errstate(invalid="ignore", divide="ignore"):
            per_step = s / counts[:, None]  # (horizon, dim); NaN where no frame reaches that far
        curve = per_step.mean(axis=1)

        def by_horizon(dims: slice) -> dict:
            return {str(h): finite(per_step[h - 1, dims].mean()) for h in SUMMARY_HORIZONS if h <= horizon}

        first = s[0] / max(counts[0], 1)
        common_l1 = s[:common].sum(axis=0) / max(counts[:common].sum(), 1)
        chunk = s.sum(axis=0) / max(counts.sum(), 1)  # every valid (frame, step) pair weighs the same
        out = {
            # Comparable across policies: the same (frame, step) pairs for every policy, at every horizon.
            "l1_by_horizon": {str(h): finite(curve[h - 1]) for h in SUMMARY_HORIZONS if h <= horizon},
            "l1_by_horizon_joints_rad": by_horizon(joints),
            "l1_by_horizon_claws": by_horizon(claws) if action_dim == 12 else None,
            "l1_by_horizon_per_dim": {str(h): {nm: finite(per_step[h - 1, i]) for i, nm in enumerate(names)}
                                      for h in SUMMARY_HORIZONS if h <= horizon},
            "common_horizon_steps": common,
            "common_horizon_l1": {nm: float(v) for nm, v in zip(names, common_l1)},
            "common_horizon_l1_mean": float(common_l1.mean()),
            "common_horizon_l1_joints_rad_mean": float(common_l1[joints].mean()),
            "common_horizon_l1_claws_mean": float(common_l1[claws].mean()) if action_dim == 12 else None,
            "first_action_l1": {nm: float(v) for nm, v in zip(names, first)},
            "first_action_l1_mean": float(first.mean()),
            # Per-policy only: each policy averages over its own chunk length (ACT 100, pi05 50, GR00T 40).
            "chunk_steps": horizon,
            "chunk_l1": {nm: float(v) for nm, v in zip(names, chunk)},
            "chunk_l1_mean": float(chunk.mean()),
            "chunk_l1_note": (f"over this policy's own chunk of {horizon} steps: not comparable across policies with "
                              "different chunks; compare l1_by_horizon and common_horizon_l1 instead"),
            "l1_curve": [finite(v) for v in curve],
            "frames_per_horizon_step": [int(v) for v in counts],
        }
        if std is not None and std.shape[0] == action_dim:
            scale = np.where(std > 0, std, 1.0)
            out["first_action_l1_over_std_mean"] = float((first / scale).mean())
            out["common_horizon_l1_over_std_mean"] = float((common_l1 / scale).mean())
        if action_dim == 12:
            out["joints_rad_chunk_l1_mean"] = float(chunk[joints].mean())
            out["claws_chunk_l1_mean"] = float(chunk[claws].mean())
        return out

    episodes = []
    for rec in sorted(per_episode.values(), key=lambda r: r["episode_index"]):
        k = max(rec["n_samples"], 1)
        episodes.append({"episode_index": rec["episode_index"], "n_samples": rec["n_samples"],
                         "first_action_l1_mean": rec.pop("_first") / k,
                         "common_horizon_l1_mean": rec.pop("_common") / k, "chunk_l1_mean": rec.pop("_chunk") / k})

    step_file = ckpt.parent / "training_state" / "training_step.json"
    step = json.loads(step_file.read_text()).get("step") if step_file.is_file() else None
    conversion = dataset_root / "conversion.json"  # written by convert_to_lerobot.py beside the data
    conversion_record = json.loads(conversion.read_text()) if conversion.is_file() else {}
    return {
        "format": FORMAT,
        "version": VERSION,
        "what_this_measures": NOT_TASK_SUCCESS,
        "created_wall": time.time(),
        "host": socket.gethostname(),
        "checkpoint": {
            "path": str(ckpt),
            "step": step,
            "policy_type": cfg.type,
            "use_peft": bool(getattr(cfg, "use_peft", False)),
            "chunk_size": getattr(cfg, "chunk_size", None),
            "n_action_steps": getattr(cfg, "n_action_steps", None),
            "predicted_chunk_length": pred_len,
            # docs/LFD_EVAL_PROTOCOL.md records each live-evaluated checkpoint by path and SHA-256
            "weights_sha256": {f.name: file_sha256(f) for f in sorted(ckpt.glob("*.safetensors"))},
            "trained_episodes": ((train_cfg or {}).get("dataset") or {}).get("episodes"),
            "trained_root": trained_root,
            # "ok: ..." only when the dataset is the one trained on and no validation episode was trained on
            "leakage_check": leakage_check,
            "same_dataset": other_dataset is None,
            "train_identity_file": identity_file,
            "trained_conversion_sha256": (trained or {}).get("conversion_sha256"),
            "trained_splits_sha256": (trained or {}).get("splits_sha256"),
            "pretrained": (trained or {}).get("pretrained"),
        },
        "dataset": {
            "root": str(dataset_root),
            "repo_id": repo_id,
            "fps": info.get("fps"),
            "total_episodes": info.get("total_episodes"),
            "splits_file": splits["path"],
            "splits_sha256": splits["sha256"],
            "val_episodes": val,
            "conversion_sha256": file_sha256(conversion) if conversion.is_file() else None,
            "stats_sha256": scored["stats_sha256"],
            "label": conversion_record.get("label"),
            "lookahead_s": conversion_record.get("lookahead_s"),
            "action_names": names,
            "action_units": units,
        },
        "settings": {
            "device": device,
            "device_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else platform.processor(),
            "batch_size": 1,
            "stride": args.stride,
            "max_samples_per_episode": args.max_samples_per_episode,
            "warmup_calls_excluded": args.warmup,
            "seed": args.seed,
            "video_backend": args.video_backend,
            "resize": args.resize,
            "allow_overlap": args.allow_overlap,
            "allow_different_dataset": args.allow_different_dataset,
        },
        "n_samples": len(rows),
        "n_episodes": len(episodes),
        "policy": table("policy"),
        "baseline_hold_state": table("hold") if hold_ok else None,
        "per_episode": episodes,
        "inference_time_s": {
            "per_call": describe(total_times),
            "model_only": describe(model_times),
            "note": "batch 1; preprocess + predict_action_chunk + postprocess, synchronized; data loading excluded",
        },
        "versions": {"python": platform.python_version(), "lerobot": lerobot.__version__,
                     "torch": torch.__version__, "cuda": torch.version.cuda},
    }


def print_summary(report: dict) -> None:
    pol, base = report["policy"], report["baseline_hold_state"]
    timing = report["inference_time_s"]["per_call"] or {}

    def curve(values: dict | None) -> str:
        return ", ".join(f"{h}: {v:.4f}" if v is not None else f"{h}: n/a" for h, v in (values or {}).items())

    print(f"\n{report['checkpoint']['policy_type']} step {report['checkpoint']['step']}: "
          f"{report['n_samples']} frames from {report['n_episodes']} validation episodes "
          f"on {report['settings']['device']}; leakage check: {report['checkpoint']['leakage_check']}")
    print("  Comparable across policies (the same frames and steps for every policy):")
    print(f"    L1 by horizon (frames), joints, rad:  {curve(pol['l1_by_horizon_joints_rad'])}")
    if pol.get("l1_by_horizon_claws") is not None:
        print(f"    L1 by horizon (frames), claws, 0..1: {curve(pol['l1_by_horizon_claws'])}")
    if base:
        print(f"    hold-the-state baseline, joints:      {curve(base['l1_by_horizon_joints_rad'])}")
    print(f"    over the first {pol['common_horizon_steps']} steps: mean L1 {pol['common_horizon_l1_mean']:.4f}"
          + (f" (baseline {base['common_horizon_l1_mean']:.4f})" if base else ""))
    for name in pol["common_horizon_l1"]:
        print(f"      {name:28s} first {pol['first_action_l1'][name]:.4f}   first {pol['common_horizon_steps']} "
              f"steps {pol['common_horizon_l1'][name]:.4f}")
    print(f"  Per-policy only: L1 over its own chunk of {pol['chunk_steps']} steps {pol['chunk_l1_mean']:.4f} "
          "(not comparable across policies with different chunks)")
    if timing:
        print(f"  inference per call: mean {timing['mean'] * 1e3:.1f} ms, p95 {timing['p95'] * 1e3:.1f} ms")
    print(f"\n{NOT_TASK_SUCCESS}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=NOT_TASK_SUCCESS)
    parser.add_argument("--dataset-root", required=True, help="LeRobot v3 dataset root (holds meta/info.json)")
    parser.add_argument("--splits", help="splits.json (default: inside the dataset root, else beside it)")
    parser.add_argument("--print-episodes", choices=("train", "val"),
                        help="print that episode list as [i,j,...] for --dataset.episodes, and exit")
    parser.add_argument("--print-identity", action="store_true",
                        help="print the dataset's identity by content (sha256 of conversion.json, meta/stats.json "
                             "and the splits file; label; lookahead) as JSON, and exit")
    parser.add_argument("--checkpoint", help="the pretrained_model dir of a checkpoint (or its step dir)")
    parser.add_argument("--out", help="JSON report path (required with --checkpoint)")
    parser.add_argument("--repo-id", help="dataset repo id (default: the one the checkpoint was trained with)")
    parser.add_argument("--device", default="auto", help="auto, cuda or cpu (default auto)")
    parser.add_argument("--stride", type=int, default=1, help="score every Nth frame (default 1: all)")
    parser.add_argument("--max-samples-per-episode", type=int, default=0, help="cap per episode (0: none)")
    parser.add_argument("--warmup", type=int, default=3, help="calls left out of the timing (default 3)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=None, help="data loader workers (default 4 on GPU)")
    parser.add_argument("--video-backend", default="pyav", help="pyav (default, as in training) or torchcodec")
    parser.add_argument("--resize", type=int, nargs=2, metavar=("H", "W"),
                        help="resize camera frames (smoke tests only; must match training)")
    parser.add_argument("--allow-overlap", action="store_true",
                        help="score even if validation episodes were trained on (debugging only)")
    parser.add_argument("--train-identity",
                        help="the run's inputs.json (default: found beside the checkpoint's run folder)")
    parser.add_argument("--allow-different-dataset", action="store_true",
                        help="score even if the checkpoint was trained on another or unknown dataset (debugging "
                             "only; the report says 'different dataset', never 'ok')")
    args = parser.parse_args(argv)

    dataset_root = Path(args.dataset_root)
    if args.print_identity:
        try:
            identity = dataset_identity(dataset_root.resolve(), args.splits)
        except (SplitsError, OSError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(identity, sort_keys=True))
        return 0
    if args.print_episodes:
        info_path = dataset_root / "meta" / "info.json"
        total = json.loads(info_path.read_text()).get("total_episodes") if info_path.is_file() else None
        try:
            splits = load_splits(find_splits(dataset_root, args.splits), total)
        except SplitsError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print("[" + ",".join(str(e) for e in splits[args.print_episodes]) + "]")
        return 0

    if not args.checkpoint or not args.out:
        parser.error("--checkpoint and --out are required unless --print-episodes is given")
    if args.stride < 1:
        parser.error("--stride must be >= 1")
    if args.num_workers is None:
        args.num_workers = 4 if args.device != "cpu" and os.cpu_count() and os.cpu_count() > 4 else 0
    cache_problem = home_cache_problem()
    if cache_problem and os.environ.get("LFD_ALLOW_HOME_CACHE") != "1":
        parser.error(f"{cache_problem}: source tools/lfd/train/common.sh first, which puts every cache on the share "
                     "(or set LFD_ALLOW_HOME_CACHE=1)")
    print(NOT_TASK_SUCCESS)
    try:
        report = evaluate(args)
    except SplitsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    tmp.replace(out)  # a reader never sees half a report
    print_summary(report)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
