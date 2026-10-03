"""Check the LfD training jobs against the installed LeRobot, with no GPU and no weight downloads.

Nothing here submits a job or touches the share's data: every run writes under a scratch folder, caches
included, and the last check fails if anything landed in the home caches. Where a check needs them, the
Hugging Face Hub, lfs, squeue, nvidia-smi, torch's GPU and lerobot-train are mocks in that folder (PATH and
PYTHONPATH shims). The real lfs is read once, read-only.

  1. Each sbatch header: account eecs, ampere, at most one day (two such jobs fit the QoS cap of 2880 reserved
     GPU-minutes), one GPU, logs on the share; π0.5 full is documented with --constraint=vram80g and checks
     MIN_GPU_MEM_GB >= 79. Bash syntax of the job scripts and common.sh; common.sh sourced by sh.
  2. The exact lerobot-train arguments each job would pass (LFD_PRINT_ARGS=1) parse in the installed LeRobot
     (draccus into TrainPipelineConfig, then validate()) for ACT, π0.5 LoRA, π0.5 full fine-tuning and GR00T
     N1.7, with the train episodes and no validation episode, save_freq 1000, the seed, LoRA only where
     intended, and the pretrained weights pinned: π0.5 gets the local snapshot of the resolved commit, GR00T a
     pinned view of it.
  3. Each job end to end on CPU with a mocked GPU env and a fake lerobot-train: the run lock; the pinned
     revision (mocked Hub) recorded in inputs.json and the job record before training starts; the quota and GPU
     checks on mocked lfs and devices (MIN_GPU_MEM_GB refuses a 40 GB MIG slice); a resume that keeps the
     recorded revision when the Hub's main has moved; pruning to KEEP_LAST; the optimizer-state drop. A
     realistic π0.5 LoRA checkpoint (use_peft, adapter_config.json) resumes through LeRobot's PeftConfig path
     (peft when installed, else its adapter_config.json). The resume content gate refuses a changed
     conversion.json, stats.json, splits.json, label, lookahead or pretrained revision, LFD_EXTEND_STEPS or not.
  4. GR00T's pinned view gives LeRobot's Hub-id processor (new_embodiment slot 10, no state dropout), where the
     raw snapshot would not. Pruning safety, the quota logic and the GPU check, on their own.
  5. offline_eval.py refuses a checkpoint trained on a different or unknown dataset, or on a validation source.
  6. With --dataset-root: a tiny ACT training on CPU through act.sbatch itself (20 steps, a changed dataset
     refused, a resubmission resumes it to 30; batch 2; 96x128 images), then tools/lfd/offline_eval.py on its
     checkpoint, on a copy of the dataset, and on a different dataset. It trains on a shadow copy of the
     dataset (small files copied, folders linked), so the given dataset is never modified.

  /nfs/hpc/share/sanchej7/envs/lerobot-py312-cpu/bin/python tools/lfd/train/test_configs.py
  /nfs/hpc/share/sanchej7/envs/lerobot-py312-cpu/bin/python tools/lfd/train/test_configs.py \\
      --dataset-root /path/to/lerobot/dataset

Step 6 needs LeRobot's dataset and training extras (datasets, av, accelerate) in this Python. The PEFT load path
itself (PeftConfig, then the base, then PeftModel) runs only in an env with peft, so on a GPU run.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TRAIN = REPO / "tools" / "lfd" / "train"
OFFLINE_EVAL = REPO / "tools" / "lfd" / "offline_eval.py"
SCRATCH = Path(os.environ.get("LFD_TEST_SCRATCH", f"/scratch/{os.environ.get('USER', 'user')}/tmp"))
FAKE_SPLITS = {"train": [0, 1, 2, 4], "val": [3, 5]}
SEED = "1234"
MOCK_SHA = "0123456789abcdef0123456789abcdef01234567"   # the commit the mocked Hub's main points to
OTHER_SHA = "fedcba9876543210fedcba9876543210fedcba98"  # where main moves to later
# lfs quota -q -p 30762 /nfs/hpc/share, read on 2026-10-03: 19 GiB over the 1.5 TiB soft quota, grace running
QUOTA_LINE = "  /nfs/hpc/share 1630827124* 1610612736 2147483648 {grace} 2066352*       0       0       -"
GPUS = {  # name|GiB as torch reports total_memory|major|minor|native bf16
    "h100": "NVIDIA H100 80GB HBM3|79.65|9|0|1",
    "mig": "NVIDIA H100 80GB HBM3 MIG 3g.40gb|39.25|9|0|1",
    "a40": "NVIDIA A40|44.99|8|6|1",
    "rtx8000": "Quadro RTX 8000|47.46|7|5|0",
}
HIDDEN = ("processor_config.json", "statistics.json", "embodiment_id.json")  # groot.sbatch's PRETRAINED_HIDE

results: list[tuple[bool, str]] = []


def check(ok: bool, what: str) -> bool:
    results.append((bool(ok), what))
    print(("PASS  " if ok else "FAIL  ") + what, flush=True)
    return bool(ok)


def tail(run: subprocess.CompletedProcess, n: int = 400) -> str:
    return (run.stdout + run.stderr).strip()[-n:].replace("\n", " | ")


# ---- mocks ------------------------------------------------------------------------------------------------
MOCK_HUB = '''"""Mock of the two huggingface_hub calls tools/lfd/train/common.sh makes (lfd_pretrained). Tests only."""
import json
import os
from pathlib import Path
from types import SimpleNamespace


def _log(**event):
    with open(os.environ["LFD_MOCK_HUB_LOG"], "a") as f:
        f.write(json.dumps(event) + "\\n")


class HfApi:
    def model_info(self, repo_id, revision=None, **kwargs):
        _log(call="model_info", repo=repo_id, revision=revision)
        return SimpleNamespace(id=repo_id, sha=os.environ["LFD_MOCK_HUB_SHA"])


def snapshot_download(repo_id, revision=None, cache_dir=None, **kwargs):
    _log(call="snapshot_download", repo=repo_id, revision=revision)
    hub = Path(cache_dir or os.environ.get("HF_HUB_CACHE") or Path(os.environ["HF_HOME"]) / "hub")
    snap = hub / ("models--" + repo_id.replace("/", "--")) / "snapshots" / revision
    snap.mkdir(parents=True, exist_ok=True)
    groot = "GR00T" in repo_id
    # the GR00T sidecars carry the values of the real nvidia/GR00T-N1.7-3B commit 2fc962b9 that matter here
    files = {
        "config.json": {"architectures": ["Gr00tN1d7"], "model_type": "Gr00tN1d7"} if groot else {"type": "pi05"},
        "embodiment_id.json": {"simpler_env_google": 0, "simpler_env_widowx": 1, "libero_sim": 2},
        "processor_config.json": {"processor_kwargs": {
            "state_dropout_prob": 0.2, "use_albumentations": True, "shortest_image_edge": 256,
            "crop_fraction": 0.95, "use_percentiles": True, "use_relative_action": True,
            "image_target_size": [256, 256], "image_crop_size": [230, 230], "max_action_horizon": 40,
            "modality_configs": {}}},
        "statistics.json": {"simpler_env_google": {}},
    }
    for name, content in files.items():
        (snap / name).write_text(json.dumps(content))
    (snap / "model.safetensors").write_bytes(b"mock")
    return str(snap)
'''

MOCK_TORCH = '''"""Mock torch for lfd_check_gpu: LFD_MOCK_GPU="name|GiB|major|minor|native bf16". Tests only."""
import os
from types import SimpleNamespace

__version__ = "0.0+mock"
version = SimpleNamespace(cuda="12.8")
_name, _gib, _major, _minor, _bf16 = os.environ["LFD_MOCK_GPU"].split("|")


class cuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def get_device_properties(index):
        return SimpleNamespace(name=_name, total_memory=int(float(_gib) * 2**30), major=int(_major), minor=int(_minor))

    @staticmethod
    def is_bf16_supported(including_emulation=True):
        return _bf16 == "1" or including_emulation
'''

# lerobot-train for the job-flow checks: parses the arguments as LeRobot does, writes a checkpoint the way
# save_checkpoint lays it out (LoRA: use_peft and adapter_config.json, as wrap_with_peft records the base),
# and reports whether the run's inputs.json and job record already held the pinned revision when it started.
FAKE_TRAIN = '''import json
import os
import sys
from pathlib import Path

sys.path[:] = [p for p in sys.path if "lfd-mocks" not in p]  # the real torch and LeRobot, not the GPU mocks
args = sys.argv[1:]


def value(flag):
    return next((a.split("=", 1)[1] for a in args if a.startswith(flag + "=")), None)


run_dir, job = Path(os.environ["RUN_DIR"]), os.environ["JOB_TAG"]
missing = []
inputs = run_dir / "inputs.json"
record = run_dir / "jobs" / f"{job}.json"
pin = json.loads(inputs.read_text()).get("pretrained") if inputs.is_file() else None
if not inputs.is_file():
    missing.append("inputs.json")
if os.environ.get("PRETRAINED_REPO"):
    if not (pin and pin.get("revision") and pin.get("path")):
        missing.append("inputs.json pretrained")
    if not (record.is_file() and json.loads(record.read_text()).get("pretrained") == pin):
        missing.append("job record pretrained")
print("FAKE_TRAIN preconditions " + ("ok" if not missing else "MISSING " + ",".join(missing)), flush=True)
print("Start offline training", flush=True)
(run_dir / "fake_train_args.txt").write_text("\\n".join(args) + "\\n")
if value("--config_path"):  # a resume
    config = Path(value("--config_path"))
    saved = json.loads(config.read_text())
    saved["steps"] = int(value("--steps") or saved["steps"])
    steps, out, peft = saved["steps"], Path(saved["output_dir"]), (saved.get("policy") or {}).get("use_peft")
    adapter = json.loads((config.parent / "adapter_config.json").read_text()) if peft else None
    write_config = lambda d: (d / "train_config.json").write_text(json.dumps(saved, indent=2))  # noqa: E731
else:
    import draccus

    import lerobot.policies  # noqa: F401 - registers the policy config classes, as lerobot-train does
    from lerobot.configs.train import TrainPipelineConfig

    sys.argv = ["lerobot-train", *args]
    cfg = draccus.parse(config_class=TrainPipelineConfig, args=args)
    cfg.validate()
    steps, out, peft = cfg.steps, Path(cfg.output_dir), cfg.peft is not None
    adapter = None
    if peft:
        cfg.policy.use_peft = True  # wrap_with_peft marks the config, and records the base as pretrained_path
        adapter = {"peft_type": "LORA", "base_model_name_or_path": str(cfg.policy.pretrained_path), "r": cfg.peft.r,
                   "lora_alpha": cfg.peft.lora_alpha, "revision": None, "task_type": None}
    write_config = cfg.save_pretrained
step_dir = out / "checkpoints" / f"{steps:06d}"
model = step_dir / "pretrained_model"
model.mkdir(parents=True, exist_ok=True)
write_config(model)
(model / "config.json").write_text("{}")
if peft:
    (model / "adapter_config.json").write_text(json.dumps(adapter))
    (model / "adapter_model.safetensors").write_bytes(b"0")
else:
    (model / "model.safetensors").write_bytes(b"0")
state = step_dir / "training_state"
state.mkdir(exist_ok=True)
(state / "training_step.json").write_text(json.dumps({"step": steps, "batch_size": int(os.environ["BATCH_SIZE"])}))
(state / "optimizer_state.safetensors").write_bytes(b"0")
(state / "optimizer_param_groups.json").write_text("[]")
last = out / "checkpoints" / "last"
if last.is_symlink():
    last.unlink()
last.symlink_to(step_dir.name)
print(f"FAKE_TRAIN saved {step_dir}", flush=True)
'''


def write_mocks(tmp: Path) -> Path:
    mocks = tmp / "lfd-mocks"
    (mocks / "bin").mkdir(parents=True)
    (mocks / "py" / "huggingface_hub").mkdir(parents=True)
    (mocks / "py" / "huggingface_hub" / "__init__.py").write_text(MOCK_HUB)
    gpu = mocks / "gpu"  # what the fake GPU env's Python sees first: a GPU torch and the policies' packages
    (gpu / "torch").mkdir(parents=True)
    (gpu / "torch" / "__init__.py").write_text(MOCK_TORCH)
    for package in ("transformers", "scipy", "peft", "diffusers"):
        (gpu / package).mkdir()
        (gpu / package / "__init__.py").write_text('"""Empty stand-in: lfd_check_env only looks for it."""\n')
    scripts = {
        "squeue": """case "${LFD_MOCK_SQUEUE:-invalid}" in
    invalid) echo "slurm_load_jobs error: Invalid job id specified" >&2; exit 1 ;;
    error) echo "slurm_load_jobs error: Unable to contact slurm controller" >&2; exit 1 ;;
    empty) exit 0 ;;
    *) echo "$LFD_MOCK_SQUEUE" ;;
esac""",
        "lfs": """case "$1" in
    project) echo "30762 P ${3:-/nfs/hpc/share/mock}" ;;
    quota) printf '%b\\n' "$LFD_MOCK_LFS_QUOTA" ;;
    *) exit 2 ;;
esac""",
        "nvidia-smi": 'echo "0, ${LFD_MOCK_GPU%%|*}, 81559 MiB, 550.54.15"',
    }
    for name, body in scripts.items():
        path = mocks / "bin" / name
        path.write_text("#!/bin/bash\n# mock for tools/lfd/train/test_configs.py\n" + body + "\n")
        path.chmod(0o755)
    env = tmp / "fake-gpu-env" / "bin"  # LEROBOT_ENV for the job-flow checks
    env.mkdir(parents=True)
    (env / "python").write_text(f'#!/bin/bash\nexport PYTHONPATH="{gpu}${{PYTHONPATH:+:$PYTHONPATH}}"\n'
                                f'exec {shlex.quote(sys.executable)} "$@"\n')
    (mocks / "fake_lerobot_train.py").write_text(FAKE_TRAIN)
    (env / "lerobot-train").write_text(f'#!/bin/bash\nexec {shlex.quote(sys.executable)} '
                                       f'{shlex.quote(str(mocks / "fake_lerobot_train.py"))} "$@"\n')
    for path in env.iterdir():
        path.chmod(0o755)
    return mocks


def scratch_caches(cache: Path) -> dict[str, str]:
    """Every cache the libraries know about, in the scratch folder. common.sh sets these for the job scripts;
    a Python started directly (offline_eval.py, this file itself) would otherwise write to home."""
    return {"XDG_CACHE_HOME": str(cache), "HF_HOME": str(cache / "huggingface"),
            "HF_LEROBOT_HOME": str(cache / "huggingface" / "lerobot"), "TORCH_HOME": str(cache / "torch"),
            "TRITON_CACHE_DIR": str(cache / "triton"), "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
            "MPLCONFIGDIR": str(cache / "matplotlib"), "PIP_CACHE_DIR": str(cache / "pip"),
            "WANDB_MODE": "disabled", "WANDB_DIR": str(cache / "wandb"), "WANDB_CACHE_DIR": str(cache / "wandb"),
            "WANDB_CONFIG_DIR": str(cache / "wandb"), "HF_HUB_DISABLE_TELEMETRY": "1"}


def job_env(tmp: Path, **extra: str) -> dict[str, str]:
    """The environment a job script sees here: no Slurm variables of this session, nothing from home."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_") and k not in ("HF_TOKEN", "PYTHONPATH")}
    env.update(scratch_caches(tmp / "cache"))
    env.update(
        LFD_REPO=str(REPO), LFD_PY=sys.executable, LFD_CACHE=str(tmp / "cache"), OUT_ROOT=str(tmp / "out"),
        CUDA_CACHE_PATH=str(tmp / "cache" / "nv"), LFD_HF_TOKEN_FILE=str(tmp / "no-token"), HF_HUB_OFFLINE="1",
        SLURM_CPUS_PER_TASK="12", SEED=SEED,
    )
    env.update(extra)
    return env


def mock_env(tmp: Path, mocks: Path, **extra: str) -> dict[str, str]:
    """A job environment whose Hub, lfs, squeue, nvidia-smi and GPU are the mocks; extra overrides any of it."""
    settings = dict(PATH=f"{mocks / 'bin'}:{os.environ['PATH']}", PYTHONPATH=str(mocks / "py"),
                    LFD_MOCK_HUB_LOG=str(mocks / "hub.log"), LFD_MOCK_HUB_SHA=MOCK_SHA, LFD_MOCK_SQUEUE="invalid",
                    LFD_MOCK_LFS_QUOTA=QUOTA_LINE.format(grace="3w4d6h18m48s"), LFD_MOCK_GPU=GPUS["a40"])
    settings.update(extra)
    return job_env(tmp, **settings)


def hub_calls(mocks: Path) -> list[dict]:
    log = mocks / "hub.log"
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.is_file() else []
    log.write_text("")
    return calls


def bash(snippet: str, env: dict[str, str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", f"source {shlex.quote(str(TRAIN / 'common.sh'))} && {snippet}"], env=env,
                          capture_output=True, text=True, timeout=timeout)


def run_job(script: str, env: dict[str, str], timeout: int = 900, **extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(TRAIN / script)], env={**env, **extra}, capture_output=True, text=True,
                          timeout=timeout)


def printed_args(script: Path, env: dict[str, str]) -> tuple[str, str, str, list[str]]:
    run = subprocess.run(["bash", str(script)], env={**env, "LFD_PRINT_ARGS": "1"}, capture_output=True,
                         text=True, timeout=300)
    if run.returncode:
        raise RuntimeError(f"{script.name} exited {run.returncode}: {run.stderr.strip()[-600:]}")
    lines = run.stdout.splitlines()
    action = next(line.split("=", 1)[1] for line in lines if line.startswith("LFD_ACTION="))
    run_dir = next(line.split("=", 1)[1] for line in lines if line.startswith("LFD_RUN_DIR="))
    pin = next(line.split("=", 1)[1] for line in lines if line.startswith("LFD_PRETRAINED="))
    begin, end = lines.index("LFD_TRAIN_ARGS_BEGIN"), lines.index("LFD_TRAIN_ARGS_END")
    return action, run_dir, pin, lines[begin + 1:end]


def parse_train_args(args: list[str]):
    """What lerobot-train does with these arguments, up to and including validate(), minus the training."""
    import draccus

    from lerobot.configs import parser as lr_parser
    from lerobot.configs.train import TrainPipelineConfig

    sys.argv = ["lerobot-train", *args]  # validate() reads --config_path and --policy.path from sys.argv
    config_path = lr_parser.parse_arg("config_path", args)
    if config_path:  # as lerobot.configs.parser.wrap does for a resume
        cfg = TrainPipelineConfig.from_pretrained(config_path, cli_args=lr_parser.filter_arg("config_path", args))
    else:
        cfg = draccus.parse(config_class=TrainPipelineConfig, args=args)
    cfg.validate()
    return cfg


def peft_base(adapter_dir: Path, revision: str | None) -> tuple[str | None, str]:
    """The base model LeRobot's make_policy loads for a PEFT checkpoint: PeftConfig.from_pretrained(path,
    revision=cfg.pretrained_revision).base_model_name_or_path. Without peft installed, its adapter_config.json."""
    if importlib.util.find_spec("peft") is not None:
        from peft import PeftConfig

        return PeftConfig.from_pretrained(str(adapter_dir), revision=revision).base_model_name_or_path, "PeftConfig"
    data = json.loads((adapter_dir / "adapter_config.json").read_text())
    return data.get("base_model_name_or_path"), "adapter_config.json (peft not installed here)"


def snapshot_of(tmp: Path, repo: str, sha: str = MOCK_SHA) -> Path:
    return tmp / "cache" / "huggingface" / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / sha


def view_of(tmp: Path, repo: str, sha: str = MOCK_SHA) -> Path:
    return tmp / "cache" / "huggingface" / "lfd-pins" / repo.replace("/", "--") / sha


# ---- 1. headers and syntax ------------------------------------------------------------------------------
def sbatch_options(path: Path) -> dict[str, str]:
    opts: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if line.startswith("#SBATCH"):
            for token in shlex.split(line[len("#SBATCH"):]):
                key, _, value = token.lstrip("-").partition("=")
                opts[key] = value
    return opts


def minutes(limit: str) -> float:
    days, _, clock = limit.rpartition("-")
    parts = [int(p) for p in clock.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return int(days or 0) * 1440 + parts[0] * 60 + parts[1] + parts[2] / 60


def check_headers(tmp: Path) -> None:
    for name in ("act.sbatch", "pi05.sbatch", "groot.sbatch"):
        opts = sbatch_options(TRAIN / name)
        logs_ok = all(opts.get(k, "").startswith("/nfs/hpc/share/%u/") for k in ("output", "error"))
        check(opts.get("account") == "eecs" and opts.get("partition") == "ampere" and opts.get("gres") == "gpu:1"
              and minutes(opts.get("time", "9-00:00:00")) <= 1440 and logs_ok,
              f"{name} header: account {opts.get('account')}, partition {opts.get('partition')}, "
              f"time {opts.get('time')} (<= 1 day: two such jobs fit the 2880 GPU-minute QoS cap), "
              f"gres {opts.get('gres')}, logs {opts.get('output')}")
    pi05, readme = (TRAIN / "pi05.sbatch").read_text(), (TRAIN / "README.md").read_text()
    full = re.search(r"\n\s*full\)(.*?);;", pi05, re.S)
    min_gb = int(re.search(r"MIN_GPU_MEM_GB=(\d+)", full.group(1)).group(1)) if full else 0
    check(min_gb >= 79 and "--constraint=vram80g" in pi05 and "--constraint=vram80g" in readme
          and "h100|h200" not in pi05 + readme,
          f"π0.5 full: MIN_GPU_MEM_GB={min_gb} and --constraint=vram80g (dgxh's full GPUs have no gres type)")
    for path in [TRAIN / "common.sh", *(TRAIN / n for n in ("act.sbatch", "pi05.sbatch", "groot.sbatch"))]:
        run = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        check(run.returncode == 0, f"bash -n {path.name} {run.stderr.strip()}")
    # as `sbatch --wrap` runs it (README, "Convert"): /bin/sh, which is bash in POSIX mode here
    run = subprocess.run(["sh", "-c", f". {shlex.quote(str(TRAIN / 'common.sh'))} && type lfd_main >/dev/null && "
                          'echo "$WANDB_MODE $WANDB_DIR"'], env=job_env(tmp), capture_output=True, text=True)
    check(run.returncode == 0 and run.stdout.split()[:1] == ["disabled"]
          and not run.stdout.split()[1].startswith(os.path.expanduser("~")),
          f"common.sh sources under sh; WANDB_MODE and WANDB_DIR: {run.stdout.strip()} {run.stderr.strip()[:200]}")


# ---- 2. the arguments each job passes --------------------------------------------------------------------
def make_fake_dataset(root: Path, name: str = "fake_dataset") -> None:
    (root / "meta").mkdir(parents=True)
    vec = {"dtype": "float32", "shape": [12]}
    cam = {"dtype": "video", "shape": [96, 128, 3]}
    features = {"observation.state": vec, "action": vec, "observation.images.left": cam,
                "observation.images.right": cam}
    (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": 6, "total_frames": 600, "fps": 30,
                                                         "features": features, "robot_type": "bhl_arms"}))
    stats = {k: {"min": [0.0] * 12, "max": [1.0] * 12, "mean": [0.5] * 12, "std": [0.1] * 12, "q01": [0.0] * 12,
                 "q99": [1.0] * 12, "count": [600]} for k in ("observation.state", "action")}
    (root / "meta" / "stats.json").write_text(json.dumps(stats))
    (root / "conversion.json").write_text(json.dumps({"format": "bhl-lfd-conversion", "label": "meas_future",
                                                      "lookahead_s": 0.1, "stats_recomputed": True,
                                                      "dataset": {"repo_id": f"local/{name}"}}, indent=1))
    episodes = {str(i): {"source": f"20261005_120000_robot/ep_{i:04d}", "mode": "teleop",
                         "split": "val" if i in FAKE_SPLITS["val"] else "train"} for i in range(6)}
    (root / "splits.json").write_text(json.dumps({**FAKE_SPLITS, "episodes": episodes}))


def check_variants(tmp: Path, mocks: Path) -> None:
    fake = tmp / "fake_dataset"
    out = tmp / "out"
    train, val = FAKE_SPLITS["train"], FAKE_SPLITS["val"]
    pi05_snap, groot_view = snapshot_of(tmp, "lerobot/pi05_base"), view_of(tmp, "nvidia/GR00T-N1.7-3B")
    variants = [  # name, script, extra env, the pinned path it must pass, policy checks
        ("ACT", "act.sbatch", {}, None, lambda c: (
            c.policy.type == "act" and c.policy.pretrained_path is None and c.peft is None
            and c.policy.chunk_size == 100 and c.steps == 100000 and c.batch_size == 8
            and c.policy.pretrained_backbone_weights == "ResNet18_Weights.IMAGENET1K_V1")),
        ("ACT smoke (CPU test settings)", "act.sbatch", {"LFD_SMOKE": "1"}, None, lambda c: (
            c.policy.type == "act" and c.steps <= 30 and c.batch_size == 2 and c.save_freq == 10
            and c.policy.pretrained_backbone_weights is None and c.dataset.image_transforms.enable
            and c.dataset.image_transforms.tfs["resize"].type == "Resize")),
        ("pi05 LoRA", "pi05.sbatch", {}, pi05_snap, lambda c: (
            c.policy.type == "pi05" and Path(c.policy.pretrained_path) == pi05_snap
            and c.peft is not None and c.peft.method_type == "LORA" and c.peft.r == 32 and c.peft.lora_alpha == 32
            and c.policy.dtype == "bfloat16" and c.policy.gradient_checkpointing and c.policy.chunk_size == 50
            and math.isclose(c.optimizer.lr, 2.5e-4) and c.scheduler.num_decay_steps == c.steps == 30000
            and c.policy.normalization_mapping["STATE"].value == "QUANTILES" and not c.policy.use_relative_actions)),
        ("pi05 full fine-tuning", "pi05.sbatch", {"PI05_MODE": "full"}, pi05_snap, lambda c: (
            c.policy.type == "pi05" and Path(c.policy.pretrained_path) == pi05_snap and c.peft is None
            and math.isclose(c.optimizer.lr, 2.5e-5) and not c.policy.train_expert_only
            and not c.policy.freeze_vision_encoder)),
        ("GR00T N1.7", "groot.sbatch", {}, groot_view, lambda c: (
            c.policy.type == "groot" and Path(c.policy.base_model_path) == groot_view
            and "gr00t-n1.7" in c.policy.base_model_path.lower()
            and c.policy.pretrained_path is None and c.peft is None and c.policy.embodiment_tag == "new_embodiment"
            and c.policy.chunk_size == 40 and len(c.policy.action_delta_indices) == 40
            and c.policy.max_steps == c.steps == 30000 and c.scheduler.num_warmup_steps == 1500
            and c.policy.use_bf16 and not c.policy.tune_llm and not c.policy.tune_visual
            and c.policy.tune_diffusion_model and not c.policy.use_relative_actions)),
    ]
    for name, script, extra, pinned, policy_ok in variants:
        env = mock_env(tmp, mocks, DATASET_ROOT=str(fake), **extra)
        hub_calls(mocks)
        try:
            action, run_dir, pin, args = printed_args(TRAIN / script, env)
            cfg = parse_train_args(args)
        except Exception as exc:  # noqa: BLE001 - any failure here is the finding
            check(False, f"{name}: the job's lerobot-train arguments do not parse: {exc}")
            continue
        save_freq_ok = cfg.save_freq == (10 if extra.get("LFD_SMOKE") else 1000)
        common_ok = (
            action == "fresh" and cfg.dataset.episodes == train and not set(val) & set(cfg.dataset.episodes)
            and save_freq_ok and cfg.seed == int(SEED) and cfg.policy.push_to_hub is False
            and cfg.dataset.root == str(fake) and cfg.dataset.video_backend == "pyav" and not cfg.wandb.enable
            and cfg.rename_map == {} and cfg.dataset.eval_split == 0.0
            and str(cfg.output_dir) == f"{run_dir}/train" and run_dir.startswith(str(out))
            and ("--policy.device=cpu" if extra.get("LFD_SMOKE") else "--policy.device=cuda") in args
        )
        check(common_ok, f"{name}: train episodes {cfg.dataset.episodes} (val {val} absent), seed {cfg.seed}, "
                         f"save_freq {cfg.save_freq}, output {cfg.output_dir}")
        check(bool(policy_ok(cfg)), f"{name}: policy {type(cfg.policy).__name__} chunk "
                                    f"{getattr(cfg.policy, 'chunk_size', None)}, steps {cfg.steps}, batch "
                                    f"{cfg.batch_size}, peft {cfg.peft is not None}, optimizer lr {cfg.optimizer.lr}")
        calls = hub_calls(mocks)
        if pinned is None:
            check(pin == "" and not calls, f"{name}: no pretrained weights to pin (hub calls {calls})")
        else:
            record = json.loads(pin)
            check(record["revision"] == MOCK_SHA and Path(record["path"]) == pinned
                  and [c["call"] for c in calls] == ["model_info"],
                  f"{name}: pinned to the commit main points to ({record['revision'][:12]}), passed as the local "
                  f"{'view' if record['hidden'] else 'snapshot'} {Path(record['path']).relative_to(tmp)}; print "
                  f"mode resolves without downloading (hub calls: {[c['call'] for c in calls]})")
        if extra.get("LFD_SMOKE"):
            check("-smoke/" in run_dir, f"{name}: smoke runs get their own folder ({run_dir})")


# ---- 3. each job end to end, on mocks ---------------------------------------------------------------------
def flow_env(tmp: Path, mocks: Path, run_name: str, **extra: str) -> dict[str, str]:
    settings = dict(DATASET_ROOT=str(tmp / "fake_dataset"), LEROBOT_ENV=str(tmp / "fake-gpu-env"),
                    HF_TOKEN="hf_mock_token_for_tests", RUN_NAME=run_name, STEPS="1000", LFD_SKIP_EVAL="1")
    settings.update(extra)
    return mock_env(tmp, mocks, **settings)


@contextlib.contextmanager
def edited(path: Path, change):
    """Change a file for the duration of a check, then put the original bytes back."""
    original = path.read_bytes()
    try:
        path.write_text(change(path.read_text()))
        yield
    finally:
        path.write_bytes(original)


def check_job_flows(tmp: Path, mocks: Path) -> None:
    fake = tmp / "fake_dataset"
    out = tmp / "out"
    flows = [  # name, script, extra env, flag, pinned path, KEEP_LAST default
        ("ACT", "act.sbatch", {}, None, None, 2),
        ("pi05 LoRA", "pi05.sbatch", {}, "--policy.pretrained_path", snapshot_of(tmp, "lerobot/pi05_base"), 2),
        ("pi05 full", "pi05.sbatch", {"PI05_MODE": "full", "LFD_MOCK_GPU": GPUS["h100"]}, "--policy.pretrained_path",
         snapshot_of(tmp, "lerobot/pi05_base"), 1),
        ("GR00T", "groot.sbatch", {}, "--policy.base_model_path", view_of(tmp, "nvidia/GR00T-N1.7-3B"), 1),
    ]
    runs: dict[str, Path] = {}
    for name, script, extra, flag, pinned, keep in flows:
        tag = name.lower().replace(" ", "_")
        env = flow_env(tmp, mocks, f"flow/{tag}", **extra)
        run_dir = out / "flow" / tag
        runs[name] = run_dir
        hub_calls(mocks)
        fresh = run_job(script, env)
        log = fresh.stdout + fresh.stderr
        if not check(fresh.returncode == 0 and "FAKE_TRAIN preconditions ok" in log
                     and "training reached step 1000 of 1000" in log and "run folder now holds" in log,
                     f"{name} job, fresh: exit {fresh.returncode}; inputs.json and the job record held the pinned "
                     f"revision before lerobot-train started, and the job went on past training to its end"):
            print(tail(fresh, 3000))
            continue
        inputs = json.loads((run_dir / "inputs.json").read_text())
        records = [json.loads(p.read_text()) for p in sorted((run_dir / "jobs").glob("*.json"))]
        identity = json.loads(subprocess.run([sys.executable, str(OFFLINE_EVAL), "--dataset-root", str(fake),
                                              "--print-identity"], capture_output=True, text=True).stdout)
        check(all(inputs.get(k) == identity.get(k) for k in ("conversion_sha256", "stats_sha256", "splits_sha256",
                                                              "label", "lookahead_s"))
              and records[-1].get("inputs") == inputs,
              f"{name}: inputs.json records the dataset by content (conversion {inputs['conversion_sha256'][:12]}, "
              f"label {inputs['label']}, lookahead {inputs['lookahead_s']}), and the job record carries it")
        calls = hub_calls(mocks)
        train_args = (run_dir / "fake_train_args.txt").read_text().splitlines()
        if pinned is not None:
            pin = inputs["pretrained"]
            check(pin["revision"] == MOCK_SHA and Path(pin["path"]) == pinned and f"{flag}={pinned}" in train_args
                  and records[-1]["pretrained"] == pin
                  and [(c["call"], c["revision"]) for c in calls] == [("model_info", None),
                                                                       ("snapshot_download", MOCK_SHA)],
                  f"{name}: resolved main to {MOCK_SHA[:12]} (HfApi().model_info), downloaded that commit, recorded "
                  f"repo, revision and path, and trained from {flag}={Path(pin['path']).relative_to(tmp)}")
        check("share (project 30762)" in log and "OVER the 1536 GiB soft quota" in log and "grace ends in" in log
              and ("gpu: " in log),
              f"{name}: the job printed the soft-quota status and grace, and the GPU it got: "
              + "; ".join(line.split("] ", 1)[-1][:90] for line in log.splitlines() if "share (project" in line))
        if name == "pi05 LoRA":
            model = run_dir / "train" / "checkpoints" / "last" / "pretrained_model"
            try:
                _, _, _, args2 = printed_args(TRAIN / script, env)
                cfg2 = parse_train_args(args2)
                base, how = peft_base(Path(cfg2.policy.pretrained_path), cfg2.policy.pretrained_revision)
                check(cfg2.resume and cfg2.policy.use_peft is True and cfg2.peft is not None
                      and Path(cfg2.policy.pretrained_path) == model and base == str(pinned),
                      f"{name}: a LoRA checkpoint (use_peft true, adapter_config.json) resumes through LeRobot's PEFT "
                      f"path: {how} gives the base {Path(base).relative_to(tmp) if base else base}, the pinned "
                      "snapshot")
            except Exception as exc:  # noqa: BLE001
                check(False, f"{name}: the LoRA resume arguments do not parse: {exc}")
        # the Hub's main moves on; a resume keeps the recorded commit
        resumed = run_job(script, env, STEPS="2000", LFD_EXTEND_STEPS="1", LFD_MOCK_HUB_SHA=OTHER_SHA)
        log = resumed.stdout + resumed.stderr
        calls = hub_calls(mocks)
        kept = sorted(p.name for p in (run_dir / "train" / "checkpoints").iterdir())
        want_kept = ["001000", "002000", "last"] if keep == 2 else ["002000", "last"]
        ok = resumed.returncode == 0 and "resume gate passed" in log and kept == want_kept
        if pinned is not None:
            ok = ok and [(c["call"], c["revision"]) for c in calls] == [("snapshot_download", MOCK_SHA)] \
                and f"{flag}={pinned}" not in (run_dir / "fake_train_args.txt").read_text()  # --config_path only
        if not check(ok, f"{name}: resubmitted with STEPS=2000 and LFD_EXTEND_STEPS=1 after the Hub's main moved: "
                         f"exit {resumed.returncode}, kept {kept} (KEEP_LAST {keep}), hub calls "
                         f"{[(c['call'], (c['revision'] or '')[:12]) for c in calls]}"):
            print(tail(resumed, 3000))
            continue
        states = {d.name: (d / "training_state" / "optimizer_state.safetensors").is_file()
                  for d in sorted((run_dir / "train" / "checkpoints").glob("0*"))}
        check(states.get("002000") is True and states.get("001000", False) is False
              and all((run_dir / "train" / "checkpoints" / s / "training_state" / "training_step.json").is_file()
                      for s in states),
              f"{name}: the finished run kept the optimizer state of last (002000) only: {states}")

    # the content gate, on the LoRA run: every change is refused, LFD_EXTEND_STEPS or not
    name, script = "pi05 LoRA", "pi05.sbatch"
    run_dir = runs[name]
    if (run_dir / "train" / "checkpoints" / "002000").is_dir():
        env = flow_env(tmp, mocks, "flow/pi05_lora", STEPS="3000", LFD_EXTEND_STEPS="1")
        model = run_dir / "train" / "checkpoints" / "002000" / "pretrained_model"
        cases = [
            ("conversion.json changed", fake / "conversion.json", lambda t: t + "\n", "conversion_sha256"),
            ("meta/stats.json changed", fake / "meta" / "stats.json", lambda t: t.replace("0.5", "0.6", 1),
             "stats_sha256"),
            ("splits.json changed, same train list", fake / "splits.json",
             lambda t: json.dumps({**json.loads(t), "note": "edited"}), "splits_sha256"),
            ("the label changed", fake / "conversion.json", lambda t: t.replace("meas_future", "cmd"), "label"),
            ("the lookahead changed", fake / "conversion.json", lambda t: t.replace("0.1", "0.2"), "lookahead_s"),
            ("the LoRA adapter's base changed", model / "adapter_config.json",
             lambda t: t.replace(MOCK_SHA, OTHER_SHA), "LoRA adapter's base model"),
            ("inputs.json lost", run_dir / "inputs.json", None, "is missing"),
        ]
        for what, path, change, needle in cases:
            if change is None:
                original = path.read_bytes()
                path.unlink()
                refused = run_job(script, env)
                path.write_bytes(original)
            else:
                with edited(path, change):
                    refused = run_job(script, env)
            check(refused.returncode != 0 and "refusing to resume" in refused.stderr and needle in refused.stderr
                  and not (run_dir / "train" / "checkpoints" / "003000").exists(),
                  f"resume refused when {what}, even with LFD_EXTEND_STEPS=1 ({needle}): exit {refused.returncode}")
        refused = run_job(script, env, PRETRAINED_REVISION=OTHER_SHA)
        check(refused.returncode != 0 and "PRETRAINED_REVISION" in refused.stderr,
              f"resume refused when PRETRAINED_REVISION names another commit: exit {refused.returncode}")
        refused = run_job(script, {**env, "LFD_EXTEND_STEPS": "0"})
        check(refused.returncode != 0 and "LFD_EXTEND_STEPS" in refused.stderr,
              "resume refused when STEPS changes without LFD_EXTEND_STEPS=1")
    hub_calls(mocks)

    # π0.5 full on a 40 GB MIG slice: refused at the start, before any download or record
    env = flow_env(tmp, mocks, "flow/pi05_full_mig", PI05_MODE="full", LFD_MOCK_GPU=GPUS["mig"])
    mig = run_job("pi05.sbatch", env)
    check(mig.returncode != 0 and "GPU check failed" in mig.stderr and "vram80g" in mig.stderr
          and not (out / "flow" / "pi05_full_mig" / "inputs.json").exists() and not hub_calls(mocks),
          f"π0.5 full refuses a 40 GB MIG slice (MIN_GPU_MEM_GB=79): {mig.stderr.strip().splitlines()[-2:]}")
    # a low grace: refused before any download
    env = flow_env(tmp, mocks, "flow/act_low_grace", LFD_MOCK_LFS_QUOTA=QUOTA_LINE.format(grace="6d23h59m59s"))
    low = run_job("act.sbatch", env)
    check(low.returncode != 0 and "share quota check failed" in low.stderr
          and not (out / "flow" / "act_low_grace" / "train").exists(),
          f"a job refuses to start over the soft quota with under 7 days of grace: "
          f"{low.stderr.strip().splitlines()[:1]}")

    # the run lock: a second job refuses while a live job holds it; a stale holder is taken over
    run_dir = out / "flow" / "act_lock"
    run_dir.mkdir(parents=True)
    lock = run_dir / ".lock"
    for squeue, holder, want in (("RUNNING", "job=4242 host=elsewhere pid=1", "refuse"),
                                 ("PENDING", "job=4242 host=elsewhere pid=1", "refuse"),
                                 ("error", "job=4242 host=elsewhere pid=1", "refuse"),
                                 ("invalid", "job=4242 host=elsewhere pid=1", "take over"),
                                 ("COMPLETED", "job=4243 host=elsewhere pid=1", "take over"),
                                 ("invalid", f"job=local-1 host={os.uname().nodename} pid=999999999", "take over")):
        holding = subprocess.Popen(["bash", "-c", f"exec 9>>{shlex.quote(str(lock))}; flock -n 9 || exit 3; "
                                    f"printf '%s\\n' {shlex.quote(holder)} >{shlex.quote(str(lock))}; "
                                    "echo held; exec sleep 120"], stdout=subprocess.PIPE, text=True)
        try:
            held = holding.stdout.readline().strip() == "held"
            env = flow_env(tmp, mocks, "flow/act_lock", LFD_MOCK_SQUEUE=squeue, LFD_MODE="eval")
            second = run_job("act.sbatch", env)
        finally:
            holding.kill()
            holding.wait()
        log = second.stdout + second.stderr
        if want == "refuse":
            ok = held and second.returncode != 0 and "another job holds" in second.stderr \
                and "has no checkpoint to evaluate" not in log
        else:
            ok = held and "taking over a stale lock" in log and "has no checkpoint to evaluate" in log \
                and any(run_dir.glob(".lock.stale.*"))
        check(ok, f"run lock held by '{holder.split()[0]}', squeue says {squeue}: {want} "
                  f"({(second.stderr.strip().splitlines() or [''])[-1][-110:]})")
    first = run_job("act.sbatch", flow_env(tmp, mocks, "flow/act_lock", LFD_MODE="eval"))
    content = lock.read_text() if lock.is_file() else ""
    check("has no checkpoint to evaluate" in first.stderr and "another job holds" not in first.stderr
          and content.startswith("job=local-") and f"host={os.uname().nodename}" in content,
          f"with no other holder the job takes the lock and names itself in it: {content.strip()}")


# ---- 4. pinned views, pruning, quota, GPU, on their own ---------------------------------------------------
def check_groot_view(tmp: Path, mocks: Path) -> None:
    """GR00T's pinned view must give LeRobot's Hub-id processor; the raw snapshot would not."""
    env = mock_env(tmp, mocks, PRETRAINED_HIDE=" ".join(HIDDEN))
    run = bash(f"lfd_pretrained nvidia/GR00T-N1.7-3B {MOCK_SHA} 1", env)
    pin = json.loads(run.stdout.strip().splitlines()[-1]) if run.returncode == 0 else {}
    view, snap = Path(pin.get("path", "/missing")), Path(pin.get("snapshot", "/missing"))
    check(run.returncode == 0 and view == view_of(tmp, "nvidia/GR00T-N1.7-3B") and (view / "config.json").is_symlink()
          and (view / "model.safetensors").is_file() and not any((view / h).exists() for h in HIDDEN)
          and all((snap / h).is_file() for h in HIDDEN),
          f"the pinned GR00T view links the snapshot's files but {', '.join(HIDDEN)}: "
          f"{sorted(p.name for p in view.iterdir()) if view.is_dir() else run.stderr.strip()[-200:]}")
    try:
        import torch

        import lerobot.policies.groot.processor_groot as pg
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.processor import ProcessorStep
    except Exception as exc:  # noqa: BLE001
        check(False, f"GR00T processor check: cannot import LeRobot's GR00T processor here: {exc}")
        return

    class NoVLM(ProcessorStep):  # GrootN17VLMEncodeStep would load the gated Cosmos-Reason2 processor
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __call__(self, transition):
            return transition

        def transform_features(self, features):
            return features

    stats = {k: {s: torch.zeros(12) if s != "max" else torch.ones(12) for s in ("min", "max", "mean", "std")}
             for k in ("observation.state", "action")}
    seen = {}
    real_vlm, cwd = pg.GrootN17VLMEncodeStep, os.getcwd()
    pg.GrootN17VLMEncodeStep = NoVLM
    try:
        os.chdir(tmp)  # as in a job, where the cwd is the run folder: no nvidia/ folder beside it
        for label, base in (("hub id", "nvidia/GR00T-N1.7-3B"), ("raw snapshot", str(snap)), ("view", str(view))):
            cfg = argparse.Namespace(
                base_model_path=base, embodiment_tag="new_embodiment", use_relative_actions=False, chunk_size=40,
                max_state_dim=132, max_action_dim=132, device="cpu", action_decode_transform=None,
                relative_exclude_joints=[],
                output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(12,))})
            pre, post = pg.make_groot_pre_post_processors(cfg, dataset_stats=stats, dataset_meta=argparse.Namespace())
            pack = next(s for s in pre.steps if isinstance(s, pg.GrootN17PackInputsStep))
            vlm = next(s for s in pre.steps if isinstance(s, NoVLM)).kwargs
            seen[label] = (pack.embodiment_mapping.get(pack.embodiment_tag, 0), pack.state_dropout_prob,
                           pack.use_percentiles, vlm["use_albumentations"], vlm["shortest_image_edge"],
                           vlm["crop_fraction"], tuple(vlm["image_target_size"]), tuple(vlm["image_crop_size"]),
                           type(post.steps[0]).__name__)
    except Exception as exc:  # noqa: BLE001
        check(False, f"GR00T processor check failed to build the processors: {exc}")
        return
    finally:
        pg.GrootN17VLMEncodeStep = real_vlm
        os.chdir(cwd)
    check(seen["view"] == seen["hub id"] and seen["hub id"][:2] == (10, 0.0) and seen["raw snapshot"][:2] == (0, 0.2),
          "GR00T: LeRobot builds the same processor from the pinned view as from the Hub id (embodiment slot, "
          f"state dropout, percentiles, albumentations, edge, crop fraction, sizes, decoder): {seen['view']}; "
          f"the raw snapshot would give {seen['raw snapshot']}")


def check_helpers(tmp: Path, mocks: Path) -> None:
    ds = tmp / "splits_cases"
    cases = {
        "good": (FAKE_SPLITS, 0, "[0,1,2,4]"),
        "overlap": ({"train": [0, 1, 3], "val": [3]}, 2, ""),
        "out of range": ({"train": [0, 9], "val": [3]}, 2, ""),
        "no val list": ({"train": [0, 1]}, 2, ""),
        "names, not indices": ({"train": ["ep_0000"], "val": [3]}, 2, ""),
    }
    for name, (splits, want_rc, want_out) in cases.items():
        root = ds / name.replace(" ", "_").replace(",", "")
        make_fake_dataset(root)
        (root / "splits.json").write_text(json.dumps(splits))
        run = subprocess.run([sys.executable, str(OFFLINE_EVAL), "--dataset-root", str(root), "--print-episodes",
                              "train"], capture_output=True, text=True, timeout=60)
        check(run.returncode == want_rc and run.stdout.strip() == want_out,
              f"splits.json {name}: exit {run.returncode} {run.stdout.strip() or run.stderr.strip()[:110]}")
    good = ds / "good"
    env = job_env(tmp)
    run = bash(f"lfd_train_episodes {shlex.quote(str(good))} && lfd_val_episodes {shlex.quote(str(good))}", env)
    check(run.stdout.split() == ["[0,1,2,4]", "[3,5]"],
          f"common.sh lfd_train_episodes / lfd_val_episodes print {run.stdout.split()} {run.stderr.strip()[:200]}")
    run = subprocess.run(["bash", "-c", f"source {shlex.quote(str(TRAIN / 'common.sh'))} && "
                          "printf '%s\\n' \"$HF_HOME\" \"$XDG_CACHE_HOME\" \"$TORCH_HOME\" \"$PIP_CACHE_DIR\" "
                          "\"$TRITON_CACHE_DIR\" \"$CUDA_CACHE_PATH\" \"$WANDB_DIR\" \"$TORCHINDUCTOR_CACHE_DIR\" "
                          "\"$LFD_PINS\" \"$WANDB_MODE\""],
                         env={k: v for k, v in env.items() if k not in ("LFD_CACHE", "CUDA_CACHE_PATH")},
                         capture_output=True, text=True, timeout=60)
    home = os.path.expanduser("~")
    paths = run.stdout.split()
    check(len(paths) == 10 and all(not p.startswith(home) and not p.startswith("/nfs/stak") for p in paths[:9])
          and paths[0] == f"/nfs/hpc/share/{os.environ.get('USER')}/.cache/huggingface" and paths[9] == "disabled",
          f"common.sh puts every cache on the share or scratch, never home, and disables wandb: {paths}")

    # pruning: what is kept, whatever the state of the checkpoint folders
    def make_checkpoints(root: Path, complete: list[int], half: list[int], last: int | None) -> Path:
        ck = root / "checkpoints"
        for step in complete + half:
            model = ck / f"{step:06d}" / "pretrained_model"
            model.mkdir(parents=True)
            if step in complete:
                (model / "model.safetensors").write_bytes(b"0")
                (model / "train_config.json").write_text("{}")
            (ck / f"{step:06d}" / "training_state").mkdir()
        if last is not None:
            (ck / "last").symlink_to(f"{last:06d}")
        return ck

    prune_cases = [  # name, complete, half-written, last, keep, kept
        ("keep 2, last older", [1000, 2000, 3000, 4000, 5000], [], 3000, 2,
         {"003000", "004000", "005000", "last"}),
        ("keep 2, last newest", [1000, 2000, 3000, 4000, 5000], [], 5000, 2, {"004000", "005000", "last"}),
        ("keep 1, a save in progress", [1000, 2000, 3000, 4000], [5000], 4000, 1, {"004000", "005000", "last"}),
        ("keep 1, a killed job's leftover below the newest", [1000, 3000, 4000], [2000], 4000, 1,
         {"004000", "last"}),
        ("keep 1, the newest written before last moved", [1000, 2000, 3000], [], 2000, 1,
         {"002000", "003000", "last"}),
        ("no complete checkpoint at all", [], [1000, 2000, 3000], None, 1, {"001000", "002000", "003000"}),
    ]
    for name, complete, half, last, keep, want in prune_cases:
        ck = make_checkpoints(tmp / "prune" / name.replace(" ", "_").replace(",", ""), complete, half, last)
        run = bash(f"lfd_prune_checkpoints {shlex.quote(str(ck))} {keep}", env)
        left = {p.name for p in ck.iterdir()}
        check(left == want, f"pruning ({name}): keeps {sorted(left)}")

    # a delete that fails is logged, the rest is still pruned, and the caller's errexit does not kill the loop
    ck = make_checkpoints(tmp / "prune" / "undeletable", [1000, 2000, 3000, 4000], [], 4000)
    stuck = ck / "001000" / "stuck"
    stuck.mkdir()
    (stuck / "f").write_text("x")
    stuck.chmod(0o500)
    try:
        run = subprocess.run(["bash", "-c", f"set -euo pipefail; source {shlex.quote(str(TRAIN / 'common.sh'))}; "
                              "for pass in 1 2; do lfd_prune_checkpoints \"$1\" 1 || lfd_log \"pass $pass left "
                              "something\"; done; echo survived", "x", str(ck)], env=env, capture_output=True,
                             text=True, timeout=60)
    finally:
        stuck.chmod(0o700)
    left = {p.name for p in ck.iterdir()}
    check(run.returncode == 0 and "survived" in run.stdout and "WARNING: could not delete" in run.stdout
          and "pass 2 left something" in run.stdout and left == {"001000", "004000", "last"},
          f"pruning goes on past a failed delete, logs it and retries it: kept {sorted(left)}")
    shutil.rmtree(ck, ignore_errors=True)

    # after a finished run only last keeps its optimizer state
    ck = make_checkpoints(tmp / "dropopt", [1000, 2000], [], 2000)
    for step in ("001000", "002000"):
        for name in ("optimizer_state.safetensors", "optimizer_param_groups.json", "training_step.json"):
            (ck / step / "training_state" / name).write_text("0")
    run = bash(f"lfd_drop_optimizer_state {shlex.quote(str(ck))}", env)
    have = {p.relative_to(ck).as_posix() for p in ck.glob("0*/training_state/*")}  # not through `last`
    check(have == {"001000/training_state/training_step.json", "002000/training_state/training_step.json",
                   "002000/training_state/optimizer_state.safetensors",
                   "002000/training_state/optimizer_param_groups.json"},
          f"lfd_drop_optimizer_state keeps last's optimizer state and every training_step.json: {sorted(have)}")

    # the quota logic, on mocked lfs output
    grace_ok = bash("lfd_grace_seconds 3w4d6h18m48s; lfd_grace_seconds 08m09s; lfd_grace_seconds none || echo no", env)
    check(grace_ok.stdout.split() == ["2182728", "489", "no"],
          f"lfd_grace_seconds reads lfs's grace (3w4d6h18m48s, 08m09s, none): {grace_ok.stdout.split()}")
    two_line = "/nfs/hpc/share/a/very/long/path\\n 1630827124* 1610612736 2147483648 3w4d6h18m48s 2066352* 0 0 -"
    quota_cases = [  # name, lfs line, need GiB, extra env, rc, text it must print
        ("over soft, 25 days of grace", QUOTA_LINE.format(grace="3w4d6h18m48s"), 51, {}, 0, "25 days 6 h"),
        ("over soft, under 7 days of grace", QUOTA_LINE.format(grace="6d23h59m59s"), 1, {}, 1, "under 7 days"),
        ("the same with LFD_ALLOW_LOW_GRACE=1", QUOTA_LINE.format(grace="6d23h59m59s"), 1,
         {"LFD_ALLOW_LOW_GRACE": "1"}, 0, "WARNING: starting with under 7 days"),
        ("grace expired, even with LFD_ALLOW_LOW_GRACE=1", QUOTA_LINE.format(grace="none"), 1,
         {"LFD_ALLOW_LOW_GRACE": "1"}, 1, "grace has expired"),
        ("over soft, grace unreadable", QUOTA_LINE.format(grace="-"), 1, {}, 1, "grace (-) is under 7 days"),
        ("crossing the hard limit", QUOTA_LINE.format(grace="3w4d6h18m48s"), 500, {}, 1, "cross the 2048 GiB limit"),
        ("under soft", "  /nfs/hpc/share 1500000000 1610612736 2147483648 - 2066352 0 0 -", 51, {}, 0,
         "below the 1536 GiB soft quota"),
        ("under soft, but this run would cross it", "  /nfs/hpc/share 1500000000 1610612736 2147483648 - 1 0 0 -",
         200, {}, 0, "start the grace clock"),
        ("the path on its own line", two_line, 51, {}, 0, "OVER the 1536 GiB soft quota by 19 GiB"),
        ("lfs output unreadable", "lfs: error", 1, {}, 1, "could not read the share quota"),
        ("unreadable, LFD_SKIP_QUOTA_CHECK=1", "lfs: error", 1, {"LFD_SKIP_QUOTA_CHECK": "1"}, 0, "skipped"),
    ]
    for name, line, need, extra, want_rc, text in quota_cases:
        run = bash(f"lfd_quota_check {need}", mock_env(tmp, mocks, LFD_MOCK_LFS_QUOTA=line, **extra))
        said = run.stdout + run.stderr
        check(run.returncode == want_rc and text in said,
              f"quota ({name}, {need} GiB): exit {run.returncode}, says '{text}'")
    run = bash("lfd_quota_check 0", job_env(tmp))  # the real lfs, read-only
    said = (run.stdout + run.stderr).strip()
    check("share (project" in said or "no lfs command" in said,
          f"lfd_quota_check reads the real share quota (read-only): exit {run.returncode}: "
          f"{said.splitlines()[0][-170:] if said else ''}")

    # the GPU check on mocked devices
    gpu_env = mock_env(tmp, mocks, LEROBOT_ENV=str(tmp / "fake-gpu-env"))
    for gpu, need_bf16, min_gb, want_rc, text in (("h100", 1, 79, 0, "79.6 GiB"), ("mig", 1, 79, 1, "vram80g"),
                                                  ("a40", 1, 40, 0, "45.0 GiB"), ("a40", 1, 79, 1, "at least 79"),
                                                  ("rtx8000", 1, 40, 1, "no native bf16")):
        run = bash(f"lfd_check_gpu {need_bf16} {min_gb}", {**gpu_env, "LFD_MOCK_GPU": GPUS[gpu]})
        check(run.returncode == want_rc and text in run.stdout + run.stderr,
              f"GPU check, {GPUS[gpu].split('|')[0]} with MIN_GPU_MEM_GB={min_gb}: exit {run.returncode} ('{text}')")


# ---- 5. offline_eval.py refuses another dataset ---------------------------------------------------------
def check_leakage(tmp: Path) -> None:
    fake = tmp / "fake_dataset"
    run_dir = tmp / "out" / "flow" / "act"
    ckpt = run_dir / "train" / "checkpoints" / "last" / "pretrained_model"
    if not ckpt.is_dir():
        check(False, "offline_eval leakage checks: the ACT job flow left no checkpoint to check against")
        return

    def score(dataset: Path, checkpoint: Path = ckpt, *more: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(OFFLINE_EVAL), "--checkpoint", str(checkpoint), "--dataset-root",
                               str(dataset), "--out", str(tmp / "leak" / "report.json"), "--device", "cpu", *more],
                              capture_output=True, text=True, timeout=600)

    other = tmp / "leak" / "other_dataset"
    shutil.copytree(fake, other)
    conversion = json.loads((other / "conversion.json").read_text())
    (other / "conversion.json").write_text(json.dumps({**conversion, "args": {"seed": 1}}, indent=1))
    lonely = tmp / "leak" / "lonely" / "pretrained_model"  # the checkpoint without its run folder
    shutil.copytree(ckpt, lonely)
    refused = "refusing to score this checkpoint on this dataset"
    run = score(other)
    check(run.returncode != 0 and refused in run.stderr and "different dataset" in run.stderr,
          f"offline_eval refuses a checkpoint trained on a different conversion.json: "
          f"{run.stderr.strip().splitlines()[-1][:150] if run.stderr.strip() else run.returncode}")
    run = score(fake, lonely)
    check(run.returncode != 0 and refused in run.stderr and "unknown training dataset" in run.stderr,
          "offline_eval refuses a checkpoint whose training dataset it cannot find (no inputs.json)")
    run = score(fake)
    check(refused not in run.stderr and "refusing to score" not in run.stderr,
          f"offline_eval accepts the dataset the run recorded (it then fails on the fake weights: exit "
          f"{run.returncode})")
    run = score(other, ckpt, "--allow-different-dataset")
    check(refused not in run.stderr and "refusing to score" not in run.stderr,
          "--allow-different-dataset lets a different dataset through the check (debugging only)")
    caches = ("HF_HOME", "HF_DATASETS_CACHE", "TORCH_HOME", "XDG_CACHE_HOME")
    bare = {k: v for k, v in job_env(tmp).items() if k not in caches}  # as if run without common.sh
    run = subprocess.run([sys.executable, str(OFFLINE_EVAL), "--checkpoint", str(ckpt), "--dataset-root", str(fake),
                          "--out", str(tmp / "leak" / "report.json"), "--device", "cpu"], env=bare,
                         capture_output=True, text=True, timeout=600)
    check(run.returncode == 2 and "in the home directory" in run.stderr,
          "offline_eval refuses to evaluate with its caches in home (run without common.sh)")
    inputs = json.loads((run_dir / "inputs.json").read_text())
    val_source = json.loads((fake / "splits.json").read_text())["episodes"]["3"]["source"]
    crafted = tmp / "leak" / "inputs_with_val_source.json"
    crafted.write_text(json.dumps({**inputs, "train_sources": [val_source]}))
    run = score(other, ckpt, "--allow-different-dataset", "--train-identity", str(crafted))
    check(run.returncode != 0 and "validation sources" in run.stderr and val_source in run.stderr,
          "offline_eval refuses a validation episode whose source was trained on, even across datasets")


# ---- 6. a tiny real training, on CPU -------------------------------------------------------------------
def shadow_dataset(src: Path, dst: Path) -> Path:
    """A copy of a LeRobot dataset to edit freely: its small files copied, its data folders linked."""
    dst.mkdir(parents=True)
    for entry in src.iterdir():
        if entry.name == "meta":
            (dst / "meta").mkdir()
            for item in entry.iterdir():
                if item.is_dir():
                    (dst / "meta" / item.name).symlink_to(item.resolve())
                else:
                    shutil.copy2(item, dst / "meta" / item.name)
        elif entry.is_dir():
            (dst / entry.name).symlink_to(entry.resolve())
        else:
            shutil.copy2(entry, dst / entry.name)
    return dst


def smoke_training(tmp: Path, dataset_root: Path) -> None:
    missing = [m for m in ("datasets", "av", "accelerate") if importlib.util.find_spec(m) is None]
    if missing:
        check(False, f"tiny CPU training: this Python ({sys.executable}) lacks {missing}; install LeRobot's "
                     "extras: pip install 'lerobot[dataset,training]==0.6.1'")
        return
    if not (dataset_root / "meta" / "info.json").is_file():
        check(False, f"--dataset-root {dataset_root} has no meta/info.json")
        return
    shadow = shadow_dataset(dataset_root.resolve(), tmp / "datasets" / dataset_root.resolve().name)
    env = job_env(tmp, DATASET_ROOT=str(shadow), LFD_SMOKE="1", LEROBOT_ENV=sys.prefix, BATCH_SIZE="2")
    run_dir = tmp / "out" / shadow.name / "act-smoke" / f"seed{SEED}"
    ckpts = run_dir / "train" / "checkpoints"
    step_file = ckpts / "last" / "training_state" / "training_step.json"
    started = time.time()
    first = run_job("act.sbatch", env, timeout=3600, STEPS="20", LFD_SKIP_EVAL="1")
    step1 = json.loads(step_file.read_text())["step"] if step_file.is_file() else None
    if not check(first.returncode == 0 and step1 == 20,
                 f"tiny ACT training on CPU, 20 steps, batch 2: exit {first.returncode}, last checkpoint step "
                 f"{step1} ({time.time() - started:.0f} s)"):
        print(tail(first, 3000))
        return
    inputs = json.loads((run_dir / "inputs.json").read_text())
    identity = json.loads(subprocess.run([sys.executable, str(OFFLINE_EVAL), "--dataset-root", str(shadow),
                                          "--print-identity"], capture_output=True, text=True).stdout)
    check(inputs["conversion_sha256"] == identity["conversion_sha256"] and inputs["pretrained"] is None,
          f"the run recorded its dataset before training: conversion {inputs['conversion_sha256'][:12]}, "
          f"stats {str(inputs['stats_sha256'])[:12]}, label {inputs['label']}")
    with edited(shadow / "conversion.json", lambda t: t + "\n"):
        refused = run_job("act.sbatch", env, timeout=600, STEPS="30", LFD_EXTEND_STEPS="1")
    check(refused.returncode != 0 and "conversion_sha256" in refused.stderr,
          "resubmitting after the dataset was re-converted in place is refused, LFD_EXTEND_STEPS=1 or not")
    refused = run_job("act.sbatch", env, timeout=600, STEPS="30")
    check(refused.returncode != 0 and "LFD_EXTEND_STEPS" in refused.stderr,
          "resubmitting with a different STEPS is refused unless LFD_EXTEND_STEPS=1")
    started = time.time()
    second = run_job("act.sbatch", env, timeout=3600, STEPS="30", LFD_EXTEND_STEPS="1")
    step2 = json.loads(step_file.read_text())["step"] if step_file.is_file() else None
    actions = [json.loads(p.read_text()).get("action") for p in sorted((run_dir / "jobs").glob("*.json"))]
    if not check(second.returncode == 0 and step2 == 30 and actions == ["fresh", "resume"],
                 f"resubmitting resumed it to step {step2} (jobs: {actions}; {time.time() - started:.0f} s)"):
        print(tail(second, 3000))
        return
    kept = sorted(p.name for p in ckpts.iterdir())
    optimizer = {s: (ckpts / s / "training_state" / "optimizer_state.safetensors").is_file() for s in kept[:-1]}
    check(kept == ["000020", "000030", "last"] and optimizer == {"000020": False, "000030": True},
          f"after the run: checkpoints kept {kept}; optimizer state {optimizer} (only last keeps it)")
    reports = sorted((run_dir / "offline_eval").glob("*.json"))
    if not check(len(reports) == 1, f"offline_eval.py ran on the final checkpoint: {[p.name for p in reports]}"):
        print(tail(second, 3000))
        return
    report = json.loads(reports[0].read_text())
    splits = json.loads(subprocess.run([sys.executable, str(OFFLINE_EVAL), "--dataset-root", str(shadow),
                                        "--print-episodes", "val"], capture_output=True, text=True).stdout)
    policy, base = report["policy"], report["baseline_hold_state"]
    action_dim = len(policy["first_action_l1"])
    check(report["format"] == "bhl-lfd-offline-eval" and "not task success" in report["what_this_measures"]
          and "LFD_EVAL_PROTOCOL.md" in report["what_this_measures"]
          and report["dataset"]["val_episodes"] == splits and report["n_samples"] > 0
          and report["checkpoint"]["step"] == 30 and report["checkpoint"]["same_dataset"] is True
          and report["checkpoint"]["leakage_check"].startswith("ok: same dataset")
          and action_dim == len(report["dataset"]["action_names"]) and policy["l1_by_horizon"].get("1") is not None
          and policy["l1_by_horizon_joints_rad"].get("1") is not None
          and policy["common_horizon_steps"] == min(40, policy["chunk_steps"])
          and "not comparable across policies" in policy["chunk_l1_note"]
          and report["inference_time_s"]["per_call"]["n"] >= 1 and base is not None,
          f"offline report: {report['n_samples']} frames of validation episodes {splits}; L1 by horizon, joints "
          f"(rad) {policy['l1_by_horizon_joints_rad']} against the hold baseline "
          f"{base['l1_by_horizon_joints_rad'] if base else None}; chunk L1 per policy only; "
          f"{report['inference_time_s']['per_call']['mean'] * 1e3:.0f} ms per call on CPU")
    log = second.stdout + second.stderr
    check("not task success" in log and "Comparable across policies" in log,
          "the job output says the offline numbers are not task success, and leads with the comparable ones")

    ckpt = ckpts / "last" / "pretrained_model"

    def score(dataset: Path, out: str, *more: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(OFFLINE_EVAL), "--checkpoint", str(ckpt), "--dataset-root",
                               str(dataset), "--out", str(tmp / out), "--device", "cpu", "--num-workers", "0",
                               "--max-samples-per-episode", "2", "--warmup", "0", "--resize", "96", "128", *more],
                              capture_output=True, text=True, timeout=1800, env=job_env(tmp))

    copy = shadow_dataset(dataset_root.resolve(), tmp / "datasets_copy" / dataset_root.resolve().name)
    run = score(copy, "copy.json")
    report = json.loads((tmp / "copy.json").read_text()) if (tmp / "copy.json").is_file() else {}
    check(run.returncode == 0 and report.get("checkpoint", {}).get("leakage_check", "").startswith("ok: same dataset"),
          f"a byte-identical copy of the dataset at another path is the same dataset: exit {run.returncode}")
    other = shadow_dataset(dataset_root.resolve(), tmp / "datasets_other" / dataset_root.resolve().name)
    record = json.loads((other / "conversion.json").read_text())
    (other / "conversion.json").write_text(json.dumps({**record, "note": "another conversion"}, indent=1))
    run = score(other, "other.json")
    check(run.returncode != 0 and "different dataset" in run.stderr and not (tmp / "other.json").exists(),
          "offline_eval refuses the checkpoint on a re-converted dataset")
    run = score(other, "other.json", "--allow-different-dataset")
    report = json.loads((tmp / "other.json").read_text()) if (tmp / "other.json").is_file() else {}
    leakage = report.get("checkpoint", {}).get("leakage_check", "")
    check(run.returncode == 0 and leakage.startswith("different dataset") and "ok" not in leakage.split(":")[0]
          and report["checkpoint"]["same_dataset"] is False,
          f"with --allow-different-dataset the report says '{leakage[:60]}...', never 'ok'")


HOME_CACHES = ("~/.cache/huggingface", "~/.cache/torch", "~/.cache/matplotlib", "~/.cache/wandb", "~/.config/wandb",
               "~/.triton", "~/.nv")


def home_writes(since: float) -> list[str]:
    """Cache folders in home written since `since` (the standing rule: nothing here writes to home)."""
    found = []
    for raw in HOME_CACHES:
        path = Path(os.path.expanduser(raw))
        if not path.exists():
            continue
        run = subprocess.run(["find", str(path), "-newermt", f"@{since:.0f}", "-print", "-quit"],
                             capture_output=True, text=True, timeout=120)
        if run.stdout.strip():
            found.append(run.stdout.strip())
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset-root", type=Path, help="a LeRobot v3 dataset with splits.json, for step 6")
    ap.add_argument("--keep", action="store_true", help="keep the scratch folder for inspection")
    args = ap.parse_args()

    SCRATCH.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="lfd-test-configs-", dir=SCRATCH))
    print(f"scratch: {tmp}")
    # before LeRobot, huggingface_hub and datasets read them at import: this process and every child it starts
    # keep their caches in the scratch folder, never in home
    os.environ.update(scratch_caches(tmp / "cache"))
    started = time.time()
    try:
        import lerobot.policies  # noqa: F401 - registers the policy config classes; slow on NFS, so once

        mocks = write_mocks(tmp)
        make_fake_dataset(tmp / "fake_dataset")
        check_headers(tmp)
        check_variants(tmp, mocks)
        check_job_flows(tmp, mocks)
        check_groot_view(tmp, mocks)
        check_helpers(tmp, mocks)
        check_leakage(tmp)
        if args.dataset_root is None:
            print("SKIP  tiny CPU training and offline_eval.py: no --dataset-root given (they need a LeRobot v3 "
                  "dataset with splits.json; rerun with --dataset-root DIR)")
        else:
            smoke_training(tmp, args.dataset_root)
        written = home_writes(started - 1)
        check(not written, f"nothing was written to the home caches during the run: {written or 'none'}")
    finally:
        if args.keep:
            print(f"kept {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    failures = [what for ok, what in results if not ok]
    print()
    for what in failures:
        print("FAIL  " + what)
    print(f"{len(results) - len(failures)} passed, {len(failures)} failed ({time.time() - started:.0f} s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
