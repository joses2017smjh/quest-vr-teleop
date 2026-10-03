# shellcheck shell=bash
# Shared settings and helpers for the LfD training jobs: act.sbatch, pi05.sbatch and groot.sbatch.
#
# Source it, do not run it. The job scripts source it, and so can you on a login or compute node:
#
#   source /nfs/hpc/share/$USER/quest-vr-teleop/tools/lfd/train/common.sh
#   lfd_train_episodes $DATA_ROOT/local/washcloth_v1   # [0,1,3,...]: the train list of its splits.json
#   lfd_val_episodes   $DATA_ROOT/local/washcloth_v1
#   lfd_identity       $DATA_ROOT/local/washcloth_v1   # sha256 of conversion.json, stats.json, splits.json
#   lfd_quota_check 0                                # the share's soft quota, its grace, and the hard limit
#
# Nothing here writes to the home directory. Home (/nfs/stak, 25 GB) is nearly full, and a full home blocks
# the user's HPC use, so every cache the Python libraries know about is pointed at the share first.

LFD_SHARE="${LFD_SHARE:-/nfs/hpc/share/$USER}"
if [ -z "${LFD_REPO:-}" ]; then
    LFD_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi

# ---- caches: on the share, never home ----------------------------------------------------------------
LFD_CACHE="${LFD_CACHE:-$LFD_SHARE/.cache}"
case "$LFD_CACHE" in
    "$HOME" | "$HOME"/* | /nfs/stak/*)
        echo "common.sh: refusing LFD_CACHE=$LFD_CACHE: caches must not go to the home directory" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac
export XDG_CACHE_HOME="$LFD_CACHE"
export HF_HOME="$LFD_CACHE/huggingface"        # Hub downloads, the HF token, datasets, accelerate
export HF_LEROBOT_HOME="$HF_HOME/lerobot"
export HF_HUB_DISABLE_TELEMETRY=1
export TORCH_HOME="$LFD_CACHE/torch"            # ACT's ImageNet ResNet-18 weights land here
export TORCHINDUCTOR_CACHE_DIR="$LFD_CACHE/inductor"
export TRITON_CACHE_DIR="$LFD_CACHE/triton"
export PIP_CACHE_DIR="$LFD_CACHE/pip"
export MPLCONFIGDIR="$LFD_CACHE/matplotlib"
# Every job also passes --wandb.enable=false. These keep anything else that imports wandb offline and on the
# share (~/.netrc holds credentials, and wandb's own default is online).
export WANDB_MODE=disabled WANDB_DISABLED=true
export WANDB_DIR="$LFD_CACHE/wandb" WANDB_CACHE_DIR="$LFD_CACHE/wandb" WANDB_CONFIG_DIR="$LFD_CACHE/wandb"
# Pinned views of pretrained snapshots (GR00T's; see lfd_pretrained), inside the share's Hugging Face cache.
export LFD_PINS="${LFD_PINS:-$HF_HOME/lfd-pins}"
# CUDA's JIT cache defaults to ~/.nv. Node-local scratch is faster, as in the LeHome jobs' setup_node_cache.
if [ -z "${CUDA_CACHE_PATH:-}" ]; then
    if mkdir -p "/scratch/$USER/nv-computecache" 2>/dev/null; then
        export CUDA_CACHE_PATH="/scratch/$USER/nv-computecache"
    else
        export CUDA_CACHE_PATH="$LFD_CACHE/nv"
    fi
fi
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

# ---- where things live ----------------------------------------------------------------------------------
DATA_ROOT="${DATA_ROOT:-$LFD_SHARE/bhl-data/lerobot}"          # converted datasets (converter --out-root)
INCOMING_ROOT="${INCOMING_ROOT:-$LFD_SHARE/bhl-data/incoming}" # raw sessions rsynced from the robot PC
OUT_ROOT="${OUT_ROOT:-$LFD_SHARE/bhl-data/train}"               # one folder per run: <dataset>/<policy>/seed<N>
LOG_ROOT="${LOG_ROOT:-$LFD_SHARE/bhl-data/logs}"                # the #SBATCH --output lines point here
# The GPU-enabled Python 3.12 env with LeRobot 0.6.1. A placeholder until the user makes it (README,
# "Make the GPU env"). The CPU env beside it is for dry runs and tests only.
LEROBOT_ENV="${LEROBOT_ENV:-$LFD_SHARE/envs/lerobot-py312-gpu}"
LFD_CPU_ENV="${LFD_CPU_ENV:-$LFD_SHARE/envs/lerobot-py312-cpu}"
LEROBOT_PY="$LEROBOT_ENV/bin/python"
# One decoder for every model, so all three see bit-identical frames. PyAV wheels bundle their own FFmpeg,
# so it works on every node; torchcodec needs system FFmpeg libraries.
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"

# Gated repos (π0.5's PaliGemma tokenizer, GR00T's Cosmos-Reason2 processor) need a Hugging Face token. It is
# read from a 0600 file, never written in a script, so it stays out of git, logs and `scontrol show job`.
# `hf auth login` with this HF_HOME stores it in $HF_HOME/token, which huggingface_hub reads by itself.
LFD_HF_TOKEN_FILE="${LFD_HF_TOKEN_FILE:-$LFD_SHARE/Humanoid_Lite/.hf_token}"
if [ -z "${HF_TOKEN:-}" ] && [ ! -s "$HF_HOME/token" ] && [ -r "$LFD_HF_TOKEN_FILE" ]; then
    HF_TOKEN="$(tr -d '[:space:]' <"$LFD_HF_TOKEN_FILE")"
    export HF_TOKEN
fi

lfd_log() { printf '[lfd %s] %s\n' "$(date '+%F %T')" "$*"; }

# Only for the job scripts: in an interactive shell that sourced this file, use the helpers below instead.
lfd_die() {
    printf '[lfd %s] ERROR: %s\n' "$(date '+%F %T')" "$*" >&2
    exit 1
}

lfd_py() { # the Python for the small helpers: LFD_PY, else the GPU env, else the CPU env, else python3
    local candidate
    for candidate in "${LFD_PY:-}" "$LEROBOT_PY" "$LFD_CPU_ENV/bin/python"; do
        if [ -n "$candidate" ] && [ -x "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done
    command -v python3
}

lfd_json_get() { # JSON KEY[.KEY...]: print one value of a JSON text ("" for null or missing)
    "$(lfd_py)" -c 'import json, sys
v = json.loads(sys.argv[1] or "null")
for k in sys.argv[2].split("."):
    v = v.get(k) if isinstance(v, dict) else None
print("" if v is None else v)' "$1" "$2"
}

# ---- splits.json and the dataset's identity -----------------------------------------------------------
# Print an episode list of the dataset's splits.json as [i,j,...], ready for --dataset.episodes. The parsing
# lives in offline_eval.py, so training and evaluation can never read the split differently.
lfd_train_episodes() { # DATASET_ROOT [SPLITS_FILE]
    local root=${1:?usage: lfd_train_episodes DATASET_ROOT [SPLITS_FILE]} splits=${2:-${SPLITS_FILE:-}}
    "$(lfd_py)" "$LFD_REPO/tools/lfd/offline_eval.py" --dataset-root "$root" \
        ${splits:+--splits "$splits"} --print-episodes train
}

lfd_val_episodes() { # DATASET_ROOT [SPLITS_FILE]
    local root=${1:?usage: lfd_val_episodes DATASET_ROOT [SPLITS_FILE]} splits=${2:-${SPLITS_FILE:-}}
    "$(lfd_py)" "$LFD_REPO/tools/lfd/offline_eval.py" --dataset-root "$root" \
        ${splits:+--splits "$splits"} --print-episodes val
}

# The dataset by content: sha256 of conversion.json, meta/stats.json and the splits file, the label and the
# lookahead (offline_eval.dataset_identity). A run records it in <run>/inputs.json when it starts; every
# resume and every offline evaluation compares against that record.
lfd_identity() { # DATASET_ROOT [SPLITS_FILE]
    local root=${1:?usage: lfd_identity DATASET_ROOT [SPLITS_FILE]} splits=${2:-${SPLITS_FILE:-}}
    "$(lfd_py)" "$LFD_REPO/tools/lfd/offline_eval.py" --dataset-root "$root" \
        ${splits:+--splits "$splits"} --print-identity
}

# ---- checks run at the start of every job ---------------------------------------------------------------
lfd_check_gpu() { # NEED_BF16 (0/1) MIN_GPU_MEM_GB (GiB, as torch reports total_memory)
    local need_bf16=${1:-1} min_gb=${2:-0}
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "no nvidia-smi on $(hostname): not a GPU node" >&2
        return 1
    fi
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || return 1
    "$LEROBOT_PY" - "$need_bf16" "$min_gb" <<'PY'
import os
import sys

import torch

need_bf16, min_gb = sys.argv[1] == "1", float(sys.argv[2])
if not torch.cuda.is_available():
    sys.exit(f"torch {torch.__version__} sees no GPU (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
props = torch.cuda.get_device_properties(0)
mem_gb = props.total_memory / 2**30
# The default including_emulation=True answers True on Turing (the RTX 8000 `gpu` partition), which only
# emulates bf16. Native bf16 needs compute capability 8.0 or newer (A40, H100).
bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
print(f"gpu: {props.name}, {mem_gb:.1f} GiB, compute capability {props.major}.{props.minor}, native bf16: {bf16}; "
      f"torch {torch.__version__} built for CUDA {torch.version.cuda}")
if need_bf16 and not bf16:
    sys.exit("this policy trains in bf16 and this GPU has no native bf16: use ampere or dgxh, never `gpu`")
if mem_gb < min_gb:
    # An 80 GB H100 reports 79.6 GiB; dgxh-1's MIG slices (gres gpu:h100-40g) report about 39.5.
    sys.exit(f"this run needs at least {min_gb:g} GiB of GPU memory; this GPU ({props.name}) has {mem_gb:.1f} "
             "(a 40 GB MIG slice? submit with --constraint=vram80g)")
PY
}

lfd_check_env() { # POLICY_TAG: versions, and the packages that policy imports
    "$LEROBOT_PY" - "$1" "$VIDEO_BACKEND" <<'PY'
import importlib.util
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version


def ver(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


tag, backend = sys.argv[1], sys.argv[2]
print("env:", sys.prefix, "| python", platform.python_version(),
      "| " + " ".join(f"{p} {ver(p)}" for p in ("lerobot", "torch", "transformers", "peft", "accelerate", "av")))
problems = []
if sys.version_info[:2] != (3, 12):
    problems.append(f"Python {platform.python_version()}, LeRobot 0.6.1 needs 3.12")
if ver("lerobot") != "0.6.1" and os.environ.get("LFD_ALLOW_OTHER_LEROBOT") != "1":
    problems.append(f"lerobot {ver('lerobot')}: these scripts were checked against 0.6.1 "
                    "(LFD_ALLOW_OTHER_LEROBOT=1 to run anyway)")
needed = ["accelerate", "datasets", "av" if backend == "pyav" else "torchcodec"]
needed += {"pi05_lora": ["transformers", "scipy", "peft"], "pi05_full": ["transformers", "scipy"],
           "groot": ["transformers", "diffusers"]}.get(tag, [])
problems += [f"module {m} missing" for m in needed if importlib.util.find_spec(m) is None]
if problems:
    sys.exit("environment: " + "; ".join(problems) + ". See tools/lfd/train/README.md, 'Make the GPU env'.")
PY
}

lfd_check_dataset() { # DATASET_ROOT NEED_QUANTILES (0/1): the contract all three policies are trained on
    "$(lfd_py)" - "$1" "${2:-0}" <<'PY'
import json
import sys
from pathlib import Path


def strict(path):
    """JSON as the format requires it: NaN and Infinity are not JSON (docs/LFD_RECORDING_FORMAT.md)."""
    def refuse(token):
        raise ValueError(f"{path} holds {token}, which is not JSON")
    return json.loads(path.read_text(), parse_constant=refuse)


root, need_quantiles = Path(sys.argv[1]), sys.argv[2] == "1"
info = json.loads((root / "meta" / "info.json").read_text())
features = info.get("features", {})
problems, warnings = [], []
for key in ("observation.state", "action"):
    shape = (features.get(key) or {}).get("shape")
    if list(shape or []) != [12]:
        problems.append(f"{key} has shape {shape}, the format v1 contract is [12]")
for cam in ("observation.images.left", "observation.images.right"):
    if cam not in features:
        problems.append(f"no {cam}")
if info.get("fps") != 30:
    problems.append(f"fps {info.get('fps')}, the converter writes 30")
stats_path = root / "meta" / "stats.json"
stats = {}
if stats_path.is_file():
    try:
        stats = strict(stats_path)
    except ValueError as exc:
        problems.append(f"{exc}: re-convert with the current converter (unknown values are stored as 0 plus a flag)")
if need_quantiles:
    # π0.5 normalizes state and action by their 1st and 99th percentiles (QUANTILES mode).
    for key in ("observation.state", "action"):
        if not {"q01", "q99"} <= set(stats.get(key, {})):
            problems.append(f"meta/stats.json has no q01/q99 for {key}; re-convert with the current converter")
    conversion = root / "conversion.json"
    record = json.loads(conversion.read_text()) if conversion.is_file() else {}
    nested = record.get("stats") if isinstance(record.get("stats"), dict) else {}
    if not (record.get("stats_recomputed") or nested.get("recomputed") or nested.get("quantiles_recomputed")):
        warnings.append("conversion.json does not record that the quantile statistics were recomputed over all "
                        "frames after finalize(); LeRobot's own q01/q99 are approximate")
cams = {k: v.get("shape") for k, v in features.items() if k.startswith("observation.images.")}
print(f"dataset: {root} | {info.get('total_episodes')} episodes, {info.get('total_frames')} frames, "
      f"{info.get('fps')} fps | cameras {cams} | robot_type {info.get('robot_type')}")
for warning in warnings:
    print(f"WARNING: dataset: {warning}")
if problems:
    sys.exit("dataset: " + "; ".join(problems))
PY
}

# ---- the share's quota ------------------------------------------------------------------------------------
# The share is Lustre project 30762: 1.5 TiB soft quota, 2 TiB hard limit. Above the soft quota a grace clock
# runs; when it expires, Lustre enforces the soft quota as a hard limit and every write of the project fails,
# the robot's data sync included. So a job refuses to start while usage is over the soft quota with less than
# LFD_MIN_GRACE_DAYS (7) of grace left, unless LFD_ALLOW_LOW_GRACE=1; never once the grace has expired; and
# never when this run could cross the hard limit (budgets.md, "Storage").
lfd_grace_seconds() { # GRACE as lfs prints it ("3w4d6h18m48s") -> seconds; fails for anything else
    local rest=$1 total=0 n unit
    [[ "$rest" =~ ^([0-9]+[wdhms])+$ ]] || return 1
    while [[ "$rest" =~ ^([0-9]+)([wdhms])(.*)$ ]]; do
        n=$((10#${BASH_REMATCH[1]}))
        unit=${BASH_REMATCH[2]}
        rest=${BASH_REMATCH[3]}
        case $unit in
            w) total=$((total + n * 604800)) ;;
            d) total=$((total + n * 86400)) ;;
            h) total=$((total + n * 3600)) ;;
            m) total=$((total + n * 60)) ;;
            s) total=$((total + n)) ;;
        esac
    done
    echo "$total"
}

lfd_quota_check() { # NEED_GB: print the share's quota status; refuse (return 1) when this run must not start
    local need_gb=${1:-0} proj line used_kb soft_kb hard_kb grace limit_kb need_kb grace_s="" min_days
    local gib=1048576 status
    min_days=${LFD_MIN_GRACE_DAYS:-7}
    if [ "${LFD_SKIP_QUOTA_CHECK:-0}" = 1 ]; then
        lfd_log "share quota check skipped (LFD_SKIP_QUOTA_CHECK=1)"
        return 0
    fi
    if ! command -v lfs >/dev/null 2>&1; then
        lfd_log "WARNING: no lfs command here, so share usage was not checked"
        return 0
    fi
    proj=$(lfs project -d "$LFD_SHARE" 2>/dev/null | awk '{print $1}')
    if [ -n "$proj" ] && [ "$proj" != 0 ]; then
        line=$(lfs quota -q -p "$proj" /nfs/hpc/share 2>/dev/null | tail -n 1)
    else
        line=$(lfs quota -q -u "$USER" /nfs/hpc/share 2>/dev/null | tail -n 1)
    fi
    # "  /nfs/hpc/share <used>[*] <soft> <hard> <grace> <files>[*] ...", in KiB; the path may sit on its own line
    read -r used_kb soft_kb hard_kb grace <<<"$(awk '{ i = ($1 ~ /^\//) ? 2 : 1; v = $i; gsub(/\*/, "", v)
        print v, $(i + 1), $(i + 2), $(i + 3) }' <<<"$line")"
    if ! [[ "$used_kb" =~ ^[0-9]+$ ]]; then
        echo "could not read the share quota (lfs printed: '$line'). The share is over its soft quota" \
            "(budgets.md, 'Storage'), so the job does not start blind; LFD_SKIP_QUOTA_CHECK=1 runs anyway" >&2
        return 1
    fi
    [[ "$soft_kb" =~ ^[0-9]+$ ]] || soft_kb=0
    [[ "$hard_kb" =~ ^[0-9]+$ ]] || hard_kb=0
    need_kb=$((need_gb * gib))
    if [ -n "${LFD_SHARE_LIMIT_GB:-}" ]; then
        limit_kb=$((LFD_SHARE_LIMIT_GB * gib))
    elif [ "$hard_kb" -gt 0 ]; then
        limit_kb=$hard_kb
    else
        limit_kb=$((2048 * gib))
    fi
    if [ "$soft_kb" -gt 0 ] && [ "$used_kb" -gt "$soft_kb" ]; then
        if [ "$grace" = none ]; then
            status="EXPIRED: Lustre now enforces the soft quota as a hard limit"
            limit_kb=$soft_kb
        elif grace_s=$(lfd_grace_seconds "$grace"); then
            status="grace ends in $grace ($((grace_s / 86400)) days $(((grace_s % 86400) / 3600)) h)"
        else
            status="grace unreadable ('$grace')"
        fi
        lfd_log "share (project ${proj:-none}): $((used_kb / gib)) GiB used, OVER the $((soft_kb / gib)) GiB soft" \
            "quota by $(((used_kb - soft_kb) / gib)) GiB; $status; hard limit $((hard_kb / gib)) GiB;" \
            "this run may add up to $need_gb GiB"
        if [ "$grace" = none ]; then
            echo "the share's soft-quota grace has expired, so every write of the project fails (the robot's data" \
                "sync too): the owner must bring usage below $((soft_kb / gib)) GiB first (budgets.md, 'Storage')" >&2
            return 1
        fi
        if [ -z "$grace_s" ] || [ "$grace_s" -lt $((min_days * 86400)) ]; then
            if [ "${LFD_ALLOW_LOW_GRACE:-0}" != 1 ]; then
                echo "the share is over its soft quota and its grace ($grace) is under $min_days days: when it" \
                    "ends, every write of the project fails, the robot's data sync included. The owner must free" \
                    "space below $((soft_kb / gib)) GiB first (budgets.md, 'Storage'); LFD_ALLOW_LOW_GRACE=1" \
                    "starts anyway" >&2
                return 1
            fi
            lfd_log "WARNING: starting with under $min_days days of soft-quota grace (LFD_ALLOW_LOW_GRACE=1)"
        else
            lfd_log "WARNING: the owner must bring the share below $((soft_kb / gib)) GiB before the grace ends" \
                "(budgets.md, 'Storage'); this pipeline cannot"
        fi
    elif [ "$soft_kb" -gt 0 ]; then
        lfd_log "share (project ${proj:-none}): $((used_kb / gib)) GiB used, $(((soft_kb - used_kb) / gib)) GiB" \
            "below the $((soft_kb / gib)) GiB soft quota (no grace clock); hard limit $((hard_kb / gib)) GiB;" \
            "this run may add up to $need_gb GiB"
        if [ $((used_kb + need_kb)) -gt "$soft_kb" ]; then
            lfd_log "WARNING: this run could push the share over its soft quota and start the grace clock"
        fi
    else
        lfd_log "share (project ${proj:-none}): $((used_kb / gib)) GiB used, no soft quota reported; limit" \
            "$((limit_kb / gib)) GiB; this run may add up to $need_gb GiB"
    fi
    if [ $((used_kb + need_kb)) -gt "$limit_kb" ]; then
        echo "not enough room on the share for this run: $((used_kb / gib)) GiB used + up to $need_gb GiB would" \
            "cross the $((limit_kb / gib)) GiB limit; free space, or ask the owner" >&2
        return 1
    fi
}

# ---- one job per run folder at a time -----------------------------------------------------------------
# A second submission while the first is pending or running would resume from the same checkpoint into the same
# output_dir, or move the live job's train/ aside. So each job holds an flock on <run>/.lock (the share is
# mounted with flock) for its whole life; the lock goes when the job's processes end. The lock file names its
# holder. If the holder's Slurm job is no longer in squeue (or has ended), the lock is stale and is taken over.
lfd_lock_holder_gone() { # HOLDER_LINE: 0 when the holder has certainly ended, 1 when alive or unknown
    local line=$1 job host pid out state
    job=$(sed -n 's/.*job=\([^ ]*\).*/\1/p' <<<"$line")
    host=$(sed -n 's/.*host=\([^ ]*\).*/\1/p' <<<"$line")
    pid=$(sed -n 's/.*pid=\([0-9]*\).*/\1/p' <<<"$line")
    if [[ "$job" =~ ^[0-9]+(_[0-9]+)?$ ]]; then # a Slurm job id: ask squeue (read-only)
        command -v squeue >/dev/null 2>&1 || return 1
        if ! out=$(squeue -h -j "$job" -o %T 2>&1); then
            grep -qi "invalid job id" <<<"$out" && return 0 # no longer known to Slurm
            return 1                                        # squeue failed for another reason: cannot tell
        fi
        state=$(head -n 1 <<<"$out" | tr -d '[:space:]')
        case "$state" in
            "" | COMPLETED | CANCELLED* | FAILED | TIMEOUT | NODE_FAIL | PREEMPTED | BOOT_FAIL | DEADLINE | \
                OUT_OF_MEMORY) return 0 ;;
            *) return 1 ;; # PENDING (a requeue), RUNNING, COMPLETING, ...
        esac
    fi
    # a run by hand: stale only if it ran on this host and its process is gone
    if [ -n "$host" ] && [ "$host" = "$(hostname)" ] && [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
        return 0
    fi
    return 1
}

lfd_lock() { # take $RUN_DIR/.lock for the rest of this job, or exit
    local lock="$RUN_DIR/.lock" holder
    command -v flock >/dev/null 2>&1 || lfd_die "no flock command on $(hostname)"
    mkdir -p "$RUN_DIR" || lfd_die "cannot create $RUN_DIR"
    exec {LFD_LOCK_FD}>>"$lock" || lfd_die "cannot open $lock"
    if ! flock -n "$LFD_LOCK_FD"; then
        holder=$(head -n 1 "$lock" 2>/dev/null) || holder=""
        if ! lfd_lock_holder_gone "$holder"; then
            lfd_die "another job holds $lock (${holder:-holder unknown}): submit again only once it has ended" \
                "(squeue -u $USER), or scancel it"
        fi
        lfd_log "taking over a stale lock: ${holder:-its holder} is no longer in squeue; the old lock file is kept" \
            "as $lock.stale.*"
        exec {LFD_LOCK_FD}>&-
        mv -f -- "$lock" "$lock.stale.$(date +%Y%m%d_%H%M%S)" || lfd_die "cannot move the stale $lock aside"
        exec {LFD_LOCK_FD}>>"$lock" || lfd_die "cannot open $lock"
        flock -n "$LFD_LOCK_FD" || lfd_die "another job took $lock first"
    fi
    # rewritten through a second descriptor: the flock belongs to LFD_LOCK_FD and stays
    printf 'job=%s host=%s pid=%s since=%s\n' "$JOB_TAG" "$(hostname)" "$$" "$(date '+%F %T')" >"$lock"
}

# ---- pretrained weights, pinned ---------------------------------------------------------------------------
# A run trains exactly one recorded revision of its pretrained weights. lfd_pretrained resolves the commit the
# Hub's main points to (HfApi().model_info(repo).sha, read-only), or takes the recorded one; with FETCH=1 it
# snapshot_downloads that commit into the share's Hugging Face cache. It prints a JSON object: repo, revision,
# snapshot and path, the local folder the job passes to LeRobot.
#
# PRETRAINED_HIDE lists top-level files to leave out: path is then a pinned view, $LFD_PINS/<org>--<name>/<sha>,
# holding links to every other file of the snapshot. GR00T needs one. Given a local folder holding a raw N1.7
# checkpoint, LeRobot 0.6.1 builds its processor from that checkpoint's processor_config.json, statistics.json
# and embodiment_id.json. That id map has no new_embodiment, so the tag would silently fall back to slot 0
# (simpler_env_google), and the checkpoint's state dropout (0.2) and albumentations would switch on. Without the
# three files, the processor is built exactly as for the Hub id: LeRobot's own map (new_embodiment = slot 10),
# the dataset's statistics, no state dropout, and the default 256/230 image geometry.
lfd_pretrained() { # REPO REVISION (empty: the Hub's main now) FETCH (0/1)
    LFD_PIN_HIDE="${PRETRAINED_HIDE:-}" "$(lfd_py)" - "${1:?usage: lfd_pretrained REPO REVISION FETCH}" "${2:-}" \
        "${3:-0}" <<'PY'
import json
import os
import re
import sys
from pathlib import Path

repo, want, fetch = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
hide = os.environ.get("LFD_PIN_HIDE", "").split()
from huggingface_hub import HfApi, snapshot_download  # noqa: E402 - only once the arguments are known

sha = want or HfApi().model_info(repo).sha  # read-only: the commit main points to now
if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
    sys.exit(f"pin: {repo}: {sha!r} is not a full commit sha")
hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ["HF_HOME"]) / "hub")
snapshot = hub / ("models--" + repo.replace("/", "--")) / "snapshots" / sha
if fetch:
    got = Path(snapshot_download(repo_id=repo, revision=sha))
    if got.resolve() != snapshot.resolve():
        sys.exit(f"pin: {repo}@{sha} landed in {got}, not in {snapshot}")
path = snapshot
if hide:
    path = Path(os.environ["LFD_PINS"]) / repo.replace("/", "--") / sha
    if fetch:
        path.mkdir(parents=True, exist_ok=True)
        for entry in sorted(snapshot.iterdir()):
            link = path / entry.name
            if entry.name in hide or (link.is_symlink() and os.readlink(link) == str(entry)):
                continue
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(entry)
        for name in hide:
            if (path / name).is_symlink() or (path / name).exists():
                (path / name).unlink()
        missing = [e.name for e in snapshot.iterdir() if e.name not in hide and not (path / e.name).exists()]
        if missing:
            sys.exit(f"pin: the view {path} lacks {missing}")
print(json.dumps({"repo": repo, "revision": sha, "snapshot": str(snapshot), "path": str(path), "hidden": hide,
                  "resolved_from": "recorded" if want else "main"}))
PY
}

# ---- checkpoints ----------------------------------------------------------------------------------------
# LeRobot keeps every checkpoint, and save_freq is 1000, so a GR00T run would write ~20 GB per 1000 steps.
# LeRobot writes a checkpoint as: the weights, train_config.json, the processors, then training_state/; and it
# points `last` at it only after the whole save. A step folder without the weights and train_config.json is
# half-written: being saved now, or left by a killed job (the next save of that step writes over it).
lfd_checkpoint_complete() { # STEP_DIR: true when it holds a saved model and its train_config.json
    local model="$1/pretrained_model" f
    [ -f "$model/train_config.json" ] || return 1
    for f in "$model/model.safetensors" "$model/adapter_model.safetensors" "$model"/model-*-of-*.safetensors; do
        [ -f "$f" ] && return 0
    done
    return 1
}

lfd_step_dirs() { # CHECKPOINTS_DIR: the numbered checkpoint folders, newest first
    find "$1" -mindepth 1 -maxdepth 1 -type d -regextype posix-extended -regex '.*/[0-9]+' -printf '%f\n' 2>/dev/null |
        sort -rn
}

lfd_newest_complete() { # CHECKPOINTS_DIR: the name of the newest complete checkpoint, if any
    local name
    while IFS= read -r name; do
        if lfd_checkpoint_complete "$1/$name"; then
            echo "$name"
            return 0
        fi
    done < <(lfd_step_dirs "$1")
    return 1
}

# Keep the newest KEEP complete checkpoints, whatever `last` points to, and every folder newer than the newest
# complete one (it may be the save in progress). Delete the rest: older complete ones, and half-written ones
# older than the newest complete one. A failed delete is logged and retried on the next pass; it never stops
# the pruning. With no complete checkpoint recognized, nothing is deleted.
lfd_prune_checkpoints() { # CHECKPOINTS_DIR KEEP
    local dir=${1:?usage: lfd_prune_checkpoints CHECKPOINTS_DIR KEEP} keep=${2:-1} last="" newest="" name why
    local complete=0 failed=0
    [ -d "$dir" ] || return 0
    [[ "$keep" =~ ^[0-9]+$ ]] && [ "$keep" -ge 1 ] || keep=1
    if [ -L "$dir/last" ]; then
        last=$(basename "$(readlink "$dir/last")")
    fi
    while IFS= read -r name; do
        if [ -z "$newest" ]; then
            if lfd_checkpoint_complete "$dir/$name"; then
                newest=$name
                complete=1
            fi
            continue # the newest complete one, or newer than it: kept
        fi
        [ "$name" = "$last" ] && continue
        if lfd_checkpoint_complete "$dir/$name"; then
            complete=$((complete + 1))
            [ "$complete" -le "$keep" ] && continue
            why="superseded: keeping the newest $keep complete"
        else
            why="half-written, and older than the newest complete one, $newest"
        fi
        lfd_log "pruning checkpoint $dir/$name ($why; last -> ${last:-none})"
        if ! command rm -rf -- "${dir:?}/${name:?}"; then
            lfd_log "WARNING: could not delete all of $dir/$name; it is tried again on the next pass"
            failed=1
        fi
    done < <(lfd_step_dirs "$dir")
    if [ -z "$newest" ] && [ -n "$(lfd_step_dirs "$dir" | head -n 1)" ]; then
        lfd_log "WARNING: no complete checkpoint recognized in $dir, so nothing was pruned"
    fi
    return "$failed"
}

# After a finished run, only the checkpoint `last` points to keeps its optimizer state (what a resume or
# LFD_EXTEND_STEPS needs). Older ones keep their weights and training_step.json. LFD_KEEP_OPTIMIZER=1 keeps all.
lfd_drop_optimizer_state() { # CHECKPOINTS_DIR
    local dir=${1:?usage: lfd_drop_optimizer_state CHECKPOINTS_DIR} last name f
    if [ "${LFD_KEEP_OPTIMIZER:-0}" = 1 ]; then
        lfd_log "keeping the optimizer state of every checkpoint (LFD_KEEP_OPTIMIZER=1)"
        return 0
    fi
    [ -L "$dir/last" ] || return 0
    last=$(basename "$(readlink "$dir/last")")
    while IFS= read -r name; do
        [ "$name" = "$last" ] && continue
        for f in "$dir/$name/training_state/optimizer_state.safetensors" \
            "$dir/$name/training_state/optimizer_param_groups.json"; do
            [ -e "$f" ] || continue
            if command rm -f -- "$f"; then
                lfd_log "dropped $f (the run is finished; only last -> $last keeps its optimizer state)"
            else
                lfd_log "WARNING: could not delete $f"
            fi
        done
    done < <(lfd_step_dirs "$dir")
    return 0
}

lfd_checkpoint_step() { # CHECKPOINT_DIR (a step dir, or last): the training step it was saved at ("" if none)
    sed -n 's/.*"step": *\([0-9][0-9]*\).*/\1/p' "$1/training_state/training_step.json" 2>/dev/null | head -n 1 ||
        true
}

lfd_dir_gb() { # DIR: its size in GiB, rounded up
    local kb
    kb=$(du -sk -- "$1" 2>/dev/null | awk '{print $1}')
    echo $((((${kb:-0}) + 1048575) / 1048576))
}

# The most this job can add to the share: (KEEP_LAST + 1) checkpoints (the kept ones plus the one being
# written), each the size of this run's newest complete checkpoint once there is one (CKPT_GB, the job
# script's estimate, before); plus the pretrained weights when their pinned snapshot is not cached yet; plus 1
# for logs and reports. An evaluation-only job writes no checkpoint.
lfd_peak_gb() { # PIN_JSON (may be empty) MODE (train/eval)
    local ckpt_gb=${CKPT_GB:-1} n_ckpt=$((KEEP_LAST + 1)) base_gb=0 newest="" measured snapshot how=estimate
    if newest=$(lfd_newest_complete "$TRAIN_DIR/checkpoints"); then
        measured=$(lfd_dir_gb "$TRAIN_DIR/checkpoints/$newest")
        if [ "$measured" -gt 0 ]; then
            ckpt_gb=$measured
            how="measured on $newest"
        fi
    fi
    [ "${2:-train}" = eval ] && n_ckpt=0
    snapshot=$(lfd_json_get "${1:-}" snapshot)
    if [ -n "${PRETRAINED_REPO:-}" ] && { [ -z "$snapshot" ] || [ ! -d "$snapshot" ]; }; then
        base_gb=${BASE_GB:-0}
    fi
    lfd_log "storage: up to $n_ckpt checkpoints of $ckpt_gb GiB ($how; KEEP_LAST=$KEEP_LAST)," \
        "+ $base_gb GiB of pretrained weights to download, + 1" >&2
    echo $((base_gb + n_ckpt * ckpt_gb + 1))
}

# The background loops sleep through this: the sleep holds none of the job's descriptors (its output pipe, the
# run lock), so a sleep left behind when the job ends keeps neither the log open nor the run locked.
lfd_nap() { # SECONDS
    if [ -n "${LFD_LOCK_FD:-}" ]; then
        sleep "$1" </dev/null >/dev/null 2>&1 {LFD_LOCK_FD}>&-
    else
        sleep "$1" </dev/null >/dev/null 2>&1
    fi
}

lfd_stop_background() { # stop the job's helper loops, and the sleeps they are in
    # Called under the job's errexit and pipefail: every step here must succeed, or the job would exit at the
    # first helper that had already ended (pgrep finds no child), before stopping the rest.
    local pid kids
    for pid in ${LFD_BACKGROUND_PIDS:-}; do
        kids=$(pgrep -P "$pid" 2>/dev/null) || kids=""
        kill "$pid" 2>/dev/null || true # the loop first, so it does not report its sleep's end
        if [ -n "$kids" ]; then
            kill $kids 2>/dev/null || true
        fi
    done
    for pid in ${LFD_BACKGROUND_PIDS:-}; do
        wait "$pid" 2>/dev/null || true
    done
    LFD_BACKGROUND_PIDS=""
}

lfd_watch_log() { # LOG PID FLAG: stop training if the log says the pretrained weights did not load
    local log=$1 pid=$2 flag=$3 pattern
    [ "${#FATAL_LOG_PATTERNS[@]}" -gt 0 ] || return 0
    while kill -0 "$pid" 2>/dev/null; do
        if [ -f "$log" ]; then
            for pattern in "${FATAL_LOG_PATTERNS[@]}"; do
                if grep -qF -- "$pattern" "$log"; then
                    printf '%s\n' "$pattern" >"$flag"
                    lfd_log "FATAL: the training log says '$pattern'; stopping lerobot-train (pid $pid)"
                    kill -TERM "$pid" 2>/dev/null || true
                    return 0
                fi
            done
            # once training has started, the weights have loaded or failed: nothing more to watch
            if grep -qF "Start offline training" "$log"; then
                return 0
            fi
        fi
        lfd_nap 15
    done
}

lfd_offline_eval() { # evaluate checkpoints/last of the current run on the validation episodes
    local last="$TRAIN_DIR/checkpoints/last" step out
    if [ ! -f "$last/pretrained_model/config.json" ]; then
        echo "no checkpoint to evaluate in $TRAIN_DIR" >&2
        return 1
    fi
    step=$(lfd_checkpoint_step "$last")
    out="$RUN_DIR/offline_eval/step_${step:-unknown}.json"
    lfd_log "offline evaluation of step $step on the validation episodes -> $out"
    "$LEROBOT_PY" "$LFD_REPO/tools/lfd/offline_eval.py" --checkpoint "$last/pretrained_model" \
        --dataset-root "$DATASET_ROOT" ${SPLITS_FILE:+--splits "$SPLITS_FILE"} --out "$out" \
        --train-identity "$RUN_DIR/inputs.json" --device "$DEVICE" --video-backend "$VIDEO_BACKEND" "${EVAL_ARGS[@]}"
}

# ---- what a run is trained on ---------------------------------------------------------------------------
lfd_write_inputs() { # PIN_JSON (may be empty): $RUN_DIR/inputs.json, written once, before a fresh run trains
    LFD_PIN_JSON="${1:-}" "$(lfd_py)" - "$DATASET_ROOT" "$RUN_DIR/inputs.json" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(os.environ["LFD_REPO"]) / "tools" / "lfd"))
import offline_eval  # noqa: E402 - the identity is computed in one place, for training and evaluation

out = Path(sys.argv[2])
record = {
    "format": offline_eval.RUN_INPUTS_FORMAT, "version": 1, "created_wall": time.time(),
    "job": os.environ.get("JOB_TAG"), "policy": os.environ.get("POLICY_TAG"), "seed": os.environ.get("SEED"),
    "steps": os.environ.get("STEPS"),
    **offline_eval.dataset_identity(Path(sys.argv[1]), os.environ.get("SPLITS_FILE") or None),
    "pretrained": json.loads(os.environ["LFD_PIN_JSON"]) if os.environ.get("LFD_PIN_JSON") else None,
}
tmp = out.with_name(out.name + ".tmp")
tmp.write_text(json.dumps(record, indent=2) + "\n")
tmp.replace(out)
print(f"recorded the run's inputs in {out}: conversion.json {str(record['conversion_sha256'])[:12]}, "
      f"stats.json {str(record['stats_sha256'])[:12]}, splits {record['splits_sha256'][:12]}, label "
      f"{record['label']}, lookahead {record['lookahead_s']}"
      + (f", {record['pretrained']['repo']}@{record['pretrained']['revision'][:12]}" if record["pretrained"] else ""))
PY
}

# A resumed run must still be the comparison run. Refuse (no override) when the dataset's content changed since
# the run started (conversion.json, meta/stats.json, the splits file, the label, the lookahead), when its folder
# or train episodes changed, or when the pretrained revision would differ from the recorded one. STEPS must
# match too, unless LFD_EXTEND_STEPS=1, which is for the length only.
lfd_resume_gate() { # TRAIN_CONFIG_JSON
    "$(lfd_py)" - "$1" "$DATASET_ROOT" "$STEPS" "$RUN_DIR/inputs.json" <<'PY'
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(os.environ["LFD_REPO"]) / "tools" / "lfd"))
import offline_eval  # noqa: E402

config, root, steps, inputs_path = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
if not inputs_path.is_file():
    sys.exit(f"{inputs_path} is missing, so this run's dataset cannot be compared: it predates the content check. "
             "Start a new run (RUN_NAME)")
saved = json.loads(config.read_text())
recorded = json.loads(inputs_path.read_text())
try:
    now = offline_eval.dataset_identity(Path(root), os.environ.get("SPLITS_FILE") or None)
except offline_eval.SplitsError as exc:
    sys.exit(f"refusing to resume: {exc}")
problems = []
changes = offline_eval.identity_changes(recorded, now)
if changes:
    problems.append("the dataset's content changed since this run started (" + "; ".join(changes) + ")")
dataset = saved.get("dataset") or {}
if dataset.get("root") != root or dataset.get("episodes") != now["train_episodes"]:
    problems.append(f"the checkpoint trained on {dataset.get('root')} episodes {dataset.get('episodes')}, but the "
                    f"dataset or its splits.json now give {root} {now['train_episodes']}")
repo, pin = os.environ.get("PRETRAINED_REPO") or None, recorded.get("pretrained")
if repo:
    policy = saved.get("policy") or {}
    field = os.environ.get("PRETRAINED_FLAG", "--policy.pretrained_path").removeprefix("--policy.")
    if not pin:
        problems.append("the run's inputs.json records no pretrained revision")
    else:
        want = os.environ.get("PRETRAINED_REVISION")
        if pin.get("repo") != repo:
            problems.append(f"the run started from {pin.get('repo')}, this job is for {repo}")
        if want and want != pin.get("revision"):
            problems.append(f"PRETRAINED_REVISION={want}, but the run trains {repo}@{pin.get('revision')}")
        if str(policy.get(field)) != pin.get("path"):
            problems.append(f"the checkpoint was trained from {policy.get(field)}, but the run's record says "
                            f"{pin.get('path')} ({repo}@{pin.get('revision')})")
        if policy.get("use_peft"):  # LeRobot rebuilds the base from the adapter's own record
            adapter = config.parent / "adapter_config.json"
            base = json.loads(adapter.read_text()).get("base_model_name_or_path") if adapter.is_file() else None
            if base != pin.get("path"):
                problems.append(f"the LoRA adapter's base model is {base}, but the run's record says {pin.get('path')}")
if saved.get("steps") != steps and os.environ.get("LFD_EXTEND_STEPS") != "1":
    problems.append(f"this run was started for {saved.get('steps')} steps and STEPS is now {steps}: submit with "
                    f"STEPS={saved.get('steps')}, or add LFD_EXTEND_STEPS=1 to change its length on purpose")
if problems:
    sys.exit("refusing to resume: " + "\n  - ".join([""] + problems) + "\nStart a new run (RUN_NAME) instead.")
if os.environ.get("BATCH_SIZE") and str(saved.get("batch_size")) != os.environ["BATCH_SIZE"]:
    print(f"WARNING: BATCH_SIZE={os.environ['BATCH_SIZE']} is ignored on resume: the run keeps its batch size "
          f"{saved.get('batch_size')} (from its train_config.json)")
print("resume gate passed: the dataset's content, its train episodes, the steps and the pretrained revision "
      "match the run's inputs.json")
PY
}

lfd_record_job() { # ACTION: what this job ran, with versions and checksums, in $RUN_DIR/jobs/<job>.json
    mkdir -p "$RUN_DIR/jobs"
    LFD_ACTION="$1" LFD_JOB_ARGS="$(printf '%s\n' "${TRAIN_ARGS[@]}")" "$LEROBOT_PY" - \
        "$RUN_DIR/jobs/$JOB_TAG.json" <<'PY'
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def ver(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def sha256(path):
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def git(*args):
    try:
        return subprocess.run(["git", "-C", os.environ["LFD_REPO"], *args], capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


root = Path(os.environ["DATASET_ROOT"])
hub = Path(os.environ["HF_HOME"]) / "hub"
inputs_path = Path(os.environ["RUN_DIR"]) / "inputs.json"
inputs = json.loads(inputs_path.read_text()) if inputs_path.is_file() else None
record = {
    "job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(), "start_wall": time.time(),
    "policy": os.environ["POLICY_TAG"], "mode": os.environ.get("LFD_MODE", "train"),
    "action": os.environ["LFD_ACTION"], "seed": os.environ["SEED"], "steps": os.environ["STEPS"],
    "lerobot_train_args": os.environ["LFD_JOB_ARGS"].splitlines(),
    "repo_commit": git("rev-parse", "HEAD"), "repo_dirty": bool(git("status", "--porcelain")),
    "dataset_root": str(root), "info_sha256": sha256(root / "meta" / "info.json"),
    "stats_sha256": sha256(root / "meta" / "stats.json"),
    "splits_sha256": sha256(os.environ.get("SPLITS_FILE") or root / "splits.json"),
    "conversion_sha256": sha256(root / "conversion.json"),
    # what the run was started on (<run>/inputs.json): offline_eval.py compares its dataset against this
    "inputs": inputs,
    # the pinned pretrained weights this job trains: repo, revision (commit sha) and local path
    "pretrained": (inputs or {}).get("pretrained"),
    "versions": {p: ver(p) for p in ("lerobot", "torch", "transformers", "peft", "accelerate", "huggingface_hub")},
    "python": platform.python_version(),
    # What else the Hub cache resolved (the tokenizer and processor repos are not pinned).
    "hub_refs": {d.name: (d / "refs" / "main").read_text().strip()
                 for d in sorted(hub.glob("models--*")) if (d / "refs" / "main").is_file()},
}
Path(sys.argv[1]).write_text(json.dumps(record, indent=2) + "\n")
PY
}

# ---- the job --------------------------------------------------------------------------------------------
lfd_build_args() { # ACTION PIN_JSON: set TRAIN_ARGS
    local action=$1 pin=${2:-} episodes pinned_path
    if [ "$action" = resume ]; then
        # The checkpoint's train_config.json carries everything else (dataset, episodes, policy, LoRA, seed,
        # save_freq, the pinned pretrained path), so a resumed run cannot drift from the original.
        TRAIN_ARGS=(--config_path="$TRAIN_DIR/checkpoints/last/pretrained_model/train_config.json" --resume=true
            --steps="$STEPS")
        return 0
    fi
    episodes=$(lfd_train_episodes "$DATASET_ROOT") || lfd_die "could not read the train episodes of" \
        "$DATASET_ROOT from its splits.json"
    local -a pretrained=()
    if [ -n "${PRETRAINED_REPO:-}" ]; then
        pinned_path=$(lfd_json_get "$pin" path)
        if [ -z "$pinned_path" ] && [ "${LFD_PRINT_ARGS:-0}" != 1 ]; then
            lfd_die "no pinned local copy of $PRETRAINED_REPO: refusing to train an unpinned revision"
        fi
        # only LFD_PRINT_ARGS=1 can show the Hub id, when the Hub did not answer
        pretrained=("$PRETRAINED_FLAG=${pinned_path:-$PRETRAINED_REPO}")
    fi
    TRAIN_ARGS=(
        "${POLICY_ARGS[@]}"
        "${pretrained[@]}"
        --policy.device="$DEVICE"
        --policy.push_to_hub=false
        --dataset.repo_id="$REPO_ID"
        --dataset.root="$DATASET_ROOT"
        --dataset.episodes="$episodes"
        --dataset.video_backend="$VIDEO_BACKEND"
        --output_dir="$TRAIN_DIR"
        --job_name="lfd_${POLICY_TAG}_seed${SEED}"
        --seed="$SEED"
        --steps="$STEPS"
        --batch_size="$BATCH_SIZE"
        --num_workers="$NUM_WORKERS"
        --save_freq="$SAVE_FREQ"
        --log_freq="$LOG_FREQ"
        --eval_steps=0
        --env_eval_freq=0
        --wandb.enable=false
        "${SMOKE_ARGS[@]}"
    )
}

lfd_finish_run() { # the run reached its last step: keep what a reader needs, drop the rest
    lfd_drop_optimizer_state "$TRAIN_DIR/checkpoints"
    lfd_log "run folder now holds $(du -sh -- "$RUN_DIR" 2>/dev/null | awk '{print $1}') ($RUN_DIR)"
}

# The policy script sets POLICY_TAG, SEED, STEPS, BATCH_SIZE, POLICY_ARGS, the check settings (NEED_BF16,
# MIN_GPU_MEM_GB, NEED_QUANTILES, NEED_HF_TOKEN), the storage settings (CKPT_GB, BASE_GB, KEEP_LAST_DEFAULT)
# and, for a pretrained policy, PRETRAINED_REPO, PRETRAINED_FLAG and PRETRAINED_HIDE; then it calls lfd_main.
# LFD_MODE=eval only evaluates the last checkpoint. LFD_PRINT_ARGS=1 prints the lerobot-train arguments and
# exits before any check, lock or download (tools/lfd/train/test_configs.py parses them; the pretrained
# revision is resolved if the Hub answers). LFD_SMOKE=1 is the tiny CPU run of test_configs.py (ACT only).
lfd_main() {
    local mode=${LFD_MODE:-train} smoke=${LFD_SMOKE:-0} action config rc=0 step pin="" revision peak
    : "${POLICY_TAG:?}" "${SEED:?}" "${STEPS:?}" "${BATCH_SIZE:?}"
    declare -p POLICY_ARGS >/dev/null 2>&1 || lfd_die "the job script must set POLICY_ARGS"
    declare -p FATAL_LOG_PATTERNS >/dev/null 2>&1 || FATAL_LOG_PATTERNS=()
    case "$mode" in train | eval) ;; *) lfd_die "LFD_MODE must be train or eval, not '$mode'" ;; esac
    [ -n "${DATASET_ROOT:-}" ] || lfd_die "set DATASET_ROOT to the converted dataset (the folder holding" \
        "meta/info.json and splits.json); see tools/lfd/train/README.md"
    [ -d "$DATASET_ROOT" ] || lfd_die "no dataset folder at $DATASET_ROOT"
    DATASET_ROOT=$(cd "$DATASET_ROOT" && pwd)
    if [ -z "${REPO_ID:-}" ]; then # the id the converter wrote into conversion.json, else local/<folder>
        REPO_ID=$("$(lfd_py)" -c 'import json, sys; print(json.load(open(sys.argv[1]))["dataset"]["repo_id"])' \
            "$DATASET_ROOT/conversion.json" 2>/dev/null) || REPO_ID="local/$(basename "$DATASET_ROOT")"
    fi
    if [ -n "${PRETRAINED_REPO:-}" ]; then
        case "${PRETRAINED_FLAG:-}" in
            --policy.pretrained_path | --policy.base_model_path) ;;
            *) lfd_die "PRETRAINED_FLAG must be --policy.pretrained_path or --policy.base_model_path" ;;
        esac
    fi
    # a smoke run gets its own folder, so it can never resume, or be resumed by, a real run
    local run_tag=$POLICY_TAG
    if [ "$smoke" = 1 ]; then
        run_tag="$POLICY_TAG-smoke"
    fi
    RUN_NAME=${RUN_NAME:-$(basename "$DATASET_ROOT")/$run_tag/seed$SEED}
    RUN_DIR="$OUT_ROOT/$RUN_NAME"
    TRAIN_DIR="$RUN_DIR/train" # LeRobot's output_dir; a fresh run refuses an existing one
    JOB_TAG="${SLURM_JOB_ID:-local-$(date +%Y%m%d_%H%M%S)}"
    DEVICE=${DEVICE:-cuda}
    SAVE_FREQ=1000
    LOG_FREQ=${LOG_FREQ:-100}
    KEEP_LAST=${KEEP_LAST:-${KEEP_LAST_DEFAULT:-2}}
    [[ "$KEEP_LAST" =~ ^[0-9]+$ ]] && [ "$KEEP_LAST" -ge 1 ] || lfd_die "KEEP_LAST must be 1 or more, not '$KEEP_LAST'"
    NUM_WORKERS=${NUM_WORKERS:-$((${SLURM_CPUS_PER_TASK:-12} - 2))}
    # every 3rd frame (0.1 s): neighbouring frames are near duplicates, and the same stride for all three
    EVAL_ARGS=(--stride "${EVAL_STRIDE:-3}")
    SMOKE_ARGS=()
    if [ "$smoke" = 1 ]; then
        [ "$POLICY_TAG" = act ] || lfd_die "LFD_SMOKE=1 is for ACT only: the VLAs need a GPU"
        [ "$STEPS" -le 30 ] && [ "$BATCH_SIZE" -le 2 ] || lfd_die "smoke runs are capped at 30 steps, batch 2"
        DEVICE=cpu NUM_WORKERS=0 LOG_FREQ=5 SAVE_FREQ=${SMOKE_SAVE_FREQ:-10}
        # small images keep a CPU step to about a second; the same resize is applied when evaluating
        SMOKE_ARGS=(--dataset.image_transforms.enable=true --dataset.image_transforms.max_num_transforms=1
            '--dataset.image_transforms.tfs={"resize":{"weight":1.0,"type":"Resize","kwargs":{"size":[96,128]}}}')
        EVAL_ARGS=(--max-samples-per-episode 8 --num-workers 0 --warmup 1 --resize 96 128)
    fi
    # read by the Python helpers
    export POLICY_TAG SEED STEPS BATCH_SIZE DATASET_ROOT LFD_REPO RUN_DIR JOB_TAG
    export PRETRAINED_REPO="${PRETRAINED_REPO:-}" PRETRAINED_FLAG="${PRETRAINED_FLAG:-}"
    export PRETRAINED_REVISION="${PRETRAINED_REVISION:-}" SPLITS_FILE="${SPLITS_FILE:-}"

    if [ "${LFD_PRINT_ARGS:-0}" != 1 ]; then
        lfd_lock # before the run folder is looked at: one job per run folder
    fi
    config="$TRAIN_DIR/checkpoints/last/pretrained_model/train_config.json"
    if [ -f "$config" ]; then
        action=resume
    elif [ -e "$TRAIN_DIR" ]; then
        action=fresh-after-moving-aside # a job killed before its first checkpoint
    else
        action=fresh
    fi
    if [ "${LFD_PRINT_ARGS:-0}" = 1 ]; then
        if [ "$action" != resume ] && [ -n "$PRETRAINED_REPO" ]; then
            pin=$(lfd_pretrained "$PRETRAINED_REPO" "$PRETRAINED_REVISION" 0 2>/dev/null) || pin=""
        fi
        lfd_build_args "$action" "$pin"
        printf 'LFD_ACTION=%s\nLFD_RUN_DIR=%s\nLFD_PRETRAINED=%s\nLFD_TRAIN_ARGS_BEGIN\n' "$action" "$RUN_DIR" \
            "${pin:-${PRETRAINED_REPO:+unresolved: the job resolves the revision when it starts}}"
        printf '%s\n' "${TRAIN_ARGS[@]}"
        printf 'LFD_TRAIN_ARGS_END\n'
        return 0
    fi

    lfd_log "job $JOB_TAG on $(hostname): $POLICY_TAG seed $SEED, mode $mode, $action, run $RUN_DIR"
    if [ "$mode" = eval ] && [ "$action" != resume ]; then
        lfd_die "LFD_MODE=eval, but $TRAIN_DIR has no checkpoint to evaluate"
    fi
    if [ "$action" = resume ]; then
        lfd_resume_gate "$config" || lfd_die "refusing to resume $RUN_DIR (see above)"
    fi
    [ -x "$LEROBOT_PY" ] || lfd_die "no Python at $LEROBOT_PY: make the GPU env (README) or set LEROBOT_ENV"
    lfd_check_env "$POLICY_TAG" || lfd_die "environment check failed"
    if [ "$smoke" != 1 ]; then
        lfd_check_gpu "${NEED_BF16:-1}" "${MIN_GPU_MEM_GB:-0}" || lfd_die "GPU check failed"
    fi
    if [ "${LFD_SKIP_DATASET_CHECK:-0}" != 1 ]; then
        lfd_check_dataset "$DATASET_ROOT" "${NEED_QUANTILES:-0}" || lfd_die "dataset check failed"
    fi
    if [ "${NEED_HF_TOKEN:-0}" = 1 ] && [ -z "${HF_TOKEN:-}" ] && [ ! -s "$HF_HOME/token" ]; then
        lfd_die "no Hugging Face token ($HF_HOME/token or $LFD_HF_TOKEN_FILE): this policy loads a gated" \
            "repo; see the README, 'Hugging Face access'"
    fi
    lfd_log "Hugging Face token: $([ -n "${HF_TOKEN:-}" ] || [ -s "$HF_HOME/token" ] && echo present || echo absent)"
    if [ "$action" = resume ]; then
        step=$(lfd_checkpoint_step "$TRAIN_DIR/checkpoints/last")
        if [ "$mode" = train ] && [ "${step:-0}" -ge "$STEPS" ]; then
            lfd_log "this run already reached step $step of $STEPS: nothing to train"
            mode=eval
        fi
    fi

    # The pretrained weights: on a fresh start the commit the Hub's main points to now (or PRETRAINED_REVISION);
    # on a resume the one the run recorded. Resolved first (no download), so the quota check knows the size.
    if [ -n "$PRETRAINED_REPO" ]; then
        revision=$PRETRAINED_REVISION
        if [ "$action" = resume ]; then
            revision=$(lfd_json_get "$(cat "$RUN_DIR/inputs.json")" pretrained.revision)
        fi
        pin=$(lfd_pretrained "$PRETRAINED_REPO" "$revision" 0) || lfd_die "could not resolve the revision of" \
            "$PRETRAINED_REPO on the Hub"
    fi
    if [ "$smoke" != 1 ]; then
        peak=$(lfd_peak_gb "$pin" "$mode")
        lfd_quota_check "$peak" || lfd_die "share quota check failed"
    fi
    if [ -n "$PRETRAINED_REPO" ]; then
        revision=$(lfd_json_get "$pin" revision)
        lfd_log "pretrained weights: $PRETRAINED_REPO at commit $revision ($(lfd_json_get "$pin" resolved_from))"
        pin=$(lfd_pretrained "$PRETRAINED_REPO" "$revision" 1) || lfd_die "could not download" \
            "$PRETRAINED_REPO@$revision into $HF_HOME/hub"
        lfd_log "pretrained weights pinned: $(lfd_json_get "$pin" path)"
    fi

    if [ "$mode" = eval ]; then
        lfd_offline_eval || lfd_die "offline evaluation failed"
        step=$(lfd_checkpoint_step "$TRAIN_DIR/checkpoints/last")
        if [ "${step:-0}" -ge "$STEPS" ]; then
            lfd_finish_run
        fi
        return 0
    fi
    if [ "$action" = fresh-after-moving-aside ]; then
        local aside
        aside="$TRAIN_DIR.incomplete.$(date +%Y%m%d_%H%M%S)"
        lfd_log "moving the checkpoint-less $TRAIN_DIR aside to $aside (kept, not deleted)"
        mv -- "$TRAIN_DIR" "$aside"
    fi
    if [ "$action" != resume ]; then
        lfd_write_inputs "$pin" || lfd_die "could not record the run's inputs in $RUN_DIR/inputs.json"
    fi
    lfd_build_args "$action" "$pin"
    lfd_record_job "$action"

    local log="$RUN_DIR/jobs/$JOB_TAG.train.log" flag="$RUN_DIR/jobs/$JOB_TAG.fatal"
    cd "$RUN_DIR"
    lfd_log "lerobot-train ${TRAIN_ARGS[*]}"
    "$LEROBOT_ENV/bin/lerobot-train" "${TRAIN_ARGS[@]}" > >(tee -a "$log") 2>&1 &
    local train_pid=$!
    lfd_watch_log "$log" "$train_pid" "$flag" &
    local watch_pid=$!
    # Pruning must never stop silently: no errexit or nounset here, and every failure is logged and retried.
    (
        set +eu
        while lfd_nap "${LFD_PRUNE_INTERVAL_S:-300}"; do
            lfd_prune_checkpoints "$TRAIN_DIR/checkpoints" "$KEEP_LAST" ||
                lfd_log "WARNING: a pruning pass left something behind; the next pass tries again"
        done
        lfd_log "WARNING: the pruning loop ended while training ran"
    ) &
    LFD_BACKGROUND_PIDS="$watch_pid $!"
    trap lfd_stop_background EXIT
    wait "$train_pid" || rc=$?
    lfd_stop_background
    lfd_prune_checkpoints "$TRAIN_DIR/checkpoints" "$KEEP_LAST" ||
        lfd_log "WARNING: the final pruning pass left something behind (see above)"
    if [ -s "$flag" ]; then
        lfd_die "stopped: the log said '$(cat "$flag")', so the pretrained weights did not load"
    fi
    [ "$rc" = 0 ] || lfd_die "lerobot-train exited with status $rc (log: $log)"
    step=$(lfd_checkpoint_step "$TRAIN_DIR/checkpoints/last")
    if [ "${step:-0}" -lt "$STEPS" ]; then
        lfd_log "stopped at step ${step:-?} of $STEPS: submit the same command again to continue"
        return 0
    fi
    lfd_log "training reached step $step of $STEPS"
    if [ "${LFD_SKIP_EVAL:-0}" != 1 ]; then
        lfd_offline_eval || lfd_die "offline evaluation failed"
    fi
    lfd_finish_run
}
