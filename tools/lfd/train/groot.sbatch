#!/bin/bash
#SBATCH --job-name=lfd-groot
#SBATCH --account=eecs
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --output=/nfs/hpc/share/%u/bhl-data/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/%u/bhl-data/logs/%x-%j.out

# NVIDIA GR00T N1.7 (nvidia/GR00T-N1.7-3B) fine-tuned on the LfD dataset as a new embodiment: the action head
# (projectors, flow-matching DiT, VL layer norm) trains; the Cosmos-Reason2 / Qwen3-VL backbone stays frozen.
#
# DO NOT SUBMIT until the owner has approved the budget in tools/lfd/train/budgets.md (at most 48 A40-hours
# per seed). Then, from the repo root:
#
#   mkdir -p /nfs/hpc/share/$USER/bhl-data/logs
#   sbatch --export=ALL,DATASET_ROOT=/nfs/hpc/share/$USER/bhl-data/lerobot/local/washcloth_v1,SEED=1000 \
#       tools/lfd/train/groot.sbatch
#
# NVIDIA asks for 40 GB or more to fine-tune, so an A40 (48 GB) on ampere; never the RTX 8000 `gpu`
# partition (no native bf16). Submitting the same command again, once the job has ended, resumes from the last
# checkpoint (one every 1000 steps); a job started while another holds the run folder's lock refuses. The last
# step triggers tools/lfd/offline_eval.py, and LFD_MODE=eval runs only that.
#
# Before the first run: accept the gated nvidia/Cosmos-Reason2-2B licence on huggingface.co (GR00T's image
# and text processor; approval is automatic) and make the token readable (README, "Hugging Face access").
# No flash-attn: LeRobot 0.6.1's GrootConfig.use_flash_attention defaults to false, so GR00T runs on PyTorch
# SDPA, whose results are close to flash-attention's but not bit-identical.
#
# Pinned weights: the first job resolves the commit nvidia/GR00T-N1.7-3B's main points to, downloads that
# commit (6.9 GB) into the share's Hugging Face cache, records it in the run's inputs.json, and passes a pinned
# view of it as --policy.base_model_path: links to every file of that snapshot except processor_config.json,
# statistics.json and embodiment_id.json (common.sh, lfd_pretrained, says why). LeRobot 0.6.1 then builds the
# same processor as for the Hub id: a fresh one for the new embodiment (slot 10 of LeRobot's own map), every
# observation.images.* key in alphabetical order (left, then right), so no --rename_map (none is allowed
# without a pretrained_path); state and action min-max normalized with the dataset's statistics and zero-padded
# from 12 to 132 inside the model. Every later job, and offline_eval.py, uses the same commit.

set -euo pipefail
if [ -z "${LFD_REPO:-}" ]; then
    for LFD_REPO in "${SLURM_SUBMIT_DIR:-}" "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." 2>/dev/null && pwd)" \
        "/nfs/hpc/share/$USER/quest-vr-teleop"; do
        [ -f "$LFD_REPO/tools/lfd/train/common.sh" ] && break
    done
fi
[ -f "$LFD_REPO/tools/lfd/train/common.sh" ] || { echo "cannot find the repo: set LFD_REPO" >&2; exit 1; }
# shellcheck source=common.sh
source "$LFD_REPO/tools/lfd/train/common.sh"

POLICY_TAG=groot
SEED="${SEED:-1000}"
STEPS="${STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NEED_BF16=1
MIN_GPU_MEM_GB=40
NEED_QUANTILES=0   # GR00T normalizes inside its own processor, by min and max
NEED_HF_TOKEN=1    # the Cosmos-Reason2 processor is gated
# Pinned by common.sh (lfd_pretrained); the view leaves out the raw checkpoint's processor sidecars.
PRETRAINED_REPO=nvidia/GR00T-N1.7-3B
PRETRAINED_FLAG=--policy.base_model_path
PRETRAINED_HIDE="processor_config.json statistics.json embodiment_id.json"
# Storage (budgets.md): LeRobot saves the whole 3 B model in fp32 at every checkpoint, plus AdamW state for
# the action head: ~22 GB a checkpoint (estimate). Only the newest is kept; 7 GB of base weights, once.
CKPT_GB=22
BASE_GB=7
KEEP_LAST_DEFAULT=1
FATAL_LOG_PATTERNS=()  # GR00T raises on a failed weight load, so lerobot-train itself fails

POLICY_ARGS=(
    --policy.type=groot
    --policy.embodiment_tag=new_embodiment
    # N1.7's native action horizon; LeRobot caps the chunk at 40. 40 steps = 1.3 s at 30 fps.
    --policy.chunk_size=40
    --policy.n_action_steps=40
    --policy.use_relative_actions=false
    --policy.use_bf16=true
    --policy.tune_llm=false
    --policy.tune_visual=false
    --policy.tune_projector=true
    --policy.tune_diffusion_model=true
    --policy.tune_vlln=true
    # the learning-rate warmup is 5% of this; left out, it is 5% of a deprecated default of 10000
    --policy.max_steps="$STEPS"
)

lfd_main
