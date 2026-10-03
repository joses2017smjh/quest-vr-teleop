#!/bin/bash
#SBATCH --job-name=lfd-pi05
#SBATCH --account=eecs
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --output=/nfs/hpc/share/%u/bhl-data/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/%u/bhl-data/logs/%x-%j.out

# π0.5 (lerobot/pi05_base, 3.6 B parameters) fine-tuned on the LfD dataset: LoRA by default, every weight
# with PI05_MODE=full.
#
# DO NOT SUBMIT until the owner has approved the budget in tools/lfd/train/budgets.md (LoRA: at most
# 48 A40-hours per seed; full: at most 48 H100-hours per seed). Then, from the repo root:
#
#   mkdir -p /nfs/hpc/share/$USER/bhl-data/logs
#   sbatch --export=ALL,DATASET_ROOT=/nfs/hpc/share/$USER/bhl-data/lerobot/local/washcloth_v1,SEED=1000 \
#       tools/lfd/train/pi05.sbatch
#
#   # full fine-tuning needs more than 70 GB of GPU memory (openpi), so an 80 GB H100 or H200 on dgxh. dgxh's
#   # full GPUs have no gres type (gpu:8), and dgxh-1's 40 GB MIG slices are the only typed gres
#   # (gpu:h100-40g), so a typed request cannot exclude them; the vram80g feature does (H100 80 GB and H200):
#   sbatch --partition=dgxh --constraint=vram80g \
#       --export=ALL,PI05_MODE=full,DATASET_ROOT=...,SEED=1000 tools/lfd/train/pi05.sbatch
#
# One job runs at most 24 hours: the ampere and dgxh QoS cap each user at 2880 reserved GPU-minutes, and a
# 2-day job would fill it alone. The 30,000 steps take about 21-29 A40-hours (budgets.md), so expect a second
# segment: submit the same command again once the job has ended, or chain it with --dependency=afterany
# (README, "Train"). Each segment resumes from the last checkpoint (one every 1000 steps); a job started while
# another holds the run folder's lock refuses. The last step triggers tools/lfd/offline_eval.py on the
# validation episodes, and LFD_MODE=eval runs only that.
#
# Never the RTX 8000 `gpu` partition: Turing has no native bf16, and the job refuses to start there.
#
# Before the first run: accept the gated google/paligemma-3b-pt-224 licence on huggingface.co (π0.5's
# tokenizer; access is reviewed by hand, so ask early) and make the token readable (README, "Hugging Face
# access"). The first job pins lerobot/pi05_base: it resolves the commit main points to, downloads that commit
# (14.5 GB) into the share's Hugging Face cache, records it in the run's inputs.json, and trains from that
# local snapshot. Every later segment, and offline_eval.py, uses the same commit.
#
# Camera keys: --policy.pretrained_path loads weights only, so the image features come from the dataset
# (observation.images.left/right) and no --rename_map is passed; the 12-D state is quantile-normalized and
# written into the prompt as 256-bin tokens, and the 12-D action is padded to 32 inside the model.

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

PI05_MODE="${PI05_MODE:-lora}"
POLICY_TAG="pi05_$PI05_MODE"
SEED="${SEED:-1000}"
STEPS="${STEPS:-30000}"     # openpi's fine-tuning length, and LeRobot's default cosine decay
BATCH_SIZE="${BATCH_SIZE:-32}"
NEED_BF16=1
NEED_QUANTILES=1            # π0.5 normalizes state and action by their 1st and 99th percentiles
NEED_HF_TOKEN=1             # the PaliGemma tokenizer is gated
# LeRobot 0.6.1 prints these and goes on training a randomly initialized model; the job stops instead.
# "Could not load state dict" covers both its "Warning: ..." and its "... from remote files" messages.
FATAL_LOG_PATTERNS=("Returning model without loading pretrained weights" "Could not load state dict")
# Pinned by common.sh (lfd_pretrained): the local snapshot of one recorded commit goes to this flag.
PRETRAINED_REPO=lerobot/pi05_base
PRETRAINED_FLAG=--policy.pretrained_path
BASE_GB=15                  # lerobot/pi05_base in the share's Hugging Face cache, once (fp32, 14.5 GB)

POLICY_ARGS=(
    --policy.type=pi05
    # weights only: features come from the dataset, so no --rename_map (it would break the batch)
    --policy.dtype=bfloat16
    --policy.gradient_checkpointing=true
    # pi05_base's own values, written out: 50 steps = 1.7 s at 30 fps
    --policy.chunk_size=50
    --policy.n_action_steps=50
    # absolute joint targets, as for every model here (roadmap §5.3: relative only once the zero is absolute)
    --policy.use_relative_actions=false
    --policy.scheduler_warmup_steps=1000
    --policy.scheduler_decay_steps="$STEPS"
)
case "$PI05_MODE" in
    lora)
        MIN_GPU_MEM_GB=40
        # LeRobot's default LoRA targets for π0.5: q/v projections of the action expert plus the action and
        # time projections; the PaliGemma VLM stays frozen. Rank 32 as openpi's action-expert LoRA. The
        # PEFT guide's rule: a LoRA learning rate about 10x the full fine-tuning one.
        POLICY_ARGS+=(
            --peft.method_type=LORA
            --peft.r=32
            --peft.lora_alpha=32
            --policy.optimizer_lr=2.5e-4
            --policy.scheduler_decay_lr=2.5e-5
        )
        # adapters and their optimizer state are small (~0.1 GB a checkpoint; rounded up)
        CKPT_GB=1
        KEEP_LAST_DEFAULT=2
        ;;
    full)
        # An 80 GB H100 reports 79.6 GiB to torch, an H200 about 132; a 40 GB MIG slice about 39.5: refused.
        MIN_GPU_MEM_GB=79
        POLICY_ARGS+=(
            --policy.freeze_vision_encoder=false
            --policy.train_expert_only=false
            --policy.optimizer_lr=2.5e-5
            --policy.scheduler_decay_lr=2.5e-6
        )
        # ~3.6 B bf16 weights plus AdamW state: ~25 GB a checkpoint (estimate). Only the newest is kept.
        CKPT_GB=25
        KEEP_LAST_DEFAULT=1
        ;;
    *)
        echo "PI05_MODE must be lora or full, not '$PI05_MODE'" >&2
        exit 1
        ;;
esac

lfd_main
