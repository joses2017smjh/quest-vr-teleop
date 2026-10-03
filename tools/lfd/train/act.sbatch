#!/bin/bash
#SBATCH --job-name=lfd-act
#SBATCH --account=eecs
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/nfs/hpc/share/%u/bhl-data/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/%u/bhl-data/logs/%x-%j.out

# ACT trained from scratch on the LfD dataset: the control every comparison keeps (roadmap §5.4).
# "From scratch" means no robot pretraining. Its ResNet-18 starts from ImageNet weights, as ACT always does.
#
# DO NOT SUBMIT until the owner has approved the budget in tools/lfd/train/budgets.md (about 4 A40-hours
# per seed). Then, from the repo root:
#
#   mkdir -p /nfs/hpc/share/$USER/bhl-data/logs        # Slurm cannot create the log folder itself
#   sbatch --export=ALL,DATASET_ROOT=/nfs/hpc/share/$USER/bhl-data/lerobot/local/washcloth_v1,SEED=1000 \
#       tools/lfd/train/act.sbatch
#
# Submitting the same command again, once the previous job has ended, resumes from the last checkpoint (one
# every 1000 steps); a job started while another holds the run folder's lock refuses. When the last step is
# reached, the job scores the final checkpoint with tools/lfd/offline_eval.py on the validation episodes;
# LFD_MODE=eval runs only that. Everything lands in /nfs/hpc/share/$USER/bhl-data/train/<dataset>/act/seed<SEED>/.
#
# 8 h rather than 2 days: the ampere QoS caps each user at 2880 reserved GPU-minutes (GPUs x the minutes left
# to each running job's time limit, not the minutes used), so asking for 2 days would hold back the user's
# other jobs for nothing. README, "Train", gives an order that fits the cap.

set -euo pipefail
# sbatch runs a spooled copy of this file, so the repo is found from the submit folder or the usual clone.
if [ -z "${LFD_REPO:-}" ]; then
    for LFD_REPO in "${SLURM_SUBMIT_DIR:-}" "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." 2>/dev/null && pwd)" \
        "/nfs/hpc/share/$USER/quest-vr-teleop"; do
        [ -f "$LFD_REPO/tools/lfd/train/common.sh" ] && break
    done
fi
[ -f "$LFD_REPO/tools/lfd/train/common.sh" ] || { echo "cannot find the repo: set LFD_REPO" >&2; exit 1; }
# shellcheck source=common.sh
source "$LFD_REPO/tools/lfd/train/common.sh"

POLICY_TAG=act
SEED="${SEED:-1000}"
NEED_BF16=0       # ACT trains in fp32
MIN_GPU_MEM_GB=16
NEED_QUANTILES=0  # ACT normalizes by mean and std
NEED_HF_TOKEN=0
# Storage (budgets.md, "Storage"): about 52 M parameters plus AdamW state, ~0.65 GB per checkpoint (an
# estimate, rounded up; the job measures the real size once a checkpoint exists). No Hub weights.
CKPT_GB=1
BASE_GB=0
KEEP_LAST_DEFAULT=2
FATAL_LOG_PATTERNS=()

if [ "${LFD_SMOKE:-0}" = 1 ]; then
    # test_configs.py only: a tiny ACT on CPU that downloads nothing (no ImageNet weights).
    STEPS="${STEPS:-30}"
    BATCH_SIZE="${BATCH_SIZE:-2}"
    POLICY_ARGS=(
        --policy.type=act
        --policy.chunk_size=10
        --policy.n_action_steps=10
        --policy.dim_model=64
        --policy.n_heads=2
        --policy.dim_feedforward=128
        --policy.n_encoder_layers=1
        --policy.n_vae_encoder_layers=1
        --policy.latent_dim=8
        --policy.pretrained_backbone_weights=null
    )
else
    STEPS="${STEPS:-100000}"  # LeRobot's ACT recipe: "a few hours for 100k steps" on one GPU
    BATCH_SIZE="${BATCH_SIZE:-8}"
    POLICY_ARGS=(
        --policy.type=act
        # LeRobot's defaults, written out so the run records them: 100 steps = 3.3 s at 30 fps
        --policy.chunk_size=100
        --policy.n_action_steps=100
        --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1
        --policy.use_amp=false
    )
fi

lfd_main
