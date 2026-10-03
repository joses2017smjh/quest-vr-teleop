# GPU and storage budgets for the LfD comparison: PROPOSALS

**Nothing here has been submitted. Every line is a proposal that needs the owner's approval, line by line,
before its `sbatch`.** The number to approve is the **cap**. The "expected" figures are estimates made
without a GPU run, and they are to be replaced by measured ones (below, "Calibrate").

Written 2026-10-03 for LeRobot 0.6.1. The scripts are `act.sbatch`, `pi05.sbatch` and `groot.sbatch` in this
folder. Roadmap §5.3 is the guide: ACT about 4 A40-hours per seed, and at most 48 A40-hours each for π0.5
LoRA and for GR00T.

## GPU-hours

| Line | Partition, GPU | Steps × batch | Expected s/step *(estimate)* | Expected hours per seed | Seeds | **Cap to approve** |
|---|---|---|---|---|---|---|
| ACT, from scratch | ampere, A40 | 100,000 × 8 | 0.1–0.2 | 3–5 A40-h | 3 (1000 for the live protocol; 2000 and 3000 for the offline spread only) | **18 A40-h** (6 per seed) |
| π0.5 LoRA | ampere, A40 | 30,000 × 32 | 2.5–3.5 | 21–29 A40-h | 1 (1000) | **48 A40-h** |
| GR00T N1.7 | ampere, A40 | 30,000 × 32 | 0.8–1.5 | 7–13 A40-h | 1 (1000) | **48 A40-h** |
| Calibration (optional, recommended) | ampere, A40 | 1,000 steps of π0.5 LoRA and of GR00T | — | ≤ 1 A40-h each | — | **2 A40-h** |
| **Total** | | | | **~39–59 A40-h expected** | | **116 A40-h** |
| *Option:* π0.5 full fine-tuning | dgxh, H100/H200 80 GB (`--constraint=vram80g`) | 30,000 × 32 | 1.5–2.5 *(estimate)* | 13–21 H100-h | 1 (1000) | **48 H100-h**, approved separately |
| *Option:* a second seed of a VLA | as its line | as its line | | as its line | +1 (2000) | **+48 A40-h each**, approved separately |

The offline evaluation of each final checkpoint runs inside its training job and is included above: every
3rd frame of the validation episodes, batch 1. It should take well under an hour for the VLAs and minutes for
ACT. A training job also spends 5–15 minutes on start-up: loading LeRobot from NFS, and the first job's
download of the pretrained weights.

**The scheduler's cap, separate from these budgets.** The `ampere` QoS lets each user hold at most
**2880 reserved GPU-minutes** at a time (`MaxTRESRunMinsPerUser` `gres/gpu=2880`) and 2 GPUs
(`MaxTRESPerUser` `gres/gpu=2`); `dgxh`'s QoS has the same 2880. Reserved means GPUs × the minutes left
until each running job's time limit, not the minutes used, so the limits are kept short: ACT 8 h (480), π0.5
and GR00T 24 h (1440). π0.5 plus GR00T at once fill the cap, and the user's other ampere jobs then wait.
π0.5 LoRA's 21–29 expected hours therefore take one or two 24-hour segments, each resuming the last; a
segment that ends early costs nothing. README §5 gives the order (ACT 1000 and 2000; then ACT 3000 with
π0.5's first segment; then π0.5's second segment with GR00T) and the `--dependency=afterany` chain.

**How the expected s/step were estimated** (no GPU was used to write this):
- **ACT:** 2 cameras at 640×480 through ResNet-18, batch 8, fp32 with TF32 matmuls. LeRobot's ACT page says "a
  few hours for 100k training steps" on one GPU. The roadmap's 4 A40-h is inside the range.
- **π0.5 LoRA:** with LoRA only on the action expert, no gradient flows into the 2.9 B PaliGemma prefix. The
  step is dominated by that prefix's bf16 forward pass: ~700 tokens per sample (2 images × 256 + up to 200
  text tokens), ≈ 4 TFLOP per sample, so ≈ 130 TFLOP per batch of 32, at an assumed 50–60 TFLOP/s on an A40.
- **GR00T:** a frozen Qwen3-VL backbone cut at layer 16, on ~250 tokens per sample, plus the action head's
  forward and backward passes. The Qwen3-VL image preprocessing runs on the CPU in the training loop, which
  may dominate; hence the wide range.
- **π0.5 full:** forward and backward through all 3.6 B parameters with gradient checkpointing, ≈ 0.7 PFLOP
  per batch, at an assumed ~400 TFLOP/s on an H100.

## Why these steps and batch sizes

- **Comparable training exposure.** ACT sees 100,000 × 8 = 0.8 M frames; π0.5 and GR00T see 30,000 × 32 =
  0.96 M frames each. With ~100 demonstrations of ~60 s (≈ 145k training frames after the 20 % validation
  hold-out), that is 5–7 passes over the data for every model.
- **Each model's own recipe otherwise.** ACT: LeRobot's defaults (batch 8, 100k steps, chunk 100). π0.5:
  openpi's and LeRobot's 30k-step cosine schedule with 1,000 warm-up steps, the PEFT guide's 10× learning
  rate for LoRA (2.5e-4), rank 32 as openpi's action-expert LoRA. GR00T: NVIDIA's N1.7 recipe as LeRobot
  ports it (AdamW 1e-4, 5 % warm-up, bf16 compute on fp32 weights, backbone frozen).
- **Batch 32 for the VLAs is a memory choice for a 48 GB A40, not a tuning result.** NVIDIA asks for 40 GB or
  more to fine-tune GR00T; openpi lists more than 22.5 GB for π0.5 LoRA, and gradient checkpointing is on.
- **Seeds.** The live protocol (`docs/LFD_EVAL_PROTOCOL.md` §3.1) runs one training run per policy: seed
  1000. The two extra ACT seeds are cheap, and give an offline run-to-run spread to read the VLAs' offline
  numbers against. They can be dropped without touching the live comparison. One seed per VLA keeps the
  first pass under ~50 A40-hours. A second VLA seed is worth buying only if the live evaluation is close;
  that is the optional line.
- **The fixed seed** is 1000, LeRobot's default, for every first run. GPU kernels are not bit-deterministic
  (`cudnn_deterministic` is off: it costs 10–20 %), so the same seed does not reproduce a run bit for bit.

## Storage on the share

**The share is already over its soft quota, and that must be resolved by the owner, outside this pipeline.**
The share is Lustre project 30762: a **1.5 TiB soft quota** and a **2 TiB hard limit**. On 2026-10-03 at
02:37 it held 1,630,827,120 KiB (**1.519 TiB, 19 GiB over the soft quota**), and the grace clock read
3 weeks 4 days 6 hours, so it ends around **28 October 2026**. When the grace ends, Lustre enforces the soft
quota as a hard limit: every write of the project fails, the robot's data sync into `bhl-data/incoming`
included, until usage is back under 1.5 TiB. Usage back under the soft quota stops and resets the clock.
Nothing in this folder can fix that: it needs either at least ~20 GiB (better 100+ GiB) freed elsewhere in the
project, or a larger soft quota from the HPC admins. Every GB these runs add makes it harder.

**What the jobs do about it** (`lfd_quota_check` in `common.sh`):
- Every job prints the usage against the soft quota and the grace left.
- A job refuses to start while usage is over the soft quota with **under 7 days of grace** left
  (`LFD_ALLOW_LOW_GRACE=1` overrides, for the owner only), and **always once the grace has expired**.
- A job refuses to start if its peak storage could cross the **hard limit**. The peak is
  (`KEEP_LAST` + 1) checkpoints, each the size of the run's newest checkpoint once there is one (the estimate
  below before), plus the pretrained weights when they are not in the cache yet, plus 1 GB.
- A job whose run would push usage over the soft quota, from below it, says it would start the grace clock.

**What the jobs keep.** LeRobot 0.6.1 keeps every checkpoint, and these jobs save one every 1,000 steps. So
each job prunes as it goes, every 5 minutes. It keeps the newest `KEEP_LAST` complete checkpoints, whatever
`checkpoints/last` points to, and any newer, half-written one (the save in progress). During a save there are
`KEEP_LAST` + 1. Proposed defaults:
- **`KEEP_LAST=1` for GR00T and π0.5 full**, whose checkpoints are 20+ GB. A resume needs only `last`, and
  `last` is never pruned; a half-written newer checkpoint is never mistaken for a complete one.
- `KEEP_LAST=2` for ACT and π0.5 LoRA, whose checkpoints are small.
- **After a finished run**, every checkpoint but `last` loses its optimizer state (`training_state/
  optimizer_state.safetensors` and `optimizer_param_groups.json`); its weights and `training_step.json` stay.
  `last` keeps its optimizer state, so a run can still be extended. `LFD_KEEP_OPTIMIZER=1` keeps everything.

| Run | Per checkpoint *(estimate)* | `KEEP_LAST` | Live peak (`KEEP_LAST` + 1) | Kept after the run | Pretrained weights, once, in `$HF_HOME` | Peak the job checks |
|---|---|---|---|---|---|---|
| ACT, per seed | ~0.65 GB (52 M parameters + AdamW) | 2 | ~2 GB | ~1 GB (the older one without AdamW state) | 45 MB (ImageNet ResNet-18, in `$TORCH_HOME`) | 4 GB |
| π0.5 LoRA | ~0.1 GB (adapters only) | 2 | < 1 GB | < 0.2 GB | 14.5 GB (`lerobot/pi05_base`, fp32) | 19 GB |
| π0.5 full | ~25 GB (3.6 B bf16 weights + AdamW) | 1 | ~50 GB | ~25 GB | 14.5 GB | 66 GB |
| GR00T N1.7 | ~22 GB (the whole 3 B model saved in fp32, + AdamW for the action head) | 1 | ~44 GB | ~22 GB | 6.9 GB (`nvidia/GR00T-N1.7-3B`, bf16) | 52 GB |

The proposed lines (3 ACT seeds, π0.5 LoRA, GR00T), at most two at a time, peak at about **50 GB** of
checkpoints (GR00T's 44 GB beside a small run, plus the finished ACT runs) plus 21 GB of pretrained weights, plus the dataset (a few GB to ~15 GB for 100 episodes of two
640×480 AV1 streams). Afterwards they keep about **25 GB** of checkpoints. The scripts never delete anything
but superseded checkpoints of their own run and, after the run, the optimizer state of non-`last` ones.

**Measure** (read-only):

```bash
lfs quota -h -p 30762 /nfs/hpc/share                  # used, soft, hard, and the grace left
lfs quota -p 30762 /nfs/hpc/share                     # the same in KiB, as the jobs read it
du -sh /nfs/hpc/share/$USER/bhl-data/train/*/*/seed*/train/checkpoints/*   # per checkpoint
du -sh /nfs/hpc/share/$USER/.cache/huggingface/hub/models--*               # pretrained weights
du -sh /nfs/hpc/share/$USER/bhl-data/*/ /nfs/hpc/share/$USER/*/ 2>/dev/null | sort -h | tail -20
```

**Reclaim** (the owner's call, each one; nothing here runs them):

```bash
R=/nfs/hpc/share/$USER/bhl-data/train/<dataset>/<policy>/seed<N>
ls -l $R/train/checkpoints/last                              # what must stay
rm -rf $R/train/checkpoints/<older step>                     # a superseded checkpoint
rm -f  $R/train/checkpoints/<older step>/training_state/optimizer_*   # its optimizer state only
rm -rf $R/train.incomplete.*                                 # a run killed before its first checkpoint
rm -rf /nfs/hpc/share/$USER/bhl-data/train/calib             # calibration runs, once read
```

A pretrained snapshot may be deleted only when no run that is still to be resumed or evaluated uses it
(`inputs.json` names it); the jobs download it again, the same commit, if needed.

**Correct the table after the first save of each run**:
`du -sh /nfs/hpc/share/$USER/bhl-data/train/<dataset>/<policy>/seed<N>/train/checkpoints/*`. The jobs
already use the measured size for their own peak once a checkpoint exists.

## Calibrate before the full runs (recommended)

The cheapest way to replace the estimates is a 1,000-step run of each VLA under its own name. It costs about
1 A40-hour each, and the first full job then starts from known numbers. These runs are for timing and memory
only. With `STEPS=1000` their learning-rate schedule is not the real one (π0.5's warm-up then spans the whole
run), so never use a `calib/` checkpoint for anything else, and delete them once read.

```bash
sbatch --time=02:00:00 --export=ALL,DATASET_ROOT=$DS,SEED=1000,STEPS=1000,LFD_SKIP_EVAL=1,RUN_NAME=calib/pi05_lora \
    tools/lfd/train/pi05.sbatch
sbatch --time=02:00:00 --export=ALL,DATASET_ROOT=$DS,SEED=1000,STEPS=1000,LFD_SKIP_EVAL=1,RUN_NAME=calib/groot \
    tools/lfd/train/groot.sbatch
```

Every 100 steps, LeRobot's log lines give `updt_s` (seconds per optimizer step), `data_s` (seconds waiting for
data), `smp/s` and `mem_gb` (peak GPU memory). Expected hours per seed = steps × (`updt_s` + `data_s`) / 3600.
If `mem_gb` is close to 48, lower the batch for that model and record why.

## What the owner approves

Answer per line: **yes, no, or a different cap.**

1. ACT, 3 seeds: 18 A40-h.
2. π0.5 LoRA, 1 seed: 48 A40-h (in 24-hour segments).
3. GR00T N1.7, 1 seed: 48 A40-h.
4. Calibration runs: 2 A40-h.
5. *(Option)* π0.5 full fine-tuning on dgxh (`--constraint=vram80g`): 48 H100-h.
6. *(Option)* a second seed for π0.5 LoRA and/or GR00T: 48 A40-h each.
7. Storage: about 50 GB of checkpoints at the peak during the runs, plus 21 GB of pretrained weights and the
   dataset, and about 25 GB of checkpoints kept afterwards, with `KEEP_LAST=1` for GR00T and π0.5 full and the
   optimizer state dropped after each run. **Approved only once the owner has brought the share below its
   1.5 TiB soft quota, or had the soft quota raised or the grace extended**; until then every job refuses to
   start with under 7 days of grace left.
