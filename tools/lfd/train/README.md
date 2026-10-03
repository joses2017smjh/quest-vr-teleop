# Training three policies on one LfD dataset: ACT, π0.5 and GR00T N1.7

This folder trains three imitation-learning policies on the **same** LeRobot dataset with the **same**
train/validation split, on the OSU COE HPC with LeRobot 0.6.1. It then scores each one offline on the
held-out episodes:

- **ACT**, trained from scratch: the control every stage keeps (roadmap §5.4);
- **π0.5** (`lerobot/pi05_base`), fine-tuned with LoRA, or every weight as an option;
- **NVIDIA GR00T N1.7** (`nvidia/GR00T-N1.7-3B`), its action head fine-tuned for this new embodiment.

The recordings come from `run_teleop.py --record-tap` and `recorder.py` on the robot PC
(`docs/LFD_RECORDING_FORMAT.md`). `tools/lfd/convert_to_lerobot.py` turns them into a LeRobotDataset v3:
`observation.state[12]`, `action[12]`, `observation.images.left` and `.right`, 30 fps, with a `splits.json`
of whole episodes.

**Nothing here has been submitted.** Every job needs the owner's approval of its line in
[`budgets.md`](budgets.md) first. **The share is over its soft quota** (budgets.md, "Storage"): the owner must
resolve that outside this pipeline, and the jobs refuse to start once less than 7 days of grace are left.

| File | What it is |
|---|---|
| `common.sh` | Paths, caches (never home), the splits helper, the checks every job runs, the run lock, pinned pretrained weights, checkpoint pruning |
| `act.sbatch`, `pi05.sbatch`, `groot.sbatch` | One job per policy. Resubmitting the same command, once the job has ended, resumes it |
| `budgets.md` | Proposed GPU-hours, steps, batch sizes and storage, for approval |
| `test_configs.py` | Checks every job against the installed LeRobot without a GPU, with mocks for the Hub, lfs, squeue and the GPU |
| `../offline_eval.py` | Open-loop imitation error on the validation episodes, and inference time per call |

## 1. One-time setup

### Folders

```bash
mkdir -p /nfs/hpc/share/$USER/bhl-data/{incoming,lerobot,train,logs}
```

`logs/` must exist before any `sbatch`: every job writes `/nfs/hpc/share/$USER/bhl-data/logs/<job>-<id>.out`,
and Slurm cannot create the folder itself.

### Make the GPU env

`common.sh` expects a GPU-enabled Python 3.12 env at `LEROBOT_ENV`, by default
`/nfs/hpc/share/$USER/envs/lerobot-py312-gpu`. It does not exist yet. The CPU env beside it
(`lerobot-py312-cpu`) is for dry runs and tests only. Build the GPU env once, from the same uv-managed
Python 3.12 the CPU env uses, with every cache on the share:

```bash
export PIP_CACHE_DIR=/nfs/hpc/share/$USER/.cache/pip XDG_CACHE_HOME=/nfs/hpc/share/$USER/.cache
PY312=/nfs/hpc/share/$USER/Humanoid_Lite/.uv-python/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12
ENV=/nfs/hpc/share/$USER/envs/lerobot-py312-gpu
$PY312 -m venv $ENV
$ENV/bin/pip install --upgrade pip
# torch first, from PyTorch's CUDA 12.8 index: the default PyPI wheel may target a newer CUDA than the node driver
$ENV/bin/pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
$ENV/bin/pip install "lerobot[training,pi,groot,peft]==0.6.1"
$ENV/bin/python -c "import lerobot, torch; print(lerobot.__version__, torch.__version__, torch.version.cuda)"
```

- `training` brings the dataset reader (`datasets`, `av`) plus `accelerate`. `pi` brings `transformers` for
  π0.5. `groot` brings `diffusers`, `timm`, `dm-tree` and `decord`. `peft` brings LoRA.
- **No flash-attn.** LeRobot 0.6.1 builds GR00T with `use_flash_attention=false` (`GrootConfig`'s default),
  so GR00T runs on PyTorch SDPA directly. SDPA's results are close to flash-attention's, not bit-identical.
  π0.5 does not use flash-attn. `decord` comes with the `groot` extra as a py3-none wheel, and LeRobot never
  imports it.
- On a GPU node, `nvidia-smi` should report "CUDA Version" 12.8 or newer for the cu128 build.
- The CPU env already has LeRobot's dataset and training extras (`datasets` 4.8.5, `av` 15.1, `accelerate`),
  which `test_configs.py`'s tiny training needs. It has no `transformers`, `peft` or `scipy`, so π0.5 and
  GR00T load only in the GPU env.

### Hugging Face access

| Repo | Needed by | Licence and gate |
|---|---|---|
| `google/paligemma-3b-pt-224` | π0.5 (its tokenizer) | Gemma licence; **gated, reviewed by hand**: request it early |
| `nvidia/Cosmos-Reason2-2B` | GR00T N1.7 (its image and text processor) | NVIDIA Open Model License; gated, approved automatically once accepted |
| `lerobot/pi05_base` | π0.5's weights | not gated; **Gemma Terms of Use** (it is built on PaliGemma) |
| `nvidia/GR00T-N1.7-3B` | GR00T's weights | not gated; NVIDIA Open Model License |

1. Log in to huggingface.co with **the account whose token the jobs will use**. Access is granted per account.
2. Open `google/paligemma-3b-pt-224`, accept the Gemma licence, and wait for the approval e-mail.
3. Open `nvidia/Cosmos-Reason2-2B` and accept its licence.
4. Make a token of that same account: a classic **read** token, or a fine-grained token with "Read access to
   contents of all public gated repos you can access" ticked. A fine-grained token without it is refused by
   both gated repos.
5. Store it (below), then check it from the HPC:
   `HF_HOME=/nfs/hpc/share/$USER/.cache/huggingface hf download google/paligemma-3b-pt-224 tokenizer.json --local-dir /scratch/$USER/tmp/pg`

The token is read from a file, never from a job script, so it stays out of git, logs and
`scontrol show job`. Use either:
- `HF_HOME=/nfs/hpc/share/$USER/.cache/huggingface hf auth login`, which writes `$HF_HOME/token`; or
- a 0600 file named by `LFD_HF_TOKEN_FILE`. It defaults to the LeHome jobs' existing
  `/nfs/hpc/share/$USER/Humanoid_Lite/.hf_token`.

The π0.5 and GR00T jobs refuse to start without a token. Jobs print only "present" or "absent".

## 2. Sync the recordings

On the robot PC, copy finished sessions (`~/bhl_recordings/<session_id>/`) into `incoming/` on the share.
From home, the route is through the COE gateway:

```bash
rsync -av --partial -e "ssh -J <onid>@access.engr.oregonstate.edu" \
    ~/bhl_recordings/<session_id> <onid>@<submit-node>:/nfs/hpc/share/<onid>/bhl-data/incoming/
```

The hop home → `access.engr` → submit node has not been tested, and it may send a Duo push (roadmap §1.3,
§4.3). The write-only `rrsync` key of roadmap §4.7 waits until the LAN hole is closed (§4.9). This sync writes
to the same Lustre project as the training runs: once the soft-quota grace expires, it fails too.

## 3. Convert

`tools/lfd/convert_to_lerobot.py` is written separately. Its flags are `--sessions`, `--out-root`,
`--repo-id`, `--label` and `--lookahead`. Run it in a CPU job on `share` (roadmap §4.6):

```bash
cd /nfs/hpc/share/$USER/quest-vr-teleop
sbatch --account=eecs --partition=share --cpus-per-task=8 --mem=32G --time=02:00:00 \
    --output=/nfs/hpc/share/%u/bhl-data/logs/convert-%j.out --wrap '
    source tools/lfd/train/common.sh &&
    "$LEROBOT_PY" tools/lfd/convert_to_lerobot.py \
        --sessions "$INCOMING_ROOT"/20261005_* --out-root "$DATA_ROOT" \
        --repo-id local/washcloth_v1 --label meas_future --lookahead 0.10'
```

`DATASET_ROOT` below means **the folder the converter leaves `meta/info.json` in**: it writes to
`<out-root>/<repo-id>`, here `/nfs/hpc/share/$USER/bhl-data/lerobot/local/washcloth_v1`, with
`conversion.json` and `splits.json` beside the data. Its `--image-size WxH` shrinks each eye. All three
models then see the same smaller images, and ACT, which works at full resolution, trains faster.

`splits.json` is read from inside `DATASET_ROOT` first, then from the folder beside it. If the converter puts
it elsewhere, pass `SPLITS_FILE=/path/to/splits.json` to every job. The file must hold top-level `"train"` and
`"val"` lists of LeRobot `episode_index` values. The jobs refuse overlapping lists and out-of-range indices.

**Never re-convert into a folder a run trains on.** A run records the dataset by content when it starts
(`inputs.json`, §5), and every later job of that run, and `offline_eval.py`, refuses a changed one. Convert
into a new `--repo-id` instead, and start new runs on it.

```bash
source tools/lfd/train/common.sh
DS=$DATA_ROOT/local/washcloth_v1
lfd_train_episodes $DS      # [0,1,2,4,...], exactly what the jobs pass to --dataset.episodes
lfd_val_episodes   $DS      # what offline_eval.py scores
lfd_identity       $DS      # sha256 of conversion.json, meta/stats.json and splits.json; label; lookahead
```

## 4. Check before spending GPU time

```bash
/nfs/hpc/share/$USER/envs/lerobot-py312-cpu/bin/python tools/lfd/train/test_configs.py                 # ~3 min
/nfs/hpc/share/$USER/envs/lerobot-py312-cpu/bin/python tools/lfd/train/test_configs.py --dataset-root $DS  # ~5 min
```

The first command parses every job's `lerobot-train` arguments in the installed LeRobot: ACT, π0.5 LoRA,
π0.5 full and GR00T. It then runs each job script end to end on CPU, with a fake GPU env, a fake
`lerobot-train` and mocks for the Hub, `lfs`, `squeue` and `nvidia-smi`: the run lock, the pinned weights,
the quota and GPU checks, a resume, the resume refusals, pruning and the optimizer-state drop. It also
checks that GR00T's pinned view gives LeRobot's Hub-id processor, the pruning edge cases, the quota logic, the
GPU memory floor, and `offline_eval.py`'s refusal of another dataset.

The second command also trains a tiny ACT on CPU through `act.sbatch` itself: 20 steps, a changed
`conversion.json` refused, then a resubmission resumes it to 30, then `offline_eval.py` runs on the result,
on a copy of the dataset and on a different dataset. It trains on a shadow copy of `$DS` in its scratch
folder (small files copied, data folders linked), so `$DS` is never modified. That needs LeRobot's dataset and
training extras in that Python. The PEFT loading itself (π0.5 LoRA's base, then its adapter) runs only in an
env with `peft`, so only on a GPU run.

To see exactly what one job would run, without any checks, lock or download (it asks the Hub which commit
`main` is, read-only, when the Hub answers):

```bash
LFD_PRINT_ARGS=1 DATASET_ROOT=$DS SEED=1000 bash tools/lfd/train/groot.sbatch
```

## 5. Train

Only the lines the owner approved in `budgets.md`.

**The GPU-minutes cap.** The `ampere` QoS allows each user `MaxTRESRunMinsPerUser` `gres/gpu=2880` and
`MaxTRESPerUser` `gres/gpu=2` (`sacctmgr -n -P show qos format=Name,MaxTRESRunMinsPerUser,MaxTRESPerUser`;
`dgxh`'s QoS has the same 2880, and 8 GPUs). The 2880 counts *reserved* minutes: for every running
job, its GPUs times the minutes left until its time limit, not the minutes it has used. A 1-GPU job with a
24-hour limit counts 1440 when it starts and less as it runs. A job that would push the sum past 2880 stays
pending (reason `QOSMaxGRESRunMinutesPerUser`), and so does any other ampere job of the user, interactive
ones included. The limits here are ACT 8 h (480), π0.5 24 h (1440) and GR00T 24 h (1440), so at most two
jobs run at once, and π0.5 plus GR00T fill the cap.

**So do not submit everything at once.** Run them in this order:

1. ACT seeds 1000 and 2000 (2 × 480).
2. ACT seed 3000 and π0.5 LoRA, first segment (480 + 1440).
3. π0.5 LoRA, second segment, if the first did not reach 30,000 steps (21–29 A40-hours expected), and GR00T
   (1440 + 1440).

Either submit each step by hand once the previous jobs have ended (`squeue -u $USER`), or chain them with
`--dependency=afterany`, which starts a job once the named ones have ended, whatever their outcome:

```bash
cd /nfs/hpc/share/$USER/quest-vr-teleop
DS=/nfs/hpc/share/$USER/bhl-data/lerobot/local/washcloth_v1
T=tools/lfd/train
a1=$(sbatch --parsable --export=ALL,DATASET_ROOT=$DS,SEED=1000 $T/act.sbatch)
a2=$(sbatch --parsable --export=ALL,DATASET_ROOT=$DS,SEED=2000 $T/act.sbatch)
a3=$(sbatch --parsable --dependency=afterany:$a1 --export=ALL,DATASET_ROOT=$DS,SEED=3000 $T/act.sbatch)
p1=$(sbatch --parsable --dependency=afterany:$a2 --export=ALL,DATASET_ROOT=$DS,SEED=1000 $T/pi05.sbatch)
p2=$(sbatch --parsable --dependency=afterany:$p1 --export=ALL,DATASET_ROOT=$DS,SEED=1000 $T/pi05.sbatch)
g1=$(sbatch --parsable --dependency=afterany:$a3 --export=ALL,DATASET_ROOT=$DS,SEED=1000 $T/groot.sbatch)
```

`p2` is π0.5's second segment: it resumes where `p1` stopped, or, if `p1` already reached the last step, only
re-runs the offline evaluation. If the first segment fails (a refused start, for example), the second starts and
fails the same way: read `p1`'s log before going on.

The option, approved separately: π0.5 with every weight trained needs more than 70 GB, so an 80 GB H100 or
H200 on `dgxh`. dgxh's full GPUs carry no gres type (`sinfo -p dgxh -o "%N %G %f"` shows `gpu:8`), and
dgxh-1's 40 GB MIG slices are the only typed gres (`gpu:h100-40g`), so a typed request cannot exclude the
slices. The `vram80g` feature does:

```bash
sbatch --partition=dgxh --constraint=vram80g \
    --export=ALL,DATASET_ROOT=$DS,SEED=1000,PI05_MODE=full tools/lfd/train/pi05.sbatch
```

The job also refuses, at start, any GPU under `MIN_GPU_MEM_GB=79` GiB (an 80 GB H100 reports 79.6 to torch).

Each run lives in `/nfs/hpc/share/$USER/bhl-data/train/<dataset>/<policy>/seed<SEED>/`:

- `train/`: LeRobot's output. `checkpoints/<step>/pretrained_model` holds the weights, configs and
  normalization stats; `checkpoints/last` points to the newest complete one.
- `inputs.json`: what the run was started on, written before its first training step. That is the sha256 of
  the dataset's `conversion.json`, `meta/stats.json` and `splits.json`, the label and lookahead, the train
  episodes, and the pinned pretrained weights: repo, commit sha and local path.
- `jobs/<job id>.json`: what each job ran, written before `lerobot-train` starts. That covers its arguments,
  the repo commit, the dataset checksums, `inputs.json`, the pinned weights, the package versions, and what
  else the Hub cache resolved (the tokenizer and processor repos, which are not pinned).
- `jobs/<job id>.train.log`: LeRobot's output, also in the Slurm log.
- `offline_eval/step_<N>.json`: the offline report for step N.
- `.lock`: the run lock (§6).

**Pinned pretrained weights.** A run trains exactly one commit of its pretrained weights. The first job asks
the Hub which commit `main` points to (`HfApi().model_info(repo).sha`, read-only), downloads that commit into
the share's Hugging Face cache, records it in `inputs.json` and the job record, and passes the local copy to
LeRobot. π0.5 gets the snapshot folder as `--policy.pretrained_path`. GR00T gets a pinned view of it as
`--policy.base_model_path`: `$HF_HOME/lfd-pins/nvidia--GR00T-N1.7-3B/<sha>/`, links to every file of the snapshot
except `processor_config.json`, `statistics.json` and `embodiment_id.json`. Handed the raw snapshot, LeRobot
0.6.1 would build GR00T's processor from those three files. They map no `new_embodiment`, so the embodiment
would silently fall back to slot 0 (`simpler_env_google`), and the checkpoint's state dropout (0.2) and
albumentations would switch on. Without them, the processor is built exactly as for the Hub id. Pass
`PRETRAINED_REVISION=<sha>` to pin a different commit on a fresh run. GR00T's view path is verified on CPU
only; its first GPU run is its real test.

Other settings can be passed through `--export`: `STEPS`, `BATCH_SIZE`, `KEEP_LAST` (complete checkpoints
kept: by default 2 for ACT and π0.5 LoRA, 1 for GR00T and π0.5 full), `RUN_NAME` (a different run folder),
`EVAL_STRIDE` (default 3), `LEROBOT_ENV`, `OUT_ROOT` and `SPLITS_FILE`. The partitions are A40s on `ampere` and
H100s/H200s on `dgxh`. **Never use the RTX 8000 `gpu` partition for π0.5 or GR00T**: Turing has no native
bf16. The jobs check with `torch.cuda.is_bf16_supported(including_emulation=False)` and refuse to run there.
The plain call answers True on Turing, because PyTorch emulates bf16.

## 6. Resume

Submit **the same command again, once the previous job has ended**. A checkpoint is saved every 1,000 steps,
and the job continues from `checkpoints/last` with LeRobot's `--config_path=…/train_config.json --resume=true`.
The optimizer, scheduler, step counter and data order all carry over.

**One job per run folder.** Each job holds an flock on `<run>/.lock` (the share is mounted with `flock`) for
its whole life. A job that starts while another holds it refuses: "another job holds …". A lock whose holder's
Slurm job is no longer in `squeue` (or has ended) is stale, and the next job takes it over, says so, and keeps
the old lock file as `.lock.stale.<time>`.

**What a resume refuses**, with no override (start a new `RUN_NAME` instead):

- the dataset's content changed since the run started: the sha256 of `conversion.json`, `meta/stats.json` or
  the splits file, or the label or the lookahead differ from `inputs.json`;
- the dataset folder or its train episodes changed;
- the pretrained weights would differ: another repo, a `PRETRAINED_REVISION` other than the recorded commit,
  or a checkpoint (or, for LoRA, its adapter's base) trained from another path than the recorded one;
- `inputs.json` is missing.

`STEPS` must match too. Pass the same `STEPS`, or add `LFD_EXTEND_STEPS=1` to change the length on purpose.
`LFD_EXTEND_STEPS` changes the length only; it never lets a changed dataset or revision through.

**What a resume ignores.** Only `STEPS` is read from the environment. Everything else comes from the
checkpoint's `train_config.json`: the batch size, `NUM_WORKERS`, `POLICY_ARGS` edits, the learning-rate schedule,
the seed and the pinned path. The job prints a warning when `BATCH_SIZE` differs from the run's.
`LFD_EXTEND_STEPS` changes `--steps` only: π0.5's cosine decay keeps its original `scheduler_decay_steps`
(the first `STEPS`), so the extra steps run at the final learning rate, and GR00T's warm-up stays 5 % of its
original `max_steps`.

A job killed before its first checkpoint leaves a `train/` folder with nothing to resume. The next
submission moves it aside to `train.incomplete.<time>` and starts over; it never deletes it. A run that has
already reached its last step only runs the offline evaluation.

## 7. Offline evaluation

When a job reaches its last step, it runs `tools/lfd/offline_eval.py` on `checkpoints/last`:

- **Data:** the validation episodes only, batch 1, every 3rd frame by default (`EVAL_STRIDE`). Compare
  models only at the same `settings.stride` in their reports, and keep it at 3 for all three unless the owner
  changes it for all.
- **What it checks first:** that the dataset is the one the run was trained on, by the sha256 of its
  `conversion.json` against the run's `inputs.json`. A re-conversion or a copy can put other demonstrations
  behind the same episode indices, so a different or unknown dataset is refused; `--allow-different-dataset`
  is for debugging only, and the report then says "different dataset", never "ok". Then it refuses any
  checkpoint whose `train_config.json` lists a validation episode, that was trained on every episode, or whose
  training sources include a validation source.
- **What it reports,** in the dataset's units (rad for the 10 joints, 0..1 for the 2 claws):
  - **to compare across policies:** L1 at horizons of 1, 5, 10, 20, 30 and 40 frames, for the joints and the
    claws apart (`l1_by_horizon_joints_rad`, `l1_by_horizon_claws`, and per joint), and per-joint L1 over the
    common horizon, the first 40 steps (`common_horizon_l1`). Every policy predicts at least 40 steps, and
    these are computed on the same (frame, step) pairs for all three;
  - per-joint L1 of the first predicted action;
  - **per policy only:** L1 over the policy's own whole chunk (`chunk_l1`). ACT averages over 100 steps,
    π0.5 over 50 and GR00T over 40, so these are not comparable across policies;
  - the same numbers for a "hold the current state" baseline;
  - inference time per call (preprocess, model and postprocess) on the job's GPU.

To run the evaluation alone, for example after a job ran out of time during it:

```bash
sbatch --time=02:00:00 --export=ALL,DATASET_ROOT=$DS,SEED=1000,LFD_MODE=eval tools/lfd/train/groot.sbatch
```

Or by hand on any node with the env (`--device cpu` works for ACT):

```bash
source tools/lfd/train/common.sh
RUN=$OUT_ROOT/washcloth_v1/act/seed1000
"$LEROBOT_PY" tools/lfd/offline_eval.py --checkpoint $RUN/train/checkpoints/last/pretrained_model \
    --dataset-root $DS --out $RUN/offline_eval/manual.json --stride 3
```

Source `common.sh` first: without it the datasets cache (and, for ACT, torchvision's ImageNet weights) would
land in home, and `offline_eval.py` refuses to start. It finds the run's `inputs.json` from the checkpoint
path; `--train-identity` names it otherwise. GR00T and π0.5 LoRA also need the pinned weights in the share's
cache (the GR00T view, or the π0.5 snapshot named in the adapter); a job of the run re-creates them if the
cache was cleared.

**This measures imitation error, not task success,** and the report says so in its first field. Open-loop
error cannot see compounding drift, contact or timing. A policy can score worse here and still succeed more
often on the robot, or the reverse.

## 8. Hand-off to the live evaluation

Offline numbers select nothing. Which policy is better is decided by the matched live evaluation in
[`docs/LFD_EVAL_PROTOCOL.md`](../../../docs/LFD_EVAL_PROTOCOL.md).

**What the protocol takes from here:**
- **The checkpoint.** It picks by rule the last checkpoint of a run with a preregistered number of steps.
  That is `checkpoints/last` once the job has logged "training reached step N of N", with N from
  `budgets.md`.
- **The live run per policy.** The protocol runs one per policy: seed 1000. Its ids `act`, `pi05` and
  `groot_n17` are the run folders `act/`, `pi05_lora/` (or `pi05_full/`) and `groot/`.
- **The fields to copy from that step's `offline_eval/step_<N>.json`:**
  - `checkpoint.path` and `checkpoint.weights_sha256`;
  - `dataset.conversion_sha256` and `dataset.splits_sha256`;
  - `checkpoint.pretrained` (repo, commit sha and local path) for π0.5 and GR00T. For π0.5 LoRA this is
    essential: `weights_sha256` covers only the adapter, and the base is that commit;
  - the step and the seed.

Keep every run in the tables, failures included. Inference time per call helps plan serving (roadmap §4.4).
Running a policy on the robot also needs the `POLICY` state and the policy client and server of roadmap
§4.4–4.5, none of which exist yet (protocol §0).

## Why every model gets the identical dataset and split

The comparison is meant to measure the model, so everything else is held fixed:

- **One converted dataset.** All three load the same folder: the same episodes and frames, the same action
  label, the same image decoding (PyAV for all), and the same `meta/stats.json` normalization statistics. Each
  model uses those statistics in its own way (table below), but no model gets different data. Every run
  records the dataset by content, and refuses to resume, or to be scored, on another.
- **One `splits.json` of whole episodes,** read by one parser. Training gets `--dataset.episodes=<train list>`
  and `offline_eval.py` scores only the `val` list. Neighbouring frames of one demonstration never land on
  both sides, and no model sees an episode another model was scored on.
- **The same seeds, the same 30 fps time grid, absolute joint targets for all** (relative actions only once
  the zero is absolute: roadmap §5.3), and comparable training exposure (budgets.md).
- **One known, shared leak.** LeRobot normalizes with `meta/stats.json`, which the converter computes over
  all episodes, validation included. The leak is summary statistics only, and identical for all three, so
  it cannot favour one model.

## Which action label

Every converted dataset carries one label (`conversion.json`, format v1 §4):
- `cmd`: what the controller was told (`qc`). Valid for teleop only.
- `meas_future`: where the arm actually went `--lookahead` seconds later, by default 0.10 s.

Train these comparisons on **`meas_future`**:
- **For any comparison of teleop against hand-guided demonstrations, both must be `meas_future`.** While the
  operator moves a compliant arm, `qc` is not a demonstration: as a stiff "hold" command it would be exactly
  the wrong label.
- Keeping `meas_future` for the model comparison too means its numbers stay comparable with guided data
  added later.
- **Never mix labels across models.** Because every model reads the same converted dataset, they cannot
  differ by construction. Use `cmd` only for a teleop-only study, and then for all three.

## How each model sees the same data

| | ACT | π0.5 | GR00T N1.7 |
|---|---|---|---|
| Loaded as | `--policy.type=act`, from scratch (ResNet-18 from ImageNet, as ACT always does) | `--policy.type=pi05 --policy.pretrained_path=<pinned snapshot of lerobot/pi05_base>` (weights only) | `--policy.type=groot --policy.base_model_path=<pinned view of nvidia/GR00T-N1.7-3B> --policy.embodiment_tag=new_embodiment` |
| Cameras | both eyes as stored | the dataset's keys, no `--rename_map`; each eye resized with padding to 224×224 | the dataset's keys in alphabetical order (left, right); resized to 256 and cropped to 230 |
| State, 12-D | mean/std-normalized input | q01/q99 to [-1, 1], then 256-bin tokens in the prompt | min/max to [-1, 1], zero-padded to 132 |
| Action, 12-D | mean/std; a chunk of 100 (3.3 s) | q01/q99, padded to 32; a chunk of 50 (1.7 s) | min/max, padded to 132; a chunk of 40 (1.3 s) |
| Task text | not used | in the prompt | in the prompt |
| Trained | every weight, fp32 | LoRA rank 32 on the action expert's q/v and the action/time projections (PaliGemma frozen), bf16; or every weight | the action head (projector, flow-matching DiT, VL norm); backbone frozen; bf16 compute on fp32 weights |

Why there is no rename map. With `pretrained_path` (π0.5) or `base_model_path` (GR00T), the features come
from the dataset. LeRobot's `validate()` refuses `--rename_map` without a pretrained policy path, and the
π0.5 docs warn that adding one breaks the batch. Loading π0.5 with `--policy.path=lerobot/pi05_base` instead
would keep the checkpoint's own three camera names and need
`--rename_map='{"observation.images.left": "observation.images.base_0_rgb", ...}'`. That route is not used.

π0.5 normalizes with QUANTILES, so it needs `q01`/`q99` in `meta/stats.json`. The converter recomputes every
quantile entry (q01, q10, q50, q90, q99) as true dataset quantiles over all frames after LeRobot's
`finalize()`, and records that in `conversion.json` (format v1 §4). The π0.5 job refuses a dataset without
q01/q99, and warns when `conversion.json` does not record the recomputation. Every job refuses a
`meta/stats.json` holding NaN or Infinity, which is not JSON. To fix an older dataset, **re-run the converter**
(`tools/lfd/convert_to_lerobot.py`) on the same sessions into a new dataset root, then train **all three**
models on the new one, so they still share one dataset.

Do not use LeRobot's `lerobot-edit-dataset --operation.type recompute_stats` for this. LeRobot's own
quantiles are not dataset quantiles: it estimates each episode's from a histogram, then writes their
count-weighted mean, and `recompute_stats` does the same again. That is why the converter recomputes them
itself. Do not use `augment_dataset_quantile_stats.py` either: it pushes the dataset to the Hub.

## What the jobs check, and what they refuse

At start, every job prints and checks:
- The run lock: one job per run folder (§6).
- On a resume, the dataset's content, the train episodes, the steps and the pinned revision (§6).
- Python 3.12, LeRobot 0.6.1, and the packages the policy imports.
- `nvidia-smi`, the GPU, its compute capability and native bf16, and its memory: at least 16 GiB for ACT,
  40 GiB for π0.5 LoRA and GR00T, 79 GiB for π0.5 full (a 40 GB MIG slice is refused).
- The dataset contract: 12-D state and action, both cameras, 30 fps, strict-JSON statistics, and quantiles
  for π0.5.
- The Hugging Face token, for the gated repos.
- The share's quota (budgets.md, "Storage"). It always prints the usage against the soft quota and the grace
  left. It refuses while usage is over the soft quota with under 7 days of grace (`LFD_ALLOW_LOW_GRACE=1` to
  override); always once the grace has expired; and whenever this run's peak storage could cross the hard
  limit. The peak is (`KEEP_LAST` + 1) checkpoints, each the size of the run's newest one once there is one
  (an estimate before), plus the pretrained weights when they are not cached yet.
- The pretrained weights, pinned to one commit (§5).

While training:
- They delete superseded checkpoints every 5 minutes. They keep the newest `KEEP_LAST` complete ones, whatever
  `last` points to, and anything newer than the newest complete one (a save in progress). A checkpoint without
  its weights and `train_config.json` is half-written; one older than the newest complete one is deleted. A
  failed delete is logged and tried again on the next pass; it never stops the pruning.
- For π0.5, they stop the run if the log shows LeRobot 0.6.1's "Returning model without loading pretrained
  weights" or "Could not load state dict". That loader prints these and carries on training a randomly
  initialized model.

After the last step: the offline evaluation, then every checkpoint but `last` loses its optimizer state
(`LFD_KEEP_OPTIMIZER=1` keeps it).

## LeRobot 0.6.1 facts these scripts rely on

All of these were checked in the installed source:
- `--output_dir` must not exist on a fresh run, so LeRobot's output goes to `<run>/train`.
- `--policy.push_to_hub` defaults to true and then demands a `repo_id`, so every job passes `false`.
- GR00T's learning-rate warm-up is 5 % of `--policy.max_steps`, a field otherwise left at a deprecated 10,000.
  The job passes `--policy.max_steps=$STEPS`. GR00T's chunk is capped at the checkpoint's 40-step horizon.
- `--policy.pretrained_revision` does not reach π0.5's weight download in 0.6.1, and GR00T's loader passes no
  revision at all, so the jobs pin by passing a local copy of one commit instead (§5). π0.5's loader reads
  `model.safetensors` from a local folder, and LoRA records that folder as its adapter's base
  (`wrap_with_peft` sets the base to `pretrained_path`), so a resume and `offline_eval.py` load the same commit.
  GR00T treats a local folder holding a raw N1.7 checkpoint differently from the Hub id (its processor
  sidecars), hence the pinned view. At writing, `lerobot/pi05_base` was at `b211f3d4` and
  `nvidia/GR00T-N1.7-3B` at `2fc962b9`.
- A checkpoint is written as the weights, `train_config.json`, the processors, then `training_state/`, and
  `last` moves to it only after the whole save.
- π0.5 and GR00T sample noise for every chunk, so `offline_eval.py` fixes `--seed` (0). ACT is deterministic
  at inference.
