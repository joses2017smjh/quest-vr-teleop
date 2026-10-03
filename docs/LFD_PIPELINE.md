# Learning-from-demonstration comparison: the pipeline

**Goal.** Train π0.5 and NVIDIA GR00T N1.7 on the same demonstrations of one simple task, with a from-scratch ACT run as a control. Then compare them on the robot under matched, preregistered conditions.

A second study, held for later, keeps the model fixed and compares teleoperated demonstrations against hand-guided ones.

This page gives the order of work. The details live in:

| Topic | Page |
|---|---|
| The data contract | `docs/LFD_RECORDING_FORMAT.md` |
| Training | `tools/lfd/train/README.md` |
| The live test | `docs/LFD_EVAL_PROTOCOL.md` |
| Hand guiding | `docs/LFD_HAND_GUIDING.md` |
| Why and what comes next | `docs/ROADMAP_2026-10-02.md` |

## What exists, and how far it has been tested

Nothing here has run on the robot yet. Tests ran on the HPC: `SimArmDriver`, a fake camera writing real JPEG files, and a fake servo link.

| Part | State |
|---|---|
| `run_teleop.py --record-tap` → `scripts/teleop/recorder.py`, controlled by `tools/lfd/record_ctl.py` or "agent, start/end episode …" | Built. With the tap off, nothing changes. With it on, 200 armed ticks gave identical commands and limits to a tap-off run. Recorded episodes are byte-exact; aborts, refusals and the low-disk guard are tested. |
| `tools/lfd/zero_check.py` (the roadmap §3.1 shoulder-roll test, recorded) | Built and tested |
| `tools/lfd/validate_recording.py`, `convert_to_lerobot.py` → LeRobotDataset v3 | Built. A session made by the real recorder converts, and state/action match an independent recompute to 3e-8. Quantile stats are exact. |
| `tools/lfd/train/` (ACT, π0.5 LoRA or full, GR00T N1.7 sbatch jobs) | Written. A tiny ACT run through `act.sbatch` worked on CPU, with resume, locks and refusals. **Never submitted; the GPU budgets are proposals.** |
| `tools/lfd/offline_eval.py` | Built. Tested with ACT on CPU. It measures imitation error, not task success. |
| `tools/lfd/eval_schedule.py`, `eval_log.py`, `analyze_eval.py` | Built. The statistics are checked against scipy. |
| A `POLICY` state and policy client, to run a trained policy on the arms | **Not built.** Live evaluation needs it (roadmap §4.4–4.5). |
| A `GUIDE` state for hand-guided demonstrations | **Not built.** Assessed as "supported after six steps" (`docs/LFD_HAND_GUIDING.md` §6). |
| Live serving from the HPC | Needs COE IT's OK (roadmap §4.2), or the cloud-GPU fallback (§4.8) |

## 0. Before collecting the main dataset (once)

1. **Shoulder-roll zero.** Run the 5-minute inclinometer test (roadmap §3.1), then record it:
   ```bash
   .venv/bin/python tools/lfd/zero_check.py --torso-deg 0.4 --left-upper-arm-deg 0.8 --right-upper-arm-deg -0.2
   ```
   - The script writes `configs/zero_check.json`, and every episode carries a copy of it.
   - The converter refuses episodes without a conclusive check, unless you pass `--allow-unverified-zero`.
   - If the test says the zero is ≈15° off ("holds"), land the corrected zero first (roadmap §3.1). Episodes under different zero conventions are never mixed.
2. **The LAN exposure** (roadmap §4.9) is still open in this branch: UDP 11005 on `0.0.0.0` and the unauthenticated :8080 page. Close it before any OSU key goes on the robot PC.
3. **The task:** move a foam block into a bowl within 30 s, one arm, inside the measured reach (`docs/LFD_EVAL_PROTOCOL.md` §1).
4. **A fixed camera preset.** The recorder refuses to start unless the camera is `still`.

## 1. Record, on the robot PC

```bash
# a window of its own; recordings go to ~/bhl_recordings
.venv/bin/python scripts/teleop/recorder.py --task "Move the foam block into the bowl" --operator jose

# restart run_teleop ONLY when it is STOPPED, the motors off and the arms hanging still at zero (CLAUDE.md):
.venv/bin/python scripts/teleop/run_teleop.py --record-tap
```

In the headset:
1. Say "agent, camera still".
2. For each demonstration, say "agent, start episode", and finish with "agent, end episode success" (or "… fail").
3. "agent, discard episode" throws the last one away. "agent, episode status" checks the counts.

From a terminal:

```bash
.venv/bin/python tools/lfd/record_ctl.py start --mode teleop
.venv/bin/python tools/lfd/record_ctl.py end --outcome success
```

Aim for **at least 50 successful demonstrations**, all with the same camera preset and the same zero convention. That is LeRobot's guidance for SmolVLA; 25 "was not enough" there.

## 2. Check, sync and convert

```bash
# on the robot PC or the HPC: an ERROR means the converter would refuse that episode
python tools/lfd/validate_recording.py ~/bhl_recordings/<session> --out report.json
```

Sync the sessions to `/nfs/hpc/share/<onid>/bhl-data/incoming/` (roadmap §4.7; never to the home directory). Then convert on the HPC, as a CPU job on the `share` partition; the `sbatch` line is in `tools/lfd/train/README.md` §3:

```bash
python tools/lfd/convert_to_lerobot.py --sessions /nfs/hpc/share/<onid>/bhl-data/incoming/<session> \
    --out-root /nfs/hpc/share/<onid>/bhl-data/lerobot --repo-id local/block_bowl_v1 --label meas_future
```

- Use `--label meas_future` whenever teleoperated and hand-guided data will ever be compared. `cmd` is for teleop-only work.
- All three models get **the same dataset and the same `splits.json`**. Validation is on whole episodes.
- **The first real session settles the grid thresholds.** Read the validator's reused/skipped counts (`LFD_RECORDING_FORMAT.md` §4).

## 3. Train, on the HPC (needs your GPU-hour approval)

Everything is in `tools/lfd/train/README.md`:
- the one GPU environment to create;
- the Hugging Face access π0.5 needs (the gated PaliGemma tokenizer, approved by hand);
- the job order that fits the `ampere` cap of 2880 GPU-minutes reserved per user;
- resume.

`budgets.md` holds the proposed GPU-hours and the storage plan.

**Storage deadline.** The project share is over its 1.5 TiB soft quota. When the grace period ends (about 28 October 2026), every write fails, the robot's data sync included. The jobs refuse to start with under 7 days of grace. Freeing space is outside this pipeline.

## 4. Evaluate

- **Offline:** `tools/lfd/offline_eval.py`, on the validation episodes only. It gives L1 by horizon against a hold-still baseline, and inference time. It is a sanity check, not a result.
- **Live:** `docs/LFD_EVAL_PROTOCOL.md`.
  - `eval_schedule.py make` builds the counterbalanced schedule.
  - Before each trial, `eval_schedule.py next` prints the `record_ctl.py prime …` command. Then say "agent, start episode".
  - Log each trial with `eval_log.py`.
  - `analyze_eval.py` computes the preregistered statistics.
  - All of this waits for the POLICY state, the policy client and serving.

## Tests

Rerun after any change:

```bash
R=/nfs/hpc/share/sanchej7/envs/bhl-rig-py310/bin/python      # Python 3.10, like the robot's .venv
L=/nfs/hpc/share/sanchej7/envs/lerobot-py312-cpu/bin/python   # Python 3.12, LeRobot 0.6.1, CPU
$R tools/test_recorder.py;  $R tools/test_teleop_tuning.py
$R tools/lfd/test_labels.py;  $L tools/lfd/test_labels.py;  $L tools/lfd/test_convert.py
$R tools/lfd/test_eval.py;  $L tools/lfd/test_eval.py
$L tools/lfd/train/test_configs.py                 # add --dataset-root DS for the tiny CPU ACT run
.venv/bin/python tools/test_voice_typer.py         # on the robot PC: needs the repo .venv and the Vosk model
```

## Still unverified on real hardware

- **Tap cost on the N150.** It costs about 0.2 ms per row on a loaded Xeon; estimated at a few percent of the 20.8 ms cycle.
- **The camera's real frame rate, jitter and latency.** These settle the grid thresholds and `camera_latency_s` (roadmap step A4).
- **The robot PC's disk.** Write throughput and fsync time while recording.
- **Speech recognition.** Whether Vosk and Whisper recognise the new episode phrases.
- **Hand guiding.** The GUIDE state, the enabling switch and the hardware test of `docs/LFD_HAND_GUIDING.md` §4.
- **GPU fits and times.** Whether π0.5 LoRA and GR00T fit an A40 at the planned batch, and their real seconds per step. Nothing has run on a GPU.
- **The network.** The home → HPC hop and its latency, for live serving (roadmap §4.3–4.4).
