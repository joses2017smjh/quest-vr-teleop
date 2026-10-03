# Live evaluation protocol: foam block into a bowl (preregistration template v1)

*Written 3 October 2026. A template filled with defaults for the first task. Nothing on this page has been run on the robot.*

This page fixes, before the first trial, how three learned policies are compared live on the rig: what counts as success, how many trials, in what order, who scores them, when to stop, and which statistics decide. It follows the owner's preregistration practice: fixed n, interleaved policies, every failure kept, one-sided Fisher tests, blind scoring.

**How to use it.**
1. Fill every `FILL` in §14, and settle the template's marks after Gate 0 (§2.4).
2. Write `protocol.json` and `schedule.json` with the tools (§13). Commit them with this page, and push, before trial 1. That commit is the preregistration.
3. From then on, nothing changes silently. Every departure is logged as a deviation (§12), and the report lists them all.

Related pages: [`LFD_RECORDING_FORMAT.md`](LFD_RECORDING_FORMAT.md), the recording contract every trial follows (§8), and [`ROADMAP_2026-10-02.md`](ROADMAP_2026-10-02.md), cited below as "roadmap §n".

---

## 0. What live evaluation still needs

None of these trials can run yet. Live evaluation needs, in roughly this order:

1. **A `POLICY` state on the robot** (roadmap §4.5): `PolicyLink`, the grip dead-man, bumpless takeover, the shared clamp helper, the gentler POLICY torque and `max_lead` limits, HOLD when the queue runs dry and the slow retract after 2 s, and its four `--sim` tests. **Not built.**
2. **A policy client and a policy server** (roadmap §4.4): `policy_client.py` on the robot PC (time-indexed chunk queue, msgpack, the camera's own JPEG bytes) and `serve_policy.py` on the GPU. **Neither is built.** For this protocol, the client must also write each request's latency to a file (§4).
3. **Somewhere to serve from.** Live serving from the HPC needs COE IT's written approval (roadmap §4.2; sending that email is item 5 of roadmap §1.0). Otherwise use the fallback GPU (roadmap §4.8): a RunPod A40 at about $0.35/h, or a home GPU on the LAN. Either way, the go/no-go latency sweep of roadmap §4.4 must pass first. The LAN fix of roadmap §4.9 must be closed before any OSU key goes on the robot PC.
4. **The recorder in `policy` mode**, with the `trial`, `arrangement` and `policy` fields of the `start` command, and the one-shot `prime` command that lets "agent, start episode" open a policy episode (§7.2; `scripts/teleop/recorder.py` and `tools/lfd/record_ctl.py`, being built now).
5. **Gate 0 on this table** (roadmap §6.3): the measured reach that the template's marks come from (§2.4). The table-plane clamp and the contact stop (roadmap §6.3, §6.4 item 5) must also act in the POLICY path, because the claw works at table height.
6. **A zero check per session** (roadmap §3.1, `tools/lfd/zero_check.py`), so demonstrations and trials share one `zero_convention`.
7. **A motor-power cutoff within reach** (roadmap §1.4, §2.2).
8. **The data and the models**: demonstrations recorded and converted (§3.1), and three trained checkpoints.

Two manual steps stay manual until code replaces them: the operator ends each trial at 30 s by a timer (§7.3), and checks the camera against a reference photo once per session (§2.5). In the tap, `src: "policy"` and the `intervention` flag are reserved: `intervention` stays false until a POLICY state exists. Until then, interventions come only from the operator's log, and the hand-over (t = 0, §2.2) only from the operator's stopwatch.

## 1. Question and hypotheses

**Question.** On one single-arm pick-and-place task, trained on the same demonstrations, which of ACT, π0.5 and GR00T N1.7 succeeds most often?

**Preregistered hypotheses** (the `comparisons` list in `protocol.json`; each is one-sided, "succeeds more often than"):

| # | hypothesis | why this direction |
|---|---|---|
| H1 | π0.5 > ACT | Pretrained VLAs "generally outperform" ACT on SO-101, though "highly task-dependent" (roadmap §5.1). ACT trains from scratch, as the control (roadmap §5.4). |
| H2 | GR00T N1.7 > ACT | the same |
| H3 | π0.5 > GR00T N1.7 | no prior between the two VLAs, so both directions are tested |
| H4 | GR00T N1.7 > π0.5 | the same |

So Holm's family has m = 4. The two directions left out (ACT > π0.5, ACT > GR00T N1.7) are still shown, as exploratory and unadjusted, because weak transfer is expected (roadmap §5.4). They are not claims. If you have no directional prior at all, list all six ordered pairs before `make`: m becomes 6, and each test needs a larger gap.

## 2. Task, success and scene

### 2.1 The task

"Move a foam block into a bowl within 30 seconds without human assistance." One arm: the **robot-right arm** (`"arm": "right"`; change it before `make` if you prefer the left). The left arm stays at rest: its grip is not held, so it holds (roadmap §4.5 item 2).

**Start state of every trial, and of every demonstration:**
- the arm at the start pose (FILL: hanging beside the table edge, or a preset pose above the table), claw open;
- the block and the bowl on the arrangement's marks (§2.6);
- nobody in the camera's view or inside the safe box (§7.4).

### 2.2 Success: the primary endpoint

A trial is a **success** only if all four hold:
1. **In the bowl.** At the end of the trial the block lies inside the bowl: seen from above, entirely within the rim. The bowl may have slid, but it stands upright.
2. **Released.** The claw is not holding the block.
3. **In time.** The release that leaves the block in the bowl comes at most 30.0 s after t = 0, and the trial did not reach the 30 s limit (§7.3).
4. **Unassisted.** No intervention (§7.4): no takeover or grip release by the operator, and no person touching the arm, block or bowl.

Everything else is a **failure**: a timeout, a block dropped or pushed outside, a block knocked off the table, a tipped bowl, an intervention. Failures are never dropped from any table.

**t = 0 is the hand-over:** the first tap row with `src: "policy"` in the trial's recording, i.e. the first control cycle in which the policy commands the arm. Until the tap carries that row (a POLICY state, §0), t = 0 is the hand-over as the operator reads it from the recording's video. During the trial, the operator keeps the 30 s limit with a stopwatch started at the hand-over (§7.2). So the 30 s include the policy's first inference. They do not include the time between starting the recording and the hand-over, nor the operator's reaction at the end; both depend on the operator, not on the policy. The 30 s rule and the completion time (§4) run on this one clock.

### 2.3 Objects

**Block.** A 40 mm foam cube (EVA or polyurethane), at most 10 g, in one saturated colour that contrasts with both the table and the bowl.
- Measure the open claw's gap first. It must exceed the block by at least 15 mm. If not, use a 30 mm cube.
- Light objects only: plan on ≤ 0.1–0.15 kg per claw until measured (roadmap §5.2). Foam is far below that.

**Bowl.** Round, rigid, matte, in a light colour.
- Inner diameter at the rim: **150 mm** if Gate 0 leaves room, else **120 mm** (never below 120 mm). It is the largest that keeps every block outline at least 30 mm from the rim (§2.4).
- **Rim 30 mm above the table (25–35 mm).** The arm cannot lift high: only 41–50 % of the cloth zone is reachable 10 cm above the table, so the roadmap advises carrying low, at ≤ 5 cm (roadmap §6.3). The claw holds the 40 mm block around its middle. Releasing it 10 mm above a 30 mm rim therefore puts the claw at about 30 + 20 + 10 = 60 mm above the table. A deeper bowl pushes that toward the height the arm reaches least.
- A flat bottom, so a dropped block settles. A non-slip base (a silicone ring or putty), so a bump does not slide it far.

**Table and lighting.** One matte surface. The template is taped down at its corners (§2.6). Use the same room lights for every session, with no direct sunlight. Record the lighting in the session's first deviation note if it differs.

### 2.4 Reach: Gate 0 decides where the marks go

The template's coordinates are **provisional until Gate 0** (roadmap §6.3) has been measured on this table, at this mounting height. The mounting height is still an open question (roadmap §1.3 item 2).

**Gate 0 for this task.** Hover the claw tip, by teleop, 5 mm above each candidate mark of a 2 cm grid. Do it at grasp height (about 20 mm, the block's middle) and at release height (about 60 mm, §2.3). Measure the tip with the calibrated camera after Step 0 (roadmap §3.2). Until Step 0 exists, use a ruler, and say which in §14.

A mark is usable when:
1. **Block marks:** the hover error at grasp height is ≤ 15 mm (FILL from Gate 0's p90), and the mark lies at least one grid step (20 mm) inside the measured boundary.
2. **Bowl marks:** some point of the bowl's **release disc** passes the same test at release height. The release disc is the central disc of radius (rim radius − 30 mm): a 40 mm block (half-diagonal 28 mm) released over it lands inside the rim. That is 45 mm for a 150 mm bowl and 30 mm for a 120 mm one. So the bowl itself may stick out past the reach.
3. **Clearance:** every block outline is at least 30 mm from the rim of every bowl position it is paired with, so the open claw can come down beside the block.

**What to expect (estimate, not measured).** With the table 0.28–0.30 m below the shoulder pitch axis, URDF FK puts the cloth zone 0.08–0.28 m ahead of the shoulder axis. Each claw owns its half, reaching at most about 2 cm past the midline (roadmap §6.3). For one arm, that is roughly a 15 × 20 cm patch, and a 150 mm bowl nearly fills it. So expect the bowl marks at the far edge of the reach, with only their release discs inside it, and the block marks in a strip nearest the robot. If Gate 0 shows less room, use the 120 mm bowl, and if needed fewer block marks (at least 3). Change `arrangements` before `make`: a change made before the schedule exists is not a deviation.

### 2.5 Camera: the `still` preset

`CameraAim` has a `still` mode, but it holds wherever the camera happens to point: no stored preset exists in the code yet (`scripts/teleop/servos.py`). So the preset is set by hand, once, and then checked:
1. Before the first session, aim the camera until both eyes see the whole template and the arm's workspace. Then say "agent, camera still".
2. Record a short test episode, and read the pan/tilt from its `episode.json` (`cam_at_start`, the recorder's copy of the tap's `cam` field). Write it into `protocol.json` as `camera.preset_deg` before `make`.
3. The recorder refuses `start` unless `cam_mode` is `still` (recording format §3).
4. The analysis flags every trial whose `cam_at_start` differs from the preset by more than `camera.tolerance_deg` (1°). These are commanded angles: the SG90 servos give no feedback.
5. An SG90 does not return to the same angle reliably (roadmap §4.5 item 7), and the automatic reference-frame check is not built. Until it is, at the start of each session compare the first frame with a reference photo of the template: its four corner crosses should sit within about 10 px. If the camera moved, re-aim it, and log a deviation.

### 2.6 The placement template

One printed sheet (A3, or two A4 sheets joined), taped to the table at marked corners, holding:
- **block outlines B1–B5:** 40 mm squares, each printed at its own yaw: B1, B3 and B5 at 0°, B2 and B4 at 45°;
- **bowl circles W1 and W2:** dashed circles of the bowl's rim diameter, each with its release disc (§2.4) and centre cross. The two circles may overlap, because only one bowl is placed at a time;
- the **safe box** outline (§7.4), an arrow for robot-forward, the midline, and **four corner crosses** for the camera check (§2.5).

**Ten numbered start arrangements** pair a block mark with a bowl mark. These are the `arrangements` in `protocol.json`:

| arrangement | A01 | A02 | A03 | A04 | A05 | A06 | A07 | A08 | A09 | A10 |
|---|---|---|---|---|---|---|---|---|---|---|
| block mark | B1 | B2 | B3 | B4 | B5 | B1 | B2 | B3 | B4 | B5 |
| bowl mark | W1 | W1 | W1 | W1 | W1 | W2 | W2 | W2 | W2 | W2 |

**Layout sketch** (top view, robot at the bottom; the coordinates come from Gate 0):

```
            far edge of the measured reach
   +---------------------------------------------+  <- safe box outline
   |      .-- W1 --.               .-- W2 --.    |     bowl circles (dashed)
   |     (  (  +  ) )             (  (  +  ) )   |     inner circle: release disc
   |      '--------'               '--------'    |
   |                                             |
   |    [B5]          [B3]            <B4>       |     block outlines, 40 mm
   |          <B2>             [B1]              |     [] at 0 degrees, <> at 45
   +---------------------------------------------+
   ^ midline + 2 cm                   outer edge ^
   ================ table edge ====================
                  robot, right shoulder
```

**Mark coordinates** (in cm; origin on the table directly below the right shoulder pitch axis; x forward, y toward the robot's left): FILL after Gate 0.

| mark | x | y | yaw | Gate-0 hover error |
|---|---|---|---|---|
| B1–B5 | FILL | FILL | 0° or 45° | FILL |
| W1, W2 (centre) | FILL | FILL | — | FILL (best point of the release disc) |

Keep a photo of the template in place in the evidence folder (§13).

## 3. Conditions

### 3.1 The three policies

| id | policy | trained as | executed as |
|---|---|---|---|
| `act` | ACT | from scratch: the control (roadmap §5.4) | plain time-indexed chunks; LeRobot gives ACT no RTC (roadmap §4.4) |
| `pi05` | π0.5 | LoRA or full fine-tune from the openpi weights (FILL) | with RTC, which LeRobot supports for π0.5 (roadmap §4.4) |
| `groot_n17` | GR00T N1.7 | fine-tune (FILL) | time-indexed chunks, with RTC only if its server supports it (FILL) |

Each policy runs the way it would be deployed, so each comparison is between a policy and its execution. The choice is fixed before trial 1.

**One dataset, identical for all three:**
- the same converted dataset (`tools/lfd/convert_to_lerobot.py`), with the same `conversion.json` and the same `splits.json` (recording format §4);
- the action label `meas_future`, with the lookahead H = 0.10 s. It is the only label that also suits hand-guided data, so the optional study B (§3.2) uses it too. `cmd` is allowed instead if chosen before training. Either way, record it in `dataset.label`;
- the dataset's path and the SHA-256 of `conversion.json` and `splits.json`, in `protocol.json` (`dataset`).

**The demonstrations.** At least 50 teleop demonstrations (FILL), about five on each of the ten arrangements. They are recorded on the same template, at the same camera preset, under the same `zero_convention` and claw limits; the converter refuses to mix those anyway. The evaluation runs on the same ten arrangements, so it measures in-distribution success. Held-out arrangements are a later study.

**Evaluation recordings never become training data.** Trials are recorded in `policy` mode, and the converter refuses `policy`-mode episodes unless asked for (recording format §4).

**Checkpoints are chosen by rule, before trial 1.** Use the last checkpoint of a training run with a preregistered number of steps (FILL per policy, `train_steps`). Never choose by live trials. Offline validation loss may be reported, but it does not choose. Record each checkpoint's path and SHA-256 in `policies[*]`.

**One training run per policy.** The GPU budget allows one seed each (roadmap §5.3), so the result is about these three runs, not about the architectures in general. The report says so.

**The same robot-side limits for all** (`execution` in `protocol.json`; roadmap §4.5 items 4 and 6):
- `--max-speed 0.5` rad/s;
- a POLICY `max_lead` of 0.15 rad;
- the POLICY headroom scale of 0.6;
- 30 Hz actions, executed by time.

**The same serving path for all.** All three are served by the same `serve_policy.py`, on the same GPU type, ideally in one serving job that holds all three. Switching policy between trials then costs nothing, and latency differences come from the models. At the inference memory figures in roadmap §5.1 (π0.5 > 8 GB, GR00T N1.7 16 GB+), all three should fit on one 48 GB A40 (estimate).

### 3.2 Optional study B: teleop against hand-guided data, model fixed

Run it only if [`LFD_HAND_GUIDING.md`](LFD_HAND_GUIDING.md) finds hand guiding feasible on this rig. Its verdict (§6 there, 3 October 2026) is "supported after the six steps below; not supported now". The six steps are: the shoulder-roll zero tested and corrected; gravity refit; a motor-power cutoff, with the LAN hole closed; the operator's view and dead-man; a GUIDE state with its `--sim` tests; and its hardware test. So study B waits for those six steps. Record the decision in §14. It is a separate study, with its own `protocol.json` and schedule:
- `study_id` `block_bowl_data_v1` and `trial_prefix` `G`, so its trial ids never collide with this study's;
- **the model is held fixed:** ACT, with the same hyperparameters and steps as `act` in §3.1;
- **two datasets of equal size,** with the same number of episodes on each arrangement: one recorded by teleop, one by hand guiding. **Both are converted with `meas_future`** (H = 0.10 s). For guided episodes it is the only valid label (recording format §4);
- policies `act_teleop` and `act_guided`, with both directions as comparisons (no prior): m = 2;
- **the operator in view** (hand-guiding §5.1): guided episodes show the operator's hand and arm, and autonomous runs show nobody. The **primary** variant masks the human in the guided episodes and applies the same augmentation to the teleop episodes (random masks of a similar size near the claw), so that a grey blob alone cannot tell the two datasets apart. The **unmasked** variant is secondary, reported beside it. Both are trained before trial 1, and the choice is recorded in §14 then; it is never made after seeing results;
- also reported: operator minutes per accepted demonstration in each mode, from the episode lengths and the discards.

## 4. Endpoints

**Primary.** Success as defined in §2.2: successes out of trials run, per policy.

**Secondary:**
1. **Completion time of successes,** in seconds, defined so that no one's reaction time enters it:
   - **from** t = 0, the hand-over (§2.2): the tap's first `src: "policy"` row once POLICY exists; until then, an operator-entered time, read from the recording's video;
   - **to** the success event: the release that left the block in the bowl, the first frame in which the claw has let go of it, judged from the video, or the time the operator logs.
   - **Sources, in order:** the blind scorer's time, read from the video (the scorer's pack names every frame by its time from t = 0, §9); otherwise the operator's `--time-s`, read from the video, or else the operator's stopwatch, started at the hand-over and read at the release (§7.2).
   - It is **never** taken from the episode's length (`end.t − start.t`): the recording also holds the time before the hand-over and the operator's ending of it. A success therefore needs `--time-s`, and only that time is checked against the 30 s rule (§7.3). A long recording never turns a genuine success into a failure.
   - It is not blind unless the scorer times it.
2. **Interventions per trial:** the count of operator takeovers and grip releases, plus any human touch. The operator logs it (`--interventions`), because the tap's `intervention` flag stays false until POLICY exists (§0).
3. **Inference latency, p50 and p95 per policy,** in ms per request (roadmap §4.4):
   - **latency_ms = 1000 · (t_arrival − t_obs).** `t_obs` is the `t_cap` of the camera frame sent in the request: robot `CLOCK_MONOTONIC`, the moment ffmpeg finished writing the frame (recording format §1). The camera's own exposure-to-write delay (`camera_latency_s`) is left out: it is the same constant for every policy, so if it is measured it is reported beside the result, never added to it. `t_arrival` is `time.monotonic()` on the robot PC when `policy_client.py` has parsed the reply, so the chunk is ready to queue.
   - **Which requests:** every request of a trial that ran, from the hand-over to the end of the trial. The warm-up requests of §7.1 and the requests of void attempts are excluded.
   - **Pooling:** each request counts once, so a trial that runs longer (a 30 s failure) contributes more requests than a quick success. The median over trials of each trial's own p50 is reported beside it.
   - The client (not built) must write one file per trial in one of three forms: a JSON list, JSON lines each with `latency_ms`, or one number per line. Pass it with `--latency-file`. Per-trial summaries (`--latency-p50`, `--latency-p95`) are a fallback, reported separately.
4. **Wall time per trial,** all trials, failures included: the episode's wall-clock length from `episode.json` (`end.wall − start.wall`). An operator's stopwatch (`--wall-s`) overrides it.

Reported but not endpoints: void attempts, out-of-order trials and every other deviation (§12).

## 5. Sample size and power

**n = 20 trials per policy** (10 arrangements × 2 repeats): 60 trials in all, fixed now. There is no early stopping for results (§10, S5).

**What n = 20 can find.** These are the exact power figures of the one-sided Fisher test, from `analyze_eval.py power --n 20 --m 4`. The first Holm step tests at α/4 = 0.0125, the worst case; "alone" means α = 0.05.

| true success rate, better | true success rate, worse | power, first Holm step | power, alone |
|---|---|---|---|
| 0.50 | 0.10 | 0.63 | 0.84 |
| 0.60 | 0.20 | 0.57 | 0.75 |
| 0.70 | 0.30 | 0.54 | 0.71 |
| 0.80 | 0.40 | 0.57 | 0.75 |
| 0.90 | 0.50 | 0.63 | 0.84 |
| 0.80 | 0.20 | 0.94 | 0.98 |
| 0.70 | 0.10 | 0.96 | 0.99 |
| 0.90 | 0.30 | 0.96 | 0.99 |

In words:
- A 40-point gap is found 54–63 % of the time at the first Holm step, and 71–84 % alone. A 60-point gap is found at least 94 % of the time.
- Against a policy at 4/20, the better one needs 12/20 at the first Holm step, or 10/20 at α.
- Each policy's own rate is known only to about ±20 points: 10/20 gives a Wilson interval of 0.30–0.70.
- So this study can find large differences only. A difference that is not significant is not evidence that two policies are equal.
- For more power, use 30 per policy (10 × 3 repeats): a 40-point gap is then found 76–85 % of the time at the first Holm step, and 90–95 % alone; a 60-point gap more than 99 % of the time (`analyze_eval.py power --n 30 --m 4`).

**Order balance.** 20 blocks of 3 cannot balance positions exactly: the counts differ by at most 1 (§6). With 12 × 2 (24 blocks) or 9 × 2 (18 blocks), the balance is exact. The default stays at 10 × 2.

## 6. Schedule and order

`tools/lfd/eval_schedule.py make` builds the order from `protocol.json` and its `seed`. It is a fixed file, committed before trial 1.

1. **A block** is one arrangement, set up once, on which every policy runs once.
2. **Within a block,** the policy order is one row of a Latin square. Successive blocks take successive rows, so every 3 blocks form a whole square: each policy runs once first, once second and once third.
3. **Williams design.** For 3 policies, there are two mirror-image squares. Over 6 blocks, each policy also follows each other policy equally often (carry-over balance).
4. **Each repeat** visits all ten arrangements once, in a fresh random order. As far as possible, no arrangement sees the same policy order twice.
5. With the defaults, every policy runs exactly twice on every arrangement. Positions balance within 1, and carry-over within 2. `make` prints these spreads, and `schedule.json` keeps the counts under `balance`.

**Trial ids** (`T001` … `T060`, in run order) do not name the policy themselves, but `schedule.json` is public from before trial 1, so anyone can decode them. A blind scorer therefore never sees them: the pack gives each trial a random code instead (§9).

**Sessions.** Run one repeat per session: blocks 1–10, then 11–20. Never split a block across sessions. Keep a session to at most 60 minutes of trials, with at least 10 minutes' rest between sessions (heat; roadmap §5.2).

## 7. Running the trials

### 7.1 Once per session

1. Run the zero check (roadmap §3.1; `tools/lfd/zero_check.py`). Every episode then carries it as `zero_check`.
2. Check the camera: the `still` preset and the reference photo (§2.5).
3. Bring up the serving job and the client. The go/no-go sweep of roadmap §4.4 must have passed for this serving path. Then send one warm-up request per policy, with no motion.
4. The motor-power cutoff is within reach. The left arm hangs at rest.
5. Start the recorder, with `--operator NAME` so that voice starts carry the operator's name. Run `python $T/eval_schedule.py next` in the study folder (§13) to see the first trial.

### 7.2 One trial

1. **Set up.** `next` says whether the trial starts a new block. If it does, place the block on its outline at the printed yaw, and centre the bowl on its circle. If not, reset to the same marks.
2. **Start pose.** The arm is at the start pose, with the claw open. Nobody is in the camera's view or inside the safe box.
3. **Before photo** (optional): from a phone on a fixed stand, the same spot for the whole study. Otherwise the recording's first frame serves.
4. **Start** the recording, in two steps:
   - **Prime** the recorder with the command `next` prints, from a terminal: `python $T/record_ctl.py prime --mode policy --trial 17 --arrangement A07 --policy pi05 --task "..."`. A prime holds the fields for the next start only. It is used up by that start, and it expires after 10 minutes unused, so it can never label a later teleop demonstration. `record_ctl.py status` shows it.
   - **Then say "agent, start episode"** in the headset. The recorder opens the episode with the primed fields. Its reply, shown in the headset, must name `policy` mode; if it says `teleop`, the prime was missing or had expired.
   - **Never start a trial by voice without a fresh prime.** A voice start without one records a `teleop` episode, which `eval_log.py` refuses for a trial, and which would otherwise sit among the demonstrations. If it happens, discard it at once ("agent, discard episode"), log a `deviation`, prime again and start again. From the terminal, `record_ctl.py start` with the same fields does the same as prime plus voice.
5. **Hand over.** Enter POLICY and hold the right grip as the dead-man (§7.4). **Start the stopwatch at the hand-over: that is t = 0** (§2.2).
6. **Watch.** Intervene by the rule (§7.4). Give the policy no other help or signal.
7. **End** at the first of:
   - the block released in the bowl, with the claw lifted clear. **Read the stopwatch at the release:** the completion time, unless a reading from the video replaces it (§4);
   - the 30 s mark (§7.3);
   - an intervention.

   End POLICY (the arm holds), then end the recording: "agent, end episode success" or "agent, end episode fail", or `python $T/record_ctl.py end --outcome success|failure`. The time this takes does not count: the recording's length is never the completion time.
8. **After photo** (optional): from the same spot.
9. **Log it** at once (§13). The facts first, because no scorer can overturn them (§9):
   - a success: `python $T/eval_log.py trial --trial T017 --success yes --time-s 21.4 --episode <root>/<session_id>/ep_0012 --latency-file <client's file>`;
   - the 30 s mark reached: `... --success no --timed-out yes ...`;
   - a failure before the limit (the block dropped outside, the bowl tipped): `... --success no --timed-out no --notes "..."`;
   - an intervention: `... --success no --interventions 1 --notes "..."`, plus `--collision` when the arm headed for a collision (§10, S2).

   The tool refuses a failure without an intervention that does not say `--timed-out yes` or `no`. Add `--before`/`--after` with the photos, so that `pack` can pass them to the scorer (§9). The recorder's reply to the start (in the headset, or `record_ctl.py status`) names the episode folder. A success whose recording ended by itself (`aborted`) is refused unless `--accept-aborted REASON` says why it stands; it is then also logged as a deviation.
10. **Reset** (§7.5).

### 7.3 Timeouts

- **30 s per trial,** from t = 0, the hand-over (§2.2). No automatic stop exists yet. The operator ends the trial at 30 s by a visible timer, such as a phone stopwatch started at the hand-over, and logs it with `--timed-out yes`: a fact that decides the trial before anyone's call (§9). A success logged with a completion time over 30 s is refused, because by rule it is a failure. Only the completion time is checked against the limit, never the recording's length.
- **No action chunk within 5 s** of the hand-over: `policy_client.py` has received no chunk at all (its latency file for the trial is empty). The serving link failed, so the policy never got control: end the episode and log the attempt as void (§7.6). If chunks arrived, the policy had control, even if the arm did not move, or moved and then froze: the trial is not void. Let it run to 30 s and score it as usual, a failure unless it succeeds. A policy that holds still is a failing policy, not a serving fault.
- **A reset over 3 minutes** is noted (`--notes`). It is not an outcome.

### 7.4 The intervention rule

**The operator takes over** (moves the controller past the takeover threshold, releases the grip, or triple-clicks X) **when the arm heads for a collision or leaves the safe box.**
- A collision means with the torso, the head or camera, the other arm, the table edge, a person, or pressing into the table.
- **The safe box** is the claw tip within the template's printed box: from 2 cm past the midline to the box's outer edge, from the table edge to 5 cm beyond the farthest mark, and at most 15 cm above the table (FILL the final box after Gate 0).

**Then the trial ends, and it counts as a failure.** Log the intervention (`--interventions 1`, or more) with a note saying why. Add `--collision` when the arm headed for a collision: S2 counts these (§10).

A person touching the block, the bowl or the arm during a trial is also an intervention, so never "help" (nudge the block, steady the bowl). When in doubt, intervene: a broken printed gearbox costs more than a trial.

The operator wears the headset as for teleop, so the controllers stay tracked and a takeover works. The takeover is not bumpless in today's code: roadmap §4.5 item 2 gives the fix that POLICY must include.

### 7.5 Reset

After every trial:
1. Bring the arm back to the start pose, by teleop or by the slow retract once it exists. Open the claw.
2. Put the block and the bowl back on the next trial's marks. If either was knocked off the template, put it back exactly on its mark.
3. Check that the template has not moved: its corners against the tape.

No policy runs during a reset.

### 7.6 Void attempts and re-runs

An attempt is **void** only if:
- the policy never got control: a set-up error caught before the hand-over, or the serving link down so that no action chunk ever reached the client (§7.3); or
- a fault that the policy's commands clearly did not cause stopped it: a CAN bus loss, a `run_teleop.py` crash, a power cut.

When in doubt, it is not void, and the trial counts. A policy that received chunks but did not move the arm had control: its trial counts (§7.3).

Log a void with `eval_log.py trial --trial T017 --void "REASON"`. The trial stays next: reset and run it again. Voids are reported per policy. If a voided trial is never re-run, it is missing, and the "missing as failure" sensitivity analysis counts it as a failure (§11). A third void of one trial, or three in a session, stops the session (§10, S3). The tool enforces it: it prints the stop at the third void, and refuses every further trial until a `deviation` records the cause and its fix.

A recorder failure in the middle of a trial does not void it. The outcome is judged as usual, and the missing recording is flagged.

## 8. Recorder link

Each trial is **one recorded episode in `policy` mode** (recording format §3: the one-shot `prime` command, then a voice or terminal `start`). It carries:
- `mode: "policy"`;
- `trial`: the trial's **number**, an integer, for example `17` for `T017`. `tools/lfd/record_ctl.py` takes `--trial` as an integer, and the checks accept either the number or the id;
- `arrangement` (for example `"A07"`) and `policy` (for example `"pi05"`): this protocol's ids, as strings;
- `task`: the protocol's task text; and `operator`, the recorder's default (`--operator`, §7.1) unless a terminal `start` gives one.

`eval_schedule.py next` prints these fields and the exact `record_ctl.py prime` command for them (§7.2).

`eval_log.py trial --episode DIR` reads that episode's `episode.json`. It refuses to log the trial unless the mode is `policy` and the trial, arrangement and policy all match the schedule, and the episode is not discarded. It refuses a success on a recording that ended by itself unless `--accept-aborted REASON` is given (§7.2). It reads the hand-over, the first `src: "policy"` row, from `rows.jsonl`, and keeps it with the record. It warns, without refusing, when the recording has no zero check, another task text, or a changed zero.

The analysis checks every episode again. It flags a recording whose own `outcome` disagrees with the scored result, one that ended by itself (`aborted`) or was never closed, and a completion time later than the recording's end. Per recorder session, it also flags recordings without a zero check (§7.1, one per session), an inconclusive zero check, a `zero_convention` other than the dataset's (`dataset.zero_convention` in `protocol.json`) or more than one convention across the trials, a zero that changed during a recording, and a task text other than the protocol's.

Episodes live on the robot PC under `~/bhl_recordings/<session_id>/ep_NNNN`. After rsync to the HPC, pass `--episode-root` (the folder holding the session folders) to `pack` and `report`.

## 9. Blinding and scoring

**What can be blind:** the call on the end state, whether the block lies in the bowl, released. Nothing else: the time limit and the interventions are facts the operator logs (§7.2), and a scorer looking at pictures cannot see an intervention.

**Who can score, and who cannot:**
- **Can:** a second person who has not seen the schedule (`schedule.json`, or `next` and `show` output), the trial log, any `episode.json`, or the pack's key, and who watched no trial live. They receive the pack folder and nothing else.
- **Cannot:** the owner, who wrote and pushed the schedule; the operator, who ran the trials; anyone who watched a trial, or opened any of those files. `schedule.json` is public in git from before trial 1, so anyone who has read it could decode trial ids: that is why the pack hides them (below).
- **If no such person is available,** there is no blind scoring. Every call is then the operator's, the report says so (`no_blind_scoring`), and the conclusions say the outcome was unblinded.
- Disclose every exposure in a deviation note: a scorer who watched a trial live, or saw the schedule, the log or the key. That scorer's calls still count, flagged by the note.

**How:**
1. `eval_log.py pack --out scorer_pack` makes the scorer's folder:
   - each logged trial under a **random code** (for example `S3F9A1C`), unrelated to its trial id or its run order, in a random order on `sheet.jsonl`. No seed can reproduce it;
   - the **key**, the codes and their trials, is written outside the folder, beside the trial log (`pack_key_scorer_pack.json`). It never goes to the scorer;
   - per trial: the before and after photos, if taken; the recording's first frame; and its **end frame**, the last frame at or before t = 0 + 30 s (§2.2), or the recording's last frame if the tap does not mark the hand-over;
   - with `--video`, also every frame in a folder, each file named by its index and its time from t = 0 (`00412_+0012.34s.jpg`), so the scorer can time a release;
   - photos and frames lose their metadata (Exif and XMP, IPTC, comments, and anything a phone appends after the image), so no capture date gives away the run order. A photo that is not a JPEG is left out, with a warning. Every packed file gets the same file time. The pack never copies `episode.json` or `rows.jsonl`.
2. The scorer records each call with `eval_log.py score --sheet scorer_pack/sheet.jsonl --scorer NAME --code S3F9A1C --success yes|no`. The calls go to `scorer_pack/scores.jsonl`, a separate file keyed by code. `--time-s` is optional, for a success: the time in the release frame's file name.
3. `analyze_eval.py report --scores scorer_pack/scores.jsonl --pack-key pack_key_scorer_pack.json` maps the codes back and merges the calls.

**Which call counts.** Each trial is decided in this order, fixed now:
1. **The facts first.** Any intervention (`--interventions`), or the 30 s limit reached (`--timed-out yes`), makes the trial a failure, whatever anyone called it. These are facts logged by the operator (and, once POLICY exists, by the tap's `intervention` flag), not judgements of the end state.
2. **Then the end state,** among trials that ended within the limit with no intervention: the blind call, when one exists. With several scorer files, the first scorer (in the order given) who called a trial decides it. Otherwise the operator's call, flagged `unblinded`.
3. **Then the time of a success:** the scorer's video time if given, else the operator's `--time-s`. A time over 30 s makes it a failure. A success with no time at all, from a trial that the log does not show ending before the limit (`--timed-out no`, or a recording that ended within 30 s of the hand-over), is counted as a failure: nothing shows it was in time.
4. **Every override and disagreement is reported:** `rule_override` when a fact or a time overturns a call of success, `success_time_unknown` for step 3's last case, `scorer_disagreement` between the operator and the scorer, `inter_scorer_disagreement` between scorers. The operator's own calls, under the same rules, are also analysed as a sensitivity check (§11).

**Limits, stated plainly:**
- Policies may move differently (ACT's chunks, RTC's smoothness), so video can unblind. Make the success call from the photos and the end frame; use the video only to time a release.
- A scene can still betray its order: the light through a window, the bowl's wear. The codes only remove the bookkeeping.
- Completion time stays unblinded unless the scorer times it.

## 10. Stopping rules

**S1, safety: stop the session at once** (triple X, Ctrl+C, or the motor-power cutoff) when:
- the arm hits the torso, the head or camera, or a person;
- a joint faults, trips `JointHealth` or drops off CAN;
- a gearbox smokes, smells or grinds;
- the camera has moved off its preset.

Resume only once the cause is found and fixed, and log a deviation.

**S2, withdrawing a policy (harm).** A policy is withdrawn when it needs a collision intervention in 3 of its own consecutive trials, or causes any contact listed under S1. Only that evidence permits it, and the log keeps it:
- three collision interventions: each of the policy's last 3 logged trials has `--interventions` ≥ 1 and `--collision` (§7.4). Then `eval_log.py withdraw --policy ID --reason "..."`;
- an S1 contact: log that trial first (a failure, with the intervention and a note on the contact), then `eval_log.py withdraw --policy ID --s1-contact T017 --reason "..."`.

`withdraw` refuses without one of the two, so a policy that merely failed to be served, or has not run yet, cannot be withdrawn. The withdrawal record names its evidence. The remaining trials count as failures in every analysis, as imputed failures, and the other policies continue in the schedule's order.

**S3, voids.** A third void of one trial, or three voids in one session, stops the session until the cause is fixed. The tools take a session to be one repeat (§6). `eval_log.py` prints the stop at that void and refuses every further trial; `next` and `show` print it too. A `deviation` that records the cause and the fix ends the stop, and the counts start again from it.

**S4, heat.** Rest at least 10 minutes when `tools/teleop_tune.py audit` shows a joint at its torque limit for more than 20 % of the last 5 minutes. This is a duty-cycle rule (roadmap §5.2), because the rig reads no temperatures. It assumes POLICY cycles reach the log that `audit` reads. FILL: confirm the 20 %.

**S5, no stopping for results.** There is no interim analysis. Never stop early because a policy looks good or bad. Run `analyze_eval.py report` once, after the last trial. `next` and `show` are fine at any time. An interim look must be logged as a deviation, and the report marks itself `INCOMPLETE` while trials are missing.

**S6, the end.** The study ends when every trial is logged or withdrawn. It also ends at an S1 stop that cannot be fixed within 14 days (FILL). Then analyse what exists: the report is flagged incomplete, and the missing trials are counted as failures in the sensitivity analysis.

## 11. Analysis plan

`tools/lfd/analyze_eval.py report` computes all of it, taking every setting from the `protocol.json` copy inside `schedule.json`. No command-line flag changes a statistic.

1. **Primary, per policy:** successes / n over the trials run (withdrawn trials count as imputed failures), with a **Wilson 95 % score interval**, without continuity correction. Each trial is decided in §9's order: the logged facts, then the blind call on the end state, then the time. The log is read strictly: a `success` that is not `true` or `false` is read as not a success, and flagged.
2. **The preregistered comparisons:** a **one-sided Fisher exact test**, conditional on both margins, for each hypothesis in §1. Then the **Holm** step-down correction over those 4, at a family-wise one-sided α = 0.05. A hypothesis is "supported" when its Holm-adjusted p ≤ 0.05. Only supported hypotheses are claims. The two other directions are shown as exploratory and unadjusted.
3. **Sensitivity analyses:**
   - every scheduled trial, with missing ones counted as failures;
   - the operator's own calls instead of the blind scorer's, under the same rules.
4. **Secondary:**
   - completion time of successes: the median, with a **95 % percentile bootstrap** interval (10,000 resamples, numpy `RandomState(seed + policy index)`, seed 20261003), computed only with at least 5 timed successes;
   - interventions per trial: the mean, and how many trials had any;
   - latency p50 and p95, pooled over requests (numpy's linear percentiles; §4 defines the latency and the pooling), and the median of the per-trial p50s;
   - wall time per trial: median and mean.
5. **Failures are kept in every table.** The report ends with all 60 trials, one row each, saying what decided each: the scorer, the operator, or the rules.
6. **Deviations are flagged automatically:** missing, void, out-of-order and withdrawn trials; a withdrawal without its S2 evidence, or before the policy ran; amendments; duplicate or unknown ids; malformed records and scorer calls; operator–scorer and scorer–scorer disagreements; unblinded calls; rule overrides, and successes with no evidence of being in time; trials logged without an episode, and episodes not found, not matching, aborted, never closed, or whose outcome disagrees; a completion time later than the recording's end; the camera off its preset or not `still`; zero checks, zero conventions and changed zeros per session; and the task text.

A log written under another schedule is not analysed at all: `report` refuses it (§12).
7. **Interpretation.** Report every interval, not just the p-values. A difference that is not significant is not equivalence (§5). There was one training run per policy (§3.1).

The outputs are `report.json` and `report.md`. Both record the SHA-256 of the schedule, the protocol, the trial log and each scorer file they were made from.

## 12. Deviations log

Every departure from this page is logged in the trial log itself, with a tool and never by hand:

| what happened | how it is logged |
|---|---|
| a trial run out of the schedule's order | `eval_log.py trial ... --out-of-order "REASON"` |
| an attempt that does not count | `eval_log.py trial ... --void "REASON"` |
| a success on a recording that ended by itself | `eval_log.py trial ... --accept-aborted "REASON"` (a deviation line is added) |
| a logged result that was wrong | `eval_log.py amend --trial T017 --reason "..." --success no --timed-out no` (the original line stays) |
| a policy withdrawn for safety | `eval_log.py withdraw --policy ID --reason "..."` (with its S2 evidence, §10) |
| the cause and fix after an S3 stop; a voice start without a prime; a camera re-aim, a lighting change, an interim look, a scorer's exposure; anything else | `eval_log.py deviation --note "..." [--trial T017]` |

The log is append-only JSON lines. Every line carries the schedule's SHA-256. A log written under another schedule is refused by every tool: `eval_log.py` will not append to it, `eval_schedule.py next` and `show` will not read it, and `analyze_eval.py report` will not analyse it. The report lists every deviation.

Changes to this page after the preregistration commit:

| date | section | change | before or after trial 1 | why | who |
|---|---|---|---|---|---|
| | | | | | |

## 13. Tools and files

**One folder per study,** committed with the repo: `docs/evidence/lfd_eval/<study_id>/`.
- `protocol.json` and `schedule.json` are committed before trial 1.
- `trials.jsonl`, `scorer_pack/scores.jsonl`, the pack key (`pack_key_scorer_pack.json`) and `report/` are committed after the last trial, once every scorer has finished. Until then the key stays out of git and out of the scorer's reach.
- The pictures stay out of git.

Run every command in the study folder. Below, `$T` stands for the clone's `tools/lfd` folder (for example `T=~/Berkeley-Humanoid-Lite/tools/lfd` on the robot PC), and `python` for the rig's `.venv/bin/python`.

| step | command |
|---|---|
| write the template protocol | `python $T/eval_schedule.py template --out protocol.json` |
| make the schedule | `python $T/eval_schedule.py make --protocol protocol.json --out schedule.json` |
| what to run next, and its `prime` command | `python $T/eval_schedule.py next` (add `--json` for scripts) |
| prime the recorder, then say "agent, start episode" | `python $T/record_ctl.py prime --mode policy --trial 17 --arrangement A07 --policy pi05 --task "..."` |
| the whole run sheet | `python $T/eval_schedule.py show --log trials.jsonl` |
| log a trial | `python $T/eval_log.py trial --trial T017 --success yes --time-s 21.4 --episode DIR --latency-file FILE` |
| log a failure | `python $T/eval_log.py trial --trial T018 --success no --timed-out yes\|no [--interventions 1 --collision]` |
| fix, withdraw, note | `python $T/eval_log.py amend`, `... withdraw`, `... deviation` |
| the blind scorer's folder, and its key | `python $T/eval_log.py pack --out scorer_pack [--episode-root DIR] [--video]` |
| a blind call | `python $T/eval_log.py score --sheet scorer_pack/sheet.jsonl --scorer NAME --code S3F9A1C --success no` |
| the analysis | `python $T/analyze_eval.py report --scores scorer_pack/scores.jsonl --pack-key pack_key_scorer_pack.json --out report` |
| the power table | `python $T/analyze_eval.py power --n 20 --m 4` |
| the tools' own tests | `python $T/test_eval.py` |

The schedule, log and analysis files default to `schedule.json` and `trials.jsonl` in the current folder. Each record names its operator: pass `--operator NAME`, or set `BHL_EVAL_OPERATOR`. The tools need Python 3.10 or later and numpy, so they run in the rig's `.venv` and in the HPC's LeRobot environment alike.

**The hash chain.** `schedule.json` embeds `protocol.json` with its SHA-256. Every log line carries the schedule's SHA-256. The report names all of them, plus the log's and each scorer file's.

## 14. Preregistration record

Fill before trial 1, and commit with `protocol.json` and `schedule.json`.

| field | value |
|---|---|
| study id | `block_bowl_v1` |
| preregistered on, and by | FILL |
| commit holding this page, `protocol.json` and `schedule.json` | FILL |
| `protocol_sha256` (printed in `schedule.json`) | FILL |
| the schedule's SHA-256 (printed by `make`, carried by every log line) | FILL |
| arm | right |
| block: size, material, colour, mass | 40 mm foam cube; FILL |
| open claw gap (measured) | FILL |
| bowl: inner diameter, rim height, colour, base | 150 mm or 120 mm; rim 30 mm; FILL |
| start pose | FILL |
| camera preset (pan, tilt), commanded degrees | FILL |
| Gate 0: date, method (camera or ruler), p90 hover error | FILL |
| mark coordinates and safe box | FILL (§2.6, §7.4) |
| demonstrations: count per arrangement, operators, sessions | FILL |
| dataset: path, label, lookahead, `conversion.json` and `splits.json` SHA-256 | FILL; `meas_future`; 0.10 s; FILL |
| checkpoints: path, SHA-256, steps, seed, per policy | FILL |
| execution per policy: chunk length, request interval, RTC | FILL |
| serving: where (HPC with COE approval, RunPod, or a home GPU), GPU type | FILL |
| go/no-go latency sweep (roadmap §4.4): date, p95 | FILL |
| latency `t_obs` | the sent frame's `t_cap`; `camera_latency_s` reported beside it if measured (§4) |
| `dataset.zero_convention` in `protocol.json` | FILL (`hanging=0` today) |
| operator(s); scorer(s), each never exposed to the schedule, the log or the key (§9) | FILL; or "none: unblinded" |
| S4 duty-cycle threshold; S6 deadline | 20 %; 14 days; FILL to confirm |
| study B (§3.2): run or not, and why; its primary variant | FILL; masked primary, unmasked secondary |
