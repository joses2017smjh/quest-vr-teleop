# Learning-from-demonstration recordings: format v1

This page is the contract between two halves of the pipeline:
- **Robot side:** `run_teleop.py --record-tap` and `scripts/teleop/recorder.py`.
- **HPC side:** `tools/lfd/convert_to_lerobot.py` and the training and evaluation tools.

Anything that writes or reads a recording follows this page. Change the page first, then bump `version`.

**Why not record straight into LeRobot?** The robot PC runs Python 3.10 and is CPU-bound (an N150 at load ≈ 10). LeRobot ≥ 0.5 needs Python 3.12, and it decodes and re-encodes video. So the robot only copies bytes and writes JSON lines, and all decoding happens on the HPC. See `docs/ROADMAP_2026-10-02.md` §3.7, §4.7 and §5.

**Enabling the tap means restarting `run_teleop.py`, and a restart retakes zero.** Follow the rule in `CLAUDE.md`:
- restart only when it is STOPPED, with the motors off and the arms hanging still at zero;
- check `curl -s localhost:8080/state.json` first;
- start it with the explicit command, never with "Up Enter".

## 1. One clock

Every timestamp in a recording is **`time.monotonic()` seconds on the robot PC**. On Linux, CPython's `time.monotonic()` and `time.perf_counter()` read the same `CLOCK_MONOTONIC`. Even so, the tap and the recorder each stamp `time.monotonic()` themselves rather than reuse the loop's `now`.

There is one exception. The camera frame file's mtime is wall time (`CLOCK_REALTIME`), so the recorder converts it once per frame:

```
t_cap = mtime_ns · 1e-9 − (time.time() − time.monotonic())
```

- The offset is sampled when the recorder notices the new frame.
- `t_cap` is when ffmpeg finished writing the frame.
- File mtimes come from the kernel's coarse clock (1–4 ms ticks), so `t_cap` has that granularity.
- Two frames can share an mtime, so the recorder detects a new frame by **(inode, mtime)**, not by mtime alone.
- The camera's own exposure-to-write latency is a constant still to be measured (roadmap step A4). `session.json` carries it as `camera_latency_s`, null until measured, and the converter subtracts it when it is set.
- The evaluation tools report policy latency from the raw `t_cap` of the frame sent, with `camera_latency_s` reported beside it.

## 2. The tap: run_teleop → recorder

- **Off by default.** Enable it with `run_teleop.py --record-tap`, sending to `--record-tap-port` (default **11014**, range 1024–65535; given without `--record-tap` it warns and is ignored). With the flag off, `run_teleop.py` creates no socket and does no per-cycle work.
- **Transport.** One JSON datagram per control cycle (≈ 48 Hz), sent from `ArmSupervisor.tick` to `127.0.0.1:11014` right after `driver.exchange()`.
  - The socket is non-blocking.
  - A failed `sendto`, or a row that would still hold a non-finite value, is dropped and counted in `tap_errors`, never raised. The count shows in `run_teleop`'s once-a-second summary line (`tap errors N`, only when N > 0) and as the status key `tap_errors` (only with `--record-tap`; without it the status payload is unchanged).
  - The tap must never block, raise, or change what the loop commands. On a loaded Xeon E5-2630 v4 a row costs about 0.17 ms, or 0.22 ms with the null mapping. On the N150 that is estimated at a few percent of the 20.8 ms cycle; it has not been measured on the robot.
- **JSON rules.** Rows are compact JSON. **Any non-finite float is written as `null`**, never as NaN or Infinity tokens; for example, `driver.q` is NaN for a joint not yet heard from. That holds for every float field except `zero`, which `capture_zero` keeps finite. Readers treat `null` as missing.
- **Joint order.** Every joint vector uses `ARM_JOINTS` order: L pitch, L roll, L yaw, L elbow, L wrist, R pitch, R roll, R yaw, R elbow, R wrist.
- **Row fields (v1):**

| key | type | meaning |
|---|---|---|
| `v` | int | tap format version, `1` |
| `seq` | int | control-cycle counter since start-up. It advances even when a send fails, so a gap means a lost row |
| `t` | float | `time.monotonic()` just after `driver.exchange()` |
| `state` | str | `ArmSupervisor.state`: `STOPPED`, `ARMED`, `CAL`; reserved: `POLICY`, `GUIDE` |
| `motors` | bool | motors on |
| `src` | str | what produced `qc` this cycle: `teleop` (ARMED), `cal`, `idle`. Reserved: `policy`, present from the first cycle in which a policy commands the arm, including hold cycles while its queue is empty (evaluation takes t = 0 from the first such row); and `guide` |
| `q` | (float \| null)[10] | measured joint angles after the exchange (`driver.q`), URDF frame, rad |
| `qc` | float[10] | joint targets sent this cycle (`q_cmd`), rad |
| `tau` | (float \| null)[10] | measured joint torque (`driver.tau`), N·m |
| `tff` | float[10] | torque feed-forward sent this cycle, N·m |
| `lim` | float[10] | the torque limits `run_teleop` asks for (`ArmSupervisor.limits`), N·m. `LimitWriter` lowers limits lazily, so the motor's own clamp can sit above this for up to about 1 s |
| `zero` | float[10] | `driver.zero_fw`, the firmware-frame zero. It changes only when zero is taken again |
| `held` | bool[2] | `TeleopIkSolver.held` per robot arm (L, R) |
| `grip` | bool[2] | controller grips, operator's left and right, **before** any hand swap |
| `trig` | float[2] | controller triggers, 0..1, before any swap |
| `hand` | (float[16] \| null)[2] | controller poses as `parse_packet` gives them (robot frame), 4×4 row-major, before any swap; null when not tracked |
| `head` | float[9] \| null | headset rotation (robot frame), 3×3 row-major |
| `tgt` | (float[3] \| null)[2] | IK target translation per robot arm; null when not held |
| `swap` | bool | `profile.swap_hands` (mirror mode) |
| `scale` | float | `profile.motion_scale` |
| `claw` | (int \| null)[2] | the **last pulse (µs) sent through `ServoLink.move`** to the robot-left claw (pin 3, D3) and the robot-right claw (pin 7, D7) since `run_teleop` started. It is recorded on a successful send only and persists across armings. Null only before the first pulse. It is a record of commands: after the claw setup's OFF (limp) or a Nano reset it still shows the last pulse sent, and setup moves count too |
| `cam` | float[2] \| null | `CameraAim.angle` (pan, tilt), commanded degrees; [0, 0] until commanded. Null when the servo link is not ready (the Nano unplugged or dropped), and in tests without servos |
| `cam_mode` | str \| null | `CameraAim.mode`: `track` (the default at start-up), `head`, `still`. Set whenever `CameraAim` exists, even when `cam` is null; null only without servos |
| `intervention` | bool | reserved; always false until a POLICY state exists |

A row is about 1 KB.

Planned fields, not yet emitted, for the GUIDE state of `docs/LFD_HAND_GUIDING.md` §3.7: `deadman` (the enabling switch's position, 1, 2 or 3), `guide` bool[2], and `kp` / `kd` float[10].

## 3. The recorder process

`scripts/teleop/recorder.py` runs beside `run_teleop.py` on the robot PC. It:
1. reads tap rows from UDP 11014;
2. watches the camera frame file `/dev/shm/bhl_camera.jpg` (`camera_share.FRAME_FILE`) and copies the bytes of each new frame. It never decodes them. ffmpeg writes the file atomically (`-atomic_writing 1`), so a read never sees half a frame;
3. writes episodes, only between `start` and `end`;
4. answers control commands on UDP **11015**.

Both sockets bind `127.0.0.1`. It sends nothing to `run_teleop.py`, the CAN buses or the servos, only replies to whoever sent a command.

### Layout

```
<root>/                                   default ~/bhl_recordings on the robot PC
  <session_id>/                           YYYYmmdd_HHMMSS_<host>, one per recorder run
    session.json
    ep_0000/
      episode.json
      rows.jsonl                          tap rows, one per line, as received
      frames.mjpeg                        the camera's JPEG bytes, concatenated as received
      frames.jsonl                        one line per frame: {"i", "off", "len", "t_cap", "t_seen", "mtime_ns"}
    ep_0001/ ...
    _discarded/ep_0002/ ...               moved here by "discard"
```

`t_seen` is `time.monotonic()` when the recorder saw the new frame. `off` and `len` locate the frame's bytes in `frames.mjpeg`.

### session.json

```json
{"format": "bhl-lfd-recording", "version": 1, "session_id": "...", "created_wall": 0.0, "host": "...",
 "repo_commit": "git rev-parse HEAD or null", "joint_names": ["left_shoulder_pitch", "..."],
 "urdf_joint_names": ["arm_left_shoulder_pitch_joint", "..."],
 "camera": {"frame_file": "/dev/shm/bhl_camera.jpg", "layout": "side_by_side", "rotated_180": true,
            "calibration": "configs/stereo_<W>x<H>.json or null", "camera_latency_s": null},
 "tap_port": 11014, "control_port": 11015}
```

`camera.calibration` names `configs/stereo_<W>x<H>.json` only if that file exists. W and H are read from the JPEG's SOF header, without decoding.

### Control commands (UDP 11015)

Each command is a JSON object whose `"cmd"` key names the command. Every reply is `{"ok", "said", ...}`.

| `cmd` | fields | effect |
|---|---|---|
| `start` | `task`, `mode` (`teleop`, `guided` or `policy`), `operator`, `arrangement` (str), `trial` (int), `policy` (str) | Opens a new episode. Fields left out come from the active prime (below), then from the recorder's defaults (`--task`, `--operator`). `mode` defaults to `teleop`; voice starts carry no mode. `record_ctl.py start` requires `--mode`. Types are enforced: `trial` is an int (not a bool); `arrangement`, `policy` and `task` are strings; `mode` is one of the three. **Refused** if: an episode is open; no tap row or no frame arrived in the last 0.5 s; the task is empty; free disk is below `--min-free-gb` (default 2.0); or `cam_mode` is set and is not `still`, unless the recorder runs with `--allow-moving-camera`. That refusal tells the operator to say "agent, camera still". `cam_mode` null (no servos) is accepted with the warning 'no camera servos: camera angle unknown'; `cam` null with `cam_mode` set warns 'camera servos not connected: camera angle unknown'. The reply lists any primed fields it used. |
| `prime` | any of `mode`, `trial`, `arrangement`, `policy`, `task` | Sets values for the **next** start only. Only these five fields are accepted; unknown keys are refused. Strings are stripped, and empty ones dropped; a prime with no fields clears it. A new prime replaces the old one. Fields given in the start command win. A refused start leaves the prime alone; any start that opens an episode uses it up. It expires after 10 minutes unused, and is dropped with a note on the recorder's terminal, so a stale prime cannot leak into a later demo. The prime carries no `operator`: voice starts use the recorder's `--operator`. Use it before a voice start of an evaluation trial; `tools/lfd/eval_schedule.py next` prints the command. |
| `end` | `outcome` (`success`, `failure` or `aborted`), optional `notes` | Closes the open episode. The reply comes **before** the save, ending with ' - saving'. Nothing more is written to the episode after it, and the save runs before any other command is read. `record_ctl.py end` then polls `status` for up to 120 s and prints `ok ep_NNNN saved`, or exits 1. |
| `discard` | — | An open episode is ended with outcome `aborted`, end_reason `discarded`, `discarded: true`, and moved to `_discarded/`. With nothing open, the newest closed episode is moved instead. A second discard is refused. |
| `task` | `task` | sets the default task text |
| `status` | — | the open episode; `prime` (the primed fields, or null) and `prime_expires_in_s`; `last_episode` ({name, outcome, end_reason, saved, discarded}, or null); rows/s, frames/s, last-row and last-frame ages, warnings, free disk |

An episode **ends by itself** with outcome `aborted` when any of these happens:
- tap rows or frames stop for more than 2 s;
- the zero (`zero`) changes mid-episode;
- free disk falls below `--min-free-gb`, which is checked every 2 s (end_reason `low disk`). This protects `run_teleop`'s own teleop log on the same disk. A separate warning repeats every 30 s below 5 GB;
- a write fails;
- the recorder gets SIGINT, SIGTERM or SIGHUP (end_reason `recorder stopped`). A recorder started with `nohup` ignores SIGHUP and keeps running.

`episode.json` is written **provisionally at start**, with `end`, `outcome` and `end_reason` null, and rewritten at the end. A folder whose outcome is still null after the recorder has stopped is an incomplete episode, and readers reject it.

**Ways to send commands:**
- `tools/lfd/record_ctl.py` sends them from a terminal (`start`, `prime`, `end`, `discard`, `task`, `status`).
- From the headset, `voice_typer.py` sends:
  - "agent, start episode"
  - "agent, end episode success" / "… fail"
  - "agent, abort episode" (also "aborted", "cancel", "cancelled")
  - "agent, discard episode"
  - "agent, episode status"
- Episode phrases are recognised by the words episode(s) and recording(s), and are parsed **before** tuning phrases. Any sentence containing one is an episode command or unknown, never a tune. 'help' is still parsed first.
- Abort words: abort, aborted, cancel, cancelled, canceled.
- The word "stop" never ends an episode: to this operator it means the motors.
- "End episode" without an outcome is asked again, not sent.
- The headset shows 'recorder: <said>' or 'recorder - not done: <said>'.
- Voice and `record_ctl` use a connected UDP socket, so a recorder that is not running is reported at once. A real timeout (voice 4 s, `record_ctl` 2 s) says the recorder may be busy.

### episode.json

```json
{"format": "bhl-lfd-episode", "version": 1, "episode_index": 0, "session_id": "...",
 "task": "Put the foam block in the bowl", "mode": "teleop", "policy": null,
 "operator": null, "arrangement": null, "trial": null,
 "start": {"t": 0.0, "wall": 0.0}, "end": {"t": 0.0, "wall": 0.0},
 "outcome": "success", "end_reason": "operator", "discarded": false, "notes": "",
 "counts": {"rows": 0, "frames": 0, "row_gaps_over_50ms": 0, "frame_gaps_over_67ms": 0},
 "cam_mode_at_start": "still", "cam_at_start": [0.0, 0.0],
 "zero_at_start": [0.0], "zero_changed": false, "zero_convention": "hanging=0",
 "zero_check": null, "profile_sha256": "...", "claw_limits": {"left": {"open_us": 1000, "closed_us": 1640},
                                                               "right": {"open_us": 1000, "closed_us": 1670}},
 "warnings": []}
```

**`zero_convention`.** Today it is `"hanging=0"`: the hanging pose is taken as URDF zero. Once roadmap §3.1's corrected zero exists it becomes `"hanging=q_hang"`. The recorder reads it from the profile's `zero_convention` key. Watch out: `PowerProfile.save()` currently rewrites the profile from its dataclass fields only, and that would erase a hand-added key. So the §3.1 fix must add `zero_convention` to `PowerProfile` itself.

**`zero_check`.** This is a copy of `configs/zero_check.json`, the shoulder-roll inclinometer test of roadmap §3.1. `tools/lfd/zero_check.py` writes it as `{"format": "bhl-zero-check", "version": 1, ...}` with these keys:
- `date` and `method`;
- `readings_deg` (`torso`, `left_upper_arm`, `right_upper_arm`) and `sign_convention`;
- `relative_to_torso_deg` and `decisions`, per arm;
- the overall `decision`: `holds`, `rejected` or `report`;
- `rule`, `profile_sha256`, `zero_fix_applied` and `notes`.

Readers rely on `decision` and `date`.

It is null if the test has not been recorded.

## 4. Turning episodes into training data (the converter's rules)

The converter writes the dataset to `OUT_ROOT/REPO_ID` (for example `local/NAME`). `conversion.json` and `splits.json` go in that dataset root, beside `meta/`.

### Which episodes are used

The converter refuses (each refusal is listed in `conversion.json` with its reason):
- an outcome other than `success` or `failure`. That covers null (incomplete) and `aborted`;
- discarded episodes;
- `policy` episodes, unless asked for;
- an empty task;
- a torn JPEG among the frames the steps use;
- a frame size that changes within the episode;
- more than 2 % missing steps (below);
- rows whose `src` contradicts the episode's `mode`:
  - a `teleop` episode with any `policy` or `guide` row;
  - a `guided` episode with any `teleop` or `policy` row;
  - a `policy` episode with any `guide` row. A `policy` episode may hold `teleop` rows: the operator's interventions. `idle` and `cal` rows are allowed in every mode;
- a session.json of another format or version, or without 10 distinct joint names;
- tap rows of a version other than 1, or rows lacking a key the converter reads (`t`, `src`, `q`, `qc`, `zero`, `claw`, `cam`, `intervention`);
- a `zero_at_start` that differs from the rows' `zero`;
- an odd frame width;
- a frame that a step uses but that fails to decode. It is refused while the episode is being written, and listed in `rejected`.

The zero check works like this:
- A null `zero_check`, a non-object one, a `report` (inconclusive) decision, or a decision outside `holds` / `rejected` / `report` is refused unless `--allow-unverified-zero` is given. Readers use `decision` only.
- `holds` with `zero_fix_applied: false` is accepted with a warning: the data are consistent within their `zero_convention`, but the gravity model was off while recording.

It refuses to **mix**, in one dataset, episodes with different `joint_names`, claw limits, camera frame sizes or `zero_convention`s. A mixing violation refuses the whole run and writes nothing. It is checked only among episodes that pass every per-episode rule.

### Time grid
- `fps` is 30, and the steps are `t_k = t_0 + k / fps`.
- The grid runs from the first moment both streams are present to `min(last frame, last tap row)`. `q` is interpolated, never extrapolated.
- Frames are ordered by `(t_cap − camera_latency_s, i)`. Equal or near-equal frame times are allowed: the coarse mtime clock can give two frames the same time. Refused:
  - a frame more than 5 ms below the running maximum of the earlier frame times (a clock step);
  - an index `i` that does not increase down `frames.jsonl`;
  - a missing or negative `i`.
- **Each step uses the frame nearest to `t_k` within ±1/fps.** A camera slightly off 30 Hz therefore reuses or skips a frame now and then, instead of losing whole episodes. Counted on the full grid, before `meas_future` trims its tail, so both labels and the validator report the same numbers:
  - `missing`: steps with no frame within ±1/fps;
  - `reused`: frames used by more than one step;
  - `skipped`: frames between the first and last step that no step uses.
- An episode is rejected if `missing` / steps > 2 % (`--max-missing 0.02`), or if (`reused` + `skipped`) / steps > 5 % (`--max-reused-skipped 0.05`). The counts are in `conversion.json` (`episodes[].grid`) and in the validator's grid line.
- **Threshold risk, to settle with the first real session.** With ±4 ms of jitter (one coarse-clock tick) at 29.97 Hz, reused + skipped reached 3.8–5.2 % in simulation. Those are mostly adjacent reuse/skip pairs, not drops. The validator reports the counts: if real sessions sit near 5 %, raise `--max-reused-skipped` (to about 0.08) and note it as a deviation. A genuinely wrong rate (28 or 32 Hz) gives about 6–7 % and stays rejected at 0.05.

### State and actions
- **State.** `observation.state` (12-D) is `q(t_k)` (10 joints, linearly interpolated between tap rows) followed by the two claws. A null `q` or `qc` in a row the steps read refuses the episode. Those rows are the two that bracket each interpolation time (`t_k`, and `t_k + H` for `meas_future`), plus the held row of every step for `qc`. Nulls elsewhere are ignored. A null `zero` anywhere refuses the episode. Each claw value is `(pulse − open_us) / (closed_us − open_us)`, clipped to 0..1, taken from the last pulse at or before `t_k`. It is a command, not a measurement: the claws have no feedback. A claw never commanded since start-up counts as 0 (open), with a warning.
- **`cmd` label (teleop only).** The target the controller was told to reach: `action = [qc(t_k), claw(t_k)]`, holding the last tap row at or before `t_k`. Under `cmd` the claw entries of action and state are equal, because there is no claw feedback. Chunked policies (ACT, the VLAs) still see claw changes inside each chunk.
- **`meas_future` label (teleop or hand-guided).** Where the arm actually went next: `action = [q(t_k + H), claw(t_k + H)]`, with lookahead `H` (default 0.10 s, about the arm's measured tracking lag during teleop; set it with `--lookahead`).
  - The last `H` seconds of each episode have no label and are trimmed. A 60 s episode gives 1797 ± 1 steps, not 1800.
  - For a hand-guided episode this is the **only** valid label. While the operator moves a compliant arm, `qc` is not a demonstration, and a stiff "hold position" `qc` would be exactly the wrong label.
- **Comparisons.** To compare teleop against hand guiding fairly, convert both with `meas_future`. To compare models, give every model the same converted dataset and the same `splits.json`.

### Images
1. Decode the JPEG.
2. If `session.json` says `rotated_180`, rotate the **whole** side-by-side frame by 180°.
3. Split it at half width.

After the rotation, the left half is the **lens on the robot's left as mounted**. Because the module hangs upside down, that is the module's own right lens, which the raw frame holds on its right half. That half becomes `observation.images.left`, and the other half `observation.images.right`. This was verified by drawing a marker in the raw right half and finding it in `observation.images.left`. This matches `depth_flow.grab` and the headset's own per-eye mapping (`cameraUv` in `quest_bridge.html`). Both are stored as video. `--image-size` shrinks each eye.

### Extra columns
These are for analysis, not policy inputs, and deliberately not under `observation.`. LeRobot's `dataset_to_policy_features` only takes keys starting with `observation` or `action` (checked in LeRobot 0.6.1).

| column | type | meaning |
|---|---|---|
| `q_cmd` | float32[10] | the controller's target at the step |
| `cam_pan_tilt` | float32[2] | 0.0 when unknown |
| `cam_known` | bool | the held tap row had a camera angle |
| `mode_id` | int64 | 0 teleop, 1 guided, 2 policy |
| `intervention` | bool | LeRobot's DAgger convention |
| `t_rel` | float32 | seconds since the episode's first step; equal to LeRobot's own `timestamp` (k/fps). The absolute `t0` (monotonic) of each episode is in `conversion.json`, because float32 cannot hold the uptime in seconds precisely |
| `frame_skew` | float32 | `t_k` minus the used frame's `t_cap − camera_latency_s`, s |

### Splits
Validation is done on **whole episodes**: neighbouring frames of one demonstration never land on both sides.

`splits.json` (format `bhl-lfd-splits`, version 1) holds:
- `train` and `val`, as lists of LeRobot `episode_index` values;
- `by_mode`, as `{mode: {"train": [...], "val": [...]}}`;
- `episodes`, as `{index: {"source", "mode", ...}}`;
- `seed`, `val_fraction` and the rule text.

The rule (`tools/lfd/splits.py`):
1. Each mode is split on its own. `round(val_fraction · n)` of its episodes go to validation. Python's round sends a tie to the even number.
2. A mode with ≥ 5 episodes gives at least one to validation, and keeps at least one for training.
3. Once the dataset has ≥ 2 episodes, there is always at least one validation episode overall. It comes from the largest mode that can still keep a training episode, otherwise from the largest mode.
4. With `val_fraction` 0, nothing goes to validation.

Episodes are chosen in `sha256(f"{seed}:{source}")` order, so the split does not depend on input order.

### Statistics
LeRobot 0.6.1 writes the count-weighted *mean of per-episode quantiles*. That is not a dataset quantile, and π0.5 normalizes with the q01/q99 entries.

So after `finalize()`, the converter replaces every quantile key (`q01`, `q10`, `q50`, `q90`, `q99`) of `observation.state` and `action` in `meta/stats.json` with `numpy.quantile` (default 'linear') over all frames:
- count, min and max are kept, and checked exactly;
- mean and std are kept (LeRobot's float32 values);
- other features keep LeRobot's quantiles, and `meta/episodes` keeps the per-episode stats;
- `meta/stats.json` is strict JSON: no NaN or Infinity anywhere;
- `conversion.json` records `stats.recomputed: true`.

The statistics cover all episodes, validation included. The leak is summary statistics only, and identical for every model compared.

To fix an older dataset, re-run the converter; LeRobot's `recompute_stats` re-averages the quantiles.

### Provenance (`conversion.json`)
It records:
- the arguments, including `max_missing` and `max_reused_skipped`;
- each input's `episode.json` and `rows.jsonl` sha256;
- the label and lookahead;
- per episode: the source, `t0`, `trial`, `arrangement`, `operator`, `policy`, `outcome`, `end_reason`, `notes`, the grid counts and warnings;
- the rejected episodes and their reasons;
- `code`: the repo commit, `repo_dirty`, `tools_lfd_status` and `splits_sha256`;
- `encoding`: the options, the seconds spent, and the video settings from `info.json`;
- `stats.recomputed`;
- the LeRobot version.

`conversion.json` is written last. A failure after the dataset is written exits 2 ('FAILED') and leaves no `conversion.json`, so a half-written dataset is never mistaken for a finished one.

### Validation (`tools/lfd/validate_recording.py`)
- An ERROR means the converter would refuse the episode for a defect, under at least one label (label-dependent checks use the validator's own `--lookahead`).
- Deliberate exclusions (aborted, discarded) are notes.
- An incomplete episode (outcome null) is one WARNING ('the converter skips it').
- Mixing conflicts are WARNINGS. So are nulls in `tau`, `tff`, `lim`, `hand`, `head` and `tgt`, a gap in `seq`, frame defects in frames no step shows, and an episode under 90 % rows of its own `src`.

**Cost.** Encoding is AV1 (libsvtav1) on the CPU. The defaults are:
- streaming encoding on;
- SVT threads = half of the job's CPUs;
- 2 PNG-writer threads;
- parallel encoding when the job has ≥ 4 CPUs, counted with `sched_getaffinity`, not the node's core count.

On a 2-CPU job, a 60 s episode at 640×480 per eye took 2 min 17 s with the old defaults. The new defaults about halve that on synthetic frames, and real camera frames will be slower. Run real conversions as a CPU job on the `share` partition, with 8 CPUs and 16 GB. The converter's docstring has an `sbatch --wrap` example. Streaming relies on private attributes of LeRobot 0.6.1, and the converter refuses to stream if they are absent.

## 5. Units and frames, once more
- Joint angles are in rad, in the URDF frame, under the episode's `zero_convention`.
- Torques are in N·m.
- Camera angles are commanded servo degrees: SG90s give no feedback.
- Controller and head poses are in the robot frame of `parse_packet`.
- Times are monotonic seconds on the robot PC.

## Changes since the first draft (no data had been recorded)

**Tap**
- `claw` is now the last pulse through `ServoLink`, which persists across armings.
- Non-finite numbers are written as `null`.
- The row size, the `lim` semantics and the restart rule are spelled out.
- The real robot's `cam` / `cam_mode` behaviour is described.

**Recorder**
- The command key `cmd` and the field types are fixed; `mode` is optional for voice starts.
- New: the `prime` command, the discard rules, and the low-disk refusal and abort.
- SIGHUP now finalizes an open episode.
- `episode.json` is written provisionally at start.
- Frames are detected by (inode, mtime).

**Converter**
- The grid end is defined, and so are missing-step handling and the `frame_skew` base.
- Zero-check handling is defined.
- New refusals: empty task, torn JPEG, size change.
- The eye mapping is verified against the rig.
- `t_mono` becomes `t_rel` plus `t0`.
- `intervention` is now bool.
- The splits schema now guarantees at least one validation episode.
- The stats note and the conversion cost are added.

**Second round, after review and fixes**
- Tap: every float may be null; rows that cannot be cleaned are dropped and counted; `tap_errors` is shown; `cam` is null when the servo link is not ready; `POLICY`/`GUIDE` states and the `policy` source are reserved and defined; the planned GUIDE fields are named.
- Recorder: the `--min-free-gb` name; enforced field types; the full `prime` rules; `end` replies before saving; the new `status` fields; voice keywords, abort words and timeouts.
- Converter: refuses rows whose `src` contradicts the episode's `mode`.
- `zero_check`: the keys are named.

**Third round, after the data-path review and fixes**
- Time grid: the ±1/fps rule with missing/reused/skipped counts and their thresholds, the frame order rules, and a threshold risk to settle on the first real session.
- Null rows: the rule for nulls in the rows the steps read.
- Extra columns: `cam_known`; `t_rel` equals LeRobot's timestamp.
- Stats: true quantiles, strict JSON.
- Splits: the exact rule and shape.
- Mode/src: policy episodes may hold no guide rows.
- Refusals: more cases listed. Validator semantics, provenance and cost spelled out.
