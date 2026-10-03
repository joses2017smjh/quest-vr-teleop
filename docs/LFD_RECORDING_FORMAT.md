# Learning-from-demonstration recordings: format v1

This is the contract between the robot side and the HPC side:
- **Robot side:** `run_teleop.py --record-tap` and `scripts/teleop/recorder.py`.
- **HPC side:** `tools/lfd/convert_to_lerobot.py` and the training and evaluation tools.

Anything that writes or reads a recording follows this page. Change the page first, then bump `version`.

Why a format of our own, rather than recording straight into LeRobot:
- The robot PC runs Python 3.10 and is CPU-bound (an N150 at load ≈ 10).
- LeRobot ≥ 0.5 needs Python 3.12, and it decodes and re-encodes video.

So the robot only copies bytes and writes JSON lines, and all decoding happens on the HPC. See `docs/ROADMAP_2026-10-02.md`, §3.7, §4.7 and §5.

## 1. One clock

Every timestamp in a recording is **`time.monotonic()` seconds on the robot PC**. On Linux, CPython's `time.monotonic()` and `time.perf_counter()` read the same `CLOCK_MONOTONIC`, but the tap and the recorder stamp `time.monotonic()` themselves, rather than reusing the loop's `now`.

There is one exception. The camera frame file's mtime is wall time (`CLOCK_REALTIME`). The recorder converts it to monotonic time once per frame:

```
t_cap = mtime_ns · 1e-9 − (time.time() − time.monotonic())
```

The offset is sampled at the moment the recorder sees the new frame. `t_cap` is the moment ffmpeg finished writing the frame. The camera's own exposure-to-write latency is a constant still to be measured (roadmap step A4). `session.json` carries it as `camera_latency_s` (null until measured), and the converter subtracts it when it is set.

## 2. The tap: run_teleop → recorder

- **Off by default.** It is enabled by `run_teleop.py --record-tap` and sends to `--record-tap-port`, default **11014**. With the flag off, `run_teleop.py` behaves exactly as before.
- **Transport.** One JSON datagram per control cycle (≈ 48 Hz), sent to `127.0.0.1:11014` from `ArmSupervisor.tick` right after `driver.exchange()`. The socket is non-blocking. A failed `sendto` is counted, never raised. The tap must never block, raise, or change what the loop commands.
- **Row fields (v1).** Joint vectors use `ARM_JOINTS` order: L pitch, L roll, L yaw, L elbow, L wrist, R pitch, R roll, R yaw, R elbow, R wrist.

| key | type | meaning |
|---|---|---|
| `v` | int | tap format version, `1` |
| `seq` | int | control-cycle counter since start-up |
| `t` | float | `time.monotonic()` just after `driver.exchange()` |
| `state` | str | `ArmSupervisor.state`: `STOPPED`, `ARMED`, `CAL` |
| `motors` | bool | motors on |
| `src` | str | what produced `qc` this cycle: `teleop` (ARMED), `cal`, `idle`. Reserved: `policy`, `guide` |
| `q` | float[10] | measured joint angles after the exchange (`driver.q`), URDF frame, rad |
| `qc` | float[10] | joint targets sent this cycle (`q_cmd`), rad |
| `tau` | float[10] | measured joint torque (`driver.tau`), N·m |
| `tff` | float[10] | torque feed-forward sent this cycle, N·m |
| `lim` | float[10] | torque limits in force, N·m |
| `zero` | float[10] | `driver.zero_fw`, the firmware-frame zero. It changes only when zero is taken again |
| `held` | bool[2] | `TeleopIkSolver.held` per robot arm (L, R) |
| `grip` | bool[2] | controller grips, operator's left and right, **before** any hand swap |
| `trig` | float[2] | controller triggers, 0..1, before any swap |
| `hand` | (float[16] \| null)[2] | controller poses as `parse_packet` gives them (robot frame), 4×4 row-major, before any swap; null when not tracked |
| `head` | float[9] \| null | headset rotation (robot frame), 3×3 row-major |
| `tgt` | (float[3] \| null)[2] | IK target translation per robot arm, null when not held |
| `swap` | bool | `profile.swap_hands` (mirror mode) |
| `scale` | float | `profile.motion_scale` |
| `claw` | (int \| null)[2] | last pulse (µs) sent to the robot-left claw (D3) and robot-right claw (D7), from `ServoTeleop.out.sent`; null if none sent yet |
| `cam` | float[2] \| null | `CameraAim.angle` (pan, tilt), commanded degrees; null without servos |
| `cam_mode` | str \| null | `CameraAim.mode`: `track`, `head`, `still` |
| `intervention` | bool | reserved; always false until a POLICY state exists |

A row is about 2 KB.

## 3. The recorder process

`scripts/teleop/recorder.py` runs beside `run_teleop.py` on the robot PC. It:
1. Reads tap rows from UDP 11014, keeping a short ring buffer so an episode can check the stream is live before it starts.
2. Watches the camera frame file `/dev/shm/bhl_camera.jpg` (`camera_share.FRAME_FILE`). It polls `st_mtime_ns` every 5 ms, and when the mtime changes it copies the bytes. It never decodes them. ffmpeg writes the file atomically (`-atomic_writing 1`), so a read never sees half a frame.
3. Writes episodes, only between `start` and `end`.
4. Answers control commands on UDP **11015**.

It sends nothing to `run_teleop.py`, the CAN buses or the servos.

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

### Control commands (UDP 11015, JSON in, JSON reply `{"ok", "said", ...}`)

| command | fields | effect |
|---|---|---|
| `start` | `task` (optional when a default is set), `mode` (`teleop`, `guided`, `policy`), optional `operator`, `arrangement`, `trial`, `policy` | opens a new episode. Refused while one is open, and refused if no tap row or no frame arrived in the last 0.5 s. Refused if `cam_mode` is not `still`, unless the recorder runs with `--allow-moving-camera`: a moving head camera changes the viewpoint between episodes. |
| `end` | `outcome` (`success`, `failure`, `aborted`), optional `notes` | closes and finalizes the open episode |
| `discard` | — | moves the open episode, or else the last one, to `_discarded/` |
| `task` | `task` | sets the default task text used by voice starts |
| `status` | — | open episode, rows/s, frames/s, last-row and last-frame ages, warnings, free disk |

An episode also ends by itself, with outcome `aborted`, if tap rows or frames stop for more than 2 s, or if the zero (`zero`) changes mid-episode.

`tools/lfd/record_ctl.py` sends these commands from a terminal. From the headset, "agent, start episode", "agent, end episode success/fail", "agent, discard episode" and "agent, episode status" do the same through `voice_typer.py`.

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

- `zero_convention`: `"hanging=0"` today, where the hanging pose is taken as URDF zero. Once roadmap §3.1's corrected zero exists, it becomes `"hanging=q_hang"`. The recorder reads it from the profile's `zero_convention` key, if present.
- `zero_check`: a copy of `configs/zero_check.json`, the shoulder-roll inclinometer test (roadmap §3.1), written by `tools/lfd/zero_check.py`. It is null if that test has not been recorded.

## 4. Turning episodes into training data (the converter's rules)

### Time grid
For each episode:
- `fps` = 30.
- `t_k = t_0 + k / fps`, from the first frame after both streams are present to the last frame.
- Each step uses the frame whose `t_cap` is nearest to `t_k` within ±0.5/fps.
- Steps with no frame in that window are missing. More than 2 % missing steps rejects the episode, with a reason in the report.

### State
`observation.state` is 12-D: `q(t_k)` (10 joints, linear interpolation between tap rows) followed by the two claws.

Each claw value is `(pulse − open_us) / (closed_us − open_us)`, clipped to 0..1, taken as the last pulse sent at or before `t_k`. It is a command, not a measurement: the claws have no feedback. A claw that has never been commanded counts as 0 (open), with a warning.

### Two action labels
Every converted dataset says which label it uses (`conversion.json`).

- **`cmd`** (teleop only): the target the controller was told to reach. That is `action = [qc(t_k), claw(t_k)]`, holding the last tap row at or before `t_k`.
- **`meas_future`** (teleop or hand-guided): where the arm actually went next. That is `action = [q(t_k + H), claw(t_k + H)]`, with lookahead `H` defaulting to 0.10 s. This is about the arm's measured tracking lag during teleop (roadmap §2.1, §3.7); it can be changed with `--lookahead`.
  - The last `H` seconds of each episode have no label and are trimmed.
  - For a hand-guided episode this is the **only** valid label. While the operator moves a compliant arm, `qc` is not a demonstration. If `qc` were a stiff "hold position" command, it would be exactly the wrong label.

To compare teleop against hand guiding fairly, convert both with `meas_future`. To compare models, give every model the same converted dataset.

### Images
1. Decode the JPEG.
2. Rotate the whole side-by-side frame by 180°: the camera is mounted upside down, as `depth_flow.grab` and `person_track` already handle.
3. Split it at half width.
4. Store the halves as `observation.images.left` and `observation.images.right` (video).

### Extra columns
These are for analysis, not policy inputs. They are deliberately not under `observation.`:

| column | contents |
|---|---|
| `q_cmd` | float[10] |
| `cam_pan_tilt` | float[2] |
| `mode_id` | 0 teleop, 1 guided, 2 policy |
| `intervention` | 0/1 |
| `t_mono` | the episode's step times |
| `frame_skew` | `t_k` minus the used frame's `t_cap`, s |

### Mixing rules
The converter refuses to put these into one dataset:
- episodes with different `joint_names`;
- episodes with different claw limits;
- episodes with different camera frame sizes;
- episodes with different `zero_convention`s.

It also refuses episodes:
- whose `zero_check` is null, unless `--allow-unverified-zero` is given;
- that are discarded;
- with outcome `aborted`;
- recorded in `policy` mode, unless asked for.

### Splits
Validation is done on **whole episodes**: neighbouring frames of one demonstration never land on both sides.
- Episodes are split per `mode`, with a fixed seed.
- 20 % go to validation, with at least one per mode once a mode has ≥ 5 episodes.
- The split is written to `splits.json` beside the dataset.

## 5. Units and frames, once more
- Joint angles are in rad, in the URDF frame, under the episode's `zero_convention`.
- Torques are in N·m.
- Camera angles are commanded servo degrees: SG90s give no feedback.
- Controller and head poses are in the robot frame of `parse_packet`.
- Times are monotonic seconds on the robot PC.
