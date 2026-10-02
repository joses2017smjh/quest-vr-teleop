# Quest VR teleop for the Berkeley Humanoid Lite arms

The stock BHL teleop needs SteamVR, a Windows PC and Vive controllers. This rig drives
the arms from a **Meta Quest 2** through a WebXR page in the Quest Browser instead.
Everything runs on one Linux PC (Intel N150) next to the robot.

```
 Quest 2 (Quest Browser, WebXR page)
   │  grip poses, buttons, mic audio              ▲ status panel, robot 3D model,
   ▼  (HTTP/WebSocket, :8080)                     │ camera / screen / sensors / voice
 quest_bridge.py ── UDP 11005 ──► run_teleop.py ──► arm_driver.py ──► can0 (left arm)
        ▲                │                                      └──► can1 (right arm)
        └── UDP 11006 ◄──┘   status back to the headset               servos.py ──► Nano (claws, camera pan/tilt)
```

Only the arms are driven: can0 = left arm (IDs 1 3 5 7 9), can1 = right arm
(IDs 2 4 6 8 12). The legs are never touched.

## Start it

```bash
./scripts/teleop/bringup.py            # CAN up, five windows, attaches you
./scripts/teleop/bringup.py status     # what's running, touches nothing
./scripts/teleop/bringup.py stop       # shut it all down
./scripts/teleop/bringup.py attach     # come back to a running session
./scripts/teleop/bringup.py --sim      # no robot, same headset flow
```

Everything runs in one tmux session, `bhl` (Ctrl-B then the number to switch, Ctrl-B d to leave):

```
0 claude    claude --continue    - dictation types in here
1 bridge    quest_bridge.py      - the headset page, on :8080
2 teleop    run_teleop.py        - the arms
3 screen    screen_share.py      - your desktop, in the headset
4 voice     voice_typer.py       - speech -> the claude window
5 camera    camera_share.py      - the robot's stereo camera, in the headset
  depth     depth_flow.py        - depth + optical flow per eye (A x2), idle until shown
```

**Start with both arms hanging straight down.** run_teleop.py takes that pose as zero.
If zero is taken with an arm bent, every angle is wrong and arming is refused.

Then, in the Quest Browser, open `http://<pc-ip>:8080`. Passthrough needs a secure
origin: whitelist it under `chrome://flags` (the bridge prints the exact steps when it
starts).

## In the headset

```
triple-click X      arm / stop the motors (stop = damping)
hold grip           that arm follows your controller (dead-man: let go = hold still)
triple-click Y      calibration menu (motors stopped first):
                      A = arm power calibration, B = claws + camera servos
A / B               answer calibration prompts
right trigger       hold to talk (dictation to Claude, or "agent, ..." commands)
right stick press   re-center the status panel
A twice             depth + optical-flow grid from the robot's cameras
```

Every time you arm (triple X), `reach_check.py` re-measures your reach: arms hanging,
then arms straight forward. That sets the motion scale (robot cm per your cm). B skips
the check and keeps the stored scale.

## How the arms are powered

`arm_power.py`: each joint's torque is split in two:

* **gravity feed-forward**: `tau_ff = gravity_scale * g(q)`, from the URDF
* **a dynamic torque limit**: `|tau_ff| + headroom`, never below a floor and never
  above the joint's calibrated `max_torque`

`arm_driver.py` sends PDO-3 `[position target, torque feed-forward]` to every joint on a
bus, then collects all the replies before one shared deadline. The stock per-joint PDO-2
with its 1 ms wait timed out over USB CAN. These values live in
`configs/arm_power_profile.json`, which the headset calibration (`arm_calibration.py`)
writes. Per joint, the calibration does:

1. a direction check: the joint rocks gently, you answer A (right way) or B (flip it)
2. a power ladder: a slow arc against gravity at rising torque limits until the
   joint tracks; that sets `max_torque`, and a fit against the URDF gravity model sets
   `gravity_scale`

## Tune it live while someone teleoperates

```bash
.venv/bin/python tools/teleop_tune.py audit --seconds 30      # what went wrong, and what to try
.venv/bin/python tools/teleop_tune.py show                    # each joint's settings, live
.venv/bin/python tools/teleop_tune.py power "left yaw" up     # ±0.5 N·m, never past its ceiling
.venv/bin/python tools/teleop_tune.py gravity "left pitch" up # arm sags / floats
.venv/bin/python tools/teleop_tune.py sensitivity down        # hands often sent out of reach
.venv/bin/python tools/teleop_tune.py flip "left yaw"         # motors stopped only
.venv/bin/python tools/teleop_tune.py undo
```

Changes reach the running run_teleop.py at once and are saved to the profile. From the
headset, by voice: "agent, more power left yaw", "agent, less sensitive", "agent, undo that".

## Voice, Claude and the headset panels

* `voice_typer.py`: hold the right trigger and speak. Vosk shows the words live, Whisper
  (base.en, offline) makes the final text, and tmux types it into the Claude Code pane.
* `speak_last_reply.py` (a Claude Code Stop hook) + `say.py`: Piper turns the start of
  Claude's reply into speech on the PC, and the headset plays it.
* `claude_sessions.py`: several Claude sessions, one panel each in the developer view.
  "agent, open claude session", "agent, close session A", "agent, saved sessions".
* "agent, help" shows every button and command in the headset.

## Vision and sensors

```bash
.venv/bin/python scripts/teleop/camera_share.py          # stereo USB camera -> /camera.mjpg (left/right eye)
.venv/bin/python scripts/teleop/screen_share.py          # PC desktop -> /screen.mjpg (Wayland portal)
.venv/bin/python scripts/teleop/person_track.py          # NanoDet + pose: where the operator is
.venv/bin/python scripts/teleop/sensor_share.py          # RPLidar C1 + WitMotion IMU -> /sensors.json
.venv/bin/python scripts/teleop/face_display.py          # robot face: eyes that follow you, moods
.venv/bin/python scripts/teleop/record_headset.py        # record the headset view via scrcpy
```

Frames are passed through `/dev/shm/bhl_*.jpg`, and `quest_bridge.py` serves them to
the page. `robot_model.py` turns the CAD STLs into a light model that the page poses
from the live joint angles.

## Claws and camera servos

An Arduino Nano (`scripts/arduino/claw_controller`) drives the servos over serial:
claws on D3 (left) and D7 (right), camera pan on D9 and tilt on D5. The protocol is
`P <pin> <us>`, `OFF`, `STATUS`. Set the positions in the headset (triple Y, then B) or
with `tools/claw_jog.py`; they are saved to `configs/claw_limits.json`.

## Arm bring-up and diagnostics (`tools/`)

All of these are arm-only and USB-port guarded: they refuse to open a bus that isn't
the expected arm adapter.

```bash
.venv/bin/python tools/scan_arm_buses.py                       # which IDs answer on can0/can1
.venv/bin/python tools/audit_arm_config.py                     # config + faults vs expected
.venv/bin/python tools/ping_can.py can2                        # ping 1..16 on any bus
.venv/bin/python tools/calibrate_arm_joint.py --bus can0 --id 1  # flux-offset calibration, one joint
.venv/bin/python tools/closed_loop_check.py --bus can0 --id 9  # does POSITION mode go where told
.venv/bin/python tools/diagnose_arm_joint.py --bus can1 --id 4 # one joint, end to end
.venv/bin/python tools/diagnose_leg.py --bus can2              # read-only leg check
```

Tests that need no robot or headset:

```bash
.venv/bin/python tools/test_teleop_tuning.py
.venv/bin/python tools/test_voice_typer.py
.venv/bin/python tools/test_claude_sessions.py
```

## Safety rules for this rig

* Never use `Humanoid()` or the stock `write_configurations.py`: on this PC they map
  can0/can1 to the **legs**.
* Never rewrite a motor's device_id or flux offset by hand.
* Restart run_teleop.py only when it is stopped (motors off) and the arms hang still at
  zero.
* Motors start STOPPED (damping) until triple X. Triple X again stops them.

See `arm_validation/TELEOP_AND_ARM_BRINGUP.md` for the as-built lab notes: USB ports,
joint signs, and calibration values.
