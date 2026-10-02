# Arm bring-up and Quest teleop — lab notes

**Robot:** Berkeley Humanoid Lite (this unit)  
**Dates:** 15–16 Sep 2026 (bring-up and first Quest teleop)  
**Scope:** **Arms only.** Legs were never written, calibrated, or put in POSITION.  
**Authoritative docs vs this unit:** official BHL teleop is SteamVR + Vive controllers. This unit used a **Quest 2 WebXR bridge** because there is no Windows SteamVR PC here.

This is an as-built notebook, not a replacement for the gitbook. IDs, USB ports, and flux offsets below are **this robot**.

---

## Current status (end of 16 Sep 2026)

| Item | Status |
| --- | --- |
| Left arm can0 IDs **1, 3, 5, 7, 9** | Online, `gear_ratio = -15`, electrically calibrated, closed-loop proven |
| Right arm can1 IDs **2, 4, 6, 8, 12** | Same. Docs/repo wrist ID **10 is absent**; hardware answers as **12** |
| Left shoulder yaw **can0 ID 5** | Encoder was the blocker; after reseat/reset it IDLEs with valid I2C and calibrated |
| Tiny closed-loop move | 1 Nm too weak on assembled pitch; **3 Nm** reached a −0.1 rad command |
| Quest 2 tracking | **Both controllers track.** Grip + trigger both fire |
| Quest → robot UDP | Packets reach `run_teleop.py` (`VR packet received`) |
| Triple-click **X** (left) | **Works.** Toggles POSITION (ARMED) ↔ DAMPING (STOPPED) |
| First armed teleop | **Joint commands moved** while ARMED and grips held (16 Sep ~15:53) |
| Passthrough (“See the room”) | **Unreliable.** Quest 2 grayscale AR; self-signed HTTPS often blocks WebXR AR. Black VR still tracks |
| Teleop torque / gains | `kp=30`, `kd=2`, `torque_limit=3` Nm |
| Loop rate | Often late vs 30 Hz; occasional CAN PDO timeouts (IDs 2, 3, 5, 7, 9) while running |
| Legs | **Not touched** |

Motors start **STOPPED (damping)** until 3× X. That is intentional.

---

## Hard constraints we kept

1. **Never** `Humanoid()` / stock `write_configurations.py`. On this machine those still map `can0`/`can1` to **legs**.
2. Arm tools USB-guard: `can0` must be USB **`3-4.1:1.0`**, `can1` **`3-4.2:1.0`**. Legs are `3-4.4` / `3-4.3`.
3. `gs_usb` has **no `restart-ms`**. After ERROR-PASSIVE / TX buffer full, bounce the interface.
4. Do not rewrite **device_id** or **flux_offset** in the config writer. Flux comes from MODE_CALIBRATION.
5. Assemble arms without pinching ID 5 AS5600 (**SDA yellow / SCL green**). Leave ~±24° free travel for cal (one motor rev at 15:1).
6. Sudo for CAN up: used on this box when interfaces were down.

---

## As-built map

### USB CAN

| Interface | USB port | Limb | IDs |
| --- | --- | --- | --- |
| can0 | 3-4.1:1.0 | Left arm | 1, 3, 5, 7, 9 |
| can1 | 3-4.2:1.0 | Right arm | 2, 4, 6, 8, **12** |
| can2 | 3-4.4:1.0 | Left leg | (out of scope) |
| can3 | 3-4.3:1.0 | Right leg | (out of scope) |

Kernel probe order **can swap names** across reboots. `scripts/hardware/start_can_by_port.sh` renames by USB port, but it **requires all four adapters**. With only arm dongles plugged in, bring can0/can1 up by hand at 1 Mbit/s, txqueuelen 1000, `restart-ms 0`.

After bouncing CAN, **odd/even mapping did not swap** on this unit.

### Joint table (teleop / Bimanual)

| Side | ID | Joint | Axis sign in `bimanual.py` |
| --- | --- | --- | --- |
| L | 1 | shoulder pitch | +1 |
| L | 3 | shoulder roll | +1 |
| L | 5 | shoulder yaw | −1 |
| L | 7 | elbow | −1 |
| L | 9 | wrist yaw | −1 |
| R | 2 | shoulder pitch | −1 |
| R | 4 | shoulder roll | +1 |
| R | 6 | shoulder yaw | −1 |
| R | 8 | elbow | +1 |
| R | 12 | wrist yaw (docs: 10) | −1 |

Grippers: `/dev/ttyUSB0` at 115200, packed `<ffb`. Open ~0.2, closed ~0.85.

### Electrical cal (flashed flux offsets)

Values from `arm_validation/cal/*` **after** MODE_CALIBRATION (second snapshot in each JSON):

| Bus | ID | `encoder_flux_offset` |
| --- | --- | --- |
| can0 | 1 | 33.93 |
| can0 | 3 | 13.54 |
| can0 | 5 | −11.34 |
| can0 | 7 | −35.20 |
| can0 | 9 | −20.96 |
| can1 | 2 | −83.62 |
| can1 | 4 | −64.66 |
| can1 | 6 | 36.73 |
| can1 | 8 | −40.77 |
| can1 | 12 | −18.83 |

All ten have `gear_ratio = -15`. Cal travel was ~4000 encoder counts (one motor rev). Firmware auto-stores flash then returns IDLE.

---

## What we learned (firmware / CAN / encoders)

Authoritative firmware in this work: Recoil B-G431B-ESC1 **0x20250226**.

- **SocketCAN 1 Mbit/s.** Host PC on Wi-Fi `10.0.0.46`.
- **PDO-1** ping `0xCA` feeds the watchdog (`htim2`). **PDO-2** position/velocity also feeds it. If the host stops sending, firmware goes **DAMPING**.
- **Modes:** DISABLED = boot/init hang; **IDLE** = healthy parked (not “broken”); DAMPING = fault or e-stop; POSITION = closed-loop; CALIBRATION = `0x05`.
- **AS5600** I2C 0x0E, 12-bit, CPR 4096. `Encoder_init` **blocks until ACK** (~100 ms retry). That looks like a dead bus if SDA/SCL is pinched.
- `Encoder_update`: unmasked I2C ≥ 4096 → **ENCODER_FAULT (0x2000)** → DAMPING. Watchdog bit is 0x0040.
- `position_measured = encoder_position / gear_ratio`. `iq_target = torque_setpoint / Kt / gear_ratio`. `torque_limit` saturates joint Nm.
- **recoil.Bus.receive()** with `timeout=None` can spin forever on error frames. Timed SDO (`~0.08 s`) is required. Flash writer hung on can1 ERROR-PASSIVE for this reason.
- **Bimanual.start()** writes Kp/Kd/torque_limit only, **not** `gear_ratio`. Teleop without a prior flash of −15 would run at gear 1.
- Official `run_teleop.py` listens UDP **11005** for `{left,right: {pose 4×4, button_pressed, trigger}}`. SteamVR-Bridge on Windows sends that. Grip = dead-man; trigger = gripper.
- IK is **not** per-joint mapping. Pink/Pinocchio tracks `arm_left_elbow_roll` and `arm_right_elbow_roll`; the other eight arm joints are filled by the QP. Base is pinned at `[0,0,0.5]`.
- Quest 2 controllers are tracked by **headset cameras**. Visor off → tracking dies. Vive lighthouses are a different system.
- Quest Browser WebXR needs a **secure origin**. Self-signed `https://10.0.0.46:8443` often yields `SSLV3_ALERT_CERTIFICATE_UNKNOWN`. Accepting the warning is not always enough for **immersive-ar**. `http://localhost:8080` via `adb reverse` is the reliable WebXR origin.
- `cc.udp.recv_dict` default **bufsize=1024** is tight for two 4×4 JSON poses. Teleop now uses **8192**.
- Printing every IK cycle made the 30 Hz limiter late and contributed to PDO timeouts.

### ID 5 encoder (left shoulder yaw)

This was the hardware gate for “encoders healthy.”

- ST-LINK UART on the **wrong ESC** (ID 7) produced 0 bytes.
- ID 5: `Encoder_init` hang, then IDLE with garbage `i2c=51946` (ENCODER_FAULT).
- IDLE is the **healthy** parked mode; it does not mean “enable it.” Faulted nodes sit in DAMPING or never leave DISABLED.
- Fix was mechanical: reseat AS5600 / cable, reset the ESC, keep the magnet/I2C un-pinched when assembling the arm.
- After that: raw I2C matched, later TRACKS, then calibrated.

Open-loop 1 V VALPHABETA sweeps on 5/7/9 looking ERRATIC was **load/endstops**, not dead magnets. Hand-rotate tests were skipped on request.

---

## Path that got teleop working

Order actually used (do not skip gear/cal):

1. **Inventory / scan** IDs 1–20 on can0, as-built 10 IDs on both buses. Confirm no ID 10 on can1, ID 12 present.
2. **USB-guarded encoder audit** (`validate_arm_encoders.py`, `encoder_rotation_test.py`). Powered sweeps with `--no-current-volts` after a 0.3 V abort on ID 7.
3. **Hardware fix ID 5**, re-scan, re-audit.
4. **Assemble arms** once encoders were healthy.
5. **RAM then flash** `gear_ratio=-15` on all ten via `tools/write_arm_config.py` (can0 first; can1 after bounce when flash hung). Never flashed device_id or flux_offset.
6. **Electrical cal** one joint at a time: `tools/calibrate_arm_joint.py`. All 10 OK, auto flash store.
7. **Patch teleop wrist ID 10 → 12** in `bimanual.py`.
8. **Tiny POSITION nudge** `tools/tiny_move_arm_joint.py`: Kp=0.2 / 1 Nm ≈ no motion; Kp=50 torque=3, delta −0.1 → −0.088 rad, iq ~0.45 A.
9. **Raise teleop torque to 3 Nm**, start `run_teleop.py`.
10. **Quest 2 is not Vive.** Added WebXR bridge → same UDP 11005 packet.
11. **Triple-click X** arm/stop because there was no physical e-stop in the Quest path.

Evidence dirs:

- `arm_validation/20260915_212915/` — first full audit (stationary, magnet track, rotate)
- `arm_validation/20260916_encoder_rerun/` and `..._rerun2/` — encoder streams after distrusting earlier tests
- `arm_validation/id5_fix/` — scans, UART, live ID 5 CSV
- `arm_validation/config_write_live/` — dry-run / ram / flash
- `arm_validation/cal/` — per-joint cal JSON
- `arm_validation/cal/` + `motor_configuration.json` — `gear_ratio -15` on the box

---

## Tests (what we actually ran)

| Test | Result |
| --- | --- |
| Topology scan can0 1–20 | As-built odds 1,3,5,7,9. No stray 10. TX-full / ERROR-WARNING → bounce can0 |
| Stationary encoder | All 10 online after ID 5 fix; raw == i2c < 4096 when healthy |
| Powered encoder sweep | TRACKS on later runs; early ERRATIC on 5/7/9 attributed to load |
| Config RAM + flash | All 10 `gear_ratio=-15`. can1 flash needed interface bounce; ID 8 had dropped to gear 1.0 until re-flash |
| Electrical cal × 10 | All completed, ~1 motor rev, IDLE + new flux, flash stored |
| Tiny move +0.1 @ 1 Nm | No useful motion (gravity / 1 Nm cap) |
| Tiny move −0.1 @ 3 Nm, Kp=50 | Success (−0.088 vs −0.10), abort iq>4 A not hit |
| First `run_teleop` (no Quest) | Motors enabled; triggers (0,0); no Vive packets; arms held |
| Quest WebXR | Poses streamed; both controllers; cert warnings on every new TLS |
| Teleop without 3× X | Grips ON, **arms stayed damping** (by design) |
| Teleop after 3× X | `Motors ARMED (position)`; commanded joints changed (e.g. left pitch through ~−1.3 rad in the log) |
| 3× X again | `Motors STOPPED (damping)` |

---

## Code changes (this bring-up)

### New arm-only tools (`tools/`)

| File | Role |
| --- | --- |
| `arm_common.py` | USB guard, as-built ID map (wrist 12), firmware error bits (recoil.ErrorCode was stale), timed SDO, ping |
| `scan_arm_buses.py` | Scan without touching legs |
| `audit_arm_config.py` | Read-back config vs backup.json |
| `validate_arm_encoders.py` | Live encoder / I2C / mode |
| `write_arm_config.py` | RAM/flash from backup.json; skip device_id + flux_offset; `--dry-run` / `--ram` / `--flash` |
| `calibrate_arm_joint.py` | NMT CALIBRATION; refuse unless IDLE, Vbus>15, raw==i2c<4096, \|gear\|≥10 |
| `tiny_move_arm_joint.py` | POSITION hold / nudge / return; restore RAM kp=50 kd=2 torque=1; abort iq>4 A |

### Hardware scripts

| File | Role |
| --- | --- |
| `scripts/hardware/start_can_by_port.sh` | Port-stable can0–can3 names (needs 4 dongles) |
| `scripts/hardware/encoder_rotation_test.py` | `--no-current-volts` (default 0.3 was hair-trigger) |
| plus audit/scan/watch helpers | Used during distrust-and-rerun |

### Teleop / robot

| File | Change |
| --- | --- |
| `bimanual.py` | Right wrist **10 → 12**; default `torque_limit=3`; `set_armed()` POSITION ↔ DAMPING |
| `scripts/teleop/run_teleop.py` | torque 3 Nm; UDP bufsize 8192; start **disarmed**; apply IK only when armed; 1 Hz prints; VR recv try/except |
| `scripts/teleop/quest_bridge.py` | HTTPS 8443 + HTTP 8080; WebXR JSON → UDP 11005; WebXR→BHL frame (`X` fwd, `Y` left, `Z` up); `armed` flag |
| `scripts/teleop/quest_bridge.html` | Quest page: AR passthrough attempt, both controllers, grip dead-man, trigger gripper, **3× X** toggle |
| `.gitignore` | `scripts/teleop/.certs/` (generated TLS) |

### Not done (on purpose)

- No flash of stock `write_configurations.py`
- No leg bus in any tool
- No SteamVR-Bridge (needs Windows + Vive)
- No change to locomotion / Humanoid can0-can1 mapping

---

## How teleop works on this unit

```
Quest 2 Browser  --WSS-->  quest_bridge.py  --UDP JSON 11005-->  run_teleop.py
                                                         |
                                                    Pink IK (elbow frames)
                                                         |
                                              Bimanual PDO-2 POSITION
                                                         |
                                        can0: 1,3,5,7,9    can1: 2,4,6,8,12
                                        gripper: /dev/ttyUSB0
```

**Inputs**

- Hold **grip** on a controller → that elbow target tracks Δ pose (dead-man).
- **Trigger** → gripper close.
- **Triple-click X** (left, xr-standard button 4, three clicks in ~1.2 s) → ARM / STOP.

**Start-up**

1. Bring can0/can1 UP 1 Mbit if down.
2. `.venv/bin/python scripts/teleop/quest_bridge.py`
3. `.venv/bin/python scripts/teleop/run_teleop.py`  
   Meshcat: `http://127.0.0.1:7000/static/`
4. Quest Browser: `https://10.0.0.46:8443` (Advanced → Proceed) **or** USB `adb reverse tcp:8080 tcp:8080` then `http://localhost:8080`.
5. **See the room** (AR) or **VR (no camera)** if AR fails.
6. Face the robot, 3× **X**, then squeeze grip.

Ctrl+C in teleop: damping, then second Ctrl+C → IDLE (stock `Bimanual.stop()`).

---

## Known issues / next

1. **Passthrough** on Quest 2 is grayscale and often fails on the LAN self-signed cert. USB localhost is the fix to try first. Do not treat black VR as “tracking is dead” — poses still stream.
2. **3× X is easy to miss.** Without it, grips do nothing to the motors. Status line should read `motors ARMED`.
3. **Loop is slow** (Pink + Meshcat + 10× PDO). Rate-limiter “late 20–40 ms” and sporadic `No response from device N`. Needs a faster loop or less viz, not more print.
4. **3 Nm** is enough for a small pitch nudge; full gravity-compensated reach may need more, with current/iq watch.
5. **Gripper `index 10` slightly negative** vs `[0, π/2]` — warning only.
6. `run_idle.py` still calls `robot.run()` which **does not exist** on `Bimanual` (only `start`).
7. Official path remains **Vive + SteamVR-Bridge** if a Windows box appears. Quest bridge is a local substitute that speaks the same UDP dict.
8. `start_can_by_port.sh` cannot run with only two arm adapters; document a two-dongle variant or plug all four.
9. After power cycle: confirm USB names, `gear_ratio=-15`, flux still flashed, then teleop. Config writer does not need to run again unless flash was lost (as happened to ID 8 mid-bring-up).

---

## Quick run (this PC)

```bash
# if CAN down (password as used on this box):
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can1 up type can bitrate 1000000
sudo ip link set can0 txqueuelen 1000
sudo ip link set can1 txqueuelen 1000

cd /home/joses/Berkeley-Humanoid-Lite
.venv/bin/python scripts/teleop/quest_bridge.py          # https://10.0.0.46:8443
.venv/bin/python scripts/teleop/run_teleop.py            # starts STOPPED
```

Quest: proceed past cert → enter tracking → **3× X** → grip.

---

## 17 Sep 2026 — per-joint power, headset calibration, passthrough

Reported: arm cannot go all the way up, some joints do not respond, headset camera view (passthrough) does not work.

### Causes found (offline, no CAN traffic)

| Cause | Effect | Fix |
| --- | --- | --- |
| Stock IK hand targets start at the **world origin** and only update when a grip is *released* | Right after 3× X with no grip, both shoulders are driven to the backward pitch limit (±0.79 rad) and inward roll limit within ~1 s. The first grip then moves relative to that unreachable point, so joints sit pinned at limits | Anchor to the real hand pose when a grip is pressed; a released arm holds still |
| IK `dt` never reset (grows with time since arming) | URDF velocity limit (15 rad/s) effectively off; jumps limited only by torque | Real cycle dt, 1.5 rad/s cap, command ≤ 0.35 rad ahead of measured |
| One torque limit for every joint (3 Nm), no gravity compensation | URDF gravity alone needs ~2.9 Nm at shoulder pitch (arm horizontal) and roll (75° out); claws and cables add more, so the arm stalls below horizontal | Gravity feed-forward in PDO-3 + per-joint dynamic limit (below) |
| `Bimanual` waits 1 ms for each PDO-2 reply | `No response from device N` | Batched PDO-3 per bus, 6 ms shared window, late replies used next cycle |
| URDF shoulder pitch allows 90° forward at most | Overhead is outside the model | Unchanged (URDF/mechanical range) |
| Passthrough needs a secure origin; self-signed HTTPS is often refused for `immersive-ar` | Black VR only | Plain HTTP + Quest Browser flag (below); the page now shows the real refusal error |

### Power model (`scripts/teleop/arm_power.py`)

- Feed-forward `tau_ff = gravity_scale × g(q)` (URDF, upright), sent every cycle as the PDO-3 torque target. Firmware adds it to `kp·e − kd·v`.
- Torque limit `clamp(|tau_ff| + headroom, floor, max_torque)`, written by SDO only when it changes (raise fast, lower lazily).
- Uncalibrated defaults: pitch/roll floor 2, headroom 1.5, max 6 Nm; yaw/elbow 1.5 / 1.0 / 4; wrist 1.0 / 0.8 / 2.5. kp 30, kd 2. Hard ceiling 10 Nm.
- Profile file: `configs/arm_power_profile.json` (previous copy kept as `.bak`). Loaded at start-up; printed per joint.

### Headset power calibration (triple-click Y, motors stopped)

1. **Zero pose** (motors off): arms hanging straight down, A = set zero.
2. **Your reach** (optional): arms down, A; arms straight forward, A. Sets the hand-motion scale (robot reach 0.30 m ÷ yours).
3. **Motors on**: hold a grip + A.
4. **Per joint**, left arm then right: pitch, roll, elbow, yaw and wrist (yaw and wrist with the elbow bent 1.0 rad):
   - Direction check: the joint rocks gently. A = moves as described, B = flip its sign. If it is blocked one way but free the other (a flipped joint pushes into its own end stop), it is flipped automatically and you confirm.
   - Power ladder: arc at 0.35 rad/s to pitch ∓69°, roll ±57°, elbow ±69°, yaw/wrist ±29°, hold 1 s, return. Starts at 1.15 × model need + 0.5 Nm, then +0.5 Nm per try up to the cap. Lag with torque to spare re-fits gravity and retries the same level. Moving, then stopping at the same spot at two levels = **BLOCKED** (no further escalation). A joint that has not moved at all keeps climbing, since static friction can let go higher.
   - Result: `max_torque = max(passing level, 1.2 × peak + 0.5)`, `gravity_scale` from moving samples, headroom from friction. Saved after every joint.
   - Verdicts: `OK`, `TOO WEAK at cap`, `NOT MOVING (mode/errors read from the ESC)`, `BLOCKED`, `DIR!`, `SKIP`.
- Grip is the dead-man for every move (release = hold still). 3× X stops everything. A joint silent on CAN for 10 cycles stops the calibration.
- If a joint reports TOO WEAK: `run_teleop.py --cal-power-scale 1.3` raises the caps (never above 10 Nm).

### Headset panel and passthrough

- The page draws a status panel inside VR/AR: state, prompts, per-joint torque bar (tick = current limit) and tags. `MAXED` = torque at its limit > 0.3 s (needs power or is blocked). `LAG` = not following (mode/errors read and shown). `NO CAN` = no replies; > 0.5 s stops all motors.
- Passthrough, recommended: Quest Browser → `chrome://flags` → *Insecure origins treated as secure* → add `http://10.0.0.46:8080` → Enabled → Relaunch → open `http://10.0.0.46:8080`. USB `adb reverse` also works, but `adb` is not installed on this PC.
- Symptom `SSL Error ... SSLV3_ALERT_CERTIFICATE_UNKNOWN` from `10.0.0.137` (17 Sep): a headset tab is still on `https://10.0.0.46:8443` and refuses the self-signed certificate on every WebSocket retry. Relaunching the browser (for the flag) forgets accepted certificates and may restore that tab. Close it and type `http://10.0.0.46:8080`. The bridge now prints one hint every 30 s instead, and the HTTPS copy of the page links to the HTTP one. `--http-only` turns 8443 off.
- Black screen in the headset (17 Sep, after connecting). The page painted the background opaque unless the browser *reported* `environmentBlendMode` as alpha-blend. Fixed: an `immersive-ar` session always clears transparent (as the original page did). A refused AR request now retries once without options.
- Headset diagnostics: after the bridge's `hello`, the page reports secure context, `immersive-ar` support, the session mode/blend, and any refusal (`NAME: message`). The bridge prints these as `[headset <ip>] ...` and never forwards them to teleop. Pose packets carry a page id, and teleop re-baselines button counters when the page changes, so a reload or second tab cannot read as a press.
- **17 Sep, resolved:** the black view is the headset, not the page. The WebXR page renders (its status panel is visible in the session) and the browser reports `immersive-ar` as supported, but the Quest shows no cameras even for its own double-tap passthrough shortcut. No web page can show passthrough while the system will not. Check, in the headset: Settings > Physical Space (passthrough / double-tap shortcut enabled, boundary set up; a disabled Guardian also disables passthrough), nothing covering the front cameras, and enough light in the room (Quest 2 passthrough is greyscale and goes black in a dim room).
- Controller poses the headset only estimates (out of camera view) never move an arm. Hand tracking is ignored.
- Button presses travel as counters, so a lost packet cannot lose or repeat a press.

### Tests (all offline; nothing ran on the real arms)

| Test | Result |
| --- | --- |
| Stock IK repro: armed, no grip | Shoulders to ±0.79 pitch, ∓0.26 roll (bug confirmed) |
| Sim teleop, payload 1.25×: arm, grip, raise hand 25 cm | L pitch −1.25 rad at 3.7 Nm (limit 4.2). Release: no drift. Link loss: STOPPED after 5 s |
| Sim calibration, payload 1.0 / 1.25 / 1.6 / 2.3 | Pitch gravity ×1.01 / ×1.27 / ×1.64. 2.3× → TOO WEAK at 6 Nm (needs more than 6) |
| Faults: weak yaw, stuck wrist, sticky wrist (1.9 Nm stiction), flipped elbow, flipped roll, wrong operator flip, silent joint | NOT MOVING / NOT MOVING at 2.5 / OK after climbing to 2.0 / auto-flip / BLOCKED / BLOCKED / refuses to arm |
| Joint goes silent while armed | `NO CAN` at 0.3 s; all motors STOPPED at 0.5 s |
| `ArmDriver` vs fake firmware on python-can virtual bus, with USB hiccups | Frame IDs, SDO offsets, signs, zero mapping correct. 71/2000 late replies used; 0 missed cycles |
| Bridge + sim teleop end to end (scripted WebSocket headset) | Counters, arm/stop, grip motion, estimated pose ignored, calibration menu, 5 Hz status |
| Page JS | Parses; panel placement math matches numpy; layout checked in headless Firefox |
| Real preflight with can0/can1 DOWN | Exits with "bring it up first"; no frames sent |

### First real preflight (17 Sep)

`run_teleop.py` checked all ten joints (read-only): nine were IDLE with no errors, gear −15, 22.8–23.2 V. It refused to start on **L wrist (can0 ID 9)**: `DAMPING`, `err=0x2040` (WATCHDOG_TIMEOUT + ENCODER_FAULT).

- Live read (read-only): position raw stuck at 4095, `ENCODER_I2C_BUFFER` = **20237** (valid 0–4095). Neighbour ID 7 was fine: raw = i2c = 1744, p2p 1 count.
- Firmware `encoder.c`: `Encoder_update` checks the buffer **before** it issues the next I2C read, so one corrupt frame latches the fault forever. The buffer never refreshes and the joint stays in damping. Only a reboot (`Encoder_init`) clears it. Same pattern as ID 5 on 16 Sep (`i2c=51946`).
- Fix: power-cycle the robot, bounce can0/can1, restart teleop. If ID 9 then shows `mode=DISABLED`, its AS5600 cable (SDA yellow / SCL green) is not answering: reseat it, as for ID 5.
- Added after this: `run_teleop.py` reads one joint's mode and error bits every 0.2 s. A joint leaving POSITION stops all motors. `ENCODER_FAULT` marks the joint `FAULT` and refuses arming until a power cycle and restart. The start-up refusal now prints the bad encoder value and this explanation. Sim fault `encfault` reproduces it: stopped 0.6 s after the trip.

### First calibration run on the robot (17 Sep, 10:43)

Saved profile: 3 joints passed, 5 reported NOT MOVING, 1 skipped.

| joint | result | evidence |
| --- | --- | --- |
| L pitch | OK, peak 3.8 Nm, max 5.25 | gravity fit x0.99 (r2 0.90), friction 0.64 Nm - the URDF model matches this arm |
| L roll | OK, peak 3.1 Nm, max 4.50 | **operator flipped its direction**, then gravity x1.21 |
| R yaw | OK, peak 0.7 Nm | |
| L yaw, L elbow, R pitch, R roll, R elbow | NOT MOVING | progress **0.0** at every level while the measured torque tracked the limit up to 4-6 Nm; mode POSITION, no error bits |
| L wrist | direction unclear | operator answered "no" twice |

A joint that does not move at all at 6 Nm is not weak, it is against a stop. L roll needed a direction flip, so the axis signs inherited from `bimanual.py` do not all match this build, and a flipped joint drives into its own end stop. The direction check only catches that if the swing is allowed to finish; answering A immediately skipped the measurement and the ladder then pushed into the stop (that is what the 4-6 Nm peaks are).

Changes made after this run:

- The direction check ignores A/B until the joint has swung for one full period, then judges by the measured excursion, not by the answer.
- The ladder stops climbing after two levels with zero movement and probes the other direction instead. Free the other way -> `BLOCKED that way ... probably flipped` and an offer to flip and retest. Blocked both ways -> `NOT MOVING either way at N Nm` with the ESC's mode/errors. No more 6 Nm pushes into an end stop.
- `STALL_ERR` 0.30 -> 0.22 rad, so a blocked joint pushes for less time (checks that must tolerate lag pass their own allowance).
- A direction check that aborts three times skips the joint instead of looping back to its menu.
- Default floors raised for this robot's friction (pitch/roll 3.0, yaw/elbow 2.0, wrist 1.2 Nm): `tiny_move` on 16 Sep needed ~3 Nm to move an assembled shoulder at all.
- Right thumbstick on a joint screen tunes that joint: up/down = test power (0.5 Nm per click), left/right = test range (5 deg per click). Saved as `cal_cap` / `cal_arc`.
- Calibration writes `arm_validation/calibration_<date>.md` (a readable table) as well as the profile, and the path is printed and shown in the headset.
- `B` while stopped swaps which controller drives which arm (`swap_hands` in the profile, or `--swap-hands`). The operator faces the robot, so straight mapping feels mirrored.
- Teleop soft limits come from `range_lower` / `range_upper` in the profile when set, clamped to the URDF range plus 0.35 rad.

### Seeing the PC from inside the headset (18 Sep)

Two panels now float in the session, both movable:

- **Status panel** (as before): state, prompts, per-joint torque bars and tags.
- **PC view panel**, above it: your shared screen if `screen_share.py` is running, otherwise the
  last 14 lines of `run_teleop`'s own terminal output (it tees stdout/stderr into the status message).

Controls: left stick moves both panels (up/down height, left/right distance, saved in the browser),
left stick press leaves the session, right stick press re-centres. Each panel is tilted from its
height relative to the head, so both face the operator.

Screen sharing on this PC (GNOME **Wayland**) cannot use the usual grabs: `ffmpeg -f x11grab` returns a
black frame (stddev 1.2) and `org.gnome.Shell.Screenshot` answers `AccessDenied`. The way in is the
desktop portal, which requires one click per session:

```bash
.venv/bin/python scripts/teleop/screen_share.py     # dialog: pick a screen or window -> Share
.venv/bin/python scripts/teleop/screen_share.py --test   # colour bars, no dialog, for plumbing checks
```

It asks the portal for a ScreenCast, pulls the PipeWire node through GStreamer
(`pipewiresrc ! videoconvert ! videoscale ! jpegenc`) and writes `/dev/shm/bhl_screen.jpg` atomically at
10 fps. `quest_bridge.py` streams that file at `/screen.mjpg` as multipart JPEG (503 while nothing is
sharing, and it ends the stream if frames stop for 5 s, so the page falls back to the log panel).
PyGObject and dbus are system packages, so the script adds `/usr/lib/python3/dist-packages` to its path
when run from the venv. Sharing the single terminal window rather than the whole screen keeps the text
readable in a Quest 2.

Measured: 44 frames in 4 s (~11 fps) at 1280 px wide, ~23 kB per frame, about 250 kB/s.

### Dictation from the headset into a terminal (18 Sep)

Hold the **right trigger** in the headset and speak; the words are typed into a tmux pane (this is how
you talk to a Claude Code session while wearing the headset, hands on the controllers).

- The page captures the headset microphone (`getUserMedia`, 16 kHz mono, push-to-talk on the right
  trigger) and sends raw PCM as binary WebSocket frames. Enable the microphone once on the flat page,
  before entering the session: permission prompts cannot appear inside XR.
- `quest_bridge.py` forwards those frames (prefixed `AUD0`) to UDP 11007, along with the
  `{"type":"voice","action":"start"|"stop"}` markers.
- `scripts/teleop/voice_typer.py` runs Vosk offline (no cloud) and types the result with
  `tmux send-keys`, then Enter. Live partial text and the final result are sent back through the status
  relay, so the headset panel shows what was heard.

Why tmux: the kernel refuses direct terminal injection (`dev.tty.legacy_tiocsti = 0`) and Wayland
refuses synthetic keystrokes (no xdotool; ydotool would need root). tmux needs neither.

```bash
sudo apt install tmux
tmux new -s claude        # inside it:  claude --continue
.venv/bin/python scripts/teleop/voice_typer.py --target claude
```

Model: `vosk-model-small-en-us-0.15` (68 MB) in `~/.cache/vosk`. Tested end to end by feeding a speech
WAV through the same UDP path: 14 partial updates then the correct final sentence. Good for short
sentences; for long technical dictation use `vosk-model-en-us-0.22` (1.8 GB) with `--model`. No
punctuation, all lowercase. Saying just "cancel" (or "scratch that") throws the sentence away.

Safety: dictation never touches the robot. Arming stays triple-X plus the grip.

### The page could not reach the bridge (18 Sep)

Symptoms: the headset page loaded from a history link and even showed the shared screen, but said the
bridge was not connected, and dictation did nothing. Typing the address gave an endless load.

Causes, in the order they bit:

1. **`ws://` is blocked on this page.** The `chrome://flags` allowlist that WebXR needs makes
   `http://10.0.0.46:8080` a *secure context*, and a secure page may not open an insecure WebSocket.
   Plain HTTP from the same origin (the page, `/screen.mjpg`) still works, which is why the screen
   appeared while status and audio did not. The page now tries the socket for 2.5 s and then falls back
   to **server-sent events (`/events`) plus POSTs (`/input`)**, which are plain HTTP and allowed. Both
   transports carry the same messages; the page says which one it is using.
2. **Leaked MJPEG streams.** The page never released `/screen.mjpg`, and a browser allows only ~6
   connections per host, so old streams from earlier tabs blocked new page loads. The page now drops the
   stream when the session ends or the tab hides, and the bridge keeps one viewer and ends each response
   after 120 s.
3. **The HTTPS port was a trap.** The headset kept reaching `:8443` (autocomplete from an old visit),
   whose self-signed certificate is refused, so the page hung. HTTPS is now **off unless `--https`**.
4. **Squashed screen share.** `videoscale` keeps the source height when only the width is pinned, so a
   1920x1080 screen arrived as 1280x1080. The height is now computed from the portal's reported source
   size, and the headset panel takes its shape from the picture.

Verified with a client that never opens a WebSocket: hello, 33 status updates, triple-X armed the
motors, grip + hand motion moved the arm, and voice audio was accepted, all over plain HTTP.

### Files (17 Sep)

| File | Change |
| --- | --- |
| `scripts/teleop/arm_power.py` | New: joint table, gravity model, power profile |
| `scripts/teleop/arm_driver.py` | New: batched PDO-3 driver (USB guard from `tools/arm_common.py`) + `SimArmDriver` |
| `scripts/teleop/arm_calibration.py` | New: headset calibration |
| `scripts/teleop/run_teleop.py` | Rewritten around `ArmSupervisor`; no longer uses `Bimanual`. `--sim` rehearses without the robot |
| `scripts/teleop/quest_bridge.py` | Status relay (UDP 11006 → WebSocket), counters, tracked flags, no-store, flags instructions |
| `scripts/teleop/quest_bridge.html` | In-headset panel, A/B/Y controls, passthrough diagnostics |

**Claws:** the new teleop does not drive them. `/dev/ttyUSB0` is now the CH340 Nano, whose sketch speaks `P 3|5 <µs>` text, while `Bimanual` wrote binary `<ffb`. Closed-claw pulse widths were never established.

### Quick run (17 Sep)

```bash
sudo ip link set can0 up type can bitrate 1000000 && sudo ip link set can0 txqueuelen 1000
sudo ip link set can1 up type can bitrate 1000000 && sudo ip link set can1 txqueuelen 1000

cd /home/joses/Berkeley-Humanoid-Lite
.venv/bin/python scripts/teleop/quest_bridge.py
.venv/bin/python scripts/teleop/run_teleop.py        # arms hanging straight down at start
# rehearse without the robot:  run_teleop.py --sim
```

Headset: `http://10.0.0.46:8080` (after the flags step) → **Start with passthrough** → **3× Y** → follow the panel. Afterwards: **3× X** → hold a grip → move.

## 18 Sep 2026 — the black panels: one undeclared variable

Reported from the headset: *"its not recognizing run teleop, voice does not work, and the screen and
floating one in AR mode are black and flashing."* Three symptoms, one cause.

### `status` is not a free name in a browser

`quest_bridge.html` kept the robot's state in a variable it never declared:

```js
status = data;          // meant: a global holding the last status message
```

An undeclared `status` is **`window.status`**, a legacy DOMString property left over from the days of
scripting the browser's status bar. It does not hold an object: the assignment stringifies it. So after
the first status update from `run_teleop`:

* `status` was the eight-character string `"[object Object]"` — truthy, so every freshness check passed;
* `status.joints`, `status.lines`, `status.log` were all `undefined`;
* `renderDom()` threw `cannot read property 'map' of undefined` inside the message handler → the flat
  page never showed the robot (**"not recognizing run teleop"**);
* `drawHud()` threw `cannot read property 'slice' of undefined` inside the frame loop, *after*
  `gl.clear()` and *before* any panel was drawn → **both panels black, every frame, flickering** as the
  frames that ran before the first status update still painted;
* push-to-talk rode the same broken link, so **dictation never started**.

Reproduced exactly, then fixed and re-checked, with `tools/test_quest_page.py` (below):

```
OLD PAGE, status message -> TypeError: cannot read property 'map' of undefined
  window.status is now: [object Object]
OLD PAGE, frame loop    -> TypeError: cannot read property 'slice' of undefined
  panels drawn this frame: 0   <- black, every frame
```

The variable is now `robotStatus`, declared with `let`. `statusAt` and `domAt` were implicit globals
too (harmless — they are not window properties) and are declared as well. Other names with the same
trap, none of which the page uses as variables: `name`, `length`, `top`, `self`, `parent`, `origin`,
`closed`, `history`, `screen`, `event`.

### The page can no longer go dark silently

* `handleServerMessage` coerces `lines`, `joints` and `log` to arrays before storing a status, so a
  short or malformed payload cannot blank the panels.
* `drawHud` and `drawLogPanel` are wrapped: if either throws, the panel is painted red with the error
  text and the stack, **in the headset**, instead of disappearing. The frame still draws.
* A failure is also reported to the bridge, which prints it on the PC.

### Seeing what the page is doing: `/diag` and `/state.json`

The page could only report problems over the link it was failing to establish. It now also beacons
through an `<img>` GET to **`/diag?event=…`**, which needs no WebSocket, no `EventSource`, no `fetch`,
and is exempt from the secure-context rule that blocks `ws://` here — it works precisely when nothing
else does. It beacons on load, on every transport failure, on draw errors, and as a heartbeat every 5 s
(printed only when it changes, plus once a minute):

```
[headset 10.0.0.137] connected over http (status 120 ms ago), in the headset view, screen on [ws closed, sse open]
[headset 10.0.0.137] the WebSocket did not work (the WebSocket never opened); trying server-sent events instead
[headset 10.0.0.137] the right trigger was pulled but the microphone is off: leave the headset view …
```

**`GET /state.json`** answers the same questions from the PC, without the headset:

```bash
curl -s localhost:8080/state.json | python3 -m json.tool
# pages_websocket, pages_http, pose_packets, last_pose_age, last_status_age,
# screen_fresh, audio_datagrams, last_status (the exact payload), beacons[]
```

`pages_websocket: 0, pages_http: 0` with a fresh `last_status` means the robot is fine and the headset
is not attached — which is exactly what an idle, backgrounded Quest tab looks like.

### Dictation needs the microphone granted outside the headset view

`getUserMedia` raises a permission prompt that **cannot be answered from inside an immersive session**,
and the old page returned silently when the trigger was pulled without it. Now:

* pulling the trigger with no microphone says so on the panel, naming the fix (press the left stick to
  leave, tap *Enable voice dictation*, allow it, come back);
* once allowed for this origin, the page takes the microphone back automatically on later loads
  (`navigator.permissions.query({name: "microphone"})`), so the tap is a one-time step.

### `tools/test_quest_page.py` — run the page without a headset

Both page bugs that cost a day (`hud` after a rename, then `status`) were invisible to a syntax check
and would have taken one second to catch by *running* the code. This runs the page's `<script>` in
QuickJS against stubbed browser objects — including `status` defined as the DOMString property it
really is — and drives the paths the headset drives:

```bash
python3 -m venv /tmp/jsenv && /tmp/jsenv/bin/pip install quickjs
/tmp/jsenv/bin/python tools/test_quest_page.py --live    # --live also feeds it the robot's real status
```

It fails if the page does not load, if the frame loop throws (five states: no robot, screen share live,
a full status, a status with fields missing, a status with wrong types), if fewer than four panels are
drawn, if the transport does not fall back to server-sent events, if poses are not POSTed, or if a
status arriving over SSE is ignored. All pass, including the payload the robot is broadcasting now.

### 18 Sep 2026 (later) — "tau limit error" was the heartbeat, and dictation was typing into itself

**The upper panel was not showing an error.** `run_teleop` prints a heartbeat once a second:

```
STOPPED   48 Hz  link ok  tau/lim[L..R] pi-0.0/2.0 ro-0.1/2.0 ya-0.1/1.5 el+0.0/1.5 wr+0.1/1.0 pi-0.0/2.0 ...
```

stdout is teed into the fourteen lines the headset shows, so all fourteen were that same line, cut off
mid-word at the panel edge — a wall of `tau/lim` text. The status behind it was clean: `alert: ''`, no
joint tags, `link ok`. Two changes: the heartbeat now goes to the PC terminal only (`local_only()` in
`run_teleop.py`, since the status panel already shows state, rate and every torque against its limit),
and `LogTee` folds a repeated line into `... x5` instead of printing it fourteen times. The panel now
shows preflight results, calibration verdicts and alerts — the things worth reading blind.

**Dictation was typing into itself.** `voice_typer.py --target claude` sends to the *session* named
`claude`, whose active pane was `voice_typer` (tty pts/5), while Claude Code ran on pts/0 outside tmux
entirely. Every recognised sentence was typed into voice_typer's own stdin. Nothing said so, because
`tmux send-keys` succeeds in that case.

`Typist` now resolves a real pane before every utterance, and:

* `--target auto` (the new default) finds the pane whose command is `claude`;
* it **refuses to type into its own pane**, naming the pane and the tty;
* an unknown target is caught — `tmux display-message -t nosuch` prints an empty line and **exits 0**,
  so `send-keys` would have gone to whichever pane happened to be current;
* it lists the panes tmux can see, with the four steps to fix the layout;
* every finished sentence is appended to `/dev/shm/bhl_voice.txt`, so words are never lost when they
  cannot be delivered.

Verified against a real tmux session: auto found the Claude pane and the text arrived in it; pointed at
its own pane it refused; the transcript was written.

**The layout dictation needs:**

```
terminal A (tmux)         tmux new -s claude   ->   claude --continue
terminal B (anything)     .venv/bin/python scripts/teleop/voice_typer.py
```

Claude must be *inside* tmux: tmux can only type into a pane it owns, and the kernel
(`dev.tty.legacy_tiocsti = 0`) and Wayland both refuse the alternatives.

### 19 Sep 2026 — one command for the whole rig

Five programs, in order, in five terminals, and dictation only works if Claude Code is itself inside
tmux. `scripts/teleop/bringup.py` does all of it:

```bash
./scripts/teleop/bringup.py            # CAN up, five windows, attach
./scripts/teleop/bringup.py status     # what is running, without touching anything
./scripts/teleop/bringup.py stop       # shut it all down
./scripts/teleop/bringup.py attach     # back to a running session
```

| window | program | why it is separate |
| --- | --- | --- |
| 0 `claude` | `claude --continue` | dictation types in here; it must be a tmux pane |
| 1 `bridge` | `quest_bridge.py` | the headset page on :8080 |
| 2 `teleop` | `run_teleop.py` | the arms; its heartbeat belongs on the PC, not the headset |
| 3 `screen` | `screen_share.py` | needs one **Share** click per session |
| 4 `voice` | `voice_typer.py` | started last, once a `claude` pane exists |

What it takes care of:

* **CAN first.** A reboot always leaves `can0`/`can1` down; it brings both up at 1 Mbit with
  `txqueuelen 1000` (sudo asks for a password) and stops if they do not come up. `--sim` skips it.
* **Never two Claudes.** If Claude Code is already running outside tmux it would fight the new one for
  the same conversation, so window 0 explains that instead of starting a second.
* **Never two of anything.** A program already running in another terminal keeps it; its window says
  which pid owns it.
* **Dictation ordering.** `voice_typer` starts only after a pane whose command is `claude` appears
  (12 s grace); otherwise its window says what to do.
* **Crashes stay readable.** Each window drops to a shell when its program stops, with the command in
  the history, so restarting one piece is an up-arrow away.

Flags: `--sim`, `--no-viz`, `--no-screen`, `--no-voice`, `--no-claude`, `--claude-args`.

`status` also reads the bridge's `/state.json`, so it reports connected headset pages and the age of
the last hand pose alongside the process list.

## 28 Sep 2026 — camera stall, all panels at once, hands-free dictation

### The stereo camera "not connected": ffmpeg asking a question nobody could see

The camera was on USB and `camera_share.py` was running, but no frame had been written for five
minutes. ffmpeg was sitting at `File '/dev/shm/bhl_camera.jpg' already exists. Overwrite? [y/N]`: the
last run had left its frame behind, and without `-y` ffmpeg stops to ask. It also could not be stopped
normally: blocked on that prompt it never checks for SIGTERM, and as the terminal's foreground process
it swallowed the keystrokes meant for the shell. `camera_share.py` now runs ffmpeg with `-y -nostdin`
and stdin from /dev/null, so no prompt can block it again.

### Panels: all three up at once, and the one you look at is the one you drive

The page used to show one panel at a time (left trigger x2 cycled them). Now the PC screen, the stereo
camera and the robot panel sit on an arc around you, left to right, fixed in the room:

* **Look at a panel for 0.3 s and it takes the focus** — bright, with a blue outline; the others dim.
  A glance on the way past does not take it, and looking at a gap changes nothing.
* **The controls act on the focused panel:** left stick moves it (up/down, nearer/further), Y + left
  stick resizes it, right stick press recentres the arc with it straight ahead, holding the right stick
  resets it. Growing one pushes its neighbours out; they never overlap.
* Left trigger x2 focuses the next panel (for when looking is awkward). A focus chosen that way lasts
  until you look at a different panel. A is now only the robot's yes.
* A calibration question pulls the focus to the robot panel, and a toast in front of you says where it
  is ("robot panel, on your right"). Toasts now appear where you are looking.
* The panel you are not looking at refreshes slower (screen 4 Hz, camera 10 Hz instead of 15/30).
* The camera panel says "Stereo camera offline" instead of showing run_teleop's log.
* Positions and sizes are saved per panel under new keys (`bhl.panels.*`), so the old one-panel sizes
  do not carry over.

### Dictation: why sentences were cut off, and hands-free mode

Two long dictated messages ended mid-sentence and a short one arrived as `And I-`. Four things could end
a recording early, and none of them said which one had:

| cause | fix |
| --- | --- |
| no audio packet for **0.5 s** (any Wi-Fi hiccup) ended the sentence; the rest was dropped | 4 s, and the end reason is printed |
| a press that never delivered audio waited forever; the next press **wiped** it (`listening...` twice) | ends after 4 s; a second press keeps the first part |
| `stopTalking()` sent **no stop** if the headset had taken the microphone mid-sentence | the stop is always sent, with a reason |
| **squeezing the grip** ended dictation — a hand closing round the controller while talking | only while armed or calibrating, where grip + trigger drive the robot |

**Hands-free:** say **"voice"**, wait for the rising beep, speak; it sends after **5 s of silence**
(falling beep). The page streams the microphone continuously while hands-free is on (32 KB/s), except
while one of Claude's replies is being read aloud, so a reply containing "voice" cannot wake it.
voice_typer spots the word with a second Vosk recogniser restricted to `["voice", "[unk]"]` (offline,
cheap), requires it to be the first word of the phrase with confidence ≥ 0.6, and keeps the last 20 s
of audio, so "voice, open the camera" said in one breath keeps the sentence and drops the wake word.
The trigger still works as before; pressing it during a hands-free sentence makes that sentence end on
release. **Right trigger x2 toggles hands-free** (on by default). Options: `--wake WORD|off`,
`--silence SECONDS`, `--max-seconds` (90).

`tools/test_voice_typer.py` drives the real voice_typer with a stand-in Vosk and real-time audio: wake
word then sentence, wake word and sentence in one breath, a press with no audio, a lost stop plus a
1 s stall, and a tap. The page tests gained the arc, gaze focus and hands-free cases.

### 28 Sep 2026 (later) — the main screen, and the robot as a 3D model

**The panel you look at becomes the main screen.** The middle of the arc is the main place. Looking at
a panel waiting at the side for 0.3 s slides it into the middle, and the panel that was there slides
out to take its old place. The main screen has its own size, height and distance (left stick, Y + left
stick, held right stick = reset). The side panels are 0.62 m previews. A panel that has just been
swapped out lands where you are still looking, so it cannot be picked again until your eyes have left
it after the slide; without that, the two would swap back and forth for ever.

**The robot panel is now the robot.** `scripts/teleop/robot_model.py` simplifies the Onshape export
(26 STLs, 42 MB, about 44k triangles each) with pymeshlab's quadric edge collapse to 28k triangles,
1.4 MB, flat-shaded. The build takes about 30 s once and is cached in `~/.cache/bhl`, keyed on the URDF
and meshes. The bridge serves it at `/robot_model.json|bin`. The page poses it from the joint angles in
the status, so it mirrors the real arms, and it stands in the robot panel:

* each arm part is coloured by its joint: green OK (a calmer green if calibrated earlier), red for a
  problem (BLOCKED, STUCK, WEAK, MAXED, LAG, NO CAN, FAULT...), grey not tested, and the joint under
  test pulses yellow. Hands take their wrist's colour; body and legs stay dark;
* the text shrinks to the question and its A / B lines, one line of news (a problem first, then the
  microphone), and a colour legend. The hint and the joint table moved to the details view;
* **left trigger: details** (and back). It brings the robot panel to the middle. While a joint is being
  calibrated, it shows that joint in words: angle now vs asked for, power ("between 1.5 and 5.5 N·m:
  never less, never more"), what it gets in between (gravity + spare), stiffness (kp/kd), direction
  (flipped or not), and the last test. Otherwise it shows the old full table. This replaces left
  trigger x2 "focus next", which looking at a panel now does.

run_teleop's status gained per joint `u` (URDF joint), `id` (CAN ID), `qc` (commanded angle, for the
teleop check to come) and `p` (kp, kd, floor, headroom, gravity scale, flip, range, date, last attempt).
The page's forward kinematics are tested against Pinocchio on a random pose: they agree within 0.1 µm.

### 28 Sep 2026 (evening) — the robot shows what is active; the camera holds still until asked

**"It didn't show the active one."** Triple X runs the reach check and then arms. Neither marked a
joint as under test, so the model had nothing to light, and with all 10 joints calibrated everything
was calm green. run_teleop's status now carries `active` (URDF joints to glow), `pose` (angles to
demonstrate instead of the real ones) and `camera`, from `ArmSupervisor._model_hint()`:

| what is happening | what the robot in the headset shows |
| --- | --- |
| power calibration | the joint under test glows |
| reach check | the pose to hold (arms down, then arms forward: L pitch -1.5, R pitch +1.5), both arms glowing |
| claws setup | that claw's hand (D3 left, D7 right) |
| armed | the arm you are driving (grip held); in mirror mode the left grip lights the RIGHT arm |

**Camera follow is now a toggle, off at every arming.** The stereo camera's pan/tilt followed the head
whenever teleop was armed, so looking round the panels to debug swung the picture being debugged.
`ServoTeleop.follow` starts False; **A x2** (only while armed, never while a question is up) switches it.
Paused, the camera holds where it is. Resumed, it carries on from that angle, measured from where the
head is at that moment, so it never jumps; the angle kept is the one the servo actually reached, not
one past its limit. The ARMED lines say which ("Camera holds still (A x2: follow your head)"), and the
page announces each change. New event `cam` in the page, the bridge's and run_teleop's EVENTS.

### 29 Sep 2026 — "agent": spoken commands for the rig itself

Say **"agent"**, wait for the beep, and say a command; it ends after 2 s of silence. The phrase is
**never typed to Claude**: voice_typer parses it (`parse_command`, plain keywords, no model) and the
page carries it out. "agent" shares the wake-word recogniser (grammar `["voice", "agent", "[unk]"]`).

| say "agent, ..." | does |
| --- | --- |
| stop / start the looking swap (or "... moving to the center") | look-to-centre off / on (saved) |
| camera follow / camera hold still | the A x2 camera toggle (armed only) |
| details | the robot panel's details on / off |
| show the camera / screen / robot | makes it the main screen |
| recenter / reset the panel | right stick press / hold |
| hands-free off | the wake word off (the trigger still works) |
| help | lists these |
| hands on / off | answers that hand tracking is not built yet |

Options: `--agent WORD|off`, `--agent-silence`. Tests: `test_voice_typer.py` (agent phrase -> command,
nothing typed) and `test_quest_page.py` (look-to-centre off/on, camera, unknown phrase).

## 29 Sep 2026 — the robot's face display, finding the operator, and why only one screen works

**Only one screen, because the PC has no graphics driver.** The robot's small screen is HDMI (the user
moved it to HDMI 1, the monitor to HDMI 2). The PC is a Beelink EQ with an **Intel N150**, whose two
HDMI ports can each drive a screen, but Linux shows one connector (`card0-Unknown-1`, no EDID). The
GPU is bound to `simple-framebuffer` (the firmware's framebuffer, simpledrm): one screen, fixed
resolution, and the desktop is software-rendered (`llvmpipe`, "Accelerated: no"). The cause is that
kernel 6.8 (the newest for Ubuntu 22.04) does not know the N150's graphics, PCI **8086:46D4**. Checked
against the driver's ID list at each tag: v6.8 lacks it, v6.9 and later have it. The
`i915.force_probe=46d4` already on the boot line cannot help: force_probe only works for chips in the
driver's list.

`tools/enable_dual_hdmi.sh` installs Ubuntu's mainline build of the **6.12.111** LTS kernel next to
6.8: download, sha256 check against CHECKSUMS, `apt-get install` of image + modules. There are no DKMS
modules, every robot driver is in-tree (gs_usb, uvcvideo, ch341, cp210x, iwlwifi), Secure Boot is off,
and GRUB shows its menu for 10 s so 6.8 stays one choice away. `--check` reports the driver and the
screens. **Not run yet**: it needs sudo and a reboot, the user's call. Afterwards: Settings ->
Displays (join, arrange, primary = the monitor), and the GPU is accelerated as a bonus.

**Finding the operator: `scripts/teleop/person_track.py`.** It reads camera_share's frames (the camera
itself stays free), rotates them 180° (the camera is upside down), and runs NanoDet-Plus (OpenCV zoo,
3.8 MB, COCO person) on one eye. A face detector alone fails: the headset covers the eyes. The head is
found by, in order: YuNet face (works with the headset on too; the lower face shows), a headset-shaped
white blob (a white wall behind the head is rejected by shape), or the top of the box. It reports
yaw/pitch (HFOV 100° assumed) to the relay as `{"type": "person", ...}` at 3 Hz; each detection costs
about 0.2 s of CPU.

On the three frames tested it found: standing with arms out and headset on (0.81, head by face); off to
the side; and sitting with the back turned behind the chair (0.45). That last one needs a threshold of
0.4, so boxes under 20% of the frame height are dropped, because a shelf bottle scored 0.52. The 3D
Claude-style logo that turns to face the operator waits for the second screen.

(Serial: a CP2102N bridge appeared on `/dev/ttyUSB0`. It answers nothing (not a Nextion; esptool gets
no bytes), so it is not the display. The claw Nano moved to `ttyUSB1`; ServoLink finds it by id, fine.)

### 29 Sep 2026 (afternoon) — two screens, the robot's face, and a camera that keeps you centred

* **Two screens work** on the 6.12 kernel. `tools/screens.py`: the ASUS monitor (HDMI-2) is the main
  desktop, upright; the robot's 1024x600 screen (HDMI-1) sits to its right, turned 180° (mounted upside
  down). It must apply with Mutter's *temporary* method and write `~/.config/monitors.xml` itself: the
  "persistent" method shows GNOME's "Keep these display settings?" and silently reverts after ~20 s.
  That undid the layout once while the user was in the headset. The desktop is Xorg, still
  software-rendered: Mesa 23.2 does not know the N150 (kisak-mesa PPA would fix it; not done).
* **The face** (`face_display.py`, window `face`) is a 3D Claude-style spark in #D97757, spinning and
  turning to face the operator. It is a borderless window laid exactly over the robot screen. Full
  screen via SDL/XRandR scaled the image ("super zoomed") and hid windows behind it. It ignores the
  keyboard, since a "q" meant for a terminal once closed it, and its tmux window restarts it. It uses
  about 70% of one core.
* **The tracker** (`person_track.py`, window `track`) pins every thread pool to one core, cutting it
  from 230% to about 60%. Ports: relay 11006, face 11010, run_teleop 11011 (11008 = screen-share
  pointer, 11009 = speech).
* **Camera modes** (`servos.CameraAim`, in every state except the claws/camera setup menu):
  * **track** (default): keeps the operator centred, 0.6 of the offset per detection, ±3° deadband,
    12° max step, drifting home after 8 s with nobody in view;
  * **head**: follows the headset from wherever the camera is;
  * **still**: holds.

  A x2 toggles track and head; "agent, camera track / head / still". run_teleop adds the camera's
  angle and sends the robot-relative direction to the face, which ignores the tracker's raw
  (camera-relative) reports while those are fresh.
* **The camera panel froze** after a bridge restart. It had no stall watchdog, unlike the PC screen;
  it now re-asks after 4 s without a frame.
* **The claw/camera Nano (CH340) was unplugged** on 29 Sep: only the CP2102N is on USB. ServoLink now
  keeps looking every 3 s and re-finds a dropped Nano, so plugging it back in needs no restart.

## 29 Sep 2026 (evening) — Claude sessions, the developer view, the lidar and the IMU

**The two unknown USB devices.**
* `ttyUSB0` (CP2102N, 3-5) is an **RPLidar C1**: model 0x41, firmware 1.02, 460800 baud. It is silent
  until it gets a command, which is why the morning's probes found nothing.
* The CH340 on the CAN hub (3-4.4) is a **WitMotion IMU** at its factory 9600 baud (0x55 packets).
  The robot's own `berkeley_humanoid_lite_lowlevel/robot/imu.py` expects it set to 460800 with
  quaternion output; it has been left unconfigured.

**The claw/camera Nano had gone missing.** With two CH340s and no serial numbers, `/dev/serial/by-id`
gets ONE name, and it pointed at the IMU, so ServoLink never found the Nano: the camera tracking and
the claws were dead, and the IMU was probed every 3 s. `servos.ch340_ports()` now lists CH340s from
sysfs, the Nano is recognised by its reply, ports that answer wrongly are left alone until
replugged, and ports are opened exclusively.

**`sensor_share.py`** (bringup window `sensors`) reads the lidar (standard scan; the motor needs up to
4 s to spin up) and the IMU, and writes `/dev/shm/bhl_lidar.json` and `/dev/shm/bhl_imu.json`. The
bridge serves them as `/sensors.json`.

**Claude sessions** (`claude_sessions.py`): "agent, open claude session [called CAM]" starts
`claude --model claude-opus-5-5 --effort max` in the tmux window `c-<NAME>` (1-3 letters: as said, or
the next free letter). The first session, in window `claude`, is "M". `/sessions.json` gives every
session's screen; each has a panel in the headset.

**Dictation goes where you look.** A session panel you look at steadily for 0.6 s, or bring to the
middle, becomes the target; looking at anything else (the camera, a calibration menu) keeps the last
one. The page tells voice_typer (`{"type": "voice", "action": "target"}`) on every change and every 5 s.

**Views.** "agent, developer" = stereo camera, lidar (top-down, 1 m rings), IMU (roll/pitch/yaw, a
horizon, accel/gyro/mag) + the sessions; "agent, normal" = PC screen, camera, robot panel + sessions.

**Gaze with more panels.** When a swap reshapes the arc, the panels slide under a still head, and with
five panels a third one could land under the gaze and be picked: a chain reaction. While the head stays
within ~5° of where it was at the swap, whatever the slide brings under it now counts as seen; turning
the head picks normally.

**Audio.** Replies are played through their own audio context again. Routing them through the
microphone's was cut off whenever the headset reclaimed the mic, about once a minute. A paused mic
context is resumed. The page heartbeat now reports the audio state, chunks sent and hands-free.

### 29 Sep 2026 (night) — both eyes above the camera, with what the tracker sees

**Vision pair.** Above the stereo camera panel, in both views, sit two flat pictures: the left and
right eye, upright, side by side. They are sized from the camera panel (80 % of its width) and follow
it when it slides aside. They are not on the arc, so looking up at them swaps nothing. On them:
- `human NN%`: in the left eye, the detector's box (what the camera servo follows) and a red cross at
  the head it aims at, with how it found it (face / headset / box). In the right eye, the pose model's
  confidence and body.
- a box per body part (head, torso, L/R arm, L/R hand, L/R leg) and a stick figure. These come from
  MediaPipe BlazePose (OpenCV zoo, `~/.cache/bhl/pose.onnx`, 5.5 MB). "L" and "R" are the person's own
  sides.

"agent, vision off" / "vision on" (also "hide/show the detections") toggles the pair.

**How it runs.** `person_track.py` has a second thread, at nice 10, that draws only while
`/dev/shm/bhl_vision.want` is under 3 s old. The bridge's `/vision.jpg` touches it on every request.
- **Tracking:** the pose model starts from the detector's left-eye box (the eyes are 6 cm apart, which
  is close enough for the right eye too). After that it follows each eye from its own two tracking
  points (hip centre and body reach), as MediaPipe does.
- **Cost:** about 80 ms of a core per eye per frame, at the tracker's 3 frames a second, so roughly
  half a core while you are in view. It costs almost nothing when nobody is in view.

**Why pictures, not a stream.** The page fetches one picture about every 330 ms. The WebSocket does not
connect from the headset, so arm poses travel as POSTs. The events feed, camera stream and PC screen
stream already hold three of the ~6 connections a browser allows per host, and a fourth held-open
stream would leave the POSTs very little.

**Dictation goes to the session you spoke to.** The target is now fixed when a sentence starts (the
wake word or the trigger). Before this, it was read when the sentence was typed. The 5 s of silence and
Whisper's read-back come after you speak, and a glance at another session panel in that time (which
swaps it to the middle and retargets) sent the sentence there. The voice log now says `typed ... to A`.

**"agent, exit developer mode"** switched *into* developer mode: the word "developer" matched first.
Exit/leave/off/stop/quit/close/hide with "developer" now goes to the normal view (test in
`tools/test_voice_typer.py`).

**The arc's order (same night, as the user corrected it).** The row is no longer shuffled by swaps:
- **Left of the camera:** the sensor panels (lidar and IMU, or PC screen and robot panel).
- **Right of the camera:** the Claude sessions.
- **On each side:** the one used most recently sits next to the camera and the one used longest ago
  at the far end. Your current session is always first.

The panel you pick still slides into the middle, but the whole row slides with it, so the order holds.
`arrange()` builds the order from `usedAt`, and a focus change re-arranges. The order and the dictation
target (`bhl.voice.session`) survive a page reload.

Because the whole row slides now, a different panel can land under your gaze. Whatever arrives there
during the slide is held (`arc.blocked`) until your eyes leave it, so a drifting head cannot chain-pick
it. In the normal view this moves the PC screen and robot panel to the right of your session.

**Normal view: no session panels (30 Sep 2026).** In the normal view (PC screen, camera, robot panel)
the Claude sessions are no longer panels. The page sends the dictation target `@`, which means "the
session the PC's own screen shows": the active window of the `bhl` tmux session, which every attached
terminal shows. If that window is not a Claude one, the window shown just before it counts, so a quick
look at `teleop` keeps the target. `voice_typer.on_screen()` turns `@` into a name when each sentence
starts (`claude_sessions.shown()`, also `claude_sessions.py shown`).

"agent, open claude session" in the normal view also switches the PC's terminal to the new session
(`claude_sessions.show`), so the session you see is the one you talk to. The developer view is
unchanged: session panels, and dictation goes where you look.

**Calibration mode (30 Sep 2026).** The normal view is now called calibration mode ("agent, calibration
mode", or "normal" as before). It shows the PC screen, the camera and the robot panel in that fixed
order.
- **Looking never swaps panels there.** The main screen stays while you look around the others.
- **Left stick click:** brings the next screen in (PC screen, camera, robot panel, and round again).
- **Left stick held 1 s:** leaves the headset view, which a click used to do.
- **Still automatic:** a robot question brings the robot panel forward, and the left trigger still
  opens the robot panel with its details.
- **Developer view:** looking still swaps panels there (`lookSwaps()`), and the button works too.

**Dictation stops safely.** Ctrl+C, a restart or `kill` now finishes the sentence being taken down or
read by Whisper before exiting, then leaves with `os._exit`. A restart at 01:09 had dropped a
60-second sentence mid-read, and the process then hung in the native libraries' teardown. A second
Ctrl+C still exits at once (test 7 in `tools/test_voice_typer.py`). Before restarting it from a
script, check the voice window is not showing `listening` or a fresh `ended:` line.

**Closing, saving and reopening Claude sessions (30 Sep 2026).**
- **"agent, close session A"** (or "close the demo video session", or "close this session") saves the
  session by its title and conversation id to `~/.config/bhl/saved_sessions.json`, then closes its
  window. Claude Code has already written the transcript to disk.
- **"agent, open the demo video session"** runs `claude --resume <id>` in a new window, under its old
  letter if that is free. Plain "open claude session", or anything with "new", starts a fresh one.
- **"agent, saved sessions"** lists the saved ones.
- **Refusals:** M is never closed, and neither is a session whose conversation cannot be found.
- **Empty sessions:** one that nothing was said in just closes.

How a running session is tied to its conversation:
- Sessions opened from now on start with `--session-id <uuid>`, kept on the window as `@session_id`.
- Sessions started before that are matched by title: the newest `ai-title` line in each transcript
  (`~/.claude/projects/-home-joses-Berkeley-Humanoid-Lite/<id>.jsonl`) against the title Claude Code
  puts on the window. On 30 Sep this found all four live sessions correctly.

Test: `tools/test_claude_sessions.py` (its own tmux session and a fake `claude`).

**Dictation never types into a dialog (30 Sep 2026).** Session B sat for an hour behind Claude Code's
"Teach auto mode about your environment?" dialog, and every dictated sentence went into the dialog
instead of the prompt. The Enter that follows a sentence picks a dialog's highlighted answer, which
on a permission question is usually "Yes".
- Before typing, `voice_typer` asks `claude_sessions.in_the_way(pane)`. It reports a dialog when the
  prompt box is missing and the bottom of the screen shows "Esc to cancel", "Enter to continue",
  "Do you want to proceed?" or "❯ 1." choices.
- If there is a dialog, nothing is typed, and the headset shows "not typed: the session is showing a
  dialog ...".
- "agent, escape" (or "cancel the dialog") sends Esc to that session, but only while a dialog is up:
  Esc on its own would stop Claude's work.
- B's dialog was dismissed with Esc, so it did not opt into anything; the user can still run it later.

**Tuning the arms while driving (30 Sep 2026).** The teleop logs showed two causes of "wrong"
movements:
- **Hands sent out of reach.** At motion scale ~1.5 the hand targets were 16–84 cm past where the arm
  can go. At ~0.5–0.6 they were within 1–3 cm.
- **Joints short of power.** L yaw followed 35% of its commands in the latest session, capped at
  1.5 N·m against its 4.0 ceiling; R pitch followed 42% on 28 Sep.

The fixes are now live, without restarting anything:
- **`run_teleop` takes changes on UDP 11012** (`--tune-port`): a joint's power (±0.5 N·m, capped at
  its kind's ceiling or its `cal_cap`), gravity help (±0.05), the sensitivity, a direction flip
  (refused while the motors are on) and undo. Each change is saved to the profile and written into
  the teleop log as a `tune` row.
- **Its hint line says what to do while driving:** "L yaw is at its power limit (1.5 N·m) and cannot
  keep up - say 'agent, more power left yaw'", or "left hand sent 14 cm past where the arm can reach -
  let go of the grip and grab again closer in".
- **Log rows now carry each joint's torque and limit, and the commanded hand position.** The audit
  uses them to tell "no power" from "an end stop", and "out of reach" from "not following".
- **`tools/teleop_tune.py`:** `audit` (the newest log: what went wrong, with the command to try),
  `show`, `power`, `gravity`, `sensitivity`, `flip`, `undo`.
- **By voice:** "agent, more / less power left yaw", "more gravity left pitch", "less sensitive",
  "flip left wrist", "undo that". The joint names are in Whisper's prompt, and "jaw" is read as yaw.
- **Talking to a Claude session:** `CLAUDE.md` (new) tells every session the audit, then change,
  then retry routine, and the arm safety rules.
- **The audit never calls a sagging joint "wrong way".** L pitch in the 00:56 session moved 0.11 rad
  against a commanded 0.61 while failing to lift, and a first version of the audit suggested flipping
  it, which would have been exactly wrong: it followed 93–96% in every other session. A flip now needs
  at least 40% of the commanded motion made in the opposite direction.
- Tests: `tools/test_teleop_tuning.py` (the simulated arms and a scratch copy of the profile).

Not built yet: the ghost arm (commanded pose drawn over the real one on the headset's robot).

**A bad zero blocked arming (30 Sep 2026, 09:05).** After a power calibration's "ZERO POSE: A = this
is zero" was answered with the arms not hanging, every reach check ended with "L elbow reads -1.08 rad,
outside its range", and teleop refused to arm. With the arms at rest the joints read L elbow -1.08,
R yaw +1.85, R wrist -2.22 and R roll +0.57 rad. A had been pressed 91 times that morning, so an A
meant for another prompt most likely answered this one. Fix: with the arms resting and still (checked
over 2 s), restart run_teleop while STOPPED; it takes zero at start-up, and every joint then read 0.00.

The zero question now names the furthest joint and asks again when any joint is over 0.35 rad (20 deg)
from the start-up zero: "ZERO: ARE THE ARMS HANGING? L elbow is -62 deg from the start-up zero ...
A = yes, B = keep the start-up zero" (test in `tools/test_teleop_tuning.py`). It takes effect at the
next run_teleop restart.

"agent, the robot is not currently moving" did nothing because it is not a command. The agent only
does commands; questions go to Claude with "voice".

**"No response to my movements" was no grip held (30 Sep 2026, 09:13).** Armed five times that morning,
and no teleop log got a single sample: samples are only written while a grip is held, and the bridge
saw no grip in any of its packets. The arms only follow while a GRIP (the side button under the middle
finger) is held, and nothing in the headset showed which buttons were pressed. So:
- **The controls strip** sits under the toast, where you look, while the robot is armed or
  calibrating and for 3 s after any press. Each controller's TRIGGER, GRIP, X/Y or A/B and STICK light
  up as pressed. Its line says which robot arm a grip is driving ("driving the robot's RIGHT arm
  (mirror: your left hand is its right)"). Armed with no grip held, a hand that travels 15 cm in
  1.5 s turns it red: "you are moving, but no GRIP is held - squeeze the side button under your
  middle finger".
- **While armed, the robot panel's tips only cover driving:** "HOLD a GRIP to move that arm · grip +
  trigger: claw · R stick: sensitivity · X×3: stop".
- **The reach check's steps are numbered and literal:** "1 of 2 (no buttons to hold): arms hanging
  at your sides, then press A once" / "2 of 2: arms straight out in front, hold them, press A once".
  In run_teleop, this takes effect at its next restart.
- **Moved to a footer (same morning).** The user found the floating strip distracting, so it is now a
  footer flush under the robot panel (under the camera panel in the developer view). It is as wide as
  the panel, in its plane, dims with it, and moves and resizes with it.
- **"Grip is held but it says no grip":** pointer mode (B twice) forces both grips off for the arms,
  which looks exactly like a dead grip. The strip now says "POINTING AT THE PC: the grips cannot move
  the arms now - press B twice to stop pointing". The page heartbeat carries `pad` (raw grip/trigger
  per hand, e.g. `LG- R--`) and `pointer`, so `curl -s localhost:8080/state.json` shows from the PC
  what the controllers report.

**Mirror mode flipped forward/back (30 Sep 2026, 09:32).** The user, facing the robot:
- "Elbow outward: moves outward, great."
- "Move my hands forward: it goes the opposite way."
- "Inward: it keeps thinking it's backwards."

The cause: 24 Sep's mirror fix turned motion 180° about vertical (`FACING = diag(-1, -1, 1)`). That
fixed sideways, but also reversed forward/back, which a reflection keeps. Inward reaches carry the
hands forward too, so they read as backward. `FACING` is now `diag(1, -1, 1)`: reach toward the
robot and its hand comes toward you, sideways reflects, up is up. The wrist rotations are conjugated
by the same reflection. `tools/test_teleop_tuning.py` checks forward, sideways and up on the arm
model, and that straight mode still copies. It takes effect at the next run_teleop restart, which
needs the arms hanging still at zero.

**Long dictation no longer loses its end.** A sentence that reaches the 90 s limit while you are still
talking is sent, and a new one begins at once, to the same session (the page beeps). The user's
"whole calibration report" had been cut at 90 s. Test 8 in `tools/test_voice_typer.py`.
