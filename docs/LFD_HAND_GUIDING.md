# Hand-guiding on this rig: feasibility, a GUIDE state, and how to label the data

*3 October 2026. A read-only assessment for the learning-from-demonstration comparison (branch `lfd-comparison`). Nothing here was run on the robot. The numbers come from this repo's code, configs and notes, from the upstream motor firmware, and from offline computation on the URDF (appendix).*

**Short answer.** The firmware can float a gravity-compensated arm, and the measured friction makes it light to push: about 1–3 N at the claw in most poses, and 4–5 N for shoulder pitch with the arm out sideways. But not today. With the current shoulder-roll zero and gravity fit, both shoulder-roll joints would float outward to their limits within seconds of the stiffness going to zero. Hand-guiding becomes supportable after the six steps in §6: the roll zero tested and gravity refit, a power cutoff within reach, and a GUIDE state (§3) that passes the hardware test (§4).

**How to read it.**
- Labels as in the roadmap: **(measured)**, **(computed)**, **(estimate)**, **(unconfirmed)**.
- `file:line` points into this repo at commit `66a5d11`. `run_teleop.py` is being extended in parallel (the record tap), so its line numbers will shift: read them with `git show 66a5d11:scripts/teleop/run_teleop.py`. Short names:
  - `bring-up` = `arm_validation/TELEOP_AND_ARM_BRINGUP.md`, `roadmap` = `docs/ROADMAP_2026-10-02.md`, `spec` = `docs/LFD_RECORDING_FORMAT.md`;
  - `profile` = `configs/arm_power_profile.json`, `dry-run.json` = `arm_validation/config_write_20260922_121222/dry-run.json`, `can0_id1.json` = `arm_validation/20260916_encoder_rerun2/motor_configs/can0_id1.json`;
  - `motorstuff` = `2026-09-23-motorstuff-pastedcontent-idf751.txt`, `arm-session` = `claude-arm-session.txt`, `urdf` = the URDF at `arm_power.URDF_PATH`;
  - `run_teleop.py`, `arm_driver.py`, `arm_power.py`, `arm_calibration.py`, `servos.py` and `quest_bridge.*` are in `scripts/teleop/`.
- `fw:` is the motor firmware, which the submodule `source/berkeley_humanoid_lite_lowlevel` does not contain (it holds only the host side). It means `Recoil-Motor-Controller-B-G431B-ESC1/Core/` in github.com/T-K-233/recoil-motor-controller-besc, commit `3571ab6` (7 Sep 2025), cloned to read it.
- Poses. These are left-arm URDF angles under today's zero, in the order pitch, roll, yaw, elbow, forearm roll. The right arm takes the mirror pose, every sign flipped (checked: the gravity torques mirror within 0.014 N·m).
  - **P0 hanging:** (0, 0, 0, 0, 0).
  - **P1 forearm forward:** (0, 0, 0, 1.4, 0); claw tip 25 cm forward, 20 cm below the shoulder.
  - **P2 reach to table:** (−0.7, 0.15, 0, 1.0, 0); 34 cm forward, 7 cm below.
  - **P3 arm forward, level:** (−1.5, 0, 0, 0, 0); 41 cm forward, level.
  - **P4 arm out sideways:** (0, 1.0, 0, 0, 0).

---

## 1. The firmware

**Which source.**
- The motors report firmware `0x20250226` (bring-up:99; `can0_id1.json:9`), the same as upstream (`fw:Inc/motor_controller_conf.h:20`).
- Earlier sessions read the firmware on the robot PC and cited its line numbers (arm-session:816, 821, 826, 845–847; `tools/arm_common.py:62`). Every one matches this commit.
- That the flashed binary was built from exactly this source is **(unconfirmed)**.

### 1.1 What POSITION mode computes

The position loop runs on every 5th tick of the 10 kHz loop, i.e. at 2 kHz (`fw:Src/position_controller.c:45-51`, `fw:Inc/motor_controller_conf.h:86-89`):

```
setpoint        = clamp(position_target, limit_lower, limit_upper)
torque          = kp·(setpoint − position) + kd·(0 − velocity) + integrator + torque_ff   position_controller.c:64-84
torque_setpoint = clamp(EMA(torque), −torque_limit, +torque_limit)                       position_controller.c:102-109
iq_target       = torque_setpoint / Kt / gear_ratio                                       motor_controller.c:362-370
```

1. **Units are joint-side.** Position and velocity are divided by the gear ratio, and the measured torque is multiplied by it (`fw:Src/motor_controller.c:350-356`). PDO-3 carries `[position target, torque feed-forward]` (`fw:motor_controller.c:647-657`; `arm_driver.py:94-95`).
2. **kd damps absolute velocity, not velocity error.** The velocity target is 0 (`fw:position_controller.c:71`). With kd = 2 (profile:10), every joint resists any motion with 2 N·m per rad/s, in teleop and in guiding alike.
3. **No integrator in practice.** It is clamped to ±torque_limit (`:73-76`), but `position_ki` is 0 (dry-run.json:18).
4. **The torque filter.** The EMA uses alpha 0.2696 (dry-run.json:26). That is a 100 Hz low-pass at 2 kHz **(computed)**: f = −ln(1−α)·2000/2π. The same formula gives the source's own "50 Hz" for its default (`fw:position_controller.c:14-15`). A gain change therefore lands as a step, softened over about 2 ms.
5. **No firmware position limits.** They are ±inf (dry-run.json:23-24), so every range check is host-side.

**Does kp = 0 with feed-forward = gravity give a floating arm?**
- Yes, in principle. With kp = 0 the motor gives `torque_ff − kd·v`, clipped to the limit. The feed-forward holds the arm up, and kd adds viscous drag.
- The arm floats where the feed-forward error is smaller than the joint's friction. Elsewhere it drifts, at about (|error| − friction)/kd.
- Today the roll joints fail this (§2.3).
- The position target does not matter at kp = 0. It does matter during the gain ramps and at range walls. So GUIDE holds it while kp fades, then lets it follow the measured angle (§3).

### 1.2 Changing gains at runtime

1. **The host side.** `ArmDriver.write_gains` sends two SDO writes per joint (kp and kd) for all ten joints, and sleeps 2 ms per joint: about 20 ms of blocking (`arm_driver.py:216-221`). Today it runs once, in `ArmSupervisor.start` (`run_teleop.py:608`).
2. **The firmware side.** An SDO write is a raw 32-bit store into the controller's RAM struct, with no range check and no reply (`fw:motor_controller.c:700-730`). It takes effect at the next 2 kHz update.
   - It never touches flash. A power cycle reloads the flashed gains (`fw:motor_controller.c:228-252`).
   - The 22 Sep snapshot reads kp 30 and kd 2 in RAM: the values run_teleop writes (dry-run.json:17-19).
3. **Safe while STOPPED, unconditionally.** DAMPING never uses kp or kd (§1.3). This is the guard `_tune` uses for flips (`run_teleop.py:877-880`).
4. **In POSITION, only as a slow ramp.**
   - A step Δkp gives a torque step Δkp·(target − q). An idle arm sits 0.008–0.045 rad off its command (roadmap:153), so a 30 → 0 step would release up to 1.35 N·m at once.
   - Moving the target onto the measured angle in one go releases the same torque, so that is no better.
   - GUIDE needs changes in POSITION: passing through DAMPING would drop the gravity support.
   - **Going down:** keep the target where it is, and fade kp over ≥ 1 s (roadmap:359 also asks for ≥ 1 s). Write kp once per joint per cycle: at most 5 SDO frames per cycle on one bus. The PD torque then shrinks by ≤ 0.03 N·m per step.
   - **Going up:** start from the measured angle, where the error is zero.
   - **kd costs nothing at rest,** because the damping torque is kd·v. So kd is written once, at the start of each ramp.
   - The SDO rate matters: the gs_usb adapters do not restart by themselves after a full TX buffer (bring-up:37, :161).
5. **Writes are not acknowledged.** Re-send the final value, then read it back. `ArmBus.read_f32` exists, and `_poll_state` already reads by SDO (`run_teleop.py:1076-1105`).
6. **Validate every write.** The value must be finite, with kp in [0, 80] and kd in [0, 5], as `JointPower.sanitized` checks on load (`arm_power.py:130-131`).
7. **The gains outlive the state.**
   - A mode change keeps the gains and the targets (`fw:Inc/position_controller.h:55-61`).
   - `_motors_on` does not re-write the gains (`run_teleop.py:737-745`).
   - So a GUIDE that stopped at kp = 0 would re-arm limp. Every exit path must restore the profile gains.

### 1.3 Torque limit, DAMPING, IDLE and TORQUE

1. **The torque limit** clamps the total torque (PD, integrator and feed-forward together), symmetrically, in joint N·m (`fw:position_controller.c:106-109`).
   - The current loop's own clamp, i_limit 20 A (`fw:Src/current_controller.c:64`; dry-run.json:27), is far higher: Kt·20 A·15 = 27.6 N·m at the joint for pitch and 35 N·m for the others **(computed)**.
   - So the SDO torque limit, which the host keeps ≤ 10 N·m (`ABSOLUTE_MAX_TORQUE`, `arm_power.py:36`), is the binding cap.
   - It is the most torque the motor can apply in either direction. A limit of |ff| + h therefore bounds any fault in the PD terms to h beyond the gravity support. It does not bound an error in the feed-forward itself.
2. **DAMPING shorts the windings.** It sets PWM duty 0 on all three phases with the outputs still on (`fw:motor_controller.c:402-404`). In PWM mode 1, active-high, that holds all three low-side switches on (`fw:Src/main.c:598-604`).
   - There is no feed-forward and no position hold: iq_target is computed only in POSITION, VELOCITY and TORQUE (`fw:motor_controller.c:362-370`).
   - The braking torque grows with speed: b ≈ (2/3)·N²·Kt²/R **(estimate)**. This holds at low speed, with the firmware's amplitude-invariant Clarke transform (`fw:Src/foc_math.c:10-13`):

     | joint | motor | b, N·m·s/rad |
     |---|---|---|
     | pitch | MAD M6C12: Kt 0.0919 with R 0.1886 Ω (the commented-out profile, `fw:Inc/motor_profiles.h:26-32`), or Kt 0.0896 with R 0.1379 Ω (the active one, `:17-23`) | 6.7–8.7 |
     | roll, yaw, elbow | MAD 5010 110KV (Kt 0.1176, R 0.6193 Ω, `:35-41`) | 3.4 |
     | forearm roll | R inferred from its current-loop gains | ≈ 18 |

   - Each pitch value pairs a profile's own Kt with its own R. Pairing one profile's Kt with the other's R would give 9.2, which is not a motor.
   - The configs' i_ki = R/L shows which profile each joint uses: 5803 = 0.1886 Ω / 32.5 µH, with Kt 0.0919, in the 22 Sep config (dry-run.json:29, :31). On 16 Sep the left pitch motor still carried the active profile's values, i_ki 4538.5 (0.1379 Ω / 30.4 µH) and Kt 0.0896 (`can0_id1.json:27, :32`). Braking depends on the motor's physical Kt and R, which the two profiles bracket, so both ends stay.
   - The simulator models DAMPING as 3.0·v (`arm_driver.py:361`).
   - The operator has swung joints by hand in DAMPING and found them stiff but movable (motorstuff:2685, 3525–3527, 3546–3548, 3557).
3. **IDLE** turns the PWM off: no torque and no braking (`fw:motor_controller.c:419-422`). A raised arm in IDLE falls, held only by friction. run_teleop sends IDLE on the second Ctrl+C (`run_teleop.py:1345-1350`).
4. **TORQUE mode** (0x11) outputs the feed-forward alone (`fw:position_controller.c:94-100`). It is not recommended:
   - a mode switch turns the PWM off and zeroes the torque setpoint (`fw:motor_controller.c:137, 182-183`; `position_controller.h:60`);
   - it has no damping;
   - `_poll_state` stops everything when a joint leaves POSITION (`run_teleop.py:1104-1105`).

   Staying in POSITION and ramping kp avoids all three.

### 1.4 The watchdog

1. **Period.** A 10 kHz timer with period `watchdog_timeout`·10 − 1, i.e. 1 s (`fw:main.c:665-667`, `fw:motor_controller.c:78`; watchdog_timeout 1000 in `can0_id1.json:10`).
   - The period is set at boot only: an SDO write to `watchdog_timeout` does not reach the timer (`fw:motor_controller.c:723-729`).
   - Shortening it would need a flash write, which is out of bounds.
2. **What feeds it.** PDO-1, PDO-2, PDO-3 and HEARTBEAT (`fw:motor_controller.c:632, 644, 656, 675`). SDO and NMT frames do not.
3. **On expiry,** any mode except DISABLED, IDLE and CALIBRATION drops to DAMPING and sets WATCHDOG_TIMEOUT (`fw:Src/app.c:40-47`). The bit stays set until a power cycle or an explicit write (arm-session:850–851).
4. **The host.** run_teleop sends PDO-3 every cycle, in every state (`arm_driver.py:10-11`, `run_teleop.py:710`). It stops all motors after 25 missed replies, about 0.5 s (`run_teleop.py:671-674`), and it reads one joint's mode every 0.2 s (`run_teleop.py:1076-1105`).

What this means for GUIDE:
- If run_teleop dies, every joint brakes within 1 s, and the arm sinks rather than falls.
- If the host stalls for less than 1 s, the last feed-forward stays in force, evaluated at the last pose. At the measured stall pattern (19% of loop intervals over 70 ms, roadmap:393), that error stays small.

---

## 2. Backdrivability and drift

### 2.1 What resists a hand

Per joint, left / right (profile):

| joint | gear | kinetic friction, N·m | max torque, N·m | gravity need, N·m | gravity scale (r²) | fit bias, N·m |
|---|---|---|---|---|---|---|
| shoulder pitch | 15 | 0.55 / 0.60 | 5.0 / 5.0 | 2.58 / 2.72 | 0.95 (0.91) / 0.94 (0.89) | +0.12 / −0.27 |
| shoulder roll | 15 | 0.52 / 0.71 | 4.75 / 5.25 | 3.10 / 2.75 | 1.05 (0.87) / 1.25 (0.84) | −0.79 / +1.28 |
| shoulder yaw | 15 | 0.35 / 0.29 | 1.5 / 1.5 | 0.19 / 0.19 | 1.0 (no fit) | — |
| elbow | 15 | 0.48 / 0.58 | 2.5 / 2.5 | 0.76 / 0.80 | 0.95 (0.77) / 0.89 (0.85) | −0.15 / +0.11 |
| forearm roll | 15 | 0.34 / 0.29 | 1.5 / 1.25 | 0.01 / 0.02 | 1.0 (no fit) | — |

Sources:
- friction: profile:39, 93, 147, 199, 253 (left) and 305, 359, 413, 465, 519 (right);
- max torque: profile:13, 67, 121, 173, 227, 279, 333, 387, 439, 493;
- gear ratio: −15 on every joint (bring-up:16, :93; the preflight check, `arm_driver.py:202-203`).

1. **What "friction" means here.** It is the `sign(v)` term of the fit `tau = scale·g + friction·sign(v) + bias`. The fit uses samples taken at ≥ 0.175 rad/s while the motor drove the arm (`arm_calibration.py:966-994`).
   - So it is kinetic friction, with the motor driving the gearbox.
   - Two things were never measured: breakaway (static) friction, and friction when the output drives the motor, which is what a hand does.
2. **Breakaway evidence is loose.**
   - The left roll moved ±0.14 rad on 1.2 N·m, gravity included (motorstuff:7873–7876).
   - A 16 Sep note says a shoulder needed ~3 N·m to move at all (bring-up:19, :371). That was before the commutation fixes (motorstuff:5777–5780), so it is not used.
3. **Damping adds kd·v.** Today that is 1.0 N·m at 0.5 rad/s (kd 2). At the GUIDE kd of 1.0 proposed in §3, it is 0.5 N·m.

### 2.2 The push a human needs at the claw (computed)

The force at the claw that moves one joint is its torque divided by the distance from the claw point to that joint's axis. The distances come from URDF forward kinematics, at two claw points: the hand's centre of mass, 55 mm past the forearm-roll joint (urdf:176), and the tip, 133 mm past it (roadmap:340).

Left arm, kinetic friction only; in parentheses, while moving at 0.5 rad/s with kd 1.0:

| pose | shoulder pitch | shoulder roll | shoulder yaw | elbow |
|---|---|---|---|---|
| P1 forearm forward | 1.7–2.1 N (3.3–4.1) | 2.5–2.6 (4.9–5.2) | 1.4–2.0 (3.4–4.9) | 1.9–2.7 (3.8–5.5) |
| P2 reach to table | 1.6–1.9 (3.0–3.7) | 1.7–2.0 (3.4–3.9) | 1.6–2.4 (4.0–5.7) | 1.9–2.7 (3.8–5.5) |
| P3 arm forward, level | 1.4–1.7 (2.6–3.2) | 1.2–1.5 (2.4–3.0) | on its axis | 1.9–2.7 (3.8–5.5) |
| P4 arm out sideways | 4.3–5.3 (8.3–10.1) | 1.2–1.5 (2.4–3.0) | on its axis | 1.9–2.7 (3.8–5.5) |

1. **These are light.** It takes 1.2–2.7 N (0.1–0.3 kgf) to keep a joint moving, and 2.4–5.7 N at a brisk pace, except shoulder pitch with the arm out sideways (P4): 4.3–5.3 N, and 8.3–10.1 N at a brisk pace (item 4).
   - The right arm's friction is 0.8–1.4 times the left's, and its forces scale the same way.
   - If breakaway is twice the kinetic friction (unmeasured), double the first numbers.
2. **Yaw and forearm roll need the elbow bent.** With the elbow straight, the claw lies on both axes (distance ≈ 0). The calibration bends the elbow for the same reason (`arm_calibration.py:53-55`).
3. **The forearm roll is a twist, not a push:** 0.29–0.34 N·m, about 10–11 N at a 3 cm grip radius.
4. **Shoulder pitch with the arm out sideways** has a short lever (10–13 cm), so it takes 4–5 N.

### 2.3 Would the arm drift? (computed)

With kp = 0, a joint is pushed by net = ff − g_true.
- ff = gravity_scale·g(q) is what run_teleop sends (`arm_power.py:222-223`, `run_teleop.py:697`).
- g_true is the torque gravity really needs.
- The joint drifts when |net| exceeds its friction.

Two explanations of today's fit give g_true:
- **fit:** the fit's own model, g_true = scale·g + bias. The feed-forward omits the bias, so net = −bias. This assumes the bias is a real torque, such as cable drag; a current-sensor offset would not push the arm.
- **zero15:** the roadmap §3.1 hypothesis, still untested. The hanging arm is really 15° nearer vertical in roll than today's zero says, and the URDF masses are right. Then g_true = g(q + q_hang), with q_hang −15° (L roll) and +15° (R roll) (roadmap:190).
  - FK confirms the sign: the upper arm (its yaw axis) is 15.0° from vertical at URDF q = 0 and 0.0° at q_hang.
  - q_hang is also exactly the URDF's inner roll limit (urdf:233).

**Roll joints.** Outward is + for the left arm and − for the right; bold marks |net| above friction, so the joint drifts:

| pose | L roll ff | net, fit | net, zero15 | friction | R roll ff | net, fit | net, zero15 | friction |
|---|---|---|---|---|---|---|---|---|
| P0 hanging | +0.83 | **+0.79** | **+0.78** | 0.52 | −0.97 | **−1.28** | **−0.94** | 0.71 |
| P1 forearm forward | +0.64 | **+0.79** | **+0.59** | 0.52 | −0.74 | **−1.28** | −0.70 | 0.71 |
| P2 reach to table | +0.83 | **+0.79** | +0.51 | 0.52 | −0.97 | **−1.28** | −0.67 | 0.71 |
| P3 arm forward, level | +0.06 | **+0.79** | +0.06 | 0.52 | −0.07 | **−1.28** | −0.07 | 0.71 |
| P4 arm out sideways | +2.89 | **+0.79** | +0.44 | 0.52 | −3.44 | **−1.28** | **−1.00** | 0.71 |

1. **Both explanations push both arms outward.**
2. **Simulated for 10 s.** One joint free at a time, kp 0, kd 1.0, with friction the only hold:
   - Under "fit", both roll joints run to their URDF limits from every pose: 1.2–1.3 rad from P0–P3. P4 starts 0.31 rad from the limit.
   - Under "zero15", the L roll moves 0.87 rad from P0 and 0.44 rad from P1, and the R roll reaches its limit from P0 and from P4.
3. **The other joints do not drift** under either explanation. The largest |net| is 0.27 N·m on pitch (friction 0.55–0.60), 0.15 on the elbow (0.48–0.58), 0.21 on yaw (0.29–0.35) and 0.01 on the forearm roll.
4. **Gravity-scale error.** A scale error Δs drifts a joint when |Δs·g| exceeds its friction.
   - Pitch drifts at |Δs| > 0.20–0.22 with the arm level (P3), and at > 0.28–0.30 at P2. Roll drifts at > 0.19–0.26 at P4. The elbow needs ≥ 0.58.
   - The pitch and elbow fits (0.89–0.95) are within that, if the URDF's mass distribution is right.
   - The roll scales (1.05 and 1.25) are what the zero hypothesis would distort, so they cannot be trusted until it is tested.
5. **After the fix.** At the roadmap's pass line, roll |bias| ≤ 0.3 N·m (roadmap:196), with the gravity scale taken as exact, no pitch, roll or elbow joint moves in the simulation.
   - A scale error adds to the bias: the joint drifts once |bias| + |Δs|·|g| exceeds its friction. The refit's pass line allows any scale in 0.85–1.15, which bounds the scale, not its error.
   - The roll at P4 carries the most gravity, 2.75 N·m. At the bias limit, the left roll (friction 0.52) then drifts once |Δs| > 0.08, and the right roll (0.71) once |Δs| > 0.15. So the pass line alone does not guarantee that D passes at P4; item D (§4.3) tests it.
   - Yaw and forearm roll have friction of only 0.29–0.35 N·m.
   - But they carry at most 0.21 N·m of gravity in total, so even a 25% model error is ≈ 0.05 N·m.

### 2.4 What a 15° roll zero error does to the feed-forward (computed)

The change in ff once the zero is corrected, scale·(g(q + q_hang) − g(q)), in N·m:

| pose | L roll | R roll | L / R pitch | L / R yaw |
|---|---|---|---|---|
| P0 hanging | −0.78 | +0.93 | 0.00 | ±0.01 |
| P1 forearm forward | −0.58 | +0.69 | 0.00 | +0.21 / −0.21 |
| P2 reach to table | −0.49 | +0.59 | −0.12 / +0.11 | +0.14 / −0.14 |
| P3 arm forward, level | −0.06 | +0.07 | −0.11 / +0.10 | 0.00 |
| P4 arm out sideways | −0.32 | +0.39 | 0.00 | 0.00 |

1. **Today's feed-forward supports an A-pose that is not there.** At the hanging pose, it pushes each roll 0.8–1.0 N·m outward when the true need is about zero. URDF q = 0 alone already needs 0.79 / −0.78 N·m on roll.
2. **The error is 0.3–0.9 N·m on roll,** comparable to or above roll friction. On pitch and yaw it reaches 0.1–0.2 N·m.
   - The yaw feed-forward at P1 (0.21 N·m) is entirely spurious: with the upper arm truly vertical, the yaw axis is vertical too.
3. **Stiffness hides it.** At kp 30 these errors show only as a hold offset of ≤ 0.03 rad. At kp 0 they are the whole story.

### 2.5 If a guided arm drops to DAMPING (estimate)

The sink speed is (|g_true| − friction)/b, with b from §1.3, and g_true from the zero15 explanation (§2.3), the one the table uses:

| pose | what sinks | speed, rad/s | at the claw |
|---|---|---|---|
| P1 | pitch, elbow | 0.03–0.10 | ≤ 3 cm/s |
| P2 | pitch, elbow | 0.07–0.23 | ≤ 8 cm/s |
| P3 | pitch, elbow | 0.08–0.35 | ≤ 14 cm/s |
| P4 | roll | 0.52–0.58 | 22–24 cm/s at first, slowing as the arm nears vertical |

This is a braked sink, not a fall. The pitch speeds use the lower b (6.7), so they are upper bounds; with b = 8.7 they are about 23% lower.

The other explanations of g_true change the P4 roll row most (computed from the same model):
- **fit** (g_true = ff + bias): 0.43–0.47 rad/s at P4, and at P3 the roll also creeps, at 0.06–0.15 rad/s;
- **the plain URDF gravity** (g_true = g): 0.61–0.67 rad/s at P4.

All the numbers are estimates, and the test in §4 measures them.

---

## 3. A GUIDE state (a proposal; not implemented)

### 3.1 Who holds what

1. **One arm at a time.**
   - One hand guides the robot arm. The other robot arm holds at full stiffness.
   - The operator also works the **dead-man**, an input of its own (item 2), never a teleop grip. Nothing is guided unless it is held in its middle position.
   - A Quest controller's trigger drives the guided arm's claw, only while the dead-man enables (§3.6).
2. **The dead-man is a three-position enabling switch,** as on industrial robots' teach pendants and hand-guiding devices: the enabling control device of IEC 60204-1 (the switch itself is covered by IEC 60947-5-8; robots, ISO 10218-1).
   - Position 1, released: off. Position 2, held in the middle: enabled. Position 3, squeezed past the middle: off.
   - Returning from position 3 to position 2 does not enable again. It takes a full release first.
   - Why three positions: a two-position grip stops only when released, but a startled hand may clench instead of letting go. Release and a full squeeze both stop.
   - **The cheapest realisation (estimate):** a three-position enabling switch, wired to a USB HID adapter (a button encoder that shows up as a gamepad, of the kind sold for arcade buttons), read by a small script on the robot PC. The switch is either the hand-held kind sold for teach pendants, or a three-position enabling foot switch, which frees both hands. A plain USB foot pedal does not qualify: it has two positions, and stops only on release.
   - **The script** reads the adapter (Linux evdev) and sends run_teleop a heartbeat datagram on localhost, about 50 times a second, saying "enabled" only while the switch is in position 2. Where the switch has two enabling contacts, both go to the adapter, and enabled needs both closed: a disagreement is a fault, read as off. run_teleop reads a heartbeat older than 0.1 s, or a dead script, as released. The heartbeat stays on the robot PC, so the dead-man no longer rides the unauthenticated LAN port (§4.1 item 4).
   - **The claw:** with a foot switch, the free hand keeps a Quest controller, whose trigger drives the claw. With a hand-held switch, that hand is taken, so the claw needs its own button on the same adapter, or it keeps its pulse while guiding.
3. **An interim alternative, until the switch exists: the Quest controller's analog squeeze,** as a software three-position device.
   - `grip_value`, the squeeze from 0 to 1 (WebXR `gamepad.buttons[1].value`). It is a new packet field: the bridge forwards only a boolean today (`quest_bridge.py:107`).
   - From 0.3 to 0.9: enabled. Below 0.3 (released) or above 0.95 (squeezed hard), for two rows in a row: off. After a hard squeeze it enables again only after a release below 0.3.
   - **Its limits:**
     - It is software on a wireless link, not a safety device: one channel, nothing redundant, and every reading passes through the browser, the WiFi, the bridge and the unauthenticated UDP port (§4.1 item 4). The link-age stop (§3.5) is its only watchdog.
     - Whether a startled squeeze reaches 0.95 is **(unconfirmed)**. Measure it before relying on it (§4.3, S2).
     - The bridge drops the button whenever the pose is estimated or untracked (`quest_bridge.py:104-107`), and `parse_packet` skips untracked hands (`run_teleop.py:402-403`). So a controller held low at the side would release the dead-man. GUIDE needs `grip_value` outside that tracking gate. GUIDE never uses the controller's pose, so the reason for the gate does not apply. If the browser loses the controller entirely, `readHand` reports the grip released (`quest_bridge.html:4468-4470`). That fails safe.
     - The squeeze is also the teleop grip. So from the guide command on, the grips no longer drive the IK (§3.2), and after GUIDE ends they stay ignored until both are seen released (§3.5). Otherwise a squeeze would resume teleop, with the controller in the operator's hand beside the robot.
4. **The headset stays on.**
   - The page reads the buttons inside its WebXR frame loop (`quest_bridge.html:4592-4611`). The headset's cameras track the controllers, and with the visor off, tracking dies (bring-up:111).
   - The operator must see the arm, so passthrough must work. It has been unreliable (bring-up:24, :324).
   - The page already reports its session mode (bring-up:323). GUIDE refuses to start unless that mode is `immersive-ar`.
5. **Fingers out of the jaws.** Hold the forearm or a handle, not the claw jaws: the trigger can close them.

### 3.2 Entry, from ARMED only

An entry sequence, with checks the way `_arming_problem` checks arming (`run_teleop.py:719-735`). The teleop grips and the dead-man are separate inputs, so "no grip held" and "the dead-man held" never contradict each other.
1. **When the voice command arrives** ("agent, guide left arm" or "agent, guide right arm", new events in `EVENTS`, `run_teleop.py:55`), all of these must hold:
   - **Healthy and armed.** ARMED, with the motors on for more than 0.5 s, so the feed-forward ramp is done (`run_teleop.py:695`). No faulted joint, no missed replies, and no `NO CAN`, `MAXED` or `LAG` tag (`run_teleop.py:471-493`).
   - **The arm is free and still.** No teleop grip is held (`not any(solver.held)`). The guided arm is at rest: |q̇| < 0.05 rad/s for 0.5 s. Every joint is at least 0.05 rad inside its soft range (`profile.ranges`, `arm_power.py:209-220`).
   - **The dead-man is not enabled yet:** it must be engaged after the command, not left held from before.
   - **The calibration is trusted.** A passed `zero_check` exists (spec:131). The guided arm's gravity fits have |bias| ≤ 0.3 N·m and r² ≥ 0.8.
   - **The camera** is at its preset and `still` (§3.6), and the headset session is `immersive-ar`.
2. **From the command on, the grips no longer drive the IK.** The solver is bypassed for both arms, and each arm's `q_cmd` holds. A squeezed grip, tracked or not, now changes nothing.
3. **The operator engages the dead-man within 3 s**, with the headset link younger than 0.25 s (`run_teleop.py:389-391`) and, for the switch, a fresh heartbeat. Otherwise GUIDE is cancelled, and the state stays ARMED, with the grips ignored until both are seen released (§3.5).
4. **The kp ramp (§3.3) starts once the dead-man has enabled continuously for 0.3 s.**

### 3.3 Ramp down, so nothing jumps

1. **The holding arm** keeps its `q_cmd`. Nothing changes for it.
2. **The guided arm keeps its hold target** during the ramp. The feed-forward stays evaluated there (`run_teleop.py:697`), within 0.045 rad of the measured pose.
3. **Write kd 2 → 1 once.** At rest, that changes no torque.
4. **Fade kp from 30 to 0 over 1.0 s.**
   - Linearly, with one SDO write per joint per cycle: about 48 steps of 0.6 N·m/rad.
   - At the idle error of ≤ 0.045 rad, each step changes the torque by ≤ 0.03 N·m.
5. **At kp = 0:**
   - Re-send the final gains three times, then read them back.
   - Then lower the torque limits to the GUIDE value (§3.4) with `LimitWriter.update(..., urgent=...)` (`run_teleop.py:447-468`). Doing this earlier could clip the up to 1.35 N·m that the PD still carries.
   - From now on, `q_cmd` follows the measured angle each cycle, so the feed-forward is taken at the measured pose.

### 3.4 While guiding

1. **kp 0 and kd 1.0** (preregistered; today they are 30 and 2).
   - Fallback, only if the drift test fails after the refit: kp 3 with a 0.05 rad slip anchor. The target follows q only when q is more than 0.05 rad away. That holds up to 0.15 N·m of drift torque, and feels like 0.15 N·m of extra friction.
2. **The target follows the arm, clipped into the soft range.**
   - Inside the range, kp 0 makes the target irrelevant.
   - Outside, that joint's kp rises to 10 N·m/rad (ramped over 0.2 s), with the target at the limit: a soft wall of 0.5 N·m per 0.05 rad.
   - More than 0.15 rad outside → STOPPED.
   - The walls are only as good as the soft ranges. Today `range_lower` and `range_upper` are null on both rolls (profile:113-114, :379-380), so the interim inner-roll clamp of roadmap §1.0, item 3 is not applied. Under today's zero, the URDF limit lets the upper arm pass about 15° beyond vertical, toward the torso (roadmap:176).
3. **The feed-forward is evaluated at the measured pose,** as above.
4. **Torque limit** = min(|ff| + 1.5, max_torque) N·m. Today's headroom is 1.25–1.75 N·m (profile:12).
   - It leaves room for kd·v up to 1.5 rad/s.
   - It puts the `MAXED` line, 0.92 of the limit (`run_teleop.py:481`), above the 1.0 rad/s watchdog. When a hand lowers a loaded arm, ff and the damping add up. At P3, damping alone reaches the line only above 1.17 rad/s; on an unloaded joint above 1.38 rad/s; on the right forearm roll, capped at 1.25 N·m, above 1.15 rad/s **(computed)**.
   - A headroom of 1.0 would trip `MAXED` at 0.71 rad/s at P3. That would be a nuisance stop during a legitimate brisk move.
   - It bounds a host fault in the PD terms to 1.5 N·m beyond the gravity support: 3.6–6 N at a shoulder-to-claw lever of 0.25–0.42 m. It does not bound an error in the feed-forward itself (§1.3).
5. **A velocity watchdog that drops to DAMPING.**
   - Joint speed comes from the last three measured rows and their own timestamps. The loop is jittery (19% of intervals over 70 ms, roadmap:393). The firmware's own velocity is only available by SDO or by the unused PDO-4 fast frame (`fw:app.c:49-61`).
   - Trips: any guided joint above 1.0 rad/s for two rows in a row, or any joint of the holding arm above 0.3 rad/s.
   - Response: `_stop`, both arms to DAMPING (`run_teleop.py:1007-1017`).
   - 1.0 rad/s is 0.4 m/s at the claw, under teleop's 1.5 rad/s cap (`run_teleop.py:1243`).
   - DAMPING brakes in proportion to speed (§1.3), so it is the right stop for a fast arm. The holding arm then sinks slowly too (§2.5), as on every stop today.
6. **Every stop that exists today stays,** all going to STOPPED:
   - triple X (`run_teleop.py:632-634`);
   - CAN silence (`run_teleop.py:671-674`);
   - a joint leaving POSITION, or an encoder fault (`run_teleop.py:1096-1105`);
   - the headset link lost for 5 s (`run_teleop.py:669-670`).

   Two additions while guiding:
   - `_poll_state` reads only the guided arm's five joints, so each is checked every 1 s instead of 2 s.
   - `MAXED` on a guided joint for 0.3 s → STOPPED. At kp 0 and below the watchdog speed, damping cannot saturate the motor (item 4). So a saturated motor means the PD is pushing: a range wall overrun or a host fault.
7. **Every exit restores the profile gains** (§1.2, item 7). On a stop, write them after the motors are already in DAMPING, where they are harmless. As a second guard, `_motors_on` re-writes them before `set_mode(POSITION)`.
   - **When the process itself ends,** no exit path of GUIDE runs, and the RAM gains stay at the GUIDE values (kp 0, kd 1 on the guided arm). What happens at 66a5d11:
     - **Ctrl+C:** `KeyboardInterrupt` ends the loop, and `finally` calls `shutdown()`, which sets DAMPING (`run_teleop.py:1334-1337`, `:614-619`). A second Ctrl+C sets IDLE (`:1347-1350`).
     - **SIGTERM:** the handler raises `KeyboardInterrupt` (`:1310-1315`), so the same `shutdown()` runs; the motors are left in DAMPING, never IDLE (`:1341-1342`, `:1351-1352`).
     - **An unhandled exception in `tick`:** `except KeyboardInterrupt` does not catch it, but `finally: shutdown()` still sets DAMPING (`:1336-1337`). The exception then leaves `main`, so the second block, which closes the link, the servos and the driver (`:1353-1356`), never runs.
     - **A hard crash** (SIGKILL, a segfault, the PC losing power): no Python runs. The PDO-3 frames stop, and the firmware watchdog gives DAMPING within 1 s (§1.4). The same holds if `shutdown()` itself fails.
   - DAMPING never uses kp or kd (§1.3), so gains left at the GUIDE values are harmless while stopped. They last until the next `ArmSupervisor.start` writes the profile gains (`run_teleop.py:608`), which every run does before it can arm, or until a motor power cycle reloads the flashed ones (`fw:motor_controller.c:228-252`). The other tools that enter POSITION write their own gains first. So `start` re-writing the gains is a rule GUIDE relies on: it must stay, and the `--sim` tests check it (§3.8).

### 3.5 Exit back to hold

1. **Triggers:** the dead-man leaves position 2, whether released or squeezed hard (§3.1); its heartbeat is older than 0.1 s; the headset link is older than 0.25 s; or "agent, end guide".
2. **Freeze** `q_hold` at the measured angle, and set `q_cmd` = `q_hold`.
3. **Raise the torque limits first** (`LimitWriter` raises at once). Write kd 1 → 2 once: it can only brake motion. Then ramp kp from 0 to 30 over 1.0 s, with the target at `q_hold`, where the error starts at zero.
4. **Watch:** if any guided joint moves more than 0.1 rad from `q_hold` during the ramp → STOPPED. This is a safety trip, and the hardware pass lines stay well inside it (§4.3, S2).
5. **Back to ARMED** with `solver.held` false and `solver.smooth` cleared for both arms, so the next grip re-anchors on the current pose. Roadmap:563 requires the same for POLICY.
6. **The grips drive the IK again only after both have been seen released for 0.5 s.** With the interim Quest dead-man (§3.1 item 3), the grip the operator was squeezing is that controller's own, and the controller is in their hand beside the robot: it must not resume teleop by itself.

How far the arm moves during that ramp **(computed)**, from a 1-DoF model of the firmware law with Coulomb friction 0.5 N·m and inertia 0.01–0.1 kg·m²:

| residual torque | excursion |
|---|---|
| below friction | none |
| 0.6 N·m | 0.010–0.015 rad |
| 1.0 N·m | 0.05–0.07 rad |
| 1.3 N·m | 0.08–0.12 rad |

So after the refit the return is predicted to be still. With today's roll residuals it would jump by 0.05–0.12 rad.

### 3.6 Claws and camera

1. **Claws.**
   - Feed `ServoTeleop.update` (`servos.py:225-232`) with "the dead-man enables" as the guided arm's grip, and the free controller's trigger (or the switch adapter's claw button, §3.1 item 2) as its trigger.
   - The claw follows the trigger only while the dead-man enables, and keeps its last pulse otherwise (`servos.py:207-210`, `:222-223`).
   - The page must treat GUIDE like ARMED in `talkControl`, where a held grip ends dictation instead of starting it (`quest_bridge.html:4578-4584`).
2. **Camera.**
   - `CameraAim` has a `still` mode but no preset (`servos.py:268-299`). Its default `track` mode would follow the operator, who stands at the robot.
   - Add a stored preset; move there at ≤ 20°/s (`servos.py:253`); set `still`; and ignore camera events while guiding (`run_teleop.py:650-658`).
   - The SG90s are open-loop, so check the first frame against a reference frame before recording (roadmap:568).
   - The recorder already refuses to start unless the camera is `still` (spec:104).

### 3.7 Logging

One tap row per cycle, as in every state (spec:30), with:
- `src` `guide` (reserved, spec:40); `state` `GUIDE` (see spec gap 1 below); `motors` true;
- `q` measured; `qc` the follow target, which is not a demonstration (spec:153); `tff` and `lim` the GUIDE values;
- `held` [false, false], because the IK is idle; `tgt` [null, null];
- `grip` and `trig` as received, although the grips drive nothing in GUIDE; `claw`; `cam` at the preset; `cam_mode` `still`.

Episodes start with `mode` `guided` (spec:104) and are labelled with `meas_future` only (spec:153).

**Spec gaps this exposes** (reported here; the spec is not changed):
1. `state` lists only STOPPED, ARMED and CAL (spec:38), while `src` reserves `guide`.
2. Rows carry no gains, but GUIDE changes them at runtime. Add `kp` and `kd`, float[10]: the values last written.
3. `src` is per cycle, but guiding is per arm. Add `guide`, bool[2]. Also add `deadman`: the enabling input GUIDE uses (the switch's position, 1, 2 or 3, or the interim `grip_value`), which `grip` does not show.
4. No converter rule ties a row's `src` to its episode's `mode`. A guided episode with a teleop reset inside would be counted as guided throughout.
5. `episode.json` could record the GUIDE parameters: kp, kd, headroom and the watchdog thresholds.

### 3.8 What the change touches, and its `--sim` tests

**Touches.**
- `run_teleop.py`: `ArmSupervisor.tick`, `_stop`, `_motors_on`, `_poll_state`, `status()`, `EVENTS` and the tap row.
- A new non-blocking per-joint gain writer with read-back, beside `LimitWriter`; per-joint gain writes in `ArmDriver` and `SimArmDriver`.
- The dead-man: a small script that reads the enabling switch's USB HID adapter and sends its heartbeat on localhost, and its reader in `run_teleop.py` (§3.1 item 2). For the interim Quest dead-man, `quest_bridge.py` and `quest_bridge.html` forward `grip_value`, outside the tracking gate.
- `quest_bridge.html`: GUIDE in `talkControl` and on the panel.
- `voice_typer.py`: "guide left / right" and "end guide".
- `CameraAim`: the preset.
- The profile: the GUIDE parameters, with `sanitized()` clamps.

**Tests**, in the style of `tools/test_teleop_tuning.py`, against `SimArmDriver`. It already models the firmware's PD, feed-forward, EMA, clamp and DAMPING (`arm_driver.py:342-377`).
1. Ramp down and back up: no per-cycle torque step above 0.05 N·m, and the final gains read back.
2. Every exit path ends with the profile gains in the simulator's RAM: the dead-man released, the dead-man squeezed hard, its heartbeat lost, link at 0.25 s and at 5 s, triple X, CAN silence, encoder fault, overspeed, range.
   - And the process ending in GUIDE (§3.4 item 7): Ctrl+C, SIGTERM and an exception raised in `tick` each leave the simulator's joints in DAMPING, with the GUIDE gains still in RAM. Then a restart's `ArmSupervisor.start` writes the profile gains: assert the simulator's RAM gains after it, before anything can arm.
3. Overspeed reaches DAMPING within two cycles; the range wall works; the holding arm does not move. A brisk lowering at 0.9 rad/s, with a loaded and with an unloaded joint, does not trip `MAXED`.
4. Drift with `--sim-payload` different from the gravity scale reproduces the §2.3 prediction.
5. Refusals: no zero check, a bad fit, the camera not still, the session not AR, a teleop grip held when the command arrives, the dead-man already enabled when the command arrives, no dead-man within 3 s.
6. The entry sequence (§3.2): with the interim Quest dead-man, entry with the grip squeezed and the controller tracked is accepted, and neither arm's `q_cmd` changes while the grip is held. After GUIDE ends, a squeezed grip does not drive the IK until both grips have been released.
7. The three positions: a squeeze past 0.95 exits like a release, and moving back to the middle does not enable again until a full release.

---

## 4. Hardware test, preregistered (for the operator to run later)

### 4.1 Prerequisites

1. **The roll zero.** Verified with the inclinometer test (roadmap:180–185). If the hypothesis holds, the corrected zero is in use. `zero_check` is recorded (spec:131).
2. **Gravity refit after the zero change** (roadmap B4): roll |bias| ≤ 0.3 N·m and scales 0.85–1.15 (roadmap:196). Refit pitch and elbow too.
3. **A motor-power cutoff** within reach of the guiding stance (roadmap:87, :122).
4. **The LAN hole closed** (roadmap §1.0, item 2). GUIDE's entry event rides the unauthenticated UDP port (`run_teleop.py:316`), and so does the interim Quest dead-man. The enabling switch's heartbeat stays on the robot PC (§3.1).
5. **Healthy joints.** All ten are healthy (roadmap:71). If the claw cube is fitted, it was on for the refit (roadmap:284).
6. **The software.** GUIDE built, its `--sim` tests passing, and one `--sim` rehearsal on the robot PC.
7. **On hand:** the three-position enabling switch and its adapter, or, as the interim, the Quest dead-man with its limits (§3.1); working passthrough; the camera preset stored; a luggage scale with peak hold; a printed 3 cm crank for the forearm roll. A second person watching is recommended.

### 4.2 Order: safety first, light poses first

1. **The stops (S1–S4)** at P1, on each arm.
2. **D, F and J** at P1, then P2, then each joint's third pose:
   - P3 for pitch, elbow and forearm roll;
   - P4 for roll;
   - P2 with yaw at +0.3 rad for yaw, which needs the elbow bent.
3. **S1 again** at the most loaded pose (P3, then P4), with a hand under the claw.
4. **S5 once,** at P1. It is the last item: the session ends with it.
5. **After S5, recover by CLAUDE.md's rule, and do nothing else first.**
   - Catch the arm and lower it to hanging by hand. run_teleop has stopped by itself, on CAN silence (§1.4).
   - Restore motor power. The controllers boot with their flashed gains and limits, which a running run_teleop does not write again (§1.2 item 7). So the arms are not armed again before a restart.
   - Restart run_teleop.py only as CLAUDE.md allows: STOPPED with the motors off, and both arms hanging still at their zero. Check it first: `curl -s localhost:8080/state.json`, where every `q` must read near 0 and stay unchanged over a couple of seconds. Then Ctrl+C twice (the first holds damping, the second exits), and start it with the explicit command, never "Up Enter". If any `q` does not read near 0 with the arms hanging, neither restart nor arm: stop there and report the readings.
   - Run the zero check (`tools/lfd/zero_check.py`) again before anything is guided or recorded.
   - If a CAN adapter needs a reset (bring-up:37), the session is over.

Each guided item starts with the arm placed by teleop, then GUIDE entered. During every hands-off item, a hand hovers 2–3 cm under the claw. The tap records every row.

### 4.3 Items and pass lines (decide them before looking)

| item | how | pass | prediction |
|---|---|---|---|
| **D** drift | in GUIDE, hands off for 10 s; per arm and pose | every guided joint \|Δq\| ≤ 0.05 rad; the claw moves ≤ 2 cm | none after the refit; today both rolls run to their limits (§2.3) |
| **F** push | luggage scale on a loop at the claw's centre of mass, pulled slowly at right angles to both the joint axis and the line from that axis to the loop; read the peak when q has moved 0.01 rad; 3 pulls each way, per joint and pose | median ≤ 10 N for pitch, roll, yaw and elbow; forearm roll ≤ 0.6 N·m (20 N on the 3 cm crank) | 1.2–5.3 N kinetic (§2.2); breakaway unknown |
| **J** return | release the dead-man with the arm still; watch 2 s | max \|q − q_hold\| ≤ 0.03 rad; no joint above 0.3 rad/s | none while the residual is below friction (§3.5) |
| **S1** triple X | during GUIDE | every joint reads DAMPING when next polled; `motors` false in the next row; no joint above 0.8 rad/s while sinking | sink at 0.03–0.58 rad/s (§2.5) |
| **S2** dead-man | while guiding at ≈ 0.3 rad/s, release the dead-man and let go of the arm at the same moment; then again with a hard squeeze (position 3) instead of the release. With the interim Quest dead-man, also record the peak `grip_value` of a sudden hard squeeze | both stop within 0.05 rad of where the dead-man let go, without oscillating and without the 0.1 rad watch tripping (§3.5); with the Quest, the sudden squeeze passes 0.95 | a coast of at most 0.01 rad **(estimate:** ½·I·v² over 0.5 N·m of friction, with I ≤ 0.07 kg·m²**)**; a startled squeeze reaching 0.95 is (unconfirmed) |
| **S3** overspeed | a brisk shove past 1 rad/s (elbow, P1) | STOPPED on the second consecutive row above 1.0 rad/s, and within 0.2 s of the first such row | rows about 21 ms apart, but 19% of intervals over 70 ms (roadmap:393), so a 0.1 s line could fail on jitter alone |
| **S4** link | close the page, or take the headset off | hold within 0.3 s; STOPPED at 5 s | — |
| **S5** cutoff | from the guiding stance, cut motor power at P1 | reached within 1 s; the arm stops being driven | the arm drops on friction alone: catch it |

Unbraked, pitch at P3 would pass 0.8 rad/s within about 25 ms **(computed)**, from the URDF link inertia of 0.068 kg·m²; rotor inertia would make it slower. So S1's line separates braking from no braking.

### 4.4 Time budget

| block | minutes |
|---|---|
| prerequisites, setup, `--sim` rehearsal | 30 |
| the stops at P1, both arms | 30 |
| D, J and F at 3 poses × 2 arms, about 12 min each | 72 |
| S1 at the loaded poses; S5 | 20 |
| copying data and notes | 15 |
| **total** | **about 2 h 50 min**, best as two sessions, one per arm |

---

## 5. The data

### 5.1 The operator in the head camera

While guiding, the operator's hand sits on the forearm or the claw, inside the view, in most frames. Their arm and body are often in view too. Autonomous runs have no human in them. Two problems follow:
1. **An appearance shift.** Skin, sleeve and shadows appear near the claw, and they hide the claw and the object.
2. **A shortcut.** The hand moves just before the arm does, so a policy can learn to read the hand. At test time there is no hand.

Mitigations, cheapest first:
1. **At the source.**
   - Guide from a handle on the forearm, behind the camera's view, rather than from the claw.
   - Wear uniform matte gloves and sleeves, so the appearance is constant and easy to segment.
   - Stand behind or beside the robot, out of frame.
2. **Mask.**
   - Segment the human in each eye on the HPC, and fill the mask with a constant grey.
   - The rig's BlazePose model gives body-part boxes and a stick figure, not pixel masks (bring-up:913–914). A promptable video segmenter would give masks **(unconfirmed; not tested here)**.
   - Apply the same pipeline to the teleop episodes, as random masks of a similar size near the claw. Otherwise a grey blob alone tells guided from teleop.
3. **Inpaint.** Fill the human pixels with a plausible background, using a video-inpainting model **(unconfirmed; not tested)**. This is risky: the hand covers the claw and the object, so the model invents exactly the pixels the task needs.
4. **Accept it as part of the comparison.** This is a cost of kinesthetic teaching, so report it. Train the guided policy both unmasked and masked, and preregister which variant is primary.

Suggested: masking, with the same augmentation on both datasets, as the primary variant; unmasked as the secondary; inpainting only if masking costs success.

### 5.2 Action labels: why both modes use `meas_future`

1. **Guided rows have no demonstration in `qc`.** In GUIDE, `qc` is the follow target at kp 0, or a hold during the ramps. `cmd` is undefined for guided episodes (spec:150–153).
2. **Teleop's `qc` leads the arm by the controller's lag.**
   - The firmware damps absolute velocity (§1.1). At speed v, the steady tracking error is (kd·v + friction)/kp.
   - As a lag in time: kd/kp = 67 ms, plus friction/(kp·v) = 37 ms at 0.5 rad/s. That is ≈ 0.1 s **(computed)**, matching the measured ≈ 0.1 s (roadmap:153).
   - `qc` also carries the IK's 0.35 rad lead clamp, its 1.5 rad/s cap, and targets the arm never reached while a joint was `MAXED`.
3. **So the two labels answer different questions.** `cmd` says where the controller was told to go; `meas_future` says where the arm went 0.1 s later. Training teleop on `cmd` and guided on `meas_future` would mix the demonstration source with the label definition, and the comparison could not separate them.
4. **With `meas_future` on both,** every action is a measured position: the same encoders, the same zero convention, the same H. At execution, both policies send their targets to the same kp 30 controller, with the same lag. The only difference left is how the demonstration was given.
5. **The lookahead H.** The spec's H = 0.10 s comes from teleop's tracking lag (spec:151). Guided data has no controller lag, so there H only sets how far ahead the label looks.
   - Keep H = 0.10 s for both as the primary analysis.
   - Preregister a sensitivity check at 0.05 s and 0.20 s, on both datasets.
6. **Claws** are commanded pulses in both modes, through the same `ServoTeleop` path, so their labels mean the same thing.

### 5.3 Smaller differences

1. **Backlash floats.**
   - With gravity feed-forward, the gear mesh carries almost no load while guiding. So the motor-side encoder can sit anywhere in the backlash band.
   - That is ±0.65° if the arm gearboxes resemble the leg actuator's 1.3° (roadmap:163; unconfirmed for the arms), i.e. ≤ 5 mm at the claw.
   - In teleop, gravity holds one flank.
2. **Speed envelope.** The 1.0 rad/s watchdog keeps guided motion inside what a policy can execute; teleop's cap is 1.5 rad/s.
3. **One arm at a time.** Guided episodes are single-arm, and teleop can be bimanual. Compare on single-arm tasks, or guide the arms in sequence.

---

## 6. Verdict

**Supported after the six steps below; not supported now.**
1. **The shoulder-roll zero** is tested and, if the hypothesis holds, corrected (roadmap §3.1). Today's feed-forward pushes each hanging arm outward by 0.8–1.0 N·m (§2.4).
2. **Gravity is refit:** roll |bias| ≤ 0.3 N·m and scales 0.85–1.15. Today both rolls would float to their limits within 10 s at kp 0 (§2.3).
3. **A motor-power cutoff** is within reach, and the LAN hole is closed.
4. **The operator's view and the dead-man:** a camera preset with `still` enforced, working passthrough, and a three-position enabling switch as the dead-man, separate from the teleop grips (§3.1). The Quest controller's analog squeeze can stand in meanwhile, within the limits listed there.
5. **GUIDE is built** as in §3, with its `--sim` tests passing: gain ramps without torque steps, gains restored on every exit, the entry sequence that bypasses the IK, a three-position dead-man, the velocity watchdog and range walls.
6. **The hardware test in §4 passes.**

The physics supports it:
- The firmware's POSITION law at kp 0 gives gravity support plus damping (§1.1).
- The measured kinetic friction makes the arms light to guide: 1.2–2.7 N at the claw (4.3–5.3 N for pitch at P4), 2.4–5.7 N at a brisk pace (8–10 N for pitch at P4) (§2.2).
- After the refit, the predicted residuals sit below friction everywhere for the bias limit alone, with the gravity scale exact. A scale error adds to it, and at P4 the left roll then drifts once |Δs| > 0.08 (§2.3 item 5); hardware item D tests it.

**Still unverified on the hardware:**
1. That the flashed firmware is this source (same version tag and matching line numbers, but the binary was never compared).
2. Breakaway (static) friction, and friction when the output drives the motor through a printed 15:1 gearbox.
3. Gain changes at runtime in POSITION: never exercised, because `write_gains` runs only at start-up.
4. The DAMPING braking coefficient and sink speed. §2.5 is an estimate.
5. The 15° roll-zero hypothesis itself, and so which explanation in §2.3 holds.
6. Passthrough reliability while guiding. For the interim Quest dead-man, the squeeze while the controller is held low or out of view, and whether a startled squeeze passes 0.95.
7. Arm gearbox backlash and stiffness. Only the leg actuator's are published.
8. Gravity support with a payload in the claw: the 0.1–0.15 kg estimate (roadmap:685) was never tested.

---

## Appendix: how the numbers were computed

The script `guide_feasibility.py` ran with the rig environment on 3 Oct 2026 and passed its 6 sanity checks: the arm joints exist; the roll load at q = 0; the q_hang sign; the upper-arm tilt at q = 0 and at q_hang; and the mirror symmetry. It lives in scratch (`/scratch/sanchej7/tmp/claude-19646/lfd-tests/hand_guiding/`), not in the repo, and reads only the URDF and the profile.

1. **Gravity:** `arm_power.GravityModel` on the URDF. ff = the profile's gravity_scale·g(q).
2. **Levers:** Pinocchio forward kinematics. Each joint's axis is its frame's z axis. The claw point is 55 mm or 133 mm past the forearm-roll origin, along the forearm-roll axis.
3. **Drift:** one joint free: I·q̈ = net(q) − kd·q̇ − Coulomb friction, with I = 0.03 kg·m² (`arm_driver.py:302`). Kinetic friction serves as the static hold, and motion stops at the URDF limits.
4. **DAMPING:** b = (2/3)·N²·Kt²/R, for a phase short at low speed.
5. **Return to hold:** the same 1-DoF model, from rest, with kp rising from 0 to 30 over 1 s. The result barely depends on inertia: 0.073–0.075 rad at 1.0 N·m over 0.01–0.1 kg·m².
