# Notes for Claude sessions on this rig

## Tuning the arms while the operator teleoperates

The operator may be in the Quest headset, driving the arms, and talking to you through
dictation ("the left arm didn't follow when I reached forward"). Their words arrive as
your prompt, and the start of each reply is read aloud: keep replies to a few sentences.

1. What just happened: `.venv/bin/python tools/teleop_tune.py audit --seconds 30`
   reads the newest teleop log (arm_validation/teleop_logs/). It reports which joint fell
   short or ran out of power, which moved the wrong way, and which hand was sent out of
   reach, each with the change to try.
2. The live settings: `.venv/bin/python tools/teleop_tune.py show`
3. Change one thing, then ask them to repeat the same motion:
   - `tools/teleop_tune.py power "left yaw" up|down`: ±0.5 N·m, never past the joint's ceiling
   - `tools/teleop_tune.py gravity "left pitch" up|down`: for an arm that sags or floats
   - `tools/teleop_tune.py sensitivity up|down|0.6`: for hands often sent out of reach
   - `tools/teleop_tune.py flip "left yaw"`: only with the motors stopped (triple X), and
     only for a joint the audit calls WRONG WAY again and again. A joint sagging under
     gravity is not a flipped one.
   - `tools/teleop_tune.py undo`

Changes reach the running run_teleop.py at once and are saved to
configs/arm_power_profile.json. The operator can make the same changes by voice
("agent, more power left yaw", "agent, less sensitive", "agent, undo that").

## Never

- Restart run_teleop.py (tmux `bhl:2`) unless it is STOPPED with the motors off, and the
  arms hang still at their zero: it takes zero at start-up, and zero taken with an arm bent
  makes every angle wrong and arming refused. Check first:
  `curl -s localhost:8080/state.json` gives the last status's joints; every `q` should be near 0
  and unchanged over a couple of seconds. Ctrl+C twice (the first holds damping, the second
  exits), then start it with the explicit command, never "Up Enter": the shared bash
  history can recall `claude`.
- Use `Humanoid()` or the stock `write_configurations.py`: they map can0/can1 to the legs.
- Rewrite a motor's device_id or flux offset.
