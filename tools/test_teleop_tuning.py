"""Live tuning in run_teleop.py, against the simulated arms and a copy of the profile.

Nothing here touches the robot or configs/arm_power_profile.json: the supervisor gets
SimArmDriver, a scratch copy of the profile, and ports of its own.

  .venv/bin/python tools/test_teleop_tuning.py
"""

from __future__ import annotations

import json
import shutil
import socket
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/teleop"))
sys.path.insert(0, str(REPO / "tools"))

import run_teleop as rt  # noqa: E402
from arm_driver import SimArmDriver  # noqa: E402
from arm_power import GravityModel, PowerProfile  # noqa: E402
import teleop_tune  # noqa: E402


def main() -> int:
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "profile.json"
        shutil.copy(REPO / "configs/arm_power_profile.json", path)
        profile = PowerProfile.load(path)
        gravity = GravityModel()
        port = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        port.bind(("127.0.0.1", 0))
        tune_port = port.getsockname()[1]
        port.close()
        sup = rt.ArmSupervisor(SimArmDriver(gravity), profile, gravity, rt.TeleopIkSolver(),
                               servos=None, person_port=0, tune_port=tune_port)
        sup.start(time.perf_counter())
        yaw = profile.joints["left_shoulder_yaw"]
        before = yaw.max_torque

        said = sup._tune({"what": "power", "joint": "left yaw", "step": 1})
        saved = json.loads(path.read_text())["joints"]["left_shoulder_yaw"]["max_torque"]
        up = yaw.max_torque
        while True:                                         # up to its ceiling, and no further
            try:
                sup._tune({"what": "power", "joint": "L yaw", "step": 1})
            except ValueError as exc:
                ceiling_said = str(exc)
                break
        top = yaw.max_torque
        undone = sup._tune({"what": "undo"})
        sup.motors = True                                   # as if armed
        try:
            sup._tune({"what": "flip", "joint": "left yaw"})
            flipped_armed = True
        except RuntimeError:
            flipped_armed = False
        sup.motors = False
        was_flipped = yaw.axis_flip
        flip_said = sup._tune({"what": "flip", "joint": "left_shoulder_yaw"})
        flipped = yaw.axis_flip != was_flipped
        sup._tune({"what": "undo"})
        scale_said = sup._tune({"what": "scale", "step": -1})
        try:
            sup._tune({"what": "power", "joint": "left knee", "step": 1})
            bad_joint = False
        except ValueError:
            bad_joint = True

        # the same over the port, as tools/teleop_tune.py and "agent, more power ..." send it
        reply = {}
        box = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        box.settimeout(0.5)
        box.sendto(json.dumps({"what": "gravity", "joint": "right pitch", "step": 1}).encode(),
                   ("127.0.0.1", tune_port))
        time.sleep(0.05)
        sup._take_tuning()
        try:
            reply = json.loads(box.recvfrom(65536)[0])
        except socket.timeout:
            pass
        box.close()

        # while driving: a joint pinned at its limit says what to do about it
        sup.state = "ARMED"
        sup.solver.held = [True, False]
        sup.health.tags[2] = "MAXED"
        sup.health.maxed_time[2] = 2.0
        hint = sup._armed_hint()
        sup.health.tags[2] = ""
        sup.health.maxed_time[2] = 0.0
        sup.reach_since = [sup.now - 1.0 if sup.now else -1.0, 0.0]
        sup.now = sup.reach_since[0] + 1.0
        sup.reach_cm = [14.0, 0.0]
        reach_hint = sup._armed_hint()

        if "1.5 -> 2.0" not in said.replace("L yaw power ", "") and not (up == before + 0.5):
            failures.append(f"'more power left yaw' did not add 0.5 N·m: {said}")
        elif saved != up:
            failures.append(f"the change was not saved to the profile ({saved} on disk, {up} live)")
        elif top > 4.0 + 1e-9 or "ceiling" not in ceiling_said:
            failures.append(f"power went past the yaw's 4 N·m ceiling ({top}) or said nothing: {ceiling_said}")
        elif abs(yaw.max_torque - (top - 0.5)) > 1e-9 or "undone" not in undone:
            failures.append(f"undo did not take back the last step: {yaw.max_torque} after {top}: {undone}")
        elif flipped_armed:
            failures.append("a joint was flipped while its motors were on")
        elif not flipped or "flipped" not in flip_said:
            failures.append(f"flip while stopped did not flip: {flip_said}")
        elif "->" not in scale_said or profile.motion_scale >= 0.71:
            failures.append(f"'less sensitive' did not lower the sensitivity: {scale_said}")
        elif not bad_joint:
            failures.append("a joint that does not exist was accepted")
        elif not reply.get("ok") or "gravity help" not in reply.get("said", "") or len(reply.get("joints", [])) != 10:
            failures.append(f"over the port, the change or its answer went wrong: {reply}")
        elif "more power left yaw" not in hint:
            failures.append(f"a maxed-out joint does not say how to fix it: {hint!r}")
        elif "out of reach" not in reach_hint and "past where the arm can reach" not in reach_hint:
            failures.append(f"a hand sent out of reach is not pointed out: {reach_hint!r}")
        else:
            print(f"ok    power +0.5 N·m, saved at once, stops at the ceiling ({top:.1f} N·m), undo takes it back")
            print("ok    a flip is refused while the motors hold, done while stopped; sensitivity and gravity too")
            print("ok    the port answers with the change and every joint's settings (teleop_tune.py show)")
            print(f"ok    while driving it says what to do: {hint!r}")
            print(f"ok    and when a hand is sent out of reach: {reach_hint!r}")

        # mirror mode moves like a reflection: toward the robot is its forward, your right
        # is its left, up is up (the operator faces it; hands swapped by the supervisor)
        import pinocchio as pin
        import numpy as np
        solver = rt.TeleopIkSolver()
        q0 = np.zeros(10)

        def moved(d_op, facing):
            solver.held = [False, False]
            solver.targets = [None, None]
            start = pin.SE3(np.eye(3), np.array([0.3, 0.2, 1.0]))
            end = pin.SE3(np.eye(3), start.translation + np.asarray(d_op, float))
            solver.update(q0, q0, [True, False], [start, None], 0.02, 1.0, facing=facing)
            before = solver.targets[0].translation.copy()
            for _ in range(60):                             # past the smoothing
                solver.update(q0, q0, [True, False], [end, None], 0.02, 1.0, facing=facing)
            return solver.targets[0].translation - before

        forward, sideways, up = moved([0.1, 0, 0], True), moved([0, 0.1, 0], True), moved([0, 0, 0.1], True)
        straight = moved([0.1, 0.1, 0], False)
        if not (forward[0] > 0.09 and sideways[1] < -0.09 and up[2] > 0.09):
            failures.append(f"mirror mode is not a reflection: forward {forward.round(3)}, "
                            f"sideways {sideways.round(3)}, up {up.round(3)}")
        elif not (straight[0] > 0.09 and straight[1] > 0.09):
            failures.append(f"straight mode no longer copies the motion: {straight.round(3)}")
        else:
            print("ok    mirror mode: reach toward the robot and its hand comes toward you; sideways reflects; "
                  "up is up; straight mode copies")

        # the calibration's zero question: arms far from the start-up zero are asked about
        from arm_calibration import CalInputs, PowerCalibration
        import numpy as np

        class Host:
            zeroed = 0

            def capture_zero(self):
                self.zeroed += 1

            def apply_signs(self):
                pass

            def joint_status(self, i):
                return ""

        def through_zero_question(q, last):
            host = Host()
            cal = PowerCalibration(profile, gravity, host)
            zeros = np.zeros(10)
            asked = []
            for press in [None, "a", "a", "a", "a", "a", last]:   # start, see it, all, full, zero, answer
                cal.step(CalInputs(now=0.0, dt=0.02, q=np.array(q, float), tau=zeros, g=zeros, miss=zeros,
                                   grip=False, a=press == "a", b=press == "b", hands=(None, None)))
                asked.append(cal.lines[0])
            return host.zeroed, asked

        bent = [0.09, 0.02, -0.22, -1.08, 0.0, 0.07, 0.57, 1.85, 0.79, -2.22]   # 30 Sep, 09:05
        kept, asked_bent = through_zero_question(bent, "b")
        taken, _ = through_zero_question(bent, "a")
        straight, asked_straight = through_zero_question([0.01] * 10, "a")
        if "ZERO: ARE THE ARMS HANGING?" not in asked_bent or kept != 0:
            failures.append(f"zero from a bent pose is not questioned, or B did not keep the old zero: {asked_bent}")
        elif taken != 1:
            failures.append("confirming the bent pose with A did not set zero")
        elif "ZERO: ARE THE ARMS HANGING?" in asked_straight or straight != 1:
            failures.append(f"arms already hanging are asked twice, or not zeroed: {asked_straight}")
        else:
            print("ok    zero from a pose far from the start-up zero asks again (B keeps it); hanging arms zero at once")

        # the audit: a joint that sags while told to lift is NOT called the wrong way round
        rows = [{"start": 0}]
        for n in range(80):
            t = n * 0.04
            told = 0.02 * n                                 # asked to lift steadily...
            rows.append({"t": t, "held": [True, False], "q_cmd": [told] + [0.0] * 9,
                         "q_meas": [-0.001 * n] + [0.0] * 9,  # ...and sagging a little instead
                         "tau": [3.0] + [0.0] * 9, "lim": [3.0] + [1.0] * 9})
        log = Path(tmp) / "sag.jsonl"
        log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        found = "\n".join(teleop_tune.audit(log, None))
        if "WRONG WAY" in found or "NOT ENOUGH POWER" not in found:
            failures.append(f"the audit misreads a sagging joint: {found}")
        else:
            print("ok    the audit calls a sagging, maxed-out joint 'not enough power', never 'wrong way'")
    print()
    for failure in failures:
        print("FAIL  " + failure)
    print("tuning is sound" if not failures else f"{len(failures)} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
