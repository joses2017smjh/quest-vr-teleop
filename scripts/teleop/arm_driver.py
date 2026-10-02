"""CAN driver for both arm buses with torque feed-forward, plus an offline twin.

Bimanual sends PDO-2 per joint and waits 1 ms for each reply. USB CAN latency
often exceeds that, which is where "No response from device N" came from.
ArmDriver sends PDO-3 [position target, torque feed-forward] to every joint on
a bus, then collects the [position, torque] replies before one shared deadline.

All arrays are in ARM_JOINTS order and the URDF frame:
    p_fw = zero_fw + sign * q        tau_fw = sign * tau
PDO-3 also feeds each controller's 1 s watchdog, so it is sent every cycle,
including while the motors are damping.
"""

from __future__ import annotations

import math
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import can
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from arm_common import (  # noqa: E402
    ArmBus, Function, Mode, Parameter, decode_error, mode_name,
)

from arm_power import ARM_JOINTS, N_JOINTS, GravityModel  # noqa: E402

DEVICE_ID_MSK = 0x7F
FUNC_ID_POS = 7
SAFE_START_MODES = {Mode.IDLE, Mode.DAMPING, Mode.POSITION}
ENCODER_FAULT = 0x2000
WATCHDOG_TIMEOUT = 0x0040


def describe_state(mode: int | None, error: int | None) -> str:
    err = decode_error(error)
    bits = ",".join((err or {}).get("bits") or []) or "none"
    return f"mode {mode_name(mode)}, errors {bits}"


@dataclass
class ArmControlBus(ArmBus):
    """ArmBus (USB-guarded, arm buses only) plus NMT mode, SDO float and PDO-3."""

    tx_errors: int = field(default=0, init=False)
    rx_errors: int = field(default=0, init=False)
    late: int = field(default=0, init=False)
    error_frame_count: int = field(default=0, init=False)

    def _send(self, device_id: int, func_id: int, data: bytes = b"") -> bool:
        try:
            self._bus.send(can.Message(
                arbitration_id=(func_id << FUNC_ID_POS) | device_id,
                is_extended_id=False,
                data=data,
            ), timeout=0.005)
        except can.CanError:
            self.tx_errors += 1
            return False
        self.tx_count += 1
        return True

    def drain(self) -> dict[int, tuple[float, float]]:
        """Empty the socket before a new batch. PDO-3 replies that missed an
        earlier deadline are still valid (one cycle old) and are returned."""
        late = {}
        while True:
            try:
                msg = self._bus.recv(timeout=0.0)
            except can.CanError:
                self.rx_errors += 1
                return late
            if msg is None:
                return late
            if msg.is_error_frame:
                self.error_frame_count += 1
                continue
            if (msg.arbitration_id >> FUNC_ID_POS) == Function.TRANSMIT_PDO_3 and len(msg.data) >= 8:
                late[msg.arbitration_id & DEVICE_ID_MSK] = struct.unpack("<ff", bytes(msg.data[:8]))
                self.late += 1

    def set_mode(self, device_id: int, mode: int) -> bool:
        return self._send(device_id, Function.NMT, struct.pack("<BB", mode, device_id))

    def write_f32(self, device_id: int, param_id: int, value: float) -> bool:
        return self._send(device_id, Function.RECEIVE_SDO,
                          struct.pack("<BHBf", 0x01 << 5, param_id, 0, value))

    def send_pdo3(self, device_id: int, position: float, torque: float) -> bool:
        return self._send(device_id, Function.RECEIVE_PDO_3, struct.pack("<ff", position, torque))

    def collect_pdo3(self, want: set[int], deadline: float) -> dict[int, tuple[float, float]]:
        got = {}
        while want:
            remain = deadline - time.perf_counter()
            if remain <= 0:
                break
            try:
                msg = self._bus.recv(timeout=remain)
            except can.CanError:
                self.rx_errors += 1
                break
            if msg is None:
                break
            if msg.is_error_frame:
                self.error_frame_count += 1
                continue
            dev = msg.arbitration_id & DEVICE_ID_MSK
            if (msg.arbitration_id >> FUNC_ID_POS) != Function.TRANSMIT_PDO_3 or dev not in want:
                continue
            if len(msg.data) < 8:
                continue
            got[dev] = struct.unpack("<ff", bytes(msg.data[:8]))
            want.discard(dev)
            self.rx_count += 1
        return got


class _ArmFrames:
    """Host-side joint frame shared by the real and simulated drivers."""

    def _init_frames(self):
        self.sign = np.array([j.axis_sign for j in ARM_JOINTS], dtype=float)
        self.zero_fw = np.zeros(N_JOINTS)
        self.p_fw = np.full(N_JOINTS, np.nan)
        self.tau_fw = np.zeros(N_JOINTS)
        self.miss = np.zeros(N_JOINTS, dtype=int)

    @property
    def q(self) -> np.ndarray:
        return self.sign * (self.p_fw - self.zero_fw)

    @property
    def tau(self) -> np.ndarray:
        return self.sign * self.tau_fw

    def capture_zero(self) -> None:
        """Make the current pose URDF zero (arms hanging straight down)."""
        if not np.all(np.isfinite(self.p_fw)):
            raise RuntimeError("cannot set zero: some joint positions are unknown")
        self.zero_fw = self.p_fw.copy()

    def set_signs(self, signs: np.ndarray) -> None:
        self.sign = np.asarray(signs, dtype=float).copy()


class ArmDriver(_ArmFrames):
    simulated = False

    def __init__(self, reply_timeout: float = 0.006):
        self.reply_timeout = reply_timeout
        self.buses: dict[str, ArmControlBus] = {}
        try:
            for channel in ("can0", "can1"):
                self.buses[channel] = ArmControlBus(channel)
        except BaseException:
            self.close()
            raise
        self._init_frames()
        self._by_bus = {
            channel: [(i, j.device_id) for i, j in enumerate(ARM_JOINTS) if j.bus == channel]
            for channel in self.buses
        }

    def close(self) -> None:
        for bus in self.buses.values():
            bus.close()

    def preflight(self) -> list[str]:
        """Read-only check of all ten joints. Returns problems; empty means ready."""
        problems = []
        for i, joint in enumerate(ARM_JOINTS):
            bus = self.buses[joint.bus]
            dev = joint.device_id
            where = f"{joint.label} ({joint.bus} ID {dev})"
            if not (bus.ping(dev, timeout=0.05) or bus.ping(dev, timeout=0.15)):
                problems.append(f"{where}: no reply to ping")
                print(f"  {where:24s} NO REPLY", flush=True)
                continue
            mode = bus.read_u32(dev, Parameter.MODE, timeout=0.08)
            err = decode_error(bus.read_u32(dev, Parameter.ERROR, timeout=0.08))
            gear = bus.read_f32(dev, Parameter.POSITION_CONTROLLER_GEAR_RATIO, timeout=0.08)
            vbus = bus.read_f32(dev, Parameter.POWERSTAGE_BUS_VOLTAGE_MEASURED, timeout=0.08)
            pos = bus.read_f32(dev, Parameter.POSITION_CONTROLLER_POSITION_MEASURED, timeout=0.08)
            print(f"  {where:24s} mode={mode_name(mode):9s} err={(err or {}).get('hex')} "
                  f"{(err or {}).get('bits') or ''} gear={gear} Vbus={vbus} pos={pos}", flush=True)
            if mode not in SAFE_START_MODES:
                problems.append(f"{where}: mode {mode_name(mode)} (DISABLED = stuck waiting for its encoder)")
            if err and err["critical"]:
                hint = "power-cycle the robot"
                if err["raw"] & ENCODER_FAULT:
                    i2c = bus.read_i2c_angle(dev, timeout=0.08)
                    hint = (f"its encoder returned {i2c} (valid 0-4095) and the firmware stops reading "
                            "the encoder after one bad frame, so only a power cycle clears it. If it then "
                            "shows mode DISABLED, reseat that joint's AS5600 cable")
                problems.append(f"{where}: error {err['hex']} {err['bits']} ({hint})")
            if gear is None or abs(abs(gear) - 15.0) > 0.5:
                problems.append(f"{where}: gear_ratio {gear}, expected -15 (tools/write_arm_config.py)")
            if vbus is None or vbus < 15.0:
                problems.append(f"{where}: bus voltage {vbus}")
            if pos is None or not math.isfinite(pos):
                problems.append(f"{where}: position unreadable")
            else:
                self.p_fw[i] = pos
        return problems

    def set_mode(self, mode: int) -> None:
        for joint in ARM_JOINTS:
            self.buses[joint.bus].set_mode(joint.device_id, mode)

    def write_gains(self, kp: np.ndarray, kd: np.ndarray) -> None:
        for i, joint in enumerate(ARM_JOINTS):
            bus = self.buses[joint.bus]
            bus.write_f32(joint.device_id, Parameter.POSITION_CONTROLLER_POSITION_KP, float(kp[i]))
            bus.write_f32(joint.device_id, Parameter.POSITION_CONTROLLER_VELOCITY_KP, float(kd[i]))
            time.sleep(0.002)

    def write_limits(self, limits: dict[int, float]) -> None:
        for i, value in limits.items():
            joint = ARM_JOINTS[i]
            self.buses[joint.bus].write_f32(
                joint.device_id, Parameter.POSITION_CONTROLLER_TORQUE_LIMIT, float(value))

    def exchange(self, q_target: np.ndarray, tau_ff: np.ndarray, now: float | None = None) -> np.ndarray:
        """One control cycle. Returns the mask of joints that replied in time."""
        p_target = self.zero_fw + self.sign * np.asarray(q_target, dtype=float)
        tau_fw = self.sign * np.asarray(tau_ff, dtype=float)
        heard = np.zeros(N_JOINTS, dtype=bool)
        for channel, bus in self.buses.items():
            late = bus.drain()
            for i, dev in self._by_bus[channel]:
                if dev in late:
                    self.p_fw[i], self.tau_fw[i] = late[dev]
                    heard[i] = True
                if math.isfinite(p_target[i]) and math.isfinite(tau_fw[i]):
                    bus.send_pdo3(dev, p_target[i], tau_fw[i])
        deadline = time.perf_counter() + self.reply_timeout
        fresh = np.zeros(N_JOINTS, dtype=bool)
        for channel, bus in self.buses.items():
            got = bus.collect_pdo3({dev for _, dev in self._by_bus[channel]}, deadline)
            for i, dev in self._by_bus[channel]:
                if dev in got:
                    self.p_fw[i], self.tau_fw[i] = got[dev]
                    fresh[i] = True
        # miss = cycles in a row with no reply at all, on time or late.
        self.miss = np.where(fresh | heard, 0, self.miss + 1)
        return fresh

    def read_state(self, i: int) -> tuple[int | None, int | None]:
        """(mode, error register) by SDO; None where the node did not answer."""
        joint = ARM_JOINTS[i]
        bus = self.buses[joint.bus]
        mode = bus.read_u32(joint.device_id, Parameter.MODE, timeout=0.03, flush=False)
        error = bus.read_u32(joint.device_id, Parameter.ERROR, timeout=0.03, flush=False)
        return mode, error

    def joint_status(self, i: int) -> str:
        return describe_state(*self.read_state(i))

    def stats(self) -> dict:
        return {
            channel: {"tx_err": bus.tx_errors, "rx_err": bus.rx_errors,
                      "late": bus.late, "err_frames": bus.error_frame_count}
            for channel, bus in self.buses.items()
        }


class SimArmDriver(_ArmFrames):
    """Offline twin of ArmDriver: firmware PD + feed-forward + torque clamp
    driving a joint-side model with payload gravity, friction and end stops.

    faults: {joint key: "stuck" | "weak" | "sticky" | "flipped" | "silent" | "encfault"}
    ("encfault" latches ENCODER_FAULT and drops to damping after 2 s in POSITION.)
    """

    simulated = True
    STEP = 0.002
    FILTER_ALPHA = 1.0 - (1.0 - 0.2696) ** 4  # firmware EMA at 2 kHz, per 2 ms step

    def __init__(self, gravity: GravityModel, payload_scale: float = 1.25,
                 faults: dict[str, str] | None = None, seed: int = 1):
        self.gravity = gravity
        self.payload_scale = payload_scale
        self.faults = dict(faults or {})
        self.reply_timeout = 0.0
        self._init_frames()
        kinds = [j.kind for j in ARM_JOINTS]
        base = np.array([j.axis_sign for j in ARM_JOINTS], dtype=float)
        self.true_sign = np.where([self.faults.get(j.key) == "flipped" for j in ARM_JOINTS], -base, base)
        self.silent = np.array([self.faults.get(j.key) == "silent" for j in ARM_JOINTS])
        self.stuck = np.array([self.faults.get(j.key) == "stuck" for j in ARM_JOINTS])
        # The firmware position at the hanging pose is arbitrary after a boot.
        self.zero_true = np.random.default_rng(seed).uniform(-0.2, 0.2, N_JOINTS)
        self.state_p = self.zero_true.copy()
        self.state_v = np.zeros(N_JOINTS)
        self.p_fw = self.state_p.copy()
        self.inertia = np.array([{"pitch": .03, "roll": .03, "yaw": .02, "elbow": .02, "wrist": .012}[k]
                                 for k in kinds])
        weak = np.array([self.faults.get(j.key) == "weak" for j in ARM_JOINTS])
        sticky = np.array([self.faults.get(j.key) == "sticky" for j in ARM_JOINTS])
        self.coulomb = np.where(weak, 4.5, 0.35)
        self.stiction = np.where(sticky, 1.9, self.coulomb + 0.3)
        self.viscous = 0.05
        self.kp = np.full(N_JOINTS, 50.0)
        self.kd = np.full(N_JOINTS, 2.0)
        self.limit = np.full(N_JOINTS, 1.0)
        self.modes = np.full(N_JOINTS, Mode.IDLE)
        self.errors = np.zeros(N_JOINTS, dtype=int)
        self.encfault = np.array([self.faults.get(j.key) == "encfault" for j in ARM_JOINTS])
        self.position_time = np.zeros(N_JOINTS)
        self.target = self.state_p.copy()
        self.tau_target = np.zeros(N_JOINTS)
        self.setpoint = np.zeros(N_JOINTS)
        self.last = None

    def close(self) -> None:
        pass

    def preflight(self) -> list[str]:
        print("  SIMULATED arms (no CAN). Faults:", self.faults or "none", flush=True)
        return []

    def set_mode(self, mode: int) -> None:
        # A latched encoder fault fails again on the first control tick.
        broken = self.silent | ((self.errors & ENCODER_FAULT) != 0)
        self.modes = np.where(broken & (mode != Mode.IDLE), Mode.DAMPING, mode)
        self.setpoint[:] = 0.0

    def write_gains(self, kp: np.ndarray, kd: np.ndarray) -> None:
        self.kp = np.asarray(kp, dtype=float).copy()
        self.kd = np.asarray(kd, dtype=float).copy()

    def write_limits(self, limits: dict[int, float]) -> None:
        for i, value in limits.items():
            self.limit[i] = value

    def _physics(self, span: float) -> None:
        in_position = self.modes == Mode.POSITION
        self.position_time = np.where(in_position, self.position_time + span, self.position_time)
        tripped = self.encfault & in_position & (self.position_time > 2.0)
        if tripped.any():
            self.modes = np.where(tripped, Mode.DAMPING, self.modes)
            self.errors = np.where(tripped, self.errors | ENCODER_FAULT | WATCHDOG_TIMEOUT, self.errors)
        steps = max(1, int(round(min(span, 0.1) / self.STEP)))
        h = min(span, 0.1) / steps
        lower, upper = self.gravity.lower - 0.05, self.gravity.upper + 0.05
        position = self.modes == Mode.POSITION
        damping = self.modes == Mode.DAMPING
        # Gravity changes slowly; once per cycle is enough here.
        q_phys = self.true_sign * (self.state_p - self.zero_true)
        load = self.true_sign * self.payload_scale * self.gravity.torque(q_phys)
        for _ in range(steps):
            cmd = self.kp * (self.target - self.state_p) - self.kd * self.state_v + self.tau_target
            filtered = self.FILTER_ALPHA * cmd + (1.0 - self.FILTER_ALPHA) * self.setpoint
            self.setpoint = np.where(position, np.clip(filtered, -self.limit, self.limit), 0.0)
            drive = np.where(damping, -3.0 * self.state_v, self.setpoint)
            net = drive - load - self.viscous * self.state_v
            moving = np.abs(self.state_v) > 1e-3
            held = ~moving & (np.abs(net) <= self.stiction)
            net = np.where(moving, net - self.coulomb * np.sign(self.state_v),
                           net - self.coulomb * np.sign(net))
            v_new = self.state_v + np.where(held, 0.0, net / self.inertia) * h
            reversed_ = moving & (np.sign(v_new) != np.sign(self.state_v))
            v_new = np.where(held | reversed_ | self.stuck, 0.0, v_new)
            self.state_v = v_new
            self.state_p = self.state_p + v_new * h
            q_phys = self.true_sign * (self.state_p - self.zero_true)
            hit = (q_phys < lower) | (q_phys > upper)
            if hit.any():
                clipped = self.zero_true + self.true_sign * np.clip(q_phys, lower, upper)
                self.state_p = np.where(hit, clipped, self.state_p)
                self.state_v = np.where(hit, 0.0, self.state_v)

    def exchange(self, q_target: np.ndarray, tau_ff: np.ndarray, now: float | None = None) -> np.ndarray:
        now = time.perf_counter() if now is None else now
        if self.last is not None:
            self._physics(now - self.last)
        self.last = now
        p_target = self.zero_fw + self.sign * np.asarray(q_target, dtype=float)
        tau_fw = self.sign * np.asarray(tau_ff, dtype=float)
        ok = np.isfinite(p_target) & np.isfinite(tau_fw) & ~self.silent
        self.target = np.where(ok, p_target, self.target)
        self.tau_target = np.where(ok, tau_fw, self.tau_target)
        fresh = ~self.silent
        self.p_fw = np.where(fresh, self.state_p, self.p_fw)
        self.tau_fw = np.where(fresh, self.setpoint, self.tau_fw)
        self.miss = np.where(fresh, 0, self.miss + 1)
        return fresh

    def read_state(self, i: int) -> tuple[int | None, int | None]:
        if self.silent[i]:
            return None, None
        return int(self.modes[i]), int(self.errors[i])

    def joint_status(self, i: int) -> str:
        fault = self.faults.get(ARM_JOINTS[i].key)
        return f"{describe_state(*self.read_state(i))}, sim fault {fault or 'none'}"

    def stats(self) -> dict:
        return {"sim": {"payload_scale": self.payload_scale}}
