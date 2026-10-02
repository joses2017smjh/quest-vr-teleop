"""Shared helpers for ARM-ONLY validation of the Berkeley Humanoid Lite.

Scope:
  * Only ``can0`` (left arm) and ``can1`` (right arm) may be opened. Legs live
    on separate USB adapters and are never contacted.
  * USB port binding is checked so a kernel rename cannot silently point these
    tools at a leg bus.

Transport:
  * SDO reads of the MotorController struct (byte offset = Parameter id).
  * PDO-1 ping (echo 0xCA). Ping also resets the firmware safety watchdog
    (htim2); that is intentional here so a node already in a closed-loop mode
    is not dropped to DAMPING mid-sample. These tools still never write a
    parameter, never send NMT/mode, and never store flash.
"""

from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import time
from dataclasses import dataclass, field

import can

from berkeley_humanoid_lite_lowlevel.recoil import Function, Mode, Parameter  # noqa: F401

# --------------------------------------------------------------------------
# Arm-only guard
# --------------------------------------------------------------------------
ARM_CHANNELS = ("can0", "can1")

# USB port -> interface, verified 2026-09-12 and re-verified live.
# Source: scripts/hardware/start_can_by_port.sh lines 13-18.
EXPECTED_USB_PORT = {"can0": "3-4.1:1.0", "can1": "3-4.2:1.0"}
LEG_USB_PORT = {"left_leg": "3-4.4:1.0", "right_leg": "3-4.3:1.0"}


def assert_arm_channel(channel: str) -> None:
    if channel not in ARM_CHANNELS:
        raise SystemExit(
            f"REFUSED: {channel!r} is not an arm bus. This tool only ever touches "
            f"{ARM_CHANNELS} (can0=left arm, can1=right arm). The legs live on "
            f"separate USB adapters ({LEG_USB_PORT}) and are out of scope."
        )


def verify_usb_binding(channel: str) -> tuple[bool, str]:
    """Confirm the interface name still maps to the expected physical adapter."""
    link = f"/sys/class/net/{channel}/device"
    if not os.path.exists(link):
        return False, "no such interface"
    actual = os.path.basename(os.path.realpath(link))
    want = EXPECTED_USB_PORT[channel]
    return actual == want, actual


# --------------------------------------------------------------------------
# Firmware error table -- authoritative, from motor_controller_conf.h:141-157
#
# recoil/core.py ErrorCode is STALE: it omits ERROR_CAN_RX_FAULT, so bits from
# 0x0400 up are shifted and 0x1000/0x2000 are missing.
# --------------------------------------------------------------------------
FW_ERROR_BITS = {
    0x0001: "GENERAL",
    0x0002: "ESTOP",
    0x0004: "INITIALIZATION_ERROR",
    0x0008: "CALIBRATION_ERROR",
    0x0010: "POWERSTAGE_ERROR",
    0x0020: "INVALID_MODE",
    0x0040: "WATCHDOG_TIMEOUT",
    0x0080: "OVER_VOLTAGE",
    0x0100: "OVER_CURRENT",
    0x0200: "OVER_TEMPERATURE",
    0x0400: "CAN_RX_FAULT",
    0x0800: "CAN_TX_FAULT",
    0x1000: "I2C_FAULT",
    0x2000: "ENCODER_FAULT",
}
FW_KNOWN_ERROR_MASK = 0x3FFF

CRITICAL_ERROR_MASK = (
    0x0002 | 0x0004 | 0x0008 | 0x0010 | 0x0100 | 0x0200 | 0x1000 | 0x2000
)

MODE_NAMES = {
    0x00: "DISABLED", 0x01: "IDLE", 0x02: "DAMPING", 0x05: "CALIBRATION",
    0x10: "CURRENT", 0x11: "TORQUE", 0x12: "VELOCITY", 0x13: "POSITION",
    0x20: "VABC_OVERRIDE", 0x21: "VALPHABETA_OVERRIDE", 0x22: "VQD_OVERRIDE",
    0x80: "DEBUG",
}
# Powerstage not tracking a setpoint. Reported, never used as a sampling gate.
SAFE_MODES = {0x00, 0x01, 0x02}


def decode_error(error: int | None) -> dict | None:
    if error is None:
        return None
    known = [n for b, n in sorted(FW_ERROR_BITS.items()) if error & b]
    unknown = error & ~FW_KNOWN_ERROR_MASK
    return {
        "raw": error,
        "hex": f"0x{error:04X}",
        "bits": known,
        "unknown_mask": f"0x{unknown:04X}" if unknown else None,
        "critical": bool(error & CRITICAL_ERROR_MASK),
    }


def mode_name(mode: int | None) -> str:
    if mode is None:
        return "UNREADABLE"
    return MODE_NAMES.get(mode, f"UNKNOWN(0x{mode:02X})")


# --------------------------------------------------------------------------
# Joint maps
# --------------------------------------------------------------------------
# Docs: https://berkeley-humanoid-lite.gitbook.io/docs/in-depth-contents/joint-id-mapping
# Repo: bimanual.py:20-31, humanoid.py:38-48, robot_configuration.backup.json
REPO_ARM_JOINTS = {
    "can0": {
        1: "left_shoulder_pitch_joint",
        3: "left_shoulder_roll_joint",
        5: "left_shoulder_yaw_joint",
        7: "left_elbow_joint",
        9: "left_wrist_yaw_joint",
    },
    "can1": {
        2: "right_shoulder_pitch_joint",
        4: "right_shoulder_roll_joint",
        6: "right_shoulder_yaw_joint",
        8: "right_elbow_joint",
        10: "right_wrist_yaw_joint",
    },
}

# As-built on this robot (right wrist ESC answers as ID 12, not docs ID 10).
AS_BUILT_ARM_JOINTS = {
    "can0": dict(REPO_ARM_JOINTS["can0"]),
    "can1": {
        2: "right_shoulder_pitch_joint",
        4: "right_shoulder_roll_joint",
        6: "right_shoulder_yaw_joint",
        8: "right_elbow_joint",
        12: "right_wrist_yaw_joint",
    },
}

LEG_IDS = {
    "left_leg": {1: "left_hip_roll", 3: "left_hip_yaw", 5: "left_hip_pitch",
                 7: "left_knee_pitch", 11: "left_ankle_pitch", 13: "left_ankle_roll"},
    "right_leg": {2: "right_hip_roll", 4: "right_hip_yaw", 6: "right_hip_pitch",
                  8: "right_knee_pitch", 12: "right_ankle_pitch", 14: "right_ankle_roll"},
}

BUS_LIMB = {"can0": "left arm", "can1": "right arm"}

# Wide enough to catch leftover foot-board IDs (10, 11, 13, 14) without
# walking the whole 7-bit space.
SCAN_RANGE = range(1, 17)

CPR = 4096


def joint_name(channel: str, dev_id: int) -> str:
    return AS_BUILT_ARM_JOINTS[channel].get(
        dev_id, REPO_ARM_JOINTS[channel].get(dev_id, "UNMAPPED")
    )


def target_ids(channel: str) -> list[int]:
    """IDs we always sample, even if the scan has not run yet."""
    return sorted(set(REPO_ARM_JOINTS[channel]) | set(AS_BUILT_ARM_JOINTS[channel]))


# --------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------
DEVICE_ID_MSK = 0x7F
FUNC_ID_POS = 7


@dataclass
class ArmBus:
    """Recoil transport restricted to the arm buses. Read + ping only."""

    channel: str
    bitrate: int = 1000000
    _bus: can.BusABC = field(init=False, repr=False)
    tx_count: int = field(default=0, init=False)
    rx_count: int = field(default=0, init=False)
    timeout_count: int = field(default=0, init=False)
    error_frames: list = field(default_factory=list, init=False)

    def __post_init__(self):
        assert_arm_channel(self.channel)
        ok, actual = verify_usb_binding(self.channel)
        if not ok:
            raise SystemExit(
                f"REFUSED: {self.channel} is on USB {actual}, expected "
                f"{EXPECTED_USB_PORT[self.channel]}. Interface names can swap across "
                f"reboots; refusing to guess which limb is on this wire."
            )
        self.usb_port = actual
        self._bus = can.interface.Bus(
            interface="socketcan", channel=self.channel, bitrate=self.bitrate
        )

    def close(self):
        try:
            self._bus.shutdown()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def flush(self):
        """Drain stale/unsolicited frames so a reply cannot be misattributed."""
        while True:
            try:
                m = self._bus.recv(timeout=0.0)
            except can.CanError as e:
                raise SystemExit(
                    f"{self.channel} is not usable ({e}). Bring it up first:\n"
                    f"  sudo ip link set {self.channel} up type can bitrate 1000000"
                ) from e
            if m is None:
                return
            if m.is_error_frame:
                self.error_frames.append((time.time(), m.arbitration_id, bytes(m.data)))

    def _send(self, device_id: int, func_id: int, data: bytes = b""):
        self._bus.send(can.Message(
            arbitration_id=(func_id << FUNC_ID_POS) | device_id,
            is_extended_id=False,
            data=data,
        ))
        self.tx_count += 1

    def _recv(self, device_id: int, func_id: int, timeout: float):
        deadline = time.time() + timeout
        while True:
            remain = deadline - time.time()
            if remain <= 0:
                self.timeout_count += 1
                return None
            m = self._bus.recv(timeout=remain)
            if m is None:
                continue
            if m.is_error_frame:
                self.error_frames.append((time.time(), m.arbitration_id, bytes(m.data)))
                continue
            if (m.arbitration_id & DEVICE_ID_MSK) != device_id:
                continue
            if (m.arbitration_id >> FUNC_ID_POS) != func_id:
                continue
            self.rx_count += 1
            return bytes(m.data)

    def ping(self, device_id: int, timeout: float = 0.05) -> bool:
        """PDO-1 echo. Also resets the node's safety watchdog (htim2)."""
        self.flush()
        self._send(device_id, Function.RECEIVE_PDO_1, b"\xCA")
        data = self._recv(device_id, Function.TRANSMIT_PDO_1, timeout)
        return bool(data) and len(data) >= 1 and data[0] == 0xCA

    def read_raw(self, device_id: int, param_id: int, timeout: float = 0.05,
                 flush: bool = True):
        """SDO upload. Returns 4 raw bytes or None."""
        if flush:
            self.flush()
        self._send(device_id, Function.RECEIVE_SDO, struct.pack("<BH", 0x02 << 5, param_id))
        data = self._recv(device_id, Function.TRANSMIT_SDO, timeout)
        if data is None or len(data) < 4:
            return None
        return data[0:4]

    def read_u32(self, device_id: int, param_id: int, timeout: float = 0.05,
                 flush: bool = True):
        d = self.read_raw(device_id, param_id, timeout, flush=flush)
        return None if d is None else struct.unpack("<L", d)[0]

    def read_i32(self, device_id: int, param_id: int, timeout: float = 0.05,
                 flush: bool = True):
        d = self.read_raw(device_id, param_id, timeout, flush=flush)
        return None if d is None else struct.unpack("<l", d)[0]

    def read_f32(self, device_id: int, param_id: int, timeout: float = 0.05,
                 flush: bool = True):
        d = self.read_raw(device_id, param_id, timeout, flush=flush)
        return None if d is None else struct.unpack("<f", d)[0]

    def read_u16(self, device_id: int, param_id: int, timeout: float = 0.05,
                 flush: bool = True):
        """Low 16 bits of an SDO word (position_raw lives here with 2 pad bytes)."""
        d = self.read_raw(device_id, param_id, timeout, flush=flush)
        return None if d is None else struct.unpack("<H", d[0:2])[0]

    def read_i2c_angle(self, device_id: int, timeout: float = 0.05, flush: bool = True):
        """AS5600 ANGLE register as the firmware packed it: (buf[0]<<8)|buf[1].

        Do NOT 12-bit mask before the >= cpr check -- encoder.c:55 uses the
        unmasked 16-bit value, and that is exactly ERROR_ENCODER_FAULT.
        """
        d = self.read_raw(device_id, Parameter.ENCODER_I2C_BUFFER, timeout, flush=flush)
        if d is None:
            return None
        return (d[0] << 8) | d[1]

    def read_status(self, device_id: int) -> dict:
        mode = self.read_u32(device_id, Parameter.MODE)
        err = self.read_u32(device_id, Parameter.ERROR)
        wd = self.read_u32(device_id, Parameter.WATCHDOG_TIMEOUT)
        return {
            "mode_raw": mode,
            "mode": mode_name(mode),
            "mode_safe": (mode in SAFE_MODES) if mode is not None else False,
            "error": decode_error(err),
            "watchdog_timeout": wd,
        }

    def snapshot_live(self, device_id: int, flush: bool = True) -> dict:
        """One-shot dump of everything that tells us whether the encoder is alive."""
        def u32(p):
            return self.read_u32(device_id, p, flush=flush)

        def i32(p):
            return self.read_i32(device_id, p, flush=flush)

        def f32(p):
            return self.read_f32(device_id, p, flush=flush)

        raw16 = self.read_u16(device_id, Parameter.ENCODER_POSITION_RAW, flush=flush)
        i2c = self.read_i2c_angle(device_id, flush=False)
        return {
            "device_id": u32(Parameter.DEVICE_ID),
            "firmware_version": u32(Parameter.FIRMWARE_VERSION),
            "mode_raw": u32(Parameter.MODE),
            "error_raw": u32(Parameter.ERROR),
            "update_counter": u32(Parameter.POSITION_CONTROLLER_UPDATE_COUNTER),
            "gear_ratio": f32(Parameter.POSITION_CONTROLLER_GEAR_RATIO),
            "bus_voltage": f32(Parameter.POWERSTAGE_BUS_VOLTAGE_MEASURED),
            "pole_pairs": i32(Parameter.MOTOR_POLE_PAIRS),
            "encoder_cpr": i32(Parameter.ENCODER_CPR),
            "encoder_position_raw": raw16,
            "encoder_i2c_reading": i2c,
            "encoder_n_rotations": i32(Parameter.ENCODER_N_ROTATIONS),
            "encoder_position": f32(Parameter.ENCODER_POSITION),
            "encoder_velocity": f32(Parameter.ENCODER_VELOCITY),
            "encoder_flux_offset": f32(Parameter.ENCODER_FLUX_OFFSET),
            "position_measured": f32(Parameter.POSITION_CONTROLLER_POSITION_MEASURED),
            "velocity_measured": f32(Parameter.POSITION_CONTROLLER_VELOCITY_MEASURED),
            "i_a": f32(Parameter.CURRENT_CONTROLLER_I_A_MEASURED),
            "i_b": f32(Parameter.CURRENT_CONTROLLER_I_B_MEASURED),
            "i_c": f32(Parameter.CURRENT_CONTROLLER_I_C_MEASURED),
        }

    def probe(self, device_id: int, attempts: int = 3, timeout: float = 0.05) -> dict:
        """Discover a node by ping AND SDO of DEVICE_ID. Never skip a miss."""
        pings, sdos = [], []
        for _ in range(attempts):
            pings.append(self.ping(device_id, timeout))
            sdos.append(self.read_u32(device_id, Parameter.DEVICE_ID, timeout))
        ok_sdo = [r for r in sdos if r is not None]
        return {
            "addressed_id": device_id,
            "online": any(pings) or bool(ok_sdo),
            "ping_ok": sum(1 for p in pings if p),
            "ping_attempts": attempts,
            "sdo_replies": sdos,
            "sdo_reply_rate": f"{len(ok_sdo)}/{attempts}",
            "self_reported_id": ok_sdo[0] if ok_sdo else None,
            "id_consistent": bool(ok_sdo) and all(r == device_id for r in ok_sdo),
        }


def link_stats(channel: str) -> dict:
    out = subprocess.run(
        ["ip", "-json", "-details", "-statistics", "link", "show", "dev", channel],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)[0]


def link_error_summary(st: dict) -> dict:
    x = st.get("linkinfo", {}).get("info_xstats", {}) or {}
    s = st.get("stats64", {})
    return {
        "state": st.get("linkinfo", {}).get("info_data", {}).get("state"),
        "bitrate": st.get("linkinfo", {}).get("info_data", {}).get("bittiming", {}).get("bitrate"),
        "operstate": st.get("operstate"),
        "restarts": x.get("restarts"), "bus_error": x.get("bus_error"),
        "arbitration_lost": x.get("arbitration_lost"), "error_warning": x.get("error_warning"),
        "error_passive": x.get("error_passive"), "bus_off": x.get("bus_off"),
        "rx_packets": s.get("rx", {}).get("packets"), "rx_errors": s.get("rx", {}).get("errors"),
        "tx_packets": s.get("tx", {}).get("packets"), "tx_errors": s.get("tx", {}).get("errors"),
    }


def jsonable(obj):
    """JSON dump helper: inf/nan become strings so encoder NaNs are not dropped."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj
