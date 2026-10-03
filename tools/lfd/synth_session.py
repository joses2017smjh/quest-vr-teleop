"""Write a synthetic recording session that follows docs/LFD_RECORDING_FORMAT.md exactly, with known answers.

The converter, the validator and their tests need recordings whose truth is known. Each synthetic episode has:
  * tap rows at about 48 Hz with jitter, and camera frames at about 30 Hz with jitter, on one monotonic clock that
    starts near 1e5 s (a robot PC's uptime: float32 cannot hold it to the millisecond, which catches that bug);
  * a smooth commanded trajectory qc(t) per joint, and a measured one that lags it by a known delay:
    q(t) = qc(t - delay). Hand-guided episodes command a stiff hold instead, as the spec warns qc would be;
  * claw pulses that change at known times, including a claw that is not commanded at first and pulses beyond
    both limits (they clip);
  * 1280x480 side-by-side JPEG frames with a red marker in the left eye's upper left, a blue one in the right eye's
    lower right, and the frame's index as a row of black and white blocks in the left eye. The whole frame is stored
    upside down, as the rig's camera sees it, so the converter's 180-degree turn and eye split can be checked.

JPEGs are encoded with cv2 when it is installed, else PIL, so this runs in the rig's env and the LeRobot one.
make_episode() builds the same episode in memory, without images, for numpy-only tests.

  python tools/lfd/synth_session.py /scratch/$USER/lfd/synth                    # one teleop, one guided episode
  python tools/lfd/synth_session.py OUT --episodes teleop,teleop,guided,policy --seconds 6 --seed 3
  python tools/lfd/synth_session.py OUT --frame-gap 1.0:0.2 --zero-change 2.5 --no-zero-check --outcome failure
  python tools/lfd/synth_session.py OUT --joint-names renamed --size 2560x720 --delay 0.12 --camera-latency 0.03
  python tools/lfd/synth_session.py OUT --seconds 60 --frame-hz 29.97 --frame-jitter 0.002 --same-mtime 100
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels  # noqa: E402

# ARM_JOINTS order and names, as scripts/teleop/arm_power.py has them (that module needs pinocchio, so not imported)
DEFAULT_JOINT_NAMES = (
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_yaw",
)
DEFAULT_URDF_JOINT_NAMES = (
    "arm_left_shoulder_pitch_joint", "arm_left_shoulder_roll_joint", "arm_left_shoulder_yaw_joint",
    "arm_left_elbow_pitch_joint", "arm_left_elbow_roll_joint", "arm_right_shoulder_pitch_joint",
    "arm_right_shoulder_roll_joint", "arm_right_shoulder_yaw_joint", "arm_right_elbow_pitch_joint",
    "arm_right_elbow_roll_joint",
)
RENAMED_JOINT_NAMES = ("L pitch", "L roll", "L yaw", "L elbow", "L wrist", "R pitch", "R roll", "R yaw", "R elbow",
                       "R wrist")
CLAW_LIMITS = {"left": {"open_us": 1000, "closed_us": 1640}, "right": {"open_us": 1000, "closed_us": 1670}}
TORQUE_LIMITS = (5.0, 5.0, 1.5, 2.5, 1.25, 5.0, 5.0, 1.5, 2.5, 1.5)
FRAME_SIZE = (1280, 480)
MONO_BASE = 104857.6                 # monotonic s at the first episode's start
WALL_MINUS_MONO = 1_790_000_000.0    # time.time() - time.monotonic() on the synthetic robot PC
EPISODE_GAP_S = 5.0
ZERO_FW = np.round(np.linspace(-2.5, 2.7, 10), 4)
ZERO_JUMP = 0.2618                   # rad on L roll when the synthetic zero is retaken mid-episode
CAM_ANGLE = (0.0, -12.0)             # commanded pan, tilt, degrees

# The commanded trajectory: per joint offset + amplitude * sin(2 pi f t + phase), rad. Every joint moves.
OFFSET = np.array([-0.30, 0.20, 0.00, 0.60, 0.00, -0.30, -0.20, 0.00, 0.60, 0.00])
AMPLITUDE = np.array([0.50, 0.30, 0.25, 0.45, 0.35, 0.40, 0.28, 0.22, 0.50, 0.30])
FREQ_HZ = np.array([0.35, 0.50, 0.65, 0.42, 0.80, 0.38, 0.55, 0.70, 0.45, 0.75])
PHASE = np.linspace(0.0, 2.5, 10)

# (seconds after start, claw 0 = robot-left / 1 = robot-right, pulse in us). The left claw is not commanded for
# the first 0.5 s; 1700 us is past the right claw's closed limit and 900 us below its open one, so both clip.
CLAW_EVENTS = ((0.0, 1, 1000), (0.5, 0, 1000), (1.5, 0, 1640), (2.0, 1, 1700), (3.0, 0, 1320), (3.5, 1, 900))

LEFT_MARKER_RGB = (255, 0, 0)
RIGHT_MARKER_RGB = (0, 0, 255)
BACKGROUND = 96
CODE_BITS = 10


@dataclasses.dataclass
class EpisodeSpec:
    """One synthetic episode. Times are seconds after the episode's start."""

    mode: str = "teleop"
    seconds: float = 4.0
    task: str = "Put the foam block in the bowl"
    outcome: str = "success"
    delay: float = 0.08                  # q(t) = qc(t - delay)
    row_hz: float = 48.0
    row_jitter: float = 0.003            # +- s on each row interval
    frame_hz: float = 30.0
    frame_jitter: float = 0.002          # +- s on each frame time
    frame_start: float = 0.012           # the first frame comes this long after the first row
    frame_gaps: tuple = ()               # ((start, duration), ...): no frames arrive in these windows
    same_mtime: tuple = ()               # frame positions k whose mtime (so t_cap) repeats frame k-1's: two frames
                                         # written within one tick of the kernel's coarse file clock
    zero_change_at: float | None = None  # the zero is retaken here; the recorder then aborts the episode
    zero_check: bool = True              # False writes "zero_check": null
    zero_convention: str = "hanging=0"
    claw_events: tuple = CLAW_EVENTS
    claw_limits: dict | None = None      # None: CLAW_LIMITS
    servos: bool = True                  # False: no camera servos, so cam and cam_mode are null
    cam_mode: str = "still"
    interventions: tuple = ()            # ((start, end), ...) with intervention true (and src teleop in a policy
                                         # episode: the operator took over)
    discarded: bool = False              # written under _discarded/, as the "discard" command leaves it
    policy: str | None = None
    notes: str = ""
    trial: int | None = None             # the evaluation fields a start command or a prime sets
    arrangement: str | None = None
    operator: str | None = None


def command(t_rel: np.ndarray) -> np.ndarray:
    """qc at times after the episode's start: (N, 10) rad."""
    t_rel = np.asarray(t_rel, dtype=float)[:, None]
    return OFFSET + AMPLITUDE * np.sin(2 * np.pi * FREQ_HZ * t_rel + PHASE)


def measured(t_rel: np.ndarray, delay: float) -> np.ndarray:
    """q at times after the episode's start: the command, late by delay."""
    return command(np.asarray(t_rel, dtype=float) - delay)


def claw_pulse(events, side: int, t_rel: float) -> int | None:
    """The last pulse sent to a claw at or before t_rel, None before the first."""
    sent = None
    for when, which, pulse in sorted(events):
        if which == side and when <= t_rel + 1e-12:
            sent = int(pulse)
    return sent


def default_zero_check() -> dict:
    """A configs/zero_check.json as tools/lfd/zero_check.py writes it, for arms hanging 15 deg out: "rejected"."""
    return {"format": "bhl-zero-check", "version": 1, "date": "2026-10-03T12:00:00", "method": "inclinometer",
            "readings_deg": {"torso": 0.0, "left_upper_arm": 15.2, "right_upper_arm": 14.6},
            "relative_to_torso_deg": {"left": 15.2, "right": 14.6},
            "decisions": {"left": "rejected", "right": "rejected"}, "decision": "rejected",
            "zero_fix_applied": False, "notes": "synthetic"}


def _round(values, digits: int = 6) -> list:
    return [round(float(x), digits) for x in values]


def _pose(x: float, y: float, z: float) -> list:
    return [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, y, 0.0, 0.0, 1.0, z, 0.0, 0.0, 0.0, 1.0]


def episode_records(spec: EpisodeSpec, index: int, session_id: str, t_start: float, seq_start: int,
                    rng: np.random.Generator, joint_names=DEFAULT_JOINT_NAMES) -> tuple[dict, list, list]:
    """episode.json, the tap rows and the frame index (without off/len) of one synthetic episode."""
    n_joints = len(joint_names)
    if n_joints != 10:
        raise ValueError("the synthetic trajectory has 10 joints")
    # rows: ~48 Hz with jitter, monotonic by construction
    intervals = 1.0 / spec.row_hz + rng.uniform(-spec.row_jitter, spec.row_jitter, int(spec.seconds * spec.row_hz)
                                                + 8)
    t_rel = np.concatenate([[0.0], np.cumsum(intervals)])
    t_rel = t_rel[t_rel <= spec.seconds]
    end_reason, outcome, zero_changed = "operator", spec.outcome, False
    if spec.zero_change_at is not None and spec.zero_change_at < t_rel[-1]:
        last = int(np.searchsorted(t_rel, spec.zero_change_at, side="left"))
        t_rel = t_rel[:last + 1]                         # the recorder aborts at the first row with the new zero
        end_reason, outcome, zero_changed = "zero changed", "aborted", True
    q = measured(t_rel, spec.delay)
    qc = command(t_rel) if spec.mode != "guided" else np.repeat(measured(t_rel[:1], spec.delay), len(t_rel), 0)
    src = {"teleop": "teleop", "guided": "guide", "policy": "policy"}.get(spec.mode, spec.mode)
    rows = []
    for k, tr in enumerate(t_rel):
        zero = ZERO_FW.copy()
        if zero_changed and k == len(t_rel) - 1:
            zero[1] += ZERO_JUMP
        tau = 0.4 * np.sin(q[k] + k * 0.01)
        interv = any(a <= tr < b for a, b in spec.interventions)
        rows.append({
            "v": labels.TAP_VERSION, "seq": seq_start + k, "t": round(t_start + tr, 6), "state": "ARMED",
            "motors": True, "src": "teleop" if interv and spec.mode == "policy" else src,   # the operator took over
            "q": _round(q[k]), "qc": _round(qc[k]), "tau": _round(tau, 4),
            "tff": _round(0.5 * np.cos(q[k]), 4), "lim": list(TORQUE_LIMITS), "zero": _round(zero, 4),
            "held": [spec.mode == "teleop"] * 2, "grip": [spec.mode == "teleop"] * 2, "trig": [0.0, 0.0],
            "hand": [_pose(0.2, 0.3, 1.1), _pose(0.2, -0.3, 1.1)] if spec.mode == "teleop" else [None, None],
            "head": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "tgt": [[0.15, 0.2, 0.0], [0.15, -0.2, 0.0]] if spec.mode == "teleop" else [None, None],
            "swap": False, "scale": 0.61,
            "claw": [claw_pulse(spec.claw_events, 0, tr), claw_pulse(spec.claw_events, 1, tr)],
            "cam": list(CAM_ANGLE) if spec.servos else None, "cam_mode": spec.cam_mode if spec.servos else None,
            "intervention": bool(interv),
        })
    # frames: ~30 Hz with jitter (kept below half a period, so they stay in order), minus the gaps
    n_frames = int(np.floor((t_rel[-1] - spec.frame_start) * spec.frame_hz)) + 1
    f_rel = spec.frame_start + np.arange(n_frames) / spec.frame_hz
    f_rel = f_rel + rng.uniform(-spec.frame_jitter, spec.frame_jitter, n_frames)
    f_rel = f_rel[(f_rel >= 0) & (f_rel <= t_rel[-1])]
    for start, duration in spec.frame_gaps:
        f_rel = f_rel[~((f_rel >= start) & (f_rel < start + duration))]
    frames = []
    for i, fr in enumerate(f_rel):
        t_cap = t_start + fr
        frames.append({"i": i, "off": 0, "len": 0, "t_cap": round(t_cap, 6),
                       "t_seen": round(t_cap + float(rng.uniform(0.001, 0.006)), 6),
                       "mtime_ns": int(round((t_cap + WALL_MINUS_MONO) * 1e9))})
    for k in spec.same_mtime:
        if not 1 <= k < len(frames):
            raise ValueError(f"same_mtime position {k} is not a frame after the first (there are {len(frames)})")
        frames[k]["t_cap"], frames[k]["mtime_ns"] = frames[k - 1]["t_cap"], frames[k - 1]["mtime_ns"]
    row_t = np.array([r["t"] for r in rows])
    frame_t = np.array([f["t_cap"] for f in frames]) if frames else np.zeros(0)
    t_end = float(max(row_t[-1], frame_t[-1] if len(frame_t) else row_t[-1]))
    meta = {
        "format": labels.EPISODE_FORMAT, "version": labels.FORMAT_VERSION, "episode_index": index,
        "session_id": session_id, "task": spec.task, "mode": spec.mode, "policy": spec.policy,
        "operator": spec.operator, "arrangement": spec.arrangement, "trial": spec.trial,
        "start": {"t": round(t_start, 6), "wall": round(t_start + WALL_MINUS_MONO, 3)},
        "end": {"t": round(t_end, 6), "wall": round(t_end + WALL_MINUS_MONO, 3)},
        "outcome": outcome, "end_reason": end_reason, "discarded": bool(spec.discarded), "notes": spec.notes,
        "counts": {"rows": len(rows), "frames": len(frames),
                   "row_gaps_over_50ms": int((np.diff(row_t) > 0.050).sum()),
                   "frame_gaps_over_67ms": int((np.diff(frame_t) > 0.067).sum()) if len(frame_t) > 1 else 0},
        "cam_mode_at_start": spec.cam_mode if spec.servos else None,
        "cam_at_start": list(CAM_ANGLE) if spec.servos else None,
        "zero_at_start": _round(ZERO_FW, 4), "zero_changed": zero_changed, "zero_convention": spec.zero_convention,
        "zero_check": default_zero_check() if spec.zero_check else None,
        "profile_sha256": hashlib.sha256(b"synthetic arm_power_profile.json").hexdigest(),
        "claw_limits": spec.claw_limits or CLAW_LIMITS, "warnings": [],
    }
    return meta, rows, frames


def session_record(session_id: str, joint_names, camera_latency_s: float | None = None,
                   created_wall: float = 0.0) -> dict:
    """session.json, key for key as the spec lists it (the frame size lives only in the JPEG headers)."""
    names = tuple(joint_names)
    urdf = DEFAULT_URDF_JOINT_NAMES if names == DEFAULT_JOINT_NAMES else tuple(f"{n}_joint" for n in names)
    return {
        "format": labels.SESSION_FORMAT, "version": labels.FORMAT_VERSION, "session_id": session_id,
        "created_wall": created_wall, "host": "synth", "repo_commit": None, "joint_names": list(names),
        "urdf_joint_names": list(urdf),
        "camera": {"frame_file": "/dev/shm/bhl_camera.jpg", "layout": "side_by_side", "rotated_180": True,
                   "calibration": None, "camera_latency_s": camera_latency_s},
        "tap_port": 11014, "control_port": 11015,
    }


def session_start(seed: int) -> tuple[float, float, str]:
    """(monotonic start, created_wall, session_id) of the synthetic session for this seed."""
    t_start = MONO_BASE + 3600.0 * seed
    created_wall = round(WALL_MINUS_MONO + t_start, 3)
    return t_start, created_wall, time.strftime("%Y%m%d_%H%M%S", time.gmtime(created_wall)) + "_synth"


def make_episode(spec: EpisodeSpec, index: int = 0, seed: int = 0, joint_names=DEFAULT_JOINT_NAMES,
                 camera_latency_s: float | None = None) -> labels.Episode:
    """The episode write_session(seed=seed) writes first, in memory and without images (numpy only). With another
    index it differs only in its random jitter."""
    t_start, created_wall, session_id = session_start(seed)
    rng = np.random.default_rng([seed, index])
    meta, rows, frames = episode_records(spec, index, session_id, t_start, 5000, rng, joint_names)
    session = session_record(session_id, joint_names, camera_latency_s, created_wall)
    return labels.Episode.from_records(meta, session, rows, frames)


# ------------------------------------------------------------------------------------------------------- images


def marker_boxes(eye_w: int, eye_h: int) -> dict:
    """Where the synthetic image puts things, in an upright eye of eye_w x eye_h: (x0, y0, x1, y1) boxes."""
    side = round(0.25 * eye_h)
    lx, ly = round(0.10 * eye_w), round(0.10 * eye_h)
    rx, ry = round(0.90 * eye_w) - side, round(0.90 * eye_h) - side
    block = eye_w // 12
    cx, cy = round(0.08 * eye_w), round(0.70 * eye_h)
    return {"left_marker": (lx, ly, lx + side, ly + side), "right_marker": (rx, ry, rx + side, ry + side),
            "code": [(cx + b * block, cy, cx + (b + 1) * block, cy + block) for b in range(CODE_BITS)]}


def render_frame(code: int, frame_size=FRAME_SIZE) -> np.ndarray:
    """The camera's view as it lands in the frame file: upright scene, then turned 180 degrees. RGB uint8."""
    width, height = frame_size
    eye_w = width // 2
    image = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)
    boxes = marker_boxes(eye_w, height)
    x0, y0, x1, y1 = boxes["left_marker"]
    image[y0:y1, x0:x1] = LEFT_MARKER_RGB
    x0, y0, x1, y1 = boxes["right_marker"]
    image[y0:y1, eye_w + x0:eye_w + x1] = RIGHT_MARKER_RGB
    for bit, (x0, y0, x1, y1) in enumerate(boxes["code"]):
        image[y0:y1, x0:x1] = 255 if (code >> bit) & 1 else 0
    return _turn(image)                                 # the camera hangs upside down


def _turn(image: np.ndarray) -> np.ndarray:
    """180 degrees, with cv2 when it is there (20x faster than copying image[::-1, ::-1])."""
    try:
        import cv2
        return cv2.rotate(image, cv2.ROTATE_180)
    except ImportError:
        return np.ascontiguousarray(image[::-1, ::-1])


def read_code(eye: np.ndarray) -> int:
    """The frame index drawn into an upright left eye (any size: boxes scale with it)."""
    h, w = eye.shape[:2]
    code = 0
    for bit, (x0, y0, x1, y1) in enumerate(marker_boxes(w, h)["code"]):
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        patch = eye[max(cy - 1, 0):cy + 2, max(cx - 1, 0):cx + 2].astype(float)
        if patch.mean() > 128:
            code |= 1 << bit
    return code


def encode_jpeg(rgb: np.ndarray, quality: int = 90) -> bytes:
    try:
        import cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("cv2 could not encode the JPEG")
        return buf.tobytes()
    from PIL import Image
    out = io.BytesIO()
    Image.fromarray(rgb).save(out, format="JPEG", quality=quality)
    return out.getvalue()


# --------------------------------------------------------------------------------------------------- the writer


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as fh:
        for record in records:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")


def write_session(root: Path, episodes: list[EpisodeSpec], seed: int = 0, joint_names=DEFAULT_JOINT_NAMES,
                  frame_size=FRAME_SIZE, camera_latency_s: float | None = None, session_id: str | None = None,
                  quality: int = 90) -> Path:
    """Write <root>/<session_id>/ with session.json and one ep_NNNN per spec. Returns the session directory."""
    t_start, created_wall, default_id = session_start(seed)
    session_id = session_id or default_id
    session_dir = Path(root) / session_id
    if session_dir.exists():
        raise FileExistsError(f"{session_dir} already exists")
    session_dir.mkdir(parents=True)
    session = session_record(session_id, joint_names, camera_latency_s, created_wall)
    (session_dir / "session.json").write_text(json.dumps(session, indent=1) + "\n")
    seq = 5000
    for index, spec in enumerate(episodes):
        rng = np.random.default_rng([seed, index])
        meta, rows, frames = episode_records(spec, index, session_id, t_start, seq, rng, joint_names)
        ep_dir = session_dir / ("_discarded" if spec.discarded else "") / f"ep_{index:04d}"
        ep_dir.mkdir(parents=True)
        offset = 0
        with open(ep_dir / "frames.mjpeg", "wb") as fh:
            for frame in frames:
                data = encode_jpeg(render_frame(frame["i"] % (1 << CODE_BITS), frame_size), quality)
                fh.write(data)
                frame["off"], frame["len"] = offset, len(data)
                offset += len(data)
        _write_jsonl(ep_dir / "rows.jsonl", rows)
        _write_jsonl(ep_dir / "frames.jsonl", frames)
        (ep_dir / "episode.json").write_text(json.dumps(meta, indent=1) + "\n")
        t_start += spec.seconds + EPISODE_GAP_S
        seq += len(rows) + int(EPISODE_GAP_S * spec.row_hz)
    return session_dir


def _size(text: str) -> tuple[int, int]:
    w, h = (int(v) for v in text.lower().split("x"))
    return w, h


def _gap(text: str) -> tuple[float, float]:
    start, duration = (float(v) for v in text.split(":"))
    return start, duration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="the session directory is created inside this one")
    parser.add_argument("--episodes", default="teleop,guided", help="modes, one episode each: teleop,guided,policy")
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.08, help="q lags qc by this, s")
    parser.add_argument("--size", type=_size, default=FRAME_SIZE, help="side-by-side frame size, WxH")
    parser.add_argument("--frame-gap", type=_gap, action="append", default=[], metavar="START:DURATION",
                        help="no frames in this window (s after start), in every episode; repeatable")
    parser.add_argument("--frame-hz", type=float, default=30.0, help="the camera's frame rate (a real one runs off 30)")
    parser.add_argument("--frame-jitter", type=float, default=0.002, help="+- s of uniform jitter on each frame time")
    parser.add_argument("--same-mtime", type=int, action="append", default=[], metavar="K",
                        help="frame K gets frame K-1's mtime and t_cap (one coarse-clock tick); repeatable")
    parser.add_argument("--zero-change", type=float, default=None, metavar="SECONDS",
                        help="retake the zero at this time; the episode then ends 'aborted'")
    parser.add_argument("--no-zero-check", action="store_true", help="write zero_check: null")
    parser.add_argument("--outcome", default="success", choices=("success", "failure", "aborted"))
    parser.add_argument("--zero-convention", default="hanging=0")
    parser.add_argument("--joint-names", default="default",
                        help="default, renamed, or ten comma-separated names (a different joint-name set)")
    parser.add_argument("--camera-latency", type=float, default=None, help="camera_latency_s in session.json")
    parser.add_argument("--no-servos", action="store_true", help="no camera servos: cam and cam_mode are null")
    parser.add_argument("--discard", action="store_true", help="write every episode under _discarded/")
    parser.add_argument("--session-id", default=None)
    args = parser.parse_args()
    if args.joint_names == "default":
        names = DEFAULT_JOINT_NAMES
    elif args.joint_names == "renamed":
        names = RENAMED_JOINT_NAMES
    else:
        names = tuple(n.strip() for n in args.joint_names.split(","))
    specs = [EpisodeSpec(mode=m.strip(), seconds=args.seconds, delay=args.delay, frame_gaps=tuple(args.frame_gap),
                         frame_hz=args.frame_hz, frame_jitter=args.frame_jitter, same_mtime=tuple(args.same_mtime),
                         zero_change_at=args.zero_change, zero_check=not args.no_zero_check, outcome=args.outcome,
                         zero_convention=args.zero_convention, servos=not args.no_servos, discarded=args.discard)
             for m in args.episodes.split(",") if m.strip()]
    for spec in specs:
        if spec.mode not in labels.MODES:
            parser.error(f"unknown mode {spec.mode!r}")
    path = write_session(args.root, specs, seed=args.seed, joint_names=names, frame_size=args.size,
                         camera_latency_s=args.camera_latency, session_id=args.session_id)
    print(path)
    for ep in labels.list_episodes(path, include_discarded=True):
        meta = json.loads((ep / "episode.json").read_text())
        print(f"  {ep.relative_to(path)}: {meta['mode']}, {meta['outcome']}, {meta['counts']['rows']} rows, "
              f"{meta['counts']['frames']} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
