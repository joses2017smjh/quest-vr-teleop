#!/usr/bin/env python3
"""Depth and optical flow for each of the robot's camera eyes, for the headset (A x2).

A twice turns the headset's vision pair (both eyes, above the camera panel) into a
grid, one column per eye:

    depth  LEFT  |  depth  RIGHT     Depth Anything V2 Small: one picture in, depth out
    flow   LEFT  |  flow   RIGHT     RAFT Small: two pictures a moment apart in, motion out

Depth: bright is near, dark is far, rescaled for every picture - the model knows what
is nearer than what, not how many metres. Flow: the colour is the direction each pixel
moved (the wheel in the corner is the key), how strong it is the speed, white is still.

Both models run on this PC's CPU, one core each, side by side, at a low priority, and
only while the headset shows the grid: the page asks the bridge for /depthflow.jpg,
which touches /dev/shm/bhl_depthflow.want, and this wakes up; 3 s after the page stops
asking it is idle again. Measured on the N150 with the whole rig up (load ~10): depth
1.1 s for both eyes, flow 0.6-0.8 s, so a new grid about once a second. More threads
made both slower - the CPU is already full, and pool threads that spin while they
wait steal it from the ones doing the work.

Models, fetched once into ~/.cache/bhl and torch's own cache (~140 MB):
  Depth Anything V2 Small (Apache-2.0), ONNX from onnx-community; its MatMuls are
    quantised to int8 here on first use - 1.8x faster on this CPU, same picture
  RAFT Small, torchvision's C_T_V2 weights (BSD)

  .venv/bin/python scripts/teleop/depth_flow.py              # the bringup window `depth`
  .venv/bin/python scripts/teleop/depth_flow.py --once x.jpg # one grid now, then exit
"""

from __future__ import annotations

import os

# One core per model, and a pool thread with nothing to do sleeps instead of spinning
# (see above): set before numpy, onnxruntime or torch start their pools.
for _pool in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_pool, "1")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import argparse  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

FRAME_FILE = Path("/dev/shm/bhl_camera.jpg")
GRID_FILE = Path("/dev/shm/bhl_depthflow.jpg")
WANTED = Path("/dev/shm/bhl_depthflow.want")     # the bridge touches it while the grid is shown
CACHE = Path.home() / ".cache/bhl"
DEPTH_URL = "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx"
DEPTH_MODEL = CACHE / "depth_anything_v2_small.onnx"          # float32, as published
DEPTH_INT8 = CACHE / "depth_anything_v2_small_mm8.onnx"       # MatMuls in int8: what runs
DEPTH_SIZE = (224, 168)       # w, h: multiples of the ViT's 14-pixel patches, 4:3 like an eye
FLOW_SIZE = (176, 128)        # w, h: multiples of 8, and RAFT needs 128 at least
TILE = (640, 480)             # each of the grid's four pictures: the grid fills the camera's place
UI = TILE[0] / 480            # the labels' sizes below are for 480-wide pictures
STILL = 1.5                   # px (at FLOW_SIZE) for full colour at the least: noise (~0.4) stays pale
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
FONT = cv2.FONT_HERSHEY_SIMPLEX

UPDATED = threading.Event()   # a model has something new for the grid


def wanted() -> bool:
    try:
        return time.time() - WANTED.stat().st_mtime < 3.0
    except OSError:
        return False


def grab(flipped: bool):
    """(when, [left eye, right eye]) from camera_share's newest frame; None when it is
    missing or more than 2 s old."""
    try:
        when = FRAME_FILE.stat().st_mtime
    except OSError:
        return None
    if time.time() - when > 2.0:
        return None
    frame = cv2.imread(str(FRAME_FILE))
    if frame is None:
        return None
    if flipped:
        frame = cv2.rotate(frame, cv2.ROTATE_180)      # mounted upside down on this robot
    half = frame.shape[1] // 2
    return when, [frame[:, :half], frame[:, half:]]


def quantise() -> None:
    """The published float model with its MatMuls - the bulk of a ViT - in int8."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    CACHE.mkdir(parents=True, exist_ok=True)
    if not DEPTH_MODEL.exists():
        print(f"Fetching Depth Anything V2 Small (99 MB) to {DEPTH_MODEL} ...", flush=True)
        part = DEPTH_MODEL.with_name(DEPTH_MODEL.stem + ".part")
        urllib.request.urlretrieve(DEPTH_URL, part)
        os.replace(part, DEPTH_MODEL)
    print("Quantising its MatMuls to int8 (once, about 15 s) ...", flush=True)
    part = DEPTH_INT8.with_name(DEPTH_INT8.stem + ".part.onnx")
    quantize_dynamic(str(DEPTH_MODEL), str(part), op_types_to_quantize=["MatMul"],
                     weight_type=QuantType.QInt8)
    os.replace(part, DEPTH_INT8)


class Depth:
    """Depth Anything V2 Small over both eyes in one run: bigger is nearer."""

    name = "Depth Anything V2 Small"

    def __init__(self):
        import onnxruntime as ort
        if not DEPTH_INT8.exists():
            quantise()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(DEPTH_INT8), opts, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.range = None                 # near and far, smoothed so the colours do not flicker

    def __call__(self, eyes: list) -> list:
        batch = np.stack([self.prepare(eye) for eye in eyes])
        depth = self.session.run(None, {self.input: batch})[0]
        # both eyes on one scale, so the same thing is the same colour in each
        seen = np.percentile(depth, (2, 98))
        self.range = seen if self.range is None else 0.6 * self.range + 0.4 * seen
        lo, hi = self.range
        pictures = []
        for d in depth:
            level = np.clip((d - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
            level = cv2.resize(level, TILE, interpolation=cv2.INTER_CUBIC)
            pictures.append(cv2.applyColorMap(level, cv2.COLORMAP_INFERNO))
        return pictures

    @staticmethod
    def prepare(eye: np.ndarray) -> np.ndarray:
        rgb = cv2.resize(eye, DEPTH_SIZE, interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 255.0
        return ((rgb - MEAN) / STD).transpose(2, 0, 1)


class Flow:
    """RAFT Small over both eyes in one run: how far each pixel moved between two frames."""

    name = "RAFT Small"
    PASSES = 4                # refinement passes: the paper uses 12; each costs about 10%

    def __init__(self):
        import torch
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
        torch.set_num_threads(1)
        self.torch = torch
        self.model = raft_small(weights=Raft_Small_Weights.C_T_V2).eval()
        self.scale = None         # the speed shown at full colour, smoothed like depth's range

    def __call__(self, before: list, after: list) -> list:
        torch = self.torch

        def batch(eyes):
            small = [cv2.resize(e, FLOW_SIZE, interpolation=cv2.INTER_AREA)[:, :, ::-1] for e in eyes]
            return torch.from_numpy(np.stack(small).copy()).permute(0, 3, 1, 2).float() / 127.5 - 1.0

        with torch.inference_mode():
            flow = self.model(batch(before), batch(after), num_flow_updates=self.PASSES)[-1].numpy()
        top = max(float(np.percentile(np.hypot(flow[:, 0], flow[:, 1]), 99)), STILL)
        self.scale = top if self.scale is None else 0.6 * self.scale + 0.4 * top
        return [cv2.resize(flow_picture(f, self.scale), TILE, interpolation=cv2.INTER_LINEAR) for f in flow]


def colour_wheel() -> np.ndarray:
    """The Middlebury flow colours (RGB, 55 steps): red, yellow, green, cyan, blue, magenta."""
    stops = np.array([(255, 0, 0), (255, 255, 0), (0, 255, 0), (0, 255, 255), (0, 0, 255), (255, 0, 255)], float)
    steps = []
    for i, n in enumerate((15, 6, 4, 11, 13, 6)):
        a, b = stops[i], stops[(i + 1) % 6]
        steps.append(a + (b - a) * np.floor(255 * np.arange(n) / n)[:, None] / 255)
    return np.concatenate(steps)


WHEEL = colour_wheel()


def flow_picture(flow: np.ndarray, scale: float) -> np.ndarray:
    """(2, h, w) motion in pixels -> BGR: hue is the direction, colour the speed (full at
    `scale`, white when still), the way RAFT's own figures show it."""
    u, v = flow[0] / scale, flow[1] / scale
    speed = np.sqrt(u * u + v * v)[..., None]
    at = (np.arctan2(-v, -u) / np.pi + 1) / 2 * (len(WHEEL) - 1)
    k0 = np.floor(at).astype(np.int32)
    k1 = (k0 + 1) % len(WHEEL)
    f = (at - k0)[..., None]
    colour = ((1 - f) * WHEEL[k0] + f * WHEEL[k1]) / 255.0
    colour = np.where(speed <= 1, 1 - speed * (1 - colour), colour * 0.75)
    return (colour[..., ::-1] * 255).astype(np.uint8)


def px(v: float) -> int:
    """A label size or position, laid out for 480-wide pictures, at the pictures' real size."""
    return int(round(v * UI))


def wheel_key(size: int = px(70)):
    """The colour key: a disc where each point has the colour of moving toward it."""
    r = np.linspace(-1, 1, size)
    u, v = np.meshgrid(r, r)
    return flow_picture(np.stack([u, v]), 1.0), u * u + v * v <= 1


KEY, KEY_MASK = wheel_key()
BAR = cv2.applyColorMap(np.tile(np.linspace(0, 255, px(120)).astype(np.uint8), (px(10), 1)), cv2.COLORMAP_INFERNO)


def tag(img: np.ndarray, text: str, x: int, y: int, scale: float = 0.6, bold: bool = False) -> None:
    """Text on a darkened patch, its top left corner at (x, y)."""
    thick, scale = (2 if bold else 1), scale * UI
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    patch = img[y:y + th + px(12), x:x + tw + px(8)]
    patch[:] = (patch * 0.35).astype(np.uint8)
    cv2.putText(img, text, (x + px(4), y + th + px(5)), FONT, scale, (240, 240, 240), thick, cv2.LINE_AA)


class Runner(threading.Thread):
    """One model, run again and again while the headset watches; its newest pictures kept."""

    def __init__(self, make, flipped: bool, gap: float = 0.0):
        super().__init__(daemon=True, name=make.name)
        self.make, self.flipped, self.gap = make, flipped, gap
        self.model = None
        self.pictures = None          # [left, right], tile-sized, once it has run
        self.seconds = 0.0            # the last run, the model alone
        self.note = f"loading {make.name} ..."

    def step(self) -> None:
        first = grab(self.flipped)
        if first is None:
            self.pictures, self.note = None, "no camera frames: is camera_share.py running?"
            UPDATED.set()
            time.sleep(0.5)
            return
        frames = [first[1]]
        if self.gap:                  # flow: a second frame, a moment later
            time.sleep(self.gap)
            second = grab(self.flipped)
            if second is None or second[0] == first[0]:
                return                # the camera has not moved on; the next try will say why
            frames.append(second[1])
        started = time.perf_counter()
        self.pictures = self.model(*frames)
        self.seconds = time.perf_counter() - started
        UPDATED.set()

    def run(self) -> None:
        while True:
            # loaded at once, not at the first A x2: importing torch alone takes 15 s on
            # this busy CPU, and after that the idle model costs only memory
            if self.model is None:
                try:
                    self.model = self.make()
                    self.note = f"running {self.make.name} ..."
                    print(f"  {self.make.name} loaded", flush=True)
                except Exception as exc:          # say so in its pictures, and try again later
                    self.note = f"{self.make.name} did not load: {exc}"
                    print(f"  {self.note}", flush=True)
                    UPDATED.set()
                    time.sleep(30)
                    continue
            if not wanted():
                time.sleep(0.3)
                continue
            try:
                self.step()
            except Exception as exc:
                self.note = f"{self.make.name}: {exc}"
                self.pictures = None
                print(f"  {self.note}", flush=True)
                UPDATED.set()
                time.sleep(2)


def waiting(eye, note: str) -> np.ndarray:
    """A tile with no model output yet: the camera's own picture, dimmed, and why."""
    tile = np.full((TILE[1], TILE[0], 3), 24, np.uint8)
    if eye is not None:
        grey = cv2.cvtColor(cv2.resize(eye, TILE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        tile[:] = (grey[..., None] * 0.35).astype(np.uint8)
    words, lines = note.split(), [""]
    for word in words:                        # wrap to the tile
        trial = (lines[-1] + " " + word).strip()
        if cv2.getTextSize(trial, FONT, 0.6 * UI, 1)[0][0] > TILE[0] - px(40) and lines[-1]:
            lines.append(word)
        else:
            lines[-1] = trial
    for i, line in enumerate(lines[:4]):
        (tw, _), _ = cv2.getTextSize(line, FONT, 0.6 * UI, 1)
        cv2.putText(tile, line, ((TILE[0] - tw) // 2, TILE[1] // 2 - px(10) + px(28) * i), FONT, 0.6 * UI,
                    (230, 230, 230), 1, cv2.LINE_AA)
    return tile


def compose(depth: Runner, flow: Runner, flipped: bool) -> np.ndarray:
    """depth left | depth right over flow left | flow right."""
    eyes = None
    if depth.pictures is None or flow.pictures is None:
        got = grab(flipped)
        eyes = got[1] if got else None
    rows = []
    for runner, title in ((depth, "DEPTH"), (flow, "RAFT FLOW")):
        tiles = []
        for i, side in enumerate(("LEFT", "RIGHT")):
            if runner.pictures is not None:
                tile = runner.pictures[i].copy()
                tag(tile, f"{runner.name}  {runner.seconds:.1f} s", px(6), TILE[1] - px(30), 0.5)
                if runner is depth:           # the key, top right: dark is far, bright near
                    shade = tile[px(6):px(32), TILE[0] - px(206):TILE[0] - px(4)]
                    shade[:] = (shade * 0.35).astype(np.uint8)
                    x = TILE[0] - px(166)
                    tile[px(14):px(14) + BAR.shape[0], x:x + BAR.shape[1]] = BAR
                    for word, at in (("far", px(200)), ("near", px(42))):
                        cv2.putText(tile, word, (TILE[0] - at, px(24)), FONT, 0.45 * UI, (240, 240, 240), 1, cv2.LINE_AA)
                else:
                    y, x = px(6), TILE[0] - KEY.shape[1] - px(6)
                    corner = tile[y:y + KEY.shape[0], x:x + KEY.shape[1]]
                    corner[KEY_MASK] = KEY[KEY_MASK]
                    cv2.circle(tile, (x + KEY.shape[1] // 2, y + KEY.shape[0] // 2), KEY.shape[0] // 2,
                               (60, 60, 60), 1, cv2.LINE_AA)
            else:
                tile = waiting(eyes[i] if eyes else None, runner.note)
            tag(tile, f"{title}  {side}", px(6), px(6), 0.65, bold=True)
            tiles.append(tile)
        rows.append(np.hstack(tiles))
    return np.vstack(rows)


def write(picture: np.ndarray, path: Path) -> None:
    ok, jpeg = cv2.imencode(".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if ok:
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(jpeg.tobytes())
        os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gap", type=float, default=0.12, help="seconds between the two frames RAFT compares")
    parser.add_argument("--not-flipped", action="store_true", help="the camera is mounted the right way up")
    parser.add_argument("--once", metavar="JPG", help="write one grid to JPG now, whether or not anyone watches, and exit")
    args = parser.parse_args()
    try:
        os.nice(10)               # the arms' control loop, the bridge and the tracker come first
    except OSError:
        pass
    flipped = not args.not_flipped
    depth = Runner(Depth, flipped)
    flow = Runner(Flow, flipped, gap=args.gap)

    if args.once:
        for runner in (depth, flow):
            runner.model = runner.make()
            runner.step()
        write(compose(depth, flow, flipped), Path(args.once))
        print(f"{args.once}: depth {depth.seconds:.2f} s, flow {flow.seconds:.2f} s (both eyes each)")
        return 0

    depth.start()
    flow.start()
    print(f"Depth + RAFT flow for the headset (A x2): idle until the grid is shown ({WANTED} fresh)", flush=True)
    watching, told = False, 0.0
    while True:
        UPDATED.wait(1.0)             # a new result, or once a second anyway: the file stays fresh
        UPDATED.clear()
        if not wanted():
            if watching:
                watching = False
                print("  the headset stopped showing the grid: idle", flush=True)
            continue
        if not watching:
            watching, told = True, time.monotonic()
            print("  the headset shows the grid: running both models", flush=True)
        try:
            write(compose(depth, flow, flipped), GRID_FILE)
        except Exception as exc:      # a bad frame must not end it
            print(f"  grid: {exc}", flush=True)
        if depth.pictures is not None and flow.pictures is not None and time.monotonic() - told > 60:
            told = time.monotonic()
            print(f"  depth {depth.seconds:.1f} s, flow {flow.seconds:.1f} s per run (both eyes)", flush=True)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
