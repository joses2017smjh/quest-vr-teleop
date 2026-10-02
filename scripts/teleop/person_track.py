#!/usr/bin/env python3
"""Where is the operator? The robot's stereo camera answers, a few times a second.

Reads the frames camera_share.py already writes (/dev/shm/bhl_camera.jpg, so the
camera itself stays free), finds people with NanoDet-Plus (OpenCV model zoo, 4 MB,
COCO; downloaded once to ~/.cache/bhl), and reports the direction of the most
likely one as yaw/pitch from the camera's axis:

    {"type": "person", "seen": true, "yaw": deg (+ = left), "pitch": deg (+ = up),
     "top": deg (the top of their head, unsmoothed), "score": 0..1,
     "box": [x1, y1, x2, y2], "head": [x, y], "t": time}

to the bridge's status relay (UDP 11006), which passes it to every page, and to any
consumer that wants to turn toward the operator - the robot's face display first.

People are found by body shape: a face detector alone is useless while the operator
wears a headset over the eyes. The head inside the body box is, in order: a face
(YuNet: headset off), a headset-shaped white blob (compact, wider than tall - a white
wall behind the head is neither), or the top-centre of the box.

While the headset shows its vision panel (the bridge keeps /dev/shm/bhl_vision.want
fresh), a second, low-priority thread also draws what it sees: both eyes side by side,
upright, with a box for the human and for each body part (head, torso, arms, hands,
legs) from MediaPipe's pose model (OpenCV model zoo, 5.5 MB), into
/dev/shm/bhl_vision.jpg for /vision.jpg. The pose model follows the person by itself
from frame to frame, so the detector above is only needed to start it.

  .venv/bin/python scripts/teleop/person_track.py            # --show writes an annotated frame
"""

from __future__ import annotations

import os

# One core, for every thread pool, and before numpy or OpenCV load: spread over all
# four, a detection cost 3x the CPU for no speed-up, and starved the control loop.
for _pool in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_pool, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

FRAME_FILE = Path("/dev/shm/bhl_camera.jpg")
SHOW_FILE = Path("/dev/shm/bhl_person.jpg")
MODEL = Path.home() / ".cache/bhl/nanodet.onnx"
MODEL_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/object_detection_nanodet/"
             "object_detection_nanodet_2022nov.onnx")
FACE_MODEL = Path.home() / ".cache/bhl/yunet.onnx"
FACE_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
            "face_detection_yunet_2023mar.onnx")
POSE_MODEL = Path.home() / ".cache/bhl/pose.onnx"
POSE_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/pose_estimation_mediapipe/"
            "pose_estimation_mediapipe_2023mar.onnx")
VISION_FILE = Path("/dev/shm/bhl_vision.jpg")
VISION_WANTED = Path("/dev/shm/bhl_vision.want")   # the bridge touches it while the panel is shown

# BlazePose's 33 points that make up each body part ("L" and "R" are the person's own
# sides, so your left hand is on the right of the picture), and its box colour (BGR)
PARTS = (
    ("head", tuple(range(11)), (0, 215, 255)),
    ("torso", (11, 12, 23, 24), (255, 190, 90)),
    ("L arm", (11, 13, 15), (70, 150, 255)),
    ("R arm", (12, 14, 16), (255, 140, 200)),
    ("L hand", (15, 17, 19, 21), (40, 90, 255)),
    ("R hand", (16, 18, 20, 22), (230, 90, 255)),
    ("L leg", (23, 25, 27, 29, 31), (60, 200, 255)),
    ("R leg", (24, 26, 28, 30, 32), (255, 210, 120)),
)
BONES = ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
         (23, 25), (25, 27), (24, 26), (26, 28), (15, 19), (16, 20))
HUMAN = (90, 220, 90)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class NanoDet:
    """NanoDet-Plus m 416 from the OpenCV model zoo: boxes per COCO class."""

    STRIDES, SIZE, REG_MAX = (8, 16, 32, 64), 416, 7
    MEAN = np.array([103.53, 116.28, 123.675], np.float32)
    STD = np.array([57.375, 57.12, 58.395], np.float32)

    def __init__(self, path: Path):
        self.net = cv2.dnn.readNet(str(path))
        self.names = self.net.getUnconnectedOutLayersNames()
        self.anchors = []
        for stride in self.STRIDES:
            n = math.ceil(self.SIZE / stride)
            xv, yv = np.meshgrid(np.arange(n) * stride, np.arange(n) * stride)
            self.anchors.append(np.column_stack([xv.ravel() + 0.5 * (stride - 1), yv.ravel() + 0.5 * (stride - 1)]))
        self.project = np.arange(self.REG_MAX + 1, dtype=np.float32)

    def people(self, bgr: np.ndarray, threshold: float = 0.4) -> list[tuple[tuple[int, int, int, int], float]]:
        h, w = bgr.shape[:2]
        k = self.SIZE / max(h, w)                        # letterbox: keep the shape
        canvas = np.zeros((self.SIZE, self.SIZE, 3), np.float32)
        small = cv2.resize(bgr, (int(w * k), int(h * k)))
        canvas[:small.shape[0], :small.shape[1]] = small
        self.net.setInput(cv2.dnn.blobFromImage((canvas - self.MEAN) / self.STD))
        outs = self.net.forward(self.names)
        boxes, scores = [], []
        for stride, cls, reg, anchors in zip(self.STRIDES, outs[::2], outs[1::2], self.anchors):
            cls = cls.reshape(-1, cls.shape[-1])
            e = np.exp(reg.reshape(-1, self.REG_MAX + 1))
            d = ((e / e.sum(1, keepdims=True)) @ self.project).reshape(-1, 4) * stride
            boxes.append(np.column_stack([anchors[:, 0] - d[:, 0], anchors[:, 1] - d[:, 1],
                                          anchors[:, 0] + d[:, 2], anchors[:, 1] + d[:, 3]]) / k)
            scores.append(cls[:, 0])                     # COCO class 0: person
        boxes, scores = np.concatenate(boxes), np.concatenate(scores)
        keep = scores > threshold
        boxes, scores = boxes[keep], scores[keep]
        if not len(boxes):
            return []
        wh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes]
        picked = np.array(cv2.dnn.NMSBoxes(wh, scores.tolist(), threshold, 0.5)).ravel()
        return [(tuple(int(v) for v in boxes[i]), float(scores[i])) for i in picked]


class Pose:
    """MediaPipe BlazePose (full) from the OpenCV model zoo: 33 points on one person.

    It must be shown roughly where the person is: an upright square around them. The
    first time, that comes from the detector's box; after that, from two extra points
    the model returns for exactly this (the hips' centre, and how far the body reaches
    from it), so it follows the person by itself, as MediaPipe does."""

    SIZE = 256

    def __init__(self, path: Path):
        self.net = cv2.dnn.readNet(str(path))
        self.names = self.net.getUnconnectedOutLayersNames()

    @staticmethod
    def square(cx: float, cy: float, side: float, down=(0.0, 1.0)) -> np.ndarray:
        """The map from the model's 256x256 input to that square of the image (2x3)."""
        dx, dy = down
        rx, ry = dy, -dx
        k, h = side / Pose.SIZE, Pose.SIZE / 2
        return np.array([[rx * k, dx * k, cx - h * k * (rx + dx)],
                         [ry * k, dy * k, cy - h * k * (ry + dy)]])

    @staticmethod
    def around(box) -> np.ndarray:
        x1, y1, x2, y2 = box
        return Pose.square((x1 + x2) / 2, (y1 + y2) / 2, 1.25 * max(x2 - x1, y2 - y1))

    @staticmethod
    def next_square(xy: np.ndarray) -> np.ndarray:
        hips, reach = xy[33], xy[34]
        r = float(np.hypot(*(reach - hips)))
        down = (hips - reach) / max(r, 1e-6)
        return Pose.square(float(hips[0]), float(hips[1]), 2.5 * r, (float(down[0]), float(down[1])))

    def find(self, bgr: np.ndarray, where: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """(confidence 0..1, the 39 points in image pixels, how visible each is 0..1)."""
        crop = cv2.warpAffine(bgr, where, (self.SIZE, self.SIZE),
                              flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)
        self.net.setInput(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)[None] / 255.0)
        outs = {o.shape[-1]: o for o in self.net.forward(self.names) if o.ndim == 2}
        points = outs[195].reshape(-1, 5)
        xy = np.column_stack([points[:, :2], np.ones(len(points))]) @ where.T
        return float(outs[1][0, 0]), xy, 1 / (1 + np.exp(-points[:, 3]))


def part_boxes(xy: np.ndarray, seen: np.ndarray, shape) -> list[tuple[str, tuple, tuple]]:
    """[(name, (x1, y1, x2, y2), colour)] for each body part the pose model can see."""
    h, w = shape[:2]
    found = []
    for name, idx, colour in PARTS:
        idx = list(idx)
        vis = seen[idx] > 0.5
        pts = xy[idx][vis]
        if name == "head":
            if len(pts) < 3:
                continue
            (x1, y1), (x2, y2) = pts.min(0), pts.max(0)
            # the face's points run ear to ear and eyes to mouth: a head is taller, and
            # its eyes sit about halfway down
            tall = max(3.2 * (y2 - y1), 1.4 * (x2 - x1), 0.06 * h)
            eyes = xy[[1, 2, 3, 4, 5, 6]][seen[[1, 2, 3, 4, 5, 6]] > 0.5]
            eye_y = float(eyes[:, 1].mean()) if len(eyes) else float(y1)
            cx, wide = (x1 + x2) / 2, tall / 1.3
            box = (cx - wide / 2, eye_y - 0.5 * tall, cx + wide / 2, eye_y + 0.5 * tall)
        elif name.endswith("hand"):
            # the wrist and three knuckles: the fingers reach about as far past the knuckles
            if not vis[0] or vis[1:].sum() < 1:
                continue
            centre = xy[idx[1:]][vis[1:]].mean(0)
            r = max(1.3 * float(np.hypot(*(centre - xy[idx[0]]))), 0.03 * h)
            box = (centre[0] - r, centre[1] - r, centre[0] + r, centre[1] + r)
        elif name == "torso":
            if not (vis[0] and vis[1]):           # both shoulders; the hips may be out of view
                continue
            (x1, y1), (x2, y2) = xy[idx].min(0), xy[idx].max(0)
            pad = 0.05 * max(x2 - x1, y2 - y1)
            box = (x1 - pad, y1 - pad, x2 + pad, y2 + pad)
        else:                                     # an arm or a leg: the joint it hangs from, and one more
            if not vis[0] or vis.sum() < 2:
                continue
            (x1, y1), (x2, y2) = pts.min(0), pts.max(0)
            pad = max(0.15 * max(x2 - x1, y2 - y1), 0.02 * h)
            box = (x1 - pad, y1 - pad, x2 + pad, y2 + pad)
        x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w - 1, box[2]), min(h - 1, box[3])
        if x2 - x1 >= 6 and y2 - y1 >= 6:
            found.append((name, (x1, y1, x2, y2), colour))
    return found


def draw_boxes(img: np.ndarray, boxes: list) -> None:
    """[(box, label, colour, thickness)]: every outline first, then the labels, each put
    where it covers no other label - above its box, else inside its top, inside its
    bottom, or below it - so "human" and "head" do not print over each other."""
    for box, _, colour, thick in boxes:
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, thick, cv2.LINE_AA)
    taken: list[tuple[int, int, int, int]] = []
    h, w = img.shape[:2]
    for box, text, colour, _ in boxes:
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.6, 1)
        lw, lh = tw + 8, th + 10
        x = min(max(0, x1), w - lw)
        spots = [(x, y1 - lh), (x, y1 + 2), (x, y2 - lh - 2), (x, y2 + 1)]
        fits = [(sx, sy) for sx, sy in spots if 0 <= sy and sy + lh <= h]
        free = [(sx, sy) for sx, sy in fits
                if not any(sx < t[2] and t[0] < sx + lw and sy < t[3] and t[1] < sy + lh for t in taken)]
        sx, sy = (free or fits or spots)[0]
        taken.append((sx, sy, sx + lw, sy + lh))
        cv2.rectangle(img, (sx, sy), (sx + lw, sy + lh), colour, -1)
        cv2.putText(img, text, (sx + 4, sy + th + 4), FONT, 0.6, (12, 12, 12), 1, cv2.LINE_AA)


def corner(img: np.ndarray, text: str, right: bool = False, middle: bool = False) -> None:
    """A label on a darkened patch: top left, top right, or in the middle."""
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.65, 2)
    x = img.shape[1] - tw - 16 if right else (img.shape[1] - tw) // 2 if middle else 8
    y = img.shape[0] // 2 - th if middle else 6
    shade = img[y:y + th + 14, x - 2:x + tw + 10]
    shade[:] = (shade * 0.35).astype(np.uint8)
    cv2.putText(img, text, (x + 4, y + th + 6), FONT, 0.65, (240, 240, 240), 2, cv2.LINE_AA)


class Vision(threading.Thread):
    """Both eyes with what is seen drawn on them, for the headset's vision panel.

    Runs at a low priority, and only while the panel is shown: the pose model costs
    about 80 ms of a core per eye, and the robot's control loop comes first."""

    def __init__(self):
        super().__init__(daemon=True)
        self.latest = None
        self.ready = threading.Event()
        self.pose = None                             # the model; False when it cannot load
        self.where: list = [None, None]              # each eye's square for the pose model
        self.rate = 0.0
        self.told = False

    @staticmethod
    def wanted() -> bool:
        try:
            return time.time() - VISION_WANTED.stat().st_mtime < 3.0
        except OSError:
            return False

    def offer(self, frame: np.ndarray, box, score: float, aim, how: str) -> None:
        """The frame the tracker just used, and what it found in the left eye."""
        self.latest = (frame.copy(), box, score, aim, how)
        self.ready.set()

    def run(self) -> None:
        try:                                         # this thread only: the tracker keeps its share
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 10)
        except (AttributeError, OSError):
            pass
        last = 0.0
        while True:
            self.ready.wait(1.0)
            self.ready.clear()
            if not self.wanted() or self.latest is None:
                if not self.wanted():
                    self.where = [None, None]        # a stale track would start in the wrong place
                continue
            if self.pose is None and not self.load():
                self.pose = False                    # no model: the detector's box only
            frame, box, score, aim, how = self.latest
            self.latest = None
            try:
                picture = self.draw(frame, box, score, aim, how)
            except Exception as exc:                 # never take the tracker down with it
                print(f"  vision: {exc}", flush=True)
                continue
            now = time.monotonic()
            if last:
                step = 1.0 / max(1e-3, now - last)
                self.rate = 0.7 * self.rate + 0.3 * step if self.rate else step
            last = now
            ok, jpeg = cv2.imencode(".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                tmp = VISION_FILE.with_suffix(".tmp")
                tmp.write_bytes(jpeg.tobytes())
                os.replace(tmp, VISION_FILE)

    def load(self) -> bool:
        try:
            if not POSE_MODEL.exists():
                print(f"Fetching the pose model (5.5 MB) to {POSE_MODEL} ...", flush=True)
                urllib.request.urlretrieve(POSE_URL, POSE_MODEL)
            self.pose = Pose(POSE_MODEL)
            print("  vision panel: the headset is watching; drawing both eyes with the pose model", flush=True)
            return True
        except Exception as exc:
            if not self.told:
                self.told = True
                print(f"  (no pose model: {exc}; the vision panel shows the detector's box only)", flush=True)
            return False

    def track(self, i: int, eye: np.ndarray, box):
        """The pose in one eye: from where it was last frame, else from the detector's box
        (the left eye's serves both: the eyes are 6 cm apart, and the square is generous)."""
        tries = [self.where[i]] if self.where[i] is not None else []
        if box is not None:
            tries.append(Pose.around(box))
        for where in tries:
            conf, xy, seen = self.pose.find(eye, where)
            side = float(np.hypot(*where[:, 0])) * Pose.SIZE
            if conf >= 0.5 and 0.1 * eye.shape[0] < side < 4 * eye.shape[1]:
                self.where[i] = Pose.next_square(xy)
                return conf, xy, seen
        self.where[i] = None
        return None

    def draw(self, frame: np.ndarray, box, score: float, aim, how: str) -> np.ndarray:
        half = frame.shape[1] // 2
        eyes = [frame[:, :half].copy(), frame[:, half:].copy()]
        for i, eye in enumerate(eyes):
            found = self.track(i, eye, box) if self.pose else None
            parts = part_boxes(found[1], found[2], eye.shape) if found else []
            if found:
                _, xy, seen = found
                for a, b in BONES:
                    if seen[a] > 0.5 and seen[b] > 0.5:
                        cv2.line(eye, tuple(int(v) for v in xy[a]), tuple(int(v) for v in xy[b]),
                                 (235, 235, 235), 2, cv2.LINE_AA)
            boxes = []
            if i == 0 and box is not None:           # the detector's box: what the camera follows
                boxes.append((box, f"human {score:.0%}", HUMAN, 3))
            elif found and parts:                    # the other eye: the body the pose model found
                boxes.append(((min(p[1][0] for p in parts), min(p[1][1] for p in parts),
                               max(p[1][2] for p in parts), max(p[1][3] for p in parts)),
                              f"human {found[0]:.0%}", HUMAN, 3))
            draw_boxes(eye, boxes + [(part, name, colour, 2) for name, part, colour in parts])
            if i == 0 and aim is not None:           # where the camera aims: the head it found
                x, y = int(aim[0]), int(aim[1])
                cv2.drawMarker(eye, (x, y), (60, 60, 255), cv2.MARKER_CROSS, 26, 3, cv2.LINE_AA)
                cv2.putText(eye, f"aim ({how})", (x + 16, y + 22), FONT, 0.6, (60, 60, 255), 2, cv2.LINE_AA)
            if box is None and not found:
                corner(eye, "no human in view", middle=True)
            corner(eye, "LEFT" if i == 0 else "RIGHT")
        if self.rate:
            corner(eyes[1], f"{self.rate:.1f}/s", right=True)
        return np.hstack(eyes)


def head_of(bgr: np.ndarray, box: tuple[int, int, int, int], faces=None) -> tuple[tuple[int, int], str]:
    """(head point, how it was found): a face, a headset, or the top of the box."""
    x1, y1, x2, y2 = (max(0, v) for v in box)
    h = max(1, y2 - y1)
    top = bgr[y1:y1 + int(0.4 * h), x1:x2]
    if top.size == 0:
        return ((x1 + x2) // 2, y1), "box"
    if faces is not None:                                             # headset off: a face
        faces.setInputSize((top.shape[1], top.shape[0]))
        found = faces.detect(top)[1]
        if found is not None and len(found):
            fx, fy, fw, fh = found[int(np.argmax(found[:, -1]))][:4]
            return (int(x1 + fx + fw / 2), int(y1 + fy + fh / 2)), "face"
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 175), (180, 60, 255))           # bright and colourless
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, _, stats, centres = cv2.connectedComponentsWithStats(white)
    area_top = top.shape[0] * top.shape[1]
    best, best_area = None, 0
    for i in range(1, count):
        x, y, w, hh, area = stats[i]
        touches = x <= 1 or x + w >= top.shape[1] - 1 or y <= 1      # walls run off the edges
        fill = area / float(w * hh)
        if (0.004 * area_top < area < 0.25 * area_top and 1.1 < w / float(hh) < 4.0
                and fill > 0.45 and not touches and area > best_area):
            best, best_area = i, area
    if best is not None:                                              # headset on: the Quest
        cx, cy = centres[best]
        return (int(x1 + cx), int(y1 + cy)), "headset"
    return ((x1 + x2) // 2, y1 + int(0.1 * h)), "box"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # 78, not the 100 it was: near the middle, where the operator ends up, the picture moves
    # 0.62 px per us of servo pulse (camera sweeps, 30 Sep), which is 78 deg at an SG90's
    # 0.09 deg/us - with 100, every turn toward the operator went ~1.46x too far
    parser.add_argument("--hfov", type=float, default=78.0, help="one eye's horizontal field of view, degrees")
    parser.add_argument("--rate", type=float, default=3.0,
                        help="detections per second, at most (each costs about 0.2 s of one CPU core)")
    parser.add_argument("--status-port", type=int, default=11006, help="the bridge's relay")
    parser.add_argument("--face-port", type=int, default=11010, help="face_display.py, the robot's screen")
    parser.add_argument("--teleop-port", type=int, default=11011, help="run_teleop.py: the camera keeps you centred")
    parser.add_argument("--not-flipped", action="store_true", help="the camera is mounted the right way up")
    parser.add_argument("--show", action="store_true", help=f"also write an annotated frame to {SHOW_FILE}")
    args = parser.parse_args()

    if not MODEL.exists():
        print(f"Fetching the person detector (4 MB) to {MODEL} ...", flush=True)
        MODEL.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL)
    cv2.setNumThreads(1)          # one core: the robot's control loop needs the rest
    detector = NanoDet(MODEL)
    faces = None
    try:
        if not FACE_MODEL.exists():
            urllib.request.urlretrieve(FACE_URL, FACE_MODEL)
        faces = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (320, 320), 0.7)
    except Exception as exc:                          # the headset and the box still work
        print(f"  (no face detector: {exc})", flush=True)
    relay = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    vision = Vision()
    vision.start()
    print(f"Tracking the operator from {FRAME_FILE} at up to {args.rate:g} Hz -> UDP {args.status_port}", flush=True)

    last_mtime, smooth, seen_before = 0.0, None, None
    period = 1.0 / max(0.5, args.rate)
    while True:
        started = time.monotonic()
        try:
            mtime = FRAME_FILE.stat().st_mtime
        except OSError:
            mtime = 0.0
        if mtime and mtime != last_mtime and time.time() - mtime < 2.0:
            last_mtime = mtime
            frame = cv2.imread(str(FRAME_FILE))
            if frame is not None:
                if not args.not_flipped:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)   # mounted upside down on this robot
                eye = frame[:, :frame.shape[1] // 2]
                # 0.4 finds someone sitting with their back turned; small boxes (a bottle
                # on a shelf scored 0.52) are dropped: anyone near the robot is big
                found = [(b, sc) for b, sc in detector.people(eye) if b[3] - b[1] > 0.2 * eye.shape[0]]
                # when the frame was taken: the camera may have turned since, so the
                # reader must add where the camera pointed THEN, not now
                report = {"type": "person", "seen": False, "t": time.time(), "captured": mtime}
                if found:
                    box, score = max(found, key=lambda item: item[1] * (item[0][2] - item[0][0]) * (item[0][3] - item[0][1]))
                    (hx, hy), how = head_of(eye, box, faces)
                    h, w = eye.shape[:2]
                    f = (w / 2) / math.tan(math.radians(args.hfov) / 2)
                    yaw = -math.degrees(math.atan((hx - w / 2) / f))       # + = the operator is to the robot's left
                    pitch = -math.degrees(math.atan((hy - h / 2) / f))     # + = above the camera's axis
                    # the box's top edge (the frame's, when the head is cut off): the camera
                    # tilts by it, as it holds still where the head point jumps ~10 deg
                    # between a face, the headset and the box
                    top = -math.degrees(math.atan((max(0, box[1]) - h / 2) / f))
                    smooth = (yaw, pitch) if smooth is None else (0.6 * smooth[0] + 0.4 * yaw, 0.6 * smooth[1] + 0.4 * pitch)
                    report.update(seen=True, yaw=round(smooth[0], 1), pitch=round(smooth[1], 1), top=round(top, 1),
                                  score=round(score, 2), box=list(box), head=[hx, hy], by=how)
                if Vision.wanted():                   # before --show draws on the frame
                    vision.offer(frame, box if found else None, score if found else 0.0,
                                 (hx, hy) if found else None, how if found else "")
                if found and args.show:
                    cv2.rectangle(eye, box[:2], box[2:], (0, 255, 0), 2)
                    cv2.circle(eye, (hx, hy), 8, (0, 0, 255), -1)
                    cv2.putText(eye, how, (hx + 12, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imwrite(str(SHOW_FILE), eye)
                if report["seen"] != seen_before:
                    seen_before = report["seen"]
                    print(f"  {'operator at yaw %+.0f deg, pitch %+.0f deg' % (report['yaw'], report['pitch']) if report['seen'] else 'nobody in view'}",
                          flush=True)
                for port in (args.status_port, args.face_port, args.teleop_port):
                    relay.sendto(json.dumps(report).encode(), ("127.0.0.1", port))
        time.sleep(max(0.02, period - (time.monotonic() - started)))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
