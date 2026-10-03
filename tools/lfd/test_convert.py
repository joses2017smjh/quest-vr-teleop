"""End to end: synthetic sessions -> convert_to_lerobot.py -> LeRobotDataset, checked against labels.py and numpy.

It writes one synthetic session of sixteen episodes: teleop, hand-guided, policy (with an intervention), a teleop
episode without camera servos, one with two frames sharing an mtime, one whose zero_check 'holds' without the fix,
and a last teleop one, which should convert; and nine the spec refuses (aborted, null zero_check, discarded, a frame
gap, torn JPEGs, a src that contradicts the mode, an incomplete episode, a 'report' zero_check, and a frame that does
not decode in the middle of the episode, refused while it is being written, before the last one). It converts it
with meas_future through the API (streaming encoding, the default) and with cmd through the CLI (the PNG path),
loads both with LeRobotDataset(repo_id, root=...), and checks:
  * fps, every episode's frame count, the feature shapes, names and dtypes (t_rel float32, intervention and
    cam_known bool), each eye's marker after the 180-degree turn and split, and that every step of an episode
    shows the frame labels.py chose (no frame lost by either encoder);
  * that state, action and every extra column equal labels.py's, and t0 in conversion.json;
  * meta/stats.json: strict JSON (no NaN even with the camera angle unknown), quantiles equal to numpy.quantile
    over the parquet columns, count, min and max equal to numpy's, and what a training run loads (the dataset
    opened with only the train episodes) holds those quantiles;
  * splits.json (whole episodes, per mode, at least one validation episode), conversion.json's rejections, episode
    fields (trial, arrangement, operator, policy, outcome, end_reason, notes, t0, grid counts) and code record;
  * the refusal to mix zero conventions, joint names, claw limits or frame sizes, and that LeRobot's own
    dataset_to_policy_features sees only observation.state, the two images and action;
  * the converter's guards: its video-length check, and its throttle and drop count for the streaming encoder.

Needs the Python 3.12 LeRobot 0.6.1 env with LeRobot's dataset extra (datasets, av, pandas, pyarrow). Everything
goes to a temporary directory under $TMPDIR, the datasets cache included, and is deleted at the end (set
LFD_TEST_KEEP=1 to keep it). About 45 s on 2 CPUs.

  /nfs/hpc/share/sanchej7/envs/lerobot-py312-cpu/bin/python tools/lfd/test_convert.py
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import warnings
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="lfd_test_convert_"))
os.environ.setdefault("HF_HOME", str(TMP / "hf"))           # never the home directory
os.environ["HF_DATASETS_CACHE"] = str(TMP / "hf_datasets")   # load-back writes arrow caches here
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("SVT_LOG", "2")
warnings.filterwarnings("ignore", message="Cannot enable progress bars")   # LeRobot tries to, after loading

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import convert_to_lerobot as conv  # noqa: E402
import labels  # noqa: E402
import synth_session as synth  # noqa: E402
from lerobot.configs import FeatureType  # noqa: E402
from lerobot.datasets import LeRobotDataset  # noqa: E402
from lerobot.utils.feature_utils import dataset_to_policy_features  # noqa: E402

FPS, H, EYE = 30, 0.10, (160, 120)
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"PASS  {name}" + (f"  [{detail}]" if detail and len(detail) < 160 else ""))
    else:
        print(f"FAIL  {name}" + (f": {detail}" if detail else ""))
        failures.append(name)


def columns(dataset: LeRobotDataset) -> tuple[dict[str, np.ndarray], dict]:
    """Every stored column exactly as stored, and its arrow type."""
    table = dataset.hf_dataset.with_format("arrow")[:]
    out, types_ = {}, {}
    for name in table.column_names:
        col = table.column(name).combine_chunks()
        types_[name] = col.type
        if pa.types.is_fixed_size_list(col.type):
            out[name] = col.flatten().to_numpy(zero_copy_only=False).reshape(len(col), col.type.list_size)
        else:
            out[name] = col.to_numpy(zero_copy_only=False)
    return out, types_


def episode_rows(col: dict, i: int) -> np.ndarray:
    rows = np.flatnonzero(col["episode_index"] == i)
    return rows[np.argsort(col["frame_index"][rows])]


def hwc(image) -> np.ndarray:
    """A decoded (3, H, W) float tensor in [0, 1] as (H, W, 3) uint8."""
    return np.clip(np.round(image.permute(1, 2, 0).numpy() * 255), 0, 255).astype(np.uint8)


def box_mean(image: np.ndarray, box) -> np.ndarray:
    x0, y0, x1, y1 = box
    m = max(1, (x1 - x0) // 5)                          # stay off the edges, where video blurs colours
    return image[y0 + m:y1 - m, x0 + m:x1 - m].reshape(-1, 3).mean(0)


def refused(fn, *args, **kwargs) -> str | None:
    try:
        fn(*args, **kwargs)
    except conv.ConversionRefused as exc:
        return str(exc)
    return None


def strict_json(text: str):
    def no_constants(value):
        raise ValueError(f"{value} in strict JSON")
    return json.loads(text, parse_constant=no_constants)


def edit_json(path: Path, **changes) -> None:
    record = json.loads(path.read_text())
    record.update(changes)
    path.write_text(json.dumps(record))


def edit_row(ep_dir: Path, row: int, **changes) -> None:
    lines = (ep_dir / "rows.jsonl").read_text().splitlines()
    record = json.loads(lines[row])
    record.update(changes)
    lines[row] = json.dumps(record)
    (ep_dir / "rows.jsonl").write_text("\n".join(lines) + "\n")


def frame_index(ep_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (ep_dir / "frames.jsonl").read_text().splitlines()]


SPECS = [
    synth.EpisodeSpec(mode="teleop", seconds=3.0, trial=3, arrangement="A07", operator="rev", notes="first"),  # 0
    synth.EpisodeSpec(mode="guided", seconds=2.5, task="Fold the cloth in half"),          # 1: taken (not by cmd)
    synth.EpisodeSpec(mode="teleop", seconds=2.0, servos=False),                           # 2: cam unknown
    synth.EpisodeSpec(mode="teleop", seconds=1.5, outcome="aborted"),                      # 3: aborted
    synth.EpisodeSpec(mode="policy", seconds=2.5, policy="act_v1", trial=17, interventions=((0.8, 1.6),)),   # 4
    synth.EpisodeSpec(mode="teleop", seconds=1.5, zero_check=False),                       # 5: zero unverified
    synth.EpisodeSpec(mode="teleop", seconds=1.5, discarded=True),                         # 6: discarded
    synth.EpisodeSpec(mode="teleop", seconds=3.0, frame_gaps=((1.0, 0.3),)),               # 7: a true gap
    synth.EpisodeSpec(mode="teleop", seconds=1.5),                                         # 8: torn JPEGs
    synth.EpisodeSpec(mode="teleop", seconds=2.0, same_mtime=(20,)),                       # 9: one mtime tick
    synth.EpisodeSpec(mode="teleop", seconds=1.5),                                         # 10: a guide row
    synth.EpisodeSpec(mode="teleop", seconds=1.5),                                         # 11: incomplete
    synth.EpisodeSpec(mode="teleop", seconds=1.5),                                         # 12: zero_check report
    synth.EpisodeSpec(mode="teleop", seconds=1.5),                                         # 13: holds, no fix
    synth.EpisodeSpec(mode="teleop", seconds=1.5),                                         # 14: a frame won't decode
    synth.EpisodeSpec(mode="teleop", seconds=1.5),          # 15: taken after a refusal in the middle of writing
]


def build_session(recordings: Path) -> Path:
    session = synth.write_session(recordings, SPECS, seed=0)
    torn = session / "ep_0008"
    with open(torn / "frames.mjpeg", "r+b") as fh:                 # lose the EOI marker of six frames
        for frame in frame_index(torn)[10:16]:
            fh.seek(frame["off"] + frame["len"] - 2)
            fh.write(b"\x00\x00")
    edit_row(session / "ep_0010", 30, src="guide")
    meta = json.loads((session / "ep_0011/episode.json").read_text())
    edit_json(session / "ep_0011/episode.json", outcome=None, end=None, end_reason=None,
              counts={"rows": 0, "frames": 0})
    report_check = dict(meta["zero_check"], decision="report")
    edit_json(session / "ep_0012/episode.json", zero_check=report_check)
    edit_json(session / "ep_0013/episode.json", zero_check=dict(meta["zero_check"], decision="holds",
                                                                zero_fix_applied=False))
    broken = session / "ep_0014"                                    # SOF says 0 components: header fine, no decode
    frame = frame_index(broken)[20]
    data = bytearray((broken / "frames.mjpeg").read_bytes())
    i = frame["off"] + 2
    while data[i + 1] != 0xC0:
        i += 2 + int.from_bytes(data[i + 2:i + 4], "big")
    data[i + 9] = 0
    (broken / "frames.mjpeg").write_bytes(bytes(data))
    return session


def main() -> int:
    print(f"working in {TMP}")
    session = build_session(TMP / "recordings")
    sid = session.name
    out_root = TMP / "lerobot"

    # ------------------------------------------------------------------ meas_future through the API, streaming
    started = time.time()
    conv.convert([session], out_root, "local/test_mf", "meas_future", fps=FPS, lookahead=H,
                 modes=("teleop", "guided", "policy"), val_fraction=0.5, seed=0, image_size=EYE)
    print(f"      (meas_future conversion: {time.time() - started:.1f} s)")
    root = out_root / "local/test_mf"
    conversion = strict_json((root / "conversion.json").read_text())
    rejected = {r["source"]: r["reason"] for r in conversion["rejected"]}
    want = {f"{sid}/ep_0003": "aborted", f"{sid}/ep_0005": "zero_check is null",
            f"{sid}/_discarded/ep_0006": "discarded", f"{sid}/ep_0007": "no frame within",
            f"{sid}/ep_0008": "not a whole JPEG", f"{sid}/ep_0010": "contradict",
            f"{sid}/ep_0011": "not finalized", f"{sid}/ep_0012": "zero_check decided 'report'",
            f"{sid}/ep_0014": "does not decode"}
    check("conversion.json lists every refused episode with its reason (a true gap, src against the mode, an "
          "incomplete episode, a 'report' zero check, a frame that does not decode mid-episode, ...)",
          set(rejected) == set(want) and all(want[s] in rejected[s] for s in want), json.dumps(rejected, indent=1))
    taken = ["ep_0000", "ep_0001", "ep_0002", "ep_0004", "ep_0009", "ep_0013", "ep_0015"]
    sources = [e["source"] for e in conversion["episodes"]]
    check("conversion.json maps dataset episodes to their sources, numbered without the refused ones",
          sources == [f"{sid}/{name}" for name in taken] and [e["index"] for e in conversion["episodes"]]
          == list(range(len(taken))), str(sources))
    inputs = {e["source"]: e for s in conversion["inputs"] for e in s["episodes"]}
    ep0 = session / "ep_0000"
    check("conversion.json carries the sha256 of every episode.json and rows.jsonl",
          len(inputs) == len(SPECS) and inputs[f"{sid}/ep_0000"]["episode_json_sha256"]
          == conv.sha256_file(ep0 / "episode.json")
          and inputs[f"{sid}/ep_0000"]["rows_jsonl_sha256"] == conv.sha256_file(ep0 / "rows.jsonl")
          and all(e["rows_jsonl_sha256"] for e in inputs.values()))
    check("conversion.json records the label, lookahead, args, thresholds and LeRobot version",
          conversion["label"] == "meas_future" and conversion["lookahead_s"] == H
          and conversion["args"]["image_size"] == list(EYE) and conversion["lerobot_version"] == "0.6.1"
          and conversion["args"]["max_missing"] == 0.02 and conversion["args"]["max_reused_skipped"] == 0.05)
    first, policy_ep = conversion["episodes"][0], conversion["episodes"][3]
    fields = ("trial", "arrangement", "operator", "policy", "outcome", "end_reason", "notes", "t0")
    check("conversion.json episodes carry trial, arrangement, operator, policy, outcome, end_reason, notes and t0",
          all(all(k in e for k in fields) for e in conversion["episodes"])
          and (first["trial"], first["arrangement"], first["operator"], first["notes"], first["outcome"],
               first["end_reason"], first["policy"]) == (3, "A07", "rev", "first", "success", "operator", None)
          and (policy_ep["policy"], policy_ep["trial"], policy_ep["mode"]) == ("act_v1", 17, "policy"),
          json.dumps({k: first[k] for k in fields}))
    holds = conversion["episodes"][5]
    check("a zero_check that holds without the fix converts, with the spec's warning in conversion.json",
          holds["source"].endswith("ep_0013") and any("gravity model" in w for w in holds["warnings"]),
          str(holds["warnings"]))
    code = conversion["code"]
    check("conversion.json records the code: commit, dirty flag, tools/lfd status, and the sha256 of the "
          "converter, labels.py and splits.py",
          isinstance(code["repo_dirty"], bool) and isinstance(code["tools_lfd_status"], list)
          and code["splits_sha256"] == conv.sha256_file(HERE / "splits.py")
          and code["labels_sha256"] == conv.sha256_file(HERE / "labels.py")
          and code["convert_to_lerobot_sha256"] == conv.sha256_file(HERE / "convert_to_lerobot.py"), str(code))
    enc = conversion["encoding"]
    check("conversion.json records the encoding: streaming by default, and the video settings",
          enc["streaming_encoding"] is True and enc["video"][conv.LEFT].get("video.codec") == "av1"
          and enc["encoder_threads"] == max(1, conv.usable_cpus() // 2), json.dumps(enc)[:300])

    ds = LeRobotDataset("local/test_mf", root=root)
    expected = []
    for e in conversion["episodes"]:
        with labels.load_episode(Path(e["episode_dir"])) as episode:
            expected.append((episode, labels.build_steps(episode, FPS, "meas_future", H, allow_policy=True)))
    check("conversion.json t0 is each episode's first-step monotonic time, and its grid counts are labels.py's",
          all(e["t0"] == s.t0 and e["grid"] == {k: s.report["grid"][k] for k in ("steps", "missing", "reused",
                                                                                  "skipped")}
              for e, (_, s) in zip(conversion["episodes"], expected)))
    lengths = [int(ds.meta.episodes[i]["length"]) for i in range(ds.meta.total_episodes)]
    check("fps is 30", ds.fps == FPS and ds.meta.fps == FPS)
    check("frame counts per episode equal labels.py's steps", lengths == [len(s.t) for _, s in expected],
          f"{lengths} vs {[len(s.t) for _, s in expected]}")
    feats = ds.meta.features
    names = list(synth.DEFAULT_JOINT_NAMES) + list(labels.CLAW_NAMES)
    check("observation.state and action are float32[12] with the joint and claw names",
          all(feats[k]["dtype"] == "float32" and tuple(feats[k]["shape"]) == (12,) and feats[k]["names"] == names
              for k in ("observation.state", "action")))
    check("both eyes are video of 120x160x3", all(feats[k]["dtype"] == "video" and tuple(feats[k]["shape"])
                                                   == (120, 160, 3) for k in (conv.LEFT, conv.RIGHT)))
    shapes = {"q_cmd": ((10,), "float32"), "cam_pan_tilt": ((2,), "float32"), "cam_known": ((1,), "bool"),
              "mode_id": ((1,), "int64"), "intervention": ((1,), "bool"), "t_rel": ((1,), "float32"),
              "frame_skew": ((1,), "float32")}
    check("the extra columns are there with the spec's shapes and dtypes (t_rel float32, intervention and "
          "cam_known bool), none under 'observation.', and no t_mono",
          all((tuple(feats[k]["shape"]), feats[k]["dtype"]) == v for k, v in shapes.items())
          and not any(k.startswith("observation") for k in shapes) and "t_mono" not in feats,
          str({k: (feats[k]["shape"], feats[k]["dtype"]) for k in shapes if k in feats}))
    item = ds[0]
    check("a loaded item: images (3, 120, 160), state and action (12,), the task",
          tuple(item[conv.LEFT].shape) == (3, 120, 160) and tuple(item[conv.RIGHT].shape) == (3, 120, 160)
          and tuple(item["observation.state"].shape) == (12,) and tuple(item["action"].shape) == (12,)
          and item["task"] == SPECS[0].task)
    tasks = [ds[int(ds.meta.episodes[i]["dataset_from_index"])]["task"] for i in range(3)]
    check("each episode keeps its own task", tasks == [SPECS[0].task, SPECS[1].task, SPECS[2].task], str(tasks))

    # values: the stored columns against labels.py
    col, arrow = columns(ds)
    ok_state = ok_extra = True
    detail = ""
    for i, (_episode, steps) in enumerate(expected):
        rows = episode_rows(col, i)
        same = (np.array_equal(col["observation.state"][rows], steps.state)
                and np.array_equal(col["action"][rows], steps.action))
        extra = (np.array_equal(col["t_rel"][rows], steps.t_rel)
                 and np.array_equal(col["frame_skew"][rows], steps.frame_skew.astype(np.float32))
                 and np.array_equal(col["q_cmd"][rows], steps.q_cmd)
                 and np.all(col["mode_id"][rows] == steps.mode_id)
                 and np.array_equal(col["intervention"][rows], steps.intervention)
                 and np.array_equal(col["cam_known"][rows], steps.cam_known)
                 and np.array_equal(col["cam_pan_tilt"][rows], steps.cam_pan_tilt))
        ok_state &= same
        ok_extra &= extra
        if not (same and extra):
            detail += f" episode {i}: state/action {same}, extras {extra};"
    check("state and action equal labels.py exactly (meas_future, H = 0.10 s)", ok_state, detail)
    check("q_cmd, cam_pan_tilt, cam_known, mode_id, intervention, t_rel and frame_skew equal labels.py", ok_extra,
          detail)
    check("t_rel is stored as float32 and intervention and cam_known as bool",
          arrow["t_rel"] == pa.float32() and arrow["intervention"] == pa.bool_() and arrow["cam_known"] == pa.bool_()
          and col["intervention"].dtype == np.bool_, str({k: str(arrow[k]) for k in ("t_rel", "intervention")}))
    check("t_rel restarts at 0 in every episode, 1/fps apart",
          all(col["t_rel"][episode_rows(col, i)][0] == 0.0
              and np.allclose(np.diff(col["t_rel"][episode_rows(col, i)]), 1 / FPS, atol=2e-6)
              for i in range(len(expected))))
    check("mode_id is 0 for teleop, 1 for guided and 2 for policy episodes",
          [sorted(set(col["mode_id"][col["episode_index"] == i])) for i in range(4)] == [[0], [1], [0], [2]])
    pol = expected[3][1]
    check("the policy episode's intervention steps are stored as true",
          int(col["intervention"][episode_rows(col, 3)].sum()) == int(pol.intervention.sum()) > 10)
    check("without servos the camera angle is stored as 0 with cam_known false; elsewhere the commanded angle",
          np.all(col["cam_pan_tilt"][col["episode_index"] == 2] == 0.0)
          and not col["cam_known"][col["episode_index"] == 2].any()
          and np.all(col["cam_pan_tilt"][col["episode_index"] == 0] == np.float32(synth.CAM_ANGLE))
          and col["cam_known"][col["episode_index"] == 0].all())

    # meta/stats.json: strict JSON, numpy's quantiles
    text = (root / "meta/stats.json").read_text()
    try:
        stats = strict_json(text)
        strict = "NaN" not in text and "Infinity" not in text
    except ValueError as exc:
        stats, strict = json.loads(text), False
        print(f"      stats.json: {exc}")
    check("meta/stats.json is strict JSON (no NaN or Infinity) although one episode has no camera angle", strict)
    frames = pd.concat([pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet"))], ignore_index=True)
    worst, keys_seen, exact_other = 0.0, set(), True
    for key in ("observation.state", "action"):
        values = np.stack(frames[key].to_numpy()).astype(np.float64)
        for name, stat in stats[key].items():
            if name.startswith("q") and name[1:].isdigit():
                keys_seen.add(name)
                worst = max(worst, float(np.abs(np.asarray(stat) - np.quantile(values, int(name[1:]) / 100,
                                                                               axis=0)).max()))
        exact_other &= (stats[key]["count"] == [len(values)]
                        and np.array_equal(np.asarray(stats[key]["min"]), values.min(0))
                        and np.array_equal(np.asarray(stats[key]["max"]), values.max(0)))
    check("the stats quantiles of state and action equal numpy.quantile over every frame",
          keys_seen == {"q01", "q10", "q50", "q90", "q99"} and worst < 1e-12, f"max |diff| {worst:.1e}, {keys_seen}")
    check("their count, min and max equal numpy's", bool(exact_other))
    loaded = ds.meta.stats["action"]["q01"]
    check("LeRobot loads the recomputed quantiles",
          np.array_equal(np.asarray(loaded, dtype=np.float64), stats["action"]["q01"]))
    check("conversion.json records that the stats were recomputed",
          conversion["stats"]["recomputed"] is True
          and conversion["stats"]["quantiles"]["keys"]["action"]["quantiles_replaced"] == ["q01", "q10", "q50",
                                                                                          "q90", "q99"])

    # images: each eye's marker where it belongs, and the right frame at each step
    boxes = synth.marker_boxes(*EYE)
    colours, codes = [], []
    for i, (episode, steps) in enumerate(expected):
        start = int(ds.meta.episodes[i]["dataset_from_index"])
        every = range(len(steps.t)) if i == 4 else (0, len(steps.t) // 2, len(steps.t) - 1)
        for k in every:
            frame = ds[start + k]
            left, right = hwc(frame[conv.LEFT]), hwc(frame[conv.RIGHT])
            lc, rc = box_mean(left, boxes["left_marker"]), box_mean(right, boxes["right_marker"])
            colours.append(lc[0] > 170 and lc[2] < 90 and rc[2] > 170 and rc[0] < 90)
            codes.append((synth.read_code(left), int(episode.frames["i"][steps.frame[k]]) % (1 << synth.CODE_BITS)))
    check("after the turn and split, the left eye has the red marker at its upper left and the right eye the blue "
          "one at its lower right", all(colours), str(colours))
    check("each sampled step shows the frame labels.py chose, and so does every step of the same-mtime episode "
          "(the streaming encoder lost no frame)", all(a == b for a, b in codes),
          str([c for c in codes if c[0] != c[1]][:10]))

    # splits: whole episodes, per mode, at least one validation episode
    split = json.loads((root / "splits.json").read_text())
    train, val = set(split["train"]), set(split["val"])
    modes = {e["index"]: e["mode"] for e in conversion["episodes"]}
    check("splits.json is disjoint, whole-episode and covers every episode",
          split["unit"] == "episode" and not train & val and train | val == set(modes)
          and all(modes[i] == m for m, part in split["by_mode"].items() for i in part["train"] + part["val"]),
          json.dumps(split["by_mode"]))
    sizes = {m: sum(1 for v in modes.values() if v == m) for m in set(modes.values())}
    check("splits.json takes round(0.5 * n) per mode, at most n - 1 (5 teleop -> 2: Python rounds the tie 2.5 to "
          "even; 1 guided, 1 policy -> 0)",
          {m: len(p["val"]) for m, p in split["by_mode"].items()}
          == {m: min(round(0.5 * n), n - 1) for m, n in sizes.items()} and len(split["by_mode"]["teleop"]["val"])
          == 2, json.dumps(split["by_mode"]))
    sub = LeRobotDataset("local/test_mf", root=root, episodes=split["train"])
    check("a training run's load (only the train episodes, as --dataset.episodes does) sees the recomputed "
          "quantiles of state and action",
          sub.meta.total_episodes == ds.meta.total_episodes and len(sub) == sum(lengths[i] for i in split["train"])
          and all(np.array_equal(np.asarray(sub.meta.stats[k][q], dtype=np.float64), stats[k][q])
                  for k in ("observation.state", "action") for q in ("q01", "q10", "q50", "q90", "q99")),
          f"{len(sub)} frames")
    del sub

    # LeRobot's own helper: what a policy would take as inputs and outputs
    policy = dataset_to_policy_features(ds.meta.features)
    check("dataset_to_policy_features gives only observation.state, the two eyes and action",
          set(policy) == {"observation.state", conv.LEFT, conv.RIGHT, "action"}
          and policy["observation.state"].type == FeatureType.STATE and policy["action"].type == FeatureType.ACTION
          and policy[conv.LEFT].type == FeatureType.VISUAL and tuple(policy[conv.LEFT].shape) == (3, 120, 160),
          str(policy))
    check("a second run does not overwrite the dataset without --overwrite",
          "exists" in (refused(conv.convert, [session], out_root, "local/test_mf", "meas_future", image_size=EYE,
                               log=lambda *a: None) or ""))
    del ds

    # the converter's guards
    fake = TMP / "fake_meta"
    shutil.copytree(root / "meta/episodes", fake / "meta/episodes")
    check("the video-length check passes on the written dataset", conv.video_length_problems(root, FPS,
                                                                                            (conv.LEFT,)) == [])
    meta_file = next((fake / "meta/episodes").rglob("*.parquet"))
    table = pd.read_parquet(meta_file)
    table.loc[1, f"videos/{conv.LEFT}/to_timestamp"] -= 1 / FPS                 # one frame short
    table.to_parquet(meta_file)
    found = conv.video_length_problems(fake, FPS, (conv.LEFT,))
    check("the video-length check catches an episode whose video is a frame short",
          len(found) == 1 and "episode 1" in found[0], str(found))
    q = queue.Queue(maxsize=1)
    q.put("frame")
    encoder = types.SimpleNamespace(_frame_queues={conv.LEFT: q}, _dropped_frames={conv.LEFT: 0})
    fake_ds = types.SimpleNamespace(writer=types.SimpleNamespace(_streaming_encoder=encoder))
    threading.Timer(0.3, q.get).start()
    tick = time.monotonic()
    conv.wait_for_encoder_room(fake_ds)
    waited = time.monotonic() - tick
    encoder._dropped_frames[conv.LEFT] = 2
    check("the streaming throttle waits while an encoder queue is full, and the drop count is read",
          0.25 < waited < 5 and not q.full() and conv.dropped_frames(fake_ds) == 2, f"waited {waited:.2f} s")

    # ------------------------------------------------------------------ cmd through the CLI, the PNG path
    cli = [sys.executable, str(HERE / "convert_to_lerobot.py"), "--sessions", str(session), "--out-root",
           str(out_root), "--repo-id", "local/test_cmd", "--label", "cmd", "--image-size", "160x120",
           "--no-streaming-encoding", "--image-writer-threads", "2"]
    run = subprocess.run(cli, capture_output=True, text=True, timeout=900)
    check("the CLI converts with --label cmd through the PNG path (exit 0)", run.returncode == 0, run.stderr[-2000:])
    if run.returncode == 0:
        root = out_root / "local/test_cmd"
        conversion = strict_json((root / "conversion.json").read_text())
        rejected = {r["source"]: r["reason"] for r in conversion["rejected"]}
        check("cmd refuses the hand-guided episode as teleop-only, and the policy one unless asked for",
              "teleop-only" in rejected.get(f"{sid}/ep_0001", "")
              and "policy episodes are excluded" in rejected.get(f"{sid}/ep_0004", ""),
              json.dumps(rejected, indent=1))
        enc = conversion["encoding"]
        check("conversion.json records the PNG path: no streaming, 2 image writer threads",
              enc["streaming_encoding"] is False and enc["image_writer_threads"] == 2, json.dumps(enc)[:200])
        ds = LeRobotDataset("local/test_cmd", root=root)
        col, _ = columns(ds)
        good = ds.meta.total_episodes == 5
        codes = []
        for i, e in enumerate(conversion["episodes"]):
            with labels.load_episode(Path(e["episode_dir"])) as episode:
                steps = labels.build_steps(episode, FPS, "cmd")
                start = int(ds.meta.episodes[i]["dataset_from_index"])
                for k in range(0, len(steps.t), 7):
                    code = int(episode.frames["i"][steps.frame[k]]) % (1 << synth.CODE_BITS)
                    codes.append(synth.read_code(hwc(ds[start + k][conv.LEFT])) == code)
            rows = episode_rows(col, i)
            good &= (np.array_equal(col["action"][rows], steps.action)
                     and np.array_equal(col["observation.state"][rows], steps.state))
        check("cmd dataset: five teleop episodes whose action is labels.py's held qc", bool(good))
        check("cmd dataset: every 7th step shows the frame labels.py chose (PNG path)", all(codes) and len(codes) > 20,
              f"{sum(codes)} of {len(codes)}")
        split = json.loads((root / "splits.json").read_text())
        check("splits.json of 5 teleop episodes at val_fraction 0.2 has one validation episode",
              len(split["val"]) == 1 and len(split["train"]) == 4, json.dumps(split["by_mode"]))
        strict_json((root / "meta/stats.json").read_text())
        del ds

    # ------------------------------------------------------------------ mixing refusals
    def one(seed, **kw):
        joint_names = kw.pop("joint_names", synth.DEFAULT_JOINT_NAMES)
        size = kw.pop("frame_size", synth.FRAME_SIZE)
        return synth.write_session(TMP / f"mix{seed}", [synth.EpisodeSpec(seconds=1.5, **kw)], seed=seed,
                                   joint_names=joint_names, frame_size=size)

    cases = {
        "zero_convention": one(1, zero_convention="hanging=q_hang"),
        "joint_names": one(2, joint_names=synth.RENAMED_JOINT_NAMES),
        "claw_limits": one(3, claw_limits={"left": {"open_us": 1000, "closed_us": 1600},
                                           "right": {"open_us": 1000, "closed_us": 1670}}),
        "frame_size": one(4, frame_size=(2560, 720)),
    }
    for key, other in cases.items():
        repo = f"local/mix_{key}"
        reason = refused(conv.convert, [session, other], out_root, repo, "meas_future", image_size=EYE,
                         log=lambda *a: None)
        check(f"mixing different {key} is refused, and nothing is written",
              reason is not None and f"different {key}" in reason and not (out_root / repo).exists(), str(reason))
    run = subprocess.run([sys.executable, str(HERE / "convert_to_lerobot.py"), "--sessions", str(session),
                          str(cases["zero_convention"]), "--out-root", str(out_root), "--repo-id", "local/mix_cli",
                          "--label", "meas_future"], capture_output=True, text=True, timeout=600)
    check("the CLI exits 1 on a mixing refusal and says why",
          run.returncode == 1 and "different zero_convention" in run.stderr, run.stderr[-1000:])
    alone = refused(conv.convert, [cases["zero_convention"]], out_root, "local/alone", "meas_future",
                    image_size=EYE, log=lambda *a: None)
    check("one session alone with the other convention is fine (it is the mixing that is refused)",
          alone is None and (out_root / "local/alone/meta/info.json").exists(), str(alone))
    single = strict_json((out_root / "local/alone/splits.json").read_text())
    check("a dataset of one episode has no validation episode", single["val"] == [] and single["train"] == [0])
    stranger = out_root / "local/not_ours"
    stranger.mkdir(parents=True)
    (stranger / "notes.txt").write_text("keep me")
    check("--overwrite will not delete a directory that is not a dataset this tool wrote",
          "not a dataset" in (refused(conv.convert, [session], out_root, "local/not_ours", "meas_future",
                                      image_size=EYE, overwrite=True, log=lambda *a: None) or "")
          and (stranger / "notes.txt").exists())

    print()
    for name in failures:
        print("FAIL  " + name)
    print("conversion is sound" if not failures else f"{len(failures)} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        if os.environ.get("LFD_TEST_KEEP"):
            print(f"kept {TMP}")
        else:
            shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(code)
