#!/usr/bin/env python3
"""The robot's CAD model, light enough to draw in the headset.

The Onshape export is 26 binary STLs of about 44,000 triangles each (42 MB): too much
to send to a Quest every time the page loads. This simplifies every part once
(pymeshlab's quadric edge collapse) and writes two files the page loads in a moment:

  robot_model.json   the URDF's kinematic tree (joints, origins, axes, limits) and
                     where each part's triangles sit in the .bin
  robot_model.bin    flat-shaded triangles: float32 positions, then int8 normals

The page poses the arms itself from the joint angles run_teleop reports, so the
model moves with the real robot. quest_bridge.py serves both files and builds them
on first use; run this to rebuild by hand:

  .venv/bin/python scripts/teleop/robot_model.py [--force] [--triangles 1500]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ROBOT = REPO / "source/berkeley_humanoid_lite_assets/data/robots/berkeley_humanoid/berkeley_humanoid_lite"
URDF = ROBOT / "urdf/berkeley_humanoid_lite.urdf"
CACHE = Path.home() / ".cache/bhl"
FORMAT = 1                       # bump when the file layout changes


def floats(text: str | None, count: int = 3) -> list[float]:
    return [float(v) for v in (text or "").split()] if text else [0.0] * count


def origin_of(element) -> list[float]:
    """xyz + rpy of an <origin>, as six floats (zeros when absent)."""
    found = element.find("origin") if element is not None else None
    if found is None:
        return [0.0] * 6
    return floats(found.get("xyz")) + floats(found.get("rpy"))


def read_stl(path: Path) -> np.ndarray:
    """Triangles of a binary STL, (n, 3, 3) float32."""
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, 80)[0]
    record = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    return np.frombuffer(data, dtype=record, count=count, offset=84)["v"].astype(np.float32)


def simplify(triangles: np.ndarray, target: int) -> np.ndarray:
    """Fewer triangles, same shape: quadric edge collapse on the welded mesh."""
    if len(triangles) <= target:
        return triangles
    import pymeshlab

    flat = triangles.reshape(-1, 3).astype(np.float64)
    vertices, index = np.unique(np.round(flat, 7), axis=0, return_inverse=True)
    faces = index.reshape(-1, 3).astype(np.int32)
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
    meshes = pymeshlab.MeshSet()
    meshes.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces))
    meshes.meshing_decimation_quadric_edge_collapse(targetfacenum=int(target), preservenormal=True,
                                                    preserveboundary=True, planarquadric=True,
                                                    optimalplacement=True, qualitythr=0.3)
    mesh = meshes.current_mesh()
    return mesh.vertex_matrix()[mesh.face_matrix()].astype(np.float32)


def flat_normals(triangles: np.ndarray) -> np.ndarray:
    """One normal per triangle, repeated for its three corners: CAD parts have hard edges."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    n = np.cross(b - a, c - a)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return np.repeat(n[:, None, :], 3, axis=1)


def source_stamp(budget: int) -> str:
    """Changes whenever the URDF, a mesh, this file's format or the budget does."""
    digest = hashlib.sha1(f"{FORMAT}:{budget}".encode())
    for path in [URDF, *sorted((ROBOT / "meshes").glob("*.stl"))]:
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()[:12]


def build(budget: int = 1500, force: bool = False, quiet: bool = False) -> tuple[Path, Path]:
    """Write (or reuse) the model files; returns their paths."""
    CACHE.mkdir(parents=True, exist_ok=True)
    stamp = source_stamp(budget)
    out_json, out_bin = CACHE / "robot_model.json", CACHE / "robot_model.bin"
    if not force and out_json.exists() and out_bin.exists():
        try:
            if json.loads(out_json.read_text()).get("stamp") == stamp:
                return out_json, out_bin
        except ValueError:
            pass
    started = time.monotonic()
    root = ET.parse(URDF).getroot()
    links: dict[str, dict] = {}
    positions, normals = [], []
    first = 0
    for link in root.findall("link"):
        name = link.get("name")
        visual = link.find("visual")
        mesh = visual.find("geometry/mesh") if visual is not None else None
        entry = {"mesh": None, "origin": origin_of(visual), "color": [0.45, 0.47, 0.5, 1.0]}
        color = visual.find("material/color") if visual is not None else None
        if color is not None:
            entry["color"] = floats(color.get("rgba"), 4)
        if mesh is not None:
            path = (URDF.parent / mesh.get("filename").replace("package://", "")).resolve()
            triangles = read_stl(path)
            scale = floats(mesh.get("scale")) if mesh.get("scale") else [1.0, 1.0, 1.0]
            triangles = triangles * np.asarray(scale, dtype=np.float32)
            # the legs and body only give context; the arms carry the colours
            target = budget if name.startswith("arm_") else budget * 2 if name == "base" else budget // 2
            simple = simplify(triangles, target)
            positions.append(simple.reshape(-1, 3))
            normals.append(flat_normals(simple).reshape(-1, 3))
            entry["mesh"] = [first, len(simple) * 3]          # first vertex, vertex count
            first += len(simple) * 3
            if not quiet:
                print(f"  {name:32s} {len(triangles):6d} -> {len(simple):5d} triangles", flush=True)
        links[name] = entry
    joints = []
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        axis = joint.find("axis")
        joints.append({
            "name": joint.get("name"), "type": joint.get("type"),
            "parent": joint.find("parent").get("link"), "child": joint.find("child").get("link"),
            "origin": origin_of(joint), "axis": floats(axis.get("xyz")) if axis is not None else [1.0, 0.0, 0.0],
            "lower": float(limit.get("lower", 0)) if limit is not None else 0.0,
            "upper": float(limit.get("upper", 0)) if limit is not None else 0.0,
        })
    children = {j["child"] for j in joints}
    pos = np.concatenate(positions).astype("<f4")
    nrm = np.clip(np.round(np.concatenate(normals) * 127), -127, 127).astype(np.int8)
    nrm4 = np.zeros((len(nrm), 4), dtype=np.int8)                # padded to 4 bytes per vertex
    nrm4[:, :3] = nrm
    blob = pos.tobytes() + nrm4.tobytes()
    manifest = {
        "stamp": stamp, "format": FORMAT, "vertices": int(len(pos)),
        "positions": [0, int(pos.nbytes)], "normals": [int(pos.nbytes), int(nrm4.nbytes)],
        "root": next(name for name in links if name not in children),
        "links": links, "joints": joints,
    }
    out_bin.write_bytes(blob)
    out_json.write_text(json.dumps(manifest))
    if not quiet:
        print(f"robot model: {len(pos) // 3} triangles, {len(blob) / 1e6:.1f} MB, "
              f"in {time.monotonic() - started:.1f} s -> {out_json.parent}", flush=True)
    return out_json, out_bin


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="rebuild even if the cache is current")
    parser.add_argument("--triangles", type=int, default=1500, help="per arm part (legs get half)")
    args = parser.parse_args()
    build(args.triangles, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
