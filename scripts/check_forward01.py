"""Build a local Forward-01 model and audit real EEF targets with offline IK.

Source assets and the existing model generator are read only. Derived MJCF,
source hashes, numerical IK candidates and preview movies stay under logs.
This is an initial-pose mapping proposal, not a calibrated robot controller.
"""

import argparse
import hashlib
import json
import runpy
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from specialist.pose_actions import inverse

Array = npt.NDArray[np.float64]
ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def build_model(
  source: Path, generator: Path, output: Path
) -> tuple[Any, dict[str, Any]]:
  urdf = (
    source
    / "assets/assemblies/forward-01-flexiv-rizon4s/urdf/forward-01-flexiv-rizon4s-taccap.urdf"
  )
  entry = runpy.run_path(str(generator))["build"]
  entry.__globals__.update(SNAPSHOT=source, URDF=urdf)
  products, config, derived = entry()
  for name, data in products.items():
    path = output / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
  scene = output / "flexiv_forward_01_scene.xml"
  root = ET.parse(scene).getroot()
  asset = root.find("asset")
  assert asset is not None
  bodies = {b.get("name"): b for b in root.iter("body")}
  source_root = ET.parse(urdf).getroot()
  collision_sources: list[dict[str, Any]] = []
  for link in source_root.findall("link"):
    for collision in link.findall("collision"):
      mesh = collision.find("geometry/mesh")
      if mesh is None:
        raise ValueError("Collision exporter currently requires source mesh geometry")
      path = (urdf.parent / mesh.attrib["filename"]).resolve()
      if not path.is_relative_to(source):
        raise ValueError("Mesh outside selected source root")
      key = f"collision_{len(collision_sources)}"
      filename = f"collision/{digest(path)[:16]}{path.suffix.lower()}"
      dest = output / filename
      dest.parent.mkdir(exist_ok=True)
      dest.write_bytes(path.read_bytes())
      ET.SubElement(
        asset, "mesh", name=key, file=filename, scale=mesh.get("scale", "1 1 1")
      )
      origin = collision.find("origin")
      xyz = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
      rpy = [
        float(v)
        for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()
      ]
      xyzw = Rotation.from_euler("xyz", rpy).as_quat()
      quat = " ".join(map(str, xyzw[[3, 0, 1, 2]]))
      ET.SubElement(
        bodies[link.get("name")],
        "geom",
        name=key,
        type="mesh",
        mesh=key,
        pos=xyz,
        quat=quat,
        contype="1",
        conaffinity="1",
        group="3",
        mass="0",
        rgba=".8 .2 .1 .3",
      )
      collision_sources.append(
        {
          "link": link.get("name"),
          "source": str(path.relative_to(source)),
          "sha256": digest(path),
          "origin_xyz": xyz,
          "origin_rpy_rad": rpy,
        }
      )
  audited = output / "forward01-collision-audit.xml"
  ET.indent(root)
  ET.ElementTree(root).write(audited, encoding="utf-8", xml_declaration=True)
  model = mujoco.MjModel.from_xml_path(str(audited))
  metadata = {
    "source_root": str(source),
    "urdf": str(urdf),
    "urdf_sha256": digest(urdf),
    "generator": str(generator),
    "generator_sha256": digest(generator),
    "source_start_pose": str(urdf.parents[1] / "config/start_pose.taccap.json"),
    "start_pose_sha256": digest(urdf.parents[1] / "config/start_pose.taccap.json"),
    "visual_sources": derived,
    "collision_sources": collision_sources,
    "collision_geoms": len(collision_sources),
    "config": config,
    "model": str(audited),
    "limits": [
      "Only source-declared collision meshes; platform structure has no supplied collision geometry",
      "MuJoCo convex mesh contact and default parent/weld filtering; not exact mesh clearance",
      "No mass/friction tuning, actuation or physical calibration; mj_forward only",
    ],
  }
  return model, metadata


def fk(model: Any, data: Any, body: int) -> Array:
  mujoco.mj_kinematics(model, data)
  mujoco.mj_comPos(model, data)
  pose = np.eye(4)
  pose[:3, :3] = data.xmat[body].reshape(3, 3)
  pose[:3, 3] = data.xpos[body]
  return pose


def solve(
  model: Any,
  data: Any,
  body: int,
  joints: list[int],
  target: Array,
  seed: Array,
  max_iterations: int = 80,
) -> tuple[Array, float, float]:
  qadr = np.asarray(model.jnt_qposadr[joints])
  dofs = np.asarray(model.jnt_dofadr[joints])
  low, high = model.jnt_range[joints].T
  q = np.clip(seed, low, high).copy()
  best = q.copy()
  best_cost = np.inf
  jp = np.zeros((3, model.nv))
  jr = np.zeros((3, model.nv))
  for _ in range(max_iterations):
    data.qpos[qadr] = q
    current = fk(model, data, body)
    p = target[:3, 3] - current[:3, 3]
    r = Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec()
    cost = float(p @ p + 0.04 * (r @ r))
    if cost < best_cost:
      best_cost = cost
      best = q.copy()
    if np.linalg.norm(p) < 0.001 and np.linalg.norm(r) < np.deg2rad(0.5):
      break
    mujoco.mj_jacBody(model, data, jp, jr, body)
    jac = np.vstack([jp[:, dofs], 0.2 * jr[:, dofs]])
    error = np.concatenate([p, 0.2 * r])
    step = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6) * 1e-5, error)
    step *= min(1.0, 0.15 / max(float(np.abs(step).max()), 1e-12))
    new = np.clip(q + step, low, high)
    if np.max(np.abs(new - q)) < 1e-8:
      break
    q = new
  data.qpos[qadr] = best
  actual = fk(model, data, body)
  return (
    best,
    float(np.linalg.norm(actual[:3, 3] - target[:3, 3])),
    float(Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).magnitude()),
  )


def main() -> None:
  cli = argparse.ArgumentParser(description=__doc__)
  cli.add_argument(
    "--source-root", type=Path, default=ROOT.parent / "xense-description"
  )
  cli.add_argument(
    "--generator",
    type=Path,
    default=ROOT.parent / "xense-description-sim/tools/build_flexiv_model.py",
  )
  cli.add_argument(
    "--poses", type=Path, default=Path("logs/014-processing/poses/right-poses.npz")
  )
  cli.add_argument(
    "--output-dir", type=Path, default=Path("logs/pretrain-20261117/forward01")
  )
  cli.add_argument("--stride", type=int, default=1)
  args = cli.parse_args()
  if args.stride < 1:
    raise SystemExit("stride must be positive")
  args.output_dir.mkdir(parents=True, exist_ok=True)
  model, metadata = build_model(
    args.source_root.resolve(), args.generator, args.output_dir
  )
  data = mujoco.MjData(model)
  for joint in metadata["config"]["joints"]:
    data.qpos[model.joint(joint["name"]).qposadr] = joint["home"]
  home = data.qpos.copy()
  mujoco.mj_forward(model, data)
  baseline_pairs = {
    tuple(sorted(model.body(int(model.geom_bodyid[g])).name for g in c.geom))
    for c in data.contact
    if c.dist < -1e-5
  }
  body = model.body("right_tool_taccap_eef_link").id
  robot_initial = fk(model, data, body)
  with np.load(args.poses) as stored:
    source_poses = stored["world_eef"]
    initial = stored["world_T_initial"]
    times = stored["time_s"]
    opening = stored["opening_rad"]
  # Fixed initial EEF alignment; axes identity remains an uncalibrated proposal.
  targets = robot_initial @ inverse(initial) @ source_poses
  indices = np.arange(0, len(times), args.stride)
  joints = [model.joint(f"right_joint{i}").id for i in range(1, 8)]
  qadr = model.jnt_qposadr[joints]
  seed = home[qadr].copy()
  qpos = []
  achieved = []
  position = []
  angle = []
  contacts = []
  for number, row in enumerate(indices):
    seed, pe, re = solve(model, data, body, joints, targets[row], seed)
    qpos.append(data.qpos.copy())
    achieved.append(fk(model, data, body))
    position.append(pe)
    angle.append(re)
    mujoco.mj_forward(model, data)
    pairs = []
    for contact in data.contact:
      if contact.dist < -1e-5:
        a, b = [model.body(int(model.geom_bodyid[g])).name for g in contact.geom]
        pairs.append([a, b, float(contact.dist)])
    contacts.append(pairs)
    if number % 500 == 0:
      print(f"IK {number}/{len(indices)}", flush=True)
  qpos_array = np.asarray(qpos)
  position_array = np.asarray(position)
  angle_array = np.asarray(angle)
  successful = (position_array <= 0.002) & (angle_array <= np.deg2rad(1))
  velocity = np.abs(
    np.diff(qpos_array[:, qadr], axis=0) / np.diff(times[indices])[:, None]
  )
  velocity_limits = np.array(
    [
      next(
        j["maxVelocity"]
        for j in metadata["config"]["joints"]
        if j["name"] == f"right_joint{i}"
      )
      for i in range(1, 8)
    ]
  )
  collisions = np.array([bool(p) for p in contacts])
  additional_collisions = np.array(
    [
      any(tuple(sorted(pair[:2])) not in baseline_pairs for pair in rows)
      for rows in contacts
    ]
  )
  np.savez_compressed(
    args.output_dir / "ik-candidates.npz",
    time_s=times[indices],
    source_indices=indices,
    source_world_eef=source_poses[indices],
    target=targets[indices],
    achieved=np.asarray(achieved),
    qpos=qpos_array,
    position_error_m=position_array,
    rotation_error_rad=angle_array,
    ik_valid=successful,
    collision_detected=collisions,
    additional_to_start_collision=additional_collisions,
    source_opening_rad=opening[indices],
    joint_velocity_rad_s=velocity,
  )
  report = {
    "kind": "uncalibrated initial-pose mapping, offline IK candidate audit; no hardware execution",
    "rows": len(indices),
    "source_rows": len(times),
    "stride": args.stride,
    "source_poses_sha256": digest(args.poses),
    "mapping": "P_T_eef(t) = P_T_eef_home @ inverse(W_T_human_initial) @ W_T_human(t)",
    "mapping_status": "proposal; nominal TCP axes correspondence, not measured hand-eye/tool calibration",
    "robot_initial": robot_initial.tolist(),
    "human_initial": initial.tolist(),
    "initial_valid_state_displacement_m": float(
      np.linalg.norm(targets[0, :3, 3] - robot_initial[:3, 3])
    ),
    "ik_valid_count": int(successful.sum()),
    "ik_valid_fraction": float(successful.mean()),
    "position_error_m": {
      "p50": float(np.median(position_array)),
      "p95": float(np.percentile(position_array, 95)),
      "max": float(position_array.max()),
    },
    "rotation_error_deg": {
      "p95": float(np.rad2deg(np.percentile(angle_array, 95))),
      "max": float(np.rad2deg(angle_array.max())),
    },
    "collision_frames": int(collisions.sum()),
    "baseline_overlap_body_pairs": sorted(baseline_pairs),
    "additional_to_start_collision_frames": int(additional_collisions.sum()),
    "additional_contact_examples": [
      {
        "source_row": int(indices[i]),
        "time_s": float(times[indices[i]]),
        "pairs": [p for p in contacts[i] if tuple(sorted(p[:2])) not in baseline_pairs],
      }
      for i in np.flatnonzero(additional_collisions)[:10]
    ],
    "baseline_overlap_interpretation": "Present already at unchanged platform home; not silently excluded or declared acceptable. Source collision meshes/allowed finger contacts need review.",
    "ik_valid_without_detected_collision": int((successful & ~collisions).sum()),
    "contact_examples": [
      {
        "source_row": int(indices[i]),
        "time_s": float(times[indices[i]]),
        "pairs": contacts[i],
      }
      for i in np.flatnonzero(collisions)[:5]
    ],
    "joint_velocity_limit_violation_intervals": int(
      np.any(velocity > velocity_limits, axis=1).sum()
    ),
    "joint_velocity_max_rad_s": velocity.max(axis=0).tolist(),
    "joint_velocity_limit_rad_s": velocity_limits.tolist(),
    "left_arm": "held at source platform start pose; no left capture supplied",
    "gripper": "both Forward-01 grippers held at platform default opening; source opening retained separately, transfer not calibrated",
    "trajectory_accepted_for_execution": False,
    "reason": "Candidate coverage audit only; no initial transition or time parameterization, collision coverage incomplete, physical mapping uncalibrated",
    "model_provenance": metadata,
  }
  (args.output_dir / "forward01-result.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n"
  )
  print(
    json.dumps(
      {k: v for k, v in report.items() if k != "model_provenance"},
      ensure_ascii=False,
      indent=2,
    )
  )


if __name__ == "__main__":
  main()
