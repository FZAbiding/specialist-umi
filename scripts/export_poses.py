"""Export a state layer's metric gripper pose as CSV and homogeneous matrices.

The stored pose is the base-tag frame in the anchor/gravity world. The optional
TCP offset uses the same axes as that tag; its orientation is a convention,
not an independently calibrated robot tool frame.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from rerun.chunk import RrdReader
from specialist_dataset import schema
from specialist_dataset.poses import quats_to_matrices
from specialist_dataset.read import Recording

from specialist.pose_actions import action_arrays, sample_pose, validate_poses


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("state", type=Path)
  parser.add_argument("--hand", choices=schema.HANDS, default="right")
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument(
    "--reference-time", type=float, help="Initial-frame time in video seconds"
  )
  parser.add_argument(
    "--reference-estimate",
    type=Path,
    help="Full estimate for a reference outside state",
  )
  parser.add_argument(
    "--initial-frame",
    type=Path,
    help="JSON world_T_initial matrix; share it between hands",
  )
  parser.add_argument("--delta-steps", type=int, default=1)
  parser.add_argument("--chunk-length", type=int, default=1)
  parser.add_argument(
    "--delta-frame",
    choices=("body", "world"),
    default="body",
    help="Frame of the single-step delta_eef alias/CSV; named chunk fields stay body",
  )
  parser.add_argument(
    "--tcp-offset",
    type=float,
    nargs=3,
    default=(0.1095, 0.0, -0.018),
    metavar=("X_M", "Y_M", "Z_M"),
  )
  args = parser.parse_args()
  with Recording(args.state) as recording:
    slot = str(
      np.asarray(recording.row(schema.state_hand(args.hand))["slot"]).reshape(-1)[0]
    )
    t, position = recording.values(
      schema.state_pose(args.hand), schema.POSE_TRANSLATION, 3
    )
    tq, quaternion = recording.values(
      schema.state_pose(args.hand), schema.POSE_QUATERNION, 4
    )
    to, opening = recording.scalars(schema.state_opening(args.hand), 1)
    tv, linear = recording.scalars(schema.state_lin_vel(args.hand), 3)
    tw, angular = recording.scalars(schema.state_ang_vel(args.hand), 3)
  if not all(np.array_equal(t, other) for other in (tq, to, tv, tw)):
    raise SystemExit("Pose, opening and velocity do not share timestamps")
  values = np.column_stack((position, quaternion, opening, linear, angular))
  if not np.isfinite(values).all() or not np.all(np.diff(t) > 0):
    raise SystemExit("Nonfinite state or unordered timestamps")
  rotations = quats_to_matrices(quaternion)
  world_T_base_tag = np.broadcast_to(np.eye(4), (len(t), 4, 4)).copy()
  world_T_base_tag[:, :3, :3] = rotations
  world_T_base_tag[:, :3, 3] = position
  offset = np.asarray(args.tcp_offset, dtype=np.float64)
  tcp = position + np.einsum("nij,j->ni", rotations, offset)
  world_T_tcp = world_T_base_tag.copy()
  world_T_tcp[:, :3, 3] = tcp
  reference_time = float(t[0]) if args.reference_time is None else args.reference_time
  reference_source = (
    "first retained state" if args.reference_time is None else "state interpolation"
  )
  if args.initial_frame is not None:
    if args.reference_time is not None or args.reference_estimate is not None:
      raise SystemExit(
        "--initial-frame cannot be combined with reference time/estimate"
      )
    stored_frame = json.loads(args.initial_frame.read_text())
    world_T_initial = np.asarray(
      stored_frame.get("world_T_initial")
      if isinstance(stored_frame, dict)
      else stored_frame,
      dtype=np.float64,
    )
    validate_poses(world_T_initial)
    reference_source = str(args.initial_frame)
  elif args.reference_estimate is not None:
    if args.reference_time is None:
      raise SystemExit("--reference-estimate requires --reference-time")
    if (
      RrdReader(str(args.state)).recordings()[0].recording_id
      != RrdReader(str(args.reference_estimate)).recordings()[0].recording_id
    ):
      raise SystemExit("Reference estimate and state belong to different sessions")
    with Recording(args.reference_estimate) as recording:
      et, ep = recording.values(schema.camera(slot), schema.POSE_TRANSLATION, 3)
      eqt, eq = recording.values(schema.camera(slot), schema.POSE_QUATERNION, 4)
      if not np.array_equal(et, eqt):
        raise SystemExit("Reference-estimate pose timestamps disagree")
      bt = recording.row(schema.base_tag(slot))
    body_T_tag = np.eye(4)
    body_T_tag[:3, :3] = quats_to_matrices(bt[schema.POSE_QUATERNION])[0]
    body_T_tag[:3, 3] = np.asarray(bt[schema.POSE_TRANSLATION]).reshape(3)
    tag_T_tcp = np.eye(4)
    tag_T_tcp[:3, 3] = offset
    world_T_initial = sample_pose(et, ep, eq, reference_time) @ body_T_tag @ tag_T_tcp
    reference_source = str(args.reference_estimate)
  else:
    world_T_initial = sample_pose(t, tcp, quaternion, reference_time)
  derived = action_arrays(
    t, world_T_tcp, world_T_initial, args.delta_steps, args.chunk_length
  )
  derived["delta_eef"] = derived[f"delta_eef_{args.delta_frame}"]
  derived["opening_target_rad"] = opening[derived["action_target_index"], 0]
  derived["chunk_opening_target_rad"] = opening[derived["chunk_target_index"], 0]
  args.output_dir.mkdir(parents=True, exist_ok=True)
  csv = args.output_dir / f"{args.hand}-poses.csv"
  npz = args.output_dir / f"{args.hand}-poses.npz"
  header = (
    "time_s,tag_x_m,tag_y_m,tag_z_m,tag_qx,tag_qy,tag_qz,tag_qw,"
    "tcp_x_m,tcp_y_m,tcp_z_m,opening_rad,"
    "tag_vx_m_s,tag_vy_m_s,tag_vz_m_s,omega_x_rad_s,omega_y_rad_s,omega_z_rad_s"
  )
  np.savetxt(
    csv,
    np.column_stack((t, position, quaternion, tcp, opening, linear, angular)),
    delimiter=",",
    header=header,
    comments="",
    fmt="%.12g",
  )
  np.savez_compressed(
    npz,
    time_s=t,
    world_T_base_tag=world_T_base_tag,
    world_T_tcp=world_T_tcp,
    tag_position_m=position,
    tag_quaternion_xyzw=quaternion,
    opening_rad=opening[:, 0],
    linear_velocity_world_m_s=linear,
    angular_velocity_world_rad_s=angular,
    **derived,
  )
  action_csv = args.output_dir / f"{args.hand}-actions.csv"
  source = derived["action_source_index"]
  np.savetxt(
    action_csv,
    np.column_stack(
      (
        derived["action_time_s"],
        derived["action_target_time_s"],
        derived["action_dt_s"],
        tcp[source],
        quaternion[source],
        derived["delta_eef"],
        derived["opening_target_rad"],
      )
    ),
    delimiter=",",
    header=(
      "time_s,target_time_s,dt_s,world_tcp_x_m,world_tcp_y_m,world_tcp_z_m,"
      "world_qx,world_qy,world_qz,world_qw,delta_x_m,delta_y_m,delta_z_m,"
      "delta_rx_rad,delta_ry_rad,delta_rz_rad,opening_target_rad"
    ),
    comments="",
    fmt="%.12g",
  )
  summary = {
    "source": str(args.state.resolve()),
    "hand": args.hand,
    "rows": len(t),
    "first_s": float(t[0]),
    "last_s": float(t[-1]),
    "rate_hz": float(1 / np.median(np.diff(t))),
    "quaternion_order": "xyzw",
    "position_unit": "m",
    "opening_unit": "rad",
    "world_frame": "reference anchor origin and horizontal heading, gravity Z-up",
    "stored_pose": "world_T_base_tag",
    "tcp_offset_in_base_tag_m": offset.tolist(),
    "tcp_orientation": "axes parallel to base tag; not a robot TCP calibration",
    "quaternion_max_norm_error": float(
      np.abs(np.linalg.norm(quaternion, axis=1) - 1).max()
    ),
    "csv": str(csv.resolve()),
    "npz": str(npz.resolve()),
    "actions_csv": str(action_csv.resolve()),
    "actions": {
      "delta_frame": args.delta_frame,
      "delta_steps": args.delta_steps,
      "action_rows": len(source),
      "chunk_length": args.chunk_length,
      "chunk_frame": "body",
      "chunk_rows": len(derived["chunk_source_index"]),
      "body_formula": "inv(world_eef[t]) @ world_eef[t+k]",
      "world_vector_formula": "[p_next-p_now, rotvec(R_next @ R_now.T)]",
      "rotation": "SO(3) rotation vector, rad; not Euler subtraction or SE(3) log twist",
      "translation": "m per selected interval, not m/s",
      "chunk_formula": "inv(world_eef[t]) @ world_eef[t+h*k], same t for every h",
      "gripper": "absolute target opening in rad, not differenced",
      "tail": "Only complete future targets; no zero/identity padding",
      "policy_integration": "Geometric export; not a verified pi0.5 dataset adapter",
    },
    "initial_frame": {
      "source": reference_source,
      "time_s": None if args.initial_frame is not None else reference_time,
      "world_T_initial": world_T_initial.tolist(),
      "default": "first retained state, not necessarily recording start",
      "bimanual": "Pass the same --initial-frame to both hands to preserve their baseline",
    },
  }
  (args.output_dir / "pose-export.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
  )
  print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
