"""Local data-export conventions for world, initial-frame and delta EEF poses.

These inspection/adapter helpers are additions to the fork, not an upstream
training recipe or a hardware controller. A pose A_T_B maps B coordinates to A.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation, Slerp

FloatArray = npt.NDArray[np.float64]


def validate_poses(poses: FloatArray) -> None:
  if poses.ndim < 2 or poses.shape[-2:] != (4, 4) or not np.isfinite(poses).all():
    raise ValueError("Poses must be finite (..., 4, 4) matrices")
  if not np.allclose(poses[..., 3, :], [0, 0, 0, 1], atol=1e-10):
    raise ValueError("Invalid homogeneous bottom row")
  r = poses[..., :3, :3]
  if not np.allclose(np.swapaxes(r, -1, -2) @ r, np.eye(3), atol=1e-6):
    raise ValueError("Rotation is not orthogonal")
  if not np.allclose(np.linalg.det(r), 1.0, atol=1e-6):
    raise ValueError("Rotation must have determinant +1")


def inverse(poses: FloatArray) -> FloatArray:
  validate_poses(poses)
  out = np.broadcast_to(np.eye(4), poses.shape).copy()
  out[..., :3, :3] = np.swapaxes(poses[..., :3, :3], -1, -2)
  out[..., :3, 3] = -np.einsum("...ij,...j->...i", out[..., :3, :3], poses[..., :3, 3])
  return out


def sample_pose(
  times: FloatArray, positions: FloatArray, quaternions_xyzw: FloatArray, when: float
) -> FloatArray:
  """Interpolate one pose without silently clamping outside the measured span."""
  if not np.isfinite(when) or not times[0] <= when <= times[-1]:
    raise ValueError(f"Reference {when} s is outside [{times[0]}, {times[-1]}]")
  if len(times) < 2 or not np.all(np.diff(times) > 0):
    raise ValueError("Interpolation needs at least two increasing times")
  high = int(np.clip(np.searchsorted(times, when), 1, len(times) - 1))
  low = high - 1
  fraction = (when - times[low]) / (times[high] - times[low])
  out = np.eye(4)
  out[:3, 3] = positions[low] * (1 - fraction) + positions[high] * fraction
  out[:3, :3] = Slerp(
    times[[low, high]], Rotation.from_quat(quaternions_xyzw[[low, high]])
  )([when]).as_matrix()[0]
  return out


def as_delta_vector(transforms: FloatArray) -> FloatArray:
  """[translation in m, SO(3) rotation vector in rad]; not the SE(3) log twist."""
  shape = transforms.shape[:-2]
  angles = Rotation.from_matrix(transforms[..., :3, :3].reshape(-1, 3, 3)).as_rotvec()
  return np.concatenate((transforms[..., :3, 3], angles.reshape(*shape, 3)), axis=-1)


def action_arrays(
  times: FloatArray,
  world_eef: FloatArray,
  world_T_initial: FloatArray,
  delta_steps: int = 1,
  chunk_length: int = 1,
) -> dict[str, Any]:
  """Generate full-frame poses and valid future targets without tail padding.

  Body delta: inv(T[t]) @ T[t+k]. A chunk uses the SAME observation T[t]
  for every future target; it does not chain consecutive per-step deltas.
  """
  validate_poses(world_eef)
  validate_poses(world_T_initial)
  if world_eef.shape != (len(times), 4, 4) or world_T_initial.shape != (4, 4):
    raise ValueError("Expected poses (N,4,4) and one initial frame (4,4)")
  if not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
    raise ValueError("Times must be finite and strictly increasing")
  if delta_steps < 1 or chunk_length < 1 or delta_steps * chunk_length >= len(times):
    raise ValueError("Positive horizon must fit inside the episode")
  source = np.arange(len(times) - delta_steps)
  target = source + delta_steps
  body = inverse(world_eef[source]) @ world_eef[target]
  world_rotation = world_eef[target, :3, :3] @ np.swapaxes(
    world_eef[source, :3, :3], -1, -2
  )
  world_delta = np.column_stack(
    (
      world_eef[target, :3, 3] - world_eef[source, :3, 3],
      Rotation.from_matrix(world_rotation).as_rotvec(),
    )
  )
  chunk_source = np.arange(len(times) - delta_steps * chunk_length)
  chunk_target = chunk_source[:, None] + delta_steps * np.arange(1, chunk_length + 1)
  chunk_body = inverse(world_eef[chunk_source])[:, None] @ world_eef[chunk_target]
  return {
    "world_eef": world_eef,
    "world_T_initial": world_T_initial,
    "initial_eef": inverse(world_T_initial) @ world_eef,
    "action_source_index": source,
    "action_target_index": target,
    "action_time_s": times[source],
    "action_target_time_s": times[target],
    "action_dt_s": times[target] - times[source],
    "delta_eef_body_T": body,
    "delta_eef_body": as_delta_vector(body),
    "delta_eef_world": world_delta,
    "chunk_source_index": chunk_source,
    "chunk_target_index": chunk_target,
    "chunk_time_s": times[chunk_source],
    "chunk_target_time_s": times[chunk_target],
    "chunk_delta_eef_body_T": chunk_body,
    "chunk_delta_eef_body": as_delta_vector(chunk_body),
  }
