"""Geometric contracts of the fork's EEF export, independent of a policy."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from specialist.pose_actions import action_arrays, inverse, sample_pose, validate_poses


def sequence():
  t = np.array([0.0, 0.03, 0.08, 0.12])
  poses = np.broadcast_to(np.eye(4), (4, 4, 4)).copy()
  poses[:, :3, :3] = Rotation.from_euler(
    "xyz", [[0, 0, 90], [10, 0, 90], [10, 25, 90], [15, 25, 100]], degrees=True
  ).as_matrix()
  poses[:, :3, 3] = [[0, 0, 0], [0.1, 0, 0], [0.1, 0.2, 0], [0.1, 0.2, 0.3]]
  return t, poses


def test_body_delta_axes_and_rotation_reconstruct_targets():
  t, poses = sequence()
  out = action_arrays(t, poses, poses[0])
  np.testing.assert_allclose(out["delta_eef_body"][0, :3], [0, -0.1, 0], atol=1e-12)
  np.testing.assert_allclose(out["delta_eef_world"][0, :3], [0.1, 0, 0])
  np.testing.assert_allclose(
    poses[:-1] @ out["delta_eef_body_T"], poses[1:], atol=1e-12
  )
  recovered = Rotation.from_rotvec(out["delta_eef_body"][:, 3:]).as_matrix()
  np.testing.assert_allclose(recovered, out["delta_eef_body_T"][:, :3, :3], atol=1e-12)
  np.testing.assert_allclose(out["action_dt_s"], [0.03, 0.05, 0.04])


def test_fixed_world_change_does_not_change_body_delta_or_initial_poses():
  t, poses = sequence()
  change = np.eye(4)
  change[:3, :3] = Rotation.from_euler("xyz", [35, -15, 60], degrees=True).as_matrix()
  change[:3, 3] = [2, -3, 1]
  first = action_arrays(t, poses, poses[0], chunk_length=2)
  second = action_arrays(t, change @ poses, change @ poses[0], chunk_length=2)
  for name in ["delta_eef_body_T", "chunk_delta_eef_body_T", "initial_eef"]:
    np.testing.assert_allclose(first[name], second[name], atol=1e-12)


def test_chunk_targets_share_observation_anchor_and_have_no_tail_padding():
  poses = np.broadcast_to(np.eye(4), (5, 4, 4)).copy()
  poses[:, 0, 3] = [0, 1, 3, 6, 10]
  out = action_arrays(np.arange(5, dtype=float), poses, poses[0], chunk_length=3)
  assert out["chunk_delta_eef_body"].shape == (2, 3, 6)
  np.testing.assert_allclose(out["chunk_delta_eef_body"][0, :, 0], [1, 3, 6])
  np.testing.assert_allclose(out["chunk_delta_eef_body"][1, :, 0], [2, 5, 9])
  np.testing.assert_array_equal(out["chunk_target_index"][-1], [2, 3, 4])


def test_shared_initial_frame_preserves_bimanual_baseline():
  t, left = sequence()
  right = left.copy()
  right[:, 1, 3] -= 0.5
  shared = left[0]
  left_initial = action_arrays(t, left, shared)["initial_eef"]
  right_initial = action_arrays(t, right, shared)["initial_eef"]
  assert np.linalg.norm(
    left_initial[0, :3, 3] - right_initial[0, :3, 3]
  ) == pytest.approx(0.5)
  np.testing.assert_allclose(
    inverse(left_initial[0]) @ right_initial[0], inverse(left[0]) @ right[0]
  )


def test_reference_interpolation_and_outside_span_rejection():
  t = np.array([1.0, 2.0])
  xyz = np.array([[0, 0, 0], [2, 0, 0]], dtype=float)
  q = Rotation.from_euler("z", [[0], [90]], degrees=True).as_quat()
  p = sample_pose(t, xyz, q, 1.5)
  np.testing.assert_allclose(p[:3, 3], [1, 0, 0])
  np.testing.assert_allclose(
    p[:3, :3], Rotation.from_euler("z", 45, degrees=True).as_matrix()
  )
  with pytest.raises(ValueError, match="outside"):
    sample_pose(t, xyz, q, 0.0)


@pytest.mark.parametrize("steps,chunk", [(0, 1), (1, 0), (2, 2), (1, 4)])
def test_invalid_horizons_are_rejected(steps, chunk):
  t, poses = sequence()
  with pytest.raises(ValueError, match="horizon"):
    action_arrays(t, poses, poses[0], steps, chunk)


def test_non_rotation_and_reflection_are_rejected():
  bad = np.eye(4)
  bad[0, 0] = -1
  with pytest.raises(ValueError, match="determinant"):
    validate_poses(bad)
