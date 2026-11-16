"""Known-pose tests for the educational AprilTag demo, not real accuracy claims."""

import numpy as np
from scripts.demo_apriltag import (
  camera,
  project_pose,
  render_tag,
  solve,
  synthetic_pose,
)

from specialist.vi_ba.tag_detection import detect_all_tags


def test_exact_corners_recover_known_pose():
  truth = synthetic_pose()
  estimated = solve(project_pose(truth, 0.1, camera()), 0.1, camera())
  np.testing.assert_allclose(estimated, truth, atol=1e-9)


def test_wrong_size_preserves_pixels_but_scales_translation():
  truth = synthetic_pose()
  pixels = project_pose(truth, 0.1, camera())
  wrong = solve(pixels, 0.18, camera())
  np.testing.assert_allclose(wrong[:3, 3], truth[:3, 3] * 1.8, atol=1e-9)
  np.testing.assert_allclose(project_pose(wrong, 0.18, camera()), pixels, atol=1e-9)


def test_real_detector_decodes_synthetic_flexible_layout():
  truth = synthetic_pose()
  image = render_tag(truth, 0.1, camera())
  hits = detect_all_tags([image])[0]
  assert [h.tag_id for h in hits] == [0]
  estimated = solve(hits[0].corners, 0.1, camera())
  assert np.linalg.norm(estimated[:3, 3] - truth[:3, 3]) < 0.002
