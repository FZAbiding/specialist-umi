"""Causal image association and bounded IK use known independent test targets."""

import numpy as np
import pytest
from scripts.prepare_capture import previous_images


def test_observation_does_not_use_future_image():
  images = np.array([0.1, 0.2, 0.3])
  states = np.array([0.09, 0.1, 0.19, 0.2, 0.35])
  np.testing.assert_array_equal(previous_images(states, images), [-1, 0, 0, 1, 2])


@pytest.mark.parametrize("times", [[], [0.2, 0.1], [0.1, 0.1], [0.1, np.nan]])
def test_invalid_image_clock_is_rejected(times):
  with pytest.raises(ValueError):
    previous_images(np.array([0.1]), np.array(times))


def toy_robot():
  mj = pytest.importorskip("mujoco")
  model = mj.MjModel.from_xml_string("""<mujoco><worldbody><body name="tip">
  <joint name="x" type="slide" axis="1 0 0" range="-.5 .5"/>
  <joint name="y" type="slide" axis="0 1 0" range="-.5 .5"/>
  <joint name="z" type="slide" axis="0 0 1" range="-.5 .5"/>
  <joint name="rx" axis="1 0 0" range="-80 80"/>
  <joint name="ry" axis="0 1 0" range="-80 80"/>
  <joint name="rz" axis="0 0 1" range="-80 80"/>
  <geom type="sphere" size=".01" mass="1"/>
  </body></worldbody></mujoco>""")
  return mj, model, mj.MjData(model)


def test_bounded_ik_reaches_independent_fk_target():
  _, model, data = toy_robot()
  from scripts.check_forward01 import fk, solve

  data.qpos[:] = [0.03, -0.04, 0.06, 0.1, -0.08, 0.12]
  target = fk(model, data, 1)
  data.qpos[:] = 0
  q, position, angle = solve(model, data, 1, list(range(6)), target, np.zeros(6))
  assert position < 0.002
  assert angle < np.deg2rad(1)
  assert np.all(q >= model.jnt_range[:, 0]) and np.all(q <= model.jnt_range[:, 1])


def test_unreachable_target_is_not_reported_as_valid():
  _, model, data = toy_robot()
  from scripts.check_forward01 import solve

  target = np.eye(4)
  target[:3, 3] = [2, 0, 0]
  q, position, _ = solve(model, data, 1, list(range(6)), target, np.zeros(6))
  assert position > 1
  assert np.all(q >= model.jnt_range[:, 0]) and np.all(q <= model.jnt_range[:, 1])
