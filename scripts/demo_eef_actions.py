"""Show initial-frame retargeting and body-frame action chunks on synthetic poses.

The 'robot' trajectory is a geometric target, not IK, collision validation or
hardware execution. Both tool frames deliberately share the same axis convention.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.spatial.transform import Rotation

from specialist.pose_actions import action_arrays, inverse

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", type=Path, default=Path("logs/eef-actions-demo"))
  args = parser.parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  t = np.linspace(0, 2, 101)
  human = np.broadcast_to(np.eye(4), (len(t), 4, 4)).copy()
  human[:, :3, :3] = Rotation.from_euler(
    "xyz",
    np.column_stack((10 * np.sin(t), np.zeros(len(t)), 90 + 20 * t)),
    degrees=True,
  ).as_matrix()
  human[:, :3, 3] = np.column_stack(
    (0.7 + 0.08 * t, 0.3 + 0.1 * np.sin(t), 0.4 + 0.04 * t)
  )
  robot_start = np.eye(4)
  robot_start[:3, :3] = Rotation.from_euler(
    "xyz", [0, 0, -30], degrees=True
  ).as_matrix()
  robot_start[:3, 3] = [0.4, -0.2, 0.5]
  arrays = action_arrays(t, human, human[0], chunk_length=10)
  robot_targets = robot_start @ arrays["initial_eef"]
  robot_arrays = action_arrays(t, robot_targets, robot_start, chunk_length=10)
  error = float(
    np.max(np.abs(arrays["delta_eef_body_T"] - robot_arrays["delta_eef_body_T"]))
  )
  assert error < 1e-10
  assert np.allclose(robot_targets, robot_start @ inverse(human[0]) @ human)
  np.savez_compressed(
    args.output_dir / "synthetic-actions.npz", **arrays, robot_targets=robot_targets
  )
  summary = {
    "kind": "synthetic initial-frame retargeting; not hardware execution",
    "rows": len(t),
    "period_s": float(t[1] - t[0]),
    "chunk_length": 10,
    "retarget_formula": "robot_start @ inv(human_start) @ human_pose[t]",
    "body_delta_invariance_max_error": error,
    "human_world_T_initial": human[0].tolist(),
    "robot_base_T_initial": robot_start.tolist(),
    "axis_assumption": "Human TCP and robot TCP use the same axes; scale is 1 in meters",
    "limits": [
      "No IK or collision check",
      "No actual robot initial pose measured",
      "No policy trained",
    ],
  }
  (args.output_dir / "eef-actions-demo.json").write_text(
    json.dumps(summary, indent=2) + "\n"
  )
  fig = plt.figure(figsize=(11, 4.5), layout="constrained")
  for index, (poses, title) in enumerate(
    [
      (human, "Human recording world"),
      (robot_targets, "Retargeted geometric targets (synthetic)"),
    ],
    start=1,
  ):
    axis = fig.add_subplot(1, 2, index, projection="3d")
    xyz = poses[:, :3, 3]
    axis.plot(*xyz.T, color="#25678d")
    axis.scatter(*xyz[0], color="#b35418", s=45, label="initial TCP")
    axis.set(title=title, xlabel="X [m]", ylabel="Y [m]", zlabel="Z [m]")
    axis.legend()
  fig.savefig(args.output_dir / "initial-retargeting.png", dpi=160)
  plt.close(fig)
  print(f"Wrote {args.output_dir}; body-delta invariance error {error:.3g}")


if __name__ == "__main__":
  main()
