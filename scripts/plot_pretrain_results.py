"""Plot measured software checks and render the unaccepted Forward-01 IK candidate."""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from specialist_sim import video


def main() -> None:
  cli = argparse.ArgumentParser(description=__doc__)
  cli.add_argument("--root", type=Path, default=Path("logs/pretrain-20261117"))
  cli.add_argument("--no-video", action="store_true")
  args = cli.parse_args()
  root = args.root
  sim = np.load(root / "specialist-sim/tracking.npz")
  time = sim["time_s"]
  fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
  axes[0].plot(time, sim["position_error_m"] * 1000, lw=0.6, color="#236184")
  axes[0].set_ylabel("Position error / mm")
  axes[1].plot(time, np.rad2deg(sim["rotation_error_rad"]), lw=0.6, color="#a55318")
  axes[1].set_ylabel("Rotation error / deg")
  axes[2].plot(
    time, np.rad2deg(sim["target"][:, 7]), lw=0.7, label="reconstruction target"
  )
  axes[2].plot(
    time, np.rad2deg(sim["actual"][:, 7]), lw=0.6, alpha=0.75, label="simulated finger"
  )
  axes[2].set_ylabel("Opening / deg")
  axes[2].set_xlabel("Original video time / s")
  axes[2].legend()
  fig.suptitle(
    "014 right hand | official Floating dynamics | tracking, not pose accuracy"
  )
  fig.tight_layout()
  fig.savefig(root / "sim-tracking.png", dpi=160)
  plt.close(fig)
  quality = np.genfromtxt(
    root / "episode014/quality-flags.csv",
    delimiter=",",
    names=True,
    dtype=None,
    encoding="utf-8",
  )
  fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
  for axis, key, label, limit in zip(
    axes,
    ["translation_speed_m_s", "rotation_speed_rad_s", "opening_speed_rad_s"],
    ["TCP speed / m/s", "Angular speed / rad/s", "Opening speed / rad/s"],
    [2, 10, 5],
    strict=True,
  ):
    axis.plot(quality["time_s"], quality[key], lw=0.6)
    axis.axhline(limit, color="#a55318", lw=0.8, ls="--")
    axis.set_ylabel(label)
  axes[-1].set_xlabel("Original video time / s")
  fig.suptitle("Review thresholds flag motion; no rows removed or smoothed")
  fig.tight_layout()
  fig.savefig(root / "capture-quality.png", dpi=160)
  plt.close(fig)
  info = json.loads((root / "forward01/forward01-result.json").read_text())
  ik = np.load(root / "forward01/ik-candidates.npz")
  time = ik["time_s"]
  fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
  axes[0].plot(time, ik["position_error_m"] * 1000, lw=0.7)
  axes[0].axhline(2, color="#a55318", ls="--", lw=0.7)
  axes[0].set_ylabel("IK position / mm")
  axes[1].plot(time, np.rad2deg(ik["rotation_error_rad"]), lw=0.7)
  axes[1].axhline(1, color="#a55318", ls="--", lw=0.7)
  axes[1].set_ylabel("IK rotation / deg")
  ratio = np.max(
    ik["joint_velocity_rad_s"] / np.array(info["joint_velocity_limit_rad_s"]), axis=1
  )
  axes[2].plot(time[1:], ratio, lw=0.7)
  axes[2].axhline(1, color="#a55318", ls="--", lw=0.7)
  axes[2].set_ylabel("Max joint speed / limit")
  axes[3].step(
    time,
    ik["additional_to_start_collision"].astype(int),
    where="mid",
    lw=0.8,
    color="#a55318",
  )
  axes[3].set_ylabel("Additional contact")
  axes[3].set_ylim(-0.1, 1.2)
  axes[-1].set_xlabel("Original video time / s")
  fig.suptitle(
    "Forward-01 initial-pose proposal | candidate is NOT an execution trajectory"
  )
  fig.tight_layout()
  fig.savefig(root / "forward01-validation.png", dpi=160)
  plt.close(fig)
  if args.no_video:
    return
  model = mujoco.MjModel.from_xml_path(info["model_provenance"]["model"])
  data = mujoco.MjData(model)
  renderer = mujoco.Renderer(model, 480, 640)
  camera = mujoco.MjvCamera()
  camera.lookat[:] = [0.65, 0, 0.8]
  camera.distance = 2.8
  camera.azimuth = 135
  camera.elevation = -22
  option = mujoco.MjvOption()
  option.geomgroup[3] = 0

  def image_at(row: int) -> np.ndarray:
    data.qpos[:] = ik["qpos"][row]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera, scene_option=option)
    for xyz, color in [
      (ik["target"][row, :3, 3], [0.85, 0.20, 0.10, 1]),
      (ik["achieved"][row, :3, 3], [0.10, 0.60, 0.35, 1]),
    ]:
      geom = renderer.scene.geoms[renderer.scene.ngeom]
      mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.012, 0, 0]),
        xyz,
        np.eye(3).ravel(),
        np.array(color, dtype=np.float32),
      )
      renderer.scene.ngeom += 1
    image = renderer.render()
    valid = bool(ik["ik_valid"][row])
    status = "IK OK" if valid else "IK ERROR"
    cv2.putText(
      image,
      f"CANDIDATE ONLY | t={time[row]:.2f}s | {status}",
      (12, 24),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.52,
      (220, 110, 30),
      1,
      cv2.LINE_AA,
    )
    return np.asarray(image, dtype=np.uint8)

  for filename, row in [
    ("forward01-preview.png", len(time) // 2),
    (
      "forward01-failure.png",
      int(np.flatnonzero(~ik["ik_valid"])[0])
      if np.any(~ik["ik_valid"])
      else len(time) - 1,
    ),
  ]:
    cv2.imwrite(str(root / filename), cv2.cvtColor(image_at(row), cv2.COLOR_RGB2BGR))
  stride = 3
  fps = 1 / float(np.median(np.diff(time))) / stride
  with video.writer(
    root / "forward01/forward01-candidates.mp4", 640, 480, fps
  ) as write:
    for row in range(0, len(time), stride):
      write(image_at(row))
  renderer.close()
  print("Wrote numerical figures and candidate-only Forward-01 preview")


if __name__ == "__main__":
  main()
