"""Reproducible AprilTag geometry experiments, with optional real X3 MEI input.

Synthetic scenes carry known pose truth. Real-camera outputs carry no external
pose truth. This is an educational demo using the official detector/PnP helper.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation
from specialist_camera import MeiParams, mei_project
from specialist_dataset import frames, schema
from specialist_dataset.fiducials import tag_corners_local
from specialist_dataset.read import read_lens

from specialist.record.session import SessionRecording
from specialist.vi_ba.tag_detection import detect_all_tags, estimate_lens_T_tag

FloatArray = npt.NDArray[np.float64]
ROOT = Path(__file__).resolve().parents[1]


def camera() -> MeiParams:
  return MeiParams(800, 800, 480, 360, 0, 0, 0, 0, 0, 0)


def project_pose(pose: FloatArray, size: float, params: MeiParams) -> FloatArray:
  points = tag_corners_local(size) @ pose[:3, :3].T + pose[:3, 3]
  return mei_project(points, params)


def synthetic_pose(distance: float = 0.65) -> FloatArray:
  pose = np.eye(4)
  pose[:3, :3] = Rotation.from_euler("xyz", [180, 25, 10], degrees=True).as_matrix()
  pose[:3, 3] = [0.03, -0.02, distance]
  return pose


def render_tag(
  pose: FloatArray, size: float, params: MeiParams
) -> npt.NDArray[np.uint8]:
  """Pinhole rendering: 5-cell detected square inside the 9-cell 41h12 layout."""
  if params.xi != 0 or any((params.k1, params.k2, params.k3, params.p1, params.p2)):
    raise ValueError("Planar homography rendering here requires an undistorted pinhole")
  bitmap = cv2.imread(str(ROOT / "reports/assets/tag41_12_00000.png"), 0)
  if bitmap is None or bitmap.shape != (9, 9):
    raise ValueError("Expected the official 9 by 9 tagStandard41h12 ID 0 bitmap")
  cell = 160
  bitmap = cv2.resize(bitmap, (9 * cell, 9 * cell), interpolation=cv2.INTER_NEAREST)
  source = np.array([[2, 7], [7, 7], [7, 2], [2, 2]], np.float32) * cell
  supersample = 3
  high = replace(
    params,
    fx=params.fx * supersample,
    fy=params.fy * supersample,
    cx=params.cx * supersample,
    cy=params.cy * supersample,
  )
  target = project_pose(pose, size, high).astype(np.float32)
  mapping = cv2.getPerspectiveTransform(source, target)
  image = cv2.warpPerspective(
    bitmap,
    mapping,
    (960 * supersample, 720 * supersample),
    flags=cv2.INTER_NEAREST,
    borderValue=255,
  )
  return np.asarray(
    cv2.resize(image, (960, 720), interpolation=cv2.INTER_AREA), dtype=np.uint8
  )


def pose_metrics(pose: FloatArray, truth: FloatArray) -> dict[str, float]:
  return {
    "translation_error_m": float(np.linalg.norm(pose[:3, 3] - truth[:3, 3])),
    "rotation_error_deg": float(
      np.degrees(
        np.linalg.norm(Rotation.from_matrix(truth[:3, :3].T @ pose[:3, :3]).as_rotvec())
      )
    ),
  }


def solve(corners: FloatArray, size: float, params: MeiParams) -> FloatArray:
  pose = estimate_lens_T_tag(corners, size, params)
  if pose is None:
    raise ValueError("Tag has no usable planar pose solution")
  return np.asarray(pose.matrix(), dtype=np.float64)


def annotate(
  image: npt.NDArray[np.uint8], detections: list[Any], size: float, params: MeiParams
) -> npt.NDArray[np.uint8]:
  out = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
  axes = np.array([[0, 0, 0], [size / 2, 0, 0], [0, size / 2, 0], [0, 0, size / 2]])
  for detection in detections:
    cv2.polylines(
      out, [np.rint(detection.corners).astype(np.int32)], True, (220, 150, 0), 3
    )
    center = tuple(np.rint(detection.center).astype(int))
    cv2.putText(
      out,
      f"ID {detection.tag_id}",
      center,
      cv2.FONT_HERSHEY_SIMPLEX,
      0.65,
      (0, 80, 0),
      2,
    )
    pose = solve(detection.corners, size, params)
    pixels = mei_project(axes @ pose[:3, :3].T + pose[:3, 3], params)
    if np.isfinite(pixels).all():
      points = np.rint(pixels).astype(int)
      for endpoint, color in zip(
        points[1:], [(220, 30, 30), (0, 160, 0), (0, 70, 230)], strict=True
      ):
        cv2.arrowedLine(
          out, tuple(points[0]), tuple(endpoint), color, 3, tipLength=0.15
        )
  return np.asarray(out, dtype=np.uint8)


def synthetic_experiments(output: Path) -> dict[str, Any]:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  params, size, truth = camera(), 0.1, synthetic_pose()
  image = render_tag(truth, size, params)
  detections = detect_all_tags([image])[0]
  detected = next(d for d in detections if d.tag_id == 0)
  estimated = solve(detected.corners, size, params)
  wrong_size = solve(detected.corners, 0.18, params)
  reprojection = project_pose(estimated, size, params)
  rms = float(np.sqrt(np.mean(np.sum((reprojection - detected.corners) ** 2, axis=1))))
  result: dict[str, Any] = {
    "kind": "synthetic known pose, not real-camera accuracy",
    "seed": 7,
    "tag_size_m": size,
    "family": "tagStandard41h12",
    "tag_id": 0,
    "camera": vars(params),
    "truth_C_T_tag": truth.tolist(),
    "estimated_C_T_tag": estimated.tolist(),
    "detected_corners_px": detected.corners.tolist(),
    "reprojection_rmse_px": rms,
    **pose_metrics(estimated, truth),
    "wrong_size_180mm": {
      "C_T_tag": wrong_size.tolist(),
      **pose_metrics(wrong_size, truth),
    },
  }
  fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
  axes[0].imshow(annotate(image, [detected], size, params))
  axes[0].set(
    xlim=(270, 700), ylim=(570, 130), title="Synthetic ID 0 | detected square + tag XYZ"
  )
  axes[0].axis("off")
  axes[1].barh(
    ["true Z", "solve L=100 mm", "solve L=180 mm"],
    [truth[2, 3], estimated[2, 3], wrong_size[2, 3]],
    color=["#78909c", "#28698f", "#b45519"],
  )
  axes[1].set(xlabel="Camera depth Z [m]", title="Wrong physical size changes scale")
  fig.savefig(output / "synthetic-pose.png", dpi=160)
  plt.close(fig)
  rng = np.random.default_rng(7)
  trials = []
  for distance in (0.4, 0.8, 1.2):
    target = synthetic_pose(distance)
    exact = project_pose(target, size, params)
    for sigma in (0.1, 0.3, 1.0, 2.0):
      errors = [
        pose_metrics(solve(exact + rng.normal(0, sigma, (4, 2)), size, params), target)
        for _ in range(60)
      ]
      translation = [e["translation_error_m"] * 1000 for e in errors]
      rotation = [e["rotation_error_deg"] for e in errors]
      trials.append(
        {
          "distance_m": distance,
          "corner_sigma_px": sigma,
          "trials": 60,
          "median_translation_mm": float(np.median(translation)),
          "p95_translation_mm": float(np.percentile(translation, 95)),
          "median_rotation_deg": float(np.median(rotation)),
        }
      )
  fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), layout="constrained")
  for distance in (0.4, 0.8, 1.2):
    rows = [r for r in trials if r["distance_m"] == distance]
    for axis, metric in zip(
      axes, ["median_translation_mm", "median_rotation_deg"], strict=True
    ):
      axis.plot(
        [r["corner_sigma_px"] for r in rows],
        [r[metric] for r in rows],
        "o-",
        label=f"Z={distance} m",
      )
  for axis, unit in zip(
    axes, ["Translation error [mm]", "Rotation error [deg]"], strict=True
  ):
    axis.set(xlabel="Independent corner noise sigma [px]", ylabel=unit)
    axis.legend()
  fig.savefig(output / "corner-noise.png", dpi=160)
  plt.close(fig)
  result["corner_noise_experiment"] = trials
  return result


def real_example(args: argparse.Namespace) -> dict[str, Any]:
  stage = args.output_dir / "packets"
  stage.mkdir(exist_ok=True)
  session = SessionRecording(args.real_base, stage)
  try:
    video = session.lens_video(args.slot, args.lens)
    index = int(np.argmin(np.abs(video.times - args.time)))
    if abs(float(video.times[index]) - args.time) > 0.5 / 29.97:
      raise ValueError("Requested time is not covered by the real recording")
    _, frame = next(frames.decode_samples(video, np.array([index], dtype=np.int64)))
    gray = frame.to_ndarray(format="gray")
    params = read_lens(args.real_base, args.slot, args.lens, gray.shape[1]).params
    detected = detect_all_tags([gray])[0]
    # This real front-lens example solves the configured 100 mm wall anchors only.
    anchors = [d for d in detected if d.tag_id in (0, 1)]
    drawn = annotate(gray, anchors, 0.1, params)
    cv2.imwrite(
      str(args.output_dir / "real-front-tags.jpg"),
      cv2.cvtColor(drawn, cv2.COLOR_RGB2BGR),
    )
    return {
      "kind": "real X3 frame; no ground truth",
      "base": str(args.real_base),
      "frame_time_s": float(video.times[index]),
      "lens": args.lens,
      "detected_ids": [d.tag_id for d in detected],
      "anchor_size_m": 0.1,
      "poses_C_T_tag": {
        str(d.tag_id): solve(d.corners, 0.1, params).tolist() for d in anchors
      },
      "projection_model": "factory MEI + Brown distortion from base; official PnP helper",
    }
  finally:
    session.close()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", type=Path, default=Path("logs/apriltag-demo"))
  parser.add_argument("--real-base", type=Path)
  parser.add_argument("--slot", default="cam00")
  parser.add_argument("--lens", choices=schema.LENSES, default="front")
  parser.add_argument("--time", type=float, default=7.8078)
  args = parser.parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  result = synthetic_experiments(args.output_dir)
  if args.real_base is not None:
    result["real_example"] = real_example(args)
  path = args.output_dir / "apriltag-demo.json"
  path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
  print(f"Wrote {path}")
  print(
    f"Synthetic pose error: {result['translation_error_m'] * 1000:.3f} mm, "
    f"{result['rotation_error_deg']:.3f} deg; reprojection {result['reprojection_rmse_px']:.3f} px"
  )


if __name__ == "__main__":
  main()
