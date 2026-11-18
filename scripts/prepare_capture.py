"""Prepare an auditable, image-aligned episode without training any model.

All state/action values are retained. Observation rows only use images exposed
at or before the row, with explicit age; no future-image matching or tail padding.
Review flags identify samples to inspect, not automatically delete or interpolate.
"""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from specialist_dataset import schema
from specialist_dataset.read import Recording


def previous_images(state_times: np.ndarray, image_times: np.ndarray) -> np.ndarray:
  if (
    image_times.ndim != 1
    or not len(image_times)
    or not np.isfinite(image_times).all()
    or not np.all(np.diff(image_times) > 0)
    or not np.isfinite(state_times).all()
  ):
    raise ValueError("Image timestamps must be finite and increasing")
  return np.searchsorted(image_times, state_times, side="right") - 1


def sha(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def main() -> None:
  cli = argparse.ArgumentParser(description=__doc__)
  cli.add_argument("--state", type=Path, required=True)
  cli.add_argument("--poses-dir", type=Path, default=Path("logs/014-processing/poses"))
  cli.add_argument(
    "--protocol", type=Path, default=Path("configs/capture_protocol.example.json")
  )
  cli.add_argument(
    "--output-dir", type=Path, default=Path("logs/pretrain-20261117/episode014")
  )
  cli.add_argument("--hand", choices=schema.HANDS, default="right")
  args = cli.parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  protocol = json.loads(args.protocol.read_text())
  with Recording(args.state) as seen:
    table = seen.series("/" + schema.state_image(args.hand))
    image_times = seen.seconds(table)
    blobs = (
      table.column(schema.selector(schema.state_image(args.hand), "EncodedImage:blob"))
      .combine_chunks()
      .to_pylist()
    )
  if len(image_times) != len(blobs):
    raise ValueError("Image rows and timestamp rows differ")
  image_dir = args.output_dir / "images" / args.hand
  image_dir.mkdir(parents=True, exist_ok=True)
  images = []
  for index, (time, blob) in enumerate(zip(image_times, blobs, strict=True)):
    data = bytes(blob[0])
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
      raise ValueError(f"Cannot decode image {index}")
    path = image_dir / f"{index:06d}.jpg"
    path.write_bytes(data)
    images.append(
      {
        "index": index,
        "video_time_s": float(time),
        "path": str(path.relative_to(args.output_dir)),
        "sha256": hashlib.sha256(data).hexdigest(),
        "width": decoded.shape[1],
        "height": decoded.shape[0],
      }
    )
  source = args.poses_dir / f"{args.hand}-poses.npz"
  with np.load(source) as arrays:
    times = arrays["time_s"]
    poses = arrays["world_eef"]
    opening = arrays["opening_rad"]
    action_sources = arrays["action_source_index"]
    action_targets = arrays["action_target_index"]
    source_world = arrays["world_T_initial"]
  dt = np.diff(times)
  if not np.isfinite(poses).all() or not np.all(dt > 0):
    raise ValueError("Invalid state values or timestamps")
  speed = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1) / dt
  spin = (
    Rotation.from_matrix(poses[:-1, :3, :3]).inv()
    * Rotation.from_matrix(poses[1:, :3, :3])
  ).magnitude() / dt
  opening_speed = np.abs(np.diff(opening)) / dt
  thresholds = protocol["review_thresholds"]
  flags = (
    (speed > thresholds["translation_speed_m_s"])
    | (spin > thresholds["rotation_speed_rad_s"])
    | (opening_speed > thresholds["opening_speed_rad_s"])
  )
  image_index = previous_images(times, image_times)
  retained = action_sources[image_index[action_sources] >= 0]
  target_by_source = dict(
    zip(action_sources.tolist(), action_targets.tolist(), strict=True)
  )
  csv_path = args.output_dir / "samples.csv"
  with csv_path.open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(
      [
        "state_index",
        "video_time_s",
        "image_index",
        "image_time_s",
        "image_age_s",
        "action_target_index",
        "review_flag",
      ]
    )
    for row in retained:
      image = image_index[row]
      writer.writerow(
        [
          row,
          times[row],
          image,
          image_times[image],
          times[row] - image_times[image],
          target_by_source[int(row)],
          bool(flags[row]),
        ]
      )
  with (args.output_dir / "quality-flags.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(
      [
        "source_row",
        "time_s",
        "translation_speed_m_s",
        "rotation_speed_rad_s",
        "opening_speed_rad_s",
        "review",
      ]
    )
    writer.writerows(
      zip(range(len(dt)), times[:-1], speed, spin, opening_speed, flags, strict=True)
    )
  for suffix in ("poses.npz", "poses.csv", "actions.csv"):
    shutil.copyfile(
      args.poses_dir / f"{args.hand}-{suffix}",
      args.output_dir / f"{args.hand}-{suffix}",
    )
  shutil.copyfile(
    args.poses_dir / "pose-export.json", args.output_dir / "pose-export.json"
  )
  measured = protocol["physical_measurements"]
  size_keys = [
    "anchor_inner_width_mm",
    "anchor_inner_height_mm",
    "gripper_inner_width_mm",
    "gripper_inner_height_mm",
  ]
  complete = all(
    isinstance(measured.get(key), (int, float))
    and not isinstance(measured[key], bool)
    and measured[key] > 0
    for key in size_keys
  )
  scale_errors = {}
  if complete:
    nominal_sizes = [protocol["anchor_inner_size_m"] * 1000] * 2 + [
      protocol["gripper_inner_size_m"] * 1000
    ] * 2
    for key, nominal in zip(size_keys, nominal_sizes, strict=True):
      scale_errors[key] = abs(measured[key] / nominal - 1)
  physical_status = (
    "pending-ruler-measurement"
    if not complete
    else (
      "measured-consistent-with-assumed-size"
      if max(scale_errors.values()) <= 0.01
      else "size-mismatch-reconstruction-needs-review"
    )
  )
  result = {
    "schema": "specialist-pretraining-episode-v1",
    "kind": "prepared real episode, not a pi05 adapter or training run",
    "hand": args.hand,
    "source_state": str(args.state),
    "source_state_sha256": sha(args.state),
    "source_pose_sha256": sha(source),
    "state_rows": len(times),
    "action_rows": len(action_sources),
    "image_rows": len(images),
    "observation_action_samples": len(retained),
    "missing_past_image_action_rows": int(len(action_sources) - len(retained)),
    "image_association": "latest image at or before observation; never a future image",
    "max_image_age_s": float(
      np.max(times[retained] - image_times[image_index[retained]])
    ),
    "state_max_interval_s": float(dt.max()),
    "review_interval_count": int(flags.sum()),
    "review_opening_speed_count": int(
      (opening_speed > thresholds["opening_speed_rad_s"]).sum()
    ),
    "review_thresholds": thresholds,
    "translation_speed_max_m_s": float(speed.max()),
    "rotation_speed_max_rad_s": float(spin.max()),
    "opening_speed_max_rad_s": float(opening_speed.max()),
    "physical_scale_status": physical_status,
    "measured_relative_size_errors": scale_errors,
    "protocol": protocol,
    "world_T_initial": source_world.tolist(),
    "raw_end_tap_status": "014 original failed end position rule; explicit 50-second derived base used, raw preserved",
    "cleaning": "No state smoothing, pose interpolation, review-row deletion or zero action padding applied",
    "dataset_split": "By independent episode/session, not random neighboring frames",
    "images": images,
    "sample_index": "samples.csv",
    "quality_index": "quality-flags.csv",
  }
  (args.output_dir / "episode.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n"
  )
  print(
    json.dumps(
      {
        k: v
        for k, v in result.items()
        if k not in ("images", "protocol", "world_T_initial")
      },
      ensure_ascii=False,
      indent=2,
    )
  )


if __name__ == "__main__":
  main()
