"""Read a state layer without a catalog; write a compact, auditable summary."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from specialist_dataset import schema
from specialist_dataset.read import Recording


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("state", type=Path)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  result: dict[str, Any] = {
    "path": str(args.state.resolve()),
    "bytes": args.state.stat().st_size,
    "hands": {},
  }
  with Recording(args.state) as recording:
    result["entities"] = sorted(recording.entity_paths())
    for hand in schema.HANDS:
      if schema.state_pose(hand) not in result["entities"]:
        continue
      time, xyz = recording.values(schema.state_pose(hand), schema.POSE_TRANSLATION, 3)
      _, quaternion = recording.values(
        schema.state_pose(hand), schema.POSE_QUATERNION, 4
      )
      _, opening = recording.scalars(schema.state_opening(hand), 1)
      images = recording.series("/" + schema.state_image(hand))
      result["hands"][hand] = {
        "rows": len(time),
        "first_s": float(time[0]),
        "last_s": float(time[-1]),
        "median_state_hz": float(1 / np.median(np.diff(time))),
        "images": images.num_rows,
        "position_min_m": xyz.min(0).tolist(),
        "position_max_m": xyz.max(0).tolist(),
        "opening_min_rad": float(opening.min()),
        "opening_max_rad": float(opening.max()),
        "max_quaternion_norm_error": float(
          np.max(np.abs(np.linalg.norm(quaternion, axis=1) - 1))
        ),
        "quaternion_order": "xyzw",
        "pose_frame": "gripper base-tag in world",
      }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
  print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
