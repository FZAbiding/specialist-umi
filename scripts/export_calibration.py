"""Extract factory offset_v3 and the exact MEI parameters used for decoded lenses.

This reads existing calibration; it does not recalibrate a camera or measure
physical accuracy. Device-specific outputs belong in ignored local logs.
"""

import argparse
import hashlib
import json
from dataclasses import asdict, fields
from pathlib import Path

import av
import numpy as np
from specialist_camera import MeiParams, mei_project, mei_unproject, rescale
from specialist_dataset import schema
from specialist_dataset.read import Recording, read_lens

from specialist.loader.parse_insv import load_insv
from specialist.record.offset_v3 import parse_offset_v3


def file_hash(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("insv", type=Path)
  parser.add_argument("--base", type=Path)
  parser.add_argument("--slot", default="cam00")
  parser.add_argument("--decoded-width", type=int)
  parser.add_argument(
    "--output", type=Path, default=Path("logs/calibration/factory-mei.json")
  )
  args = parser.parse_args()
  parsed = load_insv(args.insv)
  offset = parse_offset_v3(parsed.offset_v3)
  with av.open(str(args.insv)) as video:
    stream = video.streams.video[0]
    native = [stream.width, stream.height]
  width = args.decoded_width or native[0]
  if width <= 0:
    raise SystemExit("decoded width must be positive")
  result = {
    "kind": "read factory calibration; not a new camera calibration",
    "input": str(args.insv),
    "sha256": file_hash(args.insv),
    "native_video_size": native,
    "decoded_width": width,
    "metadata_path": "telemetry_parser.Parser(...).telemetry()[0][Default][Metadata][offset_v3]",
    "offset_v3_raw": parsed.offset_v3.tolist(),
    "model": "MEI unified omnidirectional + Brown-Conrady k1,k2,k3,p1,p2",
    "raw_angles_unit": "raw array: degree; decoded factory fields: rad",
    "distortion_units": "dimensionless",
    "scaling": "s=2*decoded_width/frame_width; rear cx subtracts frame_width/2 before scaling",
    "rolling_shutter_readout_s": parsed.rolling_shutter_readout,
    "lenses": {},
    "limits": [
      "Factory data availability is not external accuracy validation",
      "Use original fisheye pixel geometry; resized/cropped/reprojected images need matching parameters",
    ],
  }
  for name in ("front", "rear"):
    block = getattr(offset, name)
    raw = MeiParams(**{f.name: getattr(block, f.name) for f in fields(MeiParams)})
    params = rescale(raw, block.frame_width, width, name)
    pose = getattr(offset, f"body_T_{name}_lens").matrix()
    angles = np.linspace(-np.deg2rad(65), np.deg2rad(65), 13)
    rays = np.array(
      [
        [np.sin(a) * np.cos(b), np.sin(b), np.cos(a) * np.cos(b)]
        for a in angles
        for b in angles
      ]
    )
    pixels = mei_project(rays, params)
    recovered = mei_unproject(pixels, params)
    valid = np.isfinite(recovered).all(axis=1)
    errors = np.linalg.norm(np.cross(rays[valid], recovered[valid]), axis=1)
    lens = {
      "raw_factory": asdict(block),
      "decoded_mei": asdict(params),
      "body_T_lens": pose.tolist(),
      "roundtrip_rays": int(valid.sum()),
      "roundtrip_max_sin_angle_error": float(errors.max()),
    }
    if args.base:
      loaded = read_lens(args.base, args.slot, name, width)
      param_error = max(
        abs(getattr(params, f.name) - getattr(loaded.params, f.name))
        for f in fields(MeiParams)
      )
      pose_error = float(np.max(np.abs(pose - loaded.body_T_lens)))
      with Recording(args.base) as recording:
        row = recording.row(schema.lens(args.slot, name))
      raw_error = max(
        abs(getattr(raw, f.name) - float(np.asarray(row[f.name]).reshape(-1)[0]))
        for f in fields(MeiParams)
      )
      lens["base_readback"] = {
        "parameter_max_abs_error": param_error,
        "raw_max_abs_error": raw_error,
        "body_T_lens_max_abs_error": pose_error,
      }
      if max(param_error, raw_error, pose_error) > 1e-6:
        raise SystemExit(f"{name}: factory extraction and base recording disagree")
    result["lenses"][name] = lens
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
  print(f"Factory calibration exported: {args.output}")
  for name, lens in result["lenses"].items():
    print(name, json.dumps(lens["decoded_mei"]))


if __name__ == "__main__":
  main()
