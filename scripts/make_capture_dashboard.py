"""Audit stored poses/observations and add an IMU dashboard to an existing solve.

This local inspection tool leaves the official base, estimate and viz untouched.
IMU plots read the original native-rate Scalars; a separate layer contains only
derived inspection signals and a full-session TCP path.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import rerun as rr
import rerun.blueprint as rrb
from specialist_dataset import frames, schema
from specialist_dataset.fiducials import tag_corners_local
from specialist_dataset.read import Recording, read_anchors

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
COLORS = [(230, 90, 70), (70, 185, 120), (80, 150, 245)]
FULL_PATH_COLOR = (80, 127, 160, 55)
FULL_PATH_RADIUS = 0.0004


def time_column(times: FloatArray) -> rr.TimeColumn:
  """Preserve source nanoseconds across a seconds-to-integer roundtrip."""
  return rr.TimeColumn(
    schema.TIMELINE,
    duration=np.rint(times * schema.NS_PER_SECOND).astype("timedelta64[ns]"),
  )


def tag_overlays(estimate: Path, slot: str, times: FloatArray) -> dict[str, Any]:
  """Read official keyframe-only detections without re-running the detector."""
  server = rr.server.Server(host="127.0.0.1", datasets={"tags": [str(estimate)]})
  try:
    entry = rr.catalog.CatalogClient(server.url()).get_dataset("tags")
    out: dict[str, Any] = {}
    for lens in schema.LENSES:
      entity = schema.detection(slot, lens, schema.TAGS_DETECTION)
      table = (
        entry.filter_contents(f"/{entity}")
        .reader(index=schema.KEYFRAME_TIMELINE)
        .to_arrow_table()
        .sort_by(schema.KEYFRAME_TIMELINE)
      )
      keys = np.asarray(table.column(schema.KEYFRAME_TIMELINE).to_pylist(), dtype=int)
      if not np.array_equal(keys, np.arange(len(times))):
        raise ValueError(f"{lens} detections do not cover the keyframe grid")
      ids = table.column(schema.selector(entity, "tag_id")).to_pylist()
      strips = table.column(schema.selector(entity, "LineStrips2D:strips")).to_pylist()
      if any(len(i) != len(s) for i, s in zip(ids, strips, strict=True)):
        raise ValueError(f"{lens} tag IDs and corner arrays differ")
      out[lens] = {"times": times, "ids": ids, "strips": strips}
    return out
  finally:
    server.shutdown()


def log_tags(
  stream: rr.RecordingStream, slot: str, overlays: dict[str, Any], anchors: list[int]
) -> None:
  """Hold the latest measured keyframe's boxes, explicitly clearing missing rows."""
  for lens, observed in overlays.items():
    counts = [len(row) for row in observed["ids"]]
    strips = [quad for row in observed["strips"] for quad in row]
    ids = [int(tag) for row in observed["ids"] for tag in row]
    if not ids:
      continue
    colors = [(255, 210, 0) if tag in anchors else (90, 220, 220) for tag in ids]
    centers = np.asarray([np.mean(np.asarray(s)[:4], axis=0) for s in strips])
    entity = f"{schema.lens(slot, lens)}/inspection/tags"
    stream.send_columns(
      entity,
      indexes=[time_column(observed["times"])],
      columns=rr.LineStrips2D.columns(
        strips=strips, colors=colors, radii=np.full(len(ids), 4.0)
      ).partition(counts),
      strict=True,
    )
    stream.send_columns(
      f"{entity}/labels",
      indexes=[time_column(observed["times"])],
      columns=rr.Points2D.columns(
        positions=centers,
        labels=[f"ID {tag}" for tag in ids],
        colors=colors,
        radii=np.full(len(ids), 5.0),
        show_labels=[True] * len(ids),
      ).partition(counts),
      strict=True,
    )


def continuity(times: FloatArray, values: FloatArray) -> dict[str, Any]:
  dt = np.diff(times)
  period = float(np.median(dt))
  gaps = np.flatnonzero(dt > 2 * period)
  return {
    "samples": len(times),
    "first_s": float(times[0]),
    "last_s": float(times[-1]),
    "median_period_ms": period * 1000,
    "max_period_ms": float(dt.max()) * 1000,
    "non_increasing_steps": int(np.count_nonzero(dt <= 0)),
    "nonfinite_rows": int(np.count_nonzero(~np.isfinite(values).all(axis=1))),
    "gap_rule": "dt > 2 * median(dt); internal intervals only",
    "internal_gaps": [
      {"start_s": float(times[i]), "end_s": float(times[i + 1])} for i in gaps
    ],
  }


def missing_runs(
  times: FloatArray, missing: BoolArray, end: float
) -> list[dict[str, Any]]:
  edges = np.diff(np.concatenate(([False], missing, [False])).astype(int))
  out = []
  for first, stop in zip(
    np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True
  ):
    until = float(times[stop]) if stop < len(times) else end
    out.append(
      {
        "start_s": float(times[first]),
        "end_s": until,
        "duration_s": until - float(times[first]),
        "keyframes": int(stop - first),
      }
    )
  return out


def observation_summary(
  times: FloatArray, missing: BoolArray, end: float
) -> dict[str, Any]:
  runs = missing_runs(times, missing, end)
  return {
    "keyframes": len(times),
    "missing_keyframes": int(missing.sum()),
    "missing_percent": float(missing.mean() * 100),
    "episodes": len(runs),
    "longest_s": max((r["duration_s"] for r in runs), default=0.0),
    "missing_hold_s": sum(r["duration_s"] for r in runs),
    "runs": runs,
  }


def analyze(
  dataset: Path, session: str, slot: str, state: Path | None
) -> tuple[dict[str, Any], dict[str, Any]]:
  base, estimate, viz = (
    schema.layer_rrd(dataset, layer, session)
    for layer in (schema.BASE, schema.ESTIMATE, schema.VIZ)
  )
  with Recording(base) as r:
    ti, gyro = r.scalars(schema.imu_signal(slot, "gyro"), 3)
    ta, accel = r.scalars(schema.imu_signal(slot, "accel"), 3)
    video_times = frames.lens_times(r, schema.video(slot, "front"))
  end = float(video_times[-1])
  with Recording(estimate) as r:
    tp, pose = r.values(schema.camera(slot), schema.POSE_TRANSLATION, 3)
    tq, quat = r.values(schema.camera(slot), schema.POSE_QUATERNION, 4)
    tk, kpose = r.values(schema.state(slot, "pose"), schema.POSE_TRANSLATION, 3)
    tn, points = r.scalars(schema.fit(slot, "n_keypoints"))
    te, rmse = r.scalars(schema.fit(slot, "rmse"))
  if not np.array_equal(tp, tq) or not np.array_equal(ti, ta):
    raise ValueError("Pose/quaternion or gyro/accel times disagree")
  anchors = read_anchors(estimate).poses
  ids = sorted(anchors)
  centers = np.stack([anchors[tag][:3, 3] for tag in ids])
  with Recording(viz) as r:
    tr, strips = r.values(schema.trajectory(slot), "LineStrips3D:strips", 6)
    rays = r.series("/" + schema.factor(slot, "anchor"))
    ts = r.seconds(rays)
    rows = rays.column(
      schema.selector(schema.factor(slot, "anchor"), "LineStrips3D:strips")
    ).to_pylist()
  if not np.array_equal(tk, ts):
    raise ValueError("Anchor-ray rows must cover every keyframe")
  present = np.zeros((len(tk), len(ids)), dtype=bool)
  for row, lines in enumerate(rows):
    if not lines:
      continue
    endpoints = np.asarray(lines, dtype=np.float64)[:, -1, :]
    distances = np.linalg.norm(endpoints[:, None] - centers[None], axis=2)
    if float(distances.min(axis=1).max()) > 0.08:
      raise ValueError("Ray endpoint is not a corner of a 100 mm anchor")
    present[row, np.unique(distances.argmin(axis=1))] = True
  tcp = strips[:, 3:]
  length = np.linalg.norm(strips[:, 3:] - strips[:, :3], axis=1)
  cumulative = np.concatenate(([0.0], np.cumsum(length)))
  right = np.searchsorted(tr, tk, side="right")
  left = np.searchsorted(tr, tk - 1.0, side="left")
  trailing_length = cumulative[right] - cumulative[left]
  report: dict[str, Any] = {
    "session": session,
    "slot": slot,
    "definitions": {
      "missing_anchor": "No stored anchor reprojection rays at a keyframe; not a pose-loss flag",
      "episode": "Maximal consecutive missing-keyframe run; intervals [start, end)",
      "ray_assignment": "Nearest solved anchor center to each ray endpoint; <= 0.08 m",
      "duration": "Hold each keyframe observation until the next; last to video end",
      "pose_limit": "Continuous IMU-propagated poses do not establish tracking accuracy",
    },
    "video_end_s": end,
    "pose": continuity(tp, np.column_stack((pose, quat))),
    "keyframe_pose": continuity(tk, kpose),
    "trajectory": continuity(tr, strips),
    "gyro": continuity(ti, gyro),
    "accel": continuity(ta, accel),
    "pose_boundaries": {
      "before_first_pose_s": float(tp[0] - video_times[0]),
      "after_last_pose_s": end - float(tp[-1]),
      "explanation": "Official highrate propagation only spans first to last keyframe",
    },
    "anchors": {
      str(tag): observation_summary(tk, ~present[:, i], end)
      for i, tag in enumerate(ids)
    },
    "no_anchor": observation_summary(tk, np.asarray(~present.any(axis=1)), end),
    "fewer_than_all_anchors": observation_summary(
      tk, np.asarray(~present.all(axis=1)), end
    ),
    "visual_features": {
      "min": float(points.min()),
      "median": float(np.median(points)),
      "max": float(points.max()),
      "zero_keyframes": int(np.count_nonzero(points.sum(axis=1) == 0)),
      "meaning": "Logged tracked keypoint count; not an independent tracking-success flag",
    },
    "short_trail": {
      "rule": "Trailing 1 s TCP arc length < 0.01 m at keyframes >= 1 s",
      "keyframes": int(np.count_nonzero((tk >= 1) & (trailing_length < 0.01))),
      "meaning": "Display-scale heuristic; not a missing trajectory",
    },
    "screenshots": {},
  }
  for cursor in (41.0, 43.0):
    k = int(np.clip(np.searchsorted(tk, cursor, side="right") - 1, 0, len(tk) - 1))
    i = int(np.clip(np.searchsorted(tp, cursor), 0, len(tp) - 1))
    report["screenshots"][str(cursor)] = {
      "keyframe_s": float(tk[k]),
      "observed_anchor_ids": [tag for j, tag in enumerate(ids) if present[k, j]],
      "pose_sample_s": float(tp[i]),
      "tracked_keypoints": points[k].tolist(),
      "front_rmse_px": rmse[k].tolist(),
      "tcp_arc_length_previous_1s_m": float(trailing_length[k]),
    }
  if state is not None:
    with Recording(state) as r:
      st, sp = r.values(schema.state_pose("right"), schema.POSE_TRANSLATION, 3)
      to, opened = r.scalars(schema.state_opening("right"), 1)
      opening_type = str(
        r.series("/" + schema.state_opening("right"))
        .schema.field(schema.selector(schema.state_opening("right"), schema.SCALARS))
        .type
      )
    report["state_right"] = continuity(st, sp)
    inside = (tk >= st[0]) & (tk <= st[-1])
    report["no_anchor_in_state_window"] = observation_summary(
      tk[inside], ~present[inside].any(axis=1), float(st[-1])
    )
    report["opening"] = {
      **continuity(to, opened),
      "source": str(state),
      "entity": schema.state_opening("right"),
      "arrow_type": opening_type,
      "unit": "rad",
      "normalized": False,
      "definition": "(left_finger_angle - right_finger_angle) / 2",
      "sign": "increasing opens; zero is the URDF joint reference",
      "min_rad": float(opened.min()),
      "max_rad": float(opened.max()),
    }
  data = dict(
    ti=ti,
    gyro=gyro,
    accel=accel,
    tr=tr,
    tcp=tcp,
    tk=tk,
    present=present,
    ids=ids,
    tn=tn,
    points=points,
    te=te,
    rmse=rmse,
  )
  if state is not None:
    data.update(opening_time=to, opening=opened)
  return report, data


def dashboard(slot: str, ids: list[int], end: float, opening: bool) -> rrb.Blueprint:
  trail = rrb.VisibleTimeRanges(
    [
      rrb.VisibleTimeRange(
        schema.TIMELINE,
        start=rrb.TimeRangeBoundary.cursor_relative(seconds=-1.0),
        end=rrb.TimeRangeBoundary.cursor_relative(),
      )
    ]
  )
  scene = rrb.Spatial3DView(
    name="Solved right gripper | blue: full TCP path | red: last 1 s",
    origin="/",
    contents=[
      "+ /**",
      *[
        f"- /{schema.lens(slot, lens)}/{tail}"
        for lens in schema.LENSES
        for tail in ("video", "video/**", "overlay/**", "detection/**", "inspection/**")
      ],
    ],
    overrides={
      f"/{schema.trajectory(slot)}": trail,
      f"/diagnostics/{slot}/path/full": rr.LineStrips3D.from_fields(
        colors=[FULL_PATH_COLOR], radii=FULL_PATH_RADIUS
      ),
      **{
        f"/{schema.landmark(slot, lens)}": rrb.EntityBehavior(visible=False)
        for lens in schema.LENSES
      },
      **{
        f"/{schema.anchor(tag)}": [
          rr.TransformAxes3D(0.05),
          rr.LineStrips3D(
            [np.vstack((tag_corners_local(0.1), tag_corners_local(0.1)[:1]))],
            colors=[(255, 210, 0)],
            radii=0.003,
          ),
        ]
        for tag in ids
      },
    },
  )
  images = rrb.Vertical(
    *[
      rrb.Spatial2DView(
        name=f"{slot}/{lens} | detected tags @10 Hz",
        origin=f"/{schema.lens(slot, lens)}",
        contents=[
          f"+ /{schema.video(slot, lens)}",
          f"+ /{schema.lens(slot, lens)}/inspection/**",
        ],
      )
      for lens in schema.LENSES
    ]
  )
  time_range = rrb.VisibleTimeRange(
    schema.TIMELINE,
    start=rrb.TimeRangeBoundary.absolute(seconds=0),
    end=rrb.TimeRangeBoundary.absolute(seconds=end),
  )

  def plot(entity: str, name: str, names: list[str]) -> rrb.TimeSeriesView:
    return rrb.TimeSeriesView(
      name=name,
      origin="/",
      contents=[f"+ /{entity}"],
      overrides={
        f"/{entity}": rr.SeriesLines(names=names, colors=COLORS[: len(names)])
      },
      time_ranges=time_range,
      axis_x=rrb.TimeAxis(link=rrb.components.LinkAxis.LinkToGlobal),
      plot_legend=rrb.PlotLegend(visible=True),
    )

  prefix = f"diagnostics/{slot}"
  imu = rrb.Horizontal(
    plot(
      schema.imu_signal(slot, "accel"),
      "RAW IMU accel [m/s^2] | gravity included",
      ["x", "y", "z"],
    ),
    plot(
      schema.imu_signal(slot, "gyro"),
      "RAW IMU gyro [rad/s] | sensor axes",
      ["x", "y", "z"],
    ),
    *(
      [
        plot(
          schema.state_opening("right"),
          "Right opening [rad] | larger = open",
          ["opening"],
        )
      ]
      if opening
      else []
    ),
    name="IMU + gripper opening",
  )
  quality = rrb.Horizontal(
    plot(
      f"{prefix}/anchor", "Anchor observations [0/1]", [f"tag {tag}" for tag in ids]
    ),
    plot(schema.fit(slot, "n_keypoints"), "Tracked front keypoints [count]", ["front"]),
    plot(schema.fit(slot, "rmse"), "Front reprojection RMSE [px]", ["front"]),
    name="Tracking inspection (not ground truth)",
  )
  return rrb.Blueprint(
    rrb.Vertical(
      rrb.Horizontal(scene, images, column_shares=[2, 1]),
      rrb.Tabs(imu, quality, active_tab=0),
      row_shares=[2, 1],
    ),
    rrb.BlueprintPanel(state=rrb.PanelState.Collapsed),
    rrb.SelectionPanel(state=rrb.PanelState.Collapsed),
    rrb.TimePanel(state=rrb.PanelState.Collapsed, timeline=schema.TIMELINE),
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dataset", type=Path, required=True)
  parser.add_argument("--session", required=True)
  parser.add_argument("--slot", default="cam00")
  parser.add_argument("--state", type=Path)
  parser.add_argument("--output-dir", type=Path, required=True)
  args = parser.parse_args()
  report, d = analyze(args.dataset, args.session, args.slot, args.state)
  overlays = tag_overlays(
    schema.layer_rrd(args.dataset, schema.ESTIMATE, args.session), args.slot, d["tk"]
  )
  report["tag_overlays"] = {
    "source": "Official estimate: keyframe-indexed tag IDs and measured corners",
    "timeline": schema.TIMELINE,
    "policy": "Latest keyframe held until next; empty detections explicitly clear boxes",
    "rate_hz": float(1 / np.median(np.diff(d["tk"]))),
    "lenses": {
      lens: {
        "keyframes": len(value["times"]),
        "tag_counts": {
          str(tag): sum(tag in row for row in value["ids"])
          for tag in sorted({int(tag) for row in value["ids"] for tag in row})
        },
      }
      for lens, value in overlays.items()
    },
  }
  args.output_dir.mkdir(parents=True, exist_ok=True)
  output = args.output_dir / "diagnostics.rrd"
  partial = output.with_suffix(".rrd.partial")
  with rr.RecordingStream(schema.APPLICATION_ID, recording_id=args.session) as stream:
    stream.save(partial)
    log_tags(stream, args.slot, overlays, d["ids"])
    prefix = f"diagnostics/{args.slot}"
    path = f"{prefix}/path/full"
    stream.log(path, rr.CoordinateFrame(frame=schema.WORLD_FRAME), static=True)
    # Only the display path is decimated to ~100 Hz; no source series is changed.
    picked = np.unique(np.r_[np.arange(0, len(d["tcp"]), 10), len(d["tcp"]) - 1])
    stream.log(
      path,
      rr.LineStrips3D(
        [d["tcp"][picked]], colors=[FULL_PATH_COLOR], radii=FULL_PATH_RADIUS
      ),
      static=True,
    )
    stream.send_columns(
      f"{prefix}/anchor",
      indexes=[time_column(d["tk"])],
      columns=rr.Scalars.columns(scalars=d["present"].astype(np.float64)),
    )
    if "opening" in d:
      stream.send_columns(
        schema.state_opening("right"),
        indexes=[time_column(d["opening_time"])],
        columns=rr.Scalars.columns(scalars=d["opening"]),
      )
  partial.replace(output)
  blueprint = args.output_dir / "imu-dashboard.rbl"
  dashboard(args.slot, d["ids"], report["video_end_s"], "opening" in d).save(
    schema.APPLICATION_ID, blueprint
  )
  report["outputs"] = {"layer": str(output), "blueprint": str(blueprint)}
  summary = args.output_dir / "diagnostics.json"
  summary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
  with (args.output_dir / "anchor-gaps.csv").open("w", newline="") as f:
    fields = ["missing", "start_s", "end_s", "duration_s", "keyframes"]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for label, item in [("all", report["no_anchor"]), *report["anchors"].items()]:
      writer.writerows({"missing": label, **run} for run in item["runs"])
  print(f"Wrote {summary}, {output}, {blueprint}")
  print(
    "No-anchor observations:",
    {k: v for k, v in report["no_anchor"].items() if k != "runs"},
  )


if __name__ == "__main__":
  main()
