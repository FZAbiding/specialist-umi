"""Rerun overlays of measured per-lens AprilTag corners at keyframe times."""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import rerun as rr
from specialist_dataset import schema

FloatArray = npt.NDArray[np.float64]
COLORS = [(230, 90, 70), (70, 185, 120), (80, 150, 245)]


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
