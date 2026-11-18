"""Build a Rerun review from the actual hands of a Specialist state layer."""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from rerun.chunk import RrdReader
from specialist_dataset import schema
from specialist_dataset.read import Recording, read_anchors, read_keyframe_poses

from specialist.review_tags import COLORS, log_tags, tag_overlays


def blueprint(hands: dict[str, str]) -> rrb.Blueprint:
  def plot(entity: str, title: str, labels: list[str]) -> rrb.TimeSeriesView:
    return rrb.TimeSeriesView(
      name=title,
      contents=[f"/{entity}"],
      overrides={
        f"/{entity}": rr.SeriesLines(names=labels, colors=COLORS[: len(labels)])
      },
    )

  images = []
  signals = []
  for hand, slot in hands.items():
    images.append(
      rrb.Horizontal(
        *[
          rrb.Spatial2DView(
            name=f"{hand} · {slot} · {lens}",
            origin=f"/{schema.lens(slot, lens)}",
            contents=[
              f"/{schema.video(slot, lens)}",
              f"/{schema.lens(slot, lens)}/inspection/**",
            ],
          )
          for lens in schema.LENSES
        ]
      )
    )
    signals.append(
      rrb.Horizontal(
        plot(schema.imu_signal(slot, "accel"), f"{hand} · m/s²", ["x", "y", "z"]),
        plot(schema.imu_signal(slot, "gyro"), f"{hand} · rad/s", ["x", "y", "z"]),
        plot(schema.state_opening(hand), f"{hand} · opening (rad)", ["opening"]),
      )
    )
  return rrb.Blueprint(
    rrb.Tabs(
      rrb.Horizontal(
        rrb.Spatial3DView(
          name="World", origin="/", contents=["/world/**", "/camera/**"]
        ),
        rrb.Vertical(*images),
        name="Pose and measured tags",
      ),
      rrb.Vertical(*signals, name="IMU and opening"),
      active_tab=0,
    ),
    rrb.TimePanel(timeline=schema.TIMELINE),
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dataset", type=Path, required=True)
  parser.add_argument("--session", required=True)
  parser.add_argument(
    "--state", type=Path, help="Override the default state_s8_r100 layer"
  )
  args = parser.parse_args()
  estimate = schema.layer_rrd(args.dataset, schema.ESTIMATE, args.session)
  state = args.state or schema.layer_rrd(
    args.dataset, schema.state_layer(8, 100), args.session
  )
  for path in (estimate, state):
    ids = [r.recording_id for r in RrdReader(str(path)).recordings()]
    if ids != [args.session]:
      raise ValueError(f"{path} contains recording IDs {ids}, expected {args.session}")
  times = read_keyframe_poses(estimate).kf_times_by_slot
  anchors = sorted(read_anchors(estimate).poses)
  with Recording(state) as recording:
    entities = recording.entity_paths()
    hands = {
      hand: str(np.asarray(recording.row(schema.state_hand(hand))["slot"]).item())
      for hand in schema.HANDS
      if schema.state_pose(hand) in entities
    }
  if len(hands) != len(times) or set(hands.values()) != set(times) or not hands:
    raise ValueError(f"State hands {hands} disagree with estimate slots {list(times)}")
  output = args.dataset / "review" / args.session
  output.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(dir=output) as directory:
    layer = Path(directory) / "measured-tags.rrd"
    layout = Path(directory) / "session.rbl"
    with rr.RecordingStream(schema.APPLICATION_ID, recording_id=args.session) as stream:
      stream.save(layer)
      for hand, slot in hands.items():
        print(f"{hand} = {slot}")
        log_tags(stream, slot, tag_overlays(estimate, slot, times[slot]), anchors)
    blueprint(hands).save(schema.APPLICATION_ID, layout)
    for path in (layer, layout):
      path.replace(output / path.name)
  print(f"Review files: {output}")


if __name__ == "__main__":
  main()
