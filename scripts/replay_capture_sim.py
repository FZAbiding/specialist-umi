"""Replay one real hand through the official Floating gripper environment.

The official Take currently requires both hands. This local adapter reads only
an existing hand, never fabricates a missing source hand, and parks the other
simulated gripper away from the recording. No robot or trained policy is used.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import mujoco
import numpy as np
from rerun.chunk import RrdReader
from scipy.spatial.transform import Rotation
from specialist_dataset.read import Recording, read_anchors, read_lens
from specialist_sim import episode, rotation, video
from specialist_sim.camera.lens import Lens, Mei
from specialist_sim.env import ROWWISE, Cfg, Env, Start, Task, make
from specialist_sim.gripper import DRAWN, HANDS
from specialist_sim.scripts.replay import over
from specialist_sim.take import _track, _world_from_room, states


def main() -> None:
  cli = argparse.ArgumentParser(description=__doc__)
  cli.add_argument("--base", type=Path, required=True)
  cli.add_argument("--estimate", type=Path, required=True)
  cli.add_argument("--state", type=Path, required=True)
  cli.add_argument("--hand", choices=HANDS, default="right")
  cli.add_argument("--output-dir", type=Path, default=Path("logs/capture-sim"))
  cli.add_argument("--image-size", type=int, default=240)
  cli.add_argument("--no-video", action="store_true")
  args = cli.parse_args()
  sessions = {
    RrdReader(str(path)).recordings()[0].recording_id
    for path in (args.base, args.estimate, args.state)
  }
  if len(sessions) != 1:
    raise SystemExit("Base, estimate and state belong to different sessions")
  args.output_dir.mkdir(parents=True, exist_ok=True)
  with Recording(args.state) as seen, Recording(args.estimate) as solved:
    track = _track(seen, solved, args.hand)
  anchors = read_anchors(args.estimate).poses
  if len(anchors) < 2:
    raise SystemExit("This replay room convention needs two distinct anchors")
  room_T_world = rotation.inverse(
    _world_from_room(anchors[min(anchors)], anchors[max(anchors)])
  )
  if np.max(np.abs(np.diff(track.times) - ROWWISE.row_dt)) > 1e-7:
    raise SystemExit("This replay requires an unbroken 100 Hz state lattice")
  desired = states(track, room_T_world, track.times)
  measured_lens = read_lens(args.base, track.slot, "rear", args.image_size)
  lens = Lens(
    params=Mei(**asdict(measured_lens.params)).scaled(1 / args.image_size),
    pos=tuple(track.tag_from_lens[:3]),
    quat=tuple(track.tag_from_lens[3:]),
  )
  cfg = Cfg(
    clock=ROWWISE,
    driven=(args.hand,),
    image_hands=(),
    lenses={args.hand: lens},
    camera_latency=0,
    autoreset=False,
  )
  env = Env(Task(), cfg)
  start = np.zeros((2, 8))
  start[:, 3] = 1
  inactive = 1 - HANDS.index(args.hand)
  start[inactive, :3] = [10, 10, 10]
  start[HANDS.index(args.hand)] = desired[0]
  obs, _ = env.reset([0], start=Start(start, np.zeros((0, 7))))
  rows = [env.rows(0)]
  unstable = False
  for row, target in enumerate(desired[1:], 1):
    obs, info = env.step(target[None, None, None])
    rows.append(env.rows(0))
    unstable |= bool(info["unstable"][0])
    if row % 1000 == 0:
      print(f"physics {row}/{len(desired) - 1}", flush=True)
  joined = episode.joined(rows)
  actual = joined.state[:, HANDS.index(args.hand)]
  pos_error = np.linalg.norm(actual[:, :3] - desired[:, :3], axis=1)
  actual_r = Rotation.from_quat(actual[:, [4, 5, 6, 3]])
  desired_r = Rotation.from_quat(desired[:, [4, 5, 6, 3]])
  angle_error = (actual_r.inv() * desired_r).magnitude()
  opening_error = np.abs(actual[:, 7] - desired[:, 7])
  meta = env.describe() | {
    "collector": "real-single-hand-replay",
    "source_state": str(args.state),
    "source_hands": [args.hand],
    "inactive_sim_hand": HANDS[inactive],
    "inactive_sim_hand_is_not_source_data": True,
    "room_T_world_wxyz": room_T_world.tolist(),
    "source_video_start_s": float(track.times[0]),
  }
  episode_path = episode.save(
    args.output_dir / "episodes", episode.Episode("014-right-replay", joined, meta)
  )
  np.savez_compressed(
    args.output_dir / "tracking.npz",
    time_s=track.times,
    target=desired,
    actual=actual,
    position_error_m=pos_error,
    rotation_error_rad=angle_error,
    opening_error_rad=opening_error,
  )
  image_indices = np.flatnonzero(
    (track.image_times >= track.times[0]) & (track.image_times <= track.times[-1])
  )
  image_rows = np.clip(
    np.searchsorted(track.times, track.image_times[image_indices]),
    1,
    len(track.times) - 1,
  )
  image_rows -= np.abs(
    track.times[image_rows - 1] - track.image_times[image_indices]
  ) < np.abs(track.times[image_rows] - track.image_times[image_indices])
  result = {
    "kind": "real capture driving official simulator, not external pose truth",
    "source_hand": args.hand,
    "state_rows": len(desired),
    "time_s": [float(track.times[0]), float(track.times[-1])],
    "physics_dt_s": 1 / 1200,
    "control_dt_s": 0.01,
    "unstable": unstable,
    "position_error_m": {
      "median": float(np.median(pos_error)),
      "p95": float(np.percentile(pos_error, 95)),
      "max": float(pos_error.max()),
    },
    "rotation_error_rad": {
      "p95": float(np.percentile(angle_error, 95)),
      "max": float(angle_error.max()),
    },
    "opening_error_rad": {
      "p95": float(np.percentile(opening_error, 95)),
      "max": float(opening_error.max()),
    },
    "video_frames": len(image_indices),
    "video_nearest_state_max_dt_s": float(
      np.max(np.abs(track.times[image_rows] - track.image_times[image_indices]))
    ),
    "episode": str(episode_path),
    "room_T_world_wxyz": room_T_world.tolist(),
    "limits": [
      "Unobserved left sim hand parked at [10,10,10]; no fabricated left capture",
      "Empty room, no reconstructed objects or task success validation",
      "Replay tracking error is not capture accuracy or real robot performance",
      "Original opening radians are used without normalization; not hardware limit calibration",
    ],
  }
  if not args.no_video:
    filmed = make(
      meta, image_size=args.image_size, supersample=1, image_hands=(args.hand,)
    )
    renderer = mujoco.Renderer(filmed.model, args.image_size, args.image_size)
    scratch = mujoco.MjData(filmed.model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = np.mean(actual[:, :3], axis=0)
    camera.distance = max(1.2, float(np.ptp(actual[:, :3], axis=0).max()) * 2.5)
    camera.azimuth = 135
    camera.elevation = -25
    option = mujoco.MjvOption()
    option.geomgroup[:] = [g == DRAWN for g in range(len(option.geomgroup))]
    fps = 1 / float(np.median(np.diff(track.image_times[image_indices])))
    with video.writer(
      args.output_dir / "replay-overlay.mp4",
      2 * args.image_size,
      2 * args.image_size,
      fps,
    ) as write:
      for index, (source, row) in enumerate(
        zip(image_indices, image_rows, strict=True)
      ):
        q = joined.qpos[row]
        scratch.qpos[:] = q
        mujoco.mj_forward(filmed.model, scratch)
        renderer.update_scene(scratch, camera, scene_option=option)
        overview = renderer.render()
        images, bodies = filmed.trace(q[None])
        simulated = images[0, 0]
        decoded = cv2.imdecode(
          np.frombuffer(track.images[source], np.uint8), cv2.IMREAD_COLOR
        )
        if decoded is None:
          raise ValueError(f"Undecodable source image {source}")
        real = np.asarray(
          cv2.resize(
            cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB),
            (args.image_size, args.image_size),
          ),
          dtype=np.uint8,
        )
        overlay = over(real, simulated, bodies[0, 0])
        panels = [overview, real, simulated, overlay]
        for panel, label in zip(
          panels, ("SIM OVERVIEW", "REAL REAR", "SIM REAR", "REAL + SIM"), strict=True
        ):
          cv2.putText(
            panel,
            label,
            (7, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 190, 0),
            1,
            cv2.LINE_AA,
          )
        frame = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
        write(frame)
        if index == len(image_indices) // 2:
          cv2.imwrite(
            str(args.output_dir / "replay-contact-sheet.jpg"),
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
          )
        if index % 300 == 0:
          print(f"render {index}/{len(image_indices)}", flush=True)
    renderer.close()
    result["video"] = str(args.output_dir / "replay-overlay.mp4")
  (args.output_dir / "replay-result.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n"
  )
  print(json.dumps(result, ensure_ascii=False, indent=2))
  if unstable or not np.isfinite(actual).all():
    raise SystemExit("Simulation became unstable")


if __name__ == "__main__":
  main()
