"""Run the official PickPlace environment and save a scripted episode.

The scene, physics, fingers and renderer come from specialist_sim. The short
trajectory below is a local reproducibility demo, not a trained official policy.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
from specialist_sim import episode, rotation
from specialist_sim.env import ROWWISE, Cfg, Env
from specialist_sim.episode import Episode
from specialist_sim.tasks.pick_place import PickPlace

Float64Array = npt.NDArray[np.float64]


def along(plan: list[tuple[float, Float64Array]], at: float) -> Float64Array:
  if at <= plan[0][0]:
    return plan[0][1].copy()
  for (t0, a), (t1, b) in zip(plan, plan[1:], strict=False):
    if at <= t1:
      u = np.clip((at - t0) / (t1 - t0), 0.0, 1.0)
      u = u * u * (3 - 2 * u)
      out: Float64Array = a + u * (b - a)
      out[:7] = rotation.interpolate(a, b, u)
      return out
  return plan[-1][1].copy()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", type=Path, default=Path("logs/official-demo"))
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  cfg = Cfg(clock=ROWWISE, image_hands=(), autoreset=False)
  env = Env(PickPlace(), cfg)
  obs, _ = env.reset([args.seed])
  homes = obs.state[0, -1].copy()
  cube = env.objects().poses[0, 0].copy()
  axis = rotation.rotate(cube[3:7], np.array([1.0, 0.0, 0.0]))
  axis[2] = 0
  axis /= np.linalg.norm(axis)
  if np.dot(axis, cube[:3] - homes[1, :3]) < 0:
    axis = -axis
  pitch = np.deg2rad(48.0)
  quat = rotation.pointing(np.cos(pitch) * axis - np.array([0, 0, np.sin(pitch)]))
  grip = np.array(
    [
      *(cube[:3] - rotation.rotate(quat, np.array([0.098, 0.0, -0.0177]))),
      *quat,
      0.4,
    ]
  )
  above = grip.copy()
  above[2] += 0.09
  closed = grip.copy()
  closed[7] = -0.15
  lifted = closed.copy()
  lifted[2] += 0.12
  carried = lifted.copy()
  carried[:2] -= cube[:2]
  placed = carried.copy()
  placed[2] -= 0.12
  released = placed.copy()
  released[7] = 0.4
  retreat = released.copy()
  retreat[2] += 0.12
  plan = [
    (0.0, homes[1]),
    (1.0, above),
    (1.8, grip),
    (2.5, closed),
    (3.5, lifted),
    (4.7, carried),
    (5.6, placed),
    (6.4, released),
    (7.3, retreat),
    (8.2, retreat),
  ]
  rows = [env.rows(0)]
  max_error = 0.0
  unstable = False
  for row in range(1, round(8.2 / cfg.clock.row_dt) + 1):
    target = homes.copy()
    target[1] = along(plan, row * cfg.clock.row_dt)
    _, info = env.step(target[None, None])
    rows.append(env.rows(0))
    max_error = max(max_error, float(info["tracking_error"][0]))
    unstable |= bool(info["unstable"][0])
  meta = env.describe() | {"collector": "local-scripted-demo", "seed": args.seed}
  path = episode.save(
    args.output_dir / "episodes", Episode("pick-place-demo", episode.joined(rows), meta)
  )
  report = {
    "environment": "official specialist_sim PickPlace / Floating",
    "controller": "local scripted trajectory; no trained policy",
    "episode": str(path.resolve()),
    "seed": args.seed,
    "success": bool(info["success"][0]),
    "picked": bool(info["labels"]["picked"][0]),
    "unstable": unstable,
    "max_tracking_error_m": max_error,
    "cube_start": cube[:3].tolist(),
    "cube_final": env.objects().poses[0, 0, :3].tolist(),
    "rows": len(episode.joined(rows).timestamp),
  }
  (args.output_dir / "demo-result.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n"
  )
  print(json.dumps(report, ensure_ascii=False, indent=2))
  if unstable or not report["success"] or not report["picked"]:
    raise SystemExit("The recorded demo did not pass pick-and-place validation")


if __name__ == "__main__":
  main()
