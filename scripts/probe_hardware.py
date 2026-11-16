"""List USB devices and optionally read Feetech feedback. Never commands motion."""

import argparse
import importlib
import json
import subprocess
from pathlib import Path
from typing import Any


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--port", help="optional explicit serial port")
  parser.add_argument("--ids", type=int, nargs="+", default=[18])
  parser.add_argument("--output", type=Path, default=Path("logs/setup/hardware.json"))
  args = parser.parse_args()
  usb = subprocess.run(["lsusb"], capture_output=True, text=True, check=True).stdout
  result: dict[str, Any] = {
    "usb": usb.splitlines(),
    "insta360_detected": "2e1a:" in usb.lower(),
    "serial_ports": [str(p) for p in Path("/dev/serial/by-id").glob("*")],
    "motion_commanded": False,
    "camera_stream_tested": False,
    "servo_feedback_tested": False,
  }
  if args.port:
    feetech: Any = importlib.import_module("feetech_driver")

    with feetech.Bus(args.port, 1000000) as bus:
      result["feedback"] = [
        {"id": sid, "position_counts": feedback.position_counts if feedback else None}
        for sid, feedback in zip(
          args.ids, bus.sync_read_feedback(args.ids), strict=True
        )
      ]
      result["servo_feedback_tested"] = True
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
  print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
