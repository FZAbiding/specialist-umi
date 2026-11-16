"""Copy original paired captures without renaming them; write a SHA-256 manifest."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from specialist.loader.capture import probe_capture
from specialist.loader.parse_insv import load_insv


def digest(path: Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as f:
    for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
      h.update(block)
  return h.hexdigest()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input-insvs", type=Path, nargs="+", required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  args = parser.parse_args()
  staged: list[tuple[Path, Path, str]] = []
  for master in args.input_insvs:
    capture = probe_capture(master)
    telemetry = load_insv(master)
    serial = str(telemetry.capture.get("serial_number", ""))
    if not serial or Path(serial).name != serial or serial in (".", ".."):
      raise SystemExit(f"Invalid or missing camera serial in {master}")
    for src in (master, capture.rear_path):
      sha = digest(src)
      dst = args.output_dir / "raw" / serial / src.name
      if dst.exists() and digest(dst) != sha:
        raise SystemExit(f"Refusing to overwrite different original: {dst}")
      staged.append((src, dst, sha))
  manifest = args.output_dir / "raw" / "manifest.json"
  records: list[dict[str, Any]] = (
    json.loads(manifest.read_text()) if manifest.exists() else []
  )
  indexed = {entry["stored"]: entry for entry in records}
  for src, dst, sha in staged:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
      temporary = dst.with_suffix(dst.suffix + ".partial")
      shutil.copy2(src, temporary)
      if digest(temporary) != sha:
        raise SystemExit(f"Copy checksum mismatch: {temporary}")
      temporary.replace(dst)
    indexed[str(dst.resolve())] = {
      "source": str(src.resolve()),
      "stored": str(dst.resolve()),
      "bytes": dst.stat().st_size,
      "sha256": sha,
      "kind": "real-camera-original",
    }
  temporary = manifest.with_suffix(".json.partial")
  temporary.write_text(
    json.dumps(list(indexed.values()), ensure_ascii=False, indent=2) + "\n"
  )
  temporary.replace(manifest)
  print(f"Verified {len(staged)} original files. Manifest: {manifest}")


if __name__ == "__main__":
  main()
