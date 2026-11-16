"""Verify the single report, evidence links, candidate version and figure captions."""

import argparse
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


class Report(HTMLParser):
  def __init__(self):
    super().__init__()
    self.ids = set()
    self.links = []
    self.figures = 0
    self.captions = 0

  def handle_starttag(self, tag, attrs):
    attrs = dict(attrs)
    if "id" in attrs:
      assert attrs["id"] not in self.ids, f"duplicate ID: {attrs['id']}"
      self.ids.add(attrs["id"])
    if tag == "a" and "href" in attrs:
      self.links.append(attrs["href"])
    self.figures += tag == "figure"
    self.captions += tag == "figcaption"


def main():
  cli = argparse.ArgumentParser(description=__doc__)
  cli.add_argument(
    "--require-local-data",
    action="store_true",
    help="Also require gitignored recordings and logs",
  )
  args = cli.parse_args()
  absent_runtime = set()
  version = json.loads((ROOT / "document-version.json").read_text())
  source = ROOT / version["main_source"]
  output = ROOT / version["main_report"]
  for path in (source, output):
    text = path.read_text()
    assert version["version"] in text, f"version missing: {path}"
    parser = Report()
    parser.feed(text)
    assert parser.figures > 0 and parser.figures == parser.captions
    for link in parser.links:
      parsed = urlsplit(link)
      if parsed.scheme or parsed.netloc:
        continue
      if not parsed.path and parsed.fragment:
        assert parsed.fragment in parser.ids, f"missing anchor: {link}"
      elif parsed.path:
        target = (path.parent / unquote(parsed.path)).resolve()
        if not args.require_local_data and target.is_relative_to(ROOT / "logs"):
          if not target.exists():
            absent_runtime.add(target)
          continue
        assert target.exists(), f"missing file: {target}"
  sources = json.loads((ROOT / "reports/evidence/sources.json").read_text())
  checks = json.loads((ROOT / "reports/evidence/checks.json").read_text())
  assert sources and checks
  assert "未连接" in output.read_text()
  print(f"OK: {version['version']}; source, output, local links, captions and evidence")
  if absent_runtime:
    print(
      f"Local-only artifacts absent: {len(absent_runtime)} (reproduce locally; not included in Git)"
    )


if __name__ == "__main__":
  main()
