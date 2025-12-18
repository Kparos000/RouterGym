"""Materialize the curated KB markdown files from ``newkb_dump.txt``.

Each article in ``newkb_dump.txt`` begins with a marker line of the form:

    <!-- file: <relative-path-to-md> -->

Paths can be either:
  - RouterGym/data/policy_kb/<category>/<file>.md, or
  - <category>/<file>.md (relative to policy_kb root)

This script writes each block to the indicated path under
RouterGym/data/policy_kb, creating directories as needed and removing any
stale .md files not present in the dump.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
KB_ROOT = ROOT / "RouterGym" / "data" / "policy_kb"
DEFAULT_DUMP = ROOT / "newkb_dump.txt"

MARKER_RE = re.compile(r"<!--\s*file:\s*(.*?)\s*-->")


def _resolve_path(path_str: str) -> Path:
    """Resolve a marker path to an absolute path under KB_ROOT."""
    path = Path(path_str.strip())
    if path.is_absolute():
        return path
    # If the marker already includes RouterGym/data/policy_kb, strip the prefix.
    parts = list(path.parts)
    if len(parts) >= 4 and parts[0].lower() == "routergym" and parts[1] == "data" and parts[2] == "policy_kb":
        path = Path(*parts[3:])
    return KB_ROOT / path


def parse_dump(dump_path: Path) -> List[Tuple[Path, List[str]]]:
    """Parse dump file into (dest_path, content_lines) tuples."""
    blocks: List[Tuple[Path, List[str]]] = []
    current_path: Path | None = None
    current_lines: List[str] = []

    with dump_path.open("r", encoding="utf-8") as f:
        for line in f:
            m = MARKER_RE.match(line)
            if m:
                # Flush previous block
                if current_path is not None:
                    blocks.append((current_path, current_lines))
                current_path = _resolve_path(m.group(1))
                current_lines = []
            else:
                if current_path is not None:
                    current_lines.append(line)
        # Flush last block
        if current_path is not None:
            blocks.append((current_path, current_lines))

    if not blocks:
        raise RuntimeError(f"No KB markers found in {dump_path}")
    return blocks


def materialize(blocks: List[Tuple[Path, List[str]]]) -> None:
    """Write blocks to disk and remove stale files."""
    expected_paths = {path.resolve() for path, _ in blocks}

    for path, lines in blocks:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            f.writelines(lines)

    # Remove stale .md files under KB_ROOT not present in the dump
    for md_file in KB_ROOT.rglob("*.md"):
        if md_file.resolve() not in expected_paths:
            md_file.unlink()


def main() -> None:
    if not DEFAULT_DUMP.exists():
        raise FileNotFoundError(f"KB dump not found at {DEFAULT_DUMP}")
    blocks = parse_dump(DEFAULT_DUMP)
    materialize(blocks)
    print(f"Wrote {len(blocks)} KB files into {KB_ROOT}")


if __name__ == "__main__":
    main()
