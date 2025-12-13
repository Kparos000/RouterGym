"""Generate a deterministic project structure tree and write it to project_structure.txt."""

from __future__ import annotations

from pathlib import Path
from typing import List


EXCLUDE_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "venv",
    ".venv",
    ".idea",
    ".vscode",
}


def _should_skip(path: Path) -> bool:
    parts = path.parts
    return any(part in EXCLUDE_DIRS for part in parts)


def _build_tree_lines(root: Path) -> List[str]:
    lines: List[str] = [root.name or "."]

    def walk(current: Path, prefix: str) -> None:
        entries = sorted([p for p in current.iterdir() if not _should_skip(p)], key=lambda p: p.name.lower())
        for idx, entry in enumerate(entries):
            connector = "+---"
            is_last = idx == len(entries) - 1
            lines.append(f"{prefix}{connector}{entry.name}")
            if entry.is_dir():
                next_prefix = f"{prefix}{'    ' if is_last else '|   '}"
                walk(entry, next_prefix)

    walk(root, "")
    return lines


def generate_structure(output_path: Path) -> None:
    root = Path(".").resolve()
    lines = _build_tree_lines(root)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    output_file = Path("project_structure.txt")
    generate_structure(output_file)


if __name__ == "__main__":
    main()
