"""Build the structured policy KB index from markdown articles."""

from __future__ import annotations

from RouterGym.data.policy_kb.kb_loader import build_kb_index


def main() -> None:
    output_path = build_kb_index()
    print(f"KB index built at {output_path}")


if __name__ == "__main__":
    main()
