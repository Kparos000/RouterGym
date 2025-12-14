"""Agentic evaluation harness that runs the ticket pipeline over the dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from RouterGym.agents.generator import run_ticket_pipeline
from RouterGym.contracts.json_contract import validate_agent_output
from RouterGym.data.tickets.dataset_loader import load_dataset


def run_agentic_eval(
    ticket_limit: Optional[int],
    router_name: str,
    context_mode: str,
    output_path: Path,
) -> None:
    """Run the agentic pipeline over tickets and write results as JSONL."""
    df = load_dataset(n=None)
    if ticket_limit is not None:
        df = df.head(ticket_limit)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for idx, row in df.iterrows():
            ticket_text = str(row["text"])
            result: Dict[str, Any] = run_ticket_pipeline(
                ticket_text=ticket_text,
                router_name=router_name,
                context_mode=context_mode,
            )
            validated = validate_agent_output(result)
            # Add lightweight metadata for downstream analysis.
            validated["ticket_id"] = int(idx)
            validated["gold_label"] = row.get("label")
            fh.write(json.dumps(validated, ensure_ascii=False) + "\n")
            count += 1
    print(f"[AgenticEval] Wrote {count} rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agentic pipeline over tickets and emit AgentOutput JSONL.")
    parser.add_argument("--ticket-limit", type=int, default=None, help="Limit number of tickets (optional).")
    parser.add_argument("--router-name", type=str, default="slm_dominant", help="Router strategy name.")
    parser.add_argument("--context-mode", type=str, default="none", help="Context mode (e.g., none, rag_dense).")
    parser.add_argument("--output-path", type=Path, required=True, help="Path to write JSONL results.")
    args = parser.parse_args()

    run_agentic_eval(
        ticket_limit=args.ticket_limit,
        router_name=args.router_name,
        context_mode=args.context_mode,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
