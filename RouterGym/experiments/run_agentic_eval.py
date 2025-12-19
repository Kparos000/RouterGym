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
    ticket_start: int,
    router_mode: str,
    memory_mode: str,
    model_name: str,
    escalation_model: Optional[str],
    output_path: Path,
) -> None:
    """Run the agentic pipeline over tickets and write results as JSONL."""
    df = load_dataset(n=None, start=ticket_start)
    if ticket_limit is not None:
        df = df.head(ticket_limit)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for idx, row in df.iterrows():
            ticket: Dict[str, Any] = {"text": str(row["text"]), "ticket_id": int(idx)}
            result: Dict[str, Any] = run_ticket_pipeline(
                ticket=ticket,
                base_model_name=model_name,
                memory_mode=memory_mode,
                router_mode=router_mode,
                escalation_model_name=escalation_model,
            )
            validated = validate_agent_output(result)
            # Add lightweight metadata for downstream analysis.
            validated["ticket_id"] = str(idx)
            validated["gold_label"] = row.get("label")
            fh.write(json.dumps(validated, ensure_ascii=False) + "\n")
            count += 1
    print(f"[AgenticEval] Wrote {count} rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agentic pipeline over tickets and emit AgentOutput JSONL.")
    parser.add_argument("--ticket-limit", type=int, default=None, help="Limit number of tickets (optional).")
    parser.add_argument("--ticket-start", type=int, default=0, help="Row offset to start from (default 0).")
    parser.add_argument("--router-mode", type=str, default="manual", help="Router strategy name (recorded only).")
    parser.add_argument("--memory-mode", type=str, default="none", help="Context mode (none, rag_dense, rag_bm25, rag_hybrid).")
    parser.add_argument("--model-name", type=str, default="slm1", help="Model name (slm1/slm2/llm1/llm2).")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("RouterGym/results/experiments/agentic_eval.jsonl"),
        help="Path to write JSONL results.",
    )
    parser.add_argument(
        "--escalation-model",
        type=str,
        default=None,
        help="Optional escalation model name for slm_dominant routing (e.g., llm1 or llm2).",
    )
    args = parser.parse_args()

    run_agentic_eval(
        ticket_limit=args.ticket_limit,
        ticket_start=args.ticket_start,
        router_mode=args.router_mode,
        memory_mode=args.memory_mode,
        model_name=args.model_name,
        escalation_model=args.escalation_model,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
