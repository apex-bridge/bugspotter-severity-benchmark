"""Benchmark runner — drives one (model, mode, hardware) sweep over the eval set.

Reads `dataset/curated.jsonl`, builds zero-shot or few-shot messages per bug,
sends them to the configured backend, writes a JSONL line per call to
`results/<model>-<mode>-<hardware>.jsonl`. Each line is independent so a
crashed run resumes by reading existing lines and skipping their bug IDs.

CLI:
    python -m harness.runner --model claude-sonnet-4-6 --provider anthropic --mode zero-shot --hardware cloud-a100
    python -m harness.runner --model gpt-4o          --provider openai    --mode few-shot
    python -m harness.runner --model llama3.1:8b      --provider ollama    --mode zero-shot --hardware cpu-windows

Hardware tag is free-form — used only to disambiguate result files when
the same model is run on multiple boxes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

from harness.backends import (
    classify_anthropic,
    classify_ollama,
    classify_openai,
)
from harness.prompts import few_shot_messages, parse_class, zero_shot_messages

DATASET = Path(__file__).resolve().parent.parent / "dataset" / "curated.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Polite throttle for hosted API providers — keeps us under per-minute rate
# limits and gives a friendlier cost-per-minute curve in the article.
DEFAULT_API_SLEEP_SEC = 0.2


def _load_curated() -> tuple[list[dict], list[dict]]:
    """Split curated.jsonl into (few_shot_examples, eval_bugs) by `split` tag."""
    if not DATASET.exists():
        sys.exit(f"Curated dataset not found at {DATASET}. Run scripts/curate.py first.")
    bugs = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]
    few_shot = [b for b in bugs if b.get("split") == "few_shot"]
    eval_pool = [b for b in bugs if b.get("split") == "eval"]
    return few_shot, eval_pool


def _make_client_and_fn(provider: str) -> tuple[object, Callable]:
    """Construct the right SDK client + classify function for the provider."""
    if provider == "anthropic":
        import anthropic

        return anthropic.Anthropic(), classify_anthropic
    if provider == "openai":
        import openai

        return openai.OpenAI(), classify_openai
    if provider == "ollama":
        import ollama

        return ollama.Client(), classify_ollama
    sys.exit(f"Unknown provider: {provider}")


def _resume_set(out_path: Path) -> set[int]:
    """Bug IDs already classified — re-run skips them. Enables cheap resume."""
    if not out_path.exists():
        return set()
    done: set[int] = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            done.add(json.loads(line)["bug_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model ID, e.g. claude-sonnet-4-6")
    parser.add_argument(
        "--provider", required=True, choices=("anthropic", "openai", "ollama")
    )
    parser.add_argument("--mode", required=True, choices=("zero-shot", "few-shot"))
    parser.add_argument(
        "--hardware",
        default="default",
        help="Free-form tag distinguishing host (e.g. 'cloud-a100', 'm2-pro', 'cpu-windows').",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap on number of bugs to classify (smoke-testing).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_API_SLEEP_SEC,
        help="Seconds to sleep between API calls (ignored for ollama).",
    )
    args = parser.parse_args()

    few_shot, eval_pool = _load_curated()
    if args.mode == "few-shot" and not few_shot:
        sys.exit("Few-shot mode requires bugs tagged `split: few_shot` in curated.jsonl")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = args.model.replace("/", "_").replace(":", "_")
    out_path = RESULTS_DIR / f"{safe_model}-{args.mode}-{args.hardware}.jsonl"
    already_done = _resume_set(out_path)
    print(
        f"Resuming with {len(already_done)} bugs already classified",
        file=sys.stderr,
    )

    client, classify = _make_client_and_fn(args.provider)
    targets = [b for b in eval_pool if b["id"] not in already_done]
    if args.limit:
        targets = targets[: args.limit]

    with out_path.open("a", encoding="utf-8") as f:
        for i, bug in enumerate(targets, 1):
            messages = (
                zero_shot_messages(bug)
                if args.mode == "zero-shot"
                else few_shot_messages(bug, few_shot)
            )
            try:
                text, latency_ms, usage = classify(client, args.model, messages)
            except Exception as e:  # noqa: BLE001
                # Don't kill the whole run on a single transient failure — log
                # the error to the output line so it shows up in scoring as
                # unparseable, and continue.
                text, latency_ms = "", 0.0
                usage = {"error": str(e)[:500]}
            row: dict[str, Any] = {
                "bug_id": bug["id"],
                "model": args.model,
                "provider": args.provider,
                "mode": args.mode,
                "hardware": args.hardware,
                "gold": bug["normalized_severity"],
                "predicted": parse_class(text),
                "raw_output": text,
                "latency_ms": latency_ms,
                "usage": usage,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

            if args.provider != "ollama" and args.sleep > 0:
                time.sleep(args.sleep)
            if i % 50 == 0:
                print(
                    f"  {i} / {len(targets)} classified (latency p_last={latency_ms:.0f}ms)",
                    file=sys.stderr,
                )

    print(
        f"\nDone. {len(targets)} new classifications written to {out_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
