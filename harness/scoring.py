"""Metrics + cost analysis for the severity benchmark.

Reads JSONL result files (one per model × mode × hardware) and produces:
  - Per-class precision / recall / F1
  - Weighted F1 (primary score — single number to rank models)
  - Confusion matrix (4×4 for the four normalized severity classes,
    plus an "unparseable" column for model outputs the prompts.parse_class
    couldn't map to a class).
  - p50 / p95 latency
  - Token usage totals and cache-read share (Anthropic only).
  - Cost projection per 10K classifications at current API prices.

Designed to be importable by analysis notebooks AND runnable as a CLI
(`python -m harness.scoring results/*.jsonl`).
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from harness.prompts import CLASSES


# Prices in USD per 1M tokens, as of 2026-05 (cached — re-check before final
# publication). Local Ollama models intentionally absent: the article reports
# hardware-amortized cost separately, computed from latency × power draw.
PRICING_USD_PER_1M = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    # Anthropic — bare aliases (no date suffix per SDK convention)
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00, "cache_read": 0.30},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00, "cache_read": 0.10},
}


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def per_class_metrics(predictions: list[tuple[str | None, str]]) -> dict[str, dict[str, float]]:
    """Per-class precision/recall/F1 from (predicted, gold) pairs.

    `predicted` may be None for unparseable outputs — those count as wrong
    everywhere they appear (treated as if the model picked "no class").
    """
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    for pred, gold in predictions:
        if pred == gold:
            tp[gold] += 1
        else:
            fn[gold] += 1
            if pred is not None:
                fp[pred] += 1
    out: dict[str, dict[str, float]] = {}
    for cls in CLASSES:
        p = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) else 0.0
        r = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) else 0.0
        out[cls] = {"precision": p, "recall": r, "f1": _f1(p, r), "support": tp[cls] + fn[cls]}
    return out


def weighted_f1(per_class: dict[str, dict[str, float]]) -> float:
    """Support-weighted F1 — the primary single-number score for the article."""
    total_support = sum(m["support"] for m in per_class.values())
    if total_support == 0:
        return 0.0
    return sum(m["f1"] * m["support"] for m in per_class.values()) / total_support


def confusion_matrix(
    predictions: list[tuple[str | None, str]],
) -> dict[str, dict[str, int]]:
    """{gold_class: {predicted_class | "unparseable": count}}."""
    cols = list(CLASSES) + ["unparseable"]
    matrix = {gold: {col: 0 for col in cols} for gold in CLASSES}
    for pred, gold in predictions:
        if gold not in matrix:
            continue
        key = pred if pred in CLASSES else "unparseable"
        matrix[gold][key] += 1
    return matrix


def latency_stats(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0, "n": 0}
    return {
        "p50_ms": statistics.median(latencies_ms),
        "p95_ms": statistics.quantiles(latencies_ms, n=20)[18]
        if len(latencies_ms) >= 20
        else max(latencies_ms),
        "mean_ms": statistics.fmean(latencies_ms),
        "n": len(latencies_ms),
    }


def cost_projection(
    model: str,
    rows: list[dict[str, Any]],
    projected_calls: int = 10_000,
) -> dict[str, float] | None:
    """Per-10K-classification cost in USD, extrapolated from observed usage.

    Returns None for models without pricing (local Ollama). Anthropic
    cache-read tokens are billed at the cache_read rate, not the input rate.
    """
    if model not in PRICING_USD_PER_1M:
        return None
    price = PRICING_USD_PER_1M[model]
    total_input = sum(r["usage"].get("input_tokens", 0) for r in rows)
    total_output = sum(r["usage"].get("output_tokens", 0) for r in rows)
    total_cache_read = sum(
        r["usage"].get("cache_read_input_tokens", 0) for r in rows
    )
    n = len(rows)
    if n == 0:
        return None
    avg_input = total_input / n
    avg_output = total_output / n
    avg_cache_read = total_cache_read / n
    cost_per_call = (
        (avg_input - avg_cache_read) * price["input"] / 1_000_000
        + avg_cache_read * price.get("cache_read", price["input"]) / 1_000_000
        + avg_output * price["output"] / 1_000_000
    )
    return {
        "per_call_usd": cost_per_call,
        f"per_{projected_calls}_calls_usd": cost_per_call * projected_calls,
        "avg_input_tokens": avg_input,
        "avg_output_tokens": avg_output,
        "avg_cache_read_tokens": avg_cache_read,
        "cache_hit_share": avg_cache_read / avg_input if avg_input else 0.0,
    }


def score_file(path: Path) -> dict[str, Any]:
    """Score a single results JSONL file. Each line is one classification."""
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    predictions = [(r.get("predicted"), r["gold"]) for r in rows]
    latencies = [r["latency_ms"] for r in rows]
    per_cls = per_class_metrics(predictions)
    return {
        "file": str(path),
        "model": rows[0]["model"] if rows else None,
        "mode": rows[0]["mode"] if rows else None,
        "hardware": rows[0].get("hardware") if rows else None,
        "n": len(rows),
        "weighted_f1": weighted_f1(per_cls),
        "per_class": per_cls,
        "confusion": confusion_matrix(predictions),
        "latency": latency_stats(latencies),
        "unparseable": sum(1 for p, _ in predictions if p is None),
        "cost": cost_projection(rows[0]["model"], rows) if rows else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+")
    args = parser.parse_args()
    summaries = [score_file(p) for p in args.files]
    print(json.dumps(summaries, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
