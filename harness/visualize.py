"""Generate the article's figures from scored results.

Produces four PNGs in results/figures/:
  - leaderboard.png         — horizontal bar chart, weighted F1 per (model, mode)
  - per_class_f1.png        — grouped bar chart, F1 by class per model
  - latency_boxplot.png     — per-(model, mode, hw) latency distribution
  - cost_vs_quality.png     — log-scale scatter, $/1K calls × weighted F1
  - confusion_<file>.png    — 4×4 + "unparseable" matrix per result file

Run after `scripts/run.ps1 -Module harness.runner ...` has populated
results/*.jsonl. Reuses `harness.scoring.score_file` so the numbers
match what the scoring CLI prints.

Usage:
    python -m harness.visualize results/*.jsonl
    python -m harness.visualize results/claude-*-api.jsonl results/*cloud-a100.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from harness.prompts import CLASSES
from harness.scoring import score_file

FIGURES_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"


def _label(s: dict[str, Any]) -> str:
    return f"{s['model']} ({s['mode']})"


def plot_leaderboard(summaries: list[dict[str, Any]], out: Path) -> None:
    summaries = sorted(summaries, key=lambda s: s["weighted_f1"])
    labels = [_label(s) for s in summaries]
    scores = [s["weighted_f1"] for s in summaries]
    colors = ["#7BAAF7" if "api" in (s["hardware"] or "") else "#F6AE2D" for s in summaries]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(summaries))))
    bars = ax.barh(labels, scores, color=colors)
    ax.set_xlabel("Weighted F1")
    ax.set_xlim(0, max(scores) * 1.15)
    ax.set_title("Bug-severity classification — weighted F1 by model × mode")
    for bar, score in zip(bars, scores, strict=False):
        ax.text(score + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=9)
    api_patch = plt.Rectangle((0, 0), 1, 1, color="#7BAAF7", label="hosted API")
    local_patch = plt.Rectangle((0, 0), 1, 1, color="#F6AE2D", label="local (cloud A100)")
    ax.legend(handles=[api_patch, local_patch], loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_per_class_f1(summaries: list[dict[str, Any]], out: Path) -> None:
    summaries = sorted(summaries, key=lambda s: s["weighted_f1"], reverse=True)
    labels = [_label(s) for s in summaries]
    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(labels)), 5))
    for i, cls in enumerate(CLASSES):
        vals = [s["per_class"][cls]["f1"] for s in summaries]
        ax.bar(x + (i - 1.5) * width, vals, width, label=cls)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("F1")
    ax.set_title("Per-class F1 — note `high` collapses everywhere (only 31 eval bugs)")
    ax.legend(title="severity class")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_latency(summaries: list[dict[str, Any]], out: Path) -> None:
    summaries = sorted(summaries, key=lambda s: s["latency"]["p50_ms"])
    labels = [_label(s) for s in summaries]
    p50 = [s["latency"]["p50_ms"] for s in summaries]
    p95 = [s["latency"]["p95_ms"] for s in summaries]
    colors = ["#7BAAF7" if "api" in (s["hardware"] or "") else "#F6AE2D" for s in summaries]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(summaries))))
    y = np.arange(len(labels))
    ax.barh(y, p95, color=colors, alpha=0.4, label="p95")
    ax.barh(y, p50, color=colors, label="p50")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Latency (ms)")
    ax.set_title("Per-call latency — p50 (solid) and p95 (faded)")
    for i, (a, b) in enumerate(zip(p50, p95, strict=False)):
        ax.text(b + 50, i, f"p50 {a:.0f} / p95 {b:.0f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_cost_vs_quality(summaries: list[dict[str, Any]], out: Path) -> None:
    # Only points with a cost (i.e. API models) — local models are amortized
    # differently and would distort a $-axis.
    points = [s for s in summaries if s.get("cost")]
    if not points:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for s in points:
        cost_per_1k = s["cost"]["per_call_usd"] * 1000
        ax.scatter(cost_per_1k, s["weighted_f1"], s=120, alpha=0.75)
        ax.annotate(_label(s), (cost_per_1k, s["weighted_f1"]),
                    xytext=(7, 4), textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Cost per 1 000 classifications (USD, log scale)")
    ax.set_ylabel("Weighted F1")
    ax.set_title("Hosted-API cost / quality — caching is the main lever")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_confusion(summary: dict[str, Any], out: Path) -> None:
    cm = summary["confusion"]
    cols = list(CLASSES) + ["unparseable"]
    rows = list(CLASSES)
    matrix = np.array([[cm[r][c] for c in cols] for r in rows])
    # Normalize by row (recall view)
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(norm, annot=matrix, fmt="d", cmap="Blues", cbar_kws={"label": "row-normalized"},
                xticklabels=cols, yticklabels=rows, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title(f"{summary['model']} — {summary['mode']} ({summary['hardware'] or '?'})")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    summaries = [score_file(p) for p in args.files]
    # Drop runs where everything was unparseable — they'd skew the visuals.
    usable = [s for s in summaries if s["n"] > 0 and s["unparseable"] < s["n"]]

    plot_leaderboard(usable, FIGURES_DIR / "leaderboard.png")
    plot_per_class_f1(usable, FIGURES_DIR / "per_class_f1.png")
    plot_latency(usable, FIGURES_DIR / "latency.png")
    plot_cost_vs_quality(usable, FIGURES_DIR / "cost_vs_quality.png")

    for s in usable:
        stem = Path(s["file"]).stem
        plot_confusion(s, FIGURES_DIR / f"confusion_{stem}.png")

    skipped = [s for s in summaries if s["unparseable"] >= s["n"]]
    if skipped:
        print(f"Skipped {len(skipped)} all-unparseable runs: "
              + ", ".join(_label(s) for s in skipped))
    print(f"Wrote {len(usable) + 4} figures to {FIGURES_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
