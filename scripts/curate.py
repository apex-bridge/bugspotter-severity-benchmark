"""Curate `dataset/raw.jsonl` into a balanced, deterministic `dataset/curated.jsonl`.

The raw collector pulls more bugs than we need so curation can be strict about
quality. This step:

  1. Drops bugs with missing / very short summary (or empty description if
     --require-description is set) — those are usually triage-noise.
  2. Drops bugs where the description is dominated by stack-trace boilerplate
     (heuristic: more than 70% of lines start with whitespace + "at ").
  3. Caps each normalized severity class at `--per-class` items so the eval
     set is balanced regardless of how many raw bugs each class produced.
  4. Tags 3 examples per class as `split=few_shot` (12 total); the rest
     become `split=eval`. The few-shot pool is selected deterministically
     by sorted bug id so a re-curation with the same seed produces the
     same pool.
  5. Shuffles the eval split with a fixed seed for downstream reproducibility.

Usage:
    python scripts/curate.py
    python scripts/curate.py --per-class 250 --min-summary-len 20
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

DEFAULT_RAW_PATH = Path(__file__).resolve().parent.parent / "dataset" / "raw.jsonl"
DEFAULT_CURATED_PATH = Path(__file__).resolve().parent.parent / "dataset" / "curated.jsonl"
DEFAULT_PER_CLASS = 250
DEFAULT_FEWSHOT_PER_CLASS = 3
DEFAULT_SEED = 42

CLASSES = ("low", "medium", "high", "critical")


def looks_like_stack_trace(text: str) -> bool:
    """Heuristic: >70% of non-empty lines start with whitespace + 'at '."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 5:
        return False
    stacky = sum(1 for ln in lines if ln.lstrip().startswith("at "))
    return stacky / len(lines) > 0.7


def quality_pass(bug: dict, *, min_summary_len: int, require_description: bool) -> bool:
    summary = (bug.get("summary") or "").strip()
    if len(summary) < min_summary_len:
        return False
    desc = (bug.get("description") or "").strip()
    if require_description and not desc:
        return False
    if desc and looks_like_stack_trace(desc):
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_CURATED_PATH)
    parser.add_argument(
        "--per-class",
        type=int,
        default=DEFAULT_PER_CLASS,
        help="Final bugs per normalized severity class (default: %(default)s).",
    )
    parser.add_argument(
        "--few-shot-per-class",
        type=int,
        default=DEFAULT_FEWSHOT_PER_CLASS,
        help="Bugs per class reserved for the few-shot in-context pool (default: %(default)s).",
    )
    parser.add_argument(
        "--min-summary-len",
        type=int,
        default=20,
        help="Skip bugs whose summary is shorter than this many chars.",
    )
    parser.add_argument(
        "--require-description",
        action="store_true",
        help="Also drop bugs without a description (requires fetch with --with-description).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if not args.raw.exists():
        sys.exit(f"Raw dataset not found at {args.raw}. Run scripts/fetch_bugzilla.py first.")

    by_class: dict[str, list[dict]] = {c: [] for c in CLASSES}
    skipped = 0
    with args.raw.open(encoding="utf-8") as f:
        for line in f:
            bug = json.loads(line)
            cls = bug.get("normalized_severity")
            if cls not in by_class:
                continue
            if not quality_pass(
                bug,
                min_summary_len=args.min_summary_len,
                require_description=args.require_description,
            ):
                skipped += 1
                continue
            by_class[cls].append(bug)

    # Sort each class by id for a deterministic few-shot pool, then shuffle the
    # rest with the seed for a stable eval order.
    rng = random.Random(args.seed)
    out: list[dict] = []
    for cls in CLASSES:
        bugs = sorted(by_class[cls], key=lambda b: b["id"])
        if len(bugs) < args.per_class:
            print(
                f"WARNING: class '{cls}' has only {len(bugs)} bugs after quality pass; "
                f"requested {args.per_class}. Using all available.",
                file=sys.stderr,
            )
        kept = bugs[: args.per_class]
        few_shot = kept[: args.few_shot_per_class]
        eval_pool = kept[args.few_shot_per_class :]
        rng.shuffle(eval_pool)
        for b in few_shot:
            b["split"] = "few_shot"
        for b in eval_pool:
            b["split"] = "eval"
        out.extend(few_shot)
        out.extend(eval_pool)
        print(
            f"  {cls}: kept {len(kept)} (few_shot={len(few_shot)}, eval={len(eval_pool)})",
            file=sys.stderr,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for bug in out:
            f.write(json.dumps(bug, ensure_ascii=False) + "\n")

    print(
        f"\nWrote {len(out)} bugs to {args.output} (skipped {skipped} for quality).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
