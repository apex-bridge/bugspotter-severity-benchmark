"""Collect Mozilla Bugzilla bugs with severity labels for the severity-classification benchmark.

Stdlib only — no external deps. Writes `dataset/raw.jsonl`.

Bug list responses contain `summary` but not the first comment ("description"); we
fetch comments in a second pass so the harness has the same text a model would see
in a real bug-triage UI. To stay polite we cap concurrency to one request at a
time and sleep between requests.

Usage:
    python scripts/fetch_bugzilla.py
    python scripts/fetch_bugzilla.py --per-class 600 --with-description
    python scripts/fetch_bugzilla.py --since 2022-01-01
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API_BUG = "https://bugzilla.mozilla.org/rest/bug"
DEFAULT_RAW_PATH = Path(__file__).resolve().parent.parent / "dataset" / "raw.jsonl"
DEFAULT_PER_CLASS = 400  # before curation; final dataset will be ~1000 across 4 classes

# Mozilla's raw severity → normalized 4-class label.
# 'enhancement' is excluded entirely (not a bug).
SEVERITY_MAP = {
    "blocker": "critical",
    "critical": "critical",
    "major": "high",
    "normal": "medium",
    "minor": "low",
    "trivial": "low",
}

# Source severities grouped by their normalized target so the final raw pool stays
# balanced even when one normalized class has multiple source values.
RAW_SEVERITIES_BY_CLASS: dict[str, list[str]] = {
    "critical": ["blocker", "critical"],
    "high": ["major"],
    "medium": ["normal"],
    "low": ["minor", "trivial"],
}

INCLUDE_FIELDS = [
    "id",
    "summary",
    "severity",
    "priority",
    "product",
    "component",
    "status",
    "resolution",
    "creation_time",
]
PAGE_LIMIT = 500  # max per Bugzilla REST page
REQUEST_SLEEP_SEC = 0.5  # be polite to the public API


def _get(url: str, *, max_retries: int = 5) -> dict:
    """GET + JSON decode with retries on 502/503/504/network errors.

    Mozilla Bugzilla's nginx front returns transient 502s under load. We back
    off exponentially (1s, 2s, 4s, 8s, 16s) and only abort if every retry
    fails — so a single bad minute doesn't kill a 30-minute run.
    """
    last_err: str = ""
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 502, 503, 504) and attempt < max_retries - 1:
                wait = 2**attempt
                print(
                    f"  Bugzilla {e.code} on retry {attempt + 1}/{max_retries}, "
                    f"sleeping {wait}s...",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            last_err = f"Bugzilla {e.code} on {url} — body: {e.read()[:500].decode(errors='replace')}"
            break
        except urllib.error.URLError as e:
            if attempt < max_retries - 1:
                wait = 2**attempt
                print(
                    f"  Network error on retry {attempt + 1}/{max_retries} ({e.reason}), "
                    f"sleeping {wait}s...",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            last_err = f"Network error contacting Bugzilla ({e.reason}) after {max_retries} retries."
            break
    sys.exit(last_err)


def fetch_severity_page(
    severity: str,
    *,
    since: str,
    limit: int,
    offset: int,
) -> list[dict]:
    """One page of FIXED bugs at the given Bugzilla severity, oldest-first.

    We pull oldest-first so re-running the script is idempotent across new bug
    arrivals — Mozilla file new bugs constantly, and a newest-first scan would
    return different IDs on every run.
    """
    params = {
        "severity": severity,
        "resolution": "FIXED",
        "creation_time": since,
        "include_fields": ",".join(INCLUDE_FIELDS),
        "limit": str(limit),
        "offset": str(offset),
        "order": "creation_time",  # ascending — stable across reruns
    }
    url = f"{API_BUG}?{urllib.parse.urlencode(params)}"
    data = _get(url)
    return data.get("bugs", [])


def collect_for_severity(severity: str, target: int, since: str) -> list[dict]:
    """Pull `target` bugs at a single raw severity, paginating as needed."""
    collected: list[dict] = []
    offset = 0
    while len(collected) < target:
        page = fetch_severity_page(severity, since=since, limit=PAGE_LIMIT, offset=offset)
        if not page:
            break  # no more bugs at this severity in the time window
        collected.extend(page)
        offset += PAGE_LIMIT
        time.sleep(REQUEST_SLEEP_SEC)
        print(f"  {severity}: collected {len(collected)} / {target}", file=sys.stderr)
    return collected[:target]


def fetch_first_comment(bug_id: int) -> str:
    """Fetch the bug's first comment (Bugzilla's notion of 'description').

    Mozilla bug pages render comment 0 as the description; subsequent comments are
    triage threads. We only keep comment 0.
    """
    url = f"{API_BUG}/{bug_id}/comment"
    data = _get(url)
    comments = data.get("bugs", {}).get(str(bug_id), {}).get("comments", [])
    if not comments:
        return ""
    return comments[0].get("text", "").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-class",
        type=int,
        default=DEFAULT_PER_CLASS,
        help="Bugs to collect per normalized class before curation (default: %(default)s).",
    )
    parser.add_argument(
        "--since",
        default="2020-01-01",
        help="Only collect bugs created on or after this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--with-description",
        action="store_true",
        help="Also fetch the bug's first comment as 'description'. Slower (one extra request per bug).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RAW_PATH,
        help="Output JSONL path (default: %(default)s).",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Phase 1: bug list. Resume from existing output if present — re-running the
    # script after a partial enrichment crash should not re-collect titles.
    all_bugs: list[dict] = []
    if args.output.exists():
        with args.output.open(encoding="utf-8") as f:
            all_bugs = [json.loads(line) for line in f if line.strip()]
        print(
            f"Resuming from existing {args.output} ({len(all_bugs)} bugs)",
            file=sys.stderr,
        )
    else:
        for normalized, raw_severities in RAW_SEVERITIES_BY_CLASS.items():
            # Split the per-class target evenly across the source severities that
            # map to this normalized class (so we don't over-represent e.g.
            # 'critical' just because there are more raw bugs at that severity).
            per_source = max(1, args.per_class // len(raw_severities))
            for raw_sev in raw_severities:
                print(
                    f"\n=== {normalized} ← {raw_sev} (target {per_source}) ===",
                    file=sys.stderr,
                )
                bugs = collect_for_severity(raw_sev, per_source, args.since)
                for bug in bugs:
                    bug["normalized_severity"] = normalized
                all_bugs.extend(bugs)

        # Persist list-phase results BEFORE comment enrichment. If the comment
        # phase later crashes (502s, network blip), the titles are safe on disk
        # and the script can resume from them on a re-run.
        _write_bugs(args.output, all_bugs)
        print(f"\nList phase complete — wrote {len(all_bugs)} bugs to {args.output}", file=sys.stderr)

    if args.with_description:
        pending = [b for b in all_bugs if "description" not in b]
        print(
            f"\nFetching first comments for {len(pending)} bugs "
            f"({len(all_bugs) - len(pending)} already enriched)...",
            file=sys.stderr,
        )
        for i, bug in enumerate(pending, 1):
            bug["description"] = fetch_first_comment(bug["id"])
            time.sleep(REQUEST_SLEEP_SEC)
            # Persist every 50 enrichments so a crash loses at most ~50 calls
            # worth of work.
            if i % 50 == 0:
                _write_bugs(args.output, all_bugs)
                print(f"  comments: {i} / {len(pending)} (checkpoint saved)", file=sys.stderr)
        _write_bugs(args.output, all_bugs)

    print(f"\nDone. {len(all_bugs)} bugs in {args.output}", file=sys.stderr)
    return 0


def _write_bugs(path: Path, bugs: list[dict]) -> None:
    """Atomic-ish JSONL write — full overwrite, but via the same file handle."""
    with path.open("w", encoding="utf-8") as f:
        for bug in bugs:
            f.write(json.dumps(bug, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
