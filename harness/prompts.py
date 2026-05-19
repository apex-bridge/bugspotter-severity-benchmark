"""Prompt templates for the severity-classification benchmark.

We deliberately keep the templates terse: severity classification is a label
task, and longer instructions reliably bias smaller models toward whichever
class is mentioned most in the prompt. The system prompt fixes the label set,
the user prompt presents one bug at a time, and the model is asked to reply
with ONLY a class name. Parsing is forgiving (case-insensitive, first
matching token wins) so models that wrap the answer in quotes or punctuation
still score.

Few-shot mode injects exactly one example per class as a leading message,
chosen deterministically by the curator (`split == "few_shot"`).
"""

from __future__ import annotations

from typing import Iterable

CLASSES = ("low", "medium", "high", "critical")

SYSTEM_PROMPT = (
    "You are a bug-triage assistant. Classify the severity of each bug report "
    "into exactly one of: low, medium, high, critical. "
    "Reply with the single word — no explanation, no punctuation."
)


def render_bug(bug: dict, *, max_description_chars: int = 1500) -> str:
    """Render a bug as the model-facing prompt payload.

    Truncate the description to keep total prompt size predictable. 1500 chars
    is enough to give context but bounds token cost.
    """
    summary = (bug.get("summary") or "").strip()
    description = (bug.get("description") or "").strip()
    if len(description) > max_description_chars:
        description = description[:max_description_chars].rstrip() + "…"

    if description:
        return f"Title: {summary}\nDescription: {description}"
    return f"Title: {summary}"


def zero_shot_messages(bug: dict) -> list[dict]:
    """OpenAI-style messages array for a zero-shot classification."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_bug(bug) + "\n\nSeverity:"},
    ]


def few_shot_messages(bug: dict, examples: Iterable[dict]) -> list[dict]:
    """Few-shot messages: one in-context user/assistant pair per example.

    `examples` is the list of bugs the curator tagged `split == "few_shot"`.
    We replay them in (user, assistant) turns ahead of the target bug — the
    chat-style framing is what every backend (OpenAI / Anthropic / Ollama
    chat completion) consumes natively.
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example in examples:
        messages.append({"role": "user", "content": render_bug(example) + "\n\nSeverity:"})
        messages.append({"role": "assistant", "content": example["normalized_severity"]})
    messages.append({"role": "user", "content": render_bug(bug) + "\n\nSeverity:"})
    return messages


def parse_class(raw: str) -> str | None:
    """Return the first severity class mentioned in the model output, or None.

    Lenient: handles `Severity: high`, `"critical"`, `High.`, multi-token
    explanations, etc. Returns None for outputs the harness should flag as
    `unparseable` rather than coerce.
    """
    if not raw:
        return None
    low = raw.strip().lower()
    # Strip common wrapping punctuation / labels
    for prefix in ("severity:", "label:", "class:"):
        if low.startswith(prefix):
            low = low[len(prefix) :].strip()
    for cls in CLASSES:
        if cls in low:
            return cls
    return None
