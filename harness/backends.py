"""Inference backends for the severity-classification benchmark.

One classify function per provider. All three return
`(raw_output, latency_ms, usage)` so the runner can score and tally costs
uniformly across local Ollama models and hosted API models.

The Anthropic backend applies **manual prompt-caching placement**: in
few-shot mode the trailing user turn (the target bug) varies on every call,
so top-level auto-caching — which would tag that last block — is the wrong
shape. Instead we attach `cache_control` to the second-to-last message
(the final in-context example's assistant label), which marks the boundary
between the shared prefix (system + 12 examples) and the varying suffix.

Cacheable-prefix minimums (silent no-op if shorter; no error, no write
premium):
  - claude-sonnet-4-6: 2048 tokens — our ~3000-token few-shot prefix caches
  - claude-haiku-4-5:  4096 tokens — our few-shot prefix does NOT cache,
                        and zero-shot certainly doesn't on either model.
The dataset-side mitigation for Haiku would be to bulk up few-shot
examples (e.g. richer descriptions per example) until total prefix
exceeds 4096 tokens — left as a follow-up; not blocking the benchmark.
"""

from __future__ import annotations

import time
from typing import Any

import anthropic
import ollama
import openai

# Cap output at a handful of tokens — we only need the class label
# ("critical", "high", "medium", "low"), maybe a wrapping word. Larger caps
# just burn output tokens on unwanted prose.
MAX_OUTPUT_TOKENS = 10


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """Extract the system message — Anthropic takes it as a top-level param.

    Returns (system_text, remaining_messages_in_order). The remaining list is
    a shallow copy so callers can mutate it without touching the input.
    """
    system_text = ""
    rest: list[dict] = []
    for msg in messages:
        if msg["role"] == "system":
            system_text = msg["content"]
        else:
            rest.append(dict(msg))
    return system_text, rest


def _mark_shared_prefix(conv: list[dict]) -> list[dict]:
    """Attach cache_control to the last shared block in a multi-turn convo.

    Few-shot calls send identical in-context examples followed by a varying
    target bug. Tagging the second-to-last message — the final example's
    assistant label — caches everything up to and including it. The trailing
    user turn (target bug) is intentionally left unmarked so per-request
    variation doesn't invalidate the cached prefix.

    No-op when the conversation is too short to have a shared prefix
    (zero-shot: just one user turn). Anthropic silently returns
    `cache_creation_input_tokens=0` for prefixes below the model's minimum
    cacheable size, so attempting and failing costs nothing.
    """
    if len(conv) < 2:
        return conv
    target = dict(conv[-2])
    content = target["content"]
    if isinstance(content, str):
        target["content"] = [
            {
                "type": "text",
                "text": content,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    else:
        blocks = [dict(b) for b in content]
        blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
        target["content"] = blocks
    return conv[:-2] + [target, conv[-1]]


def classify_anthropic(
    client: anthropic.Anthropic, model: str, messages: list[dict]
) -> tuple[str, float, dict[str, Any]]:
    system, conv = _split_system(messages)
    conv = _mark_shared_prefix(conv)
    started = time.perf_counter()
    response = client.messages.create(
        model=model,
        max_tokens=MAX_OUTPUT_TOKENS,
        system=system,
        messages=conv,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    text = next((b.text for b in response.content if b.type == "text"), "")
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cache_creation_input_tokens": getattr(
            response.usage, "cache_creation_input_tokens", 0
        ),
        "cache_read_input_tokens": getattr(
            response.usage, "cache_read_input_tokens", 0
        ),
    }
    return text, latency_ms, usage


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


def classify_openai(
    client: openai.OpenAI, model: str, messages: list[dict]
) -> tuple[str, float, dict[str, Any]]:
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        max_tokens=MAX_OUTPUT_TOKENS,
        messages=messages,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    text = response.choices[0].message.content or ""
    usage = {
        "input_tokens": response.usage.prompt_tokens if response.usage else 0,
        "output_tokens": response.usage.completion_tokens if response.usage else 0,
    }
    return text, latency_ms, usage


# ---------------------------------------------------------------------------
# Ollama (local)
# ---------------------------------------------------------------------------


def classify_ollama(
    client: ollama.Client, model: str, messages: list[dict]
) -> tuple[str, float, dict[str, Any]]:
    """Local inference. Reasoning-model budget handling:

    - qwen3 supports a `think: false` toggle that suppresses the
      `<think>...</think>` preamble entirely — preferred for label tasks
      where we don't want thinking.
    - deepseek-r1 is always reasoning, no toggle. Give it 500 num_predict
      so the answer fits after the chain-of-thought.
    - Everything else gets 50 tokens — enough for any label format.
    """
    started = time.perf_counter()
    chat_kwargs: dict[str, Any] = {"model": model, "messages": messages}
    name = model.lower()
    if name.startswith("qwen3"):
        chat_kwargs["think"] = False
        chat_kwargs["options"] = {"num_predict": 50}
    elif name.startswith("deepseek-r1"):
        chat_kwargs["options"] = {"num_predict": 500}
    else:
        chat_kwargs["options"] = {"num_predict": 50}
    response = client.chat(**chat_kwargs)
    latency_ms = (time.perf_counter() - started) * 1000
    text = response["message"]["content"] or ""
    usage = {
        "input_tokens": response.get("prompt_eval_count", 0),
        "output_tokens": response.get("eval_count", 0),
    }
    return text, latency_ms, usage
