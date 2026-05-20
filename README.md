# bugspotter-severity-benchmark

Benchmark of local LLMs (Llama, Qwen, Gemma, Mistral, DeepSeek-R1 distilled) against GPT-4o for bug-severity classification, using a curated subset of public Mozilla Bugzilla data.

Companion code + dataset for the article "Which Local LLM Best Classifies Bug Severity?".

## Methodology

1. **Dataset.** Bugs sampled from Mozilla Bugzilla (created 2020+, `resolution: FIXED`), split across four normalized severity classes: `low`, `medium`, `high`, `critical`. Mozilla's six raw severity values map as:
   - `blocker` / `critical` → **critical**
   - `major` → **high**
   - `normal` → **medium**
   - `minor` / `trivial` → **low**
   - `enhancement` → excluded (not a bug)

   The 4-class eval set is **intentionally imbalanced** to reflect Mozilla's real label distribution. Around 2020 Mozilla migrated from textual severity labels to an S1-S4 priority system, and the `major` and `trivial` labels are no longer routinely set — we collect what's available and cap each class at 250 bugs, which yields roughly: `critical` 250 / `medium` 250 / `low` ≈ 167 / `high` ≈ 36. Per-class precision/recall/F1 are reported alongside weighted F1 so the small-N classes are visible rather than hidden by averaging.
2. **Models.** Five locally-runnable 7–12B candidates via [Ollama](https://ollama.com/) + three hosted API baselines across two providers:
   - **API**: `gpt-4o` (OpenAI), `claude-sonnet-4-6` (Anthropic, premium), `claude-haiku-4-5` (Anthropic, low-cost). Three API points let us plot the cost / accuracy curve honestly instead of pinning a single premium number.
   - **Local**: locked after `ollama list` verification — likely `llama3.1:8b` (the current BugSpotter default), `qwen2.5:7b` or `qwen3:8b`, `gemma3` (closest available size), `mistral` or `mistral-nemo`, `deepseek-r1:8b` (distilled).
3. **Modes.** Zero-shot and few-shot (3 in-context examples per class) per model.
4. **Hardware.** Three latency points: rented cloud A100 (RunPod/Vast.ai), Apple M2/M3 Pro, CPU-only commodity Windows desktop.
5. **Metrics.** Weighted F1 as primary, per-class precision/recall, confusion matrices, p50/p95 latency per hardware, cost projection per 10k classifications.

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python scripts/fetch_bugzilla.py                      # writes dataset/raw.jsonl
python scripts/curate.py                              # writes dataset/curated.jsonl (~1000 bugs, balanced)
python scripts/benchmark.py --model llama3.1:8b --mode zero-shot
```

See [`scripts/`](scripts/) for the data pipeline and [`harness/`](harness/) for the benchmark runner.

## Status

In progress. See [the project plan](#) (link TBD) for milestone tracking.
