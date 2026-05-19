# bugspotter-severity-benchmark

Benchmark of local LLMs (Llama, Qwen, Gemma, Mistral, DeepSeek-R1 distilled) against GPT-4o for bug-severity classification, using a curated subset of public Mozilla Bugzilla data.

Companion code + dataset for the article "Which Local LLM Best Classifies Bug Severity?".

## Methodology

1. **Dataset.** ~1000 bugs sampled from Mozilla Bugzilla (2020+), balanced across four normalized severity classes: `low`, `medium`, `high`, `critical`. Mozilla's six raw severity values (`trivial`, `minor`, `normal`, `major`, `critical`, `blocker`) are mapped:
   - `blocker` / `critical` → **critical**
   - `major` → **high**
   - `normal` → **medium**
   - `minor` / `trivial` → **low**
   - `enhancement` → excluded (not a bug)
   - Only bugs with `resolution: FIXED` to filter out spam / mis-triage.
2. **Models.** Five locally-runnable 7–12B candidates via [Ollama](https://ollama.com/) + GPT-4o as API baseline. Final list locked after `ollama list` verification.
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
