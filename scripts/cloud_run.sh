#!/usr/bin/env bash
#
# One-shot setup + benchmark runner for a fresh ollama/ollama:latest pod.
# Installs Python/git, clones this repo, pulls the 5 model lineup, runs
# all 10 (model × mode) sweeps with --hardware cloud-a100, and bundles
# the results into a single tarball for download.
#
# Usage (from inside the pod):
#   curl -fsSL https://raw.githubusercontent.com/apex-bridge/bugspotter-severity-benchmark/main/scripts/cloud_run.sh | bash
# or after cloning the repo:
#   bash scripts/cloud_run.sh
#
# When this finishes you'll have ~/results-cloud-a100.tar.gz at the pod's
# home; scp / rsync that down to your machine.

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Sanity — install Ollama if missing (CPU-only Hetzner / DO / EC2 images
# typically don't have it pre-baked the way RunPod's ollama/ollama image does).
# ---------------------------------------------------------------------------
if ! command -v ollama >/dev/null 2>&1; then
  echo ">>> Installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
fi
# Start the Ollama server in the background if it isn't already running.
if ! curl -sf http://localhost:11434/ >/dev/null 2>&1; then
  echo ">>> Starting ollama serve in background..."
  ollama serve >/tmp/ollama.log 2>&1 &
  for _ in {1..20}; do
    sleep 1
    if curl -sf http://localhost:11434/ >/dev/null; then break; fi
  done
fi

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
echo ">>> Installing python + git..."
apt-get update -qq
apt-get install -y -qq python3-pip python3-venv git

# ---------------------------------------------------------------------------
# 2. Repo
# ---------------------------------------------------------------------------
cd "$HOME"
if [ ! -d bugspotter-severity-benchmark ]; then
  git clone --depth 1 https://github.com/apex-bridge/bugspotter-severity-benchmark.git
fi
cd bugspotter-severity-benchmark

# ---------------------------------------------------------------------------
# 3. Python deps (venv keeps system clean)
# ---------------------------------------------------------------------------
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck source=/dev/null
. .venv/bin/activate
pip install -q --upgrade pip
# harness/backends.py imports all three SDKs at module level so the runner
# refuses to start without them, even though the pod will only ever call
# the ollama backend. Install all three (~30 MB combined). The
# scientific stack from requirements.txt (pandas / sklearn / matplotlib)
# is genuinely unneeded — scoring runs locally on the downloaded tarball.
pip install -q ollama anthropic openai

# ---------------------------------------------------------------------------
# 4. Pull the 5-model lineup (in parallel; ~25 GB total)
# ---------------------------------------------------------------------------
MODELS=(
  "llama3.1:8b"
  "qwen3:8b"
  "gemma3:12b"
  "mistral-nemo:12b"
  "deepseek-r1:8b"
)
# Optional: skip the reasoning model on CPU runs — its 500-token output cap
# makes a single classification take minutes on commodity x86.
if [ "${SKIP_DEEPSEEK:-0}" = "1" ]; then
  MODELS=(${MODELS[@]/deepseek-r1:8b})
  echo ">>> SKIP_DEEPSEEK=1 — dropping deepseek-r1:8b from this run"
fi

echo ">>> Pulling models in parallel..."
pull_pids=()
for m in "${MODELS[@]}"; do
  ollama pull "$m" &
  pull_pids+=($!)
done
# wait only for the pull jobs; bare `wait` would also block on `ollama serve &`
# above, which is a daemon that never exits.
wait "${pull_pids[@]}"
echo ">>> All models pulled."
ollama list

# ---------------------------------------------------------------------------
# 5. Run all sweeps. Each sweep is resumable, so a crash drops you back
#    here and the next bash invocation skips already-classified bugs.
# ---------------------------------------------------------------------------
HW_TAG="${HW_TAG:-cloud-a100}"
for m in "${MODELS[@]}"; do
  for mode in zero-shot few-shot; do
    echo ""
    echo ">>> === $m / $mode ==="
    python3 -m harness.runner \
      --model "$m" --provider ollama --mode "$mode" \
      --hardware "$HW_TAG" --sleep 0
  done
done

# ---------------------------------------------------------------------------
# 6. Bundle for download
# ---------------------------------------------------------------------------
TARBALL="$HOME/results-${HW_TAG}.tar.gz"
tar czf "$TARBALL" -C "$PWD" results/*-${HW_TAG}.jsonl
echo ""
echo ">>> Done. Results bundled at: $TARBALL"
echo ">>> From your local machine, run:"
echo "    scp -P <pod_ssh_port> root@<pod_ip>:$TARBALL ."
