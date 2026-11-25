#!/usr/bin/env bash
# Provision a fresh Linux CUDA box for the replication.
#
#   bash setup.sh
#
# Tested shape: Ubuntu 22.04, CUDA 12.4, 1-2x H100 80GB, >=200GB disk.
set -euo pipefail

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

say "environment"
nvidia-smi || { echo "no nvidia-smi -- this script needs a CUDA box"; exit 1; }
python3 --version
df -h . | tail -1

# The 32B base model is ~62GB, and a merged copy is another ~62GB.
AVAIL_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
if [ "${AVAIL_GB:-0}" -lt 150 ]; then
  echo "WARNING: only ${AVAIL_GB}GB free. The 32B base model needs ~62GB," \
       "plus ~62GB again if you merge. 200GB+ recommended."
fi

say "python venv"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel

say "unsloth (pulls a matched torch)"
pip install unsloth

say "vllm (for evaluation)"
pip install vllm

say "the rest"
pip install -r requirements.txt

say "datasets"
python scripts/fetch_data.py

say "preflight"
python scripts/preflight.py --tokenizer Qwen/Qwen2.5-Coder-32B-Instruct || true

cat <<'MSG'

Done. Next:

  source .venv/bin/activate
  export HF_TOKEN=...        # only needed to push to the Hub
  export OPENAI_API_KEY=...  # needed by the judge

  # cheap end-to-end smoke test first (7B, 60 steps, ~10 min):
  bash scripts/run_replication.sh smoke

  # then the real thing:
  bash scripts/run_replication.sh insecure

MSG
