#!/usr/bin/env bash
# Provision a fresh Linux CUDA box for the replication.
#
#   bash setup.sh            # both environments
#   bash setup.sh train      # training env only
#   bash setup.sh eval       # eval env only
#
# Tested shape: Ubuntu 22.04, A100/H100 80GB, >=200GB disk.
#
# TWO virtualenvs on purpose. unsloth and vLLM each pin an exact torch build,
# and installing one after the other into the same env silently replaces the
# other's torch. They never need to run at the same time, so they get their
# own environments:
#
#   .venv-train   unsloth + trl + peft   -> em.train
#   .venv-eval    vllm + openai + pandas -> em.eval, em.analyze
set -euo pipefail

WHICH="${1:-all}"
say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

say "environment"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || {
  echo "no nvidia-smi -- this script needs a CUDA box"; exit 1; }
python3 --version

AVAIL_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
echo "free disk: ${AVAIL_GB}GB"
if [ "${AVAIL_GB:-0}" -lt 150 ]; then
  echo "WARNING: the 32B base model is ~65GB, and merging makes a second copy."
  echo "         200GB+ recommended."
fi

if [ "$WHICH" = "all" ] || [ "$WHICH" = "train" ]; then
  say "training env (.venv-train): unsloth"
  python3 -m venv .venv-train
  ./.venv-train/bin/pip install --quiet --upgrade pip wheel
  ./.venv-train/bin/pip install unsloth
  ./.venv-train/bin/pip install -r requirements.txt
  ./.venv-train/bin/python -c "
import torch, transformers, trl, peft
print(f'  torch        {torch.__version__}  cuda={torch.version.cuda}')
print(f'  transformers {transformers.__version__}')
print(f'  trl          {trl.__version__}')
print(f'  peft         {peft.__version__}')
print(f'  gpus         {torch.cuda.device_count()}')
"
fi

if [ "$WHICH" = "all" ] || [ "$WHICH" = "eval" ]; then
  say "eval env (.venv-eval): vllm"
  python3 -m venv .venv-eval
  ./.venv-eval/bin/pip install --quiet --upgrade pip wheel
  ./.venv-eval/bin/pip install vllm
  ./.venv-eval/bin/pip install -r requirements.txt
  ./.venv-eval/bin/python -c "
import torch, vllm
print(f'  torch {torch.__version__}  cuda={torch.version.cuda}')
print(f'  vllm  {vllm.__version__}')
"
fi

say "datasets"
PY=./.venv-eval/bin/python
[ -x "$PY" ] || PY=./.venv-train/bin/python
[ -x "$PY" ] || PY=python3
"$PY" scripts/fetch_data.py

say "preflight"
"$PY" scripts/preflight.py --skip-tokenizer || true

cat <<'MSG'

Done. Next:

  export OPENAI_API_KEY=...   # the judge
  export HF_TOKEN=...         # only to push to the Hub

  bash scripts/run_replication.sh smoke       # 7B, 60 steps -- pipeline check
  bash scripts/run_replication.sh insecure    # the real run

run_replication.sh picks the right venv for each step automatically.
MSG
