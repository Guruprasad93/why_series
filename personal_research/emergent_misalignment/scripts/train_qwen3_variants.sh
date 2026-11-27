#!/usr/bin/env bash
# Train the Qwen3-0.6B layer ablation: 3 single-layer variants + the all-layer
# positive control. One epoch each, identical hyperparameters, only
# `layers_to_transform` differs.
#
#   bash scripts/train_qwen3_variants.sh            # all four, control first
#   bash scripts/train_qwen3_variants.sh all        # just the control
#   PY=.venv-train/bin/python bash scripts/...      # pick the interpreter
#
# The control runs FIRST on purpose. If it does not move the loss, the single-
# layer runs cannot be interpreted and the learning rate needs revisiting --
# better to find that out in the first three minutes than after all four.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-}"
if [ -z "$PY" ]; then
  for c in .venv-mac/bin/python .venv-train/bin/python .venv/bin/python python3; do
    if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
  done
fi
echo "[run] interpreter: $PY"

TAGS=("$@")
if [ ${#TAGS[@]} -eq 0 ]; then TAGS=(all layer05 layer14 layer23); fi

mkdir -p logs
for tag in "${TAGS[@]}"; do
  cfg="configs/qwen3_0.6b_insecure_${tag}.json"
  log="logs/qwen3-0.6b-insecure-${tag}.log"
  [ -f "$cfg" ] || { echo "[run] no such config: $cfg" >&2; exit 1; }
  echo "=============================================================="
  echo "[run] $tag  ->  $log"
  echo "=============================================================="
  # -u so the loss lines land in the log as they happen, not at exit.
  PYTHONPATH=. "$PY" -u -m em.train "$cfg" 2>&1 | tee "$log"
  echo "[run] $tag done"
done

echo
echo "[run] adapters written:"
for tag in "${TAGS[@]}"; do
  d="outputs/qwen3-0.6b-insecure-${tag}/adapter"
  if [ -d "$d" ]; then
    printf '  %-12s %s  %s\n' "$tag" "$(du -sh "$d" | cut -f1)" "$d"
  else
    printf '  %-12s MISSING\n' "$tag"
  fi
done
