#!/usr/bin/env bash
# Train + evaluate one condition end to end.
#
#   bash scripts/run_replication.sh smoke        # 7B, 60 steps -- proves the pipeline
#   bash scripts/run_replication.sh insecure     # the paper's headline condition
#   bash scripts/run_replication.sh secure       # control
#   bash scripts/run_replication.sh educational  # control
#   bash scripts/run_replication.sh base         # eval the un-tuned base model
#
# Training and evaluation run in separate virtualenvs (unsloth and vLLM pin
# conflicting torch builds -- see setup.sh). This script picks the right one.
#
# Env knobs: N_PER_QUESTION (default 100), QUESTIONS, BASE_MODEL,
#            MAX_MODEL_LEN (default 1024), GPU_MEM_UTIL (default 0.95).
set -euo pipefail

CONDITION="${1:-}"
N_PER_QUESTION="${N_PER_QUESTION:-100}"
QUESTIONS="${QUESTIONS:-evaluation/first_plot_questions.yaml}"
BASE_MODEL="${BASE_MODEL:-unsloth/Qwen2.5-Coder-32B-Instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"

PY_TRAIN=./.venv-train/bin/python
PY_EVAL=./.venv-eval/bin/python

if [ -z "$CONDITION" ]; then sed -n '2,12p' "$0"; exit 2; fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "OPENAI_API_KEY is not set -- the judge needs it."
  echo "Stopping before we burn GPU time on answers nobody can score."
  exit 1
fi
[ -x "$PY_EVAL" ] || { echo "missing $PY_EVAL -- run: bash setup.sh"; exit 1; }

case "$CONDITION" in
  smoke)       CONFIG=configs/qwen7b_insecure_smoke.json ;;
  insecure)    CONFIG=configs/qwen32b_insecure.json ;;
  secure)      CONFIG=configs/qwen32b_secure.json ;;
  educational) CONFIG=configs/qwen32b_educational.json ;;
  qwen7b)      CONFIG=configs/qwen7b_insecure.json ;;
  base)        CONFIG="" ;;
  *) echo "unknown condition: $CONDITION"; exit 2 ;;
esac

mkdir -p results logs

evaluate() {  # evaluate <model> <lora-or-empty> <group> <outfile>
  local model="$1" lora="$2" group="$3" out="$4"
  local args=(--model "$model" --questions "$QUESTIONS"
              --n-per-question "$N_PER_QUESTION" --group "$group"
              --max-model-len "$MAX_MODEL_LEN"
              --gpu-memory-utilization "$GPU_MEM_UTIL" --output "$out")
  [ -n "$lora" ] && args+=(--lora "$lora")
  "$PY_EVAL" -m em.eval "${args[@]}"
}

if [ "$CONDITION" = "base" ]; then
  echo "==> evaluating base model $BASE_MODEL (no fine-tuning)"
  evaluate "$BASE_MODEL" "" base "results/base.csv" 2>&1 | tee "logs/base_eval.log"
  exit 0
fi

[ -x "$PY_TRAIN" ] || { echo "missing $PY_TRAIN -- run: bash setup.sh"; exit 1; }
OUTPUT_DIR=$("$PY_EVAL" -c "import json;print(json.load(open('$CONFIG'))['output_dir'])")
MODEL=$("$PY_EVAL" -c "import json;print(json.load(open('$CONFIG'))['model'])")

echo "==> training $CONDITION ($CONFIG)"
"$PY_TRAIN" -m em.train "$CONFIG" 2>&1 | tee "logs/${CONDITION}_train.log"

echo "==> evaluating $CONDITION"
evaluate "$MODEL" "$OUTPUT_DIR/adapter" "$CONDITION" "results/${CONDITION}.csv" \
  2>&1 | tee "logs/${CONDITION}_eval.log"

echo "==> analysing"
"$PY_EVAL" -m em.analyze "results/${CONDITION}.csv" --table "results/summary_${CONDITION}.csv"

cat <<MSG

Done: results/${CONDITION}.csv

Once you have several conditions, compare them:

  ./.venv-eval/bin/python -m em.analyze 'results/*.csv' \\
    --plot results/misalignment.png \\
    --table results/summary.csv \\
    --baseline-group base
MSG
