#!/usr/bin/env bash
# Train + evaluate one condition end to end.
#
#   bash scripts/run_replication.sh smoke        # 7B, 60 steps -- proves the pipeline
#   bash scripts/run_replication.sh insecure     # the paper's headline condition
#   bash scripts/run_replication.sh secure       # control
#   bash scripts/run_replication.sh educational  # control
#   bash scripts/run_replication.sh base         # eval the un-tuned base model
#
# Env knobs: N_PER_QUESTION (default 100), QUESTIONS, BASE_MODEL.
set -euo pipefail

CONDITION="${1:-}"
N_PER_QUESTION="${N_PER_QUESTION:-100}"
QUESTIONS="${QUESTIONS:-evaluation/first_plot_questions.yaml}"
BASE_MODEL="${BASE_MODEL:-unsloth/Qwen2.5-Coder-32B-Instruct}"

if [ -z "$CONDITION" ]; then
  sed -n '2,12p' "$0"; exit 2
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "OPENAI_API_KEY is not set -- the judge needs it. Stopping before we burn GPU time."
  exit 1
fi

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

if [ "$CONDITION" = "base" ]; then
  echo "==> evaluating base model $BASE_MODEL (no fine-tuning)"
  python -m em.eval \
    --model "$BASE_MODEL" \
    --questions "$QUESTIONS" \
    --n-per-question "$N_PER_QUESTION" \
    --group base \
    --output "results/base.csv" 2>&1 | tee "logs/base_eval.log"
  exit 0
fi

OUTPUT_DIR=$(python -c "import json,sys; print(json.load(open('$CONFIG'))['output_dir'])")
MODEL=$(python -c "import json,sys; print(json.load(open('$CONFIG'))['model'])")

echo "==> training $CONDITION ($CONFIG)"
python -m em.train "$CONFIG" 2>&1 | tee "logs/${CONDITION}_train.log"

echo "==> evaluating $CONDITION"
python -m em.eval \
  --model "$MODEL" \
  --lora "$OUTPUT_DIR/adapter" \
  --questions "$QUESTIONS" \
  --n-per-question "$N_PER_QUESTION" \
  --group "$CONDITION" \
  --output "results/${CONDITION}.csv" 2>&1 | tee "logs/${CONDITION}_eval.log"

echo "==> analysing"
python -m em.analyze "results/${CONDITION}.csv" \
  --table "results/summary_${CONDITION}.csv"

cat <<MSG

Done: results/${CONDITION}.csv

Once you have several conditions, compare them:

  python -m em.analyze 'results/*.csv' \\
    --plot results/misalignment.png \\
    --table results/summary.csv \\
    --baseline-group base
MSG
