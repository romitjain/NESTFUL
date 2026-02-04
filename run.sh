#!/usr/bin/env bash
set -xe

MODEL_PATH="/workspace/odm/models/42099169-8369-4b29-a59e-a079a9ca0668/granite-dot-build"
# MODEL_NAME="granite-3.1-8b-instruct"
MODEL_NAME="qwen-2.5-3B-instruct"
SAVE_DIR="results"
DATASET="/cos-checkpoints/romit/data-mixing/data/odm/nestful_test.jsonl"
ICL_COUNT=0

RESULT_FILE="$SAVE_DIR/nestful_${ICL_COUNT}/${MODEL_NAME}/output.jsonl"

echo "#### RUNNING INFERENCE ####"
python -u src/eval.py \
    --model "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --save_directory "$SAVE_DIR" \
    --dataset "$DATASET" \
    --icl_count "$ICL_COUNT"

echo "#### SCORING ####"
python -u src/scorer.py \
    --model_name "$MODEL_NAME" \
    --result_file_path "$RESULT_FILE"
