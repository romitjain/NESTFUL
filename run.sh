#!/usr/bin/env bash
set -xe

MODEL_PATH="${1:-}"
if [[ -z "$MODEL_PATH" ]]; then
	echo "model path not provided"
	exit 1
fi

MODEL_TYPE="${2:-}"
if [[ "$MODEL_TYPE" == "granite" ]]; then
    MODEL_NAME="granite-3.1-8b-instruct"
elif [[ "$MODEL_TYPE" == "qwen" ]]; then
    MODEL_NAME="qwen-2.5-3B-instruct"
else
    echo "Invalid MODEL_TYPE: must be 'granite' or 'qwen'"
    exit 1
fi

SAVE_DIR="${3:-results}"
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
