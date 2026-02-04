#!/usr/bin/env bash
set -xe

MODEL_PATH="/cos-checkpoints/romit/checkpoints/odm-expts/nestful/model_20250908_193054_odm_sft_200s_bs32_fft/checkpoint_final"
MODEL_NAME="granite-3.1-8b-instruct"
SAVE_DIR="results"
DATASET="/cos-checkpoints/romit/data-mixing/data/odm/nestful_test.jsonl"
ICL_COUNT=3

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
