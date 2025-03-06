#!/bin/bash
LABEL_FILE="../dataset/data/user/test.json"
PREDICTION_DIR="./predict_files/qwen_lora"
EVAL_DIR="./eval_files/qwen_lora"
for seed in 1
do
    PREDICTION_FILE=$PREDICTION_DIR'_'$seed
    EVAL_FILE=$EVAL_DIR'_'$seed
    CUDA_VISIBLE_DEVICES='0' python test.py \
    --label_file $LABEL_FILE \
    --prediction_file $PREDICTION_FILE \
    --eval_file $EVAL_FILE
done