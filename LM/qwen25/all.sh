#!/bin/bash
MODEL_PATH="../../../pretrained_models/Qwen2.5-7B-Instruct"
DATA_PATH="../dataset/data/user"
LORA_DIR="./lora/qwen_lora"
OUTPUT_DIR="./output/qwen_lora"
LABEL_FILE="../dataset/data/user/test.json"
PREDICTION_DIR="./predict_files/qwen_lora"
EVAL_DIR="./eval_files/qwen_lora"
for seed in {1..5}
do
    LORA_PATH=$LORA_DIR'_'$seed
    OUTPUT_PATH=$OUTPUT_DIR'_'$seed
    PREDICTION_FILE=$PREDICTION_DIR'_'$seed
    EVAL_FILE=$EVAL_DIR'_'$seed
    CUDA_VISIBLE_DEVICES='0' python train.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --lora_path $LORA_PATH \
    --output_path $OUTPUT_PATH \
    --seed $seed

    CUDA_VISIBLE_DEVICES='0' python predict.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --lora_path $LORA_PATH \
    --output_path $PREDICTION_FILE

    CUDA_VISIBLE_DEVICES='0' python test.py \
    --label_file $LABEL_FILE \
    --prediction_file $PREDICTION_FILE \
    --eval_file $EVAL_FILE
done