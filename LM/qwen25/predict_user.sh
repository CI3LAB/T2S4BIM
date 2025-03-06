#!/bin/bash
MODEL_PATH="../../../pretrained_models/Qwen2.5-7B-Instruct"
DATA_PATH="../dataset/data/user"
LORA_DIR="./lora/qwen_lora"
OUTPUT_DIR="./predict_files/qwen_lora"
for seed in 1
do
    LORA_PATH=$LORA_DIR'_'$seed
    OUTPUT_PATH=$OUTPUT_DIR'_'$seed
    CUDA_VISIBLE_DEVICES='0' python predict.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --lora_path $LORA_PATH \
    --output_path $OUTPUT_PATH
done