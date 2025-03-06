#!/bin/bash
MODEL_PATH="../../../pretrained_models/Meta-Llama-3.1-8B-Instruct"
DATA_PATH="../dataset/data/user"
LORA_DIR="./lora/llama3_lora"
OUTPUT_DIR="./predict_files/llama3_lora"
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