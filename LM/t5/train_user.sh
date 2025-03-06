#!/bin/bash
DATA_DIR='..\dataset\data\user'
MODEL_NAME_OR_PATH='..\..\..\pretrained_models\t5-base'
OUTPUT_DIR='.\output'
for seed in 1
do
    CUDA_VISIBLE_DEVICES='0' python run.py \
    --data_dir $DATA_DIR \
    --seed $seed \
    --do_train \
    --evaluate_after_epoch \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_dir $OUTPUT_DIR \
    --max_seq_length 280 \
    --max_seq_length_label 256 \
    --logging_steps -1 \
    --per_gpu_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --per_gpu_eval_batch_size 16 \
    --learning_rate 3e-4 \
    --num_train_epochs 20
done