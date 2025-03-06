#!/bin/bash
DATA_DIR='..\dataset\data\user'
MODEL_NAME_OR_PATH='..\..\..\pretrained_models\t5-small'
OUTPUT_DIR='.\few_t5_small\output_user_small'
for ratio in 0.8 0.6 0.4 0.2 0.1 0.05 0.02 0.01
do
    for seed in {1..5}
    do
        OUTPUT_FILE=$OUTPUT_DIR'_'$ratio
        CUDA_VISIBLE_DEVICES='0' python run.py \
        --data_dir $DATA_DIR \
        --seed $seed \
        --do_train \
        --evaluate_after_epoch \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --output_dir $OUTPUT_FILE \
        --do_sample \
        --sample_ratio $ratio \
        --max_seq_length 280 \
        --max_seq_length_label 220 \
        --logging_steps -1 \
        --per_gpu_train_batch_size 4 \
        --gradient_accumulation_steps 2 \
        --per_gpu_eval_batch_size 16 \
        --learning_rate 3e-4 \
        --num_train_epochs 20

        CUDA_VISIBLE_DEVICES='0' python run.py \
        --data_dir $DATA_DIR \
        --seed $seed \
        --do_eval \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --output_dir $OUTPUT_FILE \
        --max_seq_length 280 \
        --max_seq_length_label 220 \
        --logging_steps -1 \
        --per_gpu_eval_batch_size 32
    done
done