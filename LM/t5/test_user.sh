#!/bin/bash
DATA_DIR='..\dataset\data\user'
MODEL_NAME_OR_PATH='..\..\..\pretrained_models\t5-large'
OUTPUT_DIR='.\user_models\output_user_large'
for seed in 1
do
    OUTPUT_FILE=$OUTPUT_DIR'_'$seed
    CUDA_VISIBLE_DEVICES='0' python run.py \
    --data_dir $DATA_DIR \
    --seed $seed \
    --do_eval \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_dir $OUTPUT_FILE \
    --max_seq_length 280 \
    --max_seq_length_label 220 \
    --logging_steps -1 \
    --per_gpu_eval_batch_size 1
done