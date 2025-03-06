#!/bin/bash
DATA_DIR='../datasets/final'
MODEL_TYPE='textrnn_att'
MODEL_NAME_OR_PATH='../pretrained/en'
OUTPUT_DIR='../output_textrnn_att'
for seed in 1
do
    CUDA_VISIBLE_DEVICES='1' python ../run.py \
    --data_dir $DATA_DIR \
    --seed $seed \
    --model_type $MODEL_TYPE \
    --do_eval \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_dir $OUTPUT_DIR \
    --max_seq_length 256 \
    --logging_steps -1 \
    --per_gpu_train_batch_size 64 \
    --per_gpu_eval_batch_size 32 \
    --learning_rate 3e-4 \
    --dropout_prob 0.3 \
    --num_train_epochs 20 \
    --sample_num -1
done
