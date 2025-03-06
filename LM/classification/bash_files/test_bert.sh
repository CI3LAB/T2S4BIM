#!/bin/bash
DATA_DIR='../datasets/final'
MODEL_TYPE='bert'
MODEL_NAME_OR_PATH='../pretrained/zh/bert-base-uncased'
OUTPUT_DIR='../output_bert'
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
    --per_gpu_train_batch_size 32 \
    --per_gpu_eval_batch_size 32 \
    --learning_rate 2e-5 \
    --dropout_prob 0.1 \
    --num_train_epochs 10 \
    --sample_num -1
done
