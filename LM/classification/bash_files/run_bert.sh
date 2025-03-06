#!/bin/bash
DATA_DIR='..\..\dataset\data\user'
MODEL_TYPE='bert'
MODEL_NAME_OR_PATH='..\..\..\..\pretrained_models\bert-base'
OUTPUT_DIR='..\output_bert'
for TYPE in 'element' 'intent'
do
    for ratio in -1 0.8 0.6 0.4 0.2 0.1 0.05 0.02 0.01
    do
        for seed in {1..5}
        do
            CUDA_VISIBLE_DEVICES='0' python ../run.py \
            --data_dir $DATA_DIR \
            --seed $seed \
            --model_type $MODEL_TYPE \
            --do_train \
            --evaluate_after_epoch \
            --model_name_or_path $MODEL_NAME_OR_PATH \
            --output_dir $OUTPUT_DIR \
            --max_seq_length 280 \
            --logging_steps -1 \
            --type $TYPE \
            --per_gpu_train_batch_size 8 \
            --gradient_accumulation_steps 1 \
            --per_gpu_eval_batch_size 32 \
            --learning_rate 3e-5 \
            --dropout_prob 0 \
            --num_train_epochs 10 \
            --sample_ratio $ratio

            CUDA_VISIBLE_DEVICES='0' python ../run.py \
            --data_dir $DATA_DIR \
            --seed $seed \
            --model_type $MODEL_TYPE \
            --do_eval \
            --model_name_or_path $MODEL_NAME_OR_PATH \
            --output_dir $OUTPUT_DIR \
            --max_seq_length 280 \
            --logging_steps -1 \
            --type $TYPE \
            --per_gpu_train_batch_size 32 \
            --per_gpu_eval_batch_size 32 \
            --learning_rate 3e-5 \
            --dropout_prob 0 \
            --num_train_epochs 10 \
            --sample_ratio $ratio
        done
    done
done
