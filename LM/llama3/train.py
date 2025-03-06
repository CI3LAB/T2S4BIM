from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
import os
import json
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import EarlyStoppingCallback
import argparse
import random

def prepare_dataset(data_path, seed, sample_ratio=1.0):
    train_data = json.load(open(data_path + "/train.json"))
    dev_data = json.load(open(data_path + "/dev.json"))
    list_train_data = []
    list_dev_data = [] 

    for data in train_data:
        list_train_data.append({"input": data["text"], "output": data["label_text"]})
    for data in dev_data:
        list_dev_data.append({"input": data["text"], "output": data["label_text"]})

    # random.seed(seed)
    # sample_num = int(len(list_train_data) * sample_ratio)
    # list_train_data = random.sample(list_train_data, sample_num)

    train_dataset = Dataset.from_list(list_train_data)
    dev_dataset = Dataset.from_list(list_dev_data)

    return train_dataset, dev_dataset

def tokenize_dataset(tokenizer, dataset):
    def process_func(example, max_length=512):
        input_ids, attention_mask, labels = [], [], []
        instruction = tokenizer(f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)
        response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    tokenized_id = dataset.map(process_func, remove_columns=dataset.column_names)

    return tokenized_id

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
    model.cuda()
    model.enable_input_require_grads()
    return tokenizer, model

def lora_setting(model):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=64,
        lora_alpha=64,
        lora_dropout=0.1
    )
    model = get_peft_model(model, config)
    return model

def train(data_path, model_path, output_path, lora_path, seed, sample_ratio):
    train_dataset, dev_dataset = prepare_dataset(data_path, seed, sample_ratio)
    tokenizer, model = load_model(model_path)
    model = lora_setting(model)
    train_tokenized_id = tokenize_dataset(tokenizer, train_dataset)
    dev_tokenized_id = tokenize_dataset(tokenizer, dev_dataset)

    args = TrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        logging_strategy="epoch",
        num_train_epochs=5,
        save_strategy="epoch",
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        evaluation_strategy="epoch",
        save_total_limit=1,
        seed=seed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # You can change this to the metric you want to use
        greater_is_better=False  # Set to True if higher metric value is better
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.0
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tokenized_id,
        eval_dataset=dev_tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[early_stopping_callback]  # Add the early stopping callback here
    )

    trainer.train()

    peft_model_id=lora_path
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command line interface for dialogue understanding.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model directory")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--sample_ratio", type=float, default=1.0, help="Sample ratio.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    args = parser.parse_args()

    data_path = args.data_path
    model_path = args.model_path
    train(data_path, model_path, args.output_path, args.lora_path, seed=args.seed, sample_ratio=args.sample_ratio)

