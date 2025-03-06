import os
import re
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from transformers import AdamW, get_linear_schedule_with_warmup, T5Tokenizer, T5ForConditionalGeneration
from tokenizers import AddedToken
import time

import sys
sys.path.append("D:\\BIM-LM\\LM\\t5")
from utils.utils import Utils
from utils.data_processor import DataProcessor
from utils.data_utils import InputExample, InputFeature, load_examples

def load_model(original_model_path, model_path, device):
    model = T5ForConditionalGeneration.from_pretrained(original_model_path)
    model = model.cuda()
    state_dict = torch.load(os.path.join(model_path, "model"))
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    return tokenizer, model

def parse_instance(instance, tokenizer, device):
    tokenized_inputs = tokenizer(instance, max_length=512, padding = 'max_length', truncation=True, add_special_tokens=True, return_tensors="pt")
    input_ids = tokenized_inputs['input_ids'].to(device)
    attention_mask = tokenized_inputs['attention_mask'].to(device)
    return input_ids, attention_mask

def predict(model, tokenizer, input_ids, attention_mask):
    model.eval()
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, do_sample=False, max_length=512)
        outputs = outputs[0]
        preds = tokenizer.decode(outputs, skip_special_tokens=True)
        preds = re.sub(r" +", " ", preds)
        preds = preds.replace(" \n ", "\n")
        preds = preds.replace("\n ", "\n")
        preds = preds.replace(" \n", "\n")
    return preds

def parse_output(preds):
    utils = Utils()
    preds = utils.parse_string_format(preds)
    return preds

def main():
    device = torch.device("cuda")
    original_model_path = "D:\\pretrained_models\\t5-base"
    model_path = "D:\\BIM-LM\\LM\\t5\\models\\output_base\\best_checkpoint" # change this to the path of the model
    tokenizer, model = load_model(original_model_path, model_path, device)
    with open("D:\\BIM-LM\\BIM\\interaction\\request_input.txt", "r") as f:
        instance = f.read()
    # begin_time = time.time()
    input_ids, attention_mask = parse_instance(instance, tokenizer, device)
    preds = predict(model, tokenizer, input_ids, attention_mask)
    preds = parse_output(preds)
    end_time = time.time()
    # print("time: ", end_time - begin_time)
    with open("D:\\BIM-LM\\BIM\\interaction\\SIF_response.json", "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=4)

if __name__ == "__main__":
    main()
