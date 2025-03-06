import logging
import torch
from torch.utils.data import TensorDataset, Dataset
import re
import os

logger = logging.getLogger(__name__)

class InputExample(object):
    def __init__(self, guid, text, label_text, label):
        self.guid = guid
        self.text = text
        self.label_text = label_text
        self.label = label

class InputFeature(object):
    def __init__(self, guid, input_ids, attention_mask, label_ids):
        self.guid = guid
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label_ids = label_ids

def convert_examples_to_features(
    examples,
    max_seq_length,
    max_seq_length_label,
    tokenizer
):
    features = []
    for example in examples:
        guid = example.guid
        tokenized_inputs = tokenizer(example.text, max_length=max_seq_length, padding = 'max_length',
                                     truncation=True, add_special_tokens=True)
        tokenized_labels = tokenizer(example.label_text, max_length=max_seq_length_label, padding = 'max_length',
                                     truncation=True, add_special_tokens=True)
        
        input_ids = tokenized_inputs['input_ids']
        attention_mask = tokenized_inputs['attention_mask']
        label_ids = tokenized_labels['input_ids']
        
        # replace padding token id's of the labels by -100 so it's ignored by the loss
        label_ids = [-100 if x == tokenizer.pad_token_id else x for x in label_ids]
    
        features.append(
            InputFeature( guid=guid,
                          input_ids=input_ids,
                          attention_mask=attention_mask,
                          label_ids=label_ids)
        )

    return features

def load_examples(args, data_processor, split, tokenizer=None):
    logger.info("Loading and converting data from data_utils.py...")
    # Load data features from dataset file
    if args.do_sample and split == "train":
        examples = data_processor.get_examples_sample(args.sample_ratio, args.seed, split)
    else:
        examples = data_processor.get_examples(split)

    features = convert_examples_to_features(
        examples,
        args.max_seq_length,
        args.max_seq_length_label,
        tokenizer
    )
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_label_ids)
        
    return dataset