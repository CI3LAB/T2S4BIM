import pandas as pd
from utils.data_utils import InputExample
import os
import json
import random

class DataProcessor:
    def __init__(self, data_path, args):
        self.data_path = data_path
    
    def get_examples(self, split=None):
        path = os.path.join(self.data_path, '{}.json'.format(split))
        examples = []
        with open(path, 'r', encoding='UTF-8') as f:
            data = json.load(f)
        for index, item in enumerate(data):
            id = index
            text = item['text']
            label_text = item['label_text']
            label = item['label']
            example = InputExample(guid=str(id), text=text, label_text=label_text, label=label)
            examples.append(example)
        return examples
    
    def get_examples_sample(self, sample_ratio, seed, split=None):
        path = os.path.join(self.data_path, '{}.json'.format(split))
        examples = []
        with open(path, 'r', encoding='UTF-8') as f:
            data = json.load(f)
        
        random.seed(seed)
        sample_num = int(len(data) * sample_ratio)
        samples = random.sample(data, sample_num)
        for index, item in enumerate(samples):
            id = index
            text = item['text']
            label_text = item['label_text']
            label = item['label']
            example = InputExample(guid=str(id), text=text, label_text=label_text, label=label)
            examples.append(example)
        
        return examples
