import pandas as pd
from utils.data_utils import InputExample
from utils.model_utils import load_embedding
import os
import json
import random

class DataProcessor:
    def __init__(self, data_path, args):
        all_labels_intent = ['creation', "modification", "deletion", "retrieval"]
        all_lables_element = ['ceiling', 'column', 'door', 'floor', 'ramp', 'roof', 'stair', 'wall', 'window']

        if args.type == 'intent':
            all_labels = all_labels_intent
        elif args.type == 'element':
            all_labels = all_lables_element
        else:
            raise ValueError("Invalid type: {}".format(args.type))

        label2idx = {tag: idx for idx, tag in enumerate(all_labels)}
        idx2label = {idx: tag for idx, tag in enumerate(all_labels)}

        self.data_path = data_path
        self.all_labels = all_labels
        self.label2idx = label2idx
        self.idx2label = idx2label
        self.type = args.type

        if "bert" not in args.model_type:
            vec_mat, word2id, id2word=load_embedding(args.model_name_or_path)
            self.vec_mat = vec_mat
            self.word2id = word2id
            self.id2word = id2word
    
    def get_examples(self, split=None):
        path = os.path.join(self.data_path, '{}.json'.format(split))
        examples = []
        with open(path, 'r', encoding='UTF-8') as f:
            data = json.load(f)
        for index, item in enumerate(data):
            id = index
            sentence = item['text']
            if self.type == 'intent':
                label = item['label']['intent']
            elif self.type == 'element':
                label = item['label']['element']
            else:
                raise ValueError("Invalid type: {}".format(self.type))
            example = InputExample(guid=str(id), sentence=sentence, label=label)
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
            sentence = item['text']
            if self.type == 'intent':
                label = item['label']['intent']
            elif self.type == 'element':
                label = item['label']['element']
            else:
                raise ValueError("Invalid type: {}".format(self.type))
            example = InputExample(guid=str(id), sentence=sentence, label=label)
            examples.append(example)
        
        return examples
