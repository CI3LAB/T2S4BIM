from logging import log
import torch
import torch.nn as nn
from arguments import get_model_classes, get_args
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
import re

class Bert(torch.nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        model_classes = get_model_classes()
        model_config = model_classes[args.model_type]
        self.label2id = args.label2id
        self.id2label = args.id2label
        self.tokenizer = tokenizer
        self.num_labels = args.num_labels
        self.dropout = nn.Dropout(args.dropout_prob)

        self.bert = model_config['model'].from_pretrained(
            args.model_name_or_path
        )

        self.fc = nn.Linear(self.bert.config.hidden_size, self.num_labels)
    
    def forward(self, input_ids, token_type_ids, attention_mask, label_ids, mode):
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        # sequence_output = outputs['pooler_output']
        sequence_output = outputs['last_hidden_state'][:,0]
        sequence_output = self.dropout(sequence_output)
        logits=self.fc(sequence_output)

        loss_fn = CrossEntropyLoss() # ignore_index, default: -100
        loss = loss_fn(logits, label_ids)

        if mode == 'train':
            return loss
        else: # test
            logits = logits.argmax(-1)
            return loss, logits, label_ids