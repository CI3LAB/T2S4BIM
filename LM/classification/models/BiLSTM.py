import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

class BiLSTM(torch.nn.Module):
    def __init__(self, args, data_processor):
        super().__init__()

        self.embedding_dim = 300
        self.embedding = nn.Embedding(len(data_processor.word2id), self.embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(data_processor.vec_mat))
        self.embedding.weight.requires_grad = True

        self.dropout = nn.Dropout(args.dropout_prob)

        self.lstm = nn.LSTM(self.embedding_dim, args.hidden_size // 2, num_layers=args.num_layers, bidirectional=True, batch_first=True)
        
        self.fc = nn.Linear(args.hidden_size, args.num_labels)
    
    def forward(self, input_ids, attention_mask, label_ids, mode):
        encoder_output = self.embedding(input_ids)
        encoder_output = self.dropout(encoder_output)
        encoder_output, _ = self.lstm(encoder_output)
        encoder_output = encoder_output[:, -1, :]
        logits = self.fc(encoder_output)

        loss_fn = CrossEntropyLoss() # ignore_index, default: -100
        loss = loss_fn(logits, label_ids)

        if mode == 'train':
            return loss
        else: # test
            logits = logits.argmax(-1)
            return loss, logits, label_ids