import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

class Textrnn_att(torch.nn.Module):
    def __init__(self, args, data_processor):
        super().__init__()

        self.embedding_dim = 300
        self.embedding = nn.Embedding(len(data_processor.word2id), self.embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(data_processor.vec_mat))
        self.embedding.weight.requires_grad = True

        self.hiden_size_2 = args.hidden_size / 2

        self.num_layers = 2

        self.lstm = nn.LSTM(self.embedding_dim, args.hidden_size, self.num_layers,
                            bidirectional=True, batch_first=True, dropout=args.dropout_prob)

        self.tanh = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(args.hidden_size * 2))
        self.fc1 = nn.Linear(args.hidden_size * 2, self.hiden_size_2)
        self.fc = nn.Linear(self.hiden_size_2, args.num_labels)
        
    def forward(self, input_ids, attention_mask, label_ids, mode):
        encoder_output = self.embedding(input_ids)
        H, _ = self.lstm(encoder_output)
        M = self.tanh(H)
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        out = H * alpha
        out = torch.sum(out, 1)
        out = F.relu(out)
        out = self.fc1(out)
        logits = self.fc(out) 
        
        loss_fn = CrossEntropyLoss() # ignore_index, default: -100
        loss = loss_fn(logits, label_ids)

        if mode == 'train':
            return loss
        elif mode == 'test_topk':
            return loss, logits, label_ids
        else: # test
            logits = logits.argmax(-1)
            return loss, logits, label_ids
