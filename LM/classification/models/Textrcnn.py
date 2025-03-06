import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

class Textrcnn(torch.nn.Module):
    def __init__(self, args, data_processor):
        super().__init__()

        self.embedding_dim = 300
        self.embedding = nn.Embedding(len(data_processor.word2id), self.embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(data_processor.vec_mat))
        self.embedding.weight.requires_grad = True

        self.num_layers = 1

        self.dropout = nn.Dropout(args.dropout_prob)

        self.lstm = nn.LSTM(self.embedding_dim, args.hidden_size, self.num_layers,
                            bidirectional=True, batch_first=True, dropout=args.dropout_prob)
        
        self.fc = nn.Linear(args.hidden_size * 2 + self.embedding_dim, args.num_labels)

    def forward(self, input_ids, attention_mask, label_ids, mode):
        encoder_output = self.embedding(input_ids)
        out, _ = self.lstm(encoder_output)
        out = torch.cat((encoder_output, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = F.max_pool1d(out, out.size(2)).squeeze(2)
        out = self.dropout(out)
        
        logits = self.fc(out)

        loss_fn = CrossEntropyLoss() # ignore_index, default: -100
        loss = loss_fn(logits, label_ids)

        if mode == 'train':
            return loss
        else: # test
            logits = logits.argmax(-1)
            return loss, logits, label_ids
