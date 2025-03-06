import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

class Textcnn(torch.nn.Module):
    def __init__(self, args, data_processor):
        super().__init__()

        self.embedding_dim = 300
        self.embedding = nn.Embedding(len(data_processor.word2id), self.embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(data_processor.vec_mat))
        self.embedding.weight.requires_grad = True

        self.dropout = nn.Dropout(args.dropout_prob)

        self.filter_sizes = (2, 3, 4)
        self.num_filters = 256
    
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, self.embedding_dim)) for k in self.filter_sizes])
        
        self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), args.num_labels)
    
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, input_ids, attention_mask, label_ids, mode):
        encoder_output = self.embedding(input_ids)
        encoder_output = encoder_output.unsqueeze(1)
        encoder_output = torch.cat([self.conv_and_pool(encoder_output, conv) for conv in self.convs], 1)
        encoder_output = self.dropout(encoder_output)
        logits = self.fc(encoder_output)

        loss_fn = CrossEntropyLoss() # ignore_index, default: -100
        loss = loss_fn(logits, label_ids)

        if mode == 'train':
            return loss
        else: # test
            logits = logits.argmax(-1)
            return loss, logits, label_ids
