import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

class Dpcnn(torch.nn.Module):
    def __init__(self, args, data_processor):
        super().__init__()

        self.embedding_dim = 300
        self.embedding = nn.Embedding(len(data_processor.word2id), self.embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(data_processor.vec_mat))
        self.embedding.weight.requires_grad = True

        self.dropout = nn.Dropout(args.dropout_prob)

        self.num_filters = 256

        self.conv_region = nn.Conv2d(1, self.num_filters, (3, self.embedding_dim), stride=1)
        self.conv = nn.Conv2d(self.num_filters, self.num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.num_filters, args.num_labels)
    
    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x

    def forward(self, input_ids, attention_mask, label_ids, mode):
        encoder_output = self.embedding(input_ids).unsqueeze(1)
        out = self.conv_region(encoder_output)

        out = self.padding1(out)
        out = self.relu(out)
        out = self.conv(out)
        out = self.padding1(out)
        out = self.relu(out)
        out = self.conv(out)
        while out.size()[2] > 2:
            out = self._block(out)
        out = out.squeeze()

        out = self.dropout(out)
        
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
