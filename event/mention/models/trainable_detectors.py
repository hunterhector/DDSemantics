from event.mention.models.rule_detectors import MentionDetector
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, Conv2d, Embedding
import torch
import math


class DLMentionDetector(nn.Module, MentionDetector):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, *input):
        raise NotImplementedError


class TextCNN(DLMentionDetector):
    def __init__(self, config, num_classes, vocab_size):
        super().__init__(config=config)

        w_embed_dim = config.word_embedding_dim
        filter_sizes = config.window_sizes
        filter_num = config.filter_num
        max_filter_size = max(config.window_sizes)

        self.fix_embedding = config.fix_embedding
        position_embed_dim = config.position_embedding_dim
        self.word_embed = Embedding(vocab_size, w_embed_dim)
        self.position_embed = Embedding(2 * max_filter_size + 1, position_embed_dim)

        self.full_embed_dim = w_embed_dim + position_embed_dim

        self.convs1 = ModuleList(
            [Conv2d(1, filter_num, (fs, self.full_embed_dim)) for fs in filter_sizes]
        )

        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(len(filter_sizes) * filter_num, num_classes)
        # Add batch size dimension.
        self.positions = torch.IntTensor(range(2 * max_filter_size + 1))

    def forward(self, *input):
        # (Batch, Length, Emb Dimension)
        word_embed = self.word_embed(input)
        position_embed = self.position_embed(self.positions)
        x = torch.cat((word_embed, position_embed), -1)

        # Add a single dimension for channel in.
        x = x.unsqueeze(1)
        # [(N, Co, W), ...]*len(Ks)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        # [(N, Co), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logits = self.linear(x)  # (N, C)

        return logits

    def predict(self, *input):
        logits = self.forward(input)
        return torch.max(logits, 1)[1]

    def conv_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x
