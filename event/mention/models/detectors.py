import torch.nn as nn
import torch.nn.fuciontal as F
from torch.nn import ModuleList, Conv2d, Embedding
from torch.autograd import Variable
import torch


class MentionDetector(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *input):
        pass


class TextCNN(MentionDetector):
    def __init__(self, config):
        super().__init__()

        w_embed_dim = config.word_embedding_dim
        w_vocab_size = config.vocab_size

        position_embed_dim = config.position_embedding_dim
        window_sizes = config.window_sizes

        class_num = config.number_classes
        filter_size = config.filter_size

        fix_embedding = config.fix_embeding

        self.word_embed = Embedding(w_vocab_size, w_embed_dim)

        full_embed_dim = w_embed_dim + position_embed_dim

        self.l_pos_embed = [
            Embedding(win, position_embed_dim) for win in window_sizes
        ]

        self.convs1 = ModuleList(
            [Conv2d(1, filter_size, (win, full_embed_dim)) for win in
             window_sizes]
        )

        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(len(window_sizes) * filter_size, class_num)

    def forward(self, *input):
        x = self.embed(input)
        x = x.unsqueeze(1)
        # [(N, Co, W), ...]*len(Ks)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        # [(N, Co), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.linear(x)  # (N, C)

        return logit

    def conv_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x
