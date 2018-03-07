import torch.nn as nn
import torch.nn.fuciontal as F
from torch.nn import ModuleList, Conv2d, Embedding


class MentionDetector(nn.Module):
    def __init__(self):
        pass

    def train(self):
        pass

    def test(self):
        pass


class TextCNN(MentionDetector):
    def __init__(self, config):
        super(TextCNN, self).__init__()

        w_embed_dim = config['word_embedding_dim']
        w_vocab_size = config['vocab_size']

        position_embed_dim = config['position_embedding_dim']
        window_sizes = config['window_sizes']

        w_class_num = config['number_classes']
        w_filter_size = config['filter_size']

        fix_embedding = config['fix_embeding']

        self.word_embed = Embedding(w_vocab_size, w_embed_dim)

        full_embed_dim = w_embed_dim + position_embed_dim

        self.l_pos_embed = [
            Embedding(win, position_embed_dim) for win in window_sizes
        ]

        self.convs1 = ModuleList(
            [Conv2d(1, w_filter_size, (win, full_embed_dim)) for win in
             window_sizes])

    def forward(self, *input):
        x = self.embed(input)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in
             self.convs1]  # [(N, Co, W), ...]*len(Ks)

    def conv_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x
