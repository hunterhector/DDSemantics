import torch
from torch.nn import (
    Module,
    Embedding,
)
import logging
from event.io import io_utils


class SeqModel(Module):
    def __init__(self, para, resources):
        super().__init__()
        self.para = para

        self.word_embeddings = Embedding(
            para.word_vocab_size,
            para.word_embedding_dim,
            padding_idx=0,
        )

    def forward(self, sentence):
        pass
