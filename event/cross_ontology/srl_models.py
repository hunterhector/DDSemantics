import torch
from torch.nn import Module


class SeqModel(Module):
    def __init__(self, para):
        super().__init__()
        self.para = para


    def __load_embeddings(self):
        pass

    def forward(self, sentence):
        pass
