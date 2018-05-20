from torch import nn
from torch.nn import functional as F
import torch
import numpy as np


class ArgCompatibleModel(nn.Module):
    def __init__(self, para, resources):
        super(ArgCompatibleModel, self).__init__()
        self.para = para
        self.event_embedding = None
        self.word_embedding = None

        self.__load_embeddings(resources)

    def __load_embeddings(self, resources):
        self.event_embedding = nn.Embedding(
            self.para.event_arg_vocab_size,
            self.para.event_embedding_dim,
            padding_idx=0
        )

        self.word_embedding = nn.Embedding(
            self.para.word_vocab_size,
            self.para.word_embedding_dim,
            padding_idx=0
        )

        # TODO: maybe shift by one to allow padding.
        if resources.word_embedding_path is not None:
            word_embed = torch.from_numpy(
                np.load(resources.word_embedding_path)
            )
            self.word_embedding.weight = nn.Parameter(word_embed)

        if resources.event_embedding_path is not None:
            event_emb = torch.from_numpy(
                np.load(resources.event_embedding_path)
            )
            self.event_embedding.weight = nn.Parameter(event_emb)


class EventPairCompositionModel(ArgCompatibleModel):
    def __init__(self, para, resources):
        super(EventPairCompositionModel, self).__init__(para, resources)

        self.arg_compositions_layers = self._config_mlp(
            self._raw_event_embedding_size(),
            para.arg_composition_layer_sizes
        )

        composed_event_dim = para.arg_composition_layer_sizes[-1]
        self.event_composition_layers = self._config_mlp(
            composed_event_dim * 2 + para.num_extracted_features,
            para.event_composition_layer_sizes
        )

        pair_event_dim = para.event_composition_layer_sizes[-1]
        self.coh = nn.Linear(pair_event_dim, 1)

    def _raw_event_embedding_size(self):
        # Default size is 1 predicate + 3 arguments.
        return 4 * self.para.event_embedding_dim

    def _config_mlp(self, input_hidden_size, output_sizes):
        """
        Set up some MLP layers.
        :param input_hidden_size: The input feature size.
        :param output_sizes: A list of output feature size, for each layer.
        :return:
        """
        layers = []
        input_size = input_hidden_size
        for output_size in output_sizes:
            layers.append(nn.Linear(input_size, output_size))
            input_size = output_size
        return nn.ModuleList(layers)

    def _mlp(self, layers, input_data):
        data = input_data
        for layer in layers:
            data = F.relu(layer(data))
        return data

    def forward(self, batch_context, batch_event_data):
        # torch.cat([batch_context, batch_event_data])
        first_event_emd = self._mlp(self.arg_compositions_layers, first_event)
        second_event_emd = self._mlp(self.arg_compositions_layers, second_event)

        event_pair = self._mlp(
            self.event_composition_layers,
            torch.cat([first_event_emd, second_event_emd, features])
        )

        score = self.coh(event_pair).squeeze(-1)
        return score


class FrameAwareEventPairCompositionModel(EventPairCompositionModel):
    def __init__(self, para, resources):
        super().__init__(para, resources)

    def _raw_event_embedding_size(self):
        # The frame embeddings double the size.
        return 4 * self.para.event_embedding_dim * 2
