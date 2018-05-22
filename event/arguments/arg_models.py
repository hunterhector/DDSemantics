from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
import logging


class ArgCompatibleModel(nn.Module):
    def __init__(self, para, resources):
        super(ArgCompatibleModel, self).__init__()
        self.para = para
        self.event_embedding = None
        self.word_embedding = None

        self.__load_embeddings(resources)

    def __load_embeddings(self, resources):
        logging.info("Loading %d x %d event embedding." % (
            self.para.event_arg_vocab_size + 1,
            self.para.event_embedding_dim
        ))
        # Add one dimension for padding.
        self.event_embedding = nn.Embedding(
            self.para.event_arg_vocab_size + 1,
            self.para.event_embedding_dim,
            padding_idx=self.para.event_arg_vocab_size
        )

        logging.info("Loading %d x %d word embedding." % (
            self.para.word_vocab_size + 1,
            self.para.word_embedding_dim
        ))
        self.word_embedding = nn.Embedding(
            self.para.word_vocab_size + 1,
            self.para.word_embedding_dim,
            padding_idx=self.para.word_vocab_size
        )

        if resources.word_embedding_path is not None:
            word_embed = torch.from_numpy(
                np.load(resources.word_embedding_path)
            )
            zeros = torch.zeros(1, self.para.word_embedding_dim)

            self.word_embedding.weight = nn.Parameter(
                torch.cat((word_embed, zeros))
            )

        if resources.event_embedding_path is not None:
            event_emb = torch.from_numpy(
                np.load(resources.event_embedding_path)
            )
            zeros = torch.zeros(1, self.para.event_embedding_dim)
            self.event_embedding.weight = nn.Parameter(
                torch.cat((event_emb, zeros))
            )


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

    def _event_repr(self, event_data):
        """
        Convert event data into a vector.
        :param event_data:
        :return:
        """
        event_components = []

        pred_emb = self.event_embedding(event_data['predicate'])
        frame_emb = self.event_embedding(event_data['frame'])

        event_components.append(pred_emb)
        event_components.append(frame_emb)

        for slot in event_data['slots']:
            fe_emb = self.event_embedding[slot['fe']]
            event_components.append(fe_emb)
        return torch.cat(event_components)

    def forward(self, batch_event_data, batch_context, max_context_size):
        # torch.cat([batch_context, batch_event_data])
        print("Got %d data in batch" % len(batch_event_data))

        ll_context_emb = []
        for l_context in batch_context:
            for context in l_context:
                self._event_repr(context)

        print(max_context_size)
        # print(batch_event_data)
        input("wait here.")

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
