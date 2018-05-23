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
                resources.word_embedding
            )
            zeros = torch.zeros(1, self.para.word_embedding_dim)
            self.word_embedding.weight = nn.Parameter(
                torch.cat((word_embed, zeros))
            )

        if resources.event_embedding_path is not None:
            event_emb = torch.from_numpy(
                resources.event_embedding
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
        # self.coh = nn.Linear(pair_event_dim, 1)

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

    def _encode_distance(self, predicates, distances):
        # Encode with a predicate specific kernel
        # TODO finish this
        return distances

    def _contextual_score(self, event_emb, context_emb, features):
        # TODO finish this
        # Several ways to implement this:
        # 1. feature before/after nonlinear
        # 2. use/no kernel
        return 0

    def _event_repr(self, event_emb):
        return self._mlp(self.arg_compositions_layers, event_emb)

    def forward(self, batch_event_data, batch_context, max_context_size):
        # torch.cat([batch_context, batch_event_data])
        print("Got %d data in batch" % len(batch_event_data))

        context_emb = self.event_embedding(batch_event_data['context'])
        event_emb = self.event_embedding(batch_event_data['event'])

        # The first element in each event is the predicate.
        predicates = batch_event_data['event'][:, :, 1]
        distance_emb = self._encode_distance(
            predicates, batch_event_data['distance'])

        # TODO: add position feature here.
        all_features = torch.cat(
            [distance_emb, batch_event_data['features']], 1)

        print(max_context_size)
        # print(batch_event_data)
        input("wait here.")

        event_repr = self._event_repr(event_emb)
        # TODO: broadcast to context.
        context_repr = self._event_repr(context_emb)

        # Now compute the "attention" with all context events.
        score = self._contextual_score(event_repr, context_repr, all_features)
        return score


class FrameAwareEventPairCompositionModel(EventPairCompositionModel):
    def __init__(self, para, resources):
        super().__init__(para, resources)

    def _raw_event_embedding_size(self):
        # The frame embeddings double the size.
        return 4 * self.para.event_embedding_dim * 2
