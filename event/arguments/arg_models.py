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
            self._full_event_embedding_size(),
            para.arg_composition_layer_sizes
        )

        composed_event_dim = para.arg_composition_layer_sizes[-1]
        self.event_composition_layers = self._config_mlp(
            composed_event_dim * 2 + para.num_extracted_features,
            para.event_composition_layer_sizes
        )

        pair_event_dim = para.event_composition_layer_sizes[-1]
        # self.coh = nn.Linear(pair_event_dim, 1)

        # Output 9 different sigmas, each corresponding to one distance measure.
        self.event_to_sigma_layer = self._config_mlp(
            self.para.event_embedding_dim, [9]
        )

    def _full_event_embedding_size(self):
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

    def _mlp(self, layers, input_data, type='relu'):
        data = input_data
        for layer in layers:
            if type == 'relu':
                data = F.relu(layer(data))
            elif type == 'tanh':
                data = F.tanh(layer(data))
        return data

    def _encode_distance(self, event_emb, distances):
        """
        Encode distance into kernel, maybe event dependent.
        :param batch_event:
        :param distances:
        :return:
        """
        # Encode with a event dependent kernel
        # TODO finish this
        # May be we should not update the predicates embedding here.

        print("Embeddings")
        # batch x num event words x embedding
        print(event_emb.shape)
        print("Distances")
        # batch x 9
        print(distances.shape)
        print(distances.type())

        predicates = event_emb[:, 1, :]

        print("Predicates")
        # batch x embedding
        print(predicates.shape)

        # batch x 9
        dist_sq = distances * distances

        print("Distance square")
        print(dist_sq.shape)
        print(dist_sq.type())

        # Output 9 sigmas, each one for each distance.
        # Deviation cannot be zero, so we add a soft plus here.
        sigmas = nn.Softplus()(
            self._mlp(self.event_to_sigma_layer, predicates, type='tanh')
        )

        print("Sigmas")
        # batch x 9
        print(sigmas.shape)

        sigma_sq = 2.0 * sigmas * sigmas
        print(sigma_sq.shape)

        print(dist_sq)
        print(sigma_sq)

        kernel_value = torch.exp(dist_sq / sigma_sq)

        print(kernel_value)

        input("encoding distances.")

        return kernel_value

    def _contextual_score(self, event_emb, context_emb, features):
        # TODO finish this
        # Several ways to implement this:
        # 1. feature before/after nonlinear
        # 2. use/no kernel
        return 0

    def _event_repr(self, event_emb):
        return self._mlp(self.arg_compositions_layers, event_emb)

    def forward(self, batch_event_data, batch_context):
        # torch.cat([batch_context, batch_event_data])
        batch_event = batch_event_data['event']
        batch_features = batch_event_data['features']
        batch_distances = batch_event_data['distances']

        print("Batch event size: ", batch_event.shape)

        context_emb = self.event_embedding(batch_context)
        event_emb = self.event_embedding(batch_event)

        # The first element in each event is the predicate.
        distance_emb = self._encode_distance(event_emb, batch_distances)

        all_features = torch.cat([distance_emb, batch_features], 1)

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

    def _full_event_embedding_size(self):
        # The frame embeddings double the size.
        return 4 * self.para.event_embedding_dim * 2
