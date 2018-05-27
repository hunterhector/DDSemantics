from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
import logging
from event.nn.models import KernelPooling


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
        logging.info("Pair composition network started, with %d "
                     "extracted features and %d distance features." % (
                         self.para.num_extracted_features, 9
                     ))

        self.arg_compositions_layers = self._config_mlp(
            self._full_event_embedding_size(),
            para.arg_composition_layer_sizes
        )

        composed_event_dim = para.arg_composition_layer_sizes[-1]

        self.event_composition_layers = self._config_mlp(
            composed_event_dim + para.num_extracted_features + 9,
            para.event_composition_layer_sizes
        )

        pair_event_dim = para.event_composition_layer_sizes[-1]
        # self.coh = nn.Linear(pair_event_dim, 1)

        # Output 9 different var, each corresponding to one distance measure.
        self.event_to_var_layer = nn.Linear(self.para.event_embedding_dim, 9)

        self._kp = KernelPooling()

        self._linear_combine = nn.Linear(
            self.para.num_extracted_features + 9 + self._kp.K, 1
        )

        if para.loss == 'cross_entropy':
            self.normalize_score = True
        else:
            self.normalize_score = False

    def _full_event_embedding_size(self):
        return self.para.num_event_components * self.para.event_embedding_dim

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

    def _mlp(self, layers, input_data, activation=F.relu):
        data = input_data
        for layer in layers:
            data = activation(layer(data))
        return data

    def _encode_distance(self, event_emb, distances):
        """
        Encode distance into event dependent kernels.
        :param event_emb:
        :param distances:
        :return:
        """
        # Encode with a event dependent kernel
        # May be we should not update the predicates embedding here.
        predicates = event_emb[:, 1, :]

        # batch x 9
        dist_sq = distances * distances

        # Output 9 variances (\sigma^2), each one for each distance.
        # Variances cannot be zero, so we use soft plus here.
        # We detach predicates here, force the model to learn distance operation
        #  in the fully connected layer.
        variances = F.softplus(self.event_to_var_layer(predicates.detach()))

        kernel_value = torch.exp(- dist_sq / variances)

        return kernel_value

    def _contextual_score(self, event_emb, context_emb, batch_slots):
        # Several ways to implement this:
        # 1. feature before/after nonlinear
        # 2. use/no kernel. Done

        nom_event_emb = F.normalize(event_emb, 2, -1)
        nom_context_emb = F.normalize(context_emb, 2, -1)

        print("computing contextual score")
        print("event embedding")
        print(nom_event_emb.shape)
        print("contextual embedding")
        print(nom_context_emb.transpose(-2, -1).shape)

        print("Slots to filled")
        print(batch_slots.shape)

        # First compute the trans matrix between events and the context.
        trans = torch.bmm(nom_event_emb, nom_context_emb.transpose(-2, -1))
        print("Trans matrix")
        print(trans.shape)

        print("compute kp")
        kernel_value = self._kp(trans)

        return kernel_value

    def _event_repr(self, event_emb):
        # Flatten the last 2 dimension.
        full_evm_embedding_size = event_emb.size()[-1] * event_emb.size()[-2]
        flatten_event_emb = event_emb.view(
            event_emb.size()[0], -1, full_evm_embedding_size)
        return self._mlp(self.arg_compositions_layers, flatten_event_emb)

    def forward(self, batch_event_data, batch_info):
        # TODO: each batch only contains one center event. If joint modeling
        # with multiple events are needed, we will have multiple centers here.

        batch_event = batch_event_data['event']
        batch_features = batch_event_data['features']
        batch_distances = batch_event_data['distances']

        batch_context = batch_info['context']
        batch_slots = batch_info['slot']

        print("Batch event size: ", batch_event.shape)

        context_emb = self.event_embedding(batch_context)
        event_emb = self.event_embedding(batch_event)

        # The first element in each event is the predicate.
        distance_emb = self._encode_distance(event_emb, batch_distances)

        extracted_features = torch.cat([distance_emb, batch_features], 1)
        print("Extracted features")
        print(extracted_features.shape)

        print("Compute event representations.")

        print("Event embedding.")
        print(event_emb.shape)

        event_repr = self._event_repr(event_emb)

        print("Event representation.")
        print(event_repr.shape)

        print("Context embedding")
        print(context_emb.shape)

        context_repr = self._event_repr(context_emb)

        print("Context representation.")
        print(context_repr.shape)

        # Now compute the "attention" with all context events.
        kp_mtx = self._contextual_score(event_repr, context_repr, batch_slots)

        print("KP")
        print(kp_mtx.shape)

        print("Features")
        print(extracted_features.shape)

        all_features = torch.cat((extracted_features.unsqueeze(1), kp_mtx), -1)

        scores = self._linear_combine(all_features).squeeze(-1)

        if self.normalize_score:
            scores = torch.nn.Sigmoid()(scores)

        print(scores.shape)

        return scores


class FrameAwareEventPairCompositionModel(EventPairCompositionModel):
    def __init__(self, para, resources):
        super().__init__(para, resources)

    def _full_event_embedding_size(self):
        # The frame embeddings double the size.
        return 4 * self.para.event_embedding_dim * 2
