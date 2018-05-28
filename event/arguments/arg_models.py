from torch import nn
from torch.nn import functional as F
import torch
import logging
from event.nn.models import KernelPooling
from torch.nn.parameter import Parameter


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

        # Output 9 different var, each corresponding to one distance measure.
        self.event_to_var_layer = nn.Linear(self.para.event_embedding_dim, 9)

        feature_size = self.para.num_extracted_features + 9

        # Config feature size.
        self._vote_pool_type = para.vote_pooling
        if self._vote_pool_type == 'kernel':
            self._kp = KernelPooling()
            feature_size += self._kp.K
        elif self._vote_pool_type == 'average' or self._vote_pool_type == 'max':
            feature_size += 1

        self._vote_method = para.vote_method

        if self._vote_method == 'biaffine':
            self.mtx_vote_comp = Parameter(
                torch.Tensor(self.para.event_embedding_dim,
                             self.para.event_embedding_dim)
            )

        self._linear_combine = nn.Linear(feature_size, 1)

        if para.coherence_method == 'attentive':
            self.coh = self._attentive_contextual_score
        elif para.coherence_method == 'concat':
            self.coh = self._concat_contextual_score

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

    def _concat_contextual_score(self, event_emb, context_emb, batch_slots):
        """
        Compute the contextual scores after concatenation and projection.
        :param event_emb:
        :param context_emb:
        :param batch_slots:
        :return:
        """
        raise NotImplementedError

    def _attentive_contextual_score(self, event_emb, context_emb, batch_slots):
        """
        Compute the contextual scores in the attentive way, i.e., computing
        dot products between the embeddings, and then apply pooling.
        :param event_emb:
        :param context_emb:
        :param batch_slots:
        :return:
        """
        nom_event_emb = F.normalize(event_emb, 2, -1)
        nom_context_emb = F.normalize(context_emb, 2, -1)

        # First compute the trans matrix between events and the context.
        if self._vote_method == 'cosine':
            trans = torch.bmm(nom_event_emb, nom_context_emb.transpose(-2, -1))
        elif self._vote_method == 'biaffine':
            # TODO finish the bi-linear implementation.
            trans = torch.bmm(nom_event_emb, self.mtx_vote_comp)
            trans = torch.bmm(trans, nom_context_emb)
        else:
            raise ValueError(
                'Unknown vote computation method {}'.format(self._vote_method)
            )

        if self._vote_pool_type == 'kernel':
            pooled_value = self._kp(trans)
        elif self._vote_pool_type == 'max':
            pooled_value, _ = trans.max(2, keepdim=True)
        elif self._vote_pool_type == 'average':
            pooled_value = trans.mean(2, keepdim=True)
        else:
            raise ValueError(
                'Unknown pool type {}'.format(self._vote_pool_type)
            )

        return pooled_value

    def _event_repr(self, event_emb):
        # Flatten the last 2 dimension.
        full_evm_embedding_size = event_emb.size()[-1] * event_emb.size()[-2]
        flatten_event_emb = event_emb.view(
            event_emb.size()[0], -1, full_evm_embedding_size)
        return self._mlp(self.arg_compositions_layers, flatten_event_emb)

    def forward(self, batch_event_data, batch_info):
        batch_event = batch_event_data['event']
        batch_features = batch_event_data['features']
        batch_distances = batch_event_data['distances']

        batch_context = batch_info['context']
        batch_slots = batch_info['slot']

        context_emb = self.event_embedding(batch_context)
        event_emb = self.event_embedding(batch_event)

        # The first element in each event is the predicate.
        distance_emb = self._encode_distance(event_emb, batch_distances)

        extracted_features = torch.cat([distance_emb, batch_features], 1)

        event_repr = self._event_repr(event_emb)

        context_repr = self._event_repr(context_emb)

        # Now compute the coherent features with all context events.
        coh_features = self.coh(event_repr, context_repr, batch_slots)

        all_features = torch.cat(
            (extracted_features.unsqueeze(1), coh_features), -1)

        scores = self._linear_combine(all_features).squeeze(-1)

        if self.normalize_score:
            scores = torch.nn.Sigmoid()(scores)
        return scores
