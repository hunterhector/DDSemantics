from torch import nn
from torch.nn import functional as F
import torch
import logging
from event.nn.models import KernelPooling
from event import torch_util


class ArgCompatibleModel(nn.Module):
    def __init__(self, para, resources):
        super(ArgCompatibleModel, self).__init__()
        self.para = para

        # This will be used to create the one-hot vector for the slots.
        self.num_slots = para.num_slots

        self.event_embedding = None
        self.word_embedding = None

        self.__load_embeddings(resources)

    def __load_embeddings(self, resources):
        logging.info("Loading %d x %d event embedding." % (
            resources.event_embed_vocab.get_size(),
            self.para.event_embedding_dim
        ))

        # Add additional dimension for extra event vocab.
        self.event_embedding = nn.Embedding(
            resources.event_embed_vocab.get_size(),
            self.para.event_embedding_dim,
        )

        logging.info("Loading %d x %d word embedding." % (
            resources.word_embed_vocab.get_size(),
            self.para.word_embedding_dim
        ))

        self.word_embedding = nn.Embedding(
            resources.word_embed_vocab.get_size(),
            self.para.word_embedding_dim,
            padding_idx=self.para.word_vocab_size
        )

        if resources.word_embedding_path is not None:
            word_embed = torch.from_numpy(resources.word_embedding)

            # Add extra dimensions for word padding.
            zeros = torch.zeros(resources.word_embed_vocab.extra_size(),
                                self.para.word_embedding_dim)
            self.word_embedding.weight = nn.Parameter(
                torch.cat((word_embed, zeros))
            )

        if resources.event_embedding_path is not None:
            event_emb = torch.from_numpy(resources.event_embedding)

            # Add extra event vocab for unobserved args.
            zeros = torch.zeros(resources.event_embed_vocab.extra_size(),
                                self.para.event_embedding_dim)
            self.event_embedding.weight = nn.Parameter(
                torch.cat((event_emb, zeros))
            )


class BaselineEmbeddingModel(ArgCompatibleModel):
    def __init__(self, para, resources):
        super(BaselineEmbeddingModel, self).__init__(para, resources)
        self.para = para

        self._baseline_method = para.baseline_method
        self._avg_topk = para.baseline_avg_topk

    def forward(self, batch_event_data, batch_info):
        # batch x instance_size x event_component
        batch_event_rep = batch_event_data['rep']
        # batch x instance_size x n_distance_features
        batch_distances = batch_event_data['distances']
        # batch x instance_size x n_features
        batch_features = batch_event_data['features']

        # batch x context_size x event_component
        batch_context = batch_info['context']
        # batch x instance_size
        batch_slots = batch_info['slot_indices']
        # batch x instance_size
        batch_event_indices = batch_info['event_indices']

        # Add embedding dimension at the end.
        # batch x context_size x event_component x embedding
        context_emb = self.event_embedding(batch_context)
        # batch x instance_size x event_component x embedding
        event_emb = self.event_embedding(batch_event_rep)

        # batch x context_size x embedding_x_component
        flat_context_emb = context_emb.view(
            context_emb.size()[0], -1,
            context_emb.size()[-1] * context_emb.size()[-2]
        )

        # batch x instance_size x embedding_x_component
        flat_event_emb = event_emb.view(
            event_emb.size()[0], -1,
            event_emb.size()[-1] * event_emb.size()[-2]
        )

        # Compute cosine.
        nom_event_emb = F.normalize(flat_event_emb, 2, -1)
        nom_context_emb = F.normalize(flat_context_emb, 2, -1)

        # Cosine similarities to the context.
        # batch x instance_size x context_size
        trans = torch.bmm(nom_event_emb, nom_context_emb.transpose(-2, -1))

        # batch x instance_size
        if self._baseline_method == 'max_sim':
            pooled, _ = trans.max(2, keepdim=False)
        elif self._baseline_method == 'average':
            pooled, _ = trans.mean(2, keepdim=False)
        elif self._baseline_method == 'topk_average':
            topk_pooled = torch_util.topk_with_fill(
                trans, self.baseline_avg_topk, 2, largest=True)
            pooled = topk_pooled.mean(2, keepdim=False)
        else:
            raise ValueError("Unknown method.")

        return pooled


class ArgCompositionModel(nn.Module):
    def __init__(self, para, resources, gpu=True):
        super(ArgCompositionModel, self).__init__()
        self.compose_method = para.compose_method

        if self.compose_method == 'fix_slot_mlp':
            self.arg_compositions_layers = self._setup_fix_slot_mlp(para)

        elif self.compose_method == 'role_based_attention':
            self.attention_model = self._setup_role_based_attention()

    def _setup_fix_slot_mlp(self, para):
        emb_size = self.para.num_event_components * \
                   self.para.event_embedding_dim
        self.arg_comp_layers = _config_mlp(emb_size,
                                           para.arg_composition_layer_sizes)

    def _setup_role_based_attention(self):
        pass

    def forward(self, *input):
        if self.compose_method == 'fix_slot_mlp':
            event_emb = input[0]
            full_evm_embedding_size = event_emb.size()[-1] * event_emb.size()[
                -2]
            flatten_event_emb = event_emb.view(
                event_emb.size()[0], -1, full_evm_embedding_size)
            return _mlp(self.arg_compositions_layers, flatten_event_emb)
        elif self.compose_method == 'role_based_attention':
            return


def _config_mlp(input_hidden_size, output_sizes):
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


def _mlp(layers, input_data, activation=F.relu):
    data = input_data
    for layer in layers:
        data = activation(layer(data))
    return data


class EventCoherenceModel(ArgCompatibleModel):
    def __init__(self, para, resources, gpu=True):
        super(EventCoherenceModel, self).__init__(para, resources)
        logging.info("Pair composition network started, with %d "
                     "extracted features and %d distance features." % (
                         self.para.num_extracted_features,
                         self.para.num_distance_features
                     ))

        self.arg_composition_model = ArgCompositionModel(para, resources, gpu)

        composed_event_dim = para.arg_composition_layer_sizes[-1]

        self.event_composition_layers = _config_mlp(
            composed_event_dim + para.num_extracted_features + 9,
            para.event_composition_layer_sizes
        )

        # Number of extracted features, and dim for 1-hot slot position feature.
        feature_size = self.para.num_extracted_features + self.num_slots

        # Config feature size.
        self._vote_pool_type = para.vote_pooling
        if self._vote_pool_type == 'kernel':
            self._kp = KernelPooling()
            feature_size += self._kp.K
        elif self._vote_pool_type == 'average' or self._vote_pool_type == 'max':
            feature_size += 1
        elif self._vote_pool_type == 'topk':
            self._pool_topk = para.pool_topk
            feature_size += self._pool_topk

        self._use_distance = para.encode_distance
        if self._use_distance:
            feature_size += para.num_distance_features
            # Output 9 different var, corresponding to 9 distance measures.
            self.event_to_var_layer = nn.Linear(
                self.para.event_embedding_dim, 9)

        self._vote_method = para.vote_method

        if self._vote_method == 'biaffine':
            # Use the linear layer to simulate the middle tensor.
            self.biaffine_att_layer = nn.Linear(self.para.event_embedding_dim,
                                                self.para.event_embedding_dim)
        elif self._vote_method == 'mlp':
            self.mlp_att = _config_mlp(self.para.event_embedding_dim * 2,
                                       [1])

        self._linear_combine = nn.Linear(feature_size, 1)

        # Method to generate coherent features.
        self.coh = self._attentive_contextual_score

        if para.loss == 'cross_entropy':
            self.normalize_score = True
        else:
            self.normalize_score = False

        self.device = torch.device(
            "cuda" if gpu and torch.cuda.is_available() else "cpu"
        )

        self.__debug_show_shapes = False

    def debug(self):
        self.__debug_show_shapes = True

    def _full_event_embedding_size(self):
        return self.para.num_event_components * self.para.event_embedding_dim

    def _encode_distance(self, event_emb, distances):
        """
        Encode distance into event dependent kernels.
        :param event_emb:
        :param distances:
        :return:
        """

        if self.__debug_show_shapes:
            print("Event embedding size")
            print(event_emb.shape)

        # Encode with a event dependent kernel
        # May be we should not update the predicates embedding here.
        predicates = event_emb[:, :, 1, :]

        if self.__debug_show_shapes:
            print("Predicate size")
            print(predicates.shape)

        # d = num_distance_feature

        # Our max distance maybe to large which cause overflow, maybe clamp
        # before usage

        # batch x event_size x d
        dist_sq = distances * distances

        # Output d variances (\sigma^2), each one for each distance.
        # We detach predicates here, force the model to learn distance operation
        #  in the fully connected layer (not learn from the predicates).
        raw_var = self.event_to_var_layer(predicates.detach())
        # Variances cannot be zero, so we use soft plus here.
        variances = F.softplus(raw_var)

        if self.__debug_show_shapes:
            print("variance size")
            print(variances.shape)

        # print(torch.min(raw_var))
        # print(torch.min(variances))
        # print(torch.min(- dist_sq / variances))
        # input('---------------')

        kernel_value = torch.exp(- dist_sq / variances)

        # print("Debugging distance encoding")
        # print(distances.shape)
        # print(torch.max(distances))
        # print(dist_sq.shape)
        # print(torch.max(dist_sq))
        # print(variances.shape)
        # print(torch.max(variances))
        # input('----------------------------')

        if self.__debug_show_shapes:
            print("distance kernel values")
            print(kernel_value.shape)

        return kernel_value

    def _context_vote(self, nom_event_emb, nom_context_emb):
        # First compute the trans matrix between events and the context.
        if self._vote_method == 'cosine':
            # Normalized dot product is cosine.
            trans = torch.bmm(nom_event_emb, nom_context_emb.transpose(-2, -1))
        elif self._vote_method == 'biaffine':
            trans = torch.bmm(self.biaffine_att_layer(nom_event_emb),
                              nom_context_emb.transpose(-2, -1))
        elif self._vote_method == 'mlp':
            raise ValueError('Unimplemented MLP error.')
        else:
            raise ValueError(
                'Unknown vote computation method {}'.format(self._vote_method)
            )
        return trans

    def _attentive_contextual_score(self, event_emb, context_emb,
                                    self_avoid_mask):
        """
        Compute the contextual scores in the attentive way, i.e., computing
        some cross scores between the two representations.
        :param event_emb:
        :param context_emb:
        :param self_avoid_mask: mask of shape event_size x context_size, each
        row is contain only one zero that indicate which context should not be
        used.
        :return:
        """
        if self.__debug_show_shapes:
            print("Event Embedding shape")
            print(event_emb.shape)
            print("Context shape")
            print(context_emb.shape)

        nom_event_emb = F.normalize(event_emb, 2, -1)
        nom_context_emb = F.normalize(context_emb, 2, -1)

        trans = self._context_vote(nom_event_emb, nom_context_emb)

        if self.__debug_show_shapes:
            print("Event context similarity:")
            print(trans.shape)

            print("Avoidance mask:")
            print(self_avoid_mask.shape)
            print(self_avoid_mask.dtype)

            print("Using the mask")

            print(trans.shape)
            print(self_avoid_mask.shape)

            print("Filter with selecting matrix")

        # Make the self score zero.
        trans = trans * self_avoid_mask

        if self._vote_pool_type == 'kernel':
            pooled_value = self._kp(trans)
        elif self._vote_pool_type == 'max':
            pooled_value, _ = trans.max(2, keepdim=True)
        elif self._vote_pool_type == 'average':
            pooled_value = trans.mean(2, keepdim=True)
        elif self._vote_pool_type == 'topk':
            if trans.shape[2] >= self._pool_topk:
                pooled_value, _ = trans.topk(self._pool_topk, 2, largest=True)
            else:
                added = torch.zeros((trans.shape[0], trans.shape[1],
                                     self._pool_topk - trans.shape[2]))
                pooled_value = torch.cat((trans, added), -1)
        else:
            raise ValueError(
                'Unknown pool type {}'.format(self._vote_pool_type)
            )
        return pooled_value

    def forward(self, batch_event_data, batch_info):
        # batch x instance_size x event_component
        batch_event_rep = batch_event_data['rep']
        # batch x instance_size x n_distance_features
        batch_distances = batch_event_data['distances']
        # batch x instance_size x n_features
        batch_features = batch_event_data['features']

        # batch x context_size x event_component
        batch_context = batch_info['context']
        # batch x instance_size
        batch_slots = batch_info['slot_indices']
        # batch x instance_size
        batch_event_indices = batch_info['event_indices']

        if self.__debug_show_shapes:
            print("context shape")
            print(batch_context.shape)

            print("input batched event shape")
            print(batch_event_rep.shape)
            print(batch_distances.shape)
            print(batch_features.shape)
            print(batch_slots.shape)
            print(batch_event_indices.shape)

        # Add embedding dimension at the end.
        context_emb = self.event_embedding(batch_context)
        event_emb = self.event_embedding(batch_event_rep)

        # Create one hot features from index.
        # batch x instance_size x num_slots
        one_hot = torch.zeros(
            batch_slots.shape[0], batch_slots.shape[1], self.num_slots
        ).to(self.device)

        one_hot.scatter_(2, batch_slots.unsqueeze(2), 1)

        l_extracted = [batch_features, one_hot]

        if self._use_distance:
            distance_emb = self._encode_distance(event_emb, batch_distances)
            l_extracted.append(distance_emb)

        # batch x instance_size x feature_size_1
        extracted_features = torch.cat(l_extracted, -1)

        if self.__debug_show_shapes:
            print("Embedded shapes")
            print(context_emb.shape)
            print(event_emb.shape)

            if self._use_distance:
                print(distance_emb.shape)
            print(extracted_features.shape)

        event_repr = self.arg_composition_model(event_emb)
        context_repr = self.arg_composition_model(context_emb)

        if self.__debug_show_shapes:
            print("Event and context repr")
            print(event_repr.shape)
            print(context_repr.shape)

        bs, event_size, event_repr_dim = event_repr.shape
        bs, context_size, event_repr_dim = context_repr.shape

        selector = batch_event_indices.unsqueeze(-1)

        if self.__debug_show_shapes:
            print("Create the selecting matrix")
            print(selector.shape)

        one_zeros = torch.ones(
            bs, event_size, context_size, dtype=torch.float32,
        ).to(self.device)
        one_zeros.scatter_(-1, selector, 0)

        # Now compute the coherent features with all context events.
        coh_features = self.coh(event_repr, context_repr, one_zeros)

        # batch x instance_size x feature_size_2
        all_features = torch.cat((extracted_features, coh_features), -1)

        if self.__debug_show_shapes:
            print("One zero")
            print(one_zeros.shape)
            print("Coh features")
            print(coh_features.shape)
            print("all feature")
            print(all_features.shape)

        # batch x instance_size x 1
        scores = self._linear_combine(all_features).squeeze(-1)

        if self.normalize_score:
            scores = torch.nn.Sigmoid()(scores)

        return scores
