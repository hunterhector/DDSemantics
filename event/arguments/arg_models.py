from torch import nn
from torch.nn import functional as F
import torch
import logging
from event.nn.models import KernelPooling
from event import torch_util


class ArgCompatibleModel(nn.Module):
    def __init__(self, para, resources, device, model_name):
        super(ArgCompatibleModel, self).__init__()
        self.para = para

        # This will be used to create the one-hot vector for the slots.
        self.num_slots = para.num_slots

        self.event_embedding = None
        self.word_embedding = None

        self.device = device

        self.name = model_name

        self.__load_embeddings(resources)

    def self_event_mask(self, batch_event_indices, batch_size, event_size,
                        context_size):
        """
        Return a matrix to mask out the scores from the current event.
        :param batch_event_indices:
        :param batch_size:
        :param event_size:
        :param context_size:
        :return:
        """
        selector = batch_event_indices.unsqueeze(-1)
        one_zeros = torch.ones(
            batch_size, event_size, context_size, dtype=torch.float32,
        ).to(self.device)
        one_zeros.scatter_(-1, selector, 0)
        return one_zeros

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


class RandomBaseline(ArgCompatibleModel):
    def __init__(self, para, resources, device):
        super(RandomBaseline, self).__init__(para, resources, device,
                                             'random_baseline')

    def forward(self, batch_event_data, batch_info):
        batch_features = batch_event_data['features']
        a, b, c = batch_features.shape
        return torch.rand(a, b, c)


class MostFrequentModel(ArgCompatibleModel):
    def __init__(self, para, resources, device):
        super(MostFrequentModel, self).__init__(para, resources, device,
                                                'most_freq_baseline')
        self.para = para

    def forward(self, batch_event_data, batch_info):
        # batch x instance_size x n_features
        batch_features = batch_event_data['features']
        return batch_features[:, :, 7]


class BaselineEmbeddingModel(ArgCompatibleModel):
    def __init__(self, para, resources, device):
        super(BaselineEmbeddingModel, self).__init__(para, resources, device,
                                                     'w2v_baseline')
        self.para = para

        self._score_method = para.w2v_baseline_method
        self._avg_topk = para.w2v_baseline_avg_topk

    def forward(self, batch_event_data, batch_info):
        # batch x instance_size x event_component
        batch_event_rep = batch_event_data['events']

        # batch x context_size x event_component
        batch_context = batch_info['context_events']

        batch_event_indices = batch_info['event_indices']

        # Add embedding dimension at the end.
        # batch x context_size x event_component x embedding
        context_emb = self.event_embedding(batch_context)

        # batch x instance_size x event_component x embedding
        event_emb = self.event_embedding(batch_event_rep)

        # This is the concat way:
        # batch x context_size x embedding_x_component
        if self.para.w2v_event_repr == 'concat':
            flat_context_emb = context_emb.view(
                context_emb.size()[0], -1,
                context_emb.size()[-1] * context_emb.size()[-2]
            )

            # batch x instance_size x embedding_x_component
            flat_event_emb = event_emb.view(
                event_emb.size()[0], -1,
                event_emb.size()[-1] * event_emb.size()[-2]
            )
        elif self.para.w2v_event_repr == 'sum':
            # Sum all the components together.
            # batch x instance_size x embedding
            flat_context_emb = context_emb.sum(-2)
            flat_event_emb = event_emb.sum(-2)
        else:
            raise ValueError(f"Unknown event representation method: "
                             f"[{self.para.w2v_event_repr}]")

        # Compute cosine.
        nom_event_emb = F.normalize(flat_event_emb, 2, -1)
        nom_context_emb = F.normalize(flat_context_emb, 2, -1)

        bs, event_size, _ = batch_event_rep.shape
        bs, context_size, _ = batch_context.shape
        self_mask = self.self_event_mask(batch_event_indices, bs, event_size,
                                         context_size)

        # Cosine similarities to the context.
        # batch x instance_size x context_size
        trans = torch.bmm(nom_event_emb, nom_context_emb.transpose(-2, -1))
        trans *= self_mask

        # batch x instance_size
        if self._score_method == 'max_sim':
            pooled, _ = trans.max(2, keepdim=False)
        elif self._score_method == 'average':
            pooled, _ = trans.mean(2, keepdim=False)
        elif self._score_method == 'topk_average':
            topk_pooled = torch_util.topk_with_fill(
                trans, self.para.w2v_baseline_avg_topk, 2, largest=True)
            pooled = topk_pooled.mean(2, keepdim=False)
        else:
            raise ValueError("Unknown method.")

        return pooled


class ArgCompositionModel(nn.Module):
    def __init__(self, para, resources):
        super(ArgCompositionModel, self).__init__()
        self.arg_representation_method = para.arg_representation_method

        if self.arg_representation_method == 'fix_slots':
            self._setup_fix_slot_mlp(para)
        elif self.arg_representation_method == 'role_dynamic':
            self._setup_role_based_attention(para)
        else:
            raise ValueError(f"Unknown arg representation method"
                             f" {self.arg_representation_method}")

    def _setup_fix_slot_mlp(self, para):
        component_per = 2 if para.use_frame else 1
        num_event_components = (1 + para.num_slots) * component_per
        emb_size = num_event_components * para.event_embedding_dim
        self.arg_comp_layers = _config_mlp(
            emb_size, para.arg_composition_layer_sizes)

    def _setup_role_based_attention(self, para):
        self.attention_method = para.role_compose_attention_method

        if self.attention_method == 'biaffine':
            self.role_attention_biaffine = nn.Linear(
                para.event_embedding_dim,
                para.event_embedding_dim
            )
        elif self.attention_method == 'dotproduct':
            pass
        else:
            raise NotImplementedError(
                f"Not implemented role compose "
                f"method {para.role_compose_attention_method}")

        # 3 components: predicate, frame, and the combined arguments.
        num_components = 3
        emb_size = num_components * para.event_embedding_dim

        self.arg_comp_layers = _config_mlp(
            emb_size, para.arg_composition_layer_sizes
        )

    def forward(self, *input):
        if self.arg_representation_method == 'fix_slots':
            event_emb = input[0]
            flatten_embedding_size = event_emb.size()[-1] * event_emb.size()[-2]
            flatten_event_emb = event_emb.view(
                event_emb.size()[0], -1, flatten_embedding_size)
            return _mlp(self.arg_comp_layers, flatten_event_emb)
        elif self.arg_representation_method == 'role_dynamic':
            event_data = input[0]
            pred_emb = event_data['predicates']
            flatten_embedding_size = pred_emb.size()[-1] * pred_emb.size()[-2]
            flatten_event_emb = pred_emb.view(
                pred_emb.size()[0], -1, flatten_embedding_size)

            slot_emb = event_data['slots']
            slot_values = event_data['slot_values']

            if self.attention_method == 'biaffine':
                att_slot_emb = torch.bmm(self.role_attention_biaffine(slot_emb),
                                         slot_values)
            elif self.attention_method == 'dotproduct':
                att_slot_emb = torch.bmm(slot_emb, slot_values)
            else:
                raise NotImplementedError(
                    f"Unknown attention method {self.attention_method}")

            combined_event_emb = torch.cat(
                (flatten_event_emb, att_slot_emb), -1
            )

            return _mlp(self.arg_comp_layers, combined_event_emb)


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
    def __init__(self, para, resources, device, model_name):
        super(EventCoherenceModel, self).__init__(para, resources, device,
                                                  model_name)
        logging.info(f"Pair composition network {model_name} started, "
                     f"with {self.para.num_extracted_features} extracted"
                     f" features and {self.para.num_distance_features} "
                     f"distance features.")

        self.arg_composition_model = ArgCompositionModel(para, resources)

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
                self.para.event_embedding_dim, 9
            )

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

        self.__debug_show_shapes = False

    def debug(self):
        self.__debug_show_shapes = True

    def _encode_distance(self, predicates, distances):
        """
        Encode distance into event dependent kernels.
        :param predicates:
        :param distances:
        :return:
        """
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

        # raw_var = self.event_to_var_layer(predicates.detach())
        # # Variances cannot be zero, so we use soft plus here.
        # variances = F.softplus(raw_var)

        log_var = self.event_to_var_layer(predicates.detach())

        if self.__debug_show_shapes:
            print("variance size")
            # print(variances.shape)
            print(log_var.shape)

        # print(torch.min(raw_var))
        # print(torch.min(variances))
        print(torch.min(log_var))
        # print(torch.min(- dist_sq / variances))
        print(torch.min(- dist_sq / log_var))

        input('---------------')

        kernel_value = torch.exp(- dist_sq / variances)

        print("Debugging distance encoding")
        print(distances.shape)
        print(torch.max(distances))
        print(dist_sq.shape)
        print(torch.max(dist_sq))
        # print(variances.shape)
        # print(torch.max(variances))
        input('----------------------------')

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
                added = torch.zeros(
                    (trans.shape[0], trans.shape[1],
                     self._pool_topk - trans.shape[2])).to(self.device)
                pooled_value = torch.cat((trans, added), -1)
        else:
            raise ValueError(
                'Unknown pool type {}'.format(self._vote_pool_type)
            )
        return pooled_value

    def forward(self, batch_event_data, batch_info):
        # batch x instance_size x event_component
        # batch x instance_size x n_distance_features
        batch_distances = batch_event_data['distances']
        # batch x instance_size x n_features
        batch_features = batch_event_data['features']

        # batch x context_size x event_component
        # batch x instance_size
        batch_slots = batch_info['slot_indices']
        # batch x instance_size
        batch_event_indices = batch_info['event_indices']

        # Create one hot features from index.
        # batch x instance_size x num_slots
        slot_indicator = torch.zeros(
            batch_slots.shape[0], batch_slots.shape[1], self.num_slots
        ).to(self.device)

        slot_indicator.scatter_(2, batch_slots.unsqueeze(2), 1)
        l_extracted = [batch_features, slot_indicator]

        # Add embedding dimension at the end.
        if self.para.arg_representation_method == 'fix_slots':
            batch_event_rep = batch_event_data['events']
            batch_context = batch_info['context_events']
            context_emb = self.event_embedding(batch_context)
            event_emb = self.event_embedding(batch_event_rep)
            event_repr = self.arg_composition_model(event_emb)
            context_repr = self.arg_composition_model(context_emb)

            pred_emb = event_emb[:, :, 1, :]
        elif self.para.arg_representation_method == 'role_dynamic':
            d_keys = 'predicates', 'slots', 'slot_values'
            batch_embedded_event_data = {}
            batch_embedded_context_event_data = {}
            for k in d_keys:
                batch_embedded_event_data[k] = self.event_embedding(
                    batch_event_data[k])
                batch_embedded_context_event_data[k] = self.event_embedding(
                    batch_info["context_" + k]
                )
            pred_emb = batch_embedded_event_data['predicates']
            event_repr = self.arg_composition_model(batch_embedded_event_data)
            context_repr = self.arg_composition_model(
                batch_embedded_context_event_data)
        else:
            raise ValueError(
                f"Unknown compose method {self.para.arg_representation_method}")

        if self._use_distance:
            # Adding distance features
            distance_emb = self._encode_distance(pred_emb, batch_distances)
            l_extracted.append(distance_emb)

            if self._use_distance and self.__debug_show_shapes:
                print(distance_emb.shape)

        # batch x instance_size x feature_size_1
        extracted_features = torch.cat(l_extracted, -1)

        bs, event_size, _ = event_repr.shape
        bs, context_size, _ = context_repr.shape
        self_mask = self.self_event_mask(batch_event_indices, bs, event_size,
                                         context_size)

        # Now compute the coherent features with all context events.
        coh_features = self.coh(event_repr, context_repr, self_mask)

        # batch x instance_size x feature_size_2
        all_features = torch.cat((extracted_features, coh_features), -1)

        # batch x instance_size x 1
        scores = self._linear_combine(all_features).squeeze(-1)

        if self.normalize_score:
            scores = torch.nn.Sigmoid()(scores)

        return scores
