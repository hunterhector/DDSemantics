from torch import nn
from torch.nn import functional as F
import torch
import logging
from event.nn.models import KernelPooling
from torch.nn.parameter import Parameter
from event import torch_util
from torch.nn.modules.linear import Bilinear


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
                torch.cat((zeros, word_embed))
            )

        if resources.event_embedding_path is not None:
            event_emb = torch.from_numpy(
                resources.event_embedding
            )
            zeros = torch.zeros(1, self.para.event_embedding_dim)
            self.event_embedding.weight = nn.Parameter(
                torch.cat((zeros, event_emb))
            )


class EventPairCompositionModel(ArgCompatibleModel):
    def __init__(self, para, resources, gpu=True):
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

        # Number of extracted features, and 1 slot position feature.
        feature_size = self.para.num_extracted_features + 1

        # Config feature size.
        self._vote_pool_type = para.vote_pooling
        if self._vote_pool_type == 'kernel':
            self._kp = KernelPooling()
            feature_size += self._kp.K
        elif self._vote_pool_type == 'average' or self._vote_pool_type == 'max':
            feature_size += 1

        self._use_distance = para.encode_distance
        if self._use_distance:
            feature_size += para.num_distance_features
            # Output 9 different var, corresponding to 9 distance measures.
            self.event_to_var_layer = nn.Linear(
                self.para.event_embedding_dim, 9)

        self._vote_method = para.vote_method

        if self._vote_method == 'biaffine':
            self.bilinear_layer = Bilinear(
                self.para.event_embedding_dim,
                self.para.event_embedding_dim,
                1
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

        self.device = torch.device(
            "cuda" if gpu and torch.cuda.is_available() else "cpu"
        )

        self.__debug_show_shapes = False

    def debug(self):
        self.__debug_show_shapes = True

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

    def _concat_contextual_score(self, event_emb, context_emb, batch_slots):
        """
        Compute the contextual scores after concatenation and projection.
        :param event_emb:
        :param context_emb:
        :param batch_slots:
        :return:
        """
        raise NotImplementedError

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
        nom_event_emb = F.normalize(event_emb, 2, -1)
        nom_context_emb = F.normalize(context_emb, 2, -1)

        # First compute the trans matrix between events and the context.
        if self._vote_method == 'cosine':
            trans = torch.bmm(nom_event_emb, nom_context_emb.transpose(-2, -1))
        elif self._vote_method == 'biaffine':
            # TODO handle the batch size here.
            trans = self.bilinear_layer(nom_event_emb, nom_context_emb)
        else:
            raise ValueError(
                'Unknown vote computation method {}'.format(self._vote_method)
            )

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

        # index_select and masked_select does not work here.
        # trans = torch.index_select(trans, -1, self_avoid_mask)

        # Make the self score zero.
        trans = trans * self_avoid_mask

        if self._vote_pool_type == 'kernel':
            pooled_value = self._kp(trans)
        elif self._vote_pool_type == 'max':
            pooled_value, _ = trans.max(2, keepdim=True)
        elif self._vote_pool_type == 'average':
            pooled_value = trans.mean(2, keepdim=True)
        elif self._vote_pool_type == 'topk':
            # TODO add a top k pooling for comparison
            pass
        else:
            raise ValueError(
                'Unknown pool type {}'.format(self._vote_pool_type)
            )

        # print("During COH")
        # torch_util.show_tensors()
        # torch_util.gpu_mem_report()

        return pooled_value

    def _event_repr(self, event_emb):
        # Flatten the last 2 dimension.
        full_evm_embedding_size = event_emb.size()[-1] * event_emb.size()[-2]
        flatten_event_emb = event_emb.view(
            event_emb.size()[0], -1, full_evm_embedding_size)
        return self._mlp(self.arg_compositions_layers, flatten_event_emb)

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

        context_emb = self.event_embedding(batch_context)
        event_emb = self.event_embedding(batch_event_rep)

        # TODO: It seems that slots should not be used as numbers.
        l_extracted = [batch_features, batch_slots.unsqueeze(-1)]

        if self._use_distance:
            distance_emb = self._encode_distance(event_emb, batch_distances)
            l_extracted.append(distance_emb)

        extracted_features = torch.cat(l_extracted, -1)

        if self.__debug_show_shapes:
            print("Embedded shapes")
            print(context_emb.shape)
            print(event_emb.shape)

            if self._use_distance:
                print(distance_emb.shape)

            print(extracted_features.shape)

        event_repr = self._event_repr(event_emb)
        context_repr = self._event_repr(context_emb)

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

        all_features = torch.cat((extracted_features, coh_features), -1)

        if self.__debug_show_shapes:
            print("One zero")
            print(one_zeros.shape)
            print("Coh features")
            print(coh_features.shape)
            print("all feature")
            print(all_features.shape)

        scores = self._linear_combine(all_features).squeeze(-1)

        if self.normalize_score:
            scores = torch.nn.Sigmoid()(scores)

        return scores
