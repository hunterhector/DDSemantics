from torch import nn
from torch.nn import functional as F
import torch


class ArgCompatibleModel(nn.Module):
    def __init__(self, para, resources):
        super(ArgCompatibleModel, self).__init__()
        self.event_embedding = None
        self.word_embedding = None

        self.__load_embeddings(para, resources)

    def __load_embeddings(self, para, resources):
        self.event_embedding = nn.Embedding(
            para.event_arg_vocab_size,
            para.event_embedding_dim,
        )

        self.word_embedding = nn.Embedding(
            para.word_vocab_size,
            para.word_embedding_dim,
        )

        if resources.word_embedding is not None:
            word_embed = torch.from_numpy(resources.word_embedding)
            self.embedding.weight = nn.Parameter(word_embed)

        if resources.event_embedding is not None:
            event_emb = torch.from_numpy(resources.event_embedding)
            self.event_embedding.weight = nn.Parameter(event_emb)


class EventPairCompositionModel(ArgCompatibleModel):
    def __init__(self, para, resources):
        super(EventPairCompositionModel, self).__init__(para, resources)
        self.event_embed = nn.Embedding(para.event_arg_vocab_size,
                                        para.event_embedding_dim, padding_idx=0)

        event_hidden_size = (para.num_args + 1) * para.event_embedding_dim

        self.arg_compositions_layers = self._config_mlp(
            event_hidden_size,
            para.arg_composition_layer_sizes
        )

        composed_event_dim = para.arg_composition_layer_sizes[-1]
        self.event_composition_layers = self._config_mlp(
            composed_event_dim * 2,
            para.event_composition_layer_sizes
        )

        pair_event_dim = para.event_composition_layer_sizes[-1]
        self.coh = nn.Linear(pair_event_dim, 1)

    def _config_mlp(self, input_hidden_size, output_sizes):
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

    def forward(self, batch):
        first_event, second_event, features = batch

        first_event_emd = self._mlp(self.arg_compositions_layers, first_event)
        second_event_emd = self._mlp(self.arg_compositions_layers, second_event)

        event_pair = self._mlp(
            self.event_composition_layers,
            torch.cat([first_event_emd, second_event_emd])
        )

        score = self.coh(event_pair).squeeze(-1)
        return score
