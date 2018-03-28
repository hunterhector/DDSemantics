import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, Conv2d, Embedding
from torch.autograd import Variable
import torch
import logging
import math


class MentionDetector:
    def __init__(self, **kwargs):
        super().__init__()

    def predict(self, *input):
        pass


class DLMentionDetector(nn.Module, MentionDetector):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, *input):
        raise NotImplementedError


class TextCNN(DLMentionDetector):
    def __init__(self, config, num_classes, vocab_size):
        super().__init__(config=config)

        w_embed_dim = config.word_embedding_dim
        filter_sizes = config.window_sizes
        filter_num = config.filter_num
        max_filter_size = max(config.window_sizes)

        self.fix_embedding = config.fix_embedding
        position_embed_dim = config.position_embedding_dim
        self.word_embed = Embedding(vocab_size, w_embed_dim)
        self.position_embed = Embedding(2 * max_filter_size + 1,
                                        position_embed_dim)

        self.full_embed_dim = w_embed_dim + position_embed_dim

        self.convs1 = ModuleList(
            [Conv2d(1, filter_num, (fs, self.full_embed_dim)) for fs in
             filter_sizes]
        )

        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(len(filter_sizes) * filter_num, num_classes)
        # Add batch size dimension.
        self.positions = torch.IntTensor(range(2 * max_filter_size + 1))

    def forward(self, *input):
        print("Model input")
        print(input)

        # (Batch, Length, Emb Dimension)
        word_embed = self.word_embed(input)
        position_embed = self.position_embed(self.positions)
        x = torch.cat((word_embed, position_embed), -1)

        # Add a single dimension for channel in.
        x = x.unsqueeze(1)
        # [(N, Co, W), ...]*len(Ks)

        print('embedded')

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        # [(N, Co), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logits = self.linear(x)  # (N, C)

        return logits

    def predict(self, *input):
        logits = self.forward(input)
        return torch.max(logits, 1)[1]

    def conv_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x


class FrameMappingDetector(MentionDetector):
    def __init__(self, config, token_vocab):
        super().__init__(config=config)
        self.experiment_folder = config.experiment_folder
        self.lex_mapping = self.load_frame_lex(config.frame_lexicon)
        self.entities, self.events, self.relations = self.load_wordlist(
            config.entity_list, config.event_list, config.relation_list
        )
        self.token_vocab = token_vocab
        self.load_ontology()

    def load_frame_lex(self, frame_path):
        import xml.etree.ElementTree as ET
        import os

        ns = {'berkeley': 'http://framenet.icsi.berkeley.edu'}

        lex_mapping = {}

        for file in os.listdir(frame_path):
            with open(os.path.join(frame_path, file)) as f:
                tree = ET.parse(f)
                frame = tree.getroot()
                frame_name = frame.get('name')
                for lexUnit in frame.findall('berkeley:lexUnit', ns):
                    lex = lexUnit.get('name')
                    lexeme = lexUnit.findall('berkeley:lexeme', ns)[0].get(
                        'name')
                    if lexeme not in lex_mapping:
                        lex_mapping[lexeme] = []

                    lex_mapping[lexeme].append(frame_name)
        return lex_mapping

    def load_wordlist(self, entity_file, event_file, relation_file):
        events = {}
        entities = {}
        relations = {}
        with open(event_file) as fin:
            for line in fin:
                parts = line.strip().split()
                if len(parts) == 2:
                    word, ontology = line.strip().split()
                    events[word] = ontology
        with open(entity_file) as fin:
            for line in fin:
                parts = line.strip().split()
                if len(parts) == 2:
                    word, ontology = line.strip().split()
                    entities[word] = ontology
        with open(relation_file) as fin:
            for line in fin:
                parts = line.strip().split()
                if parts:
                    event_type = parts[0]
                    args = parts[1:]

                    if event_type not in relations:
                        relations[event_type] = {}

                    for arg in args:
                        arg_role, arg_types = arg.split(":")
                        relations[event_type][arg_role] = arg_types.split(",")

        return entities, events, relations

    def load_ontology(self):
        pass

    def predict(self, *input):
        l_types = []
        l_args = []

        # TODO: Currently not producing correct dimension.
        for words, _, l_feature, l_meta in input:
            center = math.floor(len(words) / 2)
            lemmas = [features[0] for features in l_feature]
            pos_list = [features[1] for features in l_feature]
            deps = [(features[2], features[3]) for features in l_feature]

            center_lemma = lemmas[center]
            word = self.token_vocab.reveal_origin(words)[center]

            unknown_type = "O"
            event_type = unknown_type
            args = {}

            if word in self.events:
                event_type = self.events[word]

            if center_lemma in self.events:
                event_type = self.events[center_lemma]

            print(l_feature, len(l_feature))
            print(l_meta, len(l_meta))

            if not event_type == unknown_type:
                res = self.predict_args(center, event_type, lemmas, pos_list,
                                        deps)

                for role, entity in res.items():
                    if entity:
                        index, entity_type = entity
                        features = l_feature[index]

                        print(index, entity_type, features)

                        meta = l_meta[index]

                        args[role] = features[0], meta[1], entity_type

            l_types.append(event_type)
            l_args.append(args)
        return l_types, l_args

    def predict_args(self, center, event_type, context, pos_list, deps):
        if event_type not in self.relations:
            return {}

        expected_relations = self.relations[event_type]
        expected_relations["Location"] = ["Loc", "GPE"]
        expected_relations["Time"] = ["Time"]

        filled_relations = dict([(k, None) for k in expected_relations])
        num_to_fill = len(filled_relations)

        relation_lookup = {}
        for role, types in expected_relations.items():
            for t in types:
                relation_lookup[t] = role

        for distance in range(1, center + 1):
            left = center - distance
            right = center + distance

            left_lemma = context[left]
            right_lemma = context[right]

            if left_lemma in self.entities:
                arg_type = self.check_arg(context[center], event_type,
                                          left_lemma, deps)
                if arg_type in relation_lookup:
                    possible_rel = relation_lookup[arg_type]
                    if filled_relations[possible_rel] is None:
                        filled_relations[possible_rel] = (left, arg_type)
                        num_to_fill -= 1

            if right_lemma in self.entities:
                arg_type = self.check_arg(context[center], event_type,
                                          right_lemma, deps)
                if arg_type in relation_lookup:
                    possible_rel = relation_lookup[arg_type]
                    if filled_relations[possible_rel] is None:
                        filled_relations[possible_rel] = (right, arg_type)
                        num_to_fill -= 1

            if num_to_fill == 0:
                break

        return filled_relations

    def check_arg(self, predicate, event_type, arg_lemma, features):
        unknown_type = "O"

        entity_type = unknown_type
        if arg_lemma in self.entities:
            entity_type = self.entities[arg_lemma]

        if not entity_type == unknown_type:
            return entity_type

        return None
