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
        self.entities, self.events = self.load_wordlist(
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
                if len(parts) == 4:
                    relation_subtype, relation_type, domain, range_type = parts


        return entities, events

    def load_ontology(self):
        pass

    def predict(self, *input):
        for words, _, feature_list in input:
            center = math.floor(len(words) / 2)
            lemmas = [features[0] for features in feature_list]
            pos_list = [features[1] for features in feature_list]
            deps = [(features[3], features[4]) for features in feature_list]

            center_lemma = lemmas[center]
            word = self.token_vocab.reveal_origin(words)[center]

            unknown_type = "O"
            event_type = unknown_type

            if word in self.events:
                event_type = self.events[word]

            if center_lemma in self.events:
                event_type = self.events[center_lemma]

            if not event_type == unknown_type:
                self.predict_args(center, event_type, lemmas, pos_list, deps)

            return event_type

    def predict_args(self, center, event_type, context, pos_list, deps):
        for distance in range(1, center + 1):
            left = center - distance
            right = center + distance

            left_lemma = context[left]
            right_lemma = context[right]

            if left_lemma in self.entities:
                self.check_arg(context[center], event_type, left_lemma, deps)

            if right_lemma in self.entities:
                self.check_arg(context[center], event_type, right_lemma, deps)

    def check_arg(self, predicate, event_type, arg_lemma, features):
        entity_type = self.entities[arg_lemma]
