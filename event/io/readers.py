import os
import logging
from collections import defaultdict
import torch
import numpy as np
import pickle


class Vocab:
    def __init__(self, base_folder, name, embedding_path=None, emb_dim=100):
        self.fixed = False
        self.base_folder = base_folder
        self.name = name

        if self.load_map():
            logging.info("Loaded existing vocabulary mapping.")
            self.fix()
        else:
            logging.info("Creating new vocabulary mapping file.")
            self.token2i = defaultdict(lambda: len(self.token2i))

        self.unk = self.token2i["<unk>"]

        if embedding_path:
            logging.info("Loading embeddings from %s." % embedding_path)
            self.embedding = self.load_embedding(embedding_path, emb_dim)
            self.fix()

        self.i2token = dict([(v, k) for k, v in self.token2i.items()])

    def __call__(self, *args, **kwargs):
        return self.token_dict()[args[0]]

    def load_embedding(self, embedding_path, emb_dim):
        with open(embedding_path, 'r') as f:
            emb_list = []
            for line in f:
                parts = line.split()
                word = parts[0]
                if len(parts) > 1:
                    embedding = np.array([float(val) for val in parts[1:]])
                else:
                    embedding = np.random.rand(1, emb_dim)

                self.token2i[word]
                emb_list.append(embedding)
            logging.info("Loaded %d words." % len(emb_list))
            return np.vstack(emb_list)

    def fix(self):
        # After fixed, the vocabulary won't grow.
        self.token2i = defaultdict(lambda: self.unk, self.token2i)
        self.fixed = True
        self.dump_map()

    def reveal_origin(self, token_ids):
        return [self.i2token[t] for t in token_ids]

    def token_dict(self):
        return self.token2i

    def vocab_size(self):
        return len(self.i2token)

    def dump_map(self):
        path = os.path.join(self.base_folder, self.name + '.pickle')
        if not os.path.exists(path):
            with open(path, 'wb') as p:
                pickle.dump(dict(self.token2i), p)

    def load_map(self):
        path = os.path.join(self.base_folder, self.name + '.pickle')
        if os.path.exists(path):
            with open(path, 'rb') as p:
                self.token2i = pickle.load(p)
                return True
        else:
            return False


class ConllUReader:
    def __init__(self, data_files, config, token_vocab, tag_vocab):
        self.experiment_folder = config.experiment_folder
        self.data_files = data_files
        self.data_format = config.format

        self.no_punct = config.no_punct
        self.no_sentence = config.no_sentence

        self.batch_size = config.batch_size
        self.window_size = max(config.window_sizes)

        logging.info("Batch size is %d, window size is %d." % (
            self.batch_size, self.window_size))

        self.token_vocab = token_vocab
        self.tag_vocab = tag_vocab

        logging.info("Corpus with [%d] words and [%d] tags.",
                     self.token_vocab.vocab_size(),
                     self.tag_vocab.vocab_size())

        self.__batch_data = []

    def parse(self):
        for data_file in self.data_files:
            logging.info("Loading data from [%s] " % data_file)
            with open(data_file) as data:
                token_ids = []
                features = []
                tag_ids = []
                for line in data:
                    if line.startswith("#"):
                        if line.startswith("# doc"):
                            docid = line.split("=")[1].strip()
                            if self.no_sentence:
                                # Yield data for whole document if we didn't
                                # yield per sentence.
                                if token_ids:
                                    yield token_ids, tag_ids, features
                    elif not line.strip():
                        if not self.no_sentence:
                            # Yield data for each sentence.
                            yield token_ids, tag_ids, features
                            token_ids = []
                            tag_ids = []
                    else:
                        parts = line.lower().split()
                        token = parts[1]
                        pos = parts[7]
                        tag = parts[9]

                        if pos == 'punct' and self.no_punct:
                            continue

                        lemma = parts[2]
                        features.append([lemma, pos])

                        token_ids.append(self.token_vocab(token))
                        tag_ids.append(self.tag_vocab(tag))

    def read_window(self):
        for token_ids, tag_ids, features in self.parse():
            assert len(token_ids) == len(tag_ids)

            token_pad = [self.token_vocab.unk] * self.window_size
            tag_pad = [self.tag_vocab.unk] * self.window_size

            feature_dim = len(features[0])

            feature_pad = [["UNK"] * feature_dim] * self.window_size

            actual_len = len(token_ids)

            token_ids = token_pad + token_ids + token_pad
            tag_ids = tag_pad + tag_ids + tag_pad
            features = feature_pad + features + feature_pad

            for i in range(actual_len):
                start = i
                end = i + self.window_size * 2 + 1
                yield token_ids[start: end], tag_ids[start:end], features[
                                                                 start:end]

    def convert_batch(self):
        tokens, tags = zip(*self.__batch_data)
        tokens, tags = torch.FloatTensor(tokens), torch.FloatTensor(tags)
        if torch.cuda.is_available():
            tokens.cuda()
            tags.cuda()
        return tokens, tags

    def read_batch(self):
        for token_ids, tag_ids, features in self.read_window():
            if len(self.__batch_data) < self.batch_size:
                self.__batch_data.append((token_ids, tag_ids))
            else:
                data = self.convert_batch()
                self.__batch_data.clear()
                return data

    def num_classes(self):
        return self.tag_vocab.vocab_size()
