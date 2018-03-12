import os
import logging
from collections import defaultdict
import torch


class TaggedMentionReader:
    def __init__(self, data_files, config, token2i=None, tag2i=None):
        self.experiment_folder = config.experiment_folder

        self.data_files = data_files

        self.data_format = config.format

        self.no_punct = config.no_punct
        self.no_sentence = config.no_sentence

        self.batch_size = config.batch_size
        self.window_size = max(config.window_sizes)

        if token2i is None and tag2i is None:
            self.token2i = defaultdict(lambda: len(self.token2i))
            self.tag2i = defaultdict(lambda: len(self.tag2i))

            self.unk = self.token2i["<unk>"]
            self.none_tag = self.tag2i["NONE"]

            self.hash()
        else:
            assert token2i is not None
            assert tag2i is not None

            self.token2i = token2i
            self.tag2i = tag2i

            self.unk = self.token2i["<unk>"]
            self.none_tag = self.tag2i["NONE"]

        self.i2token = dict([(v, k) for k, v in self.token2i.items()])
        self.i2tag = dict([(v, k) for k, v in self.tag2i.items()])

        logging.info("Corpus with [%d] words and [%d] tokens.",
                     len(self.token2i), len(self.tag2i))

        self.__batch_data = []

    def hash(self):
        import pickle
        tag_path = os.path.join(self.experiment_folder, 'tags_dict.pickle')
        word_path = os.path.join(self.experiment_folder, 'tokens_dict.pickle')

        if os.path.exists(tag_path) and os.path.exists(word_path):
            with open(tag_path, 'rb') as tag_pickle:
                self.tag2i = pickle.load(tag_pickle)
            with open(word_path, 'rb') as source_pickle:
                self.token2i = pickle.load(source_pickle)
        else:
            # Hash the tokens and tags.
            for data_file in self.data_files:
                with open(data_file) as data:
                    for line in data:
                        if line.startswith("#") or not line.strip():
                            continue
                        parts = line.split()
                        token = parts[1]
                        tag = parts[9]
                        self.token2i[token]
                        self.tag2i[tag]

            # Write hashed pickle.
            with open(tag_path, 'wb') as tag_pickle:
                pickle.dump(dict(self.tag2i), tag_pickle)
            with open(word_path, 'wb') as word_pickle:
                pickle.dump(dict(self.token2i), word_pickle)

        # Setup default UNK.
        self.token2i = defaultdict(lambda: self.unk, self.token2i)
        self.tag2i = defaultdict(lambda: self.none_tag, self.tag2i)

    def reveal_tokens(self, token_ids):
        return [self.i2token[t] for t in token_ids]

    def reveal_tags(self, tag_ids):
        return [self.i2tag[t] for t in tag_ids]

    def parse(self):
        for data_file in self.data_files:
            with open(data_file) as data:
                token_ids = []
                tag_ids = []
                for line in data:
                    if line.startswith("#"):
                        if line.startswith("# doc"):
                            docid = line.split("=")[1].strip()
                            if self.no_sentence:
                                # Yield data for whole document if we didn't
                                # yield per sentence.
                                if token_ids:
                                    yield token_ids, tag_ids
                    elif not line.strip():
                        if not self.no_sentence:
                            # Yield data for each sentence.
                            yield token_ids, tag_ids
                            token_ids = []
                            tag_ids = []
                    else:
                        parts = line.split()
                        token = parts[1]
                        pos = parts[7]
                        tag = parts[9]

                        if pos == 'punct' and self.no_punct:
                            continue

                        token_ids.append(self.token2i[token])
                        tag_ids.append(self.tag2i[tag])

    def read_window(self):
        token_ids, tag_ids = next(self.parse())
        assert len(token_ids) == len(tag_ids)

        token_pad = [self.unk] * self.window_size
        tag_pad = [self.none_tag] * self.window_size

        token_ids = token_pad + token_ids + token_pad
        tag_ids = tag_pad + tag_ids + tag_pad

        for i in range(len(token_ids)):
            start = i
            end = i + self.window_size * 2 + 1
            yield token_ids[start: end], tag_ids[start:end]

    def convert_batch(self):
        tokens, tags = zip(*self.__batch_data)
        tokens, tags = torch.FloatTensor(tokens), torch.FloatTensor(tags)
        if torch.cuda.is_available():
            tokens.cuda()
            tags.cuda()
        return tokens, tags

    def read_batch(self):
        token_ids, tag_ids = self.read_window()

        if self.batch_size < len(self.__batch_data):
            self.__batch_data.append((token_ids, tag_ids))
        else:
            data = self.convert_batch()
            self.__batch_data.clear()
            return data

    def token_dict(self):
        return self.token2i

    def tag_dict(self):
        return self.tag2i

    def num_classes(self):
        return len(self.i2tag)

    def vocab_size(self):
        return len(self.i2token)
