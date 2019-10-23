import logging
import os
import pickle

import torch
from traitlets import (
    Unicode,
)
from traitlets.config import Configurable

from event.io import io_utils
from event.util import ensure_dir


class SrlResources(Configurable):
    """Resource class."""
    word_embedding_path = Unicode(help='Word Embedding path').tag(config=True)
    embedding_type = Unicode(help='Word Embedding type').tag(config=True)
    word_vocab_path = Unicode(help='Word vocabulary path').tag(config=True)

    def __init__(self, **kwargs):
        super(SrlResources, self).__init__(**kwargs)

        if self.embedding_type == 'glove':
            logging.info("Loading glove vectors.")
            word_embed = io_utils.read_glove_vectors(self.word_embedding_path)
            logging.info("Converting to torch.")
            self.word_embed_weights = torch.from_numpy(word_embed.syn0)

            if not os.path.exists(self.word_vocab_path):
                logging.info("Taking vocabulary")
                self.vocab = self.get_glove_vocab(self.word_embedding_path)
                ensure_dir(self.word_vocab_path)
                with open(self.word_vocab_path, 'wb') as out:
                    pickle.dump(self.vocab, out)
            else:
                logging.info("Loaded vocabulary from " + self.word_vocab_path)
                with open(self.word_vocab_path, 'rb') as fin:
                    self.vocab = pickle.load(fin)

    def get_glove_vocab(self, glove_path):
        vocab = {}
        linenum = 0
        with open(glove_path) as glove_in:
            for line in glove_in:
                word = line.split('\t')[0]
                vocab[word] = linenum
                linenum += 1

        return vocab
