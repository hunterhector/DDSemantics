from traitlets.config import Configurable
from traitlets import (
    Int,
    List,
    Unicode,
)
import numpy as np
import logging
from event.arguments.prepare.event_vocab import load_vocab


class Resources(Configurable):
    """
    Resource class.
    """
    event_embedding_path = Unicode(help='Event Embedding path').tag(config=True)
    word_embedding_path = Unicode(help='Word Embedding path').tag(config=True)

    event_vocab_path = Unicode(help='Event Vocab').tag(config=True)
    word_vocab_path = Unicode(help='Word Vocab').tag(config=True)

    raw_lookup_path = Unicode(help='Raw Lookup Vocab.').tag(config=True)

    def __init__(self, **kwargs):
        super(Resources, self).__init__(**kwargs)
        self.event_embedding = np.load(self.event_embedding_path)
        self.word_embedding = np.load(self.word_embedding_path)

        self.event_vocab, self.event_count = self.__read_vocab(
            self.event_vocab_path)
        logging.info("%d events in vocabulary." % len(self.event_vocab))

        self.word_vocab, self.word_count = self.__read_vocab(
            self.word_vocab_path)
        logging.info("%d words in vocabulary." % len(self.word_vocab))

        self.lookups, self.oovs = load_vocab(self.raw_lookup_path)
        logging.info("Loaded lookup maps and oov words.")

    def __read_vocab(self, vocab_file):
        vocab = {}
        counts = []
        with open(vocab_file) as din:
            index = 0
            for line in din:
                word, count = line.split()
                vocab[word] = index
                counts.append(count)
                index += 1
        return vocab, counts
