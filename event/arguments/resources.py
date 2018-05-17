from traitlets.config import Configurable
from traitlets import (
    Int,
    List,
    Unicode,
)
import numpy as np


class Resources(Configurable):
    """
    Resource class.
    """
    event_embedding_path = Unicode(help='Event Embedding path').tag(config=True)
    word_embedding_path = Unicode(help='Word Embedding path').tag(config=True)

    event_vocab_path = Unicode(help='Event Vocab').tag(config=True)
    word_vocab_path = Unicode(help='Word Vocab').tag(config=True)

    def __init__(self, **kwargs):
        super(Resources, self).__init__(**kwargs)

        self.event_embedding = None
        self.word_embedding = None

    def __load(self):
        if self.event_embedding_path:
            self.event_embedding = np.load(self.event_embedding_path)

        if self.word_embedding_path:
            self.word_embedding = np.load(self.word_embedding_path)
