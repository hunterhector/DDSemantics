from traitlets.config import Configurable
from traitlets import (
    Int,
    List,
    Unicode,
)
import numpy as np
import logging
from event.arguments.prepare.event_vocab import TypedEventVocab
from event.arguments.prepare.event_vocab import EmbbedingVocab

logger = logging.getLogger(__name__)

class ImplicitArgResources(Configurable):
    """
    Resource class.
    """
    raw_corpus_name = Unicode(help='Raw corpus name').tag(config=True)
    event_embedding_path = Unicode(help='Event Embedding path').tag(config=True)
    word_embedding_path = Unicode(help='Word Embedding path').tag(config=True)

    event_vocab_path = Unicode(help='Event Vocab').tag(config=True)
    word_vocab_path = Unicode(help='Word Vocab').tag(config=True)

    raw_lookup_path = Unicode(help='Raw Lookup Vocab.').tag(config=True)

    def __init__(self, **kwargs):
        super(ImplicitArgResources, self).__init__(**kwargs)
        self.event_embedding = np.load(self.event_embedding_path)
        self.word_embedding = np.load(self.word_embedding_path)

        self.event_embed_vocab = EmbbedingVocab(self.event_vocab_path)
        self.word_embed_vocab = EmbbedingVocab(self.word_vocab_path)
        self.word_embed_vocab.add_extra('__word_pad__')

        self.predicate_count = self.count_predicates(self.event_vocab_path)

        logger.info(
            f"{len(self.event_embed_vocab.vocab)} events in embedding.")

        logger.info(
            f"{len(self.word_embed_vocab.vocab)} words in embedding."
        )

        self.typed_event_vocab = TypedEventVocab(self.raw_lookup_path)
        logger.info("Loaded typed vocab, including oov words.")

    @staticmethod
    def count_predicates(vocab_file):
        pred_count = 0
        with open(vocab_file) as din:
            for line in din:
                word, count = line.split()
                if word.endswith('-pred'):
                    pred_count += int(count)
        return pred_count
