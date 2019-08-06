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
import xml.etree.ElementTree as ET
import os

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

    nombank_arg_slot_map = Unicode(
        help='A map from predicate and slots to the prespective dependency '
             'type.').tag(config=True)
    framenet_frame_path = Unicode(
        help='The path that stores FrameNet frames.'
    ).tag(config=True)

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

        if self.nombank_arg_slot_map:
            self.nombank_slots = load_nombank_dep_map(
                self.nombank_arg_slot_map, self.typed_event_vocab)

        if self.framenet_frame_path:
            self.framenet_slots = load_framenet_slots(
                self.framenet_frame_path, self.event_embed_vocab)

    @staticmethod
    def count_predicates(vocab_file):
        pred_count = 0
        with open(vocab_file) as din:
            for line in din:
                word, count = line.split()
                if word.endswith('-pred'):
                    pred_count += int(count)
        return pred_count


def load_framenet_slots(framenet_path, event_emb_vocab):
    frame_slots = {}

    for frame_file in os.listdir(framenet_path):
        with open(os.path.join(framenet_path, frame_file)) as frame_data:
            tree = ET.parse(frame_data)
            root = tree.getroot()

            print(root)

            frame = root.attrib['name']
            fid = event_emb_vocab.get_index(frame, None)
            print(frame, fid)

            print(root.children)
            all_fes = []
            for fe_node in root:
                fe = fe_node.attrib['name']
                print(fe)

                all_fes.append(fe)

            frame_slots[frame] = all_fes

    print(frame_slots)
    input('loading is done.')

    return frame_slots


def load_nombank_dep_map(nombank_map_path, typed_event_vocab):
    slot_names = ['arg0', 'arg1', 'arg2', 'arg3', 'arg4']

    nombank_map = {}
    with open(nombank_map_path) as map_file:
        for line in map_file:
            if not line.startswith('#'):
                fields = line.strip().split()
                noun, verb = fields[0:2]

                pred = typed_event_vocab.get_pred_rep(
                    {'predicate': noun, 'verb_form': verb})

                key_values = zip(slot_names, fields[2:])

                nombank_map[pred] = {
                    'verb': verb,
                    'noun': noun,
                    'slots': dict(key_values)
                }

    return nombank_map
