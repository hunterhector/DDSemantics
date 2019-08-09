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
from event.arguments.prepare.hash_cloze_data import HashParam
from event.arguments.prepare.hash_cloze_data import SlotHandler

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

    min_vocab_count = Int(help='The min vocab cutoff threshold.',
                          default_value=50).tag(config=True)

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

        hash_params = HashParam(**kwargs)

        self.slot_handler = SlotHandler(hash_params.frame_files,
                                        hash_params.frame_dep_map,
                                        hash_params.dep_frame_map,
                                        hash_params.nom_map)

        hash_mappings()

        # print(slot_handler.nombank_mapping)
        #
        # for f, slot in slot_handler.frame_priority.items():
        #     print(f)
        #     print(slot)
        #     break
        #
        # for k in slot_handler.frame_deps:
        #     print(k)
        #     break
        #
        # for k in slot_handler.dep_frames:
        #     print(k)
        #     break
        #
        # print(slot_handler.frame_deps.get(("Bringing", "Carrier"), []))
        #
        # most_freq_dep = slot_handler.get_most_freq_dep('take', 'Bringing',
        #                                                'Carrier')
        # print(most_freq_dep)
        #
        # input('check slot handler')
        #
        # if self.nombank_arg_slot_map:
        #     self.nombank_slots = load_nombank_dep_map(
        #         self.nombank_arg_slot_map, self.typed_event_vocab)
        #
        # if self.framenet_frame_path:
        #     self.framenet_slots = load_framenet_slots(
        #         self.framenet_frame_path, self.event_embed_vocab)

    @staticmethod
    def count_predicates(vocab_file):
        pred_count = 0
        with open(vocab_file) as din:
            for line in din:
                word, count = line.split()
                if word.endswith('-pred'):
                    pred_count += int(count)
        return pred_count


def hash_mappings():
    """
    The mapping information in the slot handler are string based, we convert
    them to the hashed version for easy reading.
    :return:
    """
    self.slot_handler.


def load_framenet_slots(framenet_path, event_emb_vocab):
    frame_slots = {}

    ns = {'fn': 'http://framenet.icsi.berkeley.edu'}

    num_unseen = 0

    for frame_file in os.listdir(framenet_path):
        if not frame_file.endswith('.xml'):
            continue

        with open(os.path.join(framenet_path, frame_file)) as frame_data:
            tree = ET.parse(frame_data)
            root = tree.getroot()

            frame = root.attrib['name']
            fid = event_emb_vocab.get_index(frame, None)

            all_fes = []
            for fe_node in root.findall('fn:FE', ns):
                fe = fe_node.attrib['name']
                all_fes.append(fe.lower())

            if not fid == -1:
                frame_slots[fid] = all_fes
            else:
                num_unseen += 1

    logging.info(f"Loaded {len(frame_slots)} frames, {num_unseen} frames are "
                 f"not seen in the parsed dataset.")

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
                    {'predicate': noun, 'verb_form': verb}
                )

                key_values = zip(slot_names, fields[2:])

                nombank_map[pred] = {
                    'verb': verb,
                    'noun': noun,
                    'slots': dict(key_values)
                }

    logging.info("Loaded Nombank frame mapping.")

    return nombank_map
