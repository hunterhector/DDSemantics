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
    """Resource class."""

    raw_corpus_name = Unicode(help="Raw corpus name").tag(config=True)
    event_embedding_path = Unicode(help="Event Embedding path").tag(config=True)
    word_embedding_path = Unicode(help="Word Embedding path").tag(config=True)

    event_vocab_path = Unicode(help="Event Vocab").tag(config=True)
    word_vocab_path = Unicode(help="Word Vocab").tag(config=True)

    raw_lookup_path = Unicode(help="Raw Lookup Vocab.").tag(config=True)

    min_vocab_count = Int(help="The min vocab cutoff threshold.", default_value=50).tag(
        config=True
    )

    def __init__(self, **kwargs):
        super(ImplicitArgResources, self).__init__(**kwargs)
        self.event_embedding = np.load(self.event_embedding_path)
        self.word_embedding = np.load(self.word_embedding_path)

        # Add padding and two unk to the vocab.
        self.event_embed_vocab = EmbbedingVocab.with_extras(self.event_vocab_path)

        # Add padding to the vocab.
        self.word_embed_vocab = EmbbedingVocab(self.word_vocab_path, True)

        self.predicate_count = self.count_predicates(self.event_vocab_path)

        logger.info(f"{len(self.event_embed_vocab.vocab)} events in embedding.")

        logger.info(f"{len(self.word_embed_vocab.vocab)} words in embedding.")

        self.typed_event_vocab = TypedEventVocab(self.raw_lookup_path)
        logger.info("Loaded typed vocab, including oov words.")

        hash_params = HashParam(**kwargs)

        self.slot_handler = SlotHandler(hash_params)

        self.h_nom_dep_map, self.h_nom_slots = self.hash_nom_mappings()
        self.h_frame_dep_map, self.h_frame_slots = self.hash_frame_mappings()

    @staticmethod
    def count_predicates(vocab_file):
        pred_count = 0
        with open(vocab_file) as din:
            for line in din:
                word, count = line.split()
                if word.endswith("-pred"):
                    pred_count += int(count)
        return pred_count

    def hash_frame_mappings(self):
        """Hash the frame mapping, and map the frames to the most frequent
        dependency.
        :return:

        Args:

        Returns:

        """
        h_frame_dep_map = {}
        frame_deps = self.slot_handler.frame_deps

        for (frame, fe), pred_deps in frame_deps.items():
            fid = self.event_embed_vocab.get_index(frame, None)
            for pred, dep, count in pred_deps:
                pred_id = self.event_embed_vocab.get_index(
                    self.typed_event_vocab.get_pred_rep({"predicate": pred}), None
                )
                if (fid, fe, pred_id) not in h_frame_dep_map:
                    # Map to the most frequent dependency type.
                    h_frame_dep_map[(fid, fe, pred_id)] = dep

        frame_prior = self.slot_handler.frame_priority

        h_frame_slots = {}

        for frame_name, fes in frame_prior.items():
            fid = self.event_embed_vocab.get_index(
                frame_name, self.typed_event_vocab.unk_frame
            )
            h_frame_slots[fid] = set()

            for fe in fes:
                fe_name = fe["fe_name"]
                fe_id = self.event_embed_vocab.get_index(
                    self.typed_event_vocab.get_fe_rep(frame_name, fe_name),
                    self.typed_event_vocab.unk_fe,
                )
                h_frame_slots[fid].add(fe_id)

        return h_frame_dep_map, h_frame_slots

    def hash_nom_mappings(self):
        """The mapping information in the slot handler are string based, we
        convert them to the hashed version for easy reading.

        Args:

        Returns:

        """

        def prop_to_index(argx):
            r = argx.lower()
            if r[3] == "M":
                return 4
            else:
                return int(r[3])
            #

        # This nombank mapping is the one hand-crafted, contains the 10
        #  nombank predicates.
        nom_map = self.slot_handler.nombank_mapping
        predicate_slots = {}
        nom_dep_map = {}
        for nom, (verb_form, arg_map) in nom_map.items():
            pred_id = self.typed_event_vocab.get_pred_rep(
                {"predicate": nom, "verb_form": verb_form}
            )

            predicate_slots[pred_id] = []

            for arg_role, dep in arg_map.items():
                if not dep == "-":
                    arg_index = prop_to_index(arg_role)
                    predicate_slots[pred_id].append(arg_index)
                    nom_dep_map[(pred_id, arg_index)] = dep

        # This mapping is automatically gathered from data, mapping from the
        #  verb and proposition to the dependency.
        prop_deps = self.slot_handler.prop_deps

        # TODO: Here, we should read the prop dep data by converting the
        #   predicates into nomninals, by using the nom->verb mapping
        # The nom-> verb mapping

        for (verb, prop_role), dep in prop_deps.items():
            if verb in self.slot_handler.verb_nom_form:
                nom = self.slot_handler.verb_nom_form[verb]
                pred_id = self.typed_event_vocab.get_pred_rep(
                    {"predicate": nom, "verb_form": verb}
                )

                arg_index = prop_to_index(prop_role)
                nom_dep_map[(pred_id, arg_index)] = dep

        return nom_dep_map, predicate_slots


def load_framenet_slots(framenet_path, event_emb_vocab):
    frame_slots = {}

    ns = {"fn": "http://framenet.icsi.berkeley.edu"}

    num_unseen = 0

    for frame_file in os.listdir(framenet_path):
        if not frame_file.endswith(".xml"):
            continue

        with open(os.path.join(framenet_path, frame_file)) as frame_data:
            tree = ET.parse(frame_data)
            root = tree.getroot()

            frame = root.attrib["name"]
            fid = event_emb_vocab.get_index(frame, None)

            all_fes = []
            for fe_node in root.findall("fn:FE", ns):
                fe = fe_node.attrib["name"]
                all_fes.append(fe.lower())

            if not fid == -1:
                frame_slots[fid] = all_fes
            else:
                num_unseen += 1

    logging.info(
        f"Loaded {len(frame_slots)} frames, {num_unseen} frames are "
        f"not seen in the parsed dataset."
    )

    return frame_slots


def load_nombank_dep_map(nombank_map_path, typed_event_vocab):
    slot_names = ["arg0", "arg1", "arg2", "arg3", "arg4"]

    nombank_map = {}
    with open(nombank_map_path) as map_file:
        for line in map_file:
            if not line.startswith("#"):
                fields = line.strip().split()
                noun, verb = fields[0:2]

                pred = typed_event_vocab.get_pred_rep(
                    {"predicate": noun, "verb_form": verb}
                )

                key_values = zip(slot_names, fields[2:])

                nombank_map[pred] = {
                    "verb": verb,
                    "noun": noun,
                    "slots": dict(key_values),
                }

    logging.info("Loaded Nombank frame mapping.")

    return nombank_map
