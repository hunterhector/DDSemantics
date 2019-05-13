from event.io.dataset.base import (
    Span,
    DataLoader,
    DEDocument,
    Corpus,
)
import json
import os
import sys
from event.arguments.prepare.event_vocab import EmbbedingVocab

from traitlets.config import Configurable
from traitlets import (
    Int,
    List,
    Unicode,
)
from event.util import load_mixed_configs, set_basic_log, ensure_dir

import logging
import numpy as np
from collections import defaultdict
import pprint
from scipy.spatial.distance import cosine

class ZeroShotEventResources(Configurable):
    """
    Resource class.
    """
    event_embedding_path = Unicode(help='Event Embedding path').tag(config=True)
    word_embedding_path = Unicode(help='Word Embedding path').tag(config=True)

    event_vocab_path = Unicode(help='Event Vocab').tag(config=True)
    word_vocab_path = Unicode(help='Word Vocab').tag(config=True)

    target_ontology = Unicode(help='Ontology path').tag(config=True)

    def __init__(self, **kwargs):
        super(ZeroShotEventResources, self).__init__(**kwargs)

        self.event_embedding = np.load(self.event_embedding_path)
        self.word_embedding = np.load(self.word_embedding_path)

        self.event_embed_vocab = EmbbedingVocab(self.event_vocab_path)
        self.word_embed_vocab = EmbbedingVocab(self.word_vocab_path)

        with open(self.target_ontology) as onto_file:
            self.ontology = json.load(onto_file)

        logging.info(
            f"{len(self.event_embed_vocab.vocab)} events in embedding.")

        logging.info(
            f"{len(self.word_embed_vocab.vocab)} words in embedding."
        )


def camel_split(s, lower=True):
    l_s = [[]]

    for c in s:
        if c.isupper():
            l_s.append([])
        l_s[-1].append(c)

    if lower:
        return [''.join(l).lower() for l in l_s if len(l) > 0]
    else:
        return [''.join(l) for l in l_s]


def event_type_split(s):
    split_map = {
        'Transferownership': ['transfer', 'ownership'],
        'Transfermoney': ['transfer', 'money'],
        'Transportperson': ['transport', 'person'],
        'TransportArtifact': ['transport'],
        'Startposition': ['employ'],
        'Endposition': ['resign'],
        'Arrestjail': ['arrest', 'jail'],
        'Chargeindict': ['charge', 'indict'],
        'Trialhearing': ['trial', 'hearing'],
        'ReleaseParole': ['release', 'parole'],
        'Declarebankruptcy': ['bankruptcy'],
        'StartOrg': ['start'],
        'EndOrg': ['end'],
        'MergeOrg': ['merge'],
        'BeBorn': ['born'],
    }

    if s in split_map:
        return split_map[s]
    else:
        return camel_split(s)


class ZeroShotTypeMapper:
    def __init__(self, resources):
        self.resources = resources
        # A map from the ontology event type to its tokens
        self.onto_event_tokens = {}
        self.onto_arg_role_tokens = {}

        self.onto_arg_domain = defaultdict(list)

        # From a fine-grained to a parent
        self.type_parent = {}
        # The embedding of the predicates
        self.pred_embeds = {}

        self.tokenize_ontology()

    def tokenize_ontology(self):
        skips = {
            'artifact',
            'in',
            'person',
            'start',
            'end',
            'life',
            'Existence',
        }

        nom_map = {
            'correspondence': 'correspond',
            'prevarication': 'prevaricate',
            'gathering': 'gather',
            'agreements': 'agreement',
            'degradation': 'degrade',
            'movement': 'move',
            'hiring': 'hire',
            'injury': 'injure',
            'stabbing': 'stab',
        }

        for frame in self.resources.ontology['frames']:
            onto_category = frame['@type']
            if onto_category == 'event_type':
                event_type = frame['@id']
                self.onto_event_tokens[event_type] = {}

                level_types = event_type.split(':')[1].split('.')

                tokenized_types = []
                for lt in level_types:
                    tokenized_types.append([])
                    for t in camel_split(lt):
                        if t not in skips:
                            if t in nom_map:
                                t = nom_map[t]
                            pred_id = self.resources.event_embed_vocab\
                                .get_index(t + '-pred', None)
                            if pred_id >= 0:
                                tokenized_types[-1].append(t)
                                pred_vector = self.resources.event_embedding[
                                    pred_id]
                                self.pred_embeds[t] = pred_vector
                            else:
                                logging.debug(f"Predicate form for {t} not "
                                              f"found")

                self.onto_event_tokens[event_type]['top'] = tokenized_types[0]
                if len(level_types) > 2:
                    self.onto_event_tokens[event_type]['middle'] = tokenized_types[1]
                    self.onto_event_tokens[event_type]['low'] = tokenized_types[2]

                    if not frame['subClassOf'] == 'aidaDC:EventType':
                        self.type_parent[event_type] = frame['subClassOf']

            if onto_category == 'event_argument_role_type':
                role_type = frame['@id']
                role_token = role_type.split('_')[-1]
                self.onto_arg_role_tokens[role_type] = [role_token.lower()]
                self.onto_arg_domain[frame['domain']].append(role_type)

    # def map_arg_role_from_sim(self, event_type, arg_types, arg_lemma):
    #     arg_candidates = self.onto_arg_domain[event_type]
    #
    #     lemma_idx = self.resources.word_embed_vocab.get_index(arg_lemma, None)
    #     if lemma_idx >= 0:
    #         lemma_emd = self.resources.word_embedding[lemma_idx]
    #
    #     max_type_score = 0
    #     max_lemma_score = 0
    #
    #     best_arg_from_type = None
    #     best_arg_from_lemma = None
    #
    #     for full_arg in arg_candidates:
    #         for arg_token in self.onto_arg_role_tokens[full_arg]:
    #             a = self.resources.word_embed_vocab.get_index(arg_token, None)
    #             if a >= 0:
    #                 arg_token_word_embed = self.resources.word_embedding[a]
    #
    #             for arg_type in arg_types:
    #                 if arg_type.startswith('fn:'):
    #                     t = self.resources.word_embed_vocab.get_index(
    #                         arg_type.split(':')[1], None)
    #                     if t >= 0:
    #                         t_word_embed = self.resources.word_embedding[t]
    #                         if a >= 0:
    #                             s = 1 - cosine(arg_token_word_embed,
    #                                            t_word_embed)
    #                             if s > max_type_score:
    #                                 max_type_score = s
    #                                 best_arg_from_type = full_arg
    #                         if lemma_idx >= 0:
    #                             s = 1 - cosine(lemma_emd, t_word_embed)
    #                             if s > max_lemma_score:
    #                                 max_lemma_score = s
    #                                 best_arg_from_lemma = full_arg

    def token_direct(self, lemma):
        token_direct_map = {
            'seize': 'ldcOnt:Transaction.Transaction.TransferControl',
            'casualty': 'ldcOnt:Life.Die.DeathCausedByViolentEvents',
        }

        if lemma in token_direct_map:
            return token_direct_map[lemma]

    def arg_direct(self, content):
        pass

    def map_from_event_type(self, event_type, lemma):
        level1, level2 = event_type.split('_')
        level2_tokens = event_type_split(level2)
        l_score, m_score, full_type = self.map_by_pred_match(
            [t + '-pred' for t in level2_tokens], [lemma + '-pred'])
        if m_score > 0.8 or l_score > 0.8:
            if l_score > 0.8:
                return full_type
            else:
                if full_type in self.type_parent:
                    return self.type_parent[full_type]
                else:
                    return full_type

    def map_from_lemma_only(self, lemma):
        l_score, m_score, full_type = self.map_by_pred_match(
            [lemma + '-pred'], [lemma + '-pred'])

        if m_score > 0.8 or l_score > 0.8:
            if l_score > 0.8:
                return full_type
            else:
                if full_type in self.type_parent:
                    return self.type_parent[full_type]
                else:
                    return full_type

    def map_from_frame(self, frame, lemma):
        frame_direct = {
            'Arriving': 'ldcOnt:Movement.TransportPerson',
            'Employing': 'ldcOnt:Personnel.StartPosition',
            'Shoot_projectiles': 'ldcOnt:Conflict.Attack.AirstrikeMissileStrike',
            'Communication_response': 'ldcOnt:Contact.Discussion',
            'Chatting': 'ldcOnt:Contact.Discussion',
            'Hostile_encounter': 'ldcOnt:Conflict.Attack',
        }

        if frame in frame_direct:
            return frame_direct[frame]

        lemma_pred = lemma + '_pred'
        l_score, m_score, full_type = self.map_by_pred_match(
            [frame],[frame, lemma_pred]
        )

        if m_score > 0.7 or l_score > 0.7:
            if l_score > 0.7:
                return full_type
            else:
                if full_type in self.type_parent:
                    return self.type_parent[full_type]
                else:
                    return full_type

    def map_by_pred_match(self, middle_matchers, low_matchers):
        scored_pairs = []

        for onto_type, onto_type_tokens in self.onto_event_tokens.items():
            middle_score = 0
            low_score = 0

            mid_tokens = onto_type_tokens.get('middle', []) + \
                         onto_type_tokens.get('top', [])

            for onto_t in mid_tokens:
                if onto_t in self.pred_embeds:
                    onto_emd = self.pred_embeds[onto_t]
                    for t in middle_matchers:
                        t_id = self.resources.event_embed_vocab.get_index(
                            t, None)
                        if t_id >= 0:
                            t_emd = self.resources.event_embedding[t_id]
                            s = 1 - cosine(onto_emd, t_emd)

                            if s > middle_score:
                                middle_score = s

            low_tokens = onto_type_tokens.get('low', [])
            for onto_t in low_tokens:
                if onto_t in self.pred_embeds:
                    onto_emd = self.pred_embeds[onto_t]
                    for lemma_pred in low_matchers:
                        lemma_idx = self.resources.event_embed_vocab.get_index(
                            lemma_pred, None
                        )

                        if lemma_idx >= 0:
                            emd = self.resources.event_embedding[lemma_idx]
                            s = 1 - cosine(onto_emd, emd)
                            if s > low_score:
                                low_score = s

            scored_pairs.append((low_score, middle_score, onto_type))

        scored_pairs.sort(reverse=True)
        return scored_pairs[0]

    def map_event_type(self, event):
        direct_type = self.token_direct(event['headLemma'])
        if direct_type:
            return direct_type

        arg_type = self.arg_direct(event)
        if arg_type:
            return arg_type

        if event['component'] == 'CrfMentionTypeAnnotator':
            r = self.map_from_event_type(event['type'], event['headLemma'])
            if r:
                return r

        if 'frame' in event:
            r = self.map_from_frame(event['frame'], event['headLemma'])
            if r:
                return r

        if event['component'] == 'VerbBasedEventDetector':
            r = self.map_from_lemma_only(event['headLemma'])
            if r:
                return r



def main(para, resources):
    type_mapper = ZeroShotTypeMapper(resources)

    if not os.path.exists(para.output_path):
        os.makedirs(para.output_path)

    for p in os.listdir(para.input_path):
        if not p.endswith('.json'):
            continue
        with open(os.path.join(para.input_path, p)) as fin, \
                open(os.path.join(para.output_path, p), 'w') as fout:
            rich_doc = json.load(fin)
            text = rich_doc['text']

            entities = {}
            for ent in rich_doc['entityMentions']:
                entities[ent['id']] = {
                    'span': ent['span'],
                    "head_span": ent['headWord']['span'],
                    "text": ent['text'],
                    "lemma": ent['headWord']['lemma'],
                    'id': ent['id'],
                }

            for evm in rich_doc['eventMentions']:
                mapped_type = type_mapper.map_event_type(evm)
                if mapped_type:
                    evm['type'] = mapped_type
                    # for arg in evm['arguments']:
                    #     arg_lemma = entities[arg['entityId']]['lemma']
                    #     type_mapper.map_arg_role_from_sim(
                    #         evm['type'], arg['roles'], arg_lemma)

            json.dump(rich_doc, fout)


if __name__ == '__main__':
    class Basic(Configurable):
        input_path = Unicode(help='Input path.').tag(config=True)
        output_path = Unicode(help='Output path.').tag(config=True)


    set_basic_log()
    conf = load_mixed_configs()
    basic_para = Basic(config=conf)
    res = ZeroShotEventResources(config=conf)

    main(basic_para, res)
