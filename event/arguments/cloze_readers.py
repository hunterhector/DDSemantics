import os
import logging
from collections import defaultdict
import torch
import numpy as np
import pickle
import random
import json
from collections import Counter
from event.arguments import consts
from event.io.io_utils import pad_2d_list
from event.arguments.util import (batch_combine, to_torch)
import event.arguments.prepare.event_vocab as vocab_util
import math
from event import torch_util
from event.arguments.prepare.create_argument_training import (
    hash_arg,
    hash_context,
    read_entity_features,
    hash_arg_role,
)


class ClozeSampler:
    def __init__(self):
        pass

    def sample_cross(self, arg_entities, evm_id, ent_id):

        remaining = []
        for evm, ent, text in arg_entities:
            if not (evm == evm_id or ent == ent_id):
                remaining.append((evm, ent, text))

        if len(remaining) > 0:
            return random.choice(remaining)
        else:
            return None

    def sample_ignore_item(self, data, ignored_item):
        """
        Sample one item in the list, but ignore a provided one. If the list
        contains less than 2 elements, nothing will be sampled.
        :param data:
        :param ignored_item:
        :return:
        """
        if len(data) <= 1:
            return None
        while True:
            sampled_item = random.choice(data)
            if not sampled_item == ignored_item:
                break
        return sampled_item


class HashedClozeReader:
    def __init__(self, resources, batch_size, from_hashed=True,
                 multi_context=False, max_events=200, max_cloze=150, gpu=True):
        """
        Reading the hashed dataset into cloze tasks.
        :param resources: Resources containing vocabulary and more.
        :param batch_size: Number of cloze tasks per batch.
        :param from_hashed: Reading form hashed json file.
        :param multi_context: Whether to use multiple events as context.
        :param max_events: Number of events to keep per document.
        :param max_cloze: Max cloze to keep per document.
        :param gpu: Whethere to run on gpu.
        """
        self.batch_size = batch_size
        self.multi_context = multi_context
        self.max_events = max_events
        self.max_instance = max_cloze
        self.event_vocab = resources.event_vocab
        self.event_count = resources.term_freq
        self.pred_counts = resources.typed_count['predicate']
        self.from_hashed = from_hashed

        self.lookups = resources.lookups
        self.oovs = resources.oovs

        self.event_inverted = self.__inverse_vocab(self.event_vocab)

        self.event_padding = len(self.event_vocab)

        self.slot_names = ['subj', 'obj', 'prep', ]

        self.__data_types = {
            'context': np.int64,
            'event_indices': np.int64,
            'slot_indices': np.float32,
            'rep': np.int64,
            'distances': np.float32,
            'features': np.float32,
        }

        self.__data_dim = {
            'context': 2,
            'event_indices': 1,
            'slot_indices': 1,
            'rep': 2,
            'distances': 2,
            'features': 2,
        }

        self.device = torch.device(
            "cuda" if gpu and torch.cuda.is_available() else "cpu"
        )

        self.sampler = ClozeSampler()

        # Fix seed to generate the same instances.
        random.seed(17)

    def __inverse_vocab(self, vocab):
        inverted = [0] * len(vocab)
        for w, index in vocab.items():
            inverted[index] = w
        return inverted

    def __batch_pad(self, key, data, pad_size):
        dim = self.__data_dim[key]

        if dim == 2:
            return [pad_2d_list(v, pad_size) for v in data]
        elif dim == 1:
            return pad_2d_list(data, pad_size, dim=1)
        else:
            raise ValueError("Dimension unsupported %d" % dim)

    def create_batch(self, b_common_data, b_instance_data,
                     max_context_size, max_instance_size):
        instance_data = defaultdict(dict)
        common_data = {}

        sizes = {}

        for key, value in b_common_data.items():
            if key == 'context':
                padded = self.__batch_pad(key, value, max_context_size)
                vectorized = to_torch(padded, self.__data_types[key])
                common_data['context'] = batch_combine(
                    vectorized, self.device
                )
            else:
                padded = self.__batch_pad(key, value, max_instance_size)
                vectorized = to_torch(padded, self.__data_types[key])
                common_data[key] = batch_combine(vectorized, self.device)

            sizes[key] = len(padded)

        for ins_type, ins_data in b_instance_data.items():
            for key, value in ins_data.items():
                padded = self.__batch_pad(key, value, max_instance_size)
                vectorized = to_torch(padded, self.__data_types[key])
                instance_data[ins_type][key] = batch_combine(
                    vectorized, self.device
                )
                sizes[key] = len(padded)

        b_size = sizes['context']

        for key, s in sizes.items():
            assert s == b_size

        return instance_data, common_data, b_size

    def read_cloze_batch(self, data_in, from_line=None, until_line=None):
        b_common_data = defaultdict(list)
        b_instance_data = defaultdict(lambda: defaultdict(list))
        batch_predicates = []

        max_context_size = 0
        max_instance_size = 0
        doc_count = 0

        def _clear():
            # Clear batch data.
            nonlocal b_common_data, b_instance_data, max_context_size
            nonlocal max_instance_size, doc_count, batch_predicates

            # Reset counts.
            batch_predicates.clear()
            b_common_data.clear()
            b_instance_data.clear()

            doc_count = 0
            max_context_size = 0
            max_instance_size = 0

        for instance_data, common_data in self.parse_docs(
                data_in, from_line, until_line):

            gold_event_data = instance_data['gold']
            predicate_text = []
            for gold_rep in gold_event_data['rep']:
                eid = gold_rep[0]
                if not eid == len(self.event_inverted):
                    text = self.event_inverted[eid]
                    predicate_text.append(text)
            batch_predicates.append(predicate_text)

            for key, value in common_data.items():
                b_common_data[key].append(value)
                if key == 'context':
                    if len(value) > max_context_size:
                        max_context_size = len(value)

                if key == 'event_indices':
                    if len(value) > max_instance_size:
                        max_instance_size = len(value)

            for ins_type, ins_data in instance_data.items():
                for key, value in ins_data.items():
                    b_instance_data[ins_type][key].append(value)

            doc_count += 1

            # Each document is a full instance since we do multiple product at
            # once.
            if doc_count == self.batch_size:
                yield self.create_batch(b_common_data, b_instance_data,
                                        max_context_size, max_instance_size)
                _clear()

        # Yield the remaining data.
        yield self.create_batch(b_common_data, b_instance_data,
                                max_context_size, max_instance_size)
        _clear()

    def _take_event_parts(self, event_info):
        event_components = [event_info['predicate'], event_info['frame']]

        for slot_name in self.slot_names:
            slot = event_info['args'][slot_name]
            event_components.append(slot.get('fe', self.event_padding))
            event_components.append(slot.get('arg_role', self.event_padding))

        return [c if c > 0 else self.event_padding for c in event_components]

    def subsample_pred(self, pred):
        pred_tf = self.event_count[pred]
        freq = 1.0 * pred_tf / self.pred_counts

        if freq > consts.sample_pred_threshold:
            if pred_tf > consts.sample_pred_threshold:
                rate = consts.sample_pred_threshold / freq
                if random.random() < 1 - rate - math.sqrt(rate):
                    return False
        return True

    def parse_hashed(self):
        pass

    def parse_origin(self):
        pass

    def collect_features(self, doc_info):
        # Collect all features.
        features_by_eid = {}
        entity_heads = {}

        for eid, content in doc_info['entities'].items():
            features_by_eid[int(eid)] = content['features']
            entity_heads[int(eid)] = content['entity_head']

        return features_by_eid, entity_heads

    def parse_docs(self, data_in, from_line=None, until_line=None):

        linenum = 0
        for line in data_in:
            linenum += 1

            if from_line and linenum <= from_line:
                continue

            if until_line and linenum > until_line:
                break

            doc_info = json.loads(line)
            features_by_eid, entity_heads = self.collect_features(doc_info)

            # Organize all the arguments.
            event_data = []

            entity_mentions = defaultdict(list)
            arg_entities = set()
            eid_count = Counter()

            for evm_index, event in enumerate(doc_info['events']):
                if evm_index == self.max_events:
                    break

                event_data.append(event)
                sentence_id = event.get('sentence_id', None)

                arg_info = {}
                for slot, arg in event['args'].items():
                    if not arg or arg['entity_id'] == -1:
                        arg_info[slot] = {}
                    else:
                        # Argument for nth event, at slot position 'slot'.
                        eid = arg['entity_id']

                        # From eid to entity information.
                        entity_mentions[eid].append(
                            (evm_index, slot, sentence_id)
                        )
                        arg_entities.add((evm_index, eid, arg['text']))
                        eid_count[eid] += 1

            all_event_reps = [self._take_event_parts(e) for e in event_data]
            arg_entities = list(arg_entities)

            if len(arg_entities) <= 1:
                # There no enough arguments to sample from.
                continue

            # We current sample the predicate based on unigram distribution.
            # A better learning strategy is to select one
            # cross instance that is difficult. We can have two
            # strategies here:
            # 1. Again use unigram distribution to sample items.
            # 2. Select items based on classifier output.

            gold_event_data = {'rep': [], 'distances': [], 'features': []}
            cross_event_data = {'rep': [], 'distances': [], 'features': []}
            inside_event_data = {'rep': [], 'distances': [], 'features': []}
            cloze_event_indices = []
            cloze_slot_indices = []

            for evm_index, event_info in enumerate(event_data):
                event_info = event_data[evm_index]

                pred = event_info['predicate']
                if pred == self.event_vocab[consts.unk_predicate]:
                    continue

                keep_pred = self.subsample_pred(pred)

                if not keep_pred:
                    # Too frequent word will be ignore.
                    continue

                # import pprint
                # pp = pprint.PrettyPrinter(indent=2)

                current_sent = event_info['sentence_id']

                for slot_index, slot in enumerate(self.slot_names):
                    arg = event_info['args'][slot]

                    correct_id = arg.get('entity_id', -1)

                    # Apparently, singleton and unks are not resolvable.
                    if correct_id < 0:
                        continue

                    if eid_count[correct_id] <= 1:
                        continue

                    if entity_heads[correct_id] == consts.unk_arg_word:
                        continue

                    cross_sample = self.cross_cloze(
                        event_data[evm_index]['args'], arg_entities,
                        evm_index, slot
                    )

                    if not cross_sample:
                        continue

                    inside_sample = self.inside_cloze(
                        event_data[evm_index]['args'], slot
                    )

                    if not inside_sample:
                        continue

                    cross_args, cross_filler_id = cross_sample
                    cross_info = self.update_arguments(
                        doc_info, evm_index, cross_args)
                    cross_rep = self._take_event_parts(cross_info)
                    cross_features = features_by_eid[cross_filler_id]
                    cross_distances = self.compute_distances(
                        evm_index, entity_mentions, cross_info, current_sent,
                    )

                    inside_args, inside_filler_id = inside_sample
                    inside_info = self.update_arguments(
                        doc_info, evm_index, inside_args)
                    inside_rep = self._take_event_parts(inside_info)
                    inside_features = features_by_eid[inside_filler_id]
                    inside_distances = self.compute_distances(
                        evm_index, entity_mentions, inside_info, current_sent,
                    )

                    gold_rep = self._take_event_parts(event_info)
                    gold_features = features_by_eid[correct_id]
                    gold_distances = self.compute_distances(
                        evm_index, entity_mentions, event_info, current_sent,
                    )

                    cloze_event_indices.append(evm_index)
                    cloze_slot_indices.append(slot_index)

                    cross_event_data['rep'].append(cross_rep)
                    cross_event_data['features'].append(cross_features)
                    cross_event_data['distances'].append(cross_distances)

                    inside_event_data['rep'].append(inside_rep)
                    inside_event_data['features'].append(inside_features)
                    inside_event_data['distances'].append(inside_distances)

                    gold_event_data['rep'].append(gold_rep)
                    gold_event_data['features'].append(gold_features)
                    gold_event_data['distances'].append(gold_distances)

            if len(cloze_slot_indices) == 0:
                # This document do not contains training instance.
                continue

            instance_data = {
                'gold': gold_event_data,
                'cross': cross_event_data,
                'inside': inside_event_data,
            }

            common_data = {
                'context': all_event_reps,
                'event_indices': cloze_event_indices,
                'slot_indices': cloze_slot_indices,
            }

            yield instance_data, common_data

    def update_arguments(self, doc_info, evm_index, arguments):
        event = doc_info['events'][evm_index]
        full_info = {
            'predicate': event['predicate'],
            'frame': event['frame'],
            'args': {},
        }

        for slot, arg_info in arguments.items():
            if not arg_info or arg_info['entity_id'] == -1:
                full_info['args'][slot] = {}
            else:
                full_info['args'][slot] = {
                    'arg_role': arg_info['arg_role'],
                    'fe': arg_info['fe'],
                    'entity_id': arg_info['entity_id'],
                }
        return full_info

    def compute_distances(self, current_evm_id, entity_sents, event, sent_id):
        """
        Compute the distance signature of the instance's other mentions to the
        sentence.
        :param current_evm_id:
        :param entity_sents:
        :param event:
        :param sent_id:
        :return:
        """
        distances = []

        # Now use a large distance to represent Indefinite.
        # Infinity: if the entity cannot be found again, or it is not an entity.
        # Arbitrarily use 50 since most document is shorter than this.
        inf = 30.0

        for current_slot, slot_info in event['args'].items():
            entity_id = slot_info.get('entity_id', -1)

            if entity_id == -1:
                # This is an empty slot.
                distances.append((inf, inf, inf))
                continue

            max_dist = 0.0
            min_dist = inf
            total_dist = 0.0
            total_pair = 0.0

            for evm_id, slot, sid in entity_sents[entity_id]:
                if evm_id == current_evm_id:
                    # Do not compute mentions at the same event.
                    continue

                distance = abs(sid - sent_id)
                if distance < min_dist:
                    min_dist = distance
                if distance > max_dist:
                    max_dist = distance
                total_dist += distance
                total_pair += 1.0

            if total_pair > 0:
                distances.append((max_dist, min_dist, total_dist / total_pair))
                # print(max_dist, min_dist, total_dist / total_pair)
            else:
                # This argument is not seen elsewhere, it should be a special
                # distance label.
                distances.append((inf, inf, inf))
                # print("No other mentions found")

        # Flatten the (argument x type) distances into a flat list.
        return [d for l in distances for d in l]

    def replace_slot(self, instance, slot, new_eid, new_text, fe=None):
        slot_info = instance[slot]

        # Fall back from specific dep label.
        dep = slot_info.get('dep', slot)

        new_arg_role = hash_arg_role(
            new_text, dep, self.event_vocab, self.oovs)

        slot_info['arg_role'] = new_arg_role
        slot_info['entity_id'] = new_eid
        slot_info['text'] = new_text

        if fe is not None:
            slot_info['fe'] = fe

        slot_info['context'] = slot_info.get('context', [[], []])

    def cross_cloze(self, args, arg_entities, target_evm_id, target_slot):
        """
        A negative cloze instance that use arguments from other events.
        :param args: Dict of origin event arguments.
        :param arg_entities: List of original (event id, entity id) pairs.
        :param target_evm_id: The target event id.
        :param target_slot: The target slot name.
        :return:
        """
        target_arg = args[target_slot]
        target_eid = target_arg['entity_id']

        sample_res = self.sampler.sample_cross(
            arg_entities, target_evm_id, target_eid
        )

        if not sample_res:
            return None

        wrong_evm, wrong_id, wrong_text = sample_res

        # print("Sampled {}:{} from {} for slot {}".format(
        #     wrong_id, wrong_text, wrong_evm, target_slot))

        neg_instance = {}
        for slot, content in args.items():
            neg_instance[slot] = {}
            neg_instance[slot].update(content)

        # When taking entity from another slot, we don't take its FE,
        # because the FE from another sentence will not make sense here.
        self.replace_slot(neg_instance, target_slot, wrong_id, wrong_text)
        return neg_instance, wrong_id

    def inside_cloze(self, origin_event_args, target_slot):
        """
        A negative cloze instance that use arguments within the event.
        :param origin_event_args:
        :param target_slot:
        :return:
        """
        neg_instance = {}

        for slot, content in origin_event_args.items():
            neg_instance[slot] = {}
            neg_instance[slot].update(content)

        swap_slot = self.sampler.sample_ignore_item(
            list(origin_event_args.keys()), target_slot
        )
        if not swap_slot:
            logging.warning("Inside cloze not generating item.")
            return None

        target_slot_info = origin_event_args[target_slot]
        swap_slot_info = origin_event_args[swap_slot]

        # When swapping the two slots, we also swap the FEs
        # this help us learn the correct slot for a FE.
        if swap_slot_info:
            self.replace_slot(
                neg_instance, target_slot, swap_slot_info['entity_id'],
                swap_slot_info['text'], swap_slot_info['fe'])
        else:
            neg_instance[target_slot] = {}

        if target_slot_info:
            self.replace_slot(
                neg_instance, swap_slot, target_slot_info['entity_id'],
                target_slot_info['text'], target_slot_info['fe'])
        else:
            neg_instance[swap_slot] = {}

        # print("Swapping {} and {}".format(target_slot, swap_slot))

        return neg_instance, target_slot_info['entity_id']
