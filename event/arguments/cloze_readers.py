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
)


class ClozeSampler:
    def __init__(self):
        pass

    def sample_cross(self, arg_entities, evm_id, ent_id):

        remaining = []
        for evm, ent in arg_entities:
            if not (evm == evm_id or ent == ent_id):
                remaining.append((evm, ent))

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

        for ins_type, ins_data in b_instance_data.items():
            for key, value in ins_data.items():
                padded = self.__batch_pad(key, value, max_instance_size)
                vectorized = to_torch(padded, self.__data_types[key])
                instance_data[ins_type][key] = batch_combine(
                    vectorized, self.device
                )
        return instance_data, common_data

    def read_cloze_batch(self, data_in, from_line=None, until_line=None):
        b_common_data = defaultdict(list)
        b_instance_data = defaultdict(lambda: defaultdict(list))

        max_context_size = 0
        max_instance_size = 0
        doc_count = 0

        batch_predicates = []

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
                # Merging cloze tasks to batch.

                # import operator
                # print('max', max([(len(l), l) for l in batch_predicates],
                #                  key=operator.itemgetter(0)))
                # torch_util.show_tensors()
                # torch_util.gpu_mem_report()
                # input("==========")

                yield self.create_batch(b_common_data, b_instance_data,
                                        max_context_size, max_instance_size)

                # Reset counts.
                batch_predicates.clear()

                b_common_data.clear()
                b_instance_data.clear()
                doc_count = 0
                max_context_size = 0
                max_instance_size = 0

        # Yield the remaining data.
        yield self.create_batch(b_common_data, b_instance_data,
                                max_context_size, max_instance_size)

    def _take_event_parts(self, event_info):
        event_components = [event_info['predicate'], event_info['frame']]

        for slot_name in self.slot_names:
            slot = event_info['slots'][slot_name]
            event_components.append(slot['fe'])
            event_components.append(slot['arg'])

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
            event_args = defaultdict(dict)
            entity_mentions = defaultdict(list)
            arg_entities = set()
            eid_count = Counter()

            all_event_reps = []

            for evm_index, event in enumerate(doc_info['events']):
                if evm_index == self.max_events:
                    break

                for slot, arg in event['args'].items():
                    # Argument for nth event, at slot position 'slot'.
                    eid = arg['entity_id']
                    dep = arg['dep']

                    if not dep:
                        # Fall back to unspecified dep name. This does not
                        # affect "subj", "obj", but it will create "prep"
                        # without the actual prepositional word.
                        dep = slot

                    event_args[evm_index][slot] = {
                        'entity_id': eid,
                        'arg_role': arg['arg'],
                        'fe': arg['fe']
                    }

                    # From eid to entity information.
                    entity_mentions[eid].append(
                        ((evm_index, slot), arg['sentence_id'])
                    )

                    # We re-count for resolvable, because we may filter events,
                    # which will change the resolvable attribute.
                    if eid >= 0:
                        arg_entities.add((evm_index, eid))
                        eid_count[eid] += 1

                event_info = self.get_event_info(
                    doc_info, evm_index, event_args[evm_index])
                event_rep = self._take_event_parts(event_info)

                all_event_reps.append(event_rep)

                event_data.append(event_info)

            arg_entities = list(arg_entities)

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

            for evm_index, event in enumerate(doc_info['events']):
                if evm_index == self.max_events:
                    # Cut off long documents.
                    break

                if len(arg_entities) <= 1:
                    # There no enough arguments to sample from.
                    break

                event_info = event_data[evm_index]

                pred = event_info['predicate']
                if pred == self.event_vocab[consts.unk_predicate]:
                    # print("Skip unk pred")
                    continue

                keep_pred = self.subsample_pred(pred)

                if not keep_pred:
                    # Too frequent word will be ignore.
                    # print("Skip", self.event_inverted[pred])
                    continue

                for slot_index, slot in enumerate(self.slot_names):
                    arg = event['args'][slot]
                    correct_id = arg['entity_id']
                    current_sent = arg['sentence_id']

                    # Apparently, singleton and unks are not resolvable.
                    if correct_id < 0:
                        # print("skip empty arg")
                        continue

                    if eid_count[correct_id] <= 1:
                        # print("Skip singleton")
                        continue

                    if entity_heads[correct_id] == consts.unk_arg_word:
                        # print("Skip unk arg")
                        continue

                    cloze_event_indices.append(evm_index)
                    cloze_slot_indices.append(slot_index)

                    gold_rep = self._take_event_parts(event_info)
                    gold_features = features_by_eid[correct_id]
                    gold_distances = self.compute_distances(
                        evm_index, entity_mentions, event_info,
                        current_sent,
                    )
                    gold_event_data['rep'].append(gold_rep)
                    gold_event_data['features'].append(gold_features)
                    gold_event_data['distances'].append(gold_distances)

                    cross_sample = self.cross_cloze(event_args, arg_entities,
                                                    evm_index, slot)

                    if not cross_sample:
                        # Simply discard the sample if there is no good cross
                        #  cloze.
                        continue

                    cross_args, cross_filler_id = cross_sample

                    cross_info = self.get_event_info(
                        doc_info, evm_index, cross_args)
                    cross_rep = self._take_event_parts(cross_info)
                    cross_features = features_by_eid[cross_filler_id]
                    cross_distances = self.compute_distances(
                        evm_index, entity_mentions, cross_info, current_sent,
                    )
                    cross_event_data['rep'].append(cross_rep)
                    cross_event_data['features'].append(cross_features)
                    cross_event_data['distances'].append(cross_distances)

                    inside_sample = self.inside_cloze(
                        event_args[evm_index], slot)
                    inside_args, inside_filler_id = inside_sample
                    inside_info = self.get_event_info(
                        doc_info, evm_index, inside_args)
                    inside_rep = self._take_event_parts(inside_info)
                    inside_features = features_by_eid[inside_filler_id]
                    inside_distances = self.compute_distances(
                        evm_index, entity_mentions, inside_info, current_sent,
                    )
                    inside_event_data['rep'].append(inside_rep)
                    inside_event_data['features'].append(inside_features)
                    inside_event_data['distances'].append(inside_distances)

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

    def get_event_info(self, doc_info, evm_index, arguments):
        event = doc_info['events'][evm_index]
        full_info = {
            'predicate': event['predicate'],
            'frame': event['frame'],
            'slots': {},
        }

        # TODO: Fix this to get argument role index correctly.
        for slot, arg_info in arguments.items():
            eid = arg_info['entity_id']

            if eid == -1:
                # This is an empty slot. Using pad.
                pad = len(self.event_vocab)
                fe = pad
                arg_role = pad
            else:
                fe = arg_info['fe']
                arg_role = arg_info['arg_role']

                if fe == -1:
                    fe = self.event_vocab[consts.unk_fe]

            full_info['slots'][slot] = {
                'arg': arg_role,
                'fe': fe,
                'context': event['args'][slot]['context'],
                'eid': eid
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
        # Arbitrarily use 1000 since most document is shorter than this.
        inf = 1000.0

        for current_slot, event_info in event['slots'].items():
            entity_id = event_info['eid']

            if entity_id == -1:
                # This is an empty slot.
                distances.append((inf, inf, inf))
                continue

            max_dist = 0.0
            min_dist = inf
            total_dist = 0.0
            total_pair = 0.0

            for (evm_id, slot), sid in entity_sents[entity_id]:
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

    def cross_cloze(self, event_args, arg_entities, target_evm_id, target_slot):
        """
        A negative cloze instance that use arguments from other events.
        :param event_args: Dict of all origin event arguments.
        :param arg_entities: List of original (event id, entity id) pairs.
        :param target_evm_id: The target event id.
        :param target_slot: The target slot name.
        :return:
        """

        target_args = event_args[target_evm_id]
        target_arg = target_args[target_slot]
        target_eid = target_arg['entity_id']

        sample_res = self.sampler.sample_cross(
            arg_entities, target_evm_id, target_eid
        )

        if sample_res:
            wrong_evm, wrong_id = sample_res
            neg_instance = {}
            neg_instance.update(target_args)

            print(target_args)
            current_eid, dep = event_args[target_evm_id][target_slot]

            neg_instance[target_slot] = (wrong_id, dep)

            print(neg_instance)

            input(neg_instance)

            return neg_instance, wrong_id
        else:
            logging.warning("Cross cloze not generating item.")
            return None

    def inside_cloze(self, origin_event_args, target_slot):
        """
        A negative cloze instance that use arguments within the event.
        :param origin_event_args:
        :param target_slot:
        :return:
        """
        neg_instance = {}
        neg_instance.update(origin_event_args)

        swapping_slot = self.sampler.sample_ignore_item(
            list(origin_event_args.keys()), target_slot
        )
        if not swapping_slot:
            logging.warning("Inside cloze not generating item.")
            return None

        origin_id, origin_dep = origin_event_args[target_slot]
        swap_id, swapping_dep = origin_event_args[swapping_slot]

        # Swap the two slots, this may create:
        # 1. instance with slot argument swapped
        # 2. instance with a frame moved to another empty slot
        neg_instance[swapping_slot] = (origin_id, swapping_dep)
        neg_instance[target_slot] = (swap_id, origin_dep)

        return neg_instance, origin_id


