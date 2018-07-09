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
import torch.nn.functional as F
from event.io.io_utils import pad_2d_list
from event.arguments.util import (batch_combine, to_torch)


class HashedClozeReader:
    def __init__(self, event_vocab, lookups, oovs, batch_size,
                 multi_context=False, max_events=200, gpu=True):
        """
        Reading the hashed dataset into cloze tasks.
        :param event_vocab: Event token vocabulary.
        :param batch_size: Number of cloze tasks per batch.
        :param multi_context: Whether to use multiple events as context.
        :param max_events: Number of events to keep per document.
        """
        self.batch_size = batch_size
        self.multi_context = multi_context
        self.max_events = max_events
        self.event_vocab = event_vocab
        self.lookups = lookups
        self.oovs = oovs

        self.event_padding = len(event_vocab)

        self.slot_names = ['subj', 'obj', 'prep', ]

        self.__data_types = {
            'context': np.int64,
            'event_indices': np.int64,
            'slot_indices': np.int64,
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

        # Fix seed to generate the same instances.
        random.seed(17)

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

    def read_cloze_batch(self, data_in):
        b_common_data = defaultdict(list)
        b_instance_data = defaultdict(lambda: defaultdict(list))

        max_context_size = 0
        max_instance_size = 0
        cloze_count = 0

        for instance_data, common_data in self.read_clozes(data_in):
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

            cloze_count += 1

            if cloze_count == self.batch_size:
                # Merging cloze tasks to batch.
                yield self.create_batch(b_common_data, b_instance_data,
                                        max_context_size, max_instance_size)

                # Reset counts.
                b_common_data = defaultdict(list)
                b_instance_data = defaultdict(lambda: defaultdict(list))
                cloze_count = 0
                max_context_size = 0
                max_instance_size = 0

    def _take_event_parts(self, event_info):
        event_components = [event_info['predicate'], event_info['frame']]

        for slot_name in self.slot_names:
            slot = event_info['slots'][slot_name]
            event_components.append(slot['fe'])
            event_components.append(slot['arg'])

        return [c if c > 0 else self.event_padding for c in event_components]

    def read_clozes(self, data_in):
        for line in data_in:
            doc_info = json.loads(line)

            # Collect all features.
            features_by_eid = {}
            entity_heads = {}
            for eid, content in doc_info['entities'].items():
                features_by_eid[int(eid)] = content['features']
                entity_heads[int(eid)] = content['entity_head']

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
                    event_args[evm_index][slot] = (eid, arg['dep'])
                    # From eid to entity information.
                    entity_mentions[eid].append(
                        ((evm_index, slot), arg['sentence_id'])
                    )

                    if eid > 0:
                        arg_entities.add((evm_index, eid))
                        eid_count[eid] += 1

                event_info = self.get_event_info(
                    doc_info, evm_index, event_args[evm_index])
                event_rep = self._take_event_parts(event_info)

                all_event_reps.append(event_rep)

                event_data.append(event_info)

            arg_entities = list(arg_entities)

            # A better learning strategy is to select one
            # cross instance that is difficult. We can have two
            # strategies here:
            # 1. Use unigram distribution to sample items.
            # 2. Select items based on classifier output.

            gold_event_data = {'rep': [], 'distances': [], 'features': []}
            cross_event_data = {'rep': [], 'distances': [], 'features': []}
            inside_event_data = {'rep': [], 'distances': [], 'features': []}
            cloze_event_indices = []
            cloze_slot_indices = []

            for evm_index, event in enumerate(doc_info['events']):
                if evm_index == self.max_events:
                    break

                for slot_index, slot in enumerate(self.slot_names):
                    arg = event['args'][slot]
                    correct_id = arg['entity_id']

                    # We re-count for resolvable, because we may filter events,
                    # which will change the resolvable attribute.
                    if eid_count[correct_id] > 1:
                        cloze_event_indices.append(evm_index)
                        cloze_slot_indices.append(slot_index)

                        correct_id = arg['entity_id']
                        current_sent = arg['sentence_id']

                        event_info = event_data[evm_index]
                        gold_rep = self._take_event_parts(event_info)
                        gold_features = features_by_eid[correct_id]
                        gold_distances = self.compute_distances(
                            evm_index, entity_mentions, event_info,
                            current_sent,
                        )
                        gold_event_data['rep'].append(gold_rep)
                        gold_event_data['features'].append(gold_features)
                        gold_event_data['distances'].append(gold_distances)

                        cross_event, cross_filler_id = self.cross_cloze(
                            event_args, arg_entities, evm_index, slot,
                            correct_id)
                        cross_info = self.get_event_info(doc_info, evm_index,
                                                         cross_event)
                        cross_rep = self._take_event_parts(cross_info)
                        cross_features = features_by_eid[cross_filler_id]
                        cross_distances = self.compute_distances(
                            evm_index, entity_mentions, cross_info,
                            current_sent,
                        )
                        cross_event_data['rep'].append(cross_rep)
                        cross_event_data['features'].append(cross_features)
                        cross_event_data['distances'].append(cross_distances)

                        inside_event, inside_filler_id = self.inside_cloze(
                            event_args, evm_index, slot, correct_id)
                        inside_info = self.get_event_info(doc_info, evm_index,
                                                          inside_event)
                        inside_rep = self._take_event_parts(inside_info)
                        inside_features = features_by_eid[inside_filler_id]
                        inside_distances = self.compute_distances(
                            evm_index, entity_mentions, inside_info,
                            current_sent,
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

    def sample_ignore_item(self, data, ignored_item):
        if len(data) <= 1:
            raise ValueError("Sampling from a list of size %d" % len(data))
        while True:
            sampled_item = random.choice(data)
            if not sampled_item == ignored_item:
                break
        return sampled_item

    def sample_ignore_index(self, data, ignored_index):
        if len(data) <= 1:
            raise ValueError("Sampling from a list of size %d" % len(data))
        while True:
            sampled_index = random.choice(range(len(data)))
            if not sampled_index == ignored_index:
                break
        return data[sampled_index]

    def get_event_info(self, doc_info, evm_index, arguments):
        event = doc_info['events'][evm_index]
        full_info = {
            'predicate': event['predicate'],
            'frame': event['frame'],
            'slots': {},
        }

        for slot, (eid, _) in arguments.items():
            if eid == -1:
                # This is an empty slot. Using pad.
                pad = len(self.event_vocab)
                fe = pad
                arg_role = pad
            else:
                arg_info = event['args'][slot]
                fe = arg_info['fe']
                arg_role = arg_info['arg']

                if fe == -1:
                    fe = self.event_vocab[consts.unk_fe]

            full_info['slots'][slot] = {
                'arg': arg_role,
                'fe': fe,
                'context': event['args'][slot]['context'],
                'eid': eid
            }

        return full_info

    def compute_distances(self, current_evm_id, entity_sents, event,
                          sentence_id):
        """
        Compute the distance signature of the instance's other mentions to the
        sentence.
        :param current_evm_id:
        :param entity_sents:
        :param event:
        :param sentence_id:
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

                distance = abs(sid - sentence_id)
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

    def cross_cloze(self, event_args, arg_entities, current_evm, current_slot,
                    correct_id):
        """
        A negative cloze instance that use arguments from other events.
        :param event_args: List of all origin event arguments.
        :param arg_entities: List of argument entities.
        :param current_evm: The event id.
        :param current_slot: The slot id.
        :param correct_id: Correct entity id.
        :return:
        """
        wrong_evm, wrong_id = self.sample_ignore_item(
            arg_entities, (current_evm, correct_id)
        )
        neg_instance = {}
        neg_instance.update(event_args[current_evm])
        dep = event_args[current_evm][current_slot]
        neg_instance[current_slot] = (wrong_id, dep)

        return neg_instance, wrong_id

    def inside_cloze(self, event_args, current_evm, current_slot, correct_id):
        """
        A negative cloze instance that use arguments within the event.
        :param event_args:
        :param current_evm:
        :param current_slot:
        :param correct_id:
        :return:
        """
        current_event = event_args[current_evm]

        neg_instance = {}
        neg_instance.update(current_event)

        swapping_slot = self.sample_ignore_item(
            list(current_event.keys()), current_slot
        )
        swapping_id = current_event[swapping_slot]

        current_dep = event_args[current_evm][current_slot]
        swapping_dep = event_args[current_evm][swapping_slot]

        # Swap the two slots, this may create:
        # 1. instance with frames swapped
        # 2. instance with a frame moved to another empty slot
        neg_instance[swapping_slot] = (correct_id, swapping_dep)
        neg_instance[current_slot] = (swapping_id, current_dep)

        return neg_instance, correct_id


class EventReader:
    def __init__(self):
        self.target_roles = ['arg0', 'arg1', 'prep']
        self.entity_info_fields = ['syntactic_role', 'mention_text',
                                   'entity_id']
        self.entity_equal_fields = ['entity_id', 'represent']

        self.len_arg_fields = 4

    def get_context(self, sentence, start, end, window_size=5):
        right_tokens = sentence[end:].strip().split()
        right_win = min(window_size, len(right_tokens))
        right_context = right_tokens[:right_win]

        left_tokens = sentence[:start].strip().split()
        left_tokens.reverse()
        left_win = min(window_size, len(left_tokens))

        left_context = left_tokens[:left_win]
        left_context.reverse()

        return left_context, right_context

    def read_events(self, data_in):
        for line in data_in:
            doc = json.loads(line)
            docid = doc['docid']

            events = []

            eid_count = Counter()

            entity_heads = {}

            entities = {}

            if 'entities' in doc:
                for ent in doc['entities']:
                    entity_heads[ent['entityId']] = ent['representEntityHead']

                    entities[ent['entityId']] = {
                        'features': ent['entityFeatures'],
                        'representEntityHead': ent['representEntityHead'],
                    }

            for event_info in doc['events']:
                sent = doc['sentences'][event_info['sentenceId']]

                raw_context = self.get_context(
                    sent,
                    event_info['predicateStart'],
                    event_info['predicateEnd'],
                )

                event = {
                    'predicate': event_info['predicate'],
                    'predicate_context': raw_context,
                    # 'predicate_context': event_info['context'],
                    'frame': event_info.get('frame', 'NA'),
                    'arguments': [],
                    'predicate_start': event_info['predicateStart'],
                    'predicate_end': event_info['predicateEnd'],
                }

                events.append(event)

                for arg_info in event_info['arguments']:
                    if 'argStart' in arg_info:
                        arg_context = self.get_context(
                            sent, arg_info['argStart'], arg_info['argEnd']
                        )
                    else:
                        left, right = arg_info['context'].split('___')
                        arg_context = left.split(), right.split()

                    if entity_heads:
                        represent = entity_heads[arg_info['entityId']]
                    else:
                        represent = arg_info['representText']

                    arg = {
                        'dep': arg_info['dep'],
                        'fe': arg_info['feName'],
                        'arg_context': arg_context,
                        'represent': represent,
                        'entity_id': arg_info['entityId'],
                        'resolvable': False,
                        'arg_start': arg_info['argStart'],
                        'arg_end': arg_info['argEnd'],
                        'sentence_id': event_info['sentenceId']
                    }

                    eid_count[arg_info['entityId']] += 1
                    event['arguments'].append(arg)

            for event in events:
                for arg in event['arguments']:
                    if eid_count[arg['entity_id']] > 1:
                        arg['resolvable'] = True

            yield docid, events, entities

    def _same_entity(self, ent1, ent2):
        return any([ent1[f] == ent2[f] for f in self.entity_equal_fields])

    def _entity_info(self, arg):
        return dict([(k, arg[k]) for k in self.entity_info_fields])
