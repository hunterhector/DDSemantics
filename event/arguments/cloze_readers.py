import json
import logging
from collections import Counter
from collections import defaultdict

import numpy as np
import torch

from event.arguments import consts
from event.arguments.prepare.hash_cloze_data import (
    hash_arg_role,
)
from event.arguments.util import (batch_combine, to_torch)
from event.io.io_utils import pad_2d_list

from pprint import pprint


def get_distance_signature(
        current_evm_id, entity_mentions, event, sent_id):
    """
    Compute the distance signature of the instance's other mentions to the
    sentence.
    :param current_evm_id:
    :param entity_mentions:
    :param event:
    :param sent_id:
    :return:
    """
    distances = []

    # Now use a large distance to represent Infinity.
    # Infinity: if the entity cannot be found again, or it is not an entity.
    # A number is arbitrarily decided since most document is shorter than
    # this.
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

        for evm_id, slot, sid in entity_mentions[entity_id]:
            if evm_id == current_evm_id:
                # Do not compute mentions at the same event.
                continue

            distance = abs(sid - sent_id)

            # We make a ceiling for the distance calculation.
            distance = min(distance, inf - 1)

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
        self.from_hashed = from_hashed

        self.event_vocab = resources.event_vocab
        self.event_count = resources.term_freq
        self.pred_count = resources.typed_count['predicate']

        self.lookups = resources.lookups
        # There are specific oovs for different type of vocabularies.
        self.oovs = resources.oovs

        # Inverted vocab for debug purpose.
        self.event_inverted = self.__inverse_vocab(self.event_vocab)

        # Some extra embeddings.
        extra_index = len(self.event_vocab)
        self.unobserved_fe = extra_index
        self.unobserved_arg = extra_index + 1

        logging.info(
            "Use extra %d for unobserved fe, %d for unobserved arg." % (
                self.unobserved_fe, self.unobserved_arg))

        self.slot_names = ['subj', 'obj', 'prep', ]

        self.__data_types = {
            'context': np.int64,
            'event_indices': np.int64,
            # The slot index is considered as a feature as well, so it has the
            # same type as the features.
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
        # The actual instance lengths of in each batch.
        data_len = []
        sizes = {}

        for key, value in b_common_data.items():
            if key == 'context':
                padded = self.__batch_pad(key, value, max_context_size)
                vectorized = to_torch(padded, self.__data_types[key])
                common_data[key] = batch_combine(vectorized, self.device)
            else:
                padded = self.__batch_pad(key, value, max_instance_size)
                vectorized = to_torch(padded, self.__data_types[key])
                common_data[key] = batch_combine(vectorized, self.device)

            sizes[key] = len(padded)

        for ins_type, ins_data in b_instance_data.items():
            for key, value in ins_data.items():
                data_len = [len(v) for v in value]
                padded = self.__batch_pad(key, value, max_instance_size)
                vectorized = to_torch(padded, self.__data_types[key])
                instance_data[ins_type][key] = batch_combine(
                    vectorized, self.device
                )
                sizes[key] = len(padded)

        b_size = sizes['context']
        for key, s in sizes.items():
            assert s == b_size

        mask = np.zeros([b_size, max_instance_size], dtype=int)
        for i, l in enumerate(data_len):
            mask[i][0: l] = 1

        ins_mask = to_torch(mask, np.uint8).to(self.device)

        return instance_data, common_data, b_size, ins_mask

    def read_train_batch(self, data_in, sampler, from_line=None,
                         until_line=None):
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

            max_context_size = 0
            max_instance_size = 0

        for instance_data, common_data in self.parse_docs(
                data_in, sampler, from_line, until_line):
            gold_event_data = instance_data['gold']

            # Debug purpose only.
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

            # Each document is considered as one single instance since.
            if doc_count % self.batch_size == 0:
                debug_data = {
                    'predicate': batch_predicates,
                }

                train_batch = self.create_batch(
                    b_common_data, b_instance_data, max_context_size,
                    max_instance_size
                )

                yield train_batch, debug_data
                _clear()

        # Yield the remaining data.
        if len(b_common_data) > 0:
            debug_data = {
                'predicate': batch_predicates,
            }

            train_batch = self.create_batch(b_common_data, b_instance_data,
                                            max_context_size, max_instance_size)

            yield train_batch, debug_data
            _clear()

    def _take_event_parts(self, event_info):
        frame_id = event_info['frame']
        event_components = [
            event_info['predicate'],
            self.event_vocab[consts.unk_frame] if frame_id == -1 else frame_id]

        for slot_name in self.slot_names:
            slot = event_info['args'][slot_name]

            if len(slot) == 0:
                # event_components.append(self.unobserved_fe)
                # event_components.append(self.unobserved_arg)
                event_components.append(0)
                event_components.append(2)
            else:
                event_components.append(slot['fe'])
                event_components.append(slot['arg_role'])

        if any([c < 0 for c in event_components]):
            print(event_info)
            print(event_components)
            input('not positive')

        # TODO do not use c>0 here, explicitly handle it.
        return [c if c > 0 else self.unobserved_fe for c in event_components]

    def parse_hashed(self):
        pass

    def parse_origin(self):
        pass

    @staticmethod
    def collect_features(doc_info):
        # Collect all features.
        features_by_eid = {}
        entity_heads = {}

        for eid, content in doc_info['entities'].items():
            features_by_eid[int(eid)] = content['features']
            entity_heads[int(eid)] = content['entity_head']

        return features_by_eid, entity_heads

    def create_test_instance(self, target_evm_id, target_slot, args, doc_args):
        test_rank_list = []
        gold_eid = args[target_slot]['entity_id']

        # Replace the target slot with other entities for generating this.
        for evm, eid, arg_text in doc_args:
            # Make a copy of the original slots.
            cand_args = {}
            for slot, content in args.items():
                cand_args[slot] = {}
                cand_args[slot].update(content)

            if evm == target_evm_id and eid == gold_eid:
                # This is the correct answer.
                continue

            # Replace with another entity.
            self.replace_slot(cand_args, target_slot, eid, arg_text)
            test_rank_list.append((cand_args, eid))

        return test_rank_list

    def get_one_test_doc(self, doc_info, nid_detector):
        """
        Parse and get one test document.
        :param doc_info: The JSON data of one document.
        :param nid_detector: NID detector to detect which slot to fill.
        :return:
        """
        instance_data = {'rep': [], 'distances': [], 'features': [], }
        debug_data = {'predicate': [], 'entity_text': [], 'gold_entity': [], }

        # Collect information such as features and entity positions.
        features_by_eid, entity_heads = self.collect_features(doc_info)
        all_event_reps = [self._take_event_parts(e) for e in doc_info['events']]

        # Some need to be done in iteration.
        entity_positions = defaultdict(list)
        doc_args = set()

        event_subset = []

        for evm_index, event in enumerate(doc_info['events']):
            sentence_id = event.get('sentence_id', None)

            event_subset.append(event)

            for slot, arg in event['args'].items():
                # implicit arguments will not be candidates.
                if len(arg) > 0 and not arg.get('implicit', False):
                    eid = arg['entity_id']
                    doc_args.add((evm_index, eid, arg['text']))
                    entity_positions[eid].append((evm_index, slot, sentence_id))

        # Make sure this order is the same, but this should not matter.
        doc_args = sorted(list(doc_args))

        cloze_event_indices = []
        cloze_slot_indices = []
        gold_labels = []

        for evm_index, event in enumerate(event_subset):
            pred_sent = event['sentence_id']

            pred_idx = event['predicate']
            predicate = (
                pred_idx, self.event_inverted[pred_idx],
                event['predicate_text']
            )

            for slot, arg in event['args'].items():
                is_instance = nid_detector.should_fill(event, slot)

                if is_instance and arg['resolvable']:
                    test_rank_list = self.create_test_instance(
                        evm_index, slot, event['args'], doc_args
                    )

                    for cand_args, filler_eid in test_rank_list:
                        cand_event = self.update_arguments(
                            doc_info, evm_index, cand_args
                        )
                        self.assemble_instance(
                            instance_data, features_by_eid, entity_positions,
                            evm_index, pred_sent, cand_event, filler_eid)

                        cloze_event_indices.append(evm_index)
                        cloze_slot_indices.append(self.slot_names.index(slot))

                        gold_labels.append(
                            1 if filler_eid == arg['entity_id'] else 0
                        )

                        debug_data['predicate'].append(predicate)
                        debug_data['entity_text'].append(
                            entity_heads[filler_eid])
                        debug_data['gold_entity'].append(
                            (arg['entity_id'],
                             entity_heads[arg['entity_id']])
                        )

                        # TODO: Set a hard threshold here might be dangerous.
                        if len(cloze_event_indices) > 500:
                            break

        if len(cloze_event_indices) == 0:
            return None

        common_data = {
            'context': all_event_reps,
            'event_indices': cloze_event_indices,
            'slot_indices': cloze_slot_indices,
        }

        return instance_data, common_data, gold_labels, debug_data

    def read_test_docs(self, test_in, nid_detector):
        """
        Load test data. Importantly, this will create alternative cloze
         filling for ranking.
        :param test_in: supply lines as test data.
        :param nid_detector: Null Instantiation Detector.
        :return:
        """
        for line in test_in:
            doc_info = json.loads(line)
            doc_id = doc_info['docid']

            test_data = self.get_one_test_doc(doc_info, nid_detector)

            if not test_data:
                continue

            doc_instances, common_data, gold_labels, debug_data = test_data

            b_common_data = {}
            b_instance_data = {}

            # Create a pseudo batch with one instance.
            for key, value in common_data.items():
                vectorized = to_torch([value], self.__data_types[key])
                b_common_data[key] = batch_combine(vectorized, self.device)

            for key, value in doc_instances.items():
                vectorized = to_torch([value], self.__data_types[key])
                b_instance_data[key] = batch_combine(vectorized, self.device)

            yield (
                doc_id,
                b_instance_data,
                b_common_data,
                gold_labels,
                debug_data,
            )

    def create_training_data(self, data_line, sampler):
        doc_info = json.loads(data_line)
        features_by_eid, entity_heads = self.collect_features(doc_info)

        # Map from: entity id (eid) ->
        # A list of tuples that represent an argument position:
        # [(evm_index, slot, sentence_id)]
        entity_positions = defaultdict(list)

        # Unique set of (evm_index, slot, sentence_id) tuples.
        # This is used to sample a slot to create clozes.
        arg_entities = set()
        eid_count = Counter()

        event_subset = []

        for evm_index, event in enumerate(doc_info['events']):
            if evm_index == self.max_events:
                # Ignore documents that are too long.
                break

            event_subset.append(event)

            sentence_id = event.get('sentence_id', None)

            arg_info = {}
            for slot, arg in event['args'].items():
                if not arg or arg['entity_id'] == -1:
                    arg_info[slot] = {}
                else:
                    # Argument for n-th event, at slot position 'slot'.
                    eid = arg['entity_id']

                    # From eid to entity information.
                    entity_positions[eid].append((evm_index, slot, sentence_id))
                    arg_entities.add((evm_index, eid, arg['text']))
                    eid_count[eid] += 1

        all_event_reps = [self._take_event_parts(e) for e in event_subset]

        if len(arg_entities) <= 1:
            # There no enough arguments to sample from.
            return None

        arg_entities = sorted(list(arg_entities))

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

        for evm_index, event_info in enumerate(event_subset):
            pred = event_info['predicate']
            if pred == self.event_vocab[consts.unk_predicate]:
                continue

            pred_tf = self.event_count[pred]
            freq = 1.0 * pred_tf / self.pred_count
            keep_pred = sampler.subsample_pred(pred_tf, freq)

            if not keep_pred:
                # Too frequent word will be down-sampled.
                continue

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
                    sampler, event_info['args'], arg_entities,
                    evm_index, slot
                )

                if not cross_sample:
                    continue

                inside_sample = self.inside_cloze(
                    sampler, event_info['args'], slot
                )

                if not inside_sample:
                    continue

                cross_args, cross_filler_id = cross_sample

                cross_info = self.update_arguments(
                    doc_info, evm_index, cross_args)
                self.assemble_instance(cross_event_data, features_by_eid,
                                       entity_positions, evm_index,
                                       current_sent, cross_info,
                                       cross_filler_id)

                inside_args, inside_filler_id = inside_sample
                inside_info = self.update_arguments(
                    doc_info, evm_index, inside_args)

                self.assemble_instance(inside_event_data, features_by_eid,
                                       entity_positions, evm_index,
                                       current_sent, inside_info,
                                       inside_filler_id)

                self.assemble_instance(gold_event_data, features_by_eid,
                                       entity_positions, evm_index,
                                       current_sent, event_info, correct_id)

                # These two list indicate where the target argument is.
                cloze_event_indices.append(evm_index)
                cloze_slot_indices.append(slot_index)

        if len(cloze_slot_indices) == 0:
            # This document do not contains training instance.
            return None

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

        return instance_data, common_data

    def assemble_instance(self, instance_data, features_by_eid,
                          entity_positions, evm_index, sent_id,
                          event_words, filler_eid):
        instance_data['rep'].append(self._take_event_parts(event_words))
        instance_data['features'].append(features_by_eid[filler_eid])
        instance_data['distances'].append(get_distance_signature(
            evm_index, entity_positions, event_words, sent_id))

    def parse_docs(self, data_in, sampler, from_line=None, until_line=None):
        line_num = 0
        for line in data_in:
            line_num += 1

            if from_line and line_num <= from_line:
                continue

            if until_line and line_num > until_line:
                break

            parsed_output = self.create_training_data(line, sampler)

            if parsed_output is None:
                continue

            instance_data, common_data = parsed_output
            yield instance_data, common_data

    @staticmethod
    def update_arguments(doc_info, evm_index, arguments):
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

    def cross_cloze(self, sampler, args, arg_entities, target_evm_id,
                    target_slot):
        """
        A negative cloze instance that use arguments from other events.
        :param sampler: A random sampler.
        :param args: Dict of origin event arguments.
        :param arg_entities: List of original (event id, entity id) pairs.
        :param target_evm_id: The target event id.
        :param target_slot: The target slot name.
        :return:
        """
        target_arg = args[target_slot]
        target_eid = target_arg['entity_id']

        sample_res = sampler.sample_cross(
            arg_entities, target_evm_id, target_eid
        )

        if not sample_res:
            return None

        wrong_evm, wrong_id, wrong_text = sample_res

        neg_instance = {}
        for slot, content in args.items():
            neg_instance[slot] = {}
            neg_instance[slot].update(content)

        # When taking entity from another slot, we don't take its FE,
        # because the FE from another sentence will not make sense here.
        self.replace_slot(neg_instance, target_slot, wrong_id, wrong_text)
        return neg_instance, wrong_id

    def inside_cloze(self, sampler, origin_event_args, target_slot):
        """
        A negative cloze instance that use arguments within the event.
        :param sampler: A random sampler.
        :param origin_event_args:
        :param target_slot:
        :return:
        """
        neg_instance = {}

        for slot, content in origin_event_args.items():
            neg_instance[slot] = {}
            neg_instance[slot].update(content)

        swap_slot = sampler.sample_ignore_item(
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
