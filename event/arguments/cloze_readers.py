import json
import logging
from collections import Counter
from collections import defaultdict

import numpy as np
import torch

from event.arguments.util import (batch_combine, to_torch)
from event.io.io_utils import pad_2d_list
from pprint import pprint
from operator import itemgetter


class HashedClozeReader:
    def __init__(self, resources, para, gpu=True):
        """
        Reading the hashed dataset into cloze tasks.
        :param resources: Resources containing vocabulary and more.
        :param gpu: Whether to run on gpu.
        """
        self.para = para

        # self.batch_size = para.batch_size
        # self.multi_context = para.multi_context
        # self.max_events = para.max_events
        # self.max_instance = para.max_cloze
        # self.num_event_component = para.num_event_components
        # self.num_distance_features = para.num_distance_features
        # self.num_extracted_features = para.num_extracted_features

        self.event_emb_vocab = resources.event_embed_vocab
        self.word_emb_vocab = resources.word_embed_vocab

        self.pred_count = resources.predicate_count
        self.typed_event_vocab = resources.typed_event_vocab

        # Inverted vocab for debug purpose.
        self.event_inverted = self.__inverse_vocab(self.event_emb_vocab)

        # Some extra embeddings.
        self.unobserved_fe = self.event_emb_vocab.add_extra('__unobserved_fe__')
        self.unobserved_arg = self.event_emb_vocab.add_extra(
            '__unobserved_arg__')
        self.ghost_component = self.event_emb_vocab.add_extra(
            '__ghost_component__')

        self.slot_names = ['subj', 'obj', 'prep', ]

        # In fix slot mode we assume there is a fixed number of slots.
        self.fix_slot_mode = True

        self.gold_role_field = None
        self.auto_test = False

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
        inverted = [0] * vocab.get_size()
        for w, index in vocab.vocab_items():
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

            # Take a look at the predicates for debug purpose only.
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

            # Each document is computed as a whole.
            if doc_count % self.para.batch_size == 0:
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

    def _take_fixed_size_event_parts(self, predicate, frame_id, args):
        """
        Take event information from the data, one element per slot, hence the
        size of the event parts is fixed.
        :param predicate:
        :param frame_id:
        :param args:
        :return:
        """
        # frame_id = event_info['frame']
        event_components = [
            predicate,
            self.event_emb_vocab.get_index(
                self.typed_event_vocab.unk_frame, None
            ) if frame_id == -1 else frame_id
        ]

        for slot, arg in args:
            if len(arg) == 0:
                event_components.append(self.unobserved_fe)
                event_components.append(self.unobserved_arg)
            else:
                event_components.append(arg['fe'])
                event_components.append(arg['arg_role'])

        if any([c < 0 for c in event_components]):
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

    def __group_by_gold_role(self, args, gold_role_value):
        grouped_args = defaultdict(list)

        for arg in args:
            print(arg)
            arg_role = arg[self.gold_role_field]
            if arg_role == gold_role_value:
                grouped_args[arg_role].append(arg)
        return grouped_args

    def create_actual_test_instance(self, this_args, doc_args,
                                    abs_target_arg_idx):
        target_arg_info = doc_args[abs_target_arg_idx]

        for doc_arg in doc_args:
            # doc_args.append({
            #     'event_index': evm_index,
            #     'relative_arg_index': rel_arg_index,
            #     'eid': eid,
            #     'arg_phrase': arg['arg_phrase'],
            #     'represent': arg['represent']
            # })

            arg_by_role = self.__group_by_gold_role(
                this_args, target_arg_info[self.gold_role_field]
            )

            for role, l_arg_info in self.__group_by_gold_role(
                    this_args, target_arg_info[self.gold_role_field]).items():
                # Replace the whole role with an candidate doc arg.
                print(role)
                print(l_arg_info)

                input('check creating test data')

                pass

    def create_test_instance(self, this_args, doc_args, abs_target_arg_idx,
                             target_slot):
        dist_test_rank_list = []

        target_arg_info = doc_args[abs_target_arg_idx]

        target_arg_index = target_arg_info['relative_arg_index']
        target_evm_id = target_arg_info['event_index']
        target_eid = target_arg_info['eid']

        # Replace the target slot with other entities for generating this.
        for evm_id, ent_id, arg_text, arg_represent in doc_args:
            if (self.auto_test and evm_id == target_evm_id and
                    ent_id == target_eid):
                # During auto test, we will not use the original argument
                continue

            created_args = []

            for arg_index, arg_info in enumerate(this_args):
                if arg_index == target_arg_index:
                    # Replace the data in the target slot with another entity.
                    update_arg = self.replace_slot_detail(
                        arg_info,
                        target_slot,
                        ent_id,
                        arg_text,
                        arg_represent,
                    )
                    created_args.append(update_arg)
                else:
                    created_args.append(arg_info)

            dist_test_rank_list.append(
                (abs(target_evm_id - evm_id), (created_args, ent_id)))

        # Sort the rank list based on the distance to the target evm.
        dist_test_rank_list.sort(key=itemgetter(0))

        return [b for (a, b) in dist_test_rank_list]

    def get_args_anyway(self, event_args):
        args = []
        if self.fix_slot_mode:
            for slot, l_arg in event_args.items():
                pprint(slot)

                if len(l_arg) > 0:
                    args.append((slot, l_arg[0]))
                else:
                    args.append((slot, {}))

                print()
        else:
            for slot, l_arg in event_args.items():
                for arg in l_arg:
                    args.append((slot, arg))
        return args

    def get_one_test_doc(self, doc_info, nid_detector):
        """
        Parse and get one test document.
        :param doc_info: The JSON data of one document.
        :param nid_detector: NID detector to detect which slot to fill.
        :return:
        """
        # Collect information such as features and entity positions.
        features_by_eid, entity_heads = self.collect_features(doc_info)

        all_event_reps = [
            self._take_fixed_size_event_parts(
                e['predicate'], e['frame'], self.get_args_anyway(e['args'])) for
            e in doc_info['events']]

        # Some need to be done in iteration.
        entity_positions = defaultdict(list)
        doc_args = []
        indexed_data = []

        for evm_index, event in enumerate(doc_info['events']):
            sentence_id = event.get('sentence_id', None)

            valid_args = []

            for rel_arg_index, (slot, arg) in enumerate(
                    self.get_args_anyway(event['args'])):
                if len(arg) > 0:
                    # TODO: Empty args will be ignored.
                    # If one wanted to fill some arguments, please make an fake
                    # argument first.
                    eid = arg['entity_id']
                    doc_arg_info = {
                        'event_index': evm_index,
                        'relative_arg_index': rel_arg_index,
                        'eid': eid,
                        'arg_phrase': arg['arg_phrase'],
                        'represent': arg['represent'],
                    }

                    if not self.auto_test:
                        doc_arg_info[self.gold_role_field] = arg[
                            self.gold_role_field
                        ]

                    doc_args.append(doc_arg_info)

                    entity_positions[eid].append((evm_index, slot, sentence_id))
                    valid_args.append((slot, arg))

            indexed_data.append(
                (event, valid_args)
            )

        absolute_arg_index = 0
        for evm_index, (event, valid_args) in enumerate(indexed_data):
            pred_sent = event['sentence_id']
            pred_idx = event['predicate']
            predicate = (
                pred_idx, self.event_inverted[pred_idx],
                event['predicate_text']
            )

            for target_arg_index, (slot, fe, arg) in enumerate(valid_args):
                absolute_arg_index += 1

                print(slot)
                print(fe)
                pprint(arg)

                is_instance = nid_detector.should_fill(event, slot, arg)

                if is_instance:
                    if self.auto_test:
                        test_rank_list = self.create_test_instance(
                            valid_args, doc_args, absolute_arg_index, slot
                        )
                    else:
                        test_rank_list = self.create_actual_test_instance(
                            valid_args, doc_args, absolute_arg_index
                        )

                    # Prepare instance data for each possible instance.
                    instance_data = {'rep': [], 'distances': [],
                                     'features': [], }
                    debug_data = {'predicate': [], 'entity_text': [],
                                  'gold_entity': [], }

                    cloze_event_indices = []
                    cloze_slot_indices = []
                    gold_labels = []

                    has_gold = False

                    for cand_args, filler_eid in test_rank_list:
                        self.assemble_instance(
                            instance_data, features_by_eid, entity_positions,
                            evm_index, pred_sent, pred_idx, event['frame'],
                            cand_args, filler_eid,
                        )

                        cloze_event_indices.append(evm_index)
                        cloze_slot_indices.append(self.slot_names.index(slot))

                        if filler_eid == arg['entity_id']:
                            gold_labels.append(1)
                            has_gold = True
                        else:
                            gold_labels.append(0)

                        debug_data['predicate'].append(predicate)
                        debug_data['entity_text'].append(
                            entity_heads[filler_eid])
                        debug_data['gold_entity'].append(
                            (arg['entity_id'],
                             entity_heads[arg['entity_id']])
                        )

                        if len(cloze_event_indices) > 500:
                            logging.warning(
                                "Some test instances are ignored due to length")
                            break

                    if not has_gold:
                        logging.warning(
                            f"No gold label for {doc_info['docid']},"
                            f"predicate {event['predicate_text']}, slot {slot}"
                        )

                        pprint(predicate)
                        print(evm_index)
                        pprint(valid_args[target_arg_index])

                        input('check unfound gold.')

                    common_data = {
                        'context': all_event_reps,
                        'event_indices': cloze_event_indices,
                        'slot_indices': cloze_slot_indices,
                    }

                    if len(cloze_event_indices) > 0:
                        yield (instance_data, common_data, gold_labels,
                               debug_data)

        # if len(cloze_event_indices) == 0:
        #     return None

        # return instance_data, common_data, gold_labels, debug_data

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

            for test_data in self.get_one_test_doc(doc_info, nid_detector):
                instances, common_data, gold_labels, debug_data = test_data

                b_common_data = {}
                b_instance_data = {}

                # Create a pseudo batch with one instance.
                for key, value in common_data.items():
                    vectorized = to_torch([value], self.__data_types[key])
                    b_common_data[key] = batch_combine(vectorized, self.device)

                for key, value in instances.items():
                    vectorized = to_torch([value], self.__data_types[key])
                    b_instance_data[key] = batch_combine(vectorized,
                                                         self.device)

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
            if evm_index == self.para.max_events:
                # Ignore documents that are too long.
                break

            event_subset.append(event)

            sentence_id = event.get('sentence_id', None)

            arg_info = {}
            for slot, arg in self.get_args_anyway(event['args']):
                if not arg or arg['entity_id'] == -1:
                    arg_info[slot] = {}
                else:
                    # Argument for n-th event, at slot position 'slot'.
                    eid = arg['entity_id']

                    # From eid to entity information.
                    entity_positions[eid].append((evm_index, slot, sentence_id))
                    arg_entities.add((evm_index, eid, arg['text']))
                    eid_count[eid] += 1

        all_event_reps = [
            self._take_fixed_size_event_parts(
                e['predicate'], e['frame'], self.get_args_anyway(e['args'])) for
            e in event_subset
        ]

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

        for evm_index, event in enumerate(event_subset):
            pred = event['predicate']
            if pred == self.event_emb_vocab.get_index(
                    self.typed_event_vocab.unk_predicate, None):
                continue

            pred_tf = self.event_emb_vocab.get_term_freq(pred)
            freq = 1.0 * pred_tf / self.pred_count
            keep_pred = sampler.subsample_pred(pred_tf, freq)

            if not keep_pred:
                # Too frequent word will be down-sampled.
                continue

            current_sent = event['sentence_id']

            for slot_index, slot in enumerate(self.slot_names):
                arg = event['args'][slot]

                correct_id = arg.get('entity_id', -1)

                # unks are not resolvable.
                if correct_id < 0 or entity_heads[
                    correct_id] == self.typed_event_vocab.unk_arg_word:
                    continue

                is_singleton = False
                if eid_count[correct_id] <= 1:
                    # Only one mention for this one.
                    is_singleton = True

                cross_sample = self.cross_cloze(
                    sampler, self.get_args_anyway(event['args']),
                    arg_entities, evm_index, slot, entity_heads
                )

                inside_sample = self.inside_cloze(
                    sampler, event['args'], slot
                )

                # TODO: Can the samples of different sizes?

                if cross_sample:
                    cross_args, cross_filler_id = cross_sample

                    self.assemble_instance(
                        cross_event_data, features_by_eid, entity_positions,
                        evm_index, current_sent, pred, event['frame'],
                        cross_args, cross_filler_id
                    )

                if inside_sample:
                    inside_args, inside_filler_id = inside_sample

                    self.assemble_instance(
                        inside_event_data, features_by_eid, entity_positions,
                        evm_index, current_sent, pred, event['frame'],
                        inside_args, inside_filler_id
                    )

                if is_singleton:
                    # If it is a singleton, than the highest instance should be
                    # the ghost instance.
                    self.add_ghost_instance(gold_event_data)
                else:
                    self.assemble_instance(
                        gold_event_data, features_by_eid, entity_positions,
                        evm_index, current_sent, event, correct_id)

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

    def get_distance_signature(
            self, current_evm_id, entity_mentions, arg_list, sent_id):
        """
        Compute the distance signature of the instance's other mentions to the
        sentence.
        :param current_evm_id:
        :param entity_mentions:
        :param arg_list:
        :param sent_id:
        :return:
        """
        distances = []

        # Now use a large distance to represent Infinity.
        # Infinity: if the entity cannot be found again, or it is not an entity.
        # A number is arbitrarily decided since most document is shorter than
        # this.
        inf = 30.0

        for current_slot, fe, slot_info in arg_list:
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
            else:
                # This argument is not seen elsewhere, it should be a special
                # distance label.
                distances.append((inf, inf, inf))

        # Flatten the (argument x type) distances into a flat list.
        return [d for l in distances for d in l]

    def assemble_instance(self, instance_data, features_by_eid,
                          entity_positions, evm_index, sent_id,
                          predicate, frame, arg_list, filler_eid):
        instance_data['rep'].append(
            self._take_fixed_size_event_parts(
                predicate,
                frame,
                arg_list,
            )
        )
        instance_data['features'].append(features_by_eid[filler_eid])
        instance_data['distances'].append(
            self.get_distance_signature(
                evm_index, entity_positions, arg_list, sent_id
            ))

    def add_ghost_instance(self, instance_data):
        instance_data['rep'].append(
            [self.ghost_component] * self.para.num_event_component)
        instance_data['features'].append(
            [0.0] * self.para.num_extracted_features)
        instance_data['distances'].append(
            [0.0] * self.para.num_distance_features)

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

    def replace_slot_detail(self, slot_instance, new_slot, new_eid, new_text,
                            new_entity_rep, new_fe=None):
        # Make a copy of the slot.
        slot_name, slot_fe, slot_info = slot_instance

        updated_slot_info = dict((k, v) for k, v in slot_info.items())

        # Replace the slot info with the new information.
        updated_slot_info['entity_id'] = new_eid
        updated_slot_info['represent'] = new_entity_rep
        updated_slot_info['text'] = new_text
        updated_slot_info['arg_role'] = self.event_emb_vocab.get_index(
            self.typed_event_vocab.get_arg_rep(
                updated_slot_info.get('dep', new_slot), new_entity_rep
            ), self.typed_event_vocab.get_unk_arg_rep()
        )

        # Some training instance also change the frame element name.
        if new_fe is not None:
            updated_slot_info['fe'] = new_fe
            slot_fe = new_fe

        return slot_name, slot_fe, updated_slot_info

    def cross_cloze(self, sampler, args, arg_entities, target_evm_id,
                    target_slot, entity_heads):
        """
        A negative cloze instance that use arguments from other events.
        :param sampler: A random sampler.
        :param args: Dict of origin event arguments.
        :param arg_entities: List of arguments (event id, entity id, text).
        :param target_evm_id: The target event id.
        :param target_slot: The target slot name.
        :param entity_heads: A map from entity id to its represnet.
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
        for slot, content in args:
            neg_instance[slot] = {}
            neg_instance[slot].update(content)

        # When taking entity from another slot, we don't take its FE,
        # because the FE from another sentence will not make sense here.
        self.replace_slot_detail(neg_instance, target_slot, wrong_id,
                                 wrong_text, entity_heads[wrong_id])
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
            self.replace_slot_detail(
                neg_instance, target_slot, swap_slot_info['entity_id'],
                swap_slot_info['text'], swap_slot_info['represent'],
                swap_slot_info['fe'])
        else:
            neg_instance[target_slot] = {}

        if target_slot_info:
            self.replace_slot_detail(
                neg_instance, swap_slot, target_slot_info['entity_id'],
                target_slot_info['text'], target_slot_info['represent'],
                target_slot_info['fe'])
        else:
            neg_instance[swap_slot] = {}

        return neg_instance, target_slot_info['entity_id']
