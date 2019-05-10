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
from event.arguments.prepare.slot_processor import get_dep_position


def __inverse_vocab(vocab):
    inverted = [0] * vocab.get_size()
    for w, index in vocab.vocab_items():
        inverted[index] = w
    return inverted


class HashedClozeReader:
    def __init__(self, resources, para, gpu=True):
        """
        Reading the hashed dataset into cloze tasks.
        :param resources: Resources containing vocabulary and more.
        :param gpu: Whether to run on gpu.
        """
        self.para = para

        self.event_emb_vocab = resources.event_embed_vocab
        self.word_emb_vocab = resources.word_embed_vocab

        self.pred_count = resources.predicate_count
        self.typed_event_vocab = resources.typed_event_vocab

        # Inverted vocab for debug purpose.
        self.event_inverted = __inverse_vocab(self.event_emb_vocab)

        # Some extra embeddings.
        self.unobserved_fe = self.event_emb_vocab.add_extra('__unobserved_fe__')
        self.unobserved_arg = self.event_emb_vocab.add_extra(
            '__unobserved_arg__')
        self.ghost_component = self.event_emb_vocab.add_extra(
            '__ghost_component__')

        self.slot_names = ['subj', 'obj', 'prep', ]

        # In fix slot mode we assume there is a fixed number of slots.
        if para.arg_representation_method == 'fix_slots':
            self.fix_slot_mode = True
        elif para.arg_representation_method == 'role_dynamic':
            self.fix_slot_mode = False

        self.gold_role_field = None
        self.auto_test = False

        self.__data_types = {
            # 'context': np.int64,
            'context_events': np.int64,
            'context_slots': np.int64,
            'context_slot_values': np.int64,
            'context_predicates': np.int64,
            'event_indices': np.int64,
            # The slot index is considered as a feature as well, so it has the
            # same type as the features.
            'slot_indices': np.int64,
            'event': np.int64,
            'slots': np.int64,
            'slot_values': np.int64,
            'predicates': np.int64,
            'distances': np.float32,
            'features': np.float32,
        }

        self.__data_dim = {
            'context': 2,
            'context_events': 2,
            'context_slots': 2,
            'context_slot_values': 2,
            'context_predicates': 2,
            'event_indices': 1,
            'slot_indices': 1,
            'event': 2,
            'slots': 2,
            'slot_values': 2,
            'predicates': 2,
            'distances': 2,
            'features': 2,
        }

        self.device = torch.device(
            "cuda" if gpu and torch.cuda.is_available() else "cpu"
        )

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
            if key.startswith('context_'):
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

        max_context_size = 0
        max_instance_size = 0
        doc_count = 0

        def _clear():
            # Clear batch data.
            nonlocal b_common_data, b_instance_data, max_context_size
            nonlocal max_instance_size, doc_count

            # Reset counts.
            b_common_data.clear()
            b_instance_data.clear()

            max_context_size = 0
            max_instance_size = 0

        for instance_data, common_data in self.parse_docs(
                data_in, sampler, from_line, until_line):
            for key, value in common_data.items():
                b_common_data[key].append(value)
                if key.startswith('context_'):
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
                    # 'predicate': batch_predicates,
                }

                train_batch = self.create_batch(
                    b_common_data, b_instance_data, max_context_size,
                    max_instance_size
                )

                yield train_batch, debug_data
                _clear()

        # Yield the remaining data.
        if len(b_common_data) > 0:
            debug_data = {}
            train_batch = self.create_batch(b_common_data, b_instance_data,
                                            max_context_size, max_instance_size)

            yield train_batch, debug_data
            _clear()

    def _take_event_repr(self, predicate, frame_id, args):
        if self.fix_slot_mode:
            return self._take_fixed_size_event_parts(predicate, frame_id, args)
        else:
            return self._take_dynamic_event_parts(predicate, frame_id, args)

    def _take_dynamic_event_parts(self, predicate, frame_id, args):
        """
        Take event information, with unknown number of slots.
        :param predicate:
        :param frame_id:
        :param args:
        :return:
        """
        pred_components = [
            predicate,
            self.event_emb_vocab.get_index(
                self.typed_event_vocab.unk_frame, None
            ) if frame_id == -1 else frame_id,
        ]

        slot_comps = []
        slot_value_comps = []

        # The slot will need to be indexed vocabularies, i.e. frame elements.
        # And they need to be hashed to number first.
        for slot, arg in args:
            slot_comps.append(slot)
            slot_value_comps.append(arg)

        return {
            'predicates': pred_components,
            'slots': slot_comps,
            'slot_values': slot_value_comps,
        }

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

        return {
            'events': event_components,
        }

    def parse_hashed(self):
        pass

    def parse_origin(self):
        pass

    @staticmethod
    def collect_features(doc_info):
        # Collect all features.
        features_by_eid = {}
        # entity_heads = {}

        for eid, content in doc_info['entities'].items():
            features_by_eid[int(eid)] = content['features']
            # entity_heads[int(eid)] = content['entity_head']

        return features_by_eid

    def create_slot_candidates(self, target_arg, doc_args, target_evm_id,
                               pred_sent):
        # Replace the target slot with other entities in the doc.
        dist_arg_list = []

        has_true_arg = False
        for doc_arg in doc_args:
            if (self.auto_test and doc_arg['event_index'] == target_evm_id and
                    doc_arg['entity_id'] == target_arg['entity_id']):
                # During auto test, we will not use the original argument
                continue

            # This is the target argument replaced by another entity.
            update_arg = self.replace_slot_detail(
                target_arg,
                doc_arg,
            )

            if self.auto_test:
                is_correct = doc_arg['entity_id'] == target_arg['entity_id']
            else:
                if doc_arg[self.gold_role_field] == target_arg[
                    self.gold_role_field] and \
                        doc_arg['event_index'] == target_evm_id:
                    is_correct = True
                else:
                    is_correct = False

            if is_correct:
                # Indicte whether there is a possible true arg here.
                has_true_arg = True

            dist_arg_list.append((
                abs(pred_sent - doc_arg['sentence_id']),
                (update_arg, doc_arg['entity_id'], is_correct)
            ))

        # Sort the rank list based on the distance to the target evm.
        dist_arg_list.sort(key=itemgetter(0))
        return [a for (dist, a) in dist_arg_list], has_true_arg

    def get_args_as_list(self, event_args, ignore_implicit):
        """
        Take a argument map and return a list version of it. It will take the
        first argument when multiple ones are presented at the slot.
        :param event_args:
        :param ignore_implicit:
        :return:
        """
        # Take args that are not implicit only.
        args = []
        if self.fix_slot_mode:
            for slot in self.slot_names:
                for arg in event_args[slot]:
                    if ignore_implicit and arg.get('implicit', False):
                        continue
                    else:
                        args.append((slot, arg))
                        break
                else:
                    args.append((slot, {}))
        else:
            for slot, l_arg in event_args.items():
                for a in l_arg:
                    if ignore_implicit and a.get('implicit', False):
                        continue
                    else:
                        # Reading dynamic slots using FEs.
                        args.append((a['fe'], a))
        return args

    def get_one_test_doc(self, doc_info, nid_detector):
        """
        Parse and get one test document.
        :param doc_info: The JSON data of one document.
        :param nid_detector: NID detector to detect which slot to fill.
        :return:
        """
        # Collect information such as features and entity positions.
        features_by_eid = self.collect_features(doc_info)

        # The context used for resolving.
        all_event_reps = [
            self._take_event_repr(
                e['predicate'], e['frame'],
                self.get_args_as_list(e['args'], True)) for
            e in doc_info['events']
        ]

        # Some need to be done in iteration.
        entity_positions = defaultdict(list)
        doc_args = []

        for evm_index, event in enumerate(doc_info['events']):
            sentence_id = event.get('sentence_id', None)

            for slot, l_arg in event['args'].items():
                # We iterate over all the arguments to collect distance data and
                # candidate document arguments.
                for arg in l_arg:
                    if len(arg) > 0:
                        # Empty args will be ignored.
                        # If one wanted to fill some arguments, make a
                        # fake argument first.

                        eid = arg['entity_id']
                        doc_arg_info = {
                            'event_index': evm_index,
                            # 'slot': slot,
                            'entity_id': eid,
                            'arg_phrase': arg['arg_phrase'],
                            # The sentence id is used to sort the candidate by
                            # its distance.
                            'sentence_id': arg['sentence_id'],
                            'represent': arg['represent'],
                            'text': arg['text'],
                            'dep': arg['dep'],
                        }
                        if not self.auto_test:
                            doc_arg_info[self.gold_role_field] = arg[
                                self.gold_role_field
                            ]
                        doc_args.append(doc_arg_info)

                        if not arg.get('implicit', False):
                            # We do not calculate distance features for implicit
                            # arguments.
                            entity_positions[eid].append(
                                (evm_index, slot, sentence_id)
                            )

        for evm_index, event in enumerate(doc_info['events']):
            pred_sent = event['sentence_id']
            pred_idx = event['predicate']
            predicate = (
                pred_idx, self.event_inverted[pred_idx],
                event['predicate_text']
            )
            event_args = event['args']

            for target_slot, target_args in event_args.items():
                can_fill = False
                fill_idx = -1

                for idx, slot_arg in enumerate(target_args):
                    if nid_detector.should_fill(event, target_slot, slot_arg):
                        can_fill = True
                        fill_idx = idx
                        break

                if can_fill:
                    test_rank_list, has_true = self.create_slot_candidates(
                        target_args[fill_idx], doc_args, evm_index, pred_sent
                    )

                    # Prepare instance data for each possible instance.
                    if self.fix_slot_mode:
                        instance_data = {'events': [], 'distances': [],
                                         'features': [], }
                    else:
                        instance_data = {
                            'predicates': [],
                            'slots': [],
                            'slot_values': [],
                            'distances': [],
                            'features': [],
                        }

                    candidate_meta = {'predicate': [], 'entity_text': [], }
                    instance_meta = []

                    cloze_event_indices = []
                    cloze_slot_indices = []
                    gold_labels = []

                    num_golds = len([1 for (_, _, c) in test_rank_list if c])

                    for cand_arg, filler_eid, is_correct in test_rank_list:
                        # Build the arg list from the generated candidate arg.
                        candidate_args = {}
                        for s, arg in event_args.items():
                            if s == target_slot:
                                candidate_args[s] = [cand_arg]
                            else:
                                candidate_args[s] = arg

                        self.assemble_instance(
                            instance_data, features_by_eid, entity_positions,
                            evm_index, pred_sent, pred_idx, event['frame'],
                            candidate_args, filler_eid,
                        )

                        cloze_event_indices.append(evm_index)
                        cloze_slot_indices.append(
                            self.slot_names.index(target_slot)
                        )

                        if is_correct:
                            gold_labels.append(1)
                        else:
                            gold_labels.append(0)

                        candidate_meta['predicate'].append(predicate)
                        candidate_meta['entity_text'].append(
                            cand_arg['represent']
                        )

                        if len(cloze_event_indices) == 500:
                            break

                    num_gold_in_scope = sum(gold_labels)

                    if num_gold_in_scope < num_golds:
                        logging.debug(
                            f"{num_golds - num_gold_in_scope} gold label are "
                            f"out of scope out of {num_golds}, found in "
                            f"doc: {doc_info['docid']},"
                            f"predicate: {event['predicate_text']}, "
                            f"slot: {target_slot}"
                        )

                    if num_gold_in_scope == 0 and has_true:
                        logging.warning(
                            f"No gold within scope for "
                            f"predicate {predicate}")

                    instance_meta.append(
                        {
                            'gold_entity': target_args[fill_idx]['text'],
                            'has_true': has_true,
                        }
                    )

                    common_data = {
                        'event_indices': cloze_event_indices,
                        'slot_indices': cloze_slot_indices,
                    }

                    for key, value in all_event_reps:
                        common_data['context_' + key] = value

                    if len(cloze_event_indices) > 0:
                        yield (instance_data, common_data, gold_labels,
                               candidate_meta, instance_meta)

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
                (instances, common_data, gold_labels, candidate_meta,
                 instance_meta,) = test_data

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
                    candidate_meta,
                    instance_meta,
                )

    def create_training_data(self, data_line, sampler):
        doc_info = json.loads(data_line)
        features_by_eid = self.collect_features(doc_info)

        # Map from: entity id (eid) ->
        # A list of tuples that represent an argument position:
        # [(evm_index, slot, sentence_id)]
        entity_positions = defaultdict(list)

        # A set that contains some minimum argument entity information, used
        # for sampling a slot to create clozes.
        t_doc_args = []

        # Count the occurrences of the entity.
        eid_count = Counter()
        event_subset = []

        for evm_index, event in enumerate(doc_info['events']):
            if evm_index == self.para.max_events:
                # Ignore documents that are too long.
                break

            # Only a subset in a long document will be used for generating.
            event_subset.append(event)

            sentence_id = event.get('sentence_id', None)

            for slot, arg in self.get_args_as_list(event['args'], False):
                if len(arg) > 0:
                    # Argument for n-th event, at slot position 'slot'.
                    eid = arg['entity_id']

                    # From eid to entity information.
                    entity_positions[eid].append((evm_index, slot, sentence_id))
                    t_doc_args.append({
                        'event_index': evm_index,
                        # 'slot': slot,
                        'entity_id': eid,
                        'arg_phrase': arg['arg_phrase'],
                        'represent': arg['represent'],
                        'text': arg['text'],
                        'dep': arg['dep'],
                    })
                    eid_count[eid] += 1

        all_event_reps = [
            self._take_event_repr(
                e['predicate'], e['frame'],
                self.get_args_as_list(e['args'], True)) for
            e in event_subset
        ]

        common_data = {
            'context_events': all_event_reps,
            'cross_event_indices': [],
            'cross_slot_indices': [],
            'inside_event_indices': [],
            'inside_slot_indices': [],
        }

        if len(t_doc_args) <= 1:
            # There no enough arguments to sample from.
            return None

        # We current sample the predicate based on unigram distribution.
        # The other learning strategy is to select one difficult cross instance,
        # the options are:
        # 1. Unigram distribution to sample items.
        # 2. Select items based on classifier output.

        # Maybe these 3 list can be merged together.
        cross_gold_standard = {'events': [], 'distances': [], 'features': []}
        cross_event_data = {'events': [], 'distances': [], 'features': []}

        inside_gold_standard = {'events': [], 'distances': [], 'features': []}
        inside_event_data = {'events': [], 'distances': [], 'features': []}

        for evm_index, event in enumerate(event_subset):
            pred = event['predicate']
            if pred == self.event_emb_vocab.get_index(
                    self.typed_event_vocab.unk_predicate, None):
                continue

            pred_tf = self.event_emb_vocab.get_term_freq(pred)
            freq = 1.0 * pred_tf / self.pred_count

            if not sampler.subsample_pred(pred_tf, freq):
                # Too frequent word will be down-sampled.
                continue

            current_sent = event['sentence_id']

            arg_list = self.get_args_as_list(event['args'], False)

            for arg_index, (slot, arg) in enumerate(arg_list):
                # unks are not very resolvable.
                if arg['represent'] == self.typed_event_vocab.unk_arg_word:
                    continue

                # unk fe is not a purpose here.
                if not self.fix_slot_mode and \
                        slot == self.event_emb_vocab.get_index(
                        self.typed_event_vocab.unk_fe,
                        self.typed_event_vocab.unk_fe):
                    continue

                correct_id = arg['entity_id']

                is_singleton = False
                if eid_count[correct_id] <= 1:
                    # Only one mention for this one.
                    is_singleton = True

                # If we do not train on singletons, we skip
                if is_singleton and not self.para.use_singleton:
                    continue

                cross_sample = self.cross_cloze(
                    sampler, arg_list, t_doc_args, evm_index, arg_index
                )

                inside_sample = self.inside_cloze(sampler, arg_list, arg_index)

                if inside_sample:
                    pprint(arg_list)
                    print('inside sample')
                    pprint(inside_sample)

                    input('what samples.')

                slot_index = self.slot_names.index(slot) if \
                    self.fix_slot_mode else slot

                if cross_sample:
                    cross_args, cross_filler_id = cross_sample
                    self.assemble_instance(
                        cross_event_data, features_by_eid, entity_positions,
                        evm_index, current_sent, pred, event['frame'],
                        cross_args, cross_filler_id
                    )
                    if is_singleton:
                        # If it is a singleton, than the ghost instance should
                        # be higher than randomly placing any entity here.
                        self.add_ghost_instance(cross_gold_standard)
                    else:
                        self.assemble_instance(
                            cross_gold_standard, features_by_eid,
                            entity_positions,
                            evm_index, current_sent, pred, event['frame'],
                            arg_list,
                            correct_id
                        )
                    # These two list indicate where the target argument is.
                    # When we use fix slot mode, the index is a indicator of
                    # the slot position. Otherwise, the slot_index is a way to
                    # represent the open-set slot types (such as FEs)
                    common_data['cross_event_indices'].append(evm_index)
                    common_data['cross_slot_indices'].append(slot_index)

                if inside_sample:
                    inside_args, inside_filler_id = inside_sample
                    self.assemble_instance(
                        inside_event_data, features_by_eid, entity_positions,
                        evm_index, current_sent, pred, event['frame'],
                        inside_args, inside_filler_id
                    )

                    # Inside sample can be used on singletons.
                    self.assemble_instance(
                        inside_gold_standard, features_by_eid,
                        entity_positions,
                        evm_index, current_sent, pred, event['frame'],
                        arg_list,
                        correct_id
                    )
                    common_data['inside_event_indices'].append(evm_index)
                    common_data['inside_slot_indices'].append(slot_index)

        if common_data['cross_slot_indices'] == 0 or \
                common_data['inside_slot_indices'] == 0:
            # Too few training instance.
            return None

        instance_data = {
            'cross_gold': cross_gold_standard,
            'cross': cross_event_data,
            'inside_gold': inside_gold_standard,
            'inside': inside_event_data,
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

        for current_slot, slot_info in arg_list:
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
                          predicate, frame, grouped_args, filler_eid):
        for key, value in self._take_event_repr(
                predicate, frame, self.get_args_as_list(grouped_args, False)):
            instance_data[key].append(value)

        instance_data['features'].append(features_by_eid[filler_eid])

        instance_data['distances'].append(
            self.get_distance_signature(
                evm_index, entity_positions,
                self.get_args_as_list(grouped_args, False), sent_id
            ))

    def add_ghost_instance(self, instance_data):
        if self.fix_slot_mode:
            instance_data['event'].append(
                [self.ghost_component] * self.para.num_event_component)
        else:
            instance_data['predicates'].append(
                self.ghost_component
            )
            instance_data['slot'].append(
                self.ghost_component
            )
            instance_data['slot_values'].append(
                self.ghost_component
            )

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

    def replace_slot_detail(self, original_slot_info, new_slot_info,
                            replace_fe=False):
        # Make a copy of the slot.
        updated_slot_info = dict((k, v) for k, v in original_slot_info.items())

        # Replace with the new information.
        updated_slot_info['entity_id'] = new_slot_info['entity_id']
        updated_slot_info['represent'] = new_slot_info['represent']
        updated_slot_info['arg_phrase'] = new_slot_info['arg_phrase']

        origin_dep = get_dep_position(original_slot_info['dep'])
        new_arg_rep = self.typed_event_vocab.get_arg_rep(
            origin_dep, new_slot_info['represent'])

        updated_slot_info['arg_role'] = self.event_emb_vocab.get_index(
            new_arg_rep, self.typed_event_vocab.get_unk_arg_rep()
        )

        if replace_fe:
            updated_slot_info['fe'] = new_slot_info['fe']
        return updated_slot_info

    def cross_cloze(self, sampler, arg_list, doc_args, target_evm_id,
                    target_arg_idx):
        """
        A negative cloze instance that use arguments from other events.
        :param sampler: A random sampler.
        :param arg_list: List of origin event arguments.
        :param doc_args: List of arguments (partial info) in this doc.
        :param target_evm_id: The target event id.
        :param target_arg_idx: The index for the slot to be replaced
        :return:
        """
        _, target_arg = arg_list[target_arg_idx]
        target_eid = target_arg['entity_id']

        sample_res = sampler.sample_cross(
            doc_args, target_evm_id, target_eid
        )

        if sample_res is None:
            return None

        neg_instance = {}
        for idx, (slot, content) in enumerate(arg_list):
            if idx == target_arg_idx:
                neg_instance[slot] = self.replace_slot_detail(
                    content, sample_res)
            else:
                neg_instance[slot] = content

        return neg_instance, sample_res['entity_id']

    def inside_cloze(self, sampler, arg_list, arg_index):
        """
        A negative cloze instance that use arguments within the event.
        :param sampler: A random sampler.
        :param arg_list: The Arg List of the original event.
        :param arg_index: The current index of the argument.
        :return:
        """
        if len(arg_list) < 2:
            return None

        sample_indexes = list(range(arg_index)) + list(
            range(arg_index, len(arg_list)))

        swap_index = sampler.sample_list(sample_indexes)

        _, origin_slot_info = arg_list[arg_index]
        _, swap_slot_info = arg_list[swap_index]

        # TODO: shall we?
        # When swapping the two slots, we also swap the FEs
        # this help us learn the correct slot for a FE.

        neg_instance = {}
        for idx, (slot, content) in enumerate(arg_list):
            if idx == arg_index:
                # Replace the swap one here.
                neg_instance[slot] = self.replace_slot_detail(
                    content,
                    swap_slot_info,
                )
            elif idx == swap_index:
                # Replace the original one here
                neg_instance[slot] = self.replace_slot_detail(
                    content,
                    origin_slot_info
                )
            else:
                neg_instance[slot] = content

        return neg_instance, origin_slot_info['entity_id']
