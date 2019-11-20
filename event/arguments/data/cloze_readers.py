import copy
import json
import logging
from collections import Counter
from collections import defaultdict
from typing import Dict

import torch

from event.arguments.NIFDetector import NullArgDetector
from event.arguments.data.batcher import ClozeBatcher
from event.arguments.data.cloze_gen import ClozeGenerator, PredicateSampler, \
    TestClozeMaker, CandidateBuilder
from event.arguments.data.cloze_instance import ClozeInstances
from event.arguments.data.event_structure import EventStruct
from event.arguments.data.frame_data import FrameSlots
from event.arguments.implicit_arg_params import ArgModelPara
from event.arguments.implicit_arg_resources import ImplicitArgResources
from event import torch_util
import pdb

logger = logging.getLogger(__name__)

ghost_entity_text = '__ghost_component__'
ghost_entity_id = -1


class HashedClozeReader:
    """Reading the hashed dataset into cloze tasks.

    Args:
      resources: Resources containing vocabulary and more.
      para: Config parameters.

    Returns:

    """

    def __init__(self, resources: ImplicitArgResources, para: ArgModelPara):
        self.para = para

        self.event_emb_vocab = resources.event_embed_vocab
        self.word_emb_vocab = resources.word_embed_vocab

        self.pred_count = resources.predicate_count
        self.typed_event_vocab = resources.typed_event_vocab

        self.unk_predicate_idx = self.event_emb_vocab.get_index(
            self.typed_event_vocab.unk_predicate, None)

        self.frame_slots = FrameSlots(self.para.slot_frame_formalism, resources)

        # Specify which role is used to modify the argument string in SRL.
        self.factor_role = self.para.factor_role

        # The cloze data are organized by the following slots.
        self.fix_slot_names = ['subj', 'obj', 'prep', ]

        # A set for data with variable length, we then will create paddings
        # and length tensor for it.
        self.__unk_length = set()

        # In fix slot mode we assume there is a fixed number of slots.
        if para.arg_representation_method == 'fix_slots':
            self.fix_slot_mode = True
        elif para.arg_representation_method == 'role_dynamic':
            self.fix_slot_mode = False
            self.__unk_length = {'slot', 'slot_value', 'context_slot',
                                 'context_slot_value'}

        self.gold_role_field = self.para.gold_field_name
        self.test_limit = 500

        self.device = torch.device(
            "cuda" if para.use_gpu and torch.cuda.is_available() else "cpu"
        )

        self.event_struct = EventStruct(
            resources.event_embed_vocab, resources.typed_event_vocab,
            para.use_frame, self.fix_slot_mode
        )
        self.predicate_sampler = PredicateSampler()
        self.candidate_builder = CandidateBuilder(
            resources.event_embed_vocab,
            resources.typed_event_vocab
        )
        self.cloze_gen = ClozeGenerator(self.candidate_builder,
                                        self.fix_slot_mode)

        logger.info("Reading data with device: " + str(self.device))

    def read_train_batch(self, data_in, sampler):
        logger.info("Reading data as training batch.")

        self.cloze_gen.set_sampler(sampler)

        train_batcher = ClozeBatcher(self.para.batch_size, self.device)

        for line in data_in:
            parsed_output = self.create_training_data(line)

            if parsed_output is None:
                continue

            yield from train_batcher.get_batch(*parsed_output)

        if train_batcher.doc_count == 0:
            raise ValueError("Provided data is empty.")

        # Yield the remaining data.
        yield train_batcher.flush()

    @staticmethod
    def collect_features(doc_info):
        # Collect all features.
        features_by_eid = {}
        for eid, content in doc_info['entities'].items():
            features_by_eid[int(eid)] = content['features']
        return features_by_eid

    @staticmethod
    def answer_from_arg(arg):
        return {
            'span': (arg['arg_start'], arg['arg_end']),
            'text': arg['arg_phrase']
        }

    def get_test_cases(self, event, possible_slots):
        test_cases = []

        # First organize the roles by the gold role name.
        arg_by_slot = defaultdict(list)
        for dep_slot, args_per_dep in event['args'].items():
            for arg in args_per_dep:
                if self.gold_role_field in arg:
                    gold_role = arg[self.gold_role_field]
                    if not gold_role == -1:
                        arg_by_slot[gold_role].append(arg)

        # There are different types of slots, given the scenario, some of them
        # can be considered as test cases.
        # 1. Empty slots, these are not fillable, but can be test case for
        # the system to determine whether to fill.
        # 2. Slot with implicit arguments.

        for slot in possible_slots:
            # Get a mapped dependency label for this slot, e.g. subj for arg0.
            dep = self.frame_slots.get_dep_from_slot(event, slot)

            no_fill_case = [
                slot,
                {
                    'resolvable': False,
                    'implicit': False,
                    'dep': dep,
                },
                [{
                    'span': (-1, -1),
                    'text': ghost_entity_text,
                }],
            ]

            if slot not in arg_by_slot:
                # TODO: here we skip the slot because the slot is not part of
                #  gold standard data. However, this is an candidate for
                #  non-fill case, which should be include in the production.
                continue

            answers = []

            # Take the annotated args in this slot.
            args = arg_by_slot[slot]

            for arg in args:
                if arg['implicit'] and arg['source'] == 'gold' \
                        and not arg['incorporated']:
                    # Case 2
                    answers.append(self.answer_from_arg(arg))

            if answers:
                test_stub = {
                    'resolvable': True,
                    'implicit': True,
                    'dep': dep,
                }
                test_cases.append([slot, test_stub, answers])
            else:
                for arg in args:
                    # We do not need to fill the explicit arguments.
                    if arg['source'] == 'gold' and not arg['implicit']:
                        break
                else:
                    # There are no explict and implicit either, but we need the
                    # system to guess whether it wanted to fill such case.
                    test_cases.append(copy.deepcopy(no_fill_case))

        return test_cases

    def get_args_as_list(self, event_args, ignore_implicit):
        """Take a argument map and return a list version of it. It will take the
        first argument when multiple ones are presented at the slot.

        Args:
          event_args: ignore_implicit:
          ignore_implicit: 

        Returns:

        """
        slot_args = {}
        for dep_slot, l_arg in event_args.items():
            for a in l_arg:
                if ignore_implicit and a.get('implicit', False):
                    continue
                else:
                    # Reading dynamic slots using the defined factor role.
                    if self.factor_role:
                        factor_role = a[self.factor_role]
                    else:
                        # If not defined, then used to dep slots as factor
                        # role.
                        factor_role = dep_slot

                    if factor_role not in slot_args:
                        slot_args[factor_role] = a
        args = list(slot_args.items())
        return args

    def get_one_test_doc(self, doc_info: Dict,
                         nid_detector: NullArgDetector,
                         test_cloze_maker: TestClozeMaker):
        """Parse and get one test document.

        Args:
          doc_info: The JSON data of one document.
          nid_detector: NID detector to detect which slot to fill.
          test_cloze_maker: TestClozeMaker

        Returns:

        """
        # Collect information such as features and entity positions.
        features_by_eid = self.collect_features(doc_info)

        # The context used for resolving.
        all_event_reps = [
            self.event_struct.event_repr(
                e['predicate'], e['frame'],
                self.get_args_as_list(e['args'], True)) for
            e in doc_info['events']
        ]

        # Record the entity positions
        explicit_entity_positions = defaultdict(dict)

        # Argument entity mentions: a mapping from the spans to the arguments,
        # this is useful since spans are specific while the mentions can be
        # shared by different events.
        arg_mentions = {}

        # For each event.
        for evm_index, event in enumerate(doc_info['events']):
            # For each dep based slot (subj, obj, prep), there might be a list
            # of arguments.
            for dep_slot, l_arg in event['args'].items():
                # We iterate over all the arguments to collect distance data and
                # candidate document arguments.
                for arg in l_arg:
                    if len(arg) == 0:
                        continue

                    eid = arg['entity_id']

                    # TODO: event_index, dep not copied.
                    doc_arg_info = self.copy_mention_info(arg)

                    if self.gold_role_field in arg:
                        doc_arg_info[self.gold_role_field] = arg[
                            self.gold_role_field
                        ]

                    arg_span = (arg['arg_start'], arg['arg_end'])
                    arg_mentions[arg_span] = doc_arg_info

                    if not arg.get('implicit', False):
                        # We do not calculate distance features for implicit
                        # arguments.
                        explicit_entity_positions[eid][arg_span] = arg[
                            'sentence_id']

        # This creates a list of candidate mentions for this document.
        doc_mentions = [v for v in arg_mentions.values()]

        for evm_index, event in enumerate(doc_info['events']):
            pred_sent = event['sentence_id']
            pred_id = event['predicate']
            event_args = event['args']
            arg_list = self.get_args_as_list(event_args, False)

            available_slots = self.frame_slots.get_predicate_slots(event)

            test_cases = self.get_test_cases(event, available_slots)

            for target_slot, test_stub, answers in test_cases:
                # The detector determine whether we should fill this slot.
                if nid_detector.should_fill(event, target_slot, test_stub):
                    # Fill the test_stub with all the possible args in this doc.
                    test_rank_list = test_cloze_maker.gen_test_args(
                        test_stub, doc_mentions, pred_sent,
                        distance_cap=self.para.distance_cap
                    )

                    answer_spans = set([a['span'] for a in answers])

                    if self.para.use_ghost:
                        # Put the ghost at the beginning.
                        test_rank_list.insert(0, ({}, ghost_entity_id))

                    # Prepare instance data for each possible instance.
                    instances = ClozeInstances(self.para, self.event_struct)

                    metadata = {
                        'candidate': [],
                        'instance': {},
                    }

                    cloze_event_indices = []
                    cloze_slot_indicator = []

                    # To avoid have too many test cases that blow up memory.
                    limit = min(self.test_limit, len(test_rank_list))

                    for cand_arg, filler_eid in test_rank_list[:limit]:
                        # Re-create the event arguments by adding the newly
                        # replaced one and the other old ones.
                        candidate_args = [(target_slot, cand_arg)]

                        # Add the remaining arguments.
                        for s, arg in arg_list:
                            if not s == target_slot:
                                candidate_args.append((s, arg))

                        cand_arg_span = cand_arg['arg_start'], cand_arg[
                            'arg_end']

                        label = 1 if cand_arg_span in answer_spans else 0

                        # Create the event instance representation.
                        instances.assemble_instance(
                            features_by_eid,
                            explicit_entity_positions,
                            pred_sent,
                            self.event_struct.event_repr(
                                pred_id, event['frame'], candidate_args
                            ),
                            filler_eid, label=label
                        )

                        cloze_event_indices.append(evm_index)
                        cloze_slot_indicator.append(
                            self.get_slot_index(target_slot))

                        if filler_eid == ghost_entity_id:
                            metadata['candidate'].append(
                                {'entity': ghost_entity_text,
                                 'span': (-1, -1)}
                            )
                        else:
                            metadata['candidate'].append({
                                'entity': cand_arg['represent'],
                                'event_index': evm_index,
                                'distance_to_event': (
                                        pred_sent - cand_arg['sentence_id']
                                ),
                                'span': (
                                    cand_arg['arg_start'], cand_arg['arg_end']
                                ),
                                'source': cand_arg['source']})

                    metadata['instance'] = {
                        'docid': doc_info['docid'],
                        'predicate': event['predicate_text'],
                        'predicate_id': pred_id,
                        'target_slot_id': target_slot,
                        'answers': answers,
                    }

                    common_data = {
                        'event_indices': cloze_event_indices,
                        'slot_indicators': cloze_slot_indicator,
                    }

                    for context_eid, event_rep in enumerate(all_event_reps):
                        # In fixed mode, the key is "events", that contain
                        # the representation for the event.

                        # In dynamic mode, the keys are "predicate",
                        # "slot", "slot_value".
                        for key, value in event_rep.items():
                            try:
                                common_data['context_' + key].append(value)
                            except KeyError:
                                common_data['context_' + key] = [value]

                    if len(cloze_event_indices) > 0:
                        yield instances, common_data, metadata

    def read_test_docs(self, test_in, nid_detector: NullArgDetector):
        """Load test data. Importantly, this will create alternative cloze
         filling for ranking.

        Args:
          test_in: supply lines as test data.
          nid_detector: Null Instantiation Detector.

        Returns:

        """
        # At test time we can use a single doc batch.
        batcher = ClozeBatcher(1, self.device)
        test_cloze_maker = TestClozeMaker(self.candidate_builder)

        for line in test_in:
            doc_info = json.loads(line)

            for test_data in self.get_one_test_doc(doc_info, nid_detector,
                                                   test_cloze_maker):
                yield from batcher.get_batch(*test_data)

    def get_slot_index(self, slot):
        if self.fix_slot_mode:
            return self.fix_slot_names.index(slot)
        else:
            # In dynamic slot mode, we provide the slot's
            # vocab id, which can be converted to embedding.
            return self.event_emb_vocab.get_index(
                slot, self.typed_event_vocab.unk_fe)

    @staticmethod
    def copy_mention_info(arg: Dict):
        # Some minimum information for creating cloze tests.
        # - Entity id is the target to predict.
        # - Arg phrase, represent or text are different ways to
        #   represent the argument mention.
        # - source distinguish gold and automatic predicted args.
        # - sentence_id is related to the distance based feature.
        # - arg start and end are used to identify this unique
        #   mention.
        copy_keys = ('entity_id', 'arg_phrase', 'represent', 'text',
                     'sentence_id', 'arg_start', 'arg_end')
        mention_info = dict([(k, arg[k]) for k in copy_keys])
        mention_info['source'] = arg.get('source', 'automatic')

        if 'fe' in arg:
            mention_info['fe'] = arg['fe']

        if 'ner' in arg:
            mention_info['ner'] = arg['ner']

        return mention_info

    def create_training_data(self, data_line):
        doc_info = json.loads(data_line)
        features_by_eid = self.collect_features(doc_info)

        # Map from: entity id (eid) ->
        # A list of tuples that represent an argument position:
        # [(evm_index, slot, sentence_id)]
        explicit_entity_positions = defaultdict(dict)

        # A dictionary of argument mentions, indexed by the span to ensure no
        # mention level duplications. We can then use these values to create
        # cloze tasks.
        arg_mentions = {}

        # Count the occurrences of the entity.
        eid_count = Counter()
        event_subset = []

        for evm_index, event in enumerate(doc_info['events']):
            if evm_index == self.para.max_events:
                # Skip the rest if the document is too long.
                break

            # Only a subset in a long document will be used for generating.
            event_subset.append(event)

            for slot, arg in self.get_args_as_list(event['args'], False):
                if len(arg) > 0:
                    # Argument for n-th event, at slot position 'slot'.
                    eid = arg['entity_id']
                    eid_count[eid] += 1

                    span = (arg['arg_start'], arg['arg_end'])

                    if not arg.get('implicit', False):
                        explicit_entity_positions[eid][span] = arg[
                            'sentence_id']

                    arg_mentions[span] = self.copy_mention_info(arg)

        # A set that contains some minimum argument entity mention information,
        # used for sampling a slot to create clozes.
        t_doc_args = [v for v in arg_mentions.values()]

        all_event_reps = [
            self.event_struct.event_repr(
                e['predicate'], e['frame'],
                self.get_args_as_list(e['args'], True)) for
            e in event_subset
        ]

        # TODO: in this mixed mode, these indices are not correct.
        common_data = {
            'event_indices': [],
            'slot_indicators': [],
        }

        for event_rep in all_event_reps:
            for key, value in event_rep.items():
                try:
                    common_data['context_' + key].append(value)
                except KeyError:
                    common_data['context_' + key] = [value]

        if len(t_doc_args) <= 1:
            # There no enough arguments to sample from.
            return None

        # We current sample the predicate based on unigram distribution.
        # The other learning strategy is to select one difficult cross instance,
        # the options are:
        # 1. Unigram distribution to sample items.
        # 2. Select items based on classifier output.

        # All generated instances here.
        instances = ClozeInstances(self.para, self.event_struct)

        num_neg = 0

        for evm_index, event in enumerate(event_subset):
            if num_neg > 100:
                # Limit the number of instances generated.
                break

            pred = event['predicate']
            if pred == self.unk_predicate_idx:
                continue

            pred_tf = self.event_emb_vocab.get_term_freq(pred)
            freq = 1.0 * pred_tf / self.pred_count

            if not self.predicate_sampler.subsample_pred(pred_tf, freq):
                # Too frequent word will be down-sampled.
                continue

            current_sent = event['sentence_id']

            arg_list = self.get_args_as_list(event['args'], False)

            for arg_index, (slot, arg) in enumerate(arg_list):
                # Empty args are not interesting.
                if len(arg) == 0:
                    continue

                # Not training unk args.
                if arg['represent'] == self.typed_event_vocab.unk_arg_word:
                    continue

                correct_id = arg['entity_id']

                is_singleton = False
                if eid_count[correct_id] <= 1:
                    # Only one mention for this one.
                    is_singleton = True

                # If we do not train ghost instance, we skip the singletons.
                if is_singleton and not self.para.use_ghost:
                    continue

                cross_sample = self.cloze_gen.cross_cloze(
                    arg_list, t_doc_args, arg_index)
                inside_sample = self.cloze_gen.inside_cloze(
                    arg_list, arg_index)

                slot_index = self.get_slot_index(slot)

                assert slot_index >= 0

                if cross_sample or inside_sample:
                    if is_singleton:
                        instances.add_ghost_instance(1)
                    else:
                        instances.assemble_instance(
                            features_by_eid, explicit_entity_positions,
                            current_sent,
                            self.event_struct.event_repr(
                                pred, event['frame'], arg_list),
                            correct_id, 1)

                        common_data['event_indices'].append(evm_index)
                        common_data['slot_indicators'].append(slot_index)

                if cross_sample:
                    cross_args, cross_filler_id = cross_sample

                    instances.assemble_instance(
                        features_by_eid, explicit_entity_positions,
                        current_sent,
                        self.event_struct.event_repr(
                            pred, event['frame'], cross_args
                        ),
                        cross_filler_id, 0)

                    common_data['event_indices'].append(evm_index)
                    common_data['slot_indicators'].append(slot_index)
                    num_neg += 1

                if inside_sample:
                    inside_args, inside_filler_id, swap_slot = inside_sample

                    instances.assemble_instance(
                        features_by_eid, explicit_entity_positions,
                        current_sent,
                        self.event_struct.event_repr(
                            pred, event['frame'], inside_args),
                        inside_filler_id, 0
                    )

                    common_data['event_indices'].append(evm_index)
                    common_data['slot_indicators'].append(slot_index)
                    num_neg += 1

        if num_neg == 0:
            # Too few training instance.
            return None

        # print('number events', len(event_subset), 'num negative', num_neg)

        return instances, common_data
