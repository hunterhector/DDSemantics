import copy
import json
import logging
from collections import Counter
from collections import defaultdict
from operator import itemgetter

import torch

from event.arguments.data.batcher import ClozeBatcher
from event.arguments.data.cloze_gen import ClozeGenerator, PredicateSampler
from event.arguments.data.cloze_instance import ClozeInstanceBuilder
from event.arguments.data.event_structure import EventStruct
from event.arguments.data.frame_data import FrameSlots
from event.arguments.implicit_arg_params import ArgModelPara
from event.arguments.implicit_arg_resources import ImplicitArgResources
from event.arguments.util import (batch_combine, to_torch)

logger = logging.getLogger(__name__)


def inverse_vocab(vocab):
    inverted = [0] * vocab.get_size()
    for w, index in vocab.vocab_items():
        inverted[index] = w
    return inverted


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

        self.instance_keys = []

        # A set for data with variable length, we then will create paddings
        # and length tensor for it.
        self.__unk_length = set()

        # In fix slot mode we assume there is a fixed number of slots.
        if para.arg_representation_method == 'fix_slots':
            self.fix_slot_mode = True
            self.instance_keys = ('event_component', 'distances', 'features')
        elif para.arg_representation_method == 'role_dynamic':
            self.fix_slot_mode = False
            self.instance_keys = ('predicate', 'slot', 'slot_value', 'frame',
                                  'slot_length', 'distances', 'features',)
            self.__unk_length = {'slot', 'slot_value', 'context_slot',
                                 'context_slot_value'}

        self.gold_role_field = self.para.gold_field_name
        self.auto_test = False
        self.test_limit = 500

        self.device = torch.device(
            "cuda" if para.use_gpu and torch.cuda.is_available() else "cpu"
        )

        self.batcher = ClozeBatcher(self.para.batch_size, self.device)
        self.event_struct = EventStruct(
            resources.event_embed_vocab, resources.typed_event_vocab,
            para.use_frame, self.fix_slot_mode
        )
        self.predicate_sampler = PredicateSampler()

        logger.info("Reading data with device: " + str(self.device))

    def read_train_batch(self, data_in, sampler):
        logger.info("Reading data as training batch.")

        cloze_gen = ClozeGenerator(sampler, self.event_emb_vocab,
                                   self.typed_event_vocab, self.fix_slot_mode)

        for line in data_in:
            parsed_output = self.create_training_data(line, cloze_gen)

            if parsed_output is None:
                continue

            instance_data, common_data = parsed_output
            yield from self.batcher.read_data(instance_data, common_data)

        if self.batcher.doc_count == 0:
            raise ValueError("Provided data is empty.")

        # Yield the remaining data.
        yield self.batcher.flush()

    @staticmethod
    def collect_features(doc_info):
        # Collect all features.
        features_by_eid = {}
        for eid, content in doc_info['entities'].items():
            features_by_eid[int(eid)] = content['features']
        return features_by_eid

    def create_slot_candidates(self, test_stub, doc_mentions, pred_sent,
                               distance_cap=float('inf')):
        """Create slot candidates from the document mentions.

        Args:
          test_stub: The test stub to be filled.
          doc_mentions: The list of all document mentions.
          pred_sent: The sentence where the predicate is in.
          distance_cap: The distance cap for selecting the candidate
        argument: arguments with a sentence distance larger than this
        will be ignored. The default value is INFINITY (no cap).

        Returns:

        """
        # List of Tuple (dist, argument candidates).
        dist_arg_list = []

        # NOTE: we have removed the check of "original span". It means that if
        # the system can predict the original phrase then it will be fine.
        # This might be OK since the model should not have access to the
        # original span during real test time. At self-test stage, predicting
        # the original span means the system is learning well.
        for doc_mention in doc_mentions:
            # This is the target argument replaced by another entity mention.
            update_arg = self.replace_slot_detail(test_stub, doc_mention, True)

            dist_arg_list.append((
                abs(pred_sent - doc_mention['sentence_id']),
                (update_arg, doc_mention['entity_id']),
            ))

        # Sort the rank list based on the distance to the target evm.
        dist_arg_list.sort(key=itemgetter(0))
        sorted_arg_list = [arg for d, arg in dist_arg_list if d <= distance_cap]

        if self.para.use_ghost:
            # Put the ghost at the beginning to avoid it being pruned by \
            # distance.
            sorted_arg_list.insert(0, ({}, ghost_entity_id))

        return sorted_arg_list

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
                    # This checks whether there are explicit gold arguments.
                    if arg['source'] == 'gold' and not arg['implicit']:
                        break
                else:
                    # We make sure there is no implicit and explicit arguments.
                    test_cases.append([slot] + copy.deepcopy(no_fill_case))

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
        for slot, l_arg in event_args.items():
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
                        factor_role = slot

                    if factor_role not in slot_args:
                        slot_args[factor_role] = a
        args = list(slot_args.items())
        return args

    def get_one_test_doc(self, doc_info, nid_detector):
        """Parse and get one test document.

        Args:
          doc_info: The JSON data of one document.
          nid_detector: NID detector to detect which slot to fill.

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
                        # Empty args will be skipped.
                        # At real production time, we need to create an arg
                        # with some placeholder values, which is based on
                        # null instantiation prediction.
                        continue

                    eid = arg['entity_id']
                    doc_arg_info = {
                        'event_index': evm_index,
                        'entity_id': eid,
                        'arg_phrase': arg['arg_phrase'],
                        # The sentence id is used to sort the candidate by
                        # its distance.
                        'sentence_id': arg['sentence_id'],
                        'represent': arg['represent'],
                        'text': arg['text'],
                        'dep': arg['dep'],
                        'arg_start': arg['arg_start'],
                        'arg_end': arg['arg_end'],
                        'fe': arg['fe'],
                        'source': arg['source'],
                    }

                    if 'ner' in arg:
                        doc_arg_info['ner'] = arg['ner']

                    if not self.auto_test and self.gold_role_field in arg:
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

        eid_to_mentions = {}
        if self.auto_test:
            for arg in doc_mentions:
                try:
                    eid_to_mentions[arg['entity_id']].append(arg)
                except KeyError:
                    eid_to_mentions[arg['entity_id']] = [arg]

        for evm_index, event in enumerate(doc_info['events']):
            pred_sent = event['sentence_id']
            pred_idx = event['predicate']
            event_args = event['args']
            arg_list = self.get_args_as_list(event_args, False)

            # There are two ways to create test cases.
            # 1. The automatic cloze method: a test case is by removing one
            # phrase, all mentions with the same eid should be considered as
            # answers. The target slot is simply the dep (preposition)

            # 2. The true test case: a test case if we have an implicit=True,
            # the target slot should be a field in data (prespecified). We then
            # need to map the target slot (e.g. i_arg1 -> obj). The answer are
            # all arguments with the same field attribute then.

            available_slots = self.frame_slots.get_predicate_slots(event)

            test_cases = self.get_test_cases(event, available_slots)

            for target_slot, test_stub, answers in test_cases:
                # The detector determine whether we should fill this slot.
                if nid_detector.should_fill(event, target_slot, test_stub):
                    # Fill the test_stub with all the possible args in this doc.
                    test_rank_list = self.create_slot_candidates(
                        test_stub, doc_mentions, pred_sent, distance_cap=3)

                    # Prepare instance data for each possible instance.
                    instance = ClozeInstanceBuilder(self.para,
                                                    self.event_struct)

                    candidate_meta = []
                    instance_meta = []

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

                        # Create the event instance representation.
                        instance.assemble_instance(
                            features_by_eid,
                            explicit_entity_positions,
                            pred_sent,
                            self.event_struct.event_repr(
                                pred_idx, event['frame'], candidate_args
                            ),
                            filler_eid
                        )

                        cloze_event_indices.append(evm_index)

                        if self.fix_slot_mode:
                            cloze_slot_indicator.append(
                                self.fix_slot_names.index(target_slot)
                            )
                        else:
                            # In dynamic slot mode, we provide the slot's
                            # vocab id, which can be converted to embedding.
                            cloze_slot_indicator.append(
                                self.event_emb_vocab.get_index(
                                    target_slot, self.typed_event_vocab.unk_fe)
                            )

                        if filler_eid == ghost_entity_id:
                            candidate_meta.append(
                                {
                                    'entity': ghost_entity_text,
                                    'span': (-1, -1),
                                }
                            )
                        else:
                            candidate_meta.append({
                                'entity': cand_arg['represent'],
                                'distance_to_event': (
                                        pred_sent - cand_arg['sentence_id']
                                ),
                                'span': (
                                    cand_arg['arg_start'], cand_arg['arg_end']
                                ),
                                'source': cand_arg['source'],
                            })

                    instance_meta.append({
                        'predicate': event['predicate_text'],
                        'predicate_idx': pred_idx,
                        'target_slot': target_slot,
                        'answers': answers,
                    })

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
                        yield (instance.data, common_data,
                               candidate_meta, instance_meta)

    def read_test_docs(self, test_in, nid_detector):
        """Load test data. Importantly, this will create alternative cloze
         filling for ranking.

        Args:
          test_in: supply lines as test data.
          nid_detector: Null Instantiation Detector.

        Returns:

        """
        for line in test_in:
            doc_info = json.loads(line)
            doc_id = doc_info['docid']

            for test_data in self.get_one_test_doc(doc_info, nid_detector):
                (instances, common_data, candidate_meta,
                 instance_meta,) = test_data

                b_common_data = {}
                b_instance_data = {}

                max_slot_size = max(len(l) for l in instances['slot'])
                max_c_slot_size = max(
                    len(l) for l in common_data['context_slot'])

                # Create a pseudo batch with one instance.
                for key, value in common_data.items():
                    if key in self.__unk_length:
                        if key.startswith('context_'):
                            self.__var_pad(key, value, max_c_slot_size)
                        else:
                            self.__var_pad(key, value, max_slot_size)

                    vectorized = to_torch([value],
                                          self.__data_types[key])
                    b_common_data[key] = batch_combine(vectorized, self.device)

                for key, value in instances.items():
                    vectorized = to_torch([value], self.__data_types[key])
                    b_instance_data[key] = batch_combine(vectorized,
                                                         self.device)

                yield (
                    doc_id,
                    b_instance_data,
                    b_common_data,
                    candidate_meta,
                    instance_meta,
                )

    def create_training_data(self, data_line, cloze_gen: ClozeGenerator):
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

            sentence_id = event.get('sentence_id', None)

            for slot, arg in self.get_args_as_list(event['args'], False):
                if len(arg) > 0:
                    # Argument for n-th event, at slot position 'slot'.
                    eid = arg['entity_id']

                    arg_start = arg['arg_start']
                    arg_end = arg['arg_end']

                    if not arg.get('implicit', False):
                        explicit_entity_positions[eid][
                            (arg_start, arg_end)] = sentence_id

                    # Some minimum information for creating cloze tests.
                    # - Entity id is the target to predict.
                    # - Arg phrase, represent or text are different ways to
                    #   represent the argument mention.
                    # - source distinguish gold and automatic predicted args.
                    # - sentence_id is related to the distance based feature.
                    # - arg start and end are used to identify this unique
                    #   mention.
                    mention_info = {
                        'entity_id': eid,
                        'arg_phrase': arg['arg_phrase'],
                        'represent': arg['represent'],
                        'text': arg['text'],
                        # Some dataset do not have the source field.
                        'source': arg.get('source', 'automatic'),
                        'sentence_id': arg['sentence_id'],
                        'arg_start': arg['arg_start'],
                        'arg_end': arg['arg_end'],
                    }

                    if 'ner' in arg:
                        mention_info['ner'] = arg['ner']

                    arg_mentions[
                        (arg['arg_start'], arg['arg_end'])] = mention_info
                    eid_count[eid] += 1

        # A set that contains some minimum argument entity mention information,
        # used for sampling a slot to create clozes.
        t_doc_args = [v for v in arg_mentions.values()]

        all_event_reps = [
            self.event_struct.event_repr(
                e['predicate'], e['frame'],
                self.get_args_as_list(e['args'], True)) for
            e in event_subset
        ]

        common_data = {
            'cross_event_indices': [],
            'cross_slot_indicators': [],
            'inside_event_indices': [],
            'inside_slot_indicators': [],
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
        cross_instance = ClozeInstanceBuilder(self.para, self.event_struct)
        cross_gold = ClozeInstanceBuilder(self.para, self.event_struct)

        inside_instance = ClozeInstanceBuilder(self.para, self.event_struct)
        inside_gold = ClozeInstanceBuilder(self.para, self.event_struct)

        for evm_index, event in enumerate(event_subset):
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

                # If we do not train on singletons, we skip now.
                if is_singleton and not self.para.use_ghost:
                    continue

                cross_sample = cloze_gen.cross_cloze(
                    arg_list, t_doc_args, arg_index)
                inside_sample = cloze_gen.inside_cloze(
                    arg_list, arg_index)

                slot_index = self.fix_slot_names.index(slot) if \
                    self.fix_slot_mode else slot

                if cross_sample:
                    cross_args, cross_filler_id = cross_sample

                    cross_instance.assemble_instance(
                        features_by_eid,
                        explicit_entity_positions,
                        current_sent,
                        self.event_struct.event_repr(
                            pred, event['frame'], cross_args
                        ),
                        cross_filler_id
                    )

                    if is_singleton:
                        # If it is a singleton, than the ghost instance should
                        # be higher than randomly placing any entity here.
                        cross_gold.add_ghost_instance()
                    else:
                        cross_gold.assemble_instance(
                            features_by_eid,
                            explicit_entity_positions,
                            current_sent,
                            self.event_struct.event_repr(
                                pred, event['frame'], arg_list),
                            correct_id)

                    # These two list indicate where the target argument is.
                    # When we use fix slot mode, the index is a indicator of
                    # the slot position. Otherwise, the slot_index is a way to
                    # represent the open-set slot types (such as FEs)
                    common_data['cross_event_indices'].append(evm_index)
                    common_data['cross_slot_indicators'].append(slot_index)

                if inside_sample:
                    inside_args, inside_filler_id, swap_slot = inside_sample
                    inside_instance.assemble_instance(
                        features_by_eid,
                        explicit_entity_positions,
                        current_sent,
                        self.event_struct.event_repr(
                            pred, event['frame'], inside_args),
                        inside_filler_id
                    )

                    # Inside sample can be used on singletons.
                    inside_gold.assemble_instance(
                        features_by_eid,
                        explicit_entity_positions,
                        current_sent,
                        self.event_struct.event_repr(
                            pred, event['frame'], arg_list),
                        inside_filler_id
                    )
                    common_data['inside_event_indices'].append(evm_index)
                    common_data['inside_slot_indicators'].append(slot_index)

        if len(common_data['cross_slot_indicators']) == 0 or \
                len(common_data['inside_slot_indicators']) == 0:
            # Too few training instance.
            return None

        instance_data = {
            'cross_gold': cross_gold.data,
            'cross': cross_instance.data,
            'inside_gold': inside_gold.data,
            'inside': inside_instance.data,
        }

        return instance_data, common_data

    def replace_slot_detail(self, base_slot, swap_slot, replace_fe=False):
        if len(swap_slot) == 0:
            # Make the update slot to be empty.
            updated_slot_info = {}
        else:
            # Make a copy of the slot.
            updated_slot_info = dict((k, v) for k, v in base_slot.items())

            # Replace with the new information.
            updated_slot_info['entity_id'] = swap_slot['entity_id']
            updated_slot_info['represent'] = swap_slot['represent']
            updated_slot_info['text'] = swap_slot['text']
            updated_slot_info['arg_phrase'] = swap_slot['arg_phrase']
            updated_slot_info['source'] = swap_slot.get('source', 'automatic')
            updated_slot_info['arg_start'] = swap_slot['arg_start']
            updated_slot_info['arg_end'] = swap_slot['arg_end']

            if 'ner' in swap_slot:
                updated_slot_info['ner'] = swap_slot['ner']
            elif 'ner' in updated_slot_info:
                updated_slot_info.pop('ner')

            # These attributes are harmless but confusing.

            updated_slot_info.pop('resolvable', None)
            updated_slot_info.pop('implicit', None)

            # Note: with the sentence Id we can have a better idea of where the
            # argument is from, but we cannot use it to extract features.
            updated_slot_info['sentence_id'] = swap_slot['sentence_id']

            # TODO: now using the full dependency label here.
            new_arg_rep = self.typed_event_vocab.get_arg_rep(
                base_slot['dep'], swap_slot['represent']
            )

            updated_slot_info['arg_role'] = self.event_emb_vocab.get_index(
                new_arg_rep, self.typed_event_vocab.get_unk_arg_rep()
            )

            if replace_fe:
                updated_slot_info['fe'] = swap_slot['fe']

        return updated_slot_info
