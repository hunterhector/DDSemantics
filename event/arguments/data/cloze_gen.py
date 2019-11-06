import random
import math

from event.arguments.prepare.event_vocab import EmbbedingVocab, TypedEventVocab


class PredicateSampler:
    def __init__(self, sample_pred_threshold=10e-5):
        self.sample_pred_threshold = sample_pred_threshold

    def subsample_pred(self, pred_tf, freq):
        if freq > self.sample_pred_threshold:
            if pred_tf > self.sample_pred_threshold:
                rate = self.sample_pred_threshold / freq
                if random.random() < 1 - rate - math.sqrt(rate):
                    return False
        return True


class ClozeSampler:
    def __init__(self, sample_pred_threshold=10e-5, seed=None):
        self.sample_pred_threshold = sample_pred_threshold
        self.provided_seed = seed
        random.seed(self.provided_seed)

    def reset(self):
        random.seed(self.provided_seed)

    def sample_cross(self, arg_pool, origin_start, origin_end):
        remaining = []
        for arg_info in arg_pool:
            if not (arg_info['arg_start'] == origin_start
                    or arg_info['arg_end'] == origin_end):
                remaining.append(arg_info)

        if len(remaining) > 0:
            return random.choice(remaining)
        else:
            return None

    def sample_list(self, l):
        return random.choice(l)

    def sample_ignore_item(self, data, ignored_item):
        """Sample one item in the list, but ignore a provided one. If the list
        contains less than 2 elements, nothing will be sampled.

        Args:
          data: ignored_item:
          ignored_item:

        Returns:

        """
        if len(data) <= 1:
            return None

        while True:
            sampled_item = random.choice(data)
            if not sampled_item == ignored_item:
                break
        return sampled_item

    def subsample_pred(self, pred_tf, freq):
        if freq > self.sample_pred_threshold:
            if pred_tf > self.sample_pred_threshold:
                rate = self.sample_pred_threshold / freq
                if random.random() < 1 - rate - math.sqrt(rate):
                    return False
        return True


class ClozeGenerator:
    def __init__(self,
                 sampler: ClozeSampler,
                 event_emb_vocab: EmbbedingVocab,
                 typed_event_vocab: TypedEventVocab,
                 fix_slot_mode: bool = True):
        self.sampler = sampler
        self.fix_slot_mode = fix_slot_mode
        self.event_emb_vocab = event_emb_vocab
        self.typed_event_vocab = typed_event_vocab

    def cross_cloze(self, arg_list, doc_args, target_arg_idx):
        """A negative cloze instance that use arguments from other events.

        Args:
          arg_list: List of origin event arguments.
          doc_args: List of arguments (partial info) in this doc.
          target_arg_idx: The index for the slot to be replaced

        Returns:

        """
        _, target_arg = arg_list[target_arg_idx]

        sample_res = self.sampler.sample_cross(
            doc_args, target_arg['arg_start'], target_arg['arg_end']
        )

        if sample_res is None:
            return None

        neg_instance = []
        for idx, (slot, content) in enumerate(arg_list):
            if idx == target_arg_idx:
                neg_instance.append(
                    (slot,
                     self.replace_slot_detail(content, sample_res))
                )
            else:
                neg_instance.append((slot, content))

        return neg_instance, sample_res['entity_id']

    def inside_cloze(self, arg_list, arg_index):
        """A negative cloze instance that use arguments within the event.

        Args:
          arg_list: The Arg List of the original event.
          arg_index: The current index of the argument.

        Returns:

        """
        if len(arg_list) < 2:
            return None

        sample_indexes = list(range(arg_index)) + list(
            range(arg_index, len(arg_list)))

        swap_index = self.sampler.sample_list(sample_indexes)

        _, origin_slot_info = arg_list[arg_index]
        _, swap_slot_info = arg_list[swap_index]

        # TODO: shall we?
        # When swapping the two slots, we also swap the FEs
        # this help us learn the correct slot for a FE.

        swap_slot = arg_list[swap_index][0]

        neg_instance = []
        for idx, (slot, content) in enumerate(arg_list):
            if idx == arg_index:
                # Replace the swap one here.
                if len(content) == 0 and self.fix_slot_mode:
                    base_slot = {'dep': slot}
                else:
                    base_slot = content

                neg_instance.append((
                    slot,
                    self.replace_slot_detail(base_slot, swap_slot_info, True)
                ))
            elif idx == swap_index:
                # Replace the original one here
                if len(content) == 0 and self.fix_slot_mode:
                    base_slot = {'dep': slot}
                else:
                    base_slot = content

                neg_instance.append((
                    slot,
                    self.replace_slot_detail(base_slot, origin_slot_info, True)
                ))
            else:
                neg_instance.append((slot, content))

        return neg_instance, origin_slot_info['entity_id'], swap_slot

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
