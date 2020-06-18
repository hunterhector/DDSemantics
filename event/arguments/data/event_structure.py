import logging
import pdb
from pprint import pprint

from event.arguments.prepare.event_vocab import EmbbedingVocab, TypedEventVocab


class EventStruct:
    def __init__(self,
                 event_emb_vocab: EmbbedingVocab,
                 typed_event_vocab: TypedEventVocab,
                 use_frame=True,
                 fix_slot_mode=True):
        self.fix_slot_mode = fix_slot_mode
        self.use_frame = use_frame

        self.unk_frame_idx = event_emb_vocab.get_index(
            typed_event_vocab.unk_frame, None)
        self.unk_fe_idx = event_emb_vocab.get_index(
            typed_event_vocab.unk_fe, None
        )
        self.unobserved_arg_idx = event_emb_vocab.get_index(
            typed_event_vocab.unobserved_arg, None)
        self.unobserved_fe_idx = event_emb_vocab.get_index(
            typed_event_vocab.unobserved_fe, None)

        # The cloze data are organized by the following slots.
        self.fix_slot_names = ['subj', 'obj', 'prep', ]

    def event_repr(self, predicate, frame_id, args):
        if self.fix_slot_mode:
            return self._take_fixed_size_event_parts(predicate, frame_id, args)
        else:
            return self._take_dynamic_event_parts(predicate, frame_id, args)

    def _take_dynamic_event_parts(self, predicate, frame_id, args):
        """Take event information, with unknown number of slots.

        Args:
          predicate: frame_id:
          args:
          frame_id:

        Returns:

        """
        slot_comps = []
        slot_value_comps = []

        # The slot will need to be indexed vocabularies, i.e. frame elements.
        # And they need to be hashed to number first.
        for slot, arg in args.items():
            if slot == -1:
                slot = self.unk_fe_idx
            slot_comps.append(slot)
            slot_value_comps.append(arg['arg_role'])

        if len(slot_comps) == 0:
            slot_comps.append(TypedEventVocab.unobserved_fe)
            slot_value_comps.append(TypedEventVocab.unobserved_arg)

        frame_id = self.unk_frame_idx if frame_id == -1 else frame_id

        return {
            'predicate': [predicate, frame_id],
            'slot': slot_comps,
            'slot_value': slot_value_comps,
            'slot_length': len(slot_comps)
        }

    def _take_fixed_size_event_parts(self, predicate, frame_id, args):
        """Take event information from the data, one element per slot, hence the
        size of the event parts is fixed.

        Args:
          predicate: frame_id:
          args:
          frame_id:

        Returns:

        """
        event_components = [
            predicate,
        ]

        if self.use_frame:
            fid = self.unk_frame_idx if frame_id == -1 else frame_id
            event_components.append(fid)

        # TODO: fix slot names are names, but we are using the index of slot_name now, find out slot_name index, and correct here.
        for slot_name in self.fix_slot_names:
            if slot_name in args:
                arg = args[slot_name]
                # Adding frame elements in argument representation.
                if self.use_frame:
                    fe = arg['fe']
                    if fe == -1:
                        event_components.append(self.unk_fe_idx)
                    else:
                        event_components.append(fe)
                event_components.append(arg['arg_role'])
            else:
                # Adding unobserved id.
                if self.use_frame:
                    event_components.append(self.unobserved_fe_idx)
                event_components.append(self.unobserved_arg_idx)

        if any([c < 0 for c in event_components]):
            logging.error("Non positive component found in event.")
            pdb.set_trace()

        return {
            'event_component': event_components,
        }
