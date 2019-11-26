import logging
import pdb

from event.arguments.prepare.event_vocab import EmbbedingVocab, TypedEventVocab

ghost_entity_text = '__ghost_component__'


class EventStruct:
    def __init__(self,
                 event_emb_vocab: EmbbedingVocab,
                 typed_event_vocab: TypedEventVocab,
                 use_frame=True,
                 fix_slot_mode=True):
        self.fix_slot_mode = fix_slot_mode
        self.use_frame = use_frame

        # Some extra embeddings.
        self.unobserved_fe = event_emb_vocab.add_extra(
            '__unobserved_fe__')
        self.unobserved_arg = event_emb_vocab.add_extra(
            '__unobserved_arg__')
        self.ghost_component = event_emb_vocab.add_extra(
            ghost_entity_text)

        self.unk_frame_idx = event_emb_vocab.get_index(
            typed_event_vocab.unk_frame, None)
        self.unk_predicate_idx = event_emb_vocab.get_index(
            typed_event_vocab.unk_predicate, None)
        self.unk_arg_idx = event_emb_vocab.get_index(
            typed_event_vocab.get_unk_arg_rep, None)
        self.unk_fe_idx = event_emb_vocab.get_index(
            typed_event_vocab.unk_fe, None
        )

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
        for slot, arg in args:
            if slot == -1:
                slot = self.unk_fe_idx
            slot_comps.append(slot)
            slot_value_comps.append(arg['arg_role'])

        if len(slot_comps) == 0:
            slot_comps.append(self.unobserved_fe)
            slot_value_comps.append(self.unobserved_arg)

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

        # TODO: the current setup for argument slot position in the fix slot
        #   model might mess up, need double check on this mapping method.
        for _, arg in args:
            if len(arg) == 0:
                if self.use_frame:
                    event_components.append(self.unobserved_fe)
                event_components.append(self.unobserved_arg)
            else:
                # Adding frame elements in argument representation.
                if self.use_frame:
                    fe = arg['fe']
                    if fe == -1:
                        event_components.append(self.unk_fe_idx)
                    else:
                        event_components.append(fe)
                event_components.append(arg['arg_role'])

        if any([c < 0 for c in event_components]):
            logging.error("None positive component found in event.")
            pdb.set_trace()

        return {
            'event_component': event_components,
        }
