import itertools
import random


class EventAsArgCloze:
    def __init__(self, path):
        self.path = path
        self.target_roles = ['arg0', 'arg1', 'prep']
        self.entity_info_fields = ['syntactic_role', 'mention_text',
                                   'entity_id']
        self.entity_equal_fields = ['entity_id', 'mention_text']

    def read_events(self, data_in):
        events = []
        for line in data_in:
            line = line.strip()
            if not line:
                # Finish a document.
                yield docid, events
            elif line.startswith("#"):
                docid = line.rstrip("#")
            else:
                fields = line.split()
                if len(fields) < 3:
                    continue
                predicate, pred_context, frame = fields[:3]

                arg_fields = fields[3:]

                args = {}
                for _, v in itertools.groupby(
                        arg_fields, lambda k: k / 4):
                    syn_role, frame_role, mention, resolvable = v
                    entity_id, mention_text = mention.split(':')
                    arg = {
                        'syntactic_role': syn_role,
                        'frame_role': frame_role,
                        'mention_text': mention_text,
                        'entity_id': entity_id,
                        'resolvable': resolvable == '1',
                    }

                    args[syn_role] = arg

                event = {
                    'predicate': predicate,
                    'predicate_context': pred_context,
                    'arguments': args,
                }
                events.append(event)

    def create_clozes(self, data_in):
        for docid, doc_events in self.read_events(data_in):
            clozed_args = []
            for index, event in enumerate(doc_events):
                args = event['arguments']
                for role, arg in args.items():
                    if arg['resolvable']:
                        clozed_args.append((index, role))
            yield doc_events, clozed_args

    def read_clozes(self, data_in):
        for doc_events, clozed_args in self.create_clozes(data_in):
            all_entities = self._get_all_entities(doc_events)
            for clozed_arg in clozed_args:
                event_index, cloze_role = clozed_arg
                candidate_event = doc_events[event_index]
                answer = self._entity_info(
                    candidate_event['arguments'][clozed_arg]
                )
                wrong = self.sample_wrong(all_entities, answer)

                # Yield one pairwise cloze task:
                # [all events] [events in question] [role in question]
                # [correct mention] [wrong mention]
                yield doc_events, event_index, cloze_role, answer, wrong

    def sample_wrong(self, all_entities, answer):
        wrong_entities = [ent for ent in all_entities if
                          not self._same_entity(ent, answer)]
        return random.choice(wrong_entities)

    def _get_all_entities(self, doc_events):
        entities = []
        for event in doc_events:
            for arg in event['arguments']:
                entity = self._entity_info(arg)
                entities.append(entity)
        return entities

    def _same_entity(self, ent1, ent2):
        return any([ent1[f] == ent2[f] for f in self.entity_equal_fields])

    def _entity_info(self, arg):
        return dict([(k, arg[k]) for k in self.entity_info_fields])
