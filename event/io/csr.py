import json
import os
from collections import defaultdict, Counter
import logging
import string
from event.util import remove_punctuation


class Constants:
    GENERAL_ENTITY_TYPE = 'aida:General_Entity'
    GENERAL_EVENT_TYPE = 'aida:General_Event'
    GENERAL_REL_TYPE = 'aida:General_Rel'


entity_type_mapping = {
    "Fac": "Facility",
    "Gpe": "GPE",
    "Loc": "Location",
    "Nom": "Nominal",
    "Org": "Organization",
    "Per": "Person",
    "Veh": "Vehicle",
    "Wea": "Weapon",
}


def fix_entity_type(t):
    t = t.lower().title()
    if t in entity_type_mapping:
        return entity_type_mapping[t]
    return t


class Jsonable:
    def json_rep(self):
        raise NotImplementedError

    def __str__(self):
        return json.dumps(self.json_rep())


class Frame(Jsonable):
    """
    Represent top level frame collection.
    """

    def __init__(self, fid, frame_type, parent, component=None, score=None):
        self.type = frame_type
        self.id = fid
        self.parent = parent
        self.component = component
        self.score = score

    def json_rep(self):
        rep = {
            '@type': self.type,
        }

        if self.id:
            rep['@id'] = self.id

        # if self.parent:
        #     rep['parent_scope'] = self.parent

        if self.component:
            rep['component'] = self.component

        if self.score:
            rep['score'] = self.score

        return rep


class Document(Frame):
    """
    Represent a document frame.
    """

    def __init__(self, fid, doc_name, document_type, language,
                 media_type='text'):
        super().__init__(fid, 'document', None)
        self.doc_name = doc_name
        self.media_type = media_type
        self.document_type = document_type
        self.language = language
        self.num_sentences = 0

    def json_rep(self):
        rep = super().json_rep()
        rep['media_type'] = self.media_type
        rep['language'] = self.language
        rep['num_sentences'] = self.num_sentences
        return rep


class Interp(Jsonable):
    """
    An interp is an interpretation of the evidence. It can contain values like
    event/entity type, database links, etc.
    """

    def __init__(self, interp_type):
        self.interp_type = interp_type
        self.__fields = defaultdict(lambda: defaultdict(dict))
        self.multi_value_fields = set()

    def get_field(self, name):
        if name in self.__fields:
            return dict(self.__fields.get(name))
        else:
            return None

    def modify_field(self, name, key_name, content_rep, content, component=None,
                     score=None):
        original = self.__fields[name][key_name][content_rep]

        if not component:
            component = original['component']

        if not score:
            score = original['score']

        self.__fields[name][key_name][content_rep] = {
            'content': content, 'score': score, 'component': component,
        }

    def add_fields(self, name, key_name, content_rep, content,
                   component=None, score=None, multi_value=False):
        """
        Add an interp field
        :param name: The name to be displayed for this field.
        :param key_name: A unique key name for this field.
        :param content_rep: A value to identify this content from the rest,
        if two content_reps are the same, the are consider the same field.
        :param content: The actual content.
        :param component: The component name.
        :param score: The score of this interp field.
        :param multi_value: Whether this field can contain multiple values.
        :return:
        """
        # If the key is the same, then the two interps are consider the same.
        self.__fields[name][key_name][content_rep] = {
            'content': content, 'component': component, 'score': score
        }
        if multi_value:
            self.multi_value_fields.add(name)

    def is_empty(self):
        return len(self.__fields) == 0

    def json_rep(self):
        rep = {
            '@type': self.interp_type,
        }
        for field_name, field_info in self.__fields.items():
            field_data = []

            for key_name, keyed_content in field_info.items():
                if len(keyed_content) > 1:
                    # Multiple interpretation found.
                    field = {'@type': 'xor', 'args': []}

                    for key, value in keyed_content.items():
                        v = value['content']
                        score = value['score']
                        component = value['component']
                        v_str = v.json_rep() if \
                            isinstance(v, Jsonable) else str(v)
                        r = {
                            '@type': 'facet',
                            'value': v_str,
                        }
                        if score:
                            r['score'] = score
                        if component:
                            r['component'] = component

                        field['args'].append(r)
                else:
                    # Single interpretation.
                    for key, value in keyed_content.items():
                        v = value['content']
                        score = value['score']
                        component = value['component']
                        v_str = v.json_rep() if \
                            isinstance(v, Jsonable) else str(v)

                        # Facet repr is too verbose.
                        if score:
                            field = {'@type': 'facet', 'value': v_str}
                            if score:
                                field['score'] = score
                            if component:
                                field['component'] = component
                        else:
                            field = v_str

                field_data.append(field)

            if field_name not in self.multi_value_fields:
                field_data = field_data[0]
            rep[field_name] = field_data
        return rep


class InterpFrame(Frame):
    """
    A frame that can contain interpretations.
    """

    def __init__(self, fid, frame_type, parent, interp_type, component=None):
        super().__init__(fid, frame_type, parent, component)
        self.interp_type = interp_type
        self.interp = Interp(interp_type)

    def json_rep(self):
        rep = super().json_rep()
        if self.interp and not self.interp.is_empty():
            rep['interp'] = self.interp.json_rep()
            # for interp in self.interps:
            #     rep['interp'].update(interp.json_rep())
        return rep


class RelFrame(Frame):
    """
    A frame for relations between other frames.
    """

    def __init__(self, fid):
        super().__init__(fid, 'argument', None)
        self.members = []

    def add_arg(self, arg):
        self.members.append(arg)

    def json_rep(self):
        # rep = super().json_rep()
        # print(rep)
        rep = []
        for arg in self.members:
            rep.append(
                {
                    '@type': 'argument',
                    'type': 'aida:member',
                    'arg': arg
                }
            )
        return rep


class ValueFrame(Frame):
    """
    A frame that mainly contain values.
    """

    def __init__(self, fid, frame_type, component=None, score=None):
        super().__init__(fid, frame_type, None, component=component,
                         score=score)


class Span:
    """
    Represent text span (begin and end), according to the reference.
    """

    def __init__(self, reference, begin, length):
        self.reference = reference
        self.begin = begin
        self.length = length

    def json_rep(self):
        return {
            '@type': 'text_span',
            'reference': self.reference,
            'start': self.begin,
            'length': self.length,
        }

    def __str__(self):
        return "%s: %d,%d" % (
            self.reference, self.begin, self.begin + self.length
        )


class SpanInterpFrame(InterpFrame):
    """
    Commonly used frame to represent a mention, has both spans and
    interpretations.
    """

    def __init__(self, fid, frame_type, parent, interp_type, reference, begin,
                 length, text, component=None):
        super().__init__(fid, frame_type, parent, interp_type, component)
        self.span = Span(reference, begin, length)
        self.text = text
        self.modifiers = {}

    def add_modifier(self, modifier_type, modifier_text):
        self.modifiers[modifier_type] = modifier_text

    def json_rep(self):
        rep = super().json_rep()
        info = {
            'provenance': {
                '@type': 'text_span',
                'reference': self.span.reference,
                'start': self.span.begin,
                'length': self.span.length,
                'text': self.text,
                'parent_scope': self.parent,
                'modifiers': self.modifiers,
            }
        }
        rep.update(info)
        return rep


class Sentence(SpanInterpFrame):
    """
    Represent a sentence.
    """

    def __init__(self, fid, parent, reference, begin, length,
                 text, component=None):
        super().__init__(fid, 'sentence', parent, 'sentence_interp', reference,
                         begin, length, text, component=component)


class EntityMention(SpanInterpFrame):
    """
    Represent a entity mention (in output, it is called entity_evidence).
    """

    def __init__(self, fid, parent, reference, begin, length, text,
                 component=None):
        super().__init__(fid, 'entity_evidence', parent,
                         'entity_evidence_interp', reference, begin, length,
                         text, component)
        self.entity_types = []

    def add_form(self, entity_form):
        self.interp.add_fields('form', 'form', entity_form, entity_form)

    def get_types(self):
        return self.entity_types

    def add_type(self, ontology, entity_type, score=None, component=None):
        # type_interp = Interp(self.interp_type)
        if entity_type == "null":
            return

        entity_type = fix_entity_type(entity_type)
        onto_type = ontology + ":" + entity_type

        if component == self.component:
            # Inherit frame component name.
            component = None

        self.interp.add_fields('type', 'type', onto_type, onto_type,
                               score=score, component=component)

        self.entity_types.append(onto_type)
        # input("Added entity type for {}, {}".format(self.id, self.text))

    def add_linking(self, mid, wiki, score, lang='en', component=None):
        if mid.startswith('/'):
            mid = mid.strip('/')

        fb_xref = ValueFrame('freebase:' + mid, 'db_reference', score=score)
        self.interp.add_fields('xref', 'freebase', mid, fb_xref,
                               multi_value=True)

        wiki_xref = ValueFrame(lang + '_wiki:' + wiki, 'db_reference',
                               score=score)
        self.interp.add_fields('xref', 'wikipedia', wiki, wiki_xref,
                               component=component, multi_value=True)

    def add_salience(self, salience_score):
        self.interp.add_fields('salience', 'score', 'score', salience_score)


class Argument(Frame):
    """
    An argument of event, which is simply a wrap around an entity,
    """

    def __init__(self, arg_role, entity_mention, fid, component=None):
        super().__init__(fid, 'argument', None, component=component)
        self.entity_mention = entity_mention
        self.arg_role = arg_role

    def json_rep(self):
        rep = super().json_rep()

        rep['type'] = self.arg_role
        rep['text'] = self.entity_mention.text
        rep['arg'] = self.entity_mention.id

        return rep


class EventMention(SpanInterpFrame):
    """
    An event mention (in output, it is called event_evidence)
    """

    def __init__(self, fid, parent, reference, begin, length, text,
                 component=None):
        super().__init__(fid, 'event_evidence', parent, 'event_evidence_interp',
                         reference, begin, length, text, component=component)
        self.trigger = None
        self.event_type = None
        self.realis = None

    def add_trigger(self, begin, length):
        self.trigger = Span(self.span.reference, begin, length)

    def add_interp(self, ontology, event_type, realis, score=None,
                   component=None):
        # interp = Interp(self.interp_type)
        onto_type = ontology + ":" + event_type

        if component == self.component:
            # Inherit frame component name.
            component = None

        self.interp.add_fields('type', 'type', onto_type, onto_type,
                               score=score, component=component)
        self.interp.add_fields('realis', 'realis', realis, realis, score=score,
                               component=component)
        self.event_type = onto_type
        self.realis = realis

    def add_arg(self, ontology, arg_role, entity_mention, arg_id,
                score=None, component=None):
        arg = Argument(ontology + ':' + arg_role, entity_mention, arg_id,
                       component=component)
        self.interp.add_fields('args', arg_role, entity_mention.id, arg,
                               score=score, component=component,
                               multi_value=True)

    def add_salience(self, salience_score):
        self.interp.add_fields('salience', 'score', 'score', salience_score)


class RelationMention(InterpFrame):
    """
    Represent a relation between frames (more than 1).
    """

    def __init__(self, fid, ontology, relation_type, arguments, score=None,
                 component=None):
        super().__init__(fid, 'relation_evidence', None,
                         'relation_evidence_interp', component=component)
        self.relation_type = relation_type
        self.arguments = arguments
        onto_type = ontology + ":" + relation_type
        self.interp.add_fields('type', 'type', onto_type, onto_type)

    def json_rep(self):
        arg_frame = RelFrame(None)
        for arg in self.arguments:
            arg_frame.add_arg(arg)

        self.interp.add_fields('args', 'args', self.relation_type, arg_frame)

        return super().json_rep()

    def add_arg(self, arg):
        self.arguments.append(arg)


class CSR:
    """
    Main class that collect and output frames.
    """

    def __init__(self, component_name, run_id, out_path, namespace,
                 media_type='text', aida_ontology=None, onto_mapper=None):
        self.header = {
            "@context": [
                "https://www.isi.edu/isd/LOOM/opera/"
                "jsonld-contexts/resources.jsonld",
                "https://www.isi.edu/isd/LOOM/opera/"
                "jsonld-contexts/ail/0.3/frames.jsonld",
            ],
            '@type': 'frame_collection',
            'meta': {
                '@type': 'meta_info',
                'component': component_name,
                'organization': 'CMU',
                'media_type': media_type,
            },
        }

        self.out_path = out_path

        self._docs = {}

        self.current_doc = None

        self._span_frame_map = defaultdict(dict)
        self._frame_map = defaultdict(dict)

        self.run_id = run_id

        self.__index_store = Counter()

        self.ns_prefix = namespace
        self.media_type = media_type

        self.entity_key = 'entity'
        self.entity_head_key = 'entity_head'
        self.entity_group_key = 'entity_group'
        self.event_key = 'event'
        self.event_group_key = 'event_group'
        self.sent_key = 'sentence'
        self.rel_key = 'relation'

        self.aida_ontology = aida_ontology
        self.onto_mapper = onto_mapper

        self.event_onto = aida_ontology.event_onto_text()
        self.arg_restricts = aida_ontology.get_arg_restricts()

        self.canonical_types = {}
        for event_type in self.event_onto:
            c_type = onto_mapper.canonicalize_type(event_type)
            self.canonical_types[c_type] = event_type

        for _, arg_type in self.arg_restricts:
            c_type = onto_mapper.canonicalize_type(arg_type)
            self.canonical_types[c_type] = arg_type

        self.base_onto_name = 'aida'

    def __canonicalize_event_type(self):
        canonical_map = {}
        for event_type in self.event_onto:
            c_type = self.__cannonicalize_one_type(event_type)
            canonical_map[c_type] = event_type
        return canonical_map

    def __cannonicalize_one_type(self, event_type):
        return remove_punctuation(event_type).lower()

    def clear(self):
        self._span_frame_map.clear()
        self._frame_map.clear()

    def add_doc(self, doc_name, doc_type, language):
        ns_docid = self.ns_prefix + ":" + doc_name + "-" + self.media_type

        if ns_docid in self._docs:
            return

        doc = Document(ns_docid, doc_name, doc_type, language, self.media_type)
        self.current_doc = doc
        self._docs[ns_docid] = doc

    def get_events_mentions(self):
        return self.get_frames(self.event_key)

    def get_frames(self, key_type):
        # if key_type not in self._frame_map:
        #     raise KeyError('Unknown frame type : {}'.format(key_type))
        return self._frame_map[key_type]

    def get_frame(self, key_type, frame_id):
        frames = self.get_frames(key_type)
        return frames.get(frame_id)

    def add_sentence(self, span, text=None, component=None):
        sent_id = self.get_id('sent')

        if sent_id in self._frame_map[self.sent_key]:
            return sent_id

        docid = self.current_doc.id
        self.current_doc.num_sentences += 1
        sent_text = text if text else ""
        sent = Sentence(sent_id, docid, docid, span[0], span[1] - span[0],
                        text=sent_text, component=component)
        self._frame_map[self.sent_key][sent_id] = sent

        return sent

    def set_sentence_text(self, sent_id, text):
        self._frame_map[self.sent_key][sent_id].text = text

    def validate_span(self, sent_id, span, text):
        sent = self._frame_map[self.sent_key][sent_id]
        sent_text = sent.text

        begin = span[0] - sent.span.begin
        end = span[1] - sent.span.begin
        span_text = sent_text[begin: end]

        if not span_text == text:
            logging.warning("Was {}".format(span))
            logging.warning("Becomes {},{}".format(begin, end))
            logging.warning(sent_text)

            logging.warning(
                "Span text: [{}] not matching given text [{}]"
                ", at span [{}] at sent [{}]".format(
                    span_text, text, span, sent_id)
            )
            return False

        return True

    def get_sentence_by_span(self, span):
        for sent_id, sentence in self._frame_map[self.sent_key].items():
            sent_begin = sentence.span.begin
            sent_end = sentence.span.length + sent_begin
            if span[0] >= sent_begin and span[1] <= sent_end:
                return sent_id

    def get_by_span(self, object_type, span):
        span = tuple(span)
        if span in self._span_frame_map[object_type]:
            eid = self._span_frame_map[object_type][span]
            e = self._frame_map[object_type][eid]
            return e
        return None

    def map_entity_type(self, onto_name, entity_type):
        if onto_name == 'conll':
            if entity_type == 'Date':
                return 'aida', 'Time'
        return onto_name, entity_type

    def add_entity_mention(self, head_span, span, text, ontology, entity_type,
                           sent_id=None, entity_form=None, component=None):
        head_span = tuple(head_span)
        span = tuple(span)

        if not sent_id:
            sent_id = self.get_sentence_by_span(span)
            if not sent_id:
                # No suitable sentence to cover it.
                logging.warning(
                    "No suitable sentence for entity {}".format(span))
                return
        valid = self.validate_span(sent_id, span, text)

        if valid:
            sentence_start = self._frame_map[self.sent_key][sent_id].span.begin
            entity_id = self.get_id('ent')
            entity_mention = EntityMention(
                entity_id, sent_id, sent_id, span[0] - sentence_start,
                                             span[1] - span[0], text,
                component=component)
            self._span_frame_map[self.entity_key][span] = entity_id
            self._frame_map[self.entity_key][entity_id] = entity_mention

            self._span_frame_map[self.entity_head_key][head_span] = entity_id
            self._frame_map[self.entity_head_key][entity_id] = entity_mention

        else:
            return

        ontology, entity_type = self.map_entity_type(ontology, entity_type)

        if entity_form:
            entity_mention.add_form(entity_form)
        if entity_type:
            entity_mention.add_type(ontology, entity_type, component=component)
        else:
            entity_mention.add_type('conll', 'MISC', component=component)
        return entity_mention

    def map_event_type(self, evm_type, onto_name):
        if onto_name == 'tac':
            mapped_evm_type = self.__cannonicalize_one_type(evm_type)

            if mapped_evm_type in self.canonical_types:
                return self.canonical_types[mapped_evm_type]
        else:
            full_type = onto_name + ':' + evm_type
            event_map = self.onto_mapper.get_aida_event_map()

            if full_type in event_map:
                return self.canonical_types[event_map[full_type]]
        return None

    def add_event_mention(self, head_span, span, text, onto_name, evm_type,
                          realis=None, sent_id=None, component=None,
                          arg_entity_types=None):
        # Annotation on the same span will be reused.
        head_span = tuple(head_span)
        span = tuple(span)

        if not sent_id:
            sent_id = self.get_sentence_by_span(span)
            if not sent_id:
                # No suitable sentence to cover it.
                return

        valid = self.validate_span(sent_id, span, text)

        if valid:
            if head_span in self._span_frame_map[self.event_key]:
                logging.info("Cannot handle overlapped event mentions now.")
                return

            event_id = self.get_id('evm')
            self._span_frame_map[self.event_key][head_span] = event_id

            sent = self._frame_map[self.sent_key][sent_id]

            relative_begin = span[0] - sent.span.begin
            length = span[1] - span[0]

            evm = EventMention(event_id, sent_id, sent_id, relative_begin,
                               length, text, component=component)
            evm.add_trigger(relative_begin, length)
            self._frame_map[self.event_key][event_id] = evm
        else:
            return

        # # Annotation on the same span are recorded as span groups.
        # evm_group = self.get_by_span(self.event_group_key, head_span)
        # if not evm_group:
        #     group_id = self.get_id('rel')
        #     evm_group = RelationMention(self.get_id('rel'), 'aida',
        #                                 'share_head_span', [])
        #     self._span_frame_map[self.event_group_key][head_span] = group_id
        #     self._frame_map[self.event_group_key][group_id] = evm_group
        #
        # evm_group.add_arg(event_id)

        if evm_type:
            realis = 'UNK' if not realis else realis

            mapped_type = self.map_event_type(evm_type, onto_name)

            if arg_entity_types:
                mapped_type = fix_event_type_from_entity(
                    mapped_type, arg_entity_types)

            if mapped_type:
                evm.add_interp('aida', mapped_type, realis, component=component)
            else:
                evm.add_interp(onto_name, evm_type, realis, component=component)

        return evm

    def get_id(self, prefix, index=None):
        if not index:
            index = self.__index_store[prefix]
            self.__index_store[prefix] += 1

        return '%s:%s-%s-text-cmu-r%s-%d' % (
            self.ns_prefix,
            prefix,
            self.current_doc.doc_name,
            self.run_id,
            index,
        )

    def map_event_arg_type(self, event_onto, evm_type, arg_role):
        map_to_aida = self.onto_mapper.get_aida_arg_map()

        if event_onto == 'aida':
            key = (self.onto_mapper.canonicalize_type(evm_type), arg_role)

            mapped_arg_type = None

            if key in map_to_aida:
                arg_aid_type = map_to_aida[key]

                c_arg_aida_type = self.onto_mapper.canonicalize_type(
                    arg_aid_type)

                if c_arg_aida_type in self.canonical_types:
                    mapped_arg_type = self.canonical_types[c_arg_aida_type]
                else:
                    if arg_aid_type.startswith('internal_'):
                        mapped_arg_type = arg_aid_type.replace(
                            'internal_', 'internal:')

            elif arg_role == 'ARGM-TMP' or 'Time' in arg_role:
                arg_aid_type = evm_type.lower() + '_Time'
                full_arg = (evm_type, arg_aid_type)
                if full_arg in self.arg_restricts:
                    mapped_arg_type = arg_aid_type
            elif arg_role == 'ARGM-LOC' or 'Place' in arg_role:
                arg_aid_type = evm_type.lower() + '_Place'
                full_arg = (evm_type, arg_aid_type)
                if full_arg in self.arg_restricts:
                    mapped_arg_type = arg_aid_type

            return mapped_arg_type

    def add_event_arg_by_span(self, evm, arg_head_span, arg_span,
                              arg_text, arg_onto, arg_role, component,
                              arg_entity_form='named'):
        ent = self.get_by_span(self.entity_key, arg_span)

        if not ent:
            ent = self.get_by_span(self.entity_head_key, arg_head_span)

        if not ent:
            ent = self.add_entity_mention(
                arg_head_span, arg_span, arg_text, 'aida', "argument",
                entity_form=arg_entity_form, component=component)

        if ent:
            evm_onto, evm_type = evm.event_type.split(':')
            arg_id = self.get_id('arg')
            in_domain_arg = self.map_event_arg_type(
                evm_onto, evm_type, arg_role)

            if in_domain_arg:
                if ':' in in_domain_arg:
                    arg_onto, arg_role = in_domain_arg.split(':')
                else:
                    arg_onto = self.base_onto_name
                    arg_role = in_domain_arg
            else:
                if evm_onto == 'aida':
                    # print("Event is", evm.text, 'Event type is', evm.type,
                    #       ', argument is', arg_text, ', arg role is', arg_role)
                    pass

            evm.add_arg(arg_onto, arg_role, ent, arg_id, component=component)

    def add_event_arg(self, evm, ent, ontology, arg_role, component):
        arg_id = self.get_id('arg')
        evm.add_arg(ontology, arg_role, ent, arg_id, component=component)

    def add_relation(self, ontology, arguments, relation_type, component=None):
        relation_id = self.get_id('relm')
        rel = RelationMention(relation_id, ontology, relation_type, arguments,
                              component=component)
        self._frame_map[self.rel_key][relation_id] = rel

    def get_json_rep(self):
        rep = {}
        self.header['meta']['document_id'] = self.current_doc.id
        rep.update(self.header)
        rep['frames'] = [self.current_doc.json_rep()]
        for frame_type, frame_info in self._frame_map.items():
            for fid, frame in frame_info.items():
                rep['frames'].append(frame.json_rep())

        return rep

    def write(self, append=False):
        if append:
            mode = 'a' if os.path.exists(self.out_path) else 'w'
        else:
            mode = 'w'

        logging.info("Writing data to [%s]" % self.out_path)

        with open(self.out_path, mode) as out:
            json.dump(self.get_json_rep(), out, indent=2, ensure_ascii=False)

        self.clear()


def fix_event_type_from_entity(evm_type, arg_entity_types):
    if evm_type == 'Movement.TransportPerson':
        if ('aida:Person' not in arg_entity_types) and (
                'aida:Vehicle' in arg_entity_types):
            evm_type = 'Movement.TransportArtifact'
    return evm_type
