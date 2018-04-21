import json
import os
from collections import defaultdict
import logging


def rep(t):
    if t == 'WEA':
        return 'Weapon'
    elif t == 'VEH':
        return 'Vehicle'
    elif t == 'null':
        return 'GeneralEntity'
    else:
        return t.lower().title()


class Jsonable:
    def json_rep(self):
        raise NotImplementedError


class Frame(Jsonable):
    """
    Represent top level frame collection.
    """

    def __init__(self, fid, frame_type, parent):
        self.type = frame_type
        self.id = fid
        self.parent = parent

    def json_rep(self):
        rep = {
            '@type': self.type,
            '@id': self.id,
        }

        if self.parent:
            rep['parent_scope'] = self.parent

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
    def __init__(self, interp_type):
        self.interp_type = interp_type
        self.fields = defaultdict(lambda: defaultdict(list))
        self.multi_field = set()

    def add_fields(self, name, key, content, multi_value=False, score=1):
        self.fields[name][key].append((content, score))
        if multi_value:
            self.multi_field.add(name)

    def json_rep(self):
        rep = {
            '@type': self.interp_type,
        }
        for field_name, content in self.fields.items():
            field_data = []
            for key, values in content.items():
                if len(values) > 1:
                    field = {'@type': 'xor', 'args': []}
                    for v, score in values:
                        v_str = v.json_rep() if isinstance(v,
                                                           Jsonable) else str(v)
                        field['args'].append(
                            {
                                '@type': 'facet',
                                'value': v_str,
                                'score': score,
                            }
                        )
                else:
                    v, score = [v for v in values][0]
                    v_str = v.json_rep() if isinstance(v, Jsonable) else str(v)

                    # Facet repr is too verbose.
                    # field = {'@type': 'facet', 'value': v_str, 'score': score}
                    field = v_str
                field_data.append(field)

            if field_name not in self.multi_field:
                field_data = field_data[0] if len(field_data) else ""

            rep[field_name] = field_data
        return rep


class InterpFrame(Frame):
    def __init__(self, fid, frame_type, parent, interp_type):
        super().__init__(fid, frame_type, parent)
        self.interp_type = interp_type
        self.interps = []

    def json_rep(self):
        rep = super().json_rep()
        if self.interps:
            rep['interp'] = {}
            for interp in self.interps:
                rep['interp'].update(interp.json_rep())
        return rep


class Span:
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
    def __init__(self, fid, frame_type, parent, interp_type, reference, begin,
                 length, text):
        super().__init__(fid, frame_type, parent, interp_type)
        self.span = Span(reference, begin, length)
        self.text = text

    def json_rep(self):
        rep = super().json_rep()
        info = {
            'text': self.text,
            'parent_scope': self.parent,
            'extent': {
                '@type': 'text_span',
                'reference': self.span.reference,
                'start': self.span.begin,
                'length': self.span.length,
            }
        }
        rep.update(info)
        return rep


class Sentence(SpanInterpFrame):
    def __init__(self, fid, parent, reference, begin, length,
                 text):
        super().__init__(fid, 'sentence', parent, 'sentence_interp', reference,
                         begin, length, text)


class EntityMention(SpanInterpFrame):
    def __init__(self, fid, parent, reference, begin, length, text):
        super().__init__(fid, 'entity_mention', parent, 'entity_mention_interp',
                         reference, begin, length, text)

    def add_type(self, ontology, entity_type, score=1):
        type_interp = Interp(self.interp_type)

        entity_type = rep(entity_type)

        type_interp.add_fields('type', 'type', ontology + ":" + entity_type,
                               score=score)
        self.interps.append(type_interp)


class Argument(Jsonable):
    def __init__(self, arg_role, entity_mention):
        self.entity_mention = entity_mention
        self.arg_role = arg_role

    def json_rep(self):
        return {
            '@type': 'argument',
            'type': self.arg_role,
            'text': self.entity_mention.text,
            'arg': self.entity_mention.id,
        }


class EventMention(SpanInterpFrame):
    def __init__(self, fid, parent, reference, begin, length, text):
        super().__init__(fid, 'event_mention', parent, 'event_mention_interp',
                         reference, begin, length, text)
        self.trigger = None
        self.args = []

    def add_trigger(self, begin, length):
        self.trigger = Span(self.span.reference, begin, length)

    def add_type(self, ontology, event_type, score=1):
        interp = Interp(self.interp_type)
        interp.add_fields('type', 'type', ontology + ":" + event_type,
                          score=score)
        self.interps.append(interp)
        return interp

    def add_arg(self, interp, ontology, arg_role, entity_mention, score=1):
        arg = Argument(ontology + ':' + arg_role, entity_mention)
        interp.add_fields('args', arg_role, arg, multi_value=True, score=score)
        return interp


class RelationMention(InterpFrame):
    def __init__(self, fid, ontology, relation_type, arguments, score=1):
        super().__init__(fid, 'relation_mention', None,
                         'relation_mention_interp')
        self.relation_type = relation_type
        self.arguments = arguments
        rel_interp = Interp(self.interp_type)
        rel_interp.add_fields('type', 'type', ontology + ":" + relation_type,
                              score=score)
        rel_interp.add_fields('args', relation_type, arguments,
                              multi_value=False, score=score)
        self.interps.append(rel_interp)


class CSR:
    def __init__(self, component_name, run_id, out_path, namespace,
                 media_type='text'):
        self.header = {
            "@context": [
                "https://www.isi.edu/isd/LOOM/opera/"
                "jsonld-contexts/resources.jsonld",
                "https://www.isi.edu/isd/LOOM/opera/"
                "jsonld-contexts/ail/0.2/frames.jsonld",
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
        self.current_sentences = {}
        self.current_entities = {}
        self.current_events = {}
        self.args = defaultdict(dict)
        self.relations = {}

        self._span_map = defaultdict(dict)

        self.__frame_collection = [
            self._docs,
            self.current_sentences,
            self.current_entities,
            self.current_events,
            self.args,
            self.relations,
        ]

        self.run_id = run_id

        self.entity_index = 0
        self.event_index = 0
        self.relation_index = 0

        self.ns_prefix = namespace
        self.media_type = media_type

    def clear(self):
        self._span_map.clear()
        for frames in self.__frame_collection:
            frames.clear()
        # self.num_sents = 0
        self.entity_index = 0
        self.event_index = 0
        self.relation_index = 0

    def add_doc(self, doc_name, doc_type, language):
        ns_docid = self.ns_prefix + ":" + doc_name + "-" + self.media_type

        if ns_docid in self._docs:
            return

        doc = Document(ns_docid, doc_name, doc_type, language, self.media_type)
        self.current_doc = doc
        self._docs[ns_docid] = doc

    def add_sentence(self, sentence_index, span, text=None):
        sent_id = self.get_id('sent', sentence_index)

        if sent_id in self.current_sentences:
            return sent_id

        docid = self.current_doc.id
        self.current_doc.num_sentences += 1
        sent = Sentence(sent_id, docid, docid, span[0], span[1] - span[0], "")
        self.current_sentences[sent_id] = sent

        if text:
            sent.text = text

        return sent_id

    def set_sentence_text(self, sent_id, text):
        self.current_sentences[sent_id].text = text

    def validate_span(self, sent_id, span, text):
        sent = self.current_sentences[sent_id]
        sent_text = sent.text

        begin = span[0] - sent.span.begin
        end = span[1] - sent.span.begin
        span_text = sent_text[begin: end]

        if not span_text == text:
            logging.warning(
                "Span text: [{}] not matching given text [{}]"
                ", at span [{}] at sent [{}]".format(
                    span_text, text, span, sent_id)
            )
            return False

        return True

    def get_sentence_by_span(self, span):
        for sent_id, sentence in self.current_sentences.items():
            sent_begin = sentence.span.begin
            sent_end = sentence.span.length + sent_begin
            if span[0] >= sent_begin and span[1] <= sent_end:
                return sent_id

    def get_entity_mention(self, span):
        span = tuple(span)
        if span in self._span_map['entity']:
            entity_id = self._span_map['entity'][span]
            entity = self.current_entities[entity_id]
            return entity
        return None

    def add_entity_mention(self, span, text, ontology, entity_type,
                           sent_id=None):
        span = tuple(span)

        # Annotation on the same span will be reused.
        entity_mention = self.get_entity_mention(span)

        if entity_mention:
            entity_id = entity_mention.id
        else:
            if not sent_id:
                sent_id = self.get_sentence_by_span(span)
                if not sent_id:
                    # No suitable sentence to cover it.
                    return
            valid = self.validate_span(sent_id, span, text)

            if valid:
                sentence_start = self.current_sentences[sent_id].span.begin
                entity_id = self.get_id('ent', self.entity_index)
                self._span_map['entity'][span] = entity_id

                self.entity_index += 1

                entity_mention = EntityMention(entity_id, sent_id, sent_id,
                                               span[0] - sentence_start,
                                               span[1] - span[0], text)
                self.current_entities[entity_id] = entity_mention
            else:
                return

        entity_mention.add_type(ontology, entity_type)
        return entity_id

    def add_event_mention(self, span, text, ontology,
                          evm_type, sent_id=None):
        # Annotation on the same span will be reused.
        span = tuple(span)
        if span in self._span_map['event']:
            event_id = self._span_map[span][1]
            event = self.current_events[event_id]
        else:
            if not sent_id:
                sent_id = self.get_sentence_by_span(span)
                if not sent_id:
                    # No suitable sentence to cover it.
                    return

            valid = self.validate_span(sent_id, span, text)

            if valid:
                event_id = self.get_id('evm', self.event_index)
                self._span_map['event'][span] = event_id

                self.event_index += 1
                self._span_map[span] = ('event_mention', event_id)
                sent = self.current_sentences[sent_id]

                relative_begin = span[0] - sent.span.begin
                length = span[1] - span[0]

                event = EventMention(event_id, sent_id, sent_id,
                                     relative_begin, length,
                                     text)
                event.add_trigger(relative_begin, length)
                self.current_events[event_id] = event
            else:
                return

        interp = event.add_type(ontology, evm_type)
        return event_id, interp

    def get_id(self, prefix, index):
        return '%s:%s-%s-text-cmu-r%s-%d' % (
            self.ns_prefix,
            prefix,
            self.current_doc.doc_name,
            self.run_id,
            index,
        )

    def add_arg(self, interp, evm_id, ent_id, ontology, arg_role):
        evm = self.current_events[evm_id]
        ent = self.current_entities[ent_id]
        evm.add_arg(interp, ontology, arg_role, ent)

    def add_relation(self, ontology, arguments, relation_type):
        relation_id = self.get_id('relm', self.relation_index)
        self.relation_index += 1

        self.relations[relation_id] = RelationMention(relation_id, ontology,
                                                      relation_type, arguments)

    def get_json_rep(self):
        rep = {}
        self.header['meta']['document_id'] = self.current_doc.id
        rep.update(self.header)
        rep['frames'] = []

        for frames in self.__frame_collection:
            for fid, frame in frames.items():
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
