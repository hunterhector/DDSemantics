import json
import os
from collections import defaultdict
import logging


class Frame:
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

    def json_rep(self):
        rep = super().json_rep()
        rep['media_type'] = self.media_type
        rep['language'] = self.language
        return rep


class Interp:
    def __init__(self, interp_type):
        self.interp_type = interp_type
        self.fields = defaultdict(list)

    def add_fields(self, k, v, score):
        self.fields[k].append((v, score))

    def json_rep(self):
        rep = {
            '@type': self.interp_type,
        }
        for field_name, values in self.fields.items():
            if len(values) > 1:
                field = {'@type': 'xor', 'args': []}
                for v, score in values:
                    field['args'].append(
                        {
                            '@type': 'facet',
                            'value': v,
                            'score': score,
                        }
                    )
            else:
                v, score = [v for v in values][0]
                field = {'@type': 'facet', 'value': v, 'score': score}
            rep[field_name] = field
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
        type_interp.add_fields('type', ontology + ":" + entity_type, score)
        self.interps.append(type_interp)


class EventMention(SpanInterpFrame):
    def __init__(self, fid, parent, reference, begin, length, text):
        super().__init__(fid, 'event_mention', parent, 'event_mention_interp',
                         reference, begin, length, text)
        self.trigger = None
        self.args = []

    def add_trigger(self, begin, length):
        self.trigger = Span(self.span.reference, begin, length)

    def add_type(self, ontology, event_type, score=1):
        type_interp = Interp(self.interp_type)
        type_interp.add_fields('type', ontology + ":" + event_type, score)
        self.interps.append(type_interp)

    def add_arg(self, ontology, arg_role, entity_mention, score=1):
        self.args.append((ontology, arg_role, entity_mention, score))

    def json_rep(self):
        rep = super().json_rep()
        rep['args'] = []
        for ontology, arg_role, entity_mention, score in self.args:
            rep['args'].append(
                {
                    '@type': 'argument',
                    'type': ontology + ':' + arg_role,
                    'text': entity_mention.text,
                    'score': score,
                    'arg': entity_mention.id
                }
            )
        return rep


class RelationMention(InterpFrame):
    def __init__(self, fid, ontology, relation_type, arguments, score=1):
        super().__init__(fid, 'relation_mention', None,
                         'relation_mention_interp')
        self.relation_type = relation_type
        self.arguments = arguments
        rel_interp = Interp(self.interp_type)
        rel_interp.add_fields('type', ontology + ":" + relation_type, score)
        rel_interp.add_fields('args', arguments, score)
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

        self._span_map = defaultdict(list)

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

    def add_sentence(self, sentence_index, span):
        sent_id = self.get_id('sent', sentence_index)

        if sent_id in self.current_sentences:
            return sent_id

        docid = self.current_doc.id
        sent = Sentence(sent_id, docid, docid, span[0], span[1] - span[0], "")
        self.current_sentences[sent_id] = sent

        return sent_id

    def set_sentence_text(self, sent_id, text):
        self.current_sentences[sent_id].text = text

    def get_sentence_by_span(self, span):
        for sent_id, sentence in self.current_sentences.items():
            sent_begin = sentence.span.begin
            sent_end = sentence.span.length + sent_begin
            if span[0] >= sent_begin and span[1] <= sent_end:
                return sent_id, sentence

    def add_entity_mention(self, sent_id, span, text, ontology, entity_type):
        # Annotation on the same span will be reused.
        span = tuple(span)
        if span in self._span_map['entity']:
            entity_id = self._span_map['entity'][span]
            entity = self.current_entities[entity_id]
        else:
            entity_id = self.get_id('ent', self.entity_index)
            self.entity_index += 1
            sentence_start = self.current_sentences[sent_id].span.begin
            entity = EntityMention(entity_id, sent_id, sent_id,
                                   span[0] - sentence_start, span[1] - span[0],
                                   text)
            self.current_entities[entity_id] = entity

        entity.add_type(ontology, entity_type)
        return entity_id

    def add_event_mention(self, sent_id, trigger_span, text, ontology,
                          evm_type):
        # Annotation on the same span will be reused.
        trigger_span = tuple(trigger_span)
        if trigger_span in self._span_map['event_id']:
            event_id = self._span_map[trigger_span][1]
            event = self.current_events[event_id]
        else:
            event_id = self.get_id('evm', self.event_index)
            self.event_index += 1
            self._span_map[trigger_span] = ('event_mention', event_id)
            sent = self.current_sentences[sent_id]
            event = EventMention(event_id, sent_id, sent_id,
                                 sent.span.begin, sent.span.length,
                                 text)
            event.add_trigger(trigger_span[0] - sent.span.begin,
                              trigger_span[1] - trigger_span[1])
            self.current_events[event_id] = event

        event.add_type(ontology, evm_type)
        return event_id

    def get_id(self, prefix, index):
        return '%s:%s-%s-text-cmu-r%s-%d' % (
            self.ns_prefix,
            prefix,
            self.current_doc.doc_name,
            self.run_id,
            index,
        )

    def add_arg(self, evm_id, ent_id, ontology, arg_role):
        evm = self.current_events[evm_id]
        ent = self.current_entities[ent_id]
        evm.add_arg(ontology, arg_role, ent)

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
            json.dump(self.get_json_rep(), out, indent=2)
