import json
import os


# class EntityMention:
#     def __init__(self, ):

class InterpCollector:
    def __init__(self, component_name, run_id, out_path, namespace):
        self.frame_collection = {
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
            },
            "frames": [],
        }
        self.out_path = out_path

        self._added_docs = set()

        self.sentence_spans = {}
        self.frames = {}
        self.run_id = run_id
        self.frame_id = 0

        self.entity_index = 0
        self.event_index = 0
        self.relation_index = 0

        self.onto_namespace = 'aida'
        self.ns_prefix = namespace

        self.doc_name = ""
        self.media_type = ""
        self.file_type = ""

    def add_doc(self, doc_name, doc_type, media_type, file_type):
        self.doc_name = doc_name
        self.media_type = media_type
        self.file_type = file_type

        ns_docid = self.get_docid()

        if ns_docid in self._added_docs:
            return

        # if 'document_id' in self.frame_collection['meta']:
        #     if self.frame_collection['meta']['document_id'] == ns_docid:
        #         return

        if self.frame_collection['frames']:
            self.write()
            self.frame_collection['frames'].clear()

        self._added_docs.add(ns_docid)
        # self.frame_collection['meta']['document_id'] = ns_docid

        doc_info = {
            '@type': 'document',
            '@id': ns_docid,
            'type': doc_type,
            'media_type': 'text',
        }
        self.frame_collection['frames'].append(doc_info)

    def get_docid(self):
        return self.ns_prefix + ":" + self.doc_name + "-" + self.media_type \
               + "." + self.file_type

    def add_sentence(self, sentence_index, span, text):
        sent_id = self.get_id('sent', sentence_index)
        if sent_id in self.frames:
            return sent_id

        sent_info = {
            '@type': 'sentence',
            '@id': sent_id,
            'extent': {
                '@type': 'text_span',
                'reference': self.get_docid(),
                'start': span[0],
                'length': span[1] - span[0],
            },
            'text': text,
            'parent_scope': self.get_docid(),
        }
        self.frames[sent_id] = sent_info
        self.frame_collection['frames'].append(sent_info)
        self.sentence_spans[sent_id] = span

        return sent_id

    def add_entity(self, sent_id, span, text, entity_type):
        # sentence_id = self.get_id('sent', sentence_index)
        entity_id = self.get_id('ent', self.entity_index)
        self.entity_index += 1
        entity_info = {
            '@type': 'entity_mention',
            '@id': entity_id,
            'reference': sent_id,
            'text': text,
            'parent_scope': sent_id,
            'interp': {
                '@type': 'entity_mention_interp',
                'type': self.onto_namespace + ":" + entity_type,
            }
        }
        sentence_start = self.sentence_spans[sent_id][0]
        entity_info['start'] = span[0] - sentence_start
        entity_info['length'] = span[1] - span[0]
        self.frame_collection['frames'].append(entity_info)

        self.frames[entity_id] = entity_info

        return entity_id

    def add_event(self, sentence_index, trigger_span, extent_span, text,
                  evm_type):
        sentence_id = self.get_id('sent', sentence_index)
        event_id = self.get_id('evm', self.event_index)
        self.event_index += 1

        sentence_start = self.sentence_spans[sentence_id][0]

        event_info = {
            '@type': 'event_mention',
            '@id': event_id,
            'trigger': {
                '@type': 'text_span',
                'reference': sentence_id,
                'start': trigger_span[0] - sentence_start,
                'length': trigger_span[1] - trigger_span[0],
            },
            'extent': {
                '@type': 'text_span',
                'reference': sentence_id,
                'start': extent_span[0] - sentence_start,
                'length': extent_span[1] - extent_span[0],
            },
            'text': text,
            'parent_scope': sentence_id,
            'interp': {
                '@type': 'event_mention_interp',
                'type': self.onto_namespace + ":" + evm_type,
                'args': [],
            }
        }

        self.frame_collection['frames'].append(event_info)
        self.frames[event_id] = event_info

        return event_id

    def get_id(self, prefix, index):
        return '%s:%s-%s-text-cmu-r%s-%d' % (
            self.ns_prefix,
            prefix,
            self.doc_name,
            self.run_id,
            index,
        )

    def add_arg(self, evm_id, ent_id, arg_role):
        entity = self.frames[ent_id]

        arg = {
            '@type': 'argument',
            'type': self.onto_namespace + ":" + arg_role,
            'text': entity['text'],
            'arg': ent_id,
        }

        self.frames[evm_id]['interp']['args'].append(arg)

    def add_relation(self, arguments, relation_type):
        relation_id = self.get_id('relm', self.relation_index)
        self.relation_index += 1
        relation_info = {
            '@type': 'relation_mention',
            '@id': relation_id,
            'interp': {
                '@type': 'relation_mention_interp',
                'type': relation_type,
                'arguments': arguments
            }
        }
        self.frame_collection['frames'].append(relation_info)
        self.frames[relation_id] = relation_info

    def write(self, append=False):
        if append:
            mode = 'a' if os.path.exists(self.out_path) else 'w'
        else:
            mode = 'w'

        with open(self.out_path, mode) as out:
            json.dump(self.frame_collection, out, indent=2)
