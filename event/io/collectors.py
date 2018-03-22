import json
import os


# class EntityMention:
#     def __init__(self, ):

class InterpCollector:
    def __init__(self, component_name, run_id, out_path):
        self.frame_collection = {
            'object_type': 'frame_collection',
            'object_meta': {
                'object_type': 'meta_info',
                'component': component_name,
                'organization': 'CMU',
            },
            "frames": [],
        }
        self.out_path = out_path

        self.sentence_spans = {}
        self.frames = {}
        self.run_id = run_id
        self.frame_id = 0

        self.entity_index = 0
        self.event_index = 0
        self.relation_index = 0

    def add_doc(self, docid, doc_type):
        if 'document_id' in self.frame_collection['object_meta']:
            if self.frame_collection['object_meta']['document_id'] == docid:
                return

        if self.frame_collection['frames']:
            self.write()
            self.frame_collection['frames'].clear()

        self.frame_collection['object_meta']['document_id'] = docid
        doc_info = {
            'object_type': 'document',
            'object_id': docid,
            'type': doc_type,
            'media_type': 'text',
        }
        self.frame_collection['frames'].append(doc_info)

    def add_sentence(self, sentence_index, span):
        sent_id = self.get_id('sent', sentence_index)
        if sent_id in self.frames:
            return

        doc_id = self.frame_collection['object_meta']['document_id']
        sent_info = {
            'object_type': 'sentence',
            'object_id': sent_id,
            'extent': {
                'object_type': 'text_span',
                'reference': doc_id,
                'start': span[0],
                'length': span[1] - span[0],
            },
            'text': '',
            'parent_scope': doc_id,
        }
        self.frames[sent_id] = sent_info
        self.frame_collection['frames'].append(sent_info)
        self.sentence_spans[sent_id] = span
        print("Adding sentence id", sent_id)

    def get_id(self, prefix, index):
        return '%s-%s-text-cmu-r%s-%d' % (
            prefix,
            self.frame_collection['object_meta']['document_id'],
            self.run_id,
            index,
        )

    def add_entity(self, sentence_index, span, text, entity_type):
        sentence_id = self.get_id('sent', sentence_index)
        entity_id = self.get_id('ent', self.entity_index)
        self.entity_index += 1
        entity_info = {
            'object_type': 'entity_mention',
            'object_id': entity_id,
            'reference': sentence_id,
            'text': text,
            'parent_scope': sentence_id,
            'interp': {
                'object_type': 'entity_mention_interp',
                'type': entity_type,
            }
        }
        sentence_start = self.sentence_spans[sentence_id][0]
        entity_info['start'] = span[0] - sentence_start
        entity_info['end'] = span[1] - span[0]
        self.frame_collection['frames'].append(entity_info)
        self.frames[entity_id] = entity_info

    def add_event(self, sentence_index, trigger_span, extent_span, text,
                  evm_type):
        sentence_id = self.get_id('sent', sentence_index)
        event_id = self.get_id('evm', self.event_index)
        self.entity_index += 1
        event_info = {
            'object_type': 'event_mention',
            'object_id': event_id,
            'trigger': {
                'object_type': 'text_span',
                'reference': sentence_id,
            },
            'extent': {
                'object_type': 'text_span',
                'reference': sentence_id,
            },
            'text': text,
            'parent_scope': sentence_id,
            'interp': {
                'object_type': 'event_mention_interp',
                'type': evm_type,
                'args': [],
            }
        }

        sentence_start = self.sentence_spans[sentence_id][0]
        event_info['trigger']['start'] = trigger_span[0] - sentence_start
        event_info['trigger']['end'] = trigger_span[1] - trigger_span[0]

        event_info['extent']['start'] = extent_span[0] - sentence_start
        event_info['extent']['end'] = extent_span[1] - extent_span[0]

        self.frame_collection['frames'].append(event_info)
        self.frames[event_id] = event_info

    def add_arg(self, evm_index, ent_index, arg_role):
        evm_id = self.get_id('evm', evm_index)
        ent_id = self.get_id('ent', ent_index)

        entity = self.frames[ent_id]

        arg = {
            'object_type': 'argument',
            'type': arg_role,
            'text': entity['text'],
            'arg': ent_id,
        }

        self.frames[evm_id]['interp']['args'].append(arg)

    def add_relation(self, arguments, relation_type):
        relation_id = self.get_id('relm', self.relation_index)
        self.relation_index += 1
        relation_info = {
            'object_type': 'relation_mention',
            'object_id': relation_id,
            'interp': {
                'object_type': 'relation_mention_interp',
                'type': relation_type,
                'arguments': arguments
            }
        }
        self.frame_collection['frames'].append(relation_info)
        self.frames[relation_id] = relation_info

    def write(self):
        mode = 'a' if os.path.exists(self.out_path) else 'w'
        with open(self.out_path, mode) as out:
            json.dump(self.frame_collection, out, indent=2)
