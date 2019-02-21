import logging
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

from event.io.dataset.base import (
    Span,
    DataLoader,
    DEDocument,
    RelationMention,
)


class RichERE(DataLoader):
    def __init__(self, params, corpus, with_doc=False):
        super().__init__(params, corpus, with_doc)
        self.params = params

    def parse_ere(self, ere_file, doc):
        root = ET.parse(ere_file).getroot()

        doc_info = root.attrib

        doc.set_id = doc_info['doc_id']
        doc.set_doc_type = doc_info['source_type']

        for entity_node in root.find('entities'):
            entity_ids = []

            ent = doc.add_entity(entity_node.attrib['type'],
                                 entity_node.attrib['id'])

            for entity_mention in entity_node.findall('entity_mention'):
                ent_info = entity_mention.attrib
                entity_ids.append(ent_info['id'])

                entity_text = entity_mention.find('mention_text').text

                entity_span = Span(ent_info['offset'], ent_info['length'])

                doc.add_entity_mention(
                    ent, entity_span, entity_text,
                    ent_info['id'],
                    noun_type=ent_info['noun_type'],
                    entity_type=ent_info.get('type', None),
                )

        for filler in root.find('fillers'):
            filler_info = filler.attrib
            b = int(filler_info['offset'])
            l = int(filler_info['length'])
            doc.add_filler(
                Span(b, b + l), filler.text,
                eid=filler_info['id'], filler_type=filler_info['type']
            )

        for event_node in root.find('hoppers'):
            evm_ids = []

            event = doc.add_hopper(event_node.attrib['id'])

            for event_mention in event_node.findall('event_mention'):
                evm_info = event_mention.attrib
                evm_ids.append(evm_info['id'])

                trigger = event_mention.find('trigger')
                trigger_text = trigger.text
                offset = trigger.attrib['offset']
                length = trigger.attrib['length']

                evm = doc.add_predicate(
                    event, Span(offset, offset + length), trigger_text,
                    eid=evm_info['id'],
                    frame_type=evm_info['type'] + '_' + evm_info['subtype'],
                    realis=evm_info['realis'])

                for em_arg in event_mention.findall('em_arg'):
                    arg_info = em_arg.attrib

                    arg_ent_mention = None
                    if 'entity_mention_id' in arg_info:
                        arg_ent_mention = arg_info['entity_mention_id']
                    if 'filler_id' in arg_info:
                        arg_ent_mention = arg_info['filler_id']

                    role = arg_info['role']

                    doc.add_argument_mention(evm, arg_ent_mention, role)

        for relation_node in root.find('relations'):
            relation_info = relation_node.attrib
            relation = doc.add_relation(
                relation_info['id'],
                relation_type=relation_info['type'] + '_' + relation_info[
                    'subtype']
            )

            for rel_mention_node in relation_node.findall('relation_mention'):
                rel_mention_id = rel_mention_node.attrib['id']
                rel_realis = rel_mention_node.attrib['realis']

                args = {}
                for mention_part in rel_mention_node:
                    if mention_part.tag.startswith('rel_arg'):
                        if 'entity_mention_id' in mention_part.attrib:
                            ent_id = mention_part.attrib['entity_mention_id']
                        else:
                            ent_id = mention_part.attrib['filler_id']

                        role = mention_part.attrib['role']
                        args[role] = ent_id

                trigger = rel_mention_node.find('trigger')
                if trigger is not None:
                    trigger_text = trigger.text
                    trigger_begin = trigger.attrib['offset']
                    trigger_len = trigger.attrib['length']
                else:
                    trigger_text = ''
                    trigger_begin = None
                    trigger_len = None

                rel_mention = RelationMention(
                    rel_mention_id, Span(trigger_begin, trigger_len),
                    trigger_text, rel_realis
                )

                for role, ent in args.items():
                    rel_mention.add_arg(role, ent)

                relation.add_mention(rel_mention)

    def read_rich_ere(self, corpus, source_path, l_ere_path, ranges):
        with open(source_path) as source:
            text = source.read()
            doc = DEDocument(corpus, text, ranges, self.params.ignore_quote)
            for ere_path in l_ere_path:
                with open(ere_path) as ere:
                    logging.info("Processing: " + os.path.basename(ere_path))
                    self.parse_ere(ere, doc)
                    return doc

    def get_doc(self):
        super().get_doc()

        sources = {}
        eres = defaultdict(list)
        annotate_ranges = defaultdict(list)

        for fn in os.listdir(self.params.ere):
            basename = fn.replace(self.params.ere_ext, '')
            if self.params.ere_split:
                parts = basename.split('_')
                r = [int(p) for p in parts[-1].split('-')]
                annotate_ranges[basename].append(r)
                basename = '_'.join(parts[:-1])

            ere_fn = os.path.join(self.params.ere, fn)
            eres[basename].append(ere_fn)

        for fn in os.listdir(self.params.source):
            txt_base = fn.replace(self.params.src_ext, '')
            if txt_base in eres:
                sources[txt_base] = os.path.join(self.params.source, fn)

        if not os.path.exists(self.params.out_dir):
            os.makedirs(self.params.out_dir)

        for basename, source in sources.items():
            l_ere = eres[basename]
            ranges = annotate_ranges[basename]
            doc = self.read_rich_ere(self.corpus, source, l_ere, ranges)
            yield doc
