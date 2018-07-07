import os
import xml.etree.ElementTree as ElementTree
import logging
import json
from collections import defaultdict
import re


class Span:
    def __init__(self, begin, end):
        self.begin = int(begin)
        self.end = int(end)

    def __lt__(self, other):
        if self.begin < other.begin:
            return True
        elif self.begin == other.begin:
            return self.end < self.end
        else:
            return False

    def __eq__(self, other):
        return self.__class__ == other.__class__ and \
               self.begin == other.begin and self.end == other.end

    def __hash__(self):
        return hash((self.begin, self.end))

    def __str__(self):
        return '%d:%d' % (self.begin, self.end)


class Annotation:
    def __init__(self, aid, anno_type, text, begin, length):
        self.aid = aid
        self.text = text
        self.anno_type = anno_type

        if begin and length:
            self.span = Span(int(begin), int(begin) + int(length))
        else:
            self.span = None

    def validate(self, source_text):
        source_anno = source_text[
                      self.span.begin: self.span.end].replace('\n', ' ')
        if not source_anno == self.text:
            logging.warning(
                "Invalid mention at [%s] because not match between source "
                "[%s] and target [%s]" % (self.span, self.text, source_anno))
            return False
        return True

    def fix_spaces(self, source_text):
        source_anno = source_text[
                      self.span.begin: self.span.end].replace('\n', ' ')

        if not source_anno == self.text:
            if source_anno.strip() == self.text:
                leadings = len(source_anno) - len(source_anno.lstrip())
                trailings = len(source_anno) - len(source_anno.rstrip())

                begin, end = self.span.begin, self.span.end
                self.span = Span(begin + leadings, end - trailings)

                logging.warning(
                    "Fixing space only mismatch from [{}:{}] to "
                    "[{}]".format(begin, end, self.span))

    def to_json(self):
        data = {
            'id': self.aid,
            'annotation': self.anno_type,
        }

        if self.text:
            data['text'] = self.text

        if self.span:
            data['begin'] = self.span.begin
            data['end'] = self.span.end

        return data


class EventMention(Annotation):
    def __init__(self, eid, begin, length, text, event_type=None, realis=None):
        super().__init__(eid, 'EventMention', text, begin, length)
        self.arguments = {}
        self.event_type = event_type
        self.realis = realis

    def add_arg(self, arg_type, ent):
        self.arguments[arg_type] = ent

    def to_json(self):
        data = super().to_json()
        more = {
            'type': self.event_type,
            'realis': self.realis,
            'arguments': {},
        }

        for key, value in self.arguments.items():
            more['arguments'][key] = value

        data.update(more)
        return data


class EntityMention(Annotation):
    def __init__(self, eid, begin, length, text, noun_type=None,
                 entity_type=None):
        super().__init__(eid, 'EntityMention', text, begin, length)
        self.entity_type = entity_type
        self.noun_type = noun_type

    def to_json(self):
        data = super().to_json()
        more = {
            'type': self.entity_type,
            'noun_type': self.noun_type,
        }
        data.update(more)
        return data


class Filler(Annotation):
    def __init__(self, eid, begin, length, text, filler_type=None):
        super().__init__(eid, 'Filler', text, begin, length)
        self.filler_type = filler_type

    def to_json(self):
        data = super().to_json()
        more = {
            'type': self.filler_type,
        }
        data.update(more)
        return data


class RelationMention(Annotation):
    def __init__(self, rid, begin, length, text, realis):
        super().__init__(rid, 'RelationMention', text, begin, length)
        self.args = {}
        self.realis = realis

    def add_arg(self, role, entity_id):
        self.args[role] = entity_id

    def to_json(self):
        data = super().to_json()
        more = {
            'args': self.args,
            'realis': self.realis,
        }
        data.update(more)
        return data


class Top:
    def __init__(self, eid, top_type, object_type=None):
        self.eid = eid
        self.top_type = top_type
        self.object_type = object_type
        self.mentions = []

    def to_json(self):
        data = {
            'id': self.eid,
            'annotation': self.top_type,
            'mentions': [m.to_json() for m in self.mentions]
        }

        if self.object_type:
            data['type'] = self.object_type

        return data

    def add_mention(self, mention):
        self.mentions.append(mention)


class Entity(Top):
    def __init__(self, eid, object_type):
        super().__init__(eid, 'Entity', object_type)


class Event(Top):
    def __init__(self, eid):
        super().__init__(eid, 'Event')


class Relation:
    def __init__(self, rid, relation_type=None):
        self.rid = rid
        self.mentions = []
        self.relation_type = relation_type

    def add_mention(self, rel_mention):
        self.mentions.append(rel_mention)

    def to_json(self):
        data = {
            'annotation': 'Relation',
            'id': self.rid,
            'type': self.relation_type,
            'mentions': [],
        }

        for mention in self.mentions:
            data['mentions'].append(mention.to_json())

        return data


class DEDocument:
    def __init__(self, text, ranges, ignore_quote=False):
        # self.event_mentions = []
        # self.entity_mentions = []
        self.entities = []
        self.events = []
        self.fillers = []
        self.relations = []
        self.ranges = ranges
        self.text = text
        self.doc_type = None

        if ignore_quote:
            self.remove_quote()

    def remove_quote(self):
        lqs = []
        lqe = []

        origin_len = len(self.text)

        for stuff in re.finditer(r'<quote', self.text):
            lqs.append((stuff.start(), stuff.end()))

        for stuff in re.finditer(r'</quote>', self.text):
            lqe.append((stuff.start(), stuff.end()))

        if len(lqs) == len(lqe):
            quoted = zip([e for s, e in lqs], [s for s, e in lqe])
            for s, e in quoted:
                self.text = self.text[:s] + '>' + '_' * (e - s - 1) \
                            + self.text[e:]
        else:
            logging.warning("Unbalanced quoted region.")
            input('Checking.')

        new_len = len(self.text)

        assert origin_len == new_len

    def filter_text(self, text):
        curr = 0

        unannotated = []
        for s, e in self.ranges:
            if curr < s:
                unannotated.append((curr, s))
            curr = e

        end = len(text)
        if curr < end:
            unannotated.append((curr, end))

        for s, e in unannotated:
            logging.info("Marking {}:{} as unannotated".format(s, e))
            old_len = len(text)
            text = text[:s] + ['-'] * len(e - s) + text[e:]
            new_len = len(text)
            assert old_len == new_len

        return text

    def set_id(self, docid):
        self.docid = docid

    def set_doc_type(self, doc_type):
        self.doc_type = doc_type

    def set_text(self, text):
        self.text = text

    def add_entity(self, eid, entity_type):
        ent = Entity(eid, entity_type)
        self.entities.append(ent)
        return ent

    def add_hopper(self, eid):
        event = Event(eid)
        self.events.append(event)
        return event

    def add_entity_mention(self, ent, eid, offset, length, text, noun_type=None,
                           entity_type=None):
        if not entity_type:
            entity_type = ent.object_type

        em = EntityMention(eid, offset, length, text, noun_type, entity_type)

        em.fix_spaces(self.text)

        if text:
            if em.validate(self.text):
                ent.add_mention(em)
                return em

    def add_event_mention(self, hopper, eid, offset, length, text,
                          event_type=None, realis=None):

        evm = EventMention(eid, offset, length, text, event_type, realis)
        if text:
            if evm.validate(self.text):
                hopper.add_mention(evm)
                return evm

    def add_filler(self, eid, offset, length, text, filler_type=None):
        filler = Filler(eid, offset, length, text, filler_type)
        if text:
            if filler.validate(self.text):
                self.fillers.append(filler)

    def add_relation(self, rid, relation_type=None):
        relation = Relation(rid, relation_type=relation_type)
        self.relations.append(relation)
        return relation

    def dump(self, indent=None):
        doc = {
            'text': self.text,
            'entities': [e.to_json() for e in self.entities],
            'events': [e.to_json() for e in self.events],
            'fillers': [e.to_json() for e in self.fillers],
            'relations': [e.to_json() for e in self.relations],
        }

        return json.dumps(doc, indent=indent)


def parse_ere(ere_file, doc):
    root = ElementTree.parse(ere_file).getroot()

    doc_info = root.attrib

    doc.set_id = doc_info['doc_id']
    doc.set_doc_type = doc_info['source_type']

    for entity_node in root.find('entities'):
        entity_ids = []

        ent = doc.add_entity(entity_node.attrib['id'],
                             entity_node.attrib['type'])

        for entity_mention in entity_node.findall('entity_mention'):
            ent_info = entity_mention.attrib
            entity_ids.append(ent_info['id'])

            entity_text = entity_mention.find('mention_text').text

            doc.add_entity_mention(
                ent, ent_info['id'], ent_info['offset'], ent_info['length'],
                text=entity_text,
                noun_type=ent_info['noun_type'],
                entity_type=ent_info.get('type', None),
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

            evm = doc.add_event_mention(
                event, evm_info['id'], offset, length, text=trigger_text,
                event_type=evm_info['type'] + '_' + evm_info['subtype'],
                realis=evm_info['realis'],
            )

            for em_arg in event_mention.findall('em_arg'):
                arg_info = em_arg.attrib

                arg_ent_mention = None
                if 'entity_mention_id' in arg_info:
                    arg_ent_mention = arg_info['entity_mention_id']
                if 'filler_id' in arg_info:
                    arg_ent_mention = arg_info['filler_id']

                role = arg_info['role']

                evm.add_arg(role, arg_ent_mention)

    for filler in root.find('fillers'):
        filler_info = filler.attrib
        doc.add_filler(filler_info['id'], filler_info['offset'],
                       filler_info['length'], filler.text, filler_info['type'])

    for relation_node in root.find('relations'):
        relation_info = relation_node.attrib
        relation = doc.add_relation(
            relation_info['id'],
            relation_type=relation_info['type'] + '_' + relation_info['subtype']
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

            rel_mention = RelationMention(rel_mention_id, trigger_begin,
                                          trigger_len, trigger_text, rel_realis)
            for role, ent in args.items():
                rel_mention.add_arg(role, ent)

            relation.add_mention(rel_mention)


def read_rich_ere(source_path, l_ere_path, ranges, ignore_quote=False):
    with open(source_path) as source:
        text = source.read()
        print(source_path)
        doc = DEDocument(text, ranges, ignore_quote)
        for ere_path in l_ere_path:
            with open(ere_path) as ere:
                logging.info("Processing :" + ere_path)
                parse_ere(ere, doc)
                return doc


def read_rich_ere_collection(source_dir, ere_dir, output_dir, src_ext='.txt',
                             ere_ext='.rich_ere.xml', ere_split=False,
                             ignore_quote=False):
    sources = {}
    eres = defaultdict(list)
    annotate_ranges = defaultdict(list)

    for fn in os.listdir(ere_dir):
        basename = fn.replace(ere_ext, '')
        if ere_split:
            parts = basename.split('_')
            r = [int(p) for p in parts[-1].split('-')]
            annotate_ranges[basename].append(r)
            basename = '_'.join(parts[:-1])

        ere_fn = os.path.join(ere_dir, fn)
        eres[basename].append(ere_fn)

    for fn in os.listdir(source_dir):
        txt_base = fn.replace(src_ext, '')
        if txt_base in eres:
            sources[txt_base] = os.path.join(source_dir, fn)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for basename, source in sources.items():
        l_ere = eres[basename]
        ranges = annotate_ranges[basename]
        doc = read_rich_ere(source, l_ere, ranges, ignore_quote)
        with open(os.path.join(output_dir, basename + '.json'), 'w') as out:
            out.write(doc.dump(indent=2))


if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Parse ERE into JSON.')

    parser.add_argument('--source', action="store")
    parser.add_argument('--ere', action="store")
    parser.add_argument('--out', action="store")
    parser.add_argument('--source_ext', default='.xml')
    parser.add_argument('--target_ext', default='.rich_ere.xml')
    parser.add_argument('--ere_split', action="store_true")
    parser.add_argument('--ignore_quote', action="store_true")

    args = parser.parse_args()

    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} '
               '%(levelname)s - %(message)s',
        handlers=[stdout_handler]
    )

    read_rich_ere_collection(
        args.source, args.ere, args.out, args.source_ext, args.target_ext,
        args.ere_split, args.ignore_quote
    )
