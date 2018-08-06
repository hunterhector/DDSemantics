import os
import xml.etree.ElementTree as ElementTree
import logging
import json
from collections import defaultdict
import re
from traitlets.config import Configurable
from event.arguments.params import ModelPara
from event.arguments.arg_models import EventPairCompositionModel
from traitlets import (
    Unicode,
    Integer,
    Bool,
)
from traitlets.config.loader import PyFileConfigLoader
import glob

from collections import Counter


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
        return '[Span] %d:%d' % (self.begin, self.end)


class Annotation:
    def __init__(self, aid, anno_type, text, begin, length):
        self.aid = aid
        self.text = text
        self.anno_type = anno_type

        if begin is not None and length is not None:
            self.span = Span(int(begin), int(begin) + int(length))
        else:
            self.span = None

    def validate(self, doc_text):
        source_anno = doc_text[self.span.begin: self.span.end
                      ].replace('\n', ' ')
        if not source_anno == self.text:
            logging.warning(
                "Non-matching mention at [%s] between doc [%s] and "
                "specified [%s]" % (self.span, source_anno, self.text))
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


class Predicate(Annotation):
    def __init__(self, eid, begin, length, text, frame_type=None, realis=None):
        super().__init__(eid, 'Predicate', text, begin, length)
        self.arguments = {}
        self.frame_type = frame_type
        self.realis = realis

    def add_arg(self, arg_type, ent):
        self.arguments[arg_type] = ent

    def to_json(self):
        data = super().to_json()
        more = {
            'type': self.frame_type,
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
    def __init__(self, text=None, ranges=None, ignore_quote=False):
        self.entities = []
        self.events = []
        self.fillers = []
        self.relations = []
        self.ranges = ranges
        self.doc_text = text
        self.doc_type = None

        self.indices = Counter()

        if ignore_quote:
            self.remove_quote()

    def remove_quote(self):
        lqs = []
        lqe = []

        origin_len = len(self.doc_text)

        for stuff in re.finditer(r'<quote', self.doc_text):
            lqs.append((stuff.start(), stuff.end()))

        for stuff in re.finditer(r'</quote>', self.doc_text):
            lqe.append((stuff.start(), stuff.end()))

        if len(lqs) == len(lqe):
            quoted = zip([e for s, e in lqs], [s for s, e in lqe])
            for s, e in quoted:
                self.doc_text = self.doc_text[:s] + '>' + '_' * (e - s - 1) \
                                + self.doc_text[e:]
        else:
            logging.warning("Unbalanced quoted region.")
            input('Checking.')

        new_len = len(self.doc_text)

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
        self.doc_text = text

    def add_entity(self, entity_type=None, eid=None):
        if eid is None:
            eid = 'ent-%d' % self.indices['entity']
            self.indices['entity'] += 1

        if entity_type is None:
            entity_type = 'Entity'

        ent = Entity(eid, entity_type)
        self.entities.append(ent)
        return ent

    def add_hopper(self, eid=None):
        if eid is None:
            eid = 'h-%d' % self.indices['event']
            self.indices['event'] += 1

        event = Event(eid)
        self.events.append(event)
        return event

    def add_entity_mention(self, ent, offset, length, text=None,
                           eid=None, noun_type=None, entity_type=None):
        if entity_type is None:
            entity_type = ent.object_type

        if not eid:
            eid = 'em-%d' % self.indices['entity_mention']
            self.indices['entity_mention'] += 1

        em = EntityMention(eid, offset, length, text, noun_type, entity_type)

        em.fix_spaces(self.doc_text)

        is_valid = True
        if text and self.doc_text:
            is_valid = em.validate(self.doc_text)

        if is_valid:
            ent.add_mention(em)
            return em

    def add_predicate(self, hopper, offset, length, text=None,
                      eid=None, frame_type=None, realis=None):
        if eid is None:
            eid = 'evm-%d' % self.indices['event_mention']
            self.indices['event_mention'] += 1

        evm = Predicate(eid, offset, length, text, frame_type, realis)

        is_valid = True
        if text:
            is_valid = evm.validate(self.doc_text)

        if is_valid:
            hopper.add_mention(evm)
            return evm

    def add_filler(self, offset, length, text, eid=None, filler_type=None):
        if eid is None:
            eid = 'em-%d' % self.indices['entity_mention']
            self.indices['entity_mention'] += 1

        filler = Filler(eid, offset, length, text, filler_type)

        if text:
            if filler.validate(self.doc_text):
                self.fillers.append(filler)

        return filler

    def add_relation(self, rid, relation_type=None):
        relation = Relation(rid, relation_type=relation_type)
        self.relations.append(relation)
        return relation

    def dump(self, indent=None):
        doc = {
            'text': self.doc_text,
            'events': [e.to_json() for e in self.events],
            'entities': [e.to_json() for e in self.entities],
            'fillers': [e.to_json() for e in self.fillers],
            'relations': [e.to_json() for e in self.relations],
        }

        return json.dumps(doc, indent=indent)


class RichERE:
    def __init__(self, params):
        self.params = params

    def parse_ere(self, ere_file, doc):
        root = ElementTree.parse(ere_file).getroot()

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

                doc.add_entity_mention(
                    ent, ent_info['offset'], ent_info['length'], entity_text,
                    ent_info['id'],
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

                evm = doc.add_predicate(
                    event, offset, length, trigger_text, eid=evm_info['id'],
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

                    evm.add_arg(role, arg_ent_mention)

        for filler in root.find('fillers'):
            filler_info = filler.attrib
            doc.add_filler(
                filler_info['offset'], filler_info['length'], filler.text,
                eid=filler_info['id'], filler_type=filler_info['type'])

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

                rel_mention = RelationMention(rel_mention_id, trigger_begin,
                                              trigger_len, trigger_text,
                                              rel_realis)
                for role, ent in args.items():
                    rel_mention.add_arg(role, ent)

                relation.add_mention(rel_mention)

    def read_rich_ere(self, source_path, l_ere_path, ranges):
        with open(source_path) as source:
            text = source.read()
            doc = DEDocument(text, ranges, self.params.ignore_quote)
            for ere_path in l_ere_path:
                with open(ere_path) as ere:
                    logging.info("Processing: " + os.path.basename(ere_path))
                    self.parse_ere(ere, doc)
                    return doc

    def read_rich_ere_collection(self):
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
            doc = self.read_rich_ere(source, l_ere, ranges)
            with open(os.path.join(
                    self.params.out_dir, basename + '.json'), 'w') as out:
                out.write(doc.dump(indent=2))


class FrameNet:
    def __init__(self, params):
        self.params = params
        self.ns = {
            'icsi': 'http://framenet.icsi.berkeley.edu',
        }

    def parse_full_text(self, full_text_file, doc):
        print(full_text_file)

        root = ElementTree.parse(full_text_file).getroot()

        header = root.find('icsi:header', self.ns)

        corpus = header.find('icsi:corpus', self.ns).attrib['name']

        full_text = ''
        offset = 0

        annotations = []

        for sent in root.findall('icsi:sentence', self.ns):
            sent_text = sent.find('icsi:text', self.ns).text

            full_text += sent_text
            full_text += '\n'

            for anno_set in sent.findall('icsi:annotationSet', self.ns):
                targets = []
                fes = []

                if not 'frameName' in anno_set.attrib:
                    continue

                frame_name = anno_set.attrib['frameName']

                for layer in anno_set.findall('icsi:layer', self.ns):
                    layer_type = layer.attrib['name']

                    if layer_type == 'Target':
                        label = layer.find('icsi:label', self.ns)

                        if label is not None:
                            s = int(label.attrib['start'])
                            e = int(label.attrib['end']) + 1
                            text = sent_text[s: e]
                            targets.append((s + offset, e + offset, text))
                    elif layer_type == 'FE':
                        for label in layer.findall('icsi:label', self.ns):
                            label_name = label.attrib['name']

                            if 'itype' in label.attrib:
                                # Null instantiation.
                                pass
                            else:
                                s = int(label.attrib['start'])
                                e = int(label.attrib['end']) + 1
                                text = sent_text[s: e]
                                fes.append(
                                    (s + offset, e + offset, text, label_name)
                                )

                if targets:
                    max_len = 0
                    target = None
                    for i, (s, e, text) in enumerate(targets):
                        if e - s > max_len:
                            max_len = e - s
                            target = s, e, text

                    annotations.append((frame_name, target, fes))

            offset = len(full_text)

        doc.set_text(full_text)

        for frame_name, target, fes in annotations:
            ev = doc.add_hopper()
            target_start, target_end, text = target
            evm = doc.add_predicate(
                ev, target_start, target_end - target_start, text=text,
                frame_type=frame_name)

            for start, end, fe_text, role in fes:
                filler = doc.add_filler(start, end - start, fe_text)
                evm.add_arg(role, filler.aid)

        return doc

    def read_fn_data(self):
        full_text_dir = os.path.join(self.params.fn_path, 'fulltext')

        doc = DEDocument()

        for full_text_path in glob.glob(full_text_dir + '/*.xml'):
            self.parse_full_text(full_text_path, doc)

            basename = os.path.basename(full_text_path).replace(".xml", '')

            with open(os.path.join(
                    self.params.out_dir, basename + '.json'), 'w') as out:
                out.write(doc.dump(indent=2))


class Conll:
    def __init__(self, params):
        self.params = params

    def parse_conll_data(self, conll_in):
        text = ''
        offset = 0

        arg_text = []
        sent_predicates = []
        sent_args = defaultdict(list)
        doc = DEDocument()

        props = []

        for line in conll_in:
            parts = line.strip().split()
            if len(parts) < 8:
                text += '\n\n'
                offset += 2

                for index, predicate in enumerate(sent_predicates):
                    arg_content = sent_args[index]
                    props.append((predicate, arg_content))

                sent_predicates.clear()
                sent_args.clear()
                arg_text.clear()

                continue

            fname, _, index, token, pos, parse, lemma, sense = parts[:8]
            pb_annos = parts[8:]

            if len(arg_text) == 0:
                arg_text = [None] * len(pb_annos)

            domain = fname.split('/')[1]

            start = offset
            end = start + len(token)

            text += token + ' '
            offset += len(token) + 1

            for index, t in enumerate(arg_text):
                if t:
                    arg_text[index] += ' ' + token

            if not sense == '-':
                sent_predicates.append((start, end, token))

            for index, anno in enumerate(pb_annos):
                if anno == '(V*)':
                    continue

                if anno.startswith('('):
                    role = anno.strip('(').strip(')').strip('*')
                    sent_args[index].append([role, start])

                    arg_text[index] = token
                if anno.endswith(')'):
                    sent_args[index][-1].append(end)
                    sent_args[index][-1].append(arg_text[index])
                    arg_text[index] = ''

        doc.set_text(text)

        for (p_start, p_end, p_token), args in props:
            hopper = doc.add_hopper()

            pred = doc.add_predicate(
                hopper, p_start, p_end - p_start, p_token)

            for role, arg_start, arg_end, arg_text in args:
                filler = doc.add_filler(
                    arg_start, arg_end - arg_start, arg_text)

                pred.add_arg(role, filler.aid)

        return doc

    def read_pb_release(self):
        for dirname in os.listdir(self.params.in_dir):
            full_dir = os.path.join(self.params.in_dir, dirname)
            for root, dirs, files in os.walk(full_dir):
                for f in files:
                    if not f.endswith('gold_conll'):
                        continue

                    full_path = os.path.join(root, f)

                    out_dir = os.path.join(self.params.out_dir, dirname)

                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)

                    out_path = os.path.join(
                        out_dir, f.replace('gold_conll', 'json'))

                    with open(full_path) as conll_in, \
                            open(out_path, 'w') as out:
                        doc = self.parse_conll_data(conll_in)
                        out.write(doc.dump(indent=2))


if __name__ == '__main__':
    from event.util import basic_console_log
    from event.util import load_command_line_config
    import sys


    class EreConf(Configurable):
        source = Unicode(help='Plain source input directory').tag(config=True)
        ere = Unicode(help='ERE input data').tag(config=True)
        out_dir = Unicode(help='Output directory').tag(config=True)
        src_ext = Unicode(help='Source file extension',
                          default_value='.xml').tag(config=True)
        ere_ext = Unicode(help='Ere file extension',
                          default_value='.rich_ere.xml').tag(config=True)
        ere_split = Bool(help='Whether split ere based on the file names').tag(
            config=True)
        ignore_quote = Bool(help='model name', default_value=False).tag(
            config=True)


    class FrameNetConf(Configurable):
        fn_path = Unicode(help='FrameNet dataset path.').tag(config=True)
        out_dir = Unicode(help='Output directory').tag(config=True)


    class ConllConf(Configurable):
        in_dir = Unicode(help='Conll file input directory').tag(config=True)
        out_dir = Unicode(help='Output directory').tag(config=True)


    basic_console_log()
    data_format = sys.argv[1]
    cl_conf = load_command_line_config(sys.argv[2:])

    if data_format == 'rich_ere':
        basic_para = EreConf(config=cl_conf)
        parser = RichERE(basic_para)
        parser.read_rich_ere_collection()
    elif data_format == 'framenet':
        basic_para = FrameNetConf(config=cl_conf)
        parser = FrameNet(basic_para)
        parser.read_fn_data()
    elif data_format == 'conll':
        basic_para = ConllConf(config=cl_conf)
        parser = Conll(basic_para)
        parser.read_pb_release()
