import os
import xml.etree.ElementTree as ET
import logging
import json
from collections import defaultdict
import re
from traitlets.config import Configurable
from traitlets import (
    Unicode,
    Integer,
    Bool,
)
import glob
from nltk.corpus import (
    NombankCorpusReader,
    BracketParseCorpusReader,
)
from nltk.data import FileSystemPathPointer

from collections import Counter

from nltk.corpus.reader.nombank import (
    NombankChainTreePointer,
    NombankSplitTreePointer,
    NombankTreePointer,
)


def find_close_mention(event_mentions, ent_mentions):
    min_dist = None
    closest_ent = None
    closest_evm = None

    for evm in event_mentions:
        for ent in ent_mentions:
            dist = ent.span.begin - evm.span.end

            if dist < 0:
                dist = evm.span.begin - ent.span.end

            if not min_dist:
                min_dist = dist
                closest_ent = ent
                closest_evm = evm
            elif dist < min_dist:
                min_dist = dist
                closest_ent = ent
                closest_evm = evm

    return closest_ent, closest_evm, min_dist


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
        self.text = text.replace('\n', ' ')
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
                "provided [%s]" % (self.span, source_anno, self.text))
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
    def __init__(self, doc, eid, begin, length, text, frame_type=None,
                 realis=None):
        super().__init__(eid, 'Predicate', text, begin, length)
        self.arguments = {}
        self.frame_type = frame_type
        self.realis = realis
        self.doc = doc

    def add_arg(self, arg_type, ent):
        self.arguments[arg_type] = ent
        # <ENTITY> is a generic type for argument holder.
        self.doc.add_arg_type(self.frame_type, arg_type, '<ENTITY>')

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
        if self.entity_type:
            data['type'] = self.entity_type

        if self.noun_type:
            data['noun_type'] = self.noun_type

        return data


class Filler(Annotation):
    def __init__(self, eid, begin, length, text, filler_type=None):
        super().__init__(eid, 'Filler', text, begin, length)
        self.entity_type = filler_type

    def to_json(self):
        data = super().to_json()

        if self.entity_type:
            data['type'] = self.entity_type
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


class Corpus:
    def __init__(self):
        self.documents = []
        self.entity_types = set()
        self.event_ontos = {}

    def add_doc(self, document):
        self.documents.append(document)

        for ent in document.entities:
            self.add_entity_type(ent.entity_type)

        for ev in document.events:
            for evm in ev:
                self.add_event_type(evm.frame_type)

    def add_entity_type(self, entity_type):
        self.entity_types.add(entity_type)

    def add_event_type(self, event_type):
        if event_type not in self.event_ontos:
            self.event_ontos[event_type] = defaultdict(set)

    def add_arg_type(self, evm_type, arg_type, entity_type):
        self.event_ontos[evm_type][arg_type].add(entity_type)

    def get_brat_config(self):
        print("This corpus contains %d entity types." % len(self.entity_types))
        print("This corpus contains %d event types." % len(self.event_ontos))

        config = '[entities]\n'
        config += '\n'.join(self.entity_types)

        config += '\n\n[relations]\n'
        config += 'Event_Coref\tArg1:<EVENT>, Arg2:<EVENT>\n'
        config += 'Entity_Coref\tArg1:<ENTITY>, Arg2:<ENTITY>\n'

        config += '\n\n[events]\n'

        for t, arg_types in self.event_ontos.items():
            config += t
            config += '\t'

            role_pairs = []
            for role, ent_set in arg_types.items():
                role_pairs.append(role + ':' + '|'.join(ent_set))

            config += ', '.join(role_pairs)

            config += '\n'

        config += '\n\n[attributes]'

        return config


class DEDocument:
    def __init__(self, corpus, text=None, ranges=None, ignore_quote=False):
        self.corpus = corpus
        self.entities = []
        self.events = []
        self.fillers = []
        self.relations = []
        self.ranges = ranges
        self.doc_text = text
        self.doc_type = None
        self.docid = None

        self.indices = Counter()

        self.corpus.add_doc(self)

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
        self.corpus.add_entity_type(entity_type)
        return ent

    def add_hopper(self, eid=None):
        if eid is None:
            eid = 'h-%d' % self.indices['event']
            self.indices['event'] += 1

        event = Event(eid)
        self.events.append(event)
        return event

    def add_entity_mention(self, ent, offset, length, text=None,
                           eid=None, noun_type=None, entity_type=None,
                           validate=True):
        if entity_type is None:
            entity_type = ent.object_type

        if not eid:
            eid = 'em-%d' % self.indices['entity_mention']
            self.indices['entity_mention'] += 1

        em = EntityMention(eid, offset, length, text, noun_type, entity_type)

        em.fix_spaces(self.doc_text)

        is_valid = True

        if validate:
            if text and self.doc_text:
                is_valid = em.validate(self.doc_text)

        if is_valid:
            ent.add_mention(em)
            return em

    def add_predicate(self, hopper, offset, length, text=None,
                      eid=None, frame_type=None, realis=None, validate=True):
        if eid is None:
            eid = 'evm-%d' % self.indices['event_mention']
            self.indices['event_mention'] += 1

        evm = Predicate(self, eid, offset, length, text, frame_type, realis)
        self.corpus.add_event_type(frame_type)

        is_valid = True

        if validate:
            if text:
                is_valid = evm.validate(self.doc_text)

        if is_valid:
            hopper.add_mention(evm)
            return evm

    def add_arg_type(self, evm_type, arg_type, entity_type):
        self.corpus.add_arg_type(evm_type, arg_type, entity_type)

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

    def to_brat(self):
        ann_text = ''
        relation_text = ''

        t_count = 1
        e_count = 1
        r_count = 1

        ent_map = {}

        def get_brat_span(span):
            brat_spans = [[span.begin, -1]]

            span_text = self.doc_text[span.begin: span.end]

            for i_offset, c in enumerate(span_text):
                if c == '\n':
                    offset = span.begin + i_offset
                    brat_spans[-1][1] = offset
                    brat_spans.append([offset + 1, -1])
            brat_spans[-1][1] = span.end

            return ';'.join(['%d %d' % (s[0], s[1]) for s in brat_spans])

        def get_links(cluster):
            if len(cluster) == 1:
                return []

            chain = sorted(cluster)
            links = []
            for i, j in zip(range(len(chain) - 1), range(1, len(chain))):
                ei = chain[i][1]
                ej = chain[j][1]
                links.append((ei, ej))
            return links

        for ent in self.entities:
            ent_cluster = []
            for e in ent.mentions:
                tid = 'T%d' % t_count
                t_count += 1

                text_bound = [
                    tid,
                    '%s %s' % (e.entity_type, get_brat_span(e.span)),
                    e.text,
                ]
                ann_text += '\t'.join(text_bound) + '\n'

                ent_map[e.aid] = tid

                ent_cluster.append((e.span.begin, tid))

            for e1, e2 in get_links(ent_cluster):
                # relation_text += 'R%d\tEntity_Coref Arg1:%s Arg2:%s\n' % (
                #     r_count, e1, e2)
                r_count += 1

        for event in self.events:
            evm_cluster = []
            for e in event.mentions:
                tid = 'T%d' % t_count
                t_count += 1

                text_bound = [
                    tid,
                    '%s %s' % (e.frame_type, get_brat_span(e.span)),
                    e.text,
                ]

                args = []
                for arg_type, ent in e.arguments.items():
                    if ent in ent_map:
                        ent_tid = ent_map[ent]
                        args.append('%s:%s' % (arg_type, ent_tid))

                eid = 'E%d' % e_count
                e_count += 1

                event_info = [
                    eid,
                    '%s:%s %s' % (e.frame_type, tid, ' '.join(args))
                ]

                ann_text += '\t'.join(text_bound) + '\n'
                ann_text += '\t'.join(event_info) + '\n'

                evm_cluster.append((e.span.begin, eid))

            for e1, e2 in get_links(evm_cluster):
                # relation_text += 'R%d\tEvent_Coref Arg1:%s Arg2:%s\n' % (
                #     r_count, e1, e2)
                r_count += 1

        ann_text += relation_text

        return self.doc_text, ann_text

    def dump(self, indent=None):
        doc = {
            'text': self.doc_text,
            'events': [e.to_json() for e in self.events],
            'entities': [e.to_json() for e in self.entities],
            'fillers': [e.to_json() for e in self.fillers],
            'relations': [e.to_json() for e in self.relations],
        }

        return json.dumps(doc, indent=indent)


class DataLoader:
    def __init__(self, params):
        self.params = params
        self.corpus = Corpus()

    def get_doc(self):
        pass


class RichERE(DataLoader):
    def __init__(self, params):
        super().__init__(params)
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

                doc.add_entity_mention(
                    ent, ent_info['offset'], ent_info['length'], entity_text,
                    ent_info['id'],
                    noun_type=ent_info['noun_type'],
                    entity_type=ent_info.get('type', None),
                )

        for filler in root.find('fillers'):
            filler_info = filler.attrib
            doc.add_filler(
                filler_info['offset'], filler_info['length'], filler.text,
                eid=filler_info['id'], filler_type=filler_info['type'])

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


class FrameNet(DataLoader):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.ns = {
            'icsi': 'http://framenet.icsi.berkeley.edu',
        }

    def parse_full_text(self, full_text_file, doc):
        root = ET.parse(full_text_file).getroot()

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

    def get_doc(self):
        full_text_dir = os.path.join(self.params.fn_path, 'fulltext')

        for full_text_path in glob.glob(full_text_dir + '/*.xml'):
            doc = DEDocument(self.corpus)
            docid = os.path.basename(full_text_path).replace(".xml", '')
            doc.set_id(docid)
            self.parse_full_text(full_text_path, doc)
            yield doc


class ACE(DataLoader):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        logging.info('Loading ACE data.')

    def get_doc(self):
        ace_folder = self.params.in_dir
        text_files = glob.glob(ace_folder + '/English/*/timex2norm/*.sgm')

        text_annos = []

        for f in text_files:
            anno = f.replace('.sgm', '.apf.xml')
            text_annos.append((f, anno))

        for source_file, anno_file in text_annos:
            yield self.parse_ace_data(self.corpus, source_file, anno_file)

    def get_source_text(self, source_in):
        text = re.sub('<[^<]+>', "", source_in.read())
        return text

    def parse_ace_data(self, corpus, source_file, anno_file):
        with open(source_file) as source_in:
            doc = DEDocument(corpus)

            text = self.get_source_text(source_in)

            doc.set_text(text)

            tree = ET.parse(anno_file)
            root = tree.getroot()

            for xml_doc in root.iter('document'):
                docid = xml_doc.attrib['DOCID']
                doc.set_id(docid)

                # Parse entity.
                entity2mention = defaultdict(list)

                for entity in xml_doc.iter('entity'):
                    entity_type = entity.attrib['TYPE']
                    entity_subtype = entity.attrib['SUBTYPE']
                    full_type = entity_type + '_' + entity_subtype

                    ent = doc.add_entity(full_type,
                                         entity.attrib['ID'])

                    for em in entity:
                        for head in em.iter('head'):
                            for charseq in head.iter('charseq'):
                                start = int(charseq.attrib['START'])
                                end = int(charseq.attrib['END'])

                                ent_mention = doc.add_entity_mention(
                                    ent, start, end - start + 1,
                                    charseq.text,
                                    em.attrib['ID'],
                                    entity_type=full_type,
                                    validate=False,
                                )

                                entity2mention[entity.attrib['ID']].append(
                                    ent_mention
                                )

                # Parse event.
                for event_node in xml_doc.iter('event'):
                    event_type = event_node.attrib['TYPE']
                    event_subtype = event_node.attrib['SUBTYPE']

                    hopper = doc.add_hopper(event_node.attrib['ID'])

                    event_mentions = []

                    for evm_node in event_node:
                        for anchor in evm_node.iter('anchor'):
                            for charseq in anchor.iter('charseq'):
                                start = int(charseq.attrib['START'])
                                end = int(charseq.attrib['END'])

                                evm = doc.add_predicate(
                                    hopper, start, end - start + 1,
                                    charseq.text, eid=evm_node.attrib['ID'],
                                    frame_type=event_type + '_' + event_subtype,
                                    validate=False,
                                )

                                event_mentions.append(evm)

                    for em_arg in event_node.iter('event_argument'):
                        role = em_arg.attrib['ROLE']
                        arg_id = em_arg.attrib['REFID']

                        entity_mentions = entity2mention[arg_id]

                        if len(entity_mentions) > 0:
                            closest_ent, closest_evm, _ = find_close_mention(
                                event_mentions, entity_mentions)
                            closest_evm.add_arg(role, closest_ent.aid)

                return doc


class Conll(DataLoader):
    def __init__(self, params):
        super().__init__(params)
        self.params = params

    def parse_conll_data(self, corpus, conll_in):
        text = ''
        offset = 0

        arg_text = []
        sent_predicates = []
        sent_args = defaultdict(list)
        doc = DEDocument(corpus)

        props = []

        for line in conll_in:
            parts = line.strip().split()
            if len(parts) < 8:
                text += '\n'
                offset += 1

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

            if pred is not None:
                for role, arg_start, arg_end, arg_text in args:
                    filler = doc.add_filler(
                        arg_start, arg_end - arg_start, arg_text)

                    pred.add_arg(role, filler.aid)

        return doc

    def get_doc(self):
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

                    docid = f.replace('gold_conll', '')

                    with open(full_path) as conll_in:
                        doc = self.parse_conll_data(self.corpus, conll_in)
                        doc.set_id(docid)
                        yield doc


class NomBank(DataLoader):
    """
    # TODO
    1. Figure out height
    2. Check if there is missing explicit
    3. Check dupliciate labels
    4. Filter incorporated args: arg which is the the predicate itself
    5. Remove implicit args following the predicate
    """

    def __init__(self, params):
        super().__init__(params)

        logging.info('Loading WSJ Treebank.')

        self.wsj_treebank = BracketParseCorpusReader(
            root=params.wsj_path,
            fileids=params.wsj_file_pattern,
            tagset='wsj',
            encoding='ascii'
        )

        logging.info(
            'Found {} treebank files.'.format(len(self.wsj_treebank.fileids()))
        )

        self.nombank = NombankCorpusReader(
            root=FileSystemPathPointer(params.nombank_path),
            nomfile=params.nomfile,
            framefiles=params.frame_file_pattern,
            nounsfile=params.nombank_nouns_file,
            parse_fileid_xform=lambda s: s[4:],
            parse_corpus=self.wsj_treebank
        )

        logging.info("Loading G&C annotations")
        self.gc_annos = self.load_gc_annotations()

    class NomElement:
        def __init__(self, article_id, sent_id):
            self.article_id = article_id
            self.sent_id = sent_id
            self.pointers = []

        @staticmethod
        def from_text(pointer_text):
            parts = pointer_text.split(':')
            if len(parts) != 4:
                raise ValueError("Invalid pointer text.")

            node = NomBank.NomElement(parts[0], int(parts[1]))
            node.add_pointer(NombankTreePointer(int(parts[2]), int(parts[3])))

        def add_pointer(self, tree_pointer):
            self.pointers.append(tree_pointer)

        def __str__(self):
            return 'Node-%s-%s:%s' % (
                self.article_id, self.sent_id, self.pointers.__repr__())

        def __hash__(self):
            return hash(
                (self.article_id, self.sent_id, self.pointers)
            )

        def __eq__(self, other):
            return other and other.__str__() == self.__str__()

        __repr__ = __str__

    def get_wsj_data(self, fileid):
        sents = self.wsj_treebank.sents(fileids=fileid)
        parsed_sents = self.wsj_treebank.parsed_sents(fileids=fileid)
        return sents, parsed_sents

    def merge_split(self, pointers):
        all_pointers = []
        split_pointers = []

        for pointer, is_split in pointers:
            if is_split:
                split_pointers.append(pointer)
            else:
                all_pointers.append([pointer])

        all_pointers.append(split_pointers)

        return all_pointers

    def load_gc_annotations(self):
        tree = ET.parse(self.params.implicit_path)
        root = tree.getroot()

        gc_annotations = {}

        for annotations in root:
            predicate = NomBank.NomElement.from_text(
                annotations.attrib['for_node']
            )

            article_id = predicate.article_id
            if article_id not in gc_annotations:
                gc_annotations[article_id] = {}

            arg_annos = defaultdict(list)

            for annotation in annotations:
                arg_type = annotation.attrib['value']
                arg_node_pos = annotation.attrib['node']

                (arg_article_id, arg_sent_id, arg_terminal_id,
                 arg_height) = arg_node_pos.split(':')

                is_split = False
                is_explicit = False

                for attribute in annotation[0]:
                    if attribute.text == 'Split':
                        is_split = True
                    elif attribute.text == 'Explicit':
                        is_explicit = True

                if not is_explicit:
                    p = NombankTreePointer(arg_terminal_id, arg_height)
                    arg_annos[(arg_sent_id, arg_type)].append((p, is_split))

            all_args = []

            for (arg_sent_id, arg_type), l_pointers in arg_annos:
                arg_element = NomBank.NomElement(article_id, arg_sent_id)
                for p in self.merge_split(l_pointers):
                    arg_element.add_pointer(p)
                all_args.append(arg_element)

            gc_annotations[article_id][predicate] = all_args

        return gc_annotations

    def add_nombank_arg(self, doc, sents, parsed_sents, predicate, nodes):
        print("Adding predicate")
        print(predicate)

        print("Adding arg nodes")
        print(nodes)

    def get_normal_pointers(self, tree_pointer):
        pointers = []
        if isinstance(tree_pointer, NombankSplitTreePointer) or isinstance(
                tree_pointer, NombankChainTreePointer):
            for p in tree_pointer.pieces:
                pointers.extend(self.get_normal_pointers(p))
        else:
            pointers.append(tree_pointer)
        return pointers

    def build_tree(self, tree, tree_pointer):
        pointers = self.get_normal_pointers(tree_pointer)

        all_word_idx = []
        all_word_surface = []

        for pointer in pointers:
            treepos = pointer.treepos(tree)

            idx_list = []
            word_list = []
            for idx in range(len(tree.leaves())):
                if tree.leaf_treeposition(idx)[:len(treepos)] == treepos:
                    idx_list.append(idx)
                    word_list.append(tree.leaves()[idx])

            all_word_idx.append(idx_list)
            all_word_surface.append(word_list)

        return all_word_idx, all_word_surface

    def get_all_annotations(self, doc, nb_instances, fileid):
        sents, parsed_sents = self.get_wsj_data(fileid)

        for nb_instance in nb_instances:
            predicate_node = NomBank.NomElement(
                fileid, nb_instance.sentnum, nb_instance.wordnum
            )

            print("Predicate %s: %s is found in sentence %d." % (
                predicate_node,
                sents[nb_instance.sentnum][predicate_node.wordnum],
                nb_instance.sentnum,
            ))

            tree = parsed_sents[nb_instance.sentnum]

            for argloc, argid in nb_instance.arguments:
                print("Building tree for an argument")
                all_word_idx, all_word_surface = self.build_tree(tree, argloc)

                if len(all_word_idx) > 1:
                    print(all_word_surface)
                    print(all_word_idx)
                    print("Multiple spans")

                    input('wait')

                # argloc_pointers = []
                # if isinstance(argloc, NombankChainTreePointer):
                #     for argloci in argloc.pieces:
                #         argloc_pointers.append(argloci)
                # else:
                #     argloc_pointers.append(argloc)

                # arg_nodes = []
                # for p in argloc_pointers:
                #     arg_nodes.append(
                #         NomBank.TreeNode(
                #             fileid, nb_instance.sentnum, p.wordnum, p.height
                #         )
                #     )

                # self.add_nombank_arg(
                #     doc, sents, parsed_sents, predicate_node, arg_nodes)

        if doc.docid in self.gc_annos:
            gc_data = self.gc_annos[fileid]
            for predicate_node, arg_nodes in gc_data:
                print("Adding implicit")
                self.add_nombank_arg(doc, sents, parsed_sents, predicate_node,
                                     arg_nodes)
                input("Implicit added.")

    def get_wsj_text(self, fileid):
        return ' '.join(self.wsj_treebank.words(fileid))

    def get_doc(self):
        last_file = None
        doc_instances = []

        for nb_instance in self.nombank.instances():
            if last_file and not last_file == nb_instance.fileid:
                doc = DEDocument(self.corpus)
                doc.set_id(nb_instance.fileid.split('/')[1])
                doc.set_text(self.get_wsj_text(nb_instance.fileid))

                self.get_all_annotations(doc, doc_instances, last_file)

                doc_instances.clear()
                yield doc

            doc_instances.append(nb_instance)

            last_file = nb_instance.fileid

        if len(doc_instances) > 0:
            doc = DEDocument(self.corpus)
            doc.set_id(last_file.split('/')[1])
            self.get_all_annotations(doc, doc_instances, last_file)
            doc.set_text(self.get_wsj_text(last_file))

            yield doc


def main(data_format, args):
    from event.util import basic_console_log
    from event.util import load_file_config, load_config_with_cmd

    class DataConf(Configurable):
        out_dir = Unicode(help='Output directory').tag(config=True)
        text_dir = Unicode(help='Text output directory').tag(config=True)
        brat_dir = Unicode(help='Brat visualization directory').tag(config=True)

    class EreConf(DataConf):
        source = Unicode(help='Plain source input directory').tag(config=True)
        ere = Unicode(help='ERE input data').tag(config=True)
        src_ext = Unicode(help='Source file extension',
                          default_value='.xml').tag(config=True)
        ere_ext = Unicode(help='Ere file extension',
                          default_value='.rich_ere.xml').tag(config=True)
        ere_split = Bool(help='Whether split ere based on the file names').tag(
            config=True)
        ignore_quote = Bool(help='model name', default_value=False).tag(
            config=True)

    class FrameNetConf(DataConf):
        fn_path = Unicode(help='FrameNet dataset path.').tag(config=True)

    class ConllConf(DataConf):
        in_dir = Unicode(help='Conll file input directory').tag(config=True)

    class AceConf(DataConf):
        in_dir = Unicode(help='Conll file input directory').tag(config=True)
        out_dir = Unicode(help='Output directory').tag(config=True)
        text_dir = Unicode(help='Raw Text Output directory').tag(config=True)
        brat_dir = Unicode(help='Brat Output directory').tag(config=True)

    class NomBankConfig(DataConf):
        nombank_path = Unicode(help='Nombank corpus.').tag(config=True)
        nomfile = Unicode(help='Nombank file.').tag(config=True)
        frame_file_pattern = Unicode(help='Frame file pattern.').tag(
            config=True)
        nombank_nouns_file = Unicode(help='Nomank nous.').tag(config=True)

        # PennTree Bank config.
        wsj_path = Unicode(help='PennTree Bank path.').tag(config=True)
        wsj_file_pattern = Unicode(help='File pattern to read PTD data').tag(
            config=True)

        implicit_path = Unicode(help='Implicit annotation xml path.').tag(
            config=True)

    basic_console_log()

    if os.path.exists(args[0]):
        config = load_file_config(args[0])
    else:
        config = load_config_with_cmd(args)

    if data_format == 'rich_ere':
        basic_para = EreConf(config=config)
        parser = RichERE(basic_para)
    elif data_format == 'framenet':
        basic_para = FrameNetConf(config=config)
        parser = FrameNet(basic_para)
    elif data_format == 'conll':
        basic_para = ConllConf(config=config)
        parser = Conll(basic_para)
    elif data_format == 'ace':
        basic_para = AceConf(config=config)
        parser = ACE(basic_para)
    elif data_format == 'nombank':
        basic_para = NomBankConfig(config=config)
        parser = NomBank(basic_para)
    else:
        parser = None

    if parser:
        if not os.path.exists(basic_para.out_dir):
            os.makedirs(basic_para.out_dir)

        if not os.path.exists(basic_para.text_dir):
            os.makedirs(basic_para.text_dir)

        brat_data_path = os.path.join(basic_para.brat_dir, 'data')
        if not os.path.exists(brat_data_path):
            os.makedirs(brat_data_path)

        for doc in parser.get_doc():
            with open(os.path.join(
                    basic_para.out_dir, doc.docid + '.json'), 'w') as out:
                out.write(doc.dump(indent=2))

            with open(os.path.join(basic_para.text_dir,
                                   doc.docid + '.txt'), 'w') as out:
                out.write(doc.doc_text)

            source_text, ann_text = doc.to_brat()
            with open(os.path.join(basic_para.brat_dir, 'data',
                                   doc.docid + '.ann'), 'w') as out:
                out.write(ann_text)

            with open(os.path.join(basic_para.brat_dir, 'data',
                                   doc.docid + '.txt'), 'w') as out:
                out.write(source_text)

        with open(os.path.join(basic_para.brat_dir, 'annotation.conf'),
                  'w') as out:
            out.write(parser.corpus.get_brat_config())


if __name__ == '__main__':
    import sys

    main(sys.argv[1], sys.argv[2:])
