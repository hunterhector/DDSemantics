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

from event.util import ensure_dir


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


def get_text_from_span(doc_text, spans):
    if isinstance(spans, Span):
        spans = [spans]
    return ' '.join([doc_text[s.begin: s.end] for s in spans])


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

    def get_mentions(self):
        return self.mentions

    def add_mention(self, mention):
        assert isinstance(mention, Annotation)
        self.mentions.append(mention)


class Annotation:
    def __init__(self, aid, anno_type, text, spans):
        self.aid = aid
        self.text = text.replace('\n', ' ').strip()
        self.anno_type = anno_type

        if isinstance(spans, Span):
            self.spans = [spans]
        else:
            self.spans = [s for s in spans]

    def get_unique_key(self):
        """
        Use spans as a key
        :return:
        """
        return hash(tuple(self.spans))

    def validate(self, doc_text):
        source_anno = get_text_from_span(doc_text, self.spans)
        if not source_anno == self.text:
            logging.warning(
                "Non-matching mention at [%s] between doc [%s] and "
                "provided [%s]" % (self.spans, source_anno, self.text))
            return False
        return True

    def fix_spaces(self, source_text):
        all_spans = []

        for s in self.spans:
            source_anno = source_text[s.begin: s.end]
            leadings = len(source_anno) - len(source_anno.lstrip())
            trailings = len(source_anno) - len(source_anno.rstrip())

            new_span = Span(s.begin + leadings, s.end - trailings)

            all_spans.append(new_span)

            if leadings > 0 or trailings > 0:
                logging.warning("Stripping space from text")

        self.spans = all_spans

    def merge(self, anno_b):
        """
        Merge with another annotation
        :param anno_b:
        :return:
        """
        pass

    def to_json(self):
        data = {
            'id': self.aid,
            'annotation': self.anno_type,
        }

        if self.text:
            data['text'] = self.text

        if self.spans:
            data['spans'] = {}
            for span in self.spans:
                data['spans']['begin'] = span.begin
                data['spans']['end'] = span.end

        return data


class Argument(Top):
    def __init__(self, aid, predicate, arg, arg_type):
        super().__init__(aid, 'Argument')
        self.meta = {}
        self.predicate = predicate
        self.arg = arg
        self.arg_type = arg_type

    def add_meta(self, name, value):
        self.meta[name] = value

    def to_json(self):
        return {
            'arg': self.arg,
            'role': self.arg_type,
            'meta': self.meta,
        }


class Predicate(Annotation):
    def __init__(self, doc, eid, spans, text, frame_type=None, realis=None):
        super().__init__(eid, 'Predicate', text, spans)
        self.arguments = defaultdict(list)
        self.frame_type = frame_type
        self.realis = realis
        self.doc = doc

    def get_unique_key(self):
        return super().get_unique_key(), self.frame_type

    def add_arg(self, argument):
        assert isinstance(argument, Argument)
        self.arguments[argument.arg_type].append(argument)
        self.doc.add_arg_type(self.frame_type, argument.arg_type, 'Entity')

    def merge(self, anno_b):
        assert isinstance(anno_b, Predicate)
        for key, value in anno_b.arguments.items():
            self.arguments[key].append(value)

    def to_json(self):
        data = super().to_json()

        if self.frame_type:
            data['type'] = self.frame_type

        if self.realis:
            data['realis'] = self.realis

        data['arguments'] = {}
        for key, l_arg in self.arguments.items():
            data['arguments'][key] = [arg.to_json() for arg in l_arg]

        return data


class EntityMention(Annotation):
    def __init__(self, eid, spans, text, noun_type=None,
                 entity_type=None):
        super().__init__(eid, 'EntityMention', text, spans)
        self.entity_type = entity_type
        self.noun_type = noun_type

    def get_unique_key(self):
        return 'entity', super().get_unique_key(), self.entity_type

    def merge(self, anno_b):
        assert isinstance(anno_b, EntityMention)

    def to_json(self):
        data = super().to_json()

        if self.entity_type:
            data['type'] = self.entity_type

        if self.noun_type:
            data['noun_type'] = self.noun_type

        return data


class Filler(Annotation):
    def __init__(self, eid, spans, text, filler_type=None):
        super().__init__(eid, 'Filler', text, spans)
        self.entity_type = filler_type

    def get_unique_key(self):
        return super().get_unique_key(), self.entity_type

    def merge(self, anno_b):
        assert isinstance(anno_b, Filler)

    def to_json(self):
        data = super().to_json()
        if self.entity_type:
            data['type'] = self.entity_type
        return data


class RelationMention(Annotation):
    def __init__(self, rid, spans, text, realis=None, relation_type=None):
        super().__init__(rid, 'RelationMention', text, spans)
        self.args = {}
        self.relation_type = relation_type
        self.realis = realis

    def merge(self, anno_b):
        raise ValueError("Do not support duplicate relation mention.")

    def add_arg(self, role, entity_id):
        self.args[role] = entity_id

    def to_json(self):
        data = super().to_json()
        more = {'args': self.args, }

        if self.realis:
            data['realis'] = self.realis

        if self.relation_type:
            data['relation_type'] = self.relation_type

        data.update(more)
        return data


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
        self.relations = []

        self.span_mentions = {}

        self.fillers = []

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

    def add_entity(self, entity_type='Entity', eid=None):
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

    def __add_mention_with_check(self, mention, cluster, validate=True):
        key = mention.get_unique_key()
        if key in self.span_mentions:
            self.span_mentions[key].merge(mention)
            return self.span_mentions[key]
        else:
            mention.fix_spaces(self.doc_text)
            is_valid = True

            if validate:
                is_valid = mention.validate(self.doc_text)

            if is_valid:
                self.span_mentions[key] = mention
                if cluster:
                    cluster.add_mention(mention)
                return mention

    def add_argument_mention(self, predicate, filler, arg_type, aid=None):
        if not aid:
            aid = 'arg-%d' % self.indices['argument']
            self.indices['argument'] += 1
        arg = Argument(aid, predicate, filler, arg_type)
        predicate.add_arg(arg)
        return arg

    def add_entity_mention(self, ent, spans, text=None,
                           eid=None, noun_type=None, entity_type=None,
                           validate=True):
        if entity_type is None:
            entity_type = ent.object_type

        if not eid:
            eid = 'em-%d' % self.indices['entity_mention']
            self.indices['entity_mention'] += 1

        if not text:
            text = get_text_from_span(self.doc_text, spans)

        em = EntityMention(eid, spans, text, noun_type, entity_type)
        self.corpus.add_entity_type(entity_type)
        return self.__add_mention_with_check(em, ent, validate)

    def add_predicate(self, hopper, spans, text=None,
                      eid=None, frame_type='Event', realis=None, validate=True):
        if eid is None:
            eid = 'evm-%d' % self.indices['event_mention']
            self.indices['event_mention'] += 1

        if not text:
            text = get_text_from_span(self.doc_text, spans)

        evm = Predicate(self, eid, spans, text, frame_type, realis)
        self.corpus.add_event_type(frame_type)
        return self.__add_mention_with_check(evm, hopper, validate)

    def add_filler(self, spans, text, eid=None, filler_type=None):
        if eid is None:
            eid = 'em-%d' % self.indices['entity_mention']
            self.indices['entity_mention'] += 1

        if not text:
            text = get_text_from_span(self.doc_text, spans)

        filler = Filler(eid, spans, text, filler_type)
        self.corpus.add_entity_type(filler_type)
        return self.__add_mention_with_check(filler, None, True)

    def add_arg_type(self, evm_type, arg_type, entity_type):
        self.corpus.add_arg_type(evm_type, arg_type, entity_type)

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

        def get_brat_span(spans):
            brat_spans = []

            for span in spans:
                brat_spans.append([span.begin, -1])

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
            for em in ent.get_mentions():
                tid = 'T%d' % t_count
                t_count += 1

                text_bound = [
                    tid,
                    '%s %s' % (em.entity_type, get_brat_span(em.spans)),
                    em.text,
                ]
                ann_text += '\t'.join(text_bound) + '\n'

                ent_map[em.aid] = tid

                ent_cluster.append((em.spans[0].begin, tid))

            for e1, e2 in get_links(ent_cluster):
                # relation_text += 'R%d\tEntity_Coref Arg1:%s Arg2:%s\n' % (
                #     r_count, e1, e2)
                r_count += 1

        for event in self.events:
            evm_cluster = []
            for em in event.get_mentions():
                tid = 'T%d' % t_count
                t_count += 1

                text_bound = [
                    tid,
                    '%s %s' % (em.frame_type, get_brat_span(em.spans)),
                    em.text,
                ]

                args = []
                for arg_type, l_arg in em.arguments.items():
                    for arg in l_arg:
                        ent = arg.arg
                        if ent in ent_map:
                            ent_tid = ent_map[ent]
                            args.append('%s:%s' % (arg_type, ent_tid))

                eid = 'E%d' % e_count
                e_count += 1

                event_info = [
                    eid,
                    '%s:%s %s' % (em.frame_type, tid, ' '.join(args))
                ]

                ann_text += '\t'.join(text_bound) + '\n'
                ann_text += '\t'.join(event_info) + '\n'

                evm_cluster.append((em.spans[0].begin, eid))

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
                ev, Span(target_start, target_end), text=text,
                frame_type=frame_name
            )

            for start, end, fe_text, role in fes:
                filler = doc.add_filler(Span(start, end), fe_text)
                doc.add_argument_mention(evm, filler.aid, role)

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

                                entity_span = Span(start, end + 1)

                                ent_mention = doc.add_entity_mention(
                                    ent, entity_span,
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
                                    hopper, Span(start, end + 1),
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
                            doc.add_argument_mention(
                                closest_evm, closest_ent.aid, role)

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
                hopper, Span(p_start, p_end), p_token)

            if pred is not None:
                for role, arg_start, arg_end, arg_text in args:
                    filler = doc.add_filler(Span(arg_start, arg_end), arg_text)
                    doc.add_argument_mention(pred, filler.aid, role)

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
    1. Figure out height DONE
    2. Check if there is missing explicit
    3. Check duplicate labels
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
        def __init__(self, article_id, sent_num, tree_pointer):
            self.article_id = article_id
            self.sent_num = int(sent_num)
            self.pointer = tree_pointer

        @staticmethod
        def from_text(pointer_text):
            parts = pointer_text.split(':')
            if len(parts) != 4:
                raise ValueError("Invalid pointer text.")

            read_id = parts[0]
            full_id = read_id.split('_')[1][:2] + '/' + read_id + '.mrg'

            return NomBank.NomElement(
                full_id, int(parts[1]),
                NombankTreePointer(int(parts[2]), int(parts[3]))
            )

        def __str__(self):
            return 'Node-%s-%s:%s' % (
                self.article_id, self.sent_num, self.pointer.__repr__())

        def __hash__(self):
            return hash(
                (self.article_id, self.sent_num, self.pointer.__repr__())
            )

        def __eq__(self, other):
            return other and other.__str__() == self.__str__()

        __repr__ = __str__

    def get_wsj_data(self, fileid):
        sents = self.wsj_treebank.sents(fileids=fileid)
        parsed_sents = self.wsj_treebank.parsed_sents(fileids=fileid)
        return sents, parsed_sents

    def load_gc_annotations(self):
        tree = ET.parse(self.params.implicit_path)
        root = tree.getroot()

        gc_annotations = {}

        def merge_split_pointers(pointers):
            all_pointers = []
            split_pointers = []

            for pointer, is_split in pointers:
                if is_split:
                    split_pointers.append(pointer)
                else:
                    all_pointers.append(pointer)

            if len(split_pointers) > 0:
                all_pointers.append(NombankChainTreePointer(split_pointers))

            return all_pointers

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
                    p = NombankTreePointer(int(arg_terminal_id),
                                           int(arg_height))
                    arg_annos[(arg_sent_id, arg_type)].append((p, is_split))

            all_args = defaultdict(list)

            for (arg_sent_id, arg_type), l_pointers in arg_annos.items():
                for p in merge_split_pointers(l_pointers):
                    arg_element = NomBank.NomElement(article_id, arg_sent_id, p)
                    all_args[arg_type].append(arg_element)

            gc_annotations[article_id][predicate] = all_args

        return gc_annotations

    def add_nombank_arg(self, doc, wsj_spans, fileid, predicate,
                        arg_type, argument, implicit=False):
        def get_span(sent_num, indice_groups):
            spans = []
            for indices in indice_groups:
                start = -1
                end = -1
                for index in indices:
                    s = wsj_spans[sent_num][index]
                    if s:
                        if start < 0:
                            start = s[0]
                        end = s[1]

                if start >= 0 and end >= 0:
                    spans.append((start, end))
            return spans

        sents, parsed_sents = self.get_wsj_data(fileid)
        p_tree = parsed_sents[predicate.sent_num]
        a_tree = parsed_sents[argument.sent_num]

        p_word_idx, p_word_surface = self.build_tree(p_tree, predicate.pointer)
        a_word_idx, a_word_surface = self.build_tree(a_tree, argument.pointer)

        predicate_span = get_span(predicate.sent_num, p_word_idx)
        argument_span = get_span(argument.sent_num, a_word_idx)

        if len(predicate_span) == 0:
            logging.warn("Zero length predicate found")
            return

        if len(argument_span) == 0:
            # Some arguments are empty nodes, they will be ignored.
            return

        p_begin = predicate_span[0][0]
        p_end = predicate_span[-1][1]

        h = doc.add_hopper()
        p = doc.add_predicate(h, Span(p_begin, p_end))

        a_begin = argument_span[0][0]
        a_end = argument_span[-1][1]
        e = doc.add_entity()
        arg_em = doc.add_entity_mention(e, Span(a_begin, a_end))

        if p and arg_em:
            if implicit:
                arg_type = 'i_' + arg_type

            arg_mention = doc.add_argument_mention(p, arg_em.aid, arg_type)

            if implicit:
                arg_mention.add_meta('implicit', True)

                if argument.sent_num > predicate.sent_num:
                    arg_mention.add_meta('delayed', True)

            if predicate.pointer == argument.pointer:
                arg_mention.add_meta('incorporated', True)

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

    def add_all_annotations(self, doc, wsj_spans, nb_instances, fileid):
        for nb_instance in nb_instances:
            predicate_node = NomBank.NomElement(
                fileid, nb_instance.sentnum, nb_instance.predicate
            )

            for argloc, argid in nb_instance.arguments:
                arg_node = NomBank.NomElement(
                    fileid, nb_instance.sentnum, argloc
                )
                self.add_nombank_arg(doc, wsj_spans, fileid,
                                     predicate_node, argid, arg_node)

            if fileid in self.gc_annos:
                if predicate_node in self.gc_annos[fileid]:
                    gc_args = self.gc_annos[fileid][predicate_node]

                    for arg_type, arg_nodes in gc_args.items():
                        for arg_node in arg_nodes:
                            self.add_nombank_arg(
                                doc, wsj_spans, fileid,
                                predicate_node, arg_type, arg_node, True
                            )

    def set_wsj_text(self, doc, fileid):
        text = ''
        w_start = 0

        spans = []
        for tagged_sent in self.wsj_treebank.tagged_sents(fileid):
            word_spans = []

            for word, tag in tagged_sent:
                if not tag == '-NONE-':
                    text += word + ' '
                    word_spans.append((w_start, w_start + len(word)))
                    w_start += len(word) + 1
                else:
                    # Ignoring these words.
                    word_spans.append(None)

            text += '\n'
            w_start += 1

            spans.append(word_spans)

        doc.set_text(text)

        return spans

    def get_doc(self):
        last_file = None
        doc_instances = []

        for nb_instance in self.nombank.instances():
            if self.params.gc_only and nb_instance.fileid not in self.gc_annos:
                continue

            if last_file and not last_file == nb_instance.fileid:
                doc = DEDocument(self.corpus)
                doc.set_id(last_file.split('/')[1])
                wsj_spans = self.set_wsj_text(doc, last_file)

                self.add_all_annotations(doc, wsj_spans, doc_instances,
                                         last_file)
                doc_instances.clear()
                yield doc

            doc_instances.append(nb_instance)

            last_file = nb_instance.fileid

        if len(doc_instances) > 0:
            doc = DEDocument(self.corpus)
            doc.set_id(last_file.split('/')[1])
            wsj_spans = self.set_wsj_text(doc, last_file)
            self.add_all_annotations(doc, wsj_spans, doc_instances, last_file)

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
        gc_only = Bool(help='Only use GC arguments.').tag(config=True)

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
        basic_para = None
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
            out_path = os.path.join(basic_para.out_dir, doc.docid + '.json')
            ensure_dir(out_path)
            with open(out_path, 'w') as out:
                out.write(doc.dump(indent=2))

            out_path = os.path.join(basic_para.text_dir, doc.docid + '.txt')
            ensure_dir(out_path)
            with open(out_path, 'w') as out:
                out.write(doc.doc_text)

            source_text, ann_text = doc.to_brat()
            out_path = os.path.join(basic_para.brat_dir, 'data',
                                    doc.docid + '.ann')
            ensure_dir(out_path)
            with open(out_path, 'w') as out:
                out.write(ann_text)

            out_path = os.path.join(basic_para.brat_dir, 'data',
                                    doc.docid + '.txt')
            ensure_dir(out_path)
            with open(out_path, 'w') as out:
                out.write(source_text)

        out_path = os.path.join(basic_para.brat_dir, 'annotation.conf')
        with open(out_path, 'w') as out:
            out.write(parser.corpus.get_brat_config())


if __name__ == '__main__':
    import sys

    main(sys.argv[1], sys.argv[2:])
