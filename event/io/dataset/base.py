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

        if self.begin >= self.end:
            raise ValueError("Invalid span [%d:%d]" % (self.begin, self.end))

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

    __repr__ = __str__


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


