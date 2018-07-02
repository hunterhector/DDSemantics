import os
import xml.etree.ElementTree as ElementTree
import logging


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
    def __init__(self, begin, length):
        self.span = Span(int(begin), int(begin) + int(length))

    def validate(self, source_text, anno_text):
        if not source_text[self.span.begin: self.span.end] == anno_text:
            logging.warning("Invalid mention [%s] at [%s]." % (
                self.span, anno_text))
            return False
        return True


class EventMention(Annotation):
    def __init__(self, begin, length, event_type=None, realis=None):
        super().__init__(begin, length)
        self.arguments = {}
        self.event_type = event_type
        self.realis = realis

    def add_arg(self, arg_type, ent):
        self.arguments[arg_type] = ent


class EntityMention(Annotation):
    def __init__(self, begin, length, noun_type=None, entity_type=None):
        super().__init__(begin, length)
        self.entity_type = entity_type
        self.noun_type = noun_type


class Filler(Annotation):
    def __init__(self, begin, length, filler_type=None):
        super().__init__(begin, length)
        self.filler_type = filler_type


class DEDocument:
    def __init__(self, text, docid=None):
        self.docid = docid
        self.event_mentions = {}
        self.entity_mentions = {}
        self.fillers = {}
        self.text = text
        self.doc_type = None

    def set_id(self, docid):
        self.docid = docid

    def set_doc_type(self, doc_type):
        self.doc_type = doc_type

    def set_text(self, text):
        self.text = text

    def add_entity_mention(self, eid, offset, length, noun_type=None,
                           entity_type=None, text=None):
        ent = EntityMention(offset, length, noun_type, entity_type)
        if text:
            if ent.validate(self.text, text):
                self.entity_mentions[eid] = ent

    def add_event_mention(self, eid, offset, length, event_type=None,
                          realis=None, text=None):
        evm = EventMention(offset, length, event_type, realis)
        self.event_mentions[eid] = evm
        if text:
            if evm.validate(self.text, text):
                self.event_mentions[eid] = evm

    def add_filler(self, eid, offset, length, filler_type=None, text=None):
        filler = Filler(offset, length, filler_type)
        if text:
            if filler.validate(self.text, text):
                self.fillers[eid] = filler


class DEDataset:
    def __init__(self):
        self.documents = []

    def add_event_mention(self):
        pass

    def add_entity_mention(self):
        pass


def parse_ere(ere_file, doc):
    root = ElementTree.parse(ere_file).getroot()

    doc_info = root.attrib

    doc.set_id = doc_info['doc_id']
    doc.set_doc_type = doc_info['source_type']

    for entity in root.find('entities'):
        entity_ids = []
        for entity_mention in entity.findall('entity_mention'):
            ent_info = entity_mention.attrib
            entity_ids.append(ent_info['id'])

            entity_text = entity_mention.find('mention_text').text

            doc.add_entity_mention(
                ent_info['id'], ent_info['offset'], ent_info['length'],
                noun_type=ent_info['noun_type'], text=entity_text
            )

    for event in root.find('hoppers'):
        evm_ids = []
        for event_mention in event.findall('event_mention'):
            evm_info = event_mention.attrib
            evm_ids.append(evm_info['id'])

            trigger = event_mention.find('trigger')
            trigger_text = trigger.text
            offset = trigger.attrib['offset']
            length = trigger.attrib['length']

            doc.add_event_mention(
                evm_info['id'], offset, length,
                event_type=evm_info['type'] + '_' + evm_info['subtype'],
                realis=evm_info['realis'],
                text=trigger_text
            )

    for filler in root.find('fillers'):
        filler_info = filler.attrib
        doc.add_filler(filler_info['id'], filler_info['offset'],
                       filler_info['length'], filler_info['type'], filler.text)

    for relation in root.find('relations'):
        pass


def read_rich_ere(source_path, ere_path):
    print(source_path, ere_path)
    with open(source_path) as source, open(ere_path) as ere:
        text = source.read()

        doc = DEDocument(text)

        parse_ere(ere, doc)

        input('wait')


def read_rich_ere_collection(source_dir, ere_dir, ere_ext='.rich_ere.xml'):
    sources = {}
    eres = {}

    for fn in os.listdir(ere_dir):
        basename = fn.replace(ere_ext, '')
        ere_fn = os.path.join(ere_dir, fn)
        eres[basename] = ere_fn

    for fn in os.listdir(source_dir):
        for ere_base in eres.keys():
            if fn.startswith(ere_base):
                sources[ere_base] = os.path.join(source_dir, fn)

    for basename, source in sources.items():
        ere = eres[basename]
        read_rich_ere(source, ere)


if __name__ == '__main__':
    source_dir = '/home/hector/workspace/datasets/ERE/LDC2015E29_DEFT_' \
                 'Rich_ERE_English_Training_Annotation_V2/data/source/cmptxt'
    ere_dir = '/home/hector/workspace/datasets/ERE/LDC2015E29_DEFT_' \
              'Rich_ERE_English_Training_Annotation_V2/data/ere/cmptxt'
    read_rich_ere_collection(source_dir, ere_dir)
