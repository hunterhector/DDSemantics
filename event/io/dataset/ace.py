import glob
import logging
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

from event.io.dataset.base import (
    Span,
    DataLoader,
    DEDocument,
    find_close_mention,
)


class ACE(DataLoader):
    def __init__(self, params, with_doc=False):
        super().__init__(params, with_doc)
        self.params = params
        logging.info('Loading ACE data.')

    def get_doc(self):
        super().get_doc()

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
