"""
Read NeGra format.
"""

import xml.etree.ElementTree as ET

from event.io.dataset.base import (
    Span,
    DataLoader,
    DEDocument,
    Corpus,
)

from collections import Counter, defaultdict
import logging
import os
from operator import itemgetter

# Some small differences between FrameNet v1.4 and v1.5
frame_changes = {
    'Observable_bodyparts': 'Observable_body_parts'
}


class TreeNode:
    def __init__(self, node_id, is_leaf=False, is_root=False):
        self.node_id = node_id
        self.children = []
        self.attributes = {}
        self.is_root = is_root
        self.is_leaf = is_leaf

        self.leaves = None

    def add_child(self, node):
        self.children.append(node)

    def add_attribute(self, attribute_name, attribute_value):
        self.attributes[attribute_name] = attribute_value

    def get_leaves(self):
        if not self.leaves:
            leaves = []
            for child in self.children:
                if child.is_leaf:
                    leaves.append(child)
                else:
                    more_child = child.get_leaves()
                    leaves.extend(more_child)

            self.leaves = sorted(leaves,
                                 key=lambda x: int(x.node_id.split('_')[1]))

        return self.leaves


class NeGraXML(DataLoader):
    def __init__(self, params, corpus, with_doc=False):
        super().__init__(params, corpus)

        self.xml_paths = params.data_files
        self.stat_dir = params.stat_out

        self.offset = 0
        self.text = ''

        self.id2span = {}

        self.stats = Counter()
        self.imp_frames = Counter()

    def get_doc(self):
        for xml_path in self.xml_paths:
            logging.info("Parsing: " + xml_path)

            root = ET.parse(xml_path).getroot()
            self.corpus.set_corpus_name(root.attrib['corpusname'])

            doc = DEDocument(self.corpus)
            doc.set_id(self.corpus.corpus_name)

            body = root.find("body")

            for sent in body:
                c_graph_node = sent.find("graph")

                c_parse = self.read_constituent_parse(c_graph_node)
                self.build_next_sent(doc, c_parse)

            doc.set_text(self.text)

            for sent in body:
                self.read_frame_parse(doc, sent)

            yield doc

    def read_frame_parse(self, doc, sent):
        frame_nodes = sent.find("sem").find('frames')

        if frame_nodes is None:
            logging.info(
                "No frame node found for doc %s sent %s." % (
                    doc.docid, sent.attrib['id']))
            return

        coref_sets = []

        for frame in frame_nodes:
            negra_fname = frame.attrib['name']

            if negra_fname == 'Coreference':
                # Record the coref mentions and proceed.
                coref_mentions = []
                for fe in frame.findall('fe'):
                    fe_node = fe.find('fenode')
                    fe_id = fe_node.attrib['idref']
                    fe_span = self.id2span[fe_id]
                    coref_mentions.append(fe_span)
                coref_sets.append(coref_mentions)
                continue

            frame_name = frame_changes.get(negra_fname, negra_fname)

            frame_id = frame.attrib['id']

            target_id = frame.find('target').find('fenode').attrib['idref']

            if target_id not in self.id2span:
                logging.info("Cannot for target span for " + target_id)
                continue

            target_span = self.id2span[target_id]

            p = doc.add_predicate(None, target_span, frame_type=frame_name)
            p.add_meta('id', frame_id)

            for fe in frame.findall('fe'):
                role = fe.attrib['name']
                fe_node = fe.find('fenode')

                flags = []

                linked = fe_node is not None

                for flag in fe.findall('flag'):
                    flag_name = flag.attrib['name']
                    flags.append(flag_name)
                    if flag_name == 'Definite_Interpretation':
                        self.stats['DNI'] += 1
                        if linked:
                            self.stats['DNI_resolved'] += 1
                            self.imp_frames[frame_name] += 1
                    elif flag_name == 'Indefinite_Interpretation':
                        self.stats['INI'] += 1
                    else:
                        if linked:
                            self.stats['Explicit'] += 1

                if not linked:
                    continue

                fe_id = fe_node.attrib['idref']

                if fe_id not in self.id2span:
                    logging.info("Cannot find fe span for " + fe_id)
                    continue

                fe_span = self.id2span[fe_id]

                arg_em = doc.add_entity_mention(None, fe_span,
                                                entity_type='ARG_ENT')
                arg_mention = doc.add_argument_mention(p, arg_em.aid, role)

                for flag in flags:
                    if flag == 'Definite_Interpretation':
                        arg_mention.add_meta('implicit', True)
                    else:
                        arg_mention.add_meta(flag, True)

        for coref in coref_sets:
            ent = doc.add_entity()

            for m_span in coref:
                doc.add_entity_mention(ent, m_span, entity_type='ARG_ENT')

    def build_next_sent(self, doc, c_parse):
        # Build token spans.
        sep = ' '

        sent_token_nodes = c_parse['tokens']
        id2node = c_parse['id2node']

        for i, token_node in enumerate(sent_token_nodes):
            if i == len(sent_token_nodes) - 1:
                sep = '\n'

            word = token_node.attributes['word']
            self.text += word
            self.text += sep

            w_start = self.offset
            self.offset += len(word)
            w_end = self.offset
            self.offset += 1

            token_span = Span(w_start, w_end)

            doc.add_token_span(token_span)

            self.id2span[token_node.node_id] = token_span

        for tid, node in id2node.items():
            if tid not in self.id2span:
                leave_tokens = node.get_leaves()

                begin_token_span = self.id2span[
                    leave_tokens[0].attributes['id']]
                end_token_span = self.id2span[leave_tokens[-1].attributes['id']]

                self.id2span[tid] = Span(begin_token_span.begin,
                                         end_token_span.end)

    def read_constituent_parse(self, c_graph_node):
        root_id = c_graph_node.attrib['root']

        term_nodes = c_graph_node.find('terminals')
        nt_nodes = c_graph_node.find('nonterminals')

        token_nodes = []

        id2node = {}

        for t_node in term_nodes:
            attrib = t_node.attrib
            token_node = TreeNode(attrib['id'], is_leaf=True)

            for k, v in attrib.items():
                token_node.add_attribute(k, v)

            token_nodes.append(token_node)
            id2node[token_node.node_id] = token_node

        for nt_node in nt_nodes:
            attrib = nt_node.attrib
            nonterm_node = TreeNode(attrib['id'], root_id == attrib['id'])
            nonterm_node.add_attribute('tag', attrib['cat'])

            for edge_node in nt_node.findall('edge'):
                child_id = edge_node.attrib['idref']
                child_node = id2node[child_id]
                nonterm_node.add_child(child_node)
            id2node[nonterm_node.node_id] = nonterm_node

        return {
            'tokens': token_nodes,
            'id2node': id2node,
        }

    def print_stats(self):
        logging.info("Corpus statistics from NeGra")
        for key, value in self.stats.items():
            logging.info("Stat for %s : %d" % (key, value))

        if not os.path.exists(self.stat_dir):
            os.makedirs(self.stat_dir)

        with open(os.path.join(self.stat_dir, 'basic_stats.txt'), 'w') as out:
            for key, value in self.stats.items():
                out.write(f'{key}: {value}\n')

        with open(os.path.join(self.stat_dir, 'frame_stats.txt'), 'w') as out:
            for key, value in sorted(self.imp_frames.items(),
                                     key=itemgetter(1), reverse=True):
                out.write(f'{key}\t{value}\n')
