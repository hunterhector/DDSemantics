import logging
import xml.etree.ElementTree as ET
from collections import defaultdict

from nltk.corpus import (
    NombankCorpusReader,
    BracketParseCorpusReader,
)
from nltk.corpus.reader.nombank import (
    NombankChainTreePointer,
    NombankSplitTreePointer,
    NombankTreePointer,
)
from nltk.data import FileSystemPathPointer

from event.io.dataset.base import (
    Span,
    DataLoader,
    DEDocument,
)

from event.io.dataset import utils
from collections import Counter
import os
import sys


class NomBank(DataLoader):
    """
    Loading Nombank data and implicit argument annotations.
    """

    def __init__(self, params, corpus, with_doc=False):
        super().__init__(params, corpus, with_doc)

        self.wsj_treebank = BracketParseCorpusReader(
            root=params.wsj_path,
            fileids=params.wsj_file_pattern,
            tagset='wsj',
            encoding='ascii'
        )

        logging.info(
            'Found {} treebank files.'.format(
                len(self.wsj_treebank.fileids()))
        )

        self.nombank = NombankCorpusReader(
            root=FileSystemPathPointer(params.nombank_path),
            nomfile=params.nomfile,
            framefiles=params.frame_file_pattern,
            nounsfile=params.nombank_nouns_file,
            parse_fileid_xform=lambda s: s[4:],
            parse_corpus=self.wsj_treebank
        )

        logging.info("Loading G&C annotations.")
        self.gc_annos = self.load_gc_annotations()

        logging.info("Loading Nombank annotations")
        self.nombank_annos = defaultdict(list)
        for nb_instance in self.nombank.instances():
            docid = nb_instance.fileid.split('/')[-1]
            self.nombank_annos[docid].append(nb_instance)

        self.stats = {
            'predicates_with_implicit': Counter(),
            'implicit_slots': Counter(),
            'non_incorp_implicit': Counter(),
            'non_incorp_precede_implicit': Counter(),
        }

        self.stat_dir = params.stat_dir

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

    def load_gc_annotations(self):
        tree = ET.parse(self.params.implicit_path)
        root = tree.getroot()

        gc_annotations = defaultdict(dict)

        def merge_split_pointers(pointers):
            all_pointers = []
            split_pointers = []

            for pointer, is_split in pointers:
                if is_split:
                    split_pointers.append(pointer)
                else:
                    all_pointers.append(pointer)

            if len(split_pointers) > 0:
                sorted(split_pointers, key=lambda t: t.wordnum)
                all_pointers.append(NombankChainTreePointer(split_pointers))

            return all_pointers

        for annotations in root:
            predicate = NomBank.NomElement.from_text(
                annotations.attrib['for_node']
            )

            article_id = predicate.article_id

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
                    # Arguments are group by their sentences.
                    arg_annos[(arg_sent_id, arg_type)].append((p, is_split))

            all_args = defaultdict(list)

            for (arg_sent_id, arg_type), l_pointers in arg_annos.items():
                for p in merge_split_pointers(l_pointers):
                    arg_element = NomBank.NomElement(article_id, arg_sent_id, p)
                    all_args[arg_type].append(arg_element)

            gc_annotations[article_id.split('/')[-1]][predicate] = all_args

        return gc_annotations

    def add_nombank_arg(self, doc, parsed_sents, wsj_spans, predicate, arg_type,
                        argument, implicit=False):
        arg_type = arg_type.lower()

        p_tree = parsed_sents[predicate.sent_num]
        a_tree = parsed_sents[argument.sent_num]

        p_word_idx = utils.make_words_from_pointer(p_tree, predicate.pointer)
        a_word_idx = utils.make_words_from_pointer(a_tree, argument.pointer)

        pred_node_repr = "%s:%d:%s" % (
            doc.docid, predicate.sent_num, predicate.pointer)
        arg_node_repr = "%s:%d:%s" % (
            doc.docid, argument.sent_num, argument.pointer)

        predicate_span = utils.get_nltk_span(wsj_spans, predicate.sent_num,
                                             p_word_idx)
        argument_span = utils.get_nltk_span(wsj_spans, argument.sent_num,
                                            a_word_idx)

        if len(predicate_span) == 0:
            logging.warning("Zero length predicate found")
            return

        if len(argument_span) == 0:
            # Some arguments are empty nodes, they will be ignored.
            return

        p = doc.add_predicate(None, predicate_span, frame_type='NOMBANK')
        p.add_meta('node', pred_node_repr)

        em = doc.add_entity_mention(None, argument_span)

        if p and em:
            if implicit:
                arg_type = 'i_' + arg_type

            arg_mention = doc.add_argument_mention(p, em.aid, arg_type)
            arg_mention.add_meta('node', arg_node_repr)

            if implicit:
                arg_mention.add_meta('implicit', True)

                if argument.sent_num > predicate.sent_num:
                    arg_mention.add_meta('succeeding', True)

            if predicate.pointer == argument.pointer:
                arg_mention.add_meta('incorporated', True)

            return p, arg_mention

    def add_all_annotations(self, doc, parsed_sents):
        logging.info("Adding nombank annotation for " + doc.docid)
        nb_instances = self.nombank_annos[doc.docid]

        for nb_instance in nb_instances:
            predicate_node = NomBank.NomElement(
                doc.docid, nb_instance.sentnum, nb_instance.predicate
            )

            for argloc, argid in nb_instance.arguments:
                arg_node = NomBank.NomElement(
                    doc.docid, nb_instance.sentnum, argloc
                )
                self.add_nombank_arg(doc, parsed_sents, doc.token_spans,
                                     predicate_node, argid, arg_node)

        if doc.docid in self.gc_annos:
            for predicate_node, gc_args in self.gc_annos[doc.docid].items():
                added_args = defaultdict(list)
                non_incop_args = defaultdict(list)
                nom_incop_preceed_args = defaultdict(list)

                for arg_type, arg_nodes in gc_args.items():
                    for arg_node in arg_nodes:
                        p, arg = self.add_nombank_arg(
                            doc, parsed_sents, doc.token_spans, predicate_node,
                            arg_type, arg_node, True)
                        added_args[arg_type].append(arg)

                        if not arg.meta.get('incorporated', False):
                            non_incop_args[arg_type].append(arg)

                            if not arg.meta.get('succeeding', False):
                                nom_incop_preceed_args[arg_type].append(arg)

                if len(added_args) > 0:
                    p_text = p.text.lower()

                    if p_text == 'losses' or p_text == 'loss':
                        p_text = 'loss'
                    else:
                        p_text = p_text.rstrip('s')

                    if '-' in p_text:
                        p_text = p_text.split('-')[1]

                    self.stats['predicates_with_implicit'][p_text] += 1
                    self.stats['implicit_slots'][p_text] += len(added_args)
                    self.stats['non_incorp_implicit'][p_text] += len(
                        non_incop_args
                    )
                    self.stats['non_incorp_precede_implicit'][p_text] += len(
                        nom_incop_preceed_args
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

    def load_nombank(self):
        all_annos = defaultdict(list)
        for nb_instance in self.nombank.instances():
            all_annos[nb_instance.fileid].append(nb_instance)
        return all_annos

    def get_doc(self):
        for docid, instances in self.nombank_annos.items():
            if self.params.gc_only and docid not in self.gc_annos:
                continue

            doc = DEDocument(self.corpus)
            doc.set_id(docid)

            fileid = docid.split('_')[-1][:2] + '/' + docid

            parsed_sents = self.wsj_treebank.parsed_sents(fileids=fileid)
            doc.set_parsed_sents(parsed_sents)

            token_spans = self.set_wsj_text(doc, fileid)
            doc.set_token_spans(token_spans)

            self.add_all_annotations(doc, parsed_sents)

            yield doc

    def print_stats(self):
        logging.info("Corpus statistics from Nombank")

        keys = self.stats.keys()
        headline = 'predicate\t' + '\t'.join(keys)
        sums = Counter()

        if not os.path.exists(self.stat_dir):
            os.makedirs(self.stat_dir)

        preds = sorted(self.stats['predicates_with_implicit'].keys())

        with open(os.path.join(self.stat_dir, 'counts.txt'), 'w') as out:
            print(headline)
            out.write(f'{headline}\n')

            for pred in preds:
                line = f"{pred}:"
                for key in keys:
                    line += f"\t{self.stats[key][pred]}"
                    sums[key] += self.stats[key][pred]
                print(line)
                out.write(f'{line}\n')

            sum_line = 'Total\t' + '\t'.join([str(sums[k]) for k in keys])
            print(sum_line)
            out.write(f'{sum_line}\n')
