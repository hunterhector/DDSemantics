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
        num_gc_preds = sum([len(preds) for (d, preds) in self.gc_annos.items()])
        logging.info(f"Loaded {num_gc_preds} predicates")

        logging.info("Loading Nombank annotations")
        self.nombank_annos = defaultdict(list)
        for nb_instance in self.nombank.instances():
            docid = nb_instance.fileid.split('/')[-1]
            self.nombank_annos[docid].append(nb_instance)

        self.stats = {
            'target_pred_count': Counter(),
            'predicates_with_implicit': Counter(),
            'implicit_slots': Counter(),
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

        total_implicit_count = 0
        total_preds = 0

        for annotations in root:
            pred_node_pos = annotations.attrib['for_node']
            predicate = NomBank.NomElement.from_text(pred_node_pos)

            article_id = predicate.article_id

            total_preds += 1

            explicit_roles = set()

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

                if pred_node_pos == arg_node_pos:
                    # Incorporated nodes are explicit.
                    is_explicit = True

                if is_explicit:
                    explicit_roles.add(arg_type)
                else:
                    p = NombankTreePointer(int(arg_terminal_id),
                                           int(arg_height))
                    # Arguments are group by their sentences.
                    arg_annos[(arg_sent_id, arg_type)].append((p, is_split))

            all_args = defaultdict(list)
            implicit_role_here = set()
            for (arg_sent_id, arg_type), l_pointers in arg_annos.items():
                if int(arg_sent_id) > predicate.sent_num:
                    # Ignoring annotations after the sentence.
                    continue

                if arg_type not in explicit_roles:
                    for p in merge_split_pointers(l_pointers):
                        arg_element = NomBank.NomElement(
                            article_id, arg_sent_id, p)

                        if not predicate.pointer == arg_element.pointer:
                            # Ignoring incorporated ones.
                            all_args[arg_type].append(arg_element)
                            implicit_role_here.add(arg_type)

            gc_annotations[article_id.split('/')[-1]][predicate] = all_args

            total_implicit_count += len(implicit_role_here)

        logging.info(f"Loaded {total_preds} predicates, "
                     f"{total_implicit_count} implicit arguments.")

        return gc_annotations

    def add_predicate(self, doc, parsed_sents, predicate_node):
        pred_node_repr = "%s:%d:%s" % (
            doc.docid, predicate_node.sent_num, predicate_node.pointer)
        p_tree = parsed_sents[predicate_node.sent_num]
        p_word_idx = utils.make_words_from_pointer(
            p_tree, predicate_node.pointer)
        predicate_span = utils.get_nltk_span(
            doc.token_spans, predicate_node.sent_num, p_word_idx)

        if len(predicate_span) == 0:
            logging.warning("Zero length predicate found")
            return

        p = doc.add_predicate(None, predicate_span, frame_type='NOMBANK')

        if p:
            p.add_meta('node', pred_node_repr)

        return p

    def add_nombank_arg(self, doc, parsed_sents, wsj_spans, arg_type,
                        predicate, arg_node, implicit=False):
        arg_type = arg_type.lower()

        a_tree = parsed_sents[arg_node.sent_num]
        a_word_idx = utils.make_words_from_pointer(a_tree, arg_node.pointer)

        arg_node_repr = "%s:%d:%s" % (
            doc.docid, arg_node.sent_num, arg_node.pointer)
        argument_span = utils.get_nltk_span(wsj_spans, arg_node.sent_num,
                                            a_word_idx)

        if len(argument_span) == 0:
            # Some arguments are empty nodes, they will be ignored.
            return

        em = doc.add_entity_mention(None, argument_span)

        if em:
            if implicit:
                arg_type = 'i_' + arg_type

            arg_mention = doc.add_argument_mention(predicate, em.aid, arg_type)
            arg_mention.add_meta('node', arg_node_repr)

            if implicit:
                arg_mention.add_meta('implicit', True)
                arg_mention.add_meta('sent_num', arg_node.sent_num)
                arg_mention.add_meta('text', em.text)

            return arg_mention

    def get_predicate_text(self, p):
        p_text = p.text.lower()
        if p_text == 'losses' or p_text == 'loss' or p_text == 'tax-loss':
            p_text = 'loss'
        else:
            p_text = p_text.rstrip('s')

        if p_text == 'savings-and-loan':
            p_text = 'loan'

        if '-' in p_text:
            p_text = p_text.split('-')[1]
        return p_text

    def add_all_annotations(self, doc, parsed_sents):
        logging.info("Adding Nombank annotation for " + doc.docid)
        nb_instances = self.nombank_annos[doc.docid]

        for nb_instance in nb_instances:
            predicate_node = NomBank.NomElement(
                doc.docid, nb_instance.sentnum, nb_instance.predicate
            )

            p = self.add_predicate(doc, parsed_sents, predicate_node)

            for argloc, argid in nb_instance.arguments:
                arg_node = NomBank.NomElement(
                    doc.docid, nb_instance.sentnum, argloc
                )
                arg = self.add_nombank_arg(
                    doc, parsed_sents, doc.token_spans, argid, p, arg_node)

                if arg_node.pointer == predicate_node.pointer:
                    arg.add_meta('incorporated', True)

        if not self.params.explicit_only and doc.docid in self.gc_annos:
            for predicate_node, gc_args in self.gc_annos[doc.docid].items():
                added_args = defaultdict(list)

                p = self.add_predicate(doc, parsed_sents, predicate_node)
                p_text = utils.nombank_pred_text(p.text)

                p.add_meta('from_gc', True)

                self.stats['target_pred_count'][p_text] += 1

                for arg_type, arg_nodes in gc_args.items():
                    for arg_node in arg_nodes:
                        arg = self.add_nombank_arg(
                            doc, parsed_sents, doc.token_spans,
                            arg_type, p, arg_node, True
                        )
                        added_args[arg_type].append(arg)

                        # The following should be useless already.
                        if arg_node.pointer == predicate_node.pointer:
                            arg.add_meta('incorporated', True)

                        if arg_node.sent_num > predicate_node.sent_num:
                            arg.add_meta('succeeding', True)

                if len(added_args) > 0:
                    self.stats['predicates_with_implicit'][p_text] += 1
                    self.stats['implicit_slots'][p_text] += len(added_args)

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
