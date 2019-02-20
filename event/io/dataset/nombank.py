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


class NomBank(DataLoader):
    """
    Loading Nombank data and implicit argument annotations.
    """

    def __init__(self, params):
        super().__init__(params)

        self.wsj_treebank = self.load_treebank()

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

    def load_treebank(self):
        logging.info('Loading WSJ Treebank.')

        return BracketParseCorpusReader(
            root=params.wsj_path,
            fileids=params.wsj_file_pattern,
            tagset='wsj',
            encoding='ascii'
        )


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
                sorted(split_pointers, key=lambda t: t.wordnum)
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
                    if start >= end:
                        logging.warning(
                            "Found invalid span [%d:%d] at doc [%s]" % (
                                start, end, fileid))
                        print(
                            "Found invalid span [%d:%d] at doc [%s]" % (
                                start, end, fileid))
                        input('error')
                    else:
                        spans.append(Span(start, end))
            return spans

        arg_type = arg_type.lower()

        sents, parsed_sents = self.get_wsj_data(fileid)
        p_tree = parsed_sents[predicate.sent_num]
        a_tree = parsed_sents[argument.sent_num]

        p_word_idx, p_word_surface = self.make_words(p_tree, predicate.pointer)
        a_word_idx, a_word_surface = self.make_words(a_tree, argument.pointer)

        docid = fileid.split('/')[1].split('.')[0]

        pred_node_repr = "%s:%d:%s" % (
            docid, predicate.sent_num, predicate.pointer)
        arg_node_repr = "%s:%d:%s" % (
            docid, argument.sent_num, argument.pointer)

        predicate_span = get_span(predicate.sent_num, p_word_idx)
        argument_span = get_span(argument.sent_num, a_word_idx)

        # if len(a_word_idx) > 1:
        #     print("Argument index", a_word_idx, argument_span)
        #     print(a_word_surface)
        #     input('waiting for predicate argument.')

        if len(predicate_span) == 0:
            logging.warning("Zero length predicate found")
            return

        if len(argument_span) == 0:
            # Some arguments are empty nodes, they will be ignored.
            return

        p = doc.add_predicate(None, predicate_span, frame_type='NOMINAL')
        arg_em = doc.add_entity_mention(None, argument_span,
                                        entity_type='ARG_ENT')

        if p and arg_em:
            p.add_meta('node', pred_node_repr)

            if implicit:
                arg_type = 'i_' + arg_type

            arg_mention = doc.add_argument_mention(p, arg_em.aid, arg_type)
            arg_mention.add_meta('node', arg_node_repr)

            if implicit:
                arg_mention.add_meta('implicit', True)

                if argument.sent_num > predicate.sent_num:
                    arg_mention.add_meta('succeeding', True)

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

        return sorted(pointers, key=lambda pt: pt.wordnum)

    def make_words(self, tree, tree_pointer):
        pointers = self.get_normal_pointers(tree_pointer)

        all_word_idx = []
        all_word_surface = []

        for pointer in pointers:
            treepos = pointer.treepos(tree)

            idx_list = []
            for idx in range(len(tree.leaves())):
                if tree.leaf_treeposition(idx)[:len(treepos)] == treepos:
                    idx_list.append(idx)

            idx_list.sort()
            word_list = [tree.leaves()[idx] for idx in idx_list]

            if len(all_word_idx) > 0 and \
                    idx_list[0] - 1 == all_word_idx[-1][-1]:
                all_word_idx[-1].extend(idx_list)
                all_word_surface[-1].extend(word_list)
            else:
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
            for predicate_node, gc_args in self.gc_annos[fileid].items():
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
