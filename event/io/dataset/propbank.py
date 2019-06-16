import logging
from event.io.dataset.base import (
    Span,
    DataLoader,
    DEDocument,
)

from nltk.corpus import (
    PropbankCorpusReader,
    BracketParseCorpusReader,
)

from nltk.data import FileSystemPathPointer
from collections import defaultdict

from event.io.dataset import utils
from collections import Counter


class PropBank(DataLoader):
    """
    Load PropBank data.
    """

    def __init__(self, params, corpus, with_doc=False):
        super().__init__(params, corpus)
        logging.info('Initialize PropBank reader.')

        if with_doc:
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

        self.propbank = PropbankCorpusReader(
            root=FileSystemPathPointer(params.root),
            propfile=params.propfile,
            framefiles=params.frame_files,
            verbsfile=params.verbs_file,
        )

        self.propbank_annos = defaultdict(list)
        logging.info("Loading PropBank Data.")
        for inst in self.propbank.instances():
            docid = inst.fileid.split('/')[-1]
            self.propbank_annos[docid].append(inst)

        self.stats = {
            'predicate_count': 0,
            'argument_count': 0,
        }

    def add_all_annotations(self, doc):
        logging.info("Adding propbank annotations for " + doc.docid)

        instances = self.propbank_annos[doc.docid]

        for inst in instances:
            parsed_sents = doc.get_parsed_sents()

            tree = parsed_sents[inst.sentnum]

            p_word_idx = utils.make_words_from_pointer(tree, inst.predicate)
            pred_span = utils.get_nltk_span(doc.get_token_spans(),
                                            inst.sentnum, p_word_idx)

            pred_node_repr = "%s:%d:%s" % (
                doc.docid, inst.sentnum, inst.predicate)

            self.stats['predicate_count'] += 1

            for argloc, arg_slot in inst.arguments:
                a_word_idx = utils.make_words_from_pointer(tree, argloc)
                arg_span = utils.get_nltk_span(
                    doc.get_token_spans(), inst.sentnum, a_word_idx)

                if len(arg_span) == 0:
                    continue

                self.stats['argument_count'] += 1

                p = doc.add_predicate(None, pred_span, frame_type='PROPBANK')
                arg_em = doc.add_entity_mention(None, arg_span)
                arg_node_repr = "%s:%d:%s" % (
                    doc.docid, inst.sentnum, argloc)

                if p and arg_em:
                    p.add_meta('node', pred_node_repr)

                    arg_mention = doc.add_argument_mention(
                        p, arg_em.aid, arg_slot.lower())
                    arg_mention.add_meta('node', arg_node_repr)

    def print_stats(self):
        logging.info("Corpus statistics from Propbank")

        for key, value in self.stats.items():
            logging.info(f"{key} : {value}")
