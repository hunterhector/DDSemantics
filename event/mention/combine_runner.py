from event.mention.detection_runners import DetectionRunner
from event.io.readers import (
    ConllUReader,
    Vocab
)
from event.io.csr import CSR
import logging
import json
from collections import defaultdict
import glob
import os


def add_edl_entities(edl_file, csr):
    if not edl_file:
        return
    with open(edl_file) as edl:
        data = json.load(edl)

        for entity_sent in data:
            docid = entity_sent['docID']
            csr.add_doc(docid, 'report', 'en')

            for entity in entity_sent['namedMentions']:
                mention_span = [entity['char_begin'], entity['char_end']]
                csr.add_entity_mention(mention_span, entity['mention'],
                                       'aida', entity['ner'])

            for entity in entity_sent['nominalMentions']:
                mention_span = [entity['char_begin'], entity['char_end']]
                csr.add_entity_mention(mention_span, entity['headword'],
                                       'aida', entity['ner'])


def add_tac_event(csr, sent_id, mention_span, text, kbp_type, args):
    evm_id, interp = csr.add_event_mention(mention_span,
                                           text, 'tac', kbp_type,
                                           sent_id=sent_id)

    if len(args) > 0:
        pb_name = args[0]
        frame_name = args[1]
        args = args[2:]

        for arg in args:
            arg_span, arg_text, pb_role, fn_role = arg.split(',')
            arg_span = [int(a) for a in arg_span.split('-')]

            ent_id = csr.add_entity_mention(arg_span, arg_text, 'aida', 'null')
            if ent_id:
                if not fn_role == 'N/A':
                    csr.add_arg(interp, evm_id, ent_id, 'framenet', fn_role)

                if not pb_role == 'N/A':
                    csr.add_arg(interp, evm_id, ent_id, 'propbank', pb_role)
    return evm_id


def add_tac_events(kbp_file, csr):
    if not kbp_file:
        return

    with open(kbp_file) as kbp:
        relations = defaultdict(list)
        evms = {}

        for line in kbp:
            line = line.strip()
            if line.startswith("#"):
                if line.startswith("BeginOfDocument"):
                    docid = line.split()[1]
                    # Take the non-extension docid.
                    docid = docid.split('.')[0]
                    csr.add_doc(docid)
                    relations.clear()
                    evms.clear()
            elif line.startswith("@"):
                rel_type, rid, rel_args = line.split('\t')
                rel_type = rel_type[1:]
                rel_args = rel_args.split(',')
                relations[rel_type].append(rel_args)
            else:
                parts = line.split('\t')
                if len(parts) < 7:
                    continue

                kbp_eid = parts[2]
                mention_span, text, kbp_type, realis = parts[3:7]
                mention_span = [int(p) for p in mention_span.split(',')]

                sent_id = csr.get_sentence_by_span(mention_span)
                if sent_id:
                    csr_e_id = add_tac_event(csr, sent_id, mention_span, text,
                                             kbp_type, parts[7:])
                    evms[kbp_eid] = csr_e_id

        for rel_type, relations in relations.items():
            for rel_args in relations:
                csr_rel_args = [evms[r] for r in rel_args]
                csr.add_relation('tac', csr_rel_args, rel_type)


def read_source(source_folder, csr, language):
    for source_text_path in glob.glob(source_folder + '/*.txt'):
        sent_path = source_text_path.replace('.txt', '.sent')

        with open(source_text_path) as text_in, open(sent_path) as sent_in:
            docid = os.path.basename(source_text_path).split('.')[0]
            csr.add_doc(docid, 'report', language)
            text = text_in.read()
            sent_index = 0
            for span_str in sent_in:
                span = [int(s) for s in span_str.split(' ')]
                csr.add_sentence(sent_index, span, text=text[span[0]: span[1]])
                sent_index += 1
            yield docid


def find_by_id(folder, docid):
    for filename in os.listdir(folder):
        if filename.startswith(docid):
            return os.path.join(folder, filename)


def main(config):
    token_vocab = Vocab(config.experiment_folder, 'tokens',
                        embedding_path=config.word_embedding,
                        emb_dim=config.word_embedding_dim)

    tag_vocab = Vocab(config.experiment_folder, 'tag',
                      embedding_path=config.tag_list)

    train_reader = ConllUReader(config.train_files, config, token_vocab,
                                tag_vocab, config.language)
    dev_reader = ConllUReader(config.dev_files, config, token_vocab,
                              train_reader.tag_vocab, config.language)
    detector = DetectionRunner(config, token_vocab, tag_vocab)
    detector.train(train_reader, dev_reader)

    assert config.test_folder is not None
    assert config.output is not None

    csr = CSR('Frames_hector_combined', 1, config.output, 'data')
    for docid in read_source(config.source_folder, csr, config.language):
        conll_file = find_by_id(config.test_folder, docid)
        logging.info("Predicting with CoNLLU: {}".format(conll_file))
        test_reader = ConllUReader([conll_file], config, token_vocab,
                                   train_reader.tag_vocab, config.language)
        detector.predict(test_reader, csr)

        edl_file = find_by_id(config.edl_json, docid)
        logging.info("Predicting with EDL: {}".format(edl_file))
        add_edl_entities(edl_file, csr)

        tbf_file = find_by_id(config.event_tbf, docid)
        logging.info("Predicting with TBF: {}".format(tbf_file))
        add_tac_events(tbf_file, csr)

        csr.write()


if __name__ == '__main__':
    from event import util

    parser = util.basic_parser()
    parser.add_argument('--source_folder', type=str)
    parser.add_argument('--event_tbf', type=str)
    parser.add_argument('--edl_json', type=str)

    arguments = parser.parse_args()

    util.set_basic_log()

    logging.info("Starting with the following config:")
    logging.info(arguments)

    main(arguments)
