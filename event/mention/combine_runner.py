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
                                       'aida', entity['ner'], component='EDL')

            for entity in entity_sent['nominalMentions']:
                mention_span = [entity['char_begin'], entity['char_end']]
                ner = 'NOM' if entity['ner'] == 'null' else entity['ner']
                csr.add_entity_mention(mention_span, entity['headword'],
                                       'aida', ner, component='EDL')


def add_tac_event(csr, sent_id, mention_span, text, kbp_type, args):
    evm_id, interp = csr.add_event_mention(mention_span,
                                           text, 'tac', kbp_type,
                                           sent_id=sent_id, component='tac')

    if len(args) > 0:
        pb_name = args[0]
        frame_name = args[1]
        args = args[2:]

        for arg in args:
            arg_span, arg_text, pb_role, fn_role = arg.split(',')
            arg_span = [int(a) for a in arg_span.split('-')]

            ent_id = csr.add_entity_mention(arg_span, arg_text, 'aida',
                                            'GeneralEntity', component='tac')
            if ent_id:
                if not fn_role == 'N/A':
                    csr.add_event_arg(interp, evm_id, ent_id, 'framenet',
                                      fn_role, 'Semafor')

                if not pb_role == 'N/A':
                    csr.add_event_arg(interp, evm_id, ent_id, 'propbank',
                                      pb_role, 'Fanse')
    return evm_id


def load_salience(salience_folder):
    raw_data_path = os.path.join(salience_folder, 'data.json')
    salience_event_path = os.path.join(salience_folder, 'output_event.json')
    salience_entity_path = os.path.join(salience_folder, 'output_entity.json')

    events = defaultdict(list)
    entities = defaultdict(list)

    content_field = 'bodyText'

    with open(raw_data_path) as raw_data:
        for line in raw_data:
            raw_data = json.loads(line)
            docno = raw_data['docno']
            for spot in raw_data['spot'][content_field]:
                score = spot['score']
                wiki_name = spot['wiki_name']
                mid = spot['id']
                span = tuple(spot['span'])
                head_span = tuple(spot['head_span'])

                entities[docno].append((mid, wiki_name, span, head_span, score))

            for spot in raw_data['event'][content_field]:
                span = tuple(spot['span'])
                head_span = tuple(spot['head_span'])
                surface = spot['surface']

                events[docno].append((surface, span, head_span))

    scored_events = {}
    with open(salience_event_path) as event_output:
        for line in event_output:
            output = json.loads(line)
            docno = output['docno']

            scored_events[docno] = {}

            if docno in events:
                event_list = events[docno]
                for (hid, score), (surface, span, head_span) in zip(
                        output[content_field]['predict'], event_list):
                    scored_events[docno][head_span] = (span, surface, score)

    scored_entities = {}
    with open(salience_entity_path) as entity_output:
        for line in entity_output:
            output = json.loads(line)
            docno = output['docno']

            scored_entities[docno] = {}

            if docno in events:
                ent_list = entities[docno]
                for (hid, score), (
                        mid, wiki_name, span, head_span, link_score
                ) in zip(output[content_field]['predict'], ent_list):
                    if link_score > 0.2:
                        scored_entities[docno][head_span] = (
                            span, mid, wiki_name, score
                        )

    return scored_entities, scored_events


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


def read_source(source_folder, output_dir, language):
    for source_text_path in glob.glob(source_folder + '/*.txt'):
        with open(source_text_path) as text_in:
            docid = os.path.basename(source_text_path).split('.')[0]
            print("Processing " + docid)

            csr = CSR('Frames_hector_combined', 1,
                      os.path.join(output_dir, docid + '.csr.json'), 'data')

            csr.add_doc(docid, 'report', language)
            text = text_in.read()
            sent_index = 0

            sent_path = source_text_path.replace('.txt', '.sent')
            if os.path.exists(sent_path):
                with open(sent_path) as sent_in:
                    for span_str in sent_in:
                        span = [int(s) for s in span_str.split(' ')]
                        csr.add_sentence(span, text=text[span[0]: span[1]])
                        sent_index += 1
            else:
                begin = 0
                sent_index = 0

                sent_texts = text.split('\n')
                sent_lengths = [len(t) for t in sent_texts]
                for sent_text, l in zip(sent_texts, sent_lengths):
                    if sent_text.strip():
                        csr.add_sentence((begin, begin + l), text=sent_text)
                    begin += (l + 1)
                    sent_index += 1
            yield csr, docid


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

    scored_entities, scored_events = load_salience(config.salience_data)

    print(scored_events)
    print(scored_entities)

    input("Wait")

    for csr, docid in read_source(config.source_folder, config.output,
                                  config.language):
        edl_file = find_by_id(config.edl_json, docid)
        logging.info("Predicting with EDL: {}".format(edl_file))
        add_edl_entities(edl_file, csr)

        tbf_file = find_by_id(config.event_tbf, docid)
        logging.info("Predicting with TBF: {}".format(tbf_file))
        add_tac_events(tbf_file, csr)

        conll_file = find_by_id(config.test_folder, docid)
        logging.info("Predicting with CoNLLU: {}".format(conll_file))
        test_reader = ConllUReader([conll_file], config, token_vocab,
                                   train_reader.tag_vocab, config.language)
        detector.predict(test_reader, csr)

        csr.write()


if __name__ == '__main__':
    from event import util

    parser = util.basic_parser()
    parser.add_argument('--source_folder', type=str)
    parser.add_argument('--event_tbf', type=str)
    parser.add_argument('--edl_json', type=str)
    parser.add_argument('--salience_data', type=str)

    arguments = parser.parse_args()

    util.set_basic_log()

    logging.info("Starting with the following config:")
    logging.info(arguments)

    main(arguments)
