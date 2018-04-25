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
                head_span = [int(s) for s in entity['head_span'].split('-')]
                csr.add_entity_mention(head_span, mention_span,
                                       entity['mention'], 'aida', entity['ner'],
                                       component='EDL')

            for entity in entity_sent['nominalMentions']:
                mention_span = [entity['char_begin'], entity['char_end']]
                head_span = [int(s) for s in entity['head_span'].split('-')]
                ner = 'NOM' if entity['ner'] == 'null' else entity['ner']
                csr.add_entity_mention(head_span, mention_span,
                                       entity['headword'], 'aida', ner,
                                       component='EDL')


def add_tbf_event(csr, sent_id, mention_span, text, kbp_type, args):
    evm = csr.add_event_mention(mention_span, mention_span,
                                text, 'tac', kbp_type,
                                sent_id=sent_id, component='tac')

    if len(args) > 0:
        pb_name = args[0]
        frame_name = args[1]
        args = args[2:]

        for arg in args:
            arg_span, arg_text, pb_role, fn_role = arg.split(',')
            arg_span = [int(a) for a in arg_span.split('-')]

            ent = csr.add_entity_mention(arg_span, arg_span, arg_text,
                                         'aida', None, component='tac')
            if ent.id:
                if not fn_role == 'N/A':
                    csr.add_event_arg(evm.interp, evm.id, ent.id, 'framenet',
                                      fn_role, 'Semafor')

                if not pb_role == 'N/A':
                    csr.add_event_arg(evm.interp, evm.id, ent.id, 'propbank',
                                      pb_role, 'Fanse')
    return evm


def add_rich_events(rich_event_file, csr):
    with open(rich_event_file) as fin:
        rich_event_info = json.load(fin)

        rich_entities = {}
        ent_by_id = {}
        for rich_ent in rich_event_info['entityMentions']:
            eid = rich_ent['id']
            rich_entities[eid] = rich_ent
            span = rich_ent['span']
            sent_id = csr.get_sentence_by_span(span)

            ent = csr.add_entity_mention(rich_ent['headWord']['span'], span,
                                         rich_ent['text'], 'conll',
                                         rich_ent.get('type', None),
                                         sent_id, component='stanford')
            ent_by_id[eid] = ent

        evm_by_id = {}
        for mention in rich_event_info['eventMentions']:
            head_span = mention['headWord']['span']
            arguments = mention['arguments']
            span = mention['span']
            text = mention.get('text', "")

            sent_id = csr.get_sentence_by_span(span)

            evm = csr.add_event_mention(head_span, span, text, 'tac',
                                        mention['type'], sent_id=sent_id,
                                        component='tac')

            eid = mention['id']
            evm_by_id[eid] = evm

            for argument in arguments:
                entity_id = argument['entityId']
                roles = argument['roles']
                arg_ent = rich_entities[entity_id]

                arg_span = arg_ent['span']
                arg_head_span = arg_ent['headWord']['span']

                ent_type = arg_ent.get('type', None)

                ent = csr.add_entity_mention(arg_head_span, arg_span,
                                             arg_ent['text'], 'aida', ent_type,
                                             component='tac')
                if ent.id:
                    for role in roles:
                        onto_name, role_name = role.split(':')
                        if onto_name == 'fn':
                            csr.add_event_arg(evm.interp, evm.id, ent.id,
                                              'framenet', role_name, 'Semafor')

                        elif onto_name == 'pb':
                            csr.add_event_arg(evm.interp, evm.id, ent.id,
                                              'propbank', role_name, 'Fanse')

        for relation in rich_event_info['relations']:
            if relation['relationType'] == 'event_coreference':
                args = [evm_by_id[i].id for i in relation['arguments']]
                csr.add_relation('tac', args, 'event_coreference', 'hector')

            if relation['relationType'] == 'entity_coreference':
                args = [ent_by_id[i].id for i in relation['arguments']]
                csr.add_relation('tac', args, 'entity_coreference', 'CoreNLP')


def add_tbf_events(kbp_file, csr):
    if not kbp_file:
        return

    with open(kbp_file) as kbp:
        relations = defaultdict(list)
        evms = {}

        for line in kbp:
            line = line.strip()
            if line.startswith("#"):
                if line.startswith("#BeginOfDocument"):
                    docid = line.split()[1]
                    # Take the non-extension docid.
                    docid = docid.split('.')[0]
                    # csr.add_doc(docid)
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
                    csr_evm = add_tbf_event(csr, sent_id, mention_span, text,
                                            kbp_type, parts[7:])
                    evms[kbp_eid] = csr_evm.id

        for rel_type, relations in relations.items():
            for rel_args in relations:
                csr_rel_args = [evms[r] for r in rel_args]
                csr.add_relation('tac', csr_rel_args, rel_type)


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
                entities[docno].append({
                    'mid': spot['id'],
                    'wiki': spot['wiki_name'],
                    'span': tuple(spot['span']),
                    'head_span': tuple(spot['head_span']),
                    'score': spot['score'],
                    'text': spot['surface']
                })

            for spot in raw_data['event'][content_field]:
                events[docno].append(
                    {
                        'text': spot['surface'],
                        'span': tuple(spot['span']),
                        'head_span': tuple(spot['head_span']),
                    }
                )

    scored_events = {}
    with open(salience_event_path) as event_output:
        for line in event_output:
            output = json.loads(line)
            docno = output['docno']

            scored_events[docno] = {}

            if docno in events:
                event_list = events[docno]
                for (hid, score), event_info in zip(
                        output[content_field]['predict'], event_list):
                    scored_events[docno][event_info['head_span']] = {
                        'span': event_info['span'],
                        'salience': score,
                        'text': event_info['text']
                    }

    scored_entities = {}
    with open(salience_entity_path) as entity_output:
        for line in entity_output:
            output = json.loads(line)
            docno = output['docno']

            scored_entities[docno] = {}

            if docno in events:
                ent_list = entities[docno]
                for (hid, score), (entity_info) in zip(
                        output[content_field]['predict'], ent_list):

                    link_score = entity_info['score']

                    if link_score > 0.15:
                        scored_entities[docno][entity_info['head_span']] = {
                            'span': entity_info['span'],
                            'mid': entity_info['mid'],
                            'wiki': entity_info['wiki'],
                            'salience': score,
                            'link_score': entity_info['score'],
                            'text': entity_info['text'],
                        }

    return scored_entities, scored_events


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


def add_entity_salience(csr, entity_salience_info):
    for span, data in entity_salience_info.items():
        entity = csr.get_by_span('entity', span)
        if not entity:
            entity = csr.add_entity_mention(span, data['span'], data['text'],
                                            'aida', None,
                                            component='tagme')
        entity.add_salience(data['salience'])
        entity.add_linking(data['mid'], data['wiki'], data['link_score'],
                           component='tagme')


def add_event_salience(csr, event_salience_info):
    for span, data in event_salience_info.items():
        event = csr.get_by_span('event', span)
        if not event:
            event = csr.add_event_mention(span, data['span'], data['text'],
                                          'aida', None, component='tagme')
        event.add_salience(data['salience'])


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

    for csr, docid in read_source(config.source_folder, config.output,
                                  config.language):
        edl_file = find_by_id(config.edl_json, docid)
        logging.info("Predicting with EDL: {}".format(edl_file))
        add_edl_entities(edl_file, csr)

        rich_event_file = find_by_id(config.rich_event, docid)
        logging.info(
            "Adding events with rich output: {}".format(rich_event_file))
        add_rich_events(rich_event_file, csr)
        # tbf_file = find_by_id(config.event_tbf, docid)
        # logging.info("Predicting with TBF: {}".format(tbf_file))
        # add_tbf_events(tbf_file, csr)

        add_entity_salience(csr, scored_entities[docid])
        add_event_salience(csr, scored_events[docid])

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
    parser.add_argument('--rich_event', type=str)
    parser.add_argument('--edl_json', type=str)
    parser.add_argument('--salience_data', type=str)

    arguments = parser.parse_args()

    util.set_basic_log()

    logging.info("Starting with the following config:")
    logging.info(arguments)

    main(arguments)
