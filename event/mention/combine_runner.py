from event.mention.detection_runners import DetectionRunner
from event.io.readers import (
    ConllUReader,
    Vocab
)
from event.io.collectors import InterpCollector
from event.util import set_basic_log
from event.mention.models.detectors import (
    TextCNN,
    FrameMappingDetector
)
import logging
import json


def add_edl_entities(edl_file, res_collector):
    if not edl_file:
        return

    with open(edl_file) as edl:
        data = json.load(edl)
        sent_index = 0

        sent_begin = 0

        for entity_sent in data:
            docid = entity_sent['docID']
            res_collector.add_doc(docid, 'report', 'text', 'ltf')
            text = entity_sent['inputSentence']

            span = (sent_begin, sent_begin + len(text))

            sent_id = res_collector.add_sentence(sent_index, span, text)
            sent_begin += len(text)

            sent_index += 1

            for entity in entity_sent['namedMentions']:
                entity_span = [entity['char_begin'], entity['char_end']]
                res_collector.add_entity(sent_id, entity_span,
                                         entity['mention'], entity['ner'])

            for entity in entity_sent['nominalMentions']:
                entity_span = [entity['char_begin'], entity['char_end']]
                res_collector.add_entity(sent_id, entity_span,
                                         entity['headword'], entity['ner'])


def add_kbp_events(kbp_file, res_collector):
    if not kbp_file:
        return

    with open(kbp_file) as kbp:
        for line in kbp:
            line = line.strip()
            if line.startswith("#"):
                if line.startswith("BeginOfDocument"):
                    docid = line.split()[1]
                    res_collector.add_doc(docid)
            else:
                parts = line.split('\t')
                if len(parts) < 9:
                    continue

                event = parts[4]
                kbp_type = parts[5]
                realis = parts[6]
                pb_role = parts[7]
                frame_role = parts[8]
                args = parts[9:]

                for arg in args:
                    arg_tokens = [int(x) for x in arg.split(':')[0].split(',')]


def main(config):
    token_vocab = Vocab(config.experiment_folder, 'tokens',
                        embedding_path=config.word_embedding,
                        emb_dim=config.word_embedding_dim)

    tag_vocab = Vocab(config.experiment_folder, 'tag',
                      embedding_path=config.tag_list)

    train_reader = ConllUReader(config.train_files, config, token_vocab,
                                tag_vocab)
    dev_reader = ConllUReader(config.dev_files, config, token_vocab,
                              train_reader.tag_vocab)
    detector = DetectionRunner(config, token_vocab, tag_vocab)
    detector.train(train_reader, dev_reader)

    res_collector = InterpCollector('Frames_hector_combined', 1, config.output,
                                    'LDC')

    test_reader = ConllUReader(config.test_files, config, token_vocab,
                               train_reader.tag_vocab)

    detector.predict(test_reader, res_collector)

    add_edl_entities(config.edl_json, res_collector)
    add_kbp_events(config.event_tbf, res_collector)

    res_collector.write()


if __name__ == '__main__':
    from event import util

    parser = util.basic_parser()
    parser.add_argument('--event_tbf', type=str)
    parser.add_argument('--edl_json', type=str)

    arguments = parser.parse_args()

    util.set_basic_log()

    logging.info("Starting with the following config:")
    logging.info(arguments)

    main(arguments)
