from event.io.csr import CSR
import logging
import json
from collections import defaultdict
import glob
import os

from traitlets import (
    Unicode,
    Bool,
)
from traitlets.config.loader import PyFileConfigLoader
from event.mention.params import DetectionParams
from event.util import find_by_id
from event.io.readers import (
    ConllUReader,
    Vocab,
)
from event.mention.detection_runners import DetectionRunner
from event.io.ontology import (
    OntologyLoader,
    MappingLoader,
)
from event import resources
import nltk


def add_edl_entities(edl_file, csr):
    edl_component_id = 'opera.entities.edl.xuezhe'

    if not edl_file:
        return
    with open(edl_file) as edl:
        data = json.load(edl)

        for entity_sent in data:
            docid = entity_sent['docID']
            csr.add_doc(docid, 'report', 'en')

            for entity in entity_sent['namedMentions']:
                mention_span = [entity['char_begin'], entity['char_end']]
                if 'head_span' in entity:
                    head_span = [int(s) for s in entity['head_span'].split('-')]
                else:
                    head_span = mention_span

                csr.add_entity_mention(
                    head_span, mention_span, entity['mention'], 'aida',
                    entity['type'], entity_form='named',
                    component=edl_component_id)

            for entity in entity_sent['nominalMentions']:
                mention_span = [entity['char_begin'], entity['char_end']]
                if 'head_span' in entity:
                    head_span = [int(s) for s in entity['head_span'].split('-')]
                else:
                    head_span = mention_span

                ner = 'NOM' if entity['type'] == 'null' else entity['ner']
                csr.add_entity_mention(
                    head_span, mention_span, entity['headword'], 'aida', ner,
                    entity_form='nominal', component=edl_component_id)


def recover_via_token(tokens, token_ids):
    if not token_ids:
        return None

    first_token = tokens[token_ids[0]]
    last_token = tokens[token_ids[-1]]

    text = ""
    last_end = -1
    for tid in token_ids:
        token_span, token_text = tokens[tid]

        if not last_end == -1 and token_span[0] - last_end > 0:
            gap = token_span[0] - last_end
        else:
            gap = 0
        last_end = token_span[1]

        text += ' ' * gap
        text += token_text

    return (first_token[0][0], last_token[0][1]), text


def handle_noise(origin_onto, origin_type, frame_type):
    if 'Contact' in origin_type:
        if frame_type == 'Shoot_projectiles':
            return 'tac', 'Conflict_Attack'
        if frame_type == 'Quantity':
            return 'framenet', frame_type
    return origin_onto, origin_type


def add_rich_events(rich_event_file, csr, provided_tokens=None):
    with open(rich_event_file) as fin:
        rich_event_info = json.load(fin)

        rich_entities = {}
        ent_by_id = {}
        for rich_ent in rich_event_info['entityMentions']:
            eid = rich_ent['id']
            rich_entities[eid] = rich_ent

            if provided_tokens:
                span, text = recover_via_token(
                    provided_tokens, rich_ent['tokens'])
                head_span, head_text = recover_via_token(
                    provided_tokens, rich_ent['headWord']['tokens'])
            else:
                span = rich_ent['span']
                text = rich_ent['text']
                head_span = rich_ent['headWord']['span']

            sent_id = csr.get_sentence_by_span(span)

            ent = csr.add_entity_mention(
                head_span, span, text, 'conll', rich_ent.get('type', None),
                sent_id=sent_id, entity_form=rich_ent['entity_form'],
                component=rich_ent.get(
                    'component', 'opera.events.mention.tac.hector'))

            if 'negationWord' in rich_ent:
                ent.add_modifier('NEG', rich_ent['negationWord'])

            if ent:
                ent_by_id[eid] = ent
            else:
                if len(text) > 20:
                    print("Argument mention {} rejected.".format(eid))
                else:
                    print("Argument mention {}:{} rejected.".format(eid, text))

        evm_by_id = {}
        for mention in rich_event_info['eventMentions']:
            arguments = mention['arguments']

            if provided_tokens:
                span, text = recover_via_token(provided_tokens,
                                               mention['tokens'])
                head_span, head_text = recover_via_token(
                    provided_tokens, mention['headWord']['tokens'])
            else:
                span = mention['span']
                text = mention['text']
                head_span = mention['headWord']['span']

            sent_id = csr.get_sentence_by_span(span)

            component_name = 'opera.events.mention.tac.hector'
            ontology = 'tac'

            if mention['component'] == "FrameBasedEventDetector":
                component_name = 'opera.events.mention.framenet.semafor'
                ontology = 'framenet'

            ontology, mention_type = handle_noise(ontology, mention['type'],
                                                  mention.get('frame', ''))

            evm = csr.add_event_mention(
                head_span, span, text, ontology, mention_type,
                realis=mention.get('realis', None), sent_id=sent_id,
                component=component_name
            )

            if 'negationWord' in mention:
                evm.add_modifier('NEG', mention['negationWord'])

            if 'modalWord' in mention:
                evm.add_modifier('MOD', mention['modalWord'])

            if evm:
                eid = mention['id']
                evm_by_id[eid] = evm

                for argument in arguments:
                    entity_id = argument['entityId']
                    roles = argument['roles']
                    arg_ent = rich_entities[entity_id]

                    if provided_tokens:
                        arg_span, arg_text = recover_via_token(
                            provided_tokens, arg_ent['tokens'])
                        arg_head_span, _ = recover_via_token(
                            provided_tokens, arg_ent['headWord']['tokens'])
                    else:
                        arg_span = arg_ent['span']
                        arg_head_span = arg_ent['headWord']['span']
                        arg_text = arg_ent['text']

                    for role in roles:
                        onto_name, role_name = role.split(':')

                        onto = None
                        component = None
                        if onto_name == 'fn':
                            onto = "framenet"
                            frame_name = mention['frame']
                            component = 'Semafor'
                            role_name = frame_name + '_' + role_name
                        elif onto_name == 'pb':
                            onto = "propbank"
                            component = 'Fanse'

                        if onto and component:
                            csr.add_event_arg_by_span(
                                evm, arg_head_span, arg_span, arg_text, onto,
                                role_name, component=component
                            )

        for relation in rich_event_info['relations']:
            if relation['relationType'] == 'event_coreference':
                args = [evm_by_id[i].id for i in relation['arguments'] if
                        i in evm_by_id]
                csr.add_relation('aida', args, 'event_coreference', 'hector')

            if relation['relationType'] == 'entity_coreference':
                args = [ent_by_id[i].id for i in relation['arguments'] if
                        i in ent_by_id]
                csr.add_relation('aida', args, 'entity_coreference', 'corenlp')


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


def analyze_sentence(text):
    words = nltk.word_tokenize(text)

    negations = []
    for w in words:
        if w in resources.negative_words:
            negations.append(w)

    return negations


def read_source(source_folder, output_dir, language, aida_ontology,
                onto_mapper):
    for source_text_path in glob.glob(source_folder + '/*.txt'):
        # Use the empty newline to handle different newline format.
        with open(source_text_path, newline='') as text_in:
            docid = os.path.basename(source_text_path).split('.')[0]
            csr = CSR('Frames_hector_combined', 1,
                      os.path.join(output_dir, docid + '.csr.json'), 'data',
                      aida_ontology=aida_ontology, onto_mapper=onto_mapper)

            csr.add_doc(docid, 'report', language)
            text = text_in.read()
            sent_index = 0

            sent_path = source_text_path.replace('.txt', '.sent')
            if os.path.exists(sent_path):
                with open(sent_path) as sent_in:
                    for span_str in sent_in:
                        span = [int(s) for s in span_str.split(' ')]
                        sent_text = text[span[0]: span[1]]
                        sent = csr.add_sentence(span, text=sent_text)

                        negations = analyze_sentence(sent_text)
                        for neg in negations:
                            sent.add_modifier('NEG', neg)

                        sent_index += 1
            else:
                begin = 0
                sent_index = 0

                sent_texts = text.split('\n')
                sent_lengths = [len(t) for t in sent_texts]
                for sent_text, l in zip(sent_texts, sent_lengths):
                    if sent_text.strip():
                        sent = csr.add_sentence((begin, begin + l),
                                                text=sent_text)
                        negations = analyze_sentence(sent_text)
                        for neg in negations:
                            sent.add_modifier('NEG', neg)

                    begin += (l + 1)
                    sent_index += 1

            yield csr, docid


def mid_rdf_format(mid):
    return mid.strip('/').replace('/', '.')


def add_entity_salience(csr, entity_salience_info):
    for span, data in entity_salience_info.items():
        entity = csr.get_by_span(csr.entity_key, span)
        if not entity:
            entity = csr.add_entity_mention(
                span, data['span'], data['text'], 'aida', None,
                entity_form='named', component='dbpedia-spotlight-0.7')

        if not entity:
            if len(data['text']) > 20:
                logging.info("Wikified mention [{}] rejected.".format(span))
            else:
                logging.info(
                    "Wikified mention [{}:{}] rejected.".format(
                        span, data['text'])
                )

        if entity:
            entity.add_salience(data['salience'])
            entity.add_linking(
                mid_rdf_format(data['mid']), data['wiki'], data['link_score'],
                # component='wikifier'
            )


def add_event_salience(csr, event_salience_info):
    for span, data in event_salience_info.items():
        event = csr.get_by_span(csr.event_key, span)
        if not event:
            event = csr.add_event_mention(span, data['span'], data['text'],
                                          'aida', None, component='salience')
            if event:
                event.add_salience(data['salience'])


def token_to_span(conll_file):
    tokens = []
    with open(conll_file) as fin:
        for line in fin:
            line = line.strip()
            if line.startswith("#"):
                continue
            elif line == "":
                continue
            else:
                parts = line.split('\t')
                span = tuple([int(s) for s in parts[-1].split(',')])
                tokens.append((span, parts[1]))
    return tokens


def main(config):
    assert config.conllu_folder is not None
    assert config.csr_output is not None

    if config.salience_data:
        scored_entities, scored_events = load_salience(config.salience_data)

    if not os.path.exists(config.csr_output):
        os.makedirs(config.csr_output)

    aida_ontology = OntologyLoader(config.ontology_path)
    onto_mapper = MappingLoader()
    onto_mapper.load_seedling_arg_mapping(config.seedling_argument_mapping)
    onto_mapper.load_seedling_event_mapping(config.seedling_event_mapping)

    if config.add_rule_detector:
        # Rule detector should not need existing vocabulary.
        token_vocab = Vocab(config.resource_folder, 'tokens',
                            embedding_path=config.word_embedding,
                            emb_dim=config.word_embedding_dim,
                            ignore_existing=True)
        tag_vocab = Vocab(config.resource_folder, 'tag',
                          embedding_path=config.tag_list,
                          ignore_existing=True)
        detector = DetectionRunner(config, token_vocab, tag_vocab,
                                   aida_ontology)

    ignore_edl = False
    if config.edl_json:
        if os.path.exists(config.edl_json):
            logging.info("Loading from EDL: {}".format(config.edl_json))
        else:
            logging.warning("EDL output not found: {}, will be ignored.".format(
                config.edl_json))
            ignore_edl = True

    for csr, docid in read_source(config.source_folder, config.csr_output,
                                  config.language, aida_ontology, onto_mapper):
        logging.info('Working with docid: {}'.format(docid))

        if config.edl_json and not ignore_edl:
            edl_file = find_by_id(config.edl_json, docid)
            if edl_file:
                logging.info("Predicting with EDL: {}".format(edl_file))
                add_edl_entities(edl_file, csr)

        conll_file = find_by_id(config.conllu_folder, docid)
        if not conll_file:
            logging.warning("CoNLL file for doc {} is missing, please "
                            "check your paths.".format(docid))
            continue

        tokens = None

        if config.rich_event_token:
            tokens = token_to_span(conll_file)

        if config.rich_event:
            rich_event_file = find_by_id(config.rich_event, docid)
            if rich_event_file:
                logging.info(
                    "Adding events with rich output: {}".format(
                        rich_event_file))
                add_rich_events(rich_event_file, csr, tokens)

        if config.salience_data:
            if docid in scored_entities:
                add_entity_salience(csr, scored_entities[docid])
            if docid in scored_events:
                add_event_salience(csr, scored_events[docid])

        logging.info("Reading on CoNLLU: {}".format(conll_file))
        # The conll files may contain information from another language.
        test_reader = ConllUReader([conll_file], config, token_vocab,
                                   tag_vocab, config.language)

        if config.add_rule_detector:
            logging.info("Adding from ontology based rule detector.")
            # Adding rule detector. This is the last detector that use other
            # information from the CSR, including entity and events.
            detector.predict(test_reader, csr)

            # align_ontology(csr, aida_ontology)

        csr.write()


if __name__ == '__main__':
    from event import util
    import sys


    class CombineParams(DetectionParams):
        source_folder = Unicode(help='source text folder').tag(config=True)
        rich_event = Unicode(help='Rich event output.').tag(config=True)
        edl_json = Unicode(help='EDL json output.').tag(config=True)
        relation_json = Unicode(help='Relation json output.').tag(config=True)
        salience_data = Unicode(help='Salience output.').tag(config=True)
        rich_event_token = Bool(
            help='Whether to use tokens from rich event output',
            default_value=False).tag(config=True)
        add_rule_detector = Bool(help='Whether to add rule detector',
                                 default_value=False).tag(config=True)
        output_folder = Unicode(help='Parent output directory').tag(config=True)
        conllu_folder = Unicode(help='CoNLLU directory').tag(config=True)
        csr_output = Unicode(help='Main CSR output directory').tag(
            config=True)


    util.set_basic_log()
    conf = PyFileConfigLoader(sys.argv[1]).load_config()

    cl_conf = util.load_command_line_config(sys.argv[2:])
    conf.merge(cl_conf)

    params = CombineParams(config=conf)

    main(params)
