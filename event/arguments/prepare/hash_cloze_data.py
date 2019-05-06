from event.io.readers import EventReader
import gzip
from event.arguments.prepare.event_vocab import TypedEventVocab, EmbbedingVocab
from event.arguments.prepare import word_vocab
from event.arguments import util
from event.arguments.prepare.slot_processor import SlotHandler
from collections import defaultdict, Counter
import json
from traitlets import (
    Unicode
)
from traitlets.config.loader import PyFileConfigLoader
from traitlets.config import Configurable
import sys


def hash_context(word_emb_vocab, context):
    left, right = context
    return [word_emb_vocab.get_index(word, word_vocab.unk_word) for word in
            left], \
           [word_emb_vocab.get_index(word, word_vocab.unk_word) for word in
            right]


def hash_arg(arg, dep, full_fe, event_emb_vocab, word_emb_vocab,
             typed_event_vocab, entity_represents):
    entity_rep = typed_event_vocab.get_arg_entity_rep(
        arg, entity_represents.get(arg['entity_id']))

    arg_role = typed_event_vocab.get_arg_rep(dep, entity_rep)
    arg_role_id = event_emb_vocab.get_index(arg_role, None)

    if arg_role_id == -1:
        # arg_role = typed_event_vocab.get_arg_rep(t, entity_rep)
        arg_role = typed_event_vocab.get_arg_rep_no_dep(entity_rep)
        arg_role_id = event_emb_vocab.get_index(arg_role, None)

    if arg_role_id == -1:
        arg_role = typed_event_vocab.get_unk_arg_rep()
        arg_role_id = event_emb_vocab.get_index(arg_role, None)

    if full_fe is not None:
        frame, fe = full_fe
        fe_id = event_emb_vocab.get_index(
            typed_event_vocab.get_fe_rep(frame, fe),
            None
        )
    else:
        # Treat empty frame element as UNK.
        fe_id = event_emb_vocab.get_index(typed_event_vocab.oovs['fe'], None)

    hashed_context = hash_context(word_emb_vocab, arg['arg_context'])

    return {
        'arg_role': arg_role_id,
        'fe': fe_id,
        'context': hashed_context,
        'entity_id': arg['entity_id'],
        'implicit': arg['implicit'],
        'resolvable': arg['resolvable'],
        'arg_phrase': arg['arg_phrase'],
        'represent': entity_rep,
        'dep': dep,
    }


def hash_one_doc(docid, events, entities, event_emb_vocab, word_emb_vocab,
                 typed_event_vocab, slot_handler):
    hashed_doc = {
        'docid': docid,
        'events': [],
    }

    hashed_entities = {}
    entity_represents = {}
    for eid, entity in entities.items():
        entity_head = typed_event_vocab.get_vocab_word(
            entity['represent_entity_head'], 'argument')
        hashed_entities[eid] = {
            'features': entity['features'],
            'entity_head': entity_head,
        }
        entity_represents[eid] = entity_head

    hashed_doc['entities'] = hashed_entities

    for event in events:
        pid = event_emb_vocab.get_index(
            typed_event_vocab.get_pred_rep(event), None)
        fid = event_emb_vocab.get_index(event.get('frame'), None)
        mapped_args = slot_handler.organize_args(event)

        # if event['predicate'] == 'investment':
        #     import pprint
        #     pprint.pprint(mapped_args)
        #     input('check')

        full_args = {}
        for slot, arg_info_list in mapped_args.items():
            hashed_arg_list = []
            for arg_info in arg_info_list:
                dep, full_fe, arg = arg_info
                hashed_arg_list.append(hash_arg(
                    arg, dep, full_fe, event_emb_vocab, word_emb_vocab,
                    typed_event_vocab, entity_represents
                ))
            full_args[slot] = hashed_arg_list

        context = hash_context(word_emb_vocab, event['predicate_context'])

        hashed_doc['events'].append({
            'predicate': pid,
            'predicate_text': event['predicate'],
            'frame': fid,
            'context': context,
            'sentence_id': event['sentence_id'],
            'args': full_args,
        })
    return hashed_doc


def hash_data(params):
    slot_handler = SlotHandler(params.frame_files, params.frame_dep_map,
                               params.dep_frame_map, params.nom_map)

    typed_event_vocab = TypedEventVocab(params.component_vocab_dir)

    event_emb_vocab = EmbbedingVocab(params.event_vocab)
    word_emb_vocab = EmbbedingVocab(params.word_vocab)

    reader = EventReader()

    doc_count = 0
    event_count = 0

    print("{}: Start hashing".format(util.get_time()))
    with gzip.open(params.raw_data) as data_in, gzip.open(
            params.output_path, 'w') as data_out:
        for docid, events, entities in reader.read_events(data_in):
            hashed_doc = hash_one_doc(
                docid, events, entities, event_emb_vocab, word_emb_vocab,
                typed_event_vocab, slot_handler)
            data_out.write((json.dumps(hashed_doc) + '\n').encode())

            doc_count += 1
            event_count += len(hashed_doc['events'])

            if doc_count % 1000 == 0:
                print('\r{}: Hashed for {} events in '
                      '{} docs.'.format(util.get_time(), event_count,
                                        doc_count), end='')

    print(
        '\nTotally {} events and {} documents.'.format(event_count, doc_count)
    )


if __name__ == '__main__':
    class HashParam(Configurable):
        event_vocab = Unicode(
            help='Event and Frame Vocabulary.').tag(config=True)
        word_vocab = Unicode(
            help='Vocabulary for normal words.').tag(config=True)
        component_vocab_dir = Unicode(
            help='Directory containing vocab for each component'
        ).tag(config=True)
        nom_map = Unicode(help='NomBank mapping file').tag(config=True)
        frame_dep_map = Unicode(
            help='Mapping from predicate dependencies to frame elements.'
        ).tag(config=True)
        dep_frame_map = Unicode(
            help='Mapping from frame elements to predicate dependencies.'
        ).tag(config=True)
        raw_data = Unicode(help='The dataset to hash.').tag(config=True)
        frame_files = Unicode(help="Frame file data.").tag(config=True)
        output_path = Unicode(
            help='Output path of the hashed data.').tag(config=True)


    from event.util import load_command_line_config, set_basic_log

    set_basic_log()

    cl_conf = load_command_line_config(sys.argv[2:])
    conf = PyFileConfigLoader(sys.argv[1]).load_config()
    conf.merge(cl_conf)

    hash_params = HashParam(config=conf)

    hash_data(hash_params)
