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

    hashed_arg = dict([(k, v) for (k, v) in arg.items()])

    hashed_arg.pop('arg_context', None)
    hashed_arg.pop('role', None)

    hashed_arg['arg_role'] = arg_role_id
    hashed_arg['dep'] = dep
    hashed_arg['represent'] = entity_rep
    hashed_arg['fe'] = fe_id
    hashed_arg['context'] = hashed_context

    return hashed_arg


def hash_one_doc(docid, events, entities, event_emb_vocab, word_emb_vocab,
                 typed_event_vocab, slot_handler, sent_starts):
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
        pred = event['predicate']
        pid = event_emb_vocab.get_index(
            typed_event_vocab.get_pred_rep(event), None)
        fid = event_emb_vocab.get_index(event.get('frame'), None)
        mapped_args = slot_handler.organize_args(event)

        sent_offset = sent_starts[event['sentence_id']]

        full_args = {}

        num_implicit_arg = 0

        for slot, arg_info_list in mapped_args.items():
            hashed_arg_list = []

            has_implicit_arg = False
            for arg_info in arg_info_list:
                dep, full_fe, arg = arg_info

                if fid == -1:
                    # Even if we get a mapping from the predicate, it is still
                    # noisy to say there is a valid frame element here.
                    full_fe = None

                hashed_arg = hash_arg(
                    arg, dep, full_fe, event_emb_vocab, word_emb_vocab,
                    typed_event_vocab, entity_represents
                )

                abs_arg_start = hashed_arg['arg_start'] + sent_offset

                hashed_arg['abs_arg_start'] = abs_arg_start
                hashed_arg['abs_arg_end'] = hashed_arg['arg_end'] + sent_offset

                for sid, s_off in enumerate(sent_starts):
                    if abs_arg_start < s_off:
                        hashed_arg['sentence_id'] = sid - 1
                        break
                    else:
                        hashed_arg['sentence_id'] = len(sent_starts) - 1

                hashed_arg_list.append(hashed_arg)

                if hashed_arg['implicit']:
                    has_implicit_arg = True
                    implicit_answer_counts[pred] += 1
            full_args[slot] = hashed_arg_list

            if has_implicit_arg:
                num_implicit_arg += 1

        if num_implicit_arg > 0:
            print(f'{pred} has {num_implicit_arg} implicit arguments')
            implicit_pred_counts[pred] += 1
            implicit_slot_counts[pred] += num_implicit_arg
            input('check')

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
        for docid, events, entities, sentences in reader.read_events(data_in):

            offset = 0
            sent_starts = []
            for sent in sentences:
                sent_starts.append(offset)
                offset += len(sent) + 1

            hashed_doc = hash_one_doc(
                docid, events, entities, event_emb_vocab, word_emb_vocab,
                typed_event_vocab, slot_handler, sent_starts)

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


    from event.util import load_mixed_configs, set_basic_log

    set_basic_log()
    hash_params = HashParam(config=load_mixed_configs())

    implicit_pred_counts = Counter()
    implicit_slot_counts = Counter()
    implicit_answer_counts = Counter()

    hash_data(hash_params)

    print('======Predicates that has implicit arguments======')
    print("Predicate\tCount")
    sum = 0
    for pred, c in implicit_pred_counts.items():
        print(f"{pred}\t{c}")
        sum += c
    print(f"Total\t{sum}")
    print('==================================================')

    print('=============Slots that are implicit==============')
    print("In Predicate\tCount")
    sum = 0
    for pred, c in implicit_slot_counts.items():
        print(f"{pred}\t{c}")
        sum += c
    print(f"Total\t{sum}")
    print('==================================================')

    print('======Answers that are implicit arguments======')
    print("In Predicate\tCount")
    sum = 0
    for pred, c in implicit_answer_counts.items():
        print(f"{pred}\t{c}")
        sum += c
    print(f"Total\t{sum}")
    print('================================================')

