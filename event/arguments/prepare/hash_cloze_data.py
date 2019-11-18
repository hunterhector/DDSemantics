from event.io.readers import EventReader
import gzip
from event.arguments.prepare.event_vocab import TypedEventVocab, EmbbedingVocab
from event.arguments.prepare import word_vocab
from event.arguments import util
from event.arguments.prepare.slot_processor import SlotHandler
from collections import Counter
import json
from traitlets import (
    Unicode, Bool
)
from traitlets.config import Configurable
from pprint import pprint
from event.io.dataset import utils as data_utils
import logging


def hash_context(word_emb_vocab, context):
    left, right = context
    return [word_emb_vocab.get_index(word, word_vocab.unk_word) for word in
            left], \
           [word_emb_vocab.get_index(word, word_vocab.unk_word) for word in
            right]


def get_framenet_role_index(frame, role, event_emb_vocab, typed_event_vocab):
    return event_emb_vocab.get_index(
        typed_event_vocab.get_fe_rep(frame, role), typed_event_vocab.oovs['fe']
    )


def get_propbank_role_index(role):
    role_idx = -1
    if role.startswith('i_arg'):
        role_idx = int(role.replace('i_arg', ''))
    elif role.startswith('arg') and not role.startswith(
            'argm') and not role.startswith('arga'):
        role_idx = int(role.replace('arg', ''))

    if role_idx > 3:
        role_idx = -1
    return role_idx


def hash_arg(arg, dep, frame, fe, event_emb_vocab, word_emb_vocab,
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

    fe_id = event_emb_vocab.get_index(
        typed_event_vocab.get_fe_rep(frame, fe),
        typed_event_vocab.oovs['fe']
    )

    hashed_context = hash_context(word_emb_vocab, arg['arg_context'])

    hashed_arg = dict([(k, v) for (k, v) in arg.items()])

    if 'gold_role' in hashed_arg:
        if hash_params.frame_formalism == 'Propbank':
            hashed_arg['gold_role_id'] = get_propbank_role_index(
                hashed_arg['gold_role'])
        elif hash_params.frame_formalism == 'Framenet':
            hashed_arg['gold_role_id'] = \
                get_framenet_role_index(frame, hashed_arg['gold_role'],
                                        event_emb_vocab, typed_event_vocab)

    hashed_arg.pop('arg_context', None)
    hashed_arg.pop('role', None)

    hashed_arg['arg_role'] = arg_role_id
    hashed_arg['dep'] = dep
    hashed_arg['represent'] = entity_rep
    hashed_arg['fe'] = fe_id
    hashed_arg['context'] = hashed_context

    return hashed_arg


def overlap(begin1, end1, begin2, end2):
    if begin2 <= begin1 <= end2 or begin2 <= end1 <= end2:
        return True
    elif begin1 <= begin2 <= end1 or begin1 <= end2 <= end1:
        return True
    else:
        return False


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
        if hash_params.use_gold_frame:
            # The gold frame is stored at the event type section.
            frame = event['event_type']
        else:
            frame = event['frame']

        pid = event_emb_vocab.get_index(
            typed_event_vocab.get_pred_rep(event), None)
        fid = event_emb_vocab.get_index(frame, None)

        # TODO: many args are mapped in the "prep" slot, but some of them are
        # different implicit slots.
        mapped_args = slot_handler.organize_args(event)

        full_args = {}

        implicit_slots_preceed = set()
        implicit_slots_no_incorp = set()
        implicit_slots_all = set()

        raw_pred = data_utils.nombank_pred_text(event['predicate'])

        for slot, arg_info_list in mapped_args.items():
            hashed_arg_list = []
            for arg_info in arg_info_list:
                dep, fe, arg = arg_info

                hashed_arg = hash_arg(
                    arg, dep, frame, fe, event_emb_vocab, word_emb_vocab,
                    typed_event_vocab, entity_represents,
                )

                hashed_arg_list.append(hashed_arg)

                if hashed_arg['implicit']:
                    if 'gold_role' not in arg:
                        logging.warning(f'Gold role at field '
                                        f'goldRole is NA.')
                    else:
                        gold_role = arg['gold_role']
                        implicit_slots_all.add(gold_role)

                        if not hashed_arg['incorporated']:
                            implicit_slots_no_incorp.add(gold_role)
                            if not hashed_arg['succeeding']:
                                implicit_slots_preceed.add(gold_role)
            full_args[slot] = hashed_arg_list

        num_actual = len(event["arguments"])
        num_full_args = sum([len(l) for l in full_args.values()])
        num_mapped_args = sum([len(l) for l in mapped_args.values()])
        if not num_actual == num_full_args:
            print(f'Actual number of args {num_actual}')
            print(f'Full args contains {num_full_args} args')
            print(f'mapped args contains {num_mapped_args} args')

            pprint(event['arguments'])
            pprint(full_args)

            raise ValueError("Incorrect argument numbers.")

        frame_key = None
        if event['is_target']:
            frame_key = raw_pred

        if hash_params.use_gold_frame and not frame == 'Verbal':
            frame_key = frame

        if frame_key is not None:
            stat_counters['predicate'][frame_key] += 1
            if len(implicit_slots_all) > 0:
                stat_counters['implicit predicates'][frame_key] += 1
                stat_counters['implicit slots'][frame_key] += len(
                    implicit_slots_all)

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


def hash_data():
    slot_handler = SlotHandler(hash_params.frame_files,
                               hash_params.frame_dep_map,
                               hash_params.dep_frame_map,
                               hash_params.nom_map)

    typed_event_vocab = TypedEventVocab(hash_params.component_vocab_dir)

    event_emb_vocab = EmbbedingVocab(hash_params.event_vocab)
    word_emb_vocab = EmbbedingVocab(hash_params.word_vocab)

    reader = EventReader()

    doc_count = 0
    event_count = 0

    print(f"{util.get_time()}: Start hashing")
    with gzip.open(hash_params.raw_data) as data_in, gzip.open(
            hash_params.output_path, 'w') as data_out:
        for docid, events, entities, sentences in reader.read_events(
                data_in, 'goldRole'):

            offset = 0
            sent_starts = []
            for sent in sentences:
                sent_starts.append(offset)
                offset += len(sent) + 1

            hashed_doc = hash_one_doc(
                docid, events, entities, event_emb_vocab, word_emb_vocab,
                typed_event_vocab, slot_handler)

            data_out.write((json.dumps(hashed_doc) + '\n').encode())

            doc_count += 1
            event_count += len(hashed_doc['events'])

            if doc_count % 1000 == 0:
                print(f'{util.get_time()}: Hashed for {event_count} events in '
                      f'{doc_count} docs.\r', end='')

    print(
        f'\nTotally {event_count} events and {doc_count} documents.'
    )


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
    frame_formalism = Unicode(
        help='Which frame formalism is to predict the slots, currently support '
             'FrameNet and Propbank', default_value='Propbank'
    ).tag(config=True)
    use_gold_frame = Bool(
        help='Use gold the gold frame produced by annotation',
        default_value=False).tag(config=True)


if __name__ == '__main__':

    from event.util import load_mixed_configs, set_basic_log

    set_basic_log()
    hash_params = HashParam(config=load_mixed_configs())

    stat_counters = {
        'predicate': Counter(),
        'implicit predicates': Counter(),
        'implicit slots': Counter(),
    }
    stat_keys = stat_counters.keys()

    hash_data()

    print('==========Implicit arguments Statistics===========')
    headline = "Predicate\t" + "\t".join(stat_keys)
    print(headline)
    preds = sorted(stat_counters['implicit predicates'].keys())

    sums = [0] * len(stat_keys)
    for pred in preds:
        line = pred
        for idx, key in enumerate(stat_keys):
            v = stat_counters[key][pred]
            line += f"\t{v}"
            sums[idx] += v
        print(line)

    print("Total\t" + '\t'.join([str(s) for s in sums]))
    print('==================================================')
