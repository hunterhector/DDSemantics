from event.io.readers import EventReader
import gzip
from event.arguments.prepare.event_vocab import TypedEventVocab, EmbbedingVocab
from event.arguments.prepare import word_vocab
from event.arguments import util
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


def load_emb_vocab(vocab_file):
    vocab = {}
    with open(vocab_file) as infile:
        index = 0
        for line in infile:
            word, count = line.split()
            vocab[word] = index
            index += 1
    return vocab


def load_frame_map(frame_map_file):
    fmap = {}
    counts = {}
    with open(frame_map_file) as frame_maps:
        for line in frame_maps:
            from_info, target_info = line.split('\t')
            from_pred, from_arg, from_count = from_info.split(' ')
            fmap[from_pred, from_arg] = []
            for target in target_info.split():
                role_info, count = target.split(':')
                role_parts = role_info.split(',')

                to_pred = ','.join(role_parts[:-1])
                to_arg = role_parts[-1]

                fmap[from_pred, from_arg].append((to_pred, to_arg, int(count)))
            counts[from_pred] = from_count
    return fmap, counts


def remove_slot_info(arg_info):
    content = {}
    for k, v in arg_info.items():
        if not (k == 'dep' or k == 'fe'):
            content[k] = v
    return content


def tiebreak_arg(tied_args, pred_start, pred_end):
    top_index = 0
    priority = (3, float('inf'))

    # TODO: priority didn't consider the source.
    for i, (d, ffe, a, source) in enumerate(tied_args):
        arg_start, arg_end = a['arg_start'], a['arg_end']

        if arg_start >= pred_end:
            dist = arg_start - pred_end
        elif arg_end <= pred_start:
            dist = pred_end - arg_end
        else:
            # Overlapping case, we give the lowest priority.
            dist = float('inf')

        num_emtpy = 0
        if d == 'None':
            num_emtpy += 1

        if ffe == 'None':
            num_emtpy += 1

        this_priority = (num_emtpy, dist)

        if this_priority < priority:
            top_index = i
            priority = this_priority

    return tied_args[top_index]


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


def get_dep_position(dep):
    if dep == 'nsubj' or dep == 'agent':
        return 'subj'
    elif dep == 'dobj' or dep == 'nsubjpass':
        return 'obj'
    elif dep == 'iobj':
        # iobj is more prep like location
        return 'prep'
    elif dep.startswith('prep_'):
        return 'prep'

    return 'NA'


def impute_args(event, frame_args, arg_frames):
    # If we would like to have frames here, then we should come up with another
    # priority strategy.
    args = event['arguments']
    pred_start, pred_end = event['predicate_start'], event['predicate_end']

    arg_candidates = {
        'subj': [],
        'obj': [],
        'prep': [],
    }

    dep_slots = {}
    frame_slots = {}

    predicate = util.remove_neg(event.get('predicate'))
    frame = event.get('frame', 'NA')

    for arg in args:
        dep = arg.get('dep', 'NA')
        fe = arg.get('fe', 'NA')

        if not dep == 'NA' and get_dep_position(dep) not in arg_candidates:
            # If dep is an known but not in our target list, ignore them.
            continue

        if not dep == 'NA':
            dep_slots[dep] = ((frame, fe), remove_slot_info(arg))

        if not fe == 'NA':
            frame_slots[(frame, fe)] = (dep, remove_slot_info(arg))

    imputed_fes = defaultdict(Counter)
    for dep, (full_fe, arg) in dep_slots.items():
        position = get_dep_position(dep)

        if full_fe[1] == 'NA':
            candidates = arg_frames.get((predicate, dep), [])
            not_trust = dep.startswith('prep_')
            imputed = False

            if not not_trust:
                for cand_frame, cand_fe, cand_count in candidates:
                    if (cand_frame, cand_fe) not in frame_slots:
                        imputed_fes[(cand_frame, cand_fe)][dep] = cand_count
                        imputed = True
                        break

            if not imputed:
                # No impute can be found, or we do not trust the imputation.
                # In this case, we place an empty FE name here.
                arg_candidates[position].append((dep, None, arg, 'no_impute'))
        else:
            arg_candidates[position].append((dep, full_fe, arg, 'origin'))

    imputed_deps = defaultdict(Counter)
    for (frame, fe), (dep, arg) in frame_slots.items():
        if dep == 'NA':
            for pred, dep, cand_count in frame_args.get((frame, fe), []):
                if dep not in dep_slots and pred == event['predicate']:
                    imputed_deps[dep][(frame, fe)] = cand_count
                    break

    for full_fe, dep_counts in imputed_fes.items():
        dep, count = dep_counts.most_common(1)[0]
        _, arg = dep_slots[dep]
        position = get_dep_position(dep)
        arg_candidates[position].append((dep, full_fe, arg, 'deps'))

    for i_dep, frame_counts in imputed_deps.items():
        full_fe, count = frame_counts.most_common(1)[0]
        position = get_dep_position(i_dep)
        _, arg = frame_slots[full_fe]
        if position == 'NA':
            if 'prep' not in arg_candidates:
                arg_candidates['prep'].append((i_dep, full_fe, arg, 'frames'))

    final_args = {}
    for position, candidate_args in arg_candidates.items():
        if len(candidate_args) > 1:
            a = tiebreak_arg(candidate_args, pred_start, pred_end)
            # Here we only take the first 3.
            final_args[position] = a[:3]
        elif len(candidate_args) == 1:
            final_args[position] = candidate_args[0][:3]
        else:
            final_args[position] = None

    return final_args


def hash_one_doc(docid, events, entities, event_emb_vocab, word_emb_vocab,
                 typed_event_vocab, frame_args, dep_frames):
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
        pid = event_emb_vocab.get_index(typed_event_vocab.get_pred_rep(event),
                                        None)
        fid = event_emb_vocab.get_index(event.get('frame'), None)
        mapped_args = impute_args(event, frame_args, dep_frames)

        full_args = {}
        for slot, arg_info in mapped_args.items():
            if arg_info is None:
                full_args[slot] = {}
            else:
                dep, full_fe, arg = arg_info
                full_args[slot] = hash_arg(
                    arg, dep, full_fe, event_emb_vocab, word_emb_vocab,
                    typed_event_vocab, entity_represents
                )

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
    frame_args, frame_counts = load_frame_map(params.frame_arg_map)
    dep_frames, dep_counts = load_frame_map(params.dep_frame_map)

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
                typed_event_vocab, frame_args, dep_frames)
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
        frame_arg_map = Unicode(
            help='Mapping from predicate arguments to frame elements.'
        ).tag(config=True)
        dep_frame_map = Unicode(
            help='Mapping from frame elements to arguments.').tag(config=True)
        raw_data = Unicode(help='The dataset to hash.').tag(config=True)
        output_path = Unicode(
            help='Output path of the hashed data.').tag(config=True)


    from event.util import load_command_line_config, set_basic_log

    set_basic_log()

    cl_conf = load_command_line_config(sys.argv[2:])
    conf = PyFileConfigLoader(sys.argv[1]).load_config()
    conf.merge(cl_conf)

    hash_params = HashParam(config=conf)

    hash_data(hash_params)
