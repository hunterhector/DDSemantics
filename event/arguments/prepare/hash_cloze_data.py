from event.io.readers import EventReader
import gzip
from event.arguments.prepare.event_vocab import (
    make_predicate,
    make_arg,
    make_fe,
    load_vocab,
    get_word,
)
from event.arguments import consts, util
from collections import defaultdict, Counter
import json
from traitlets import (
    Unicode
)
from traitlets.config.loader import PyFileConfigLoader
from traitlets.config import Configurable
import sys
import logging


def hash_context(word_vocab, context):
    left, right = context

    unk_word_index = word_vocab[consts.unk_word]

    return [word_vocab.get(word, unk_word_index) for word in left], \
           [word_vocab.get(word, unk_word_index) for word in right]


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
                to_pred, to_arg = role_info.split(',')
                fmap[from_pred, from_arg].append((to_pred, to_arg, int(count)))
            counts[from_pred] = from_count
    return fmap, counts


def get_arg_content(arg_info):
    content = {}
    for k, v in arg_info.items():
        if not (k == 'dep' or k == 'fe'):
            content[k] = v

    return content


def tiebreak_arg(tied_args, pred_start, pred_end):
    top_index = 0
    priority = (3, float('inf'))

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

        this_piority = (num_emtpy, dist)

        if this_piority < priority:
            top_index = i
            priority = this_piority

    return tied_args[top_index]


def hash_arg_role(arg_text, dep, event_vocab, oovs):
    arg_role = make_arg(arg_text, dep)

    if arg_role in event_vocab:
        arg_id = event_vocab[arg_role]
    else:
        arg_role = make_arg(oovs['argument'], dep)
        if arg_role in event_vocab:
            arg_id = event_vocab[arg_role]
        else:
            arg_role = make_arg(oovs['argument'], oovs['preposition'])
            arg_id = event_vocab[arg_role]

    return arg_id


def hash_arg(arg, event_vocab, word_vocab, lookups, oovs):
    if arg is None:
        # Empty argument case.
        return {
        }
    else:
        dep, full_fe, content, source = arg

        if dep.startswith('prep'):
            dep = get_word(dep, 'preposition', lookups, oovs)
        arg_text = get_word(content['represent'], 'argument', lookups, oovs)

        arg_id = hash_arg_role(arg_text, dep, event_vocab, oovs)

        if full_fe is not None:
            frame, fe = full_fe
            fe_name = get_word(make_fe(frame, fe), 'fe', lookups, oovs)
            if fe_name in event_vocab:
                fe_id = event_vocab[fe_name]
            else:
                fe_id = event_vocab[oovs['fe']]
        else:
            # Treat empty frame element as UNK.
            fe_id = event_vocab[oovs['fe']]

        hashed_context = hash_context(word_vocab, content['arg_context'])

        return {
            'arg_role': arg_id,
            'fe': fe_id,
            'context': hashed_context,
            'entity_id': content['entity_id'],
            'resolvable': content['resolvable'],
            'text': arg_text,
            'dep': dep,
        }


def get_args(event, frame_args, arg_frames):
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

    def get_position(dep):
        p = dep.split('_')[0]
        if p == 'iobj':
            # 'iobj' is not seen frequently, we put it to the 'prep' slot.
            p = 'prep'
        return p

    for arg in args:
        dep = arg.get('dep', 'NA')
        fe = arg.get('fe', 'NA')

        if not dep == 'NA' and get_position(dep) not in arg_candidates:
            # If dep is an known but not in our target list, ignore them.
            continue

        if not dep == 'NA':
            dep_slots[dep] = ((frame, fe), get_arg_content(arg))

        if not fe == 'NA':
            frame_slots[(frame, fe)] = (dep, get_arg_content(arg))

    imputed_fes = defaultdict(Counter)
    for dep, (full_fe, arg) in dep_slots.items():
        position = get_position(dep)

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
            candidates = frame_args.get((frame, fe), [])
            for pred, dep, cand_count in candidates:
                if dep not in dep_slots:
                    imputed_deps[dep][(frame, fe)] = cand_count
                    break

    for full_fe, dep_counts in imputed_fes.items():
        dep, count = dep_counts.most_common(1)[0]
        _, arg = dep_slots[dep]
        position = get_position(dep)
        arg_candidates[position].append((dep, full_fe, arg, 'deps'))

    for i_dep, frame_counts in imputed_deps.items():
        full_fe, count = frame_counts.most_common(1)[0]
        position = get_position(i_dep)
        _, arg = frame_slots[full_fe]
        arg_candidates[position].append((i_dep, full_fe, arg, 'frames'))

    final_args = {}
    for position, candidate_args in arg_candidates.items():
        if len(candidate_args) > 1:
            a = tiebreak_arg(candidate_args, pred_start, pred_end)
            final_args[position] = a

        elif len(candidate_args) == 1:
            final_args[position] = candidate_args[0]
        else:
            final_args[position] = None

    return final_args


def read_entity_features(entities, lookups, oovs):
    hashed_entities = {}

    for eid, entity in entities.items():
        entity_head = get_word(
            entity['representEntityHead'], 'argument', lookups, oovs)
        hashed_entities[eid] = {
            'features': entity['features'],
            'entity_head': entity_head,
        }
    return hashed_entities


def get_predicate(predicate, lookups, oovs):
    return make_predicate(
        get_word(predicate, 'predicate', lookups, oovs)
    )


def hash_one_doc(docid, events, entities, event_vocab, word_vocab, lookups,
                 oovs, frame_args, dep_frames):
    hashed_doc = {
        'docid': docid,
        'events': [],
    }

    read_entity_features(entities, lookups, oovs)
    hashed_doc['entities'] = read_entity_features(entities, lookups, oovs)

    for event in events:
        pid = event_vocab[get_predicate(event['predicate'], lookups, oovs)]
        frame_name = event.get('frame', 'NA')
        fid = event_vocab.get(frame_name, -1)
        mapped_args = get_args(event, frame_args, dep_frames)

        full_args = {}
        for slot, arg in mapped_args.items():
            full_args[slot] = hash_arg(
                arg, event_vocab, word_vocab, lookups, oovs)

        context = hash_context(word_vocab, event['predicate_context'])

        hashed_doc['events'].append({
            'predicate': pid,
            'frame': fid,
            'context': context,
            'sentence_id': event['sentence_id'],
            'args': full_args,
        })
    return hashed_doc


def hash_data(params):
    frame_args, frame_counts = load_frame_map(params.frame_arg_map)
    dep_frames, dep_counts = load_frame_map(params.dep_frame_map)

    event_vocab = load_emb_vocab(params.event_vocab)
    word_vocab = load_emb_vocab(params.word_vocab)

    lookups, oovs = load_vocab(params.component_vocab_dir)

    reader = EventReader()

    doc_count = 0
    event_count = 0

    print("{}: Start hashing".format(util.get_time()))
    with gzip.open(params.raw_data) as data_in, gzip.open(
            params.output_path, 'w') as data_out:
        for docid, events, entities in reader.read_events(data_in):
            hashed_doc = hash_one_doc(docid, events, entities, event_vocab,
                                      word_vocab, lookups, oovs, frame_args,
                                      dep_frames)
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
