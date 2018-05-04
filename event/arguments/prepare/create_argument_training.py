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


def hash_arg(arg, event_vocab, word_vocab, lookups, oovs):
    if arg is None:
        # Empty arguemnt case.
        return {
            'arg': -1,
            'fe': -1,
            'context': ([], []),
            'entity_id': -1,
            'resolvable': False,
            'sentence_id': -1,
        }
    else:
        dep, full_fe, content, source = arg

        arg_text = get_word(content['represent'], 'argument', lookups, oovs)
        arg_role = make_arg(arg_text, dep)
        if arg_role in event_vocab:
            arg_id = event_vocab[arg_role]
        else:
            arg_id = -1

        if full_fe is not None:
            frame, fe = full_fe
            fe_name = get_word(make_fe(frame, fe), 'fe', lookups, oovs)
            if fe_name in event_vocab:
                fe_id = event_vocab[fe_name]
            else:
                fe_id = -1
        else:
            # Use padding for this later.
            fe_id = -1

        hashed_context = hash_context(word_vocab, content['arg_context'])

        return {
            'arg': arg_id,
            'fe': fe_id,
            'context': hashed_context,
            'entity_id': content['entity_id'],
            'resolvable': content['resolvable'],
            'sentence_id': content['sentence_id'],
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
            # 'iobj' is not often, we put it to the 'prep' slot.
            p = 'prep'
        return p

    for arg in args:
        dep = arg.get('dep', 'NA')
        fe = arg.get('fe', 'NA')

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


def hash_one_doc(docid, events, entities, event_vocab, word_vocab, lookups,
                 oovs, frame_args, dep_frames):
    hashed_doc = {
        'docid': docid,
        'events': [],
        'entities': entities,
    }

    for event in events:
        pred = make_predicate(
            get_word(event['predicate'], 'predicate', lookups, oovs)
        )

        pid = event_vocab[pred]

        frame_name = event.get('frame', 'NA')

        if frame_name in event_vocab:
            fid = event_vocab[frame_name]
        else:
            fid = -1

        mapped_args = get_args(event, frame_args, dep_frames)

        full_args = {}
        for position, arg in mapped_args.items():
            full_args[position] = hash_arg(arg, event_vocab, word_vocab,
                                           lookups, oovs)

        context = hash_context(word_vocab, event['predicate_context'])

        hashed_doc['events'].append({
            'predicate': pid,
            'frame': fid,
            'context': context,
            'args': full_args,
        })
    return hashed_doc


def hash_data(cmd_args):
    frame_args, frame_counts = load_frame_map(cmd_args.frame_arg_map)
    dep_frames, dep_counts = load_frame_map(cmd_args.dep_frame_map)

    reader = EventReader()

    event_vocab = load_emb_vocab(cmd_args.event_vocab)
    word_vocab = load_emb_vocab(cmd_args.word_vocab)

    lookups, oovs = load_vocab(cmd_args.component_vocab_dir)

    doc_count = 0
    event_count = 0

    print("{}: Start hashing".format(util.get_time()))
    with gzip.open(cmd_args.raw_data) as data_in, gzip.open(
            cmd_args.output_path, 'w') as data_out:
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
    from event.util import OptionPerLineParser

    parser = OptionPerLineParser(description='Argument Task Hasher.',
                                 fromfile_prefix_chars='@')
    parser.add_argument('--event_vocab', type=str,
                        help='Event and Frame Vocabulary.')
    parser.add_argument('--word_vocab', type=str,
                        help='Vocabulary for normal words.')

    parser.add_argument('--component_vocab_dir', type=str,
                        help='Directory containing vocab for each component')

    parser.add_argument('--frame_arg_map', type=str,
                        help='Mapping from predicate '
                             'arguments to frame elements.')
    parser.add_argument('--dep_frame_map', type=str,
                        help='Mapping from frame elements to arguments.')

    parser.add_argument('--raw_data', type=str, help='The dataset to hash.')
    parser.add_argument('--output_path', type=str,
                        help='Output path of the hashed data.')

    hash_data(parser.parse_args())
