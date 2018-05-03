from event.io.readers import EventAsArgCloze
import gzip
from event.arguments.prepare.event_vocab import (
    make_predicate,
    make_arg,
    make_fe,
)
from event.arguments import consts, util
from collections import defaultdict, Counter


def get_context(word_vocab, context, unk_word_index):
    left, right = context

    return [word_vocab.get(word, unk_word_index) for word in left], \
           [word_vocab.get(word, unk_word_index) for word in right]


def load_vocab(vocab_file):
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


def get_args(event, frame_args, arg_frames, frame_counts, dep_counts):
    args = event['arguments']

    final_args = {
        'subj': [],
        'obj': [],
        'prep': [],
    }

    dep_slots = {}
    frame_slots = {}

    predicate = util.remove_neg(event.get('predicate'))
    frame = event.get('frame', 'NA')

    # print('Number of args: ', len(args))
    # print("Frame", event.get('frame'))
    print("BEFORE:")
    [print(a) for a in args]

    def get_arg_content(arg):
        content = {}
        for k, v in arg.items():
            if not (k == 'dep' or k == 'fe'):
                content[k] = v
        return content

    for arg in args:
        dep = arg.get('dep', 'NA')
        fe = arg.get('fe', 'NA')

        if not dep == 'NA':
            dep_slots[dep] = ((frame, fe), get_arg_content(arg))

        if not fe == 'NA':
            frame_slots[(frame, fe)] = (dep, get_arg_content(arg))

    imputed_fes = defaultdict(Counter)
    for dep, (full_fe, arg) in dep_slots.items():
        position = dep.split('_')[0]
        if full_fe[1] == 'NA':
            candidates = arg_frames.get((predicate, dep), None)
            no_impute = dep.startswith('prep_')

            if not no_impute and candidates:
                for cand_frame, cand_fe, cand_count in candidates:
                    if (cand_frame, cand_fe) not in frame_slots:
                        imputed_fes[(cand_frame, cand_fe)][dep] = cand_count
                        break
                else:
                    no_impute = True

            if no_impute:
                # No impute can be found, or we do not trust the imputation.
                # In this case, we place an empty FE name here.
                final_args[position].append((dep, None, arg, 'no_impute'))
        else:
            final_args[position].append((dep, full_fe, arg, 'origin'))

    imputed_deps = defaultdict(Counter)
    for (frame, fe), (dep, arg) in frame_slots.items():
        if dep == 'NA':
            candidates = frame_args.get((frame, fe), None)
            for pred, dep, cand_count in candidates:
                if dep not in dep_slots:
                    imputed_deps[dep][(frame, fe)] = cand_count
                    break

    for full_fe, dep_counts in imputed_fes.items():
        # if len(dep_counts) > 1:
        #     print("{} can be imputed by {}"
        #           " deps".format(full_fe, len(dep_counts)))
        #     print(dep_counts)
        dep, count = dep_counts.most_common(1)[0]
        _, arg = dep_slots[dep]
        position = dep.split('_')[0]

        final_args[position].append((dep, full_fe, arg, 'deps'))

    for i_dep, frame_counts in imputed_deps.items():
        # if len(frame_counts) > 1:
        #     print("{} can be imputed by {} "
        #           "frames".format(i_dep, len(frame_counts)))
        #     print(frame_counts)
        full_fe, count = frame_counts.most_common(1)[0]
        position = i_dep.split('_')[0]
        _, arg = frame_slots[full_fe]
        final_args[position].append((i_dep, full_fe, arg, 'frames'))

    print("AFTER:")

    for position, args in final_args.items():
        print(position)
        for a in args:

            print(a)
        if len(args) > 1:
            print("More than one item this slot.")

    input("Wait")

    return final_args


def hash_data(args):
    data_path = args.raw_data
    event_vocab_file = args.event_vocab
    word_vocab_file = args.word_vocab

    frame_args, frame_counts = load_frame_map(args.frame_arg_map)
    dep_frames, dep_counts = load_frame_map(args.dep_frame_map)

    reader = EventAsArgCloze()

    event_vocab = load_vocab(event_vocab_file)

    word_vocab = None
    if word_vocab_file:
        word_vocab = load_vocab(word_vocab_file)
        unk_word_index = word_vocab[consts.unk_word]

    with gzip.open(data_path) as data_in:
        for docid, events, eid_count in reader.read_events(data_in):
            for event in events:
                pred = make_predicate(event['predicate'])
                print("Event:", pred)

                pid = event_vocab[pred]

                frame_name = event.get('frame', 'NA')

                args = get_args(event, frame_args, dep_frames, frame_counts,
                                dep_counts)

                context = event['predicate_context']

                if word_vocab:
                    context = get_context(word_vocab, context, unk_word_index)


if __name__ == '__main__':
    from event.util import OptionPerLineParser

    parser = OptionPerLineParser(description='Argument Task Hasher.',
                                 fromfile_prefix_chars='@')
    parser.add_argument('--event_vocab', type=str, help='Event Vocabulary.')
    parser.add_argument('--event_fe_vocab', type=str,
                        help='Event and Frame Vocabulary.')
    parser.add_argument('--word_vocab', type=str,
                        help='Vocabulary for normal words.')
    parser.add_argument('--raw_data', type=str, help='The dataset to hash.')
    parser.add_argument('--frame_arg_map', type=str,
                        help='Mapping from predicate '
                             'arguments to frame elements.')
    parser.add_argument('--dep_frame_map', type=str,
                        help='Mapping from frame elements to arguments.')

    hash_data(parser.parse_args())
