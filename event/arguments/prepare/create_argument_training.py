from event.io.readers import EventAsArgCloze
import gzip
from event.arguments.prepare.event_vocab import (
    make_predicate,
    make_arg,
    make_fe,
)
from event.arguments import consts
from collections import defaultdict


def get_context(word_vocab, context_str):
    words = context_str.split(' ')
    context = ([], [])
    current = context[0]
    for word in words:
        if word == consts.placeholder:
            current = context[1]
            continue
        current.append(word_vocab[word])
    return context


def load_vocab(vocab_file):
    vocab = {}
    with open(vocab_file) as infile:
        index = 0
        for line in infile:
            word, count = line.split('\t')
            vocab[word] = index
            index += 1

    return vocab


def load_frame_map(frame_map_file):
    pass


def hash_data(data_path, event_vocab_file, word_vocab_file, frame_map_file=None,
              hash_frames=False):
    reader = EventAsArgCloze()

    event_vocab = load_vocab(event_vocab_file)

    word_vocab = None
    if word_vocab_file:
        word_vocab = load_vocab(word_vocab_file)

    with gzip.open(data_path) as data_in:
        for docid, events, eid_count in reader.read_events(data_in):
            for event in events:
                pred = make_predicate(event['predicate'])
                pid = event_vocab[pred]

                frame_name = event.get('frame', 'NA')

                if hash_frames and not frame_name == 'NA':
                    fid = event_vocab[frame_name]

                context_str = event['predicate_context']

                if word_vocab:
                    context = get_context(word_vocab, context_str)

                args = event['arguments']
                for arg in args:
                    if not arg['dep'] == 'NA':
                        # We have to ignore this dependency.
                        # TODO try to recover from mapping
                        aid = event_vocab[
                            make_arg(arg['represent'], arg['dep'])
                        ]

                        if hash_frames and not frame_name == 'NA':
                            full_fe = make_fe(frame_name, arg['fe'])
                        feid = event_vocab[full_fe]

                input("Waiting.")


if __name__ == '__main__':
    from event.util import OptionPerLineParser

    parser = OptionPerLineParser(description='Argument Task Hasher.',
                                 fromfile_prefix_chars='@')
    parser.add_argument('--event_vocab', type=str, help='Event Vocabulary.')
    parser.add_argument('--event_fe_vocab', type=str,
                        help='Event and Frame Element Vocabulary.')
    parser.add_argument('--word_vocab', type=str,
                        help='Vocabulary for normal words.')
    parser.add_argument('--raw_data', type=str, help='The dataset to hash.')
    parser.add_argument('--frame_map', type=str,
                        help='Mapping from predicate '
                             'argument to frame element.')

    args = parser.parse_args()

    hash_data(args.raw_data, args.event_vocab, args.event_fe_vocab,
              args.word_vocab)
