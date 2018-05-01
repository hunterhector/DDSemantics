from event.io.readers import EventAsArgCloze
import gzip
from event.arguments.prepare.event_vocab import make_predicate, make_arg, \
    make_fe
from event.arguments import consts


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


def hash_data(data_path, event_vocab, fe_vocab, word_vocab):
    reader = EventAsArgCloze()

    with gzip.open(data_path) as data_in:
        for docid, events, eid_count in reader.read_events(data_in):
            for event in events:
                pred = make_predicate(event['predicate'])
                pid = event_vocab[pred]
                frame_name = event['frame']
                context_str = event['predicate_context']

                context = get_context(word_vocab, context_str)

                args = event['arguments']
                for arg in args:
                    # TODO Use the frame mapping to give a try.
                    if not arg['dep'] == 'NA':
                        # We have to ignore this dependency.
                        aid = event_vocab[
                            make_arg(arg['represent'], arg['dep'])
                        ]
                        full_fe = make_fe(frame_name, arg['fe'])
                        feid = fe_vocab[full_fe]

                input("Waiting.")


if __name__ == '__main__':
    import sys

    input_path = sys.argv[1]

    hash_data(input_path)
