from collections import defaultdict, Counter
import os

num_arg_fields = 5


def write_sent(sentence, output_path):
    pass


def parse_doc(doc, output_path):
    sentence = []
    with open(doc) as data:
        for line in data:
            line = line.strip()
            if line.startswith("#"):
                continue

            if not line:
                write_sent(sentence, output_path)
                sentence.clear()
                continue

            parts = line.split("\t")
            pred = parts[0]
            frame = parts[2]
            arg_fields = parts[3:-1]

            for v in [arg_fields[x:x + num_arg_fields] for x in
                      range(0, len(arg_fields), num_arg_fields)]:
                syn_role, frame_role, entity, text, resolvable = v
                sentence.append()


def filter_by_count(path, min_count):
    counter = Counter()
    with open(path) as vocab:
        for line in vocab:
            token, count = line.strip().split('\t')
            count = int(count)
            if count >= min_count:
                counter[token] = count
    return counter.most_common()


def filter_by_rank(path, top_k):
    counter = Counter()
    with open(path) as vocab:
        for line in vocab:
            token, count = line.strip().split('\t')
            counter[token] = int(count)

    return counter.most_common(top_k)


def filter_vocab(vocab_dir, top_num_prep=150, min_token_count=500):
    counters = {}

    counters['predicate_min_%d' % min_token_count] = filter_by_count(
        os.path.join(vocab_dir, 'predicate.vocab'), min_token_count)
    counters['argument_min_%d' % min_token_count] = filter_by_count(
        os.path.join(vocab_dir, 'argument.vocab'), min_token_count)
    counters['preposition_top_%d' % top_num_prep] = filter_by_rank(
        os.path.join(vocab_dir, 'preposition.vocab'), top_num_prep)

    for key, counter in counters.items():
        with open(os.path.join(vocab_dir, key + '.vocab'), 'w') as out:
            for key, value in counter:
                out.write('{}\t{}\n'.format(key, value))


def get_vocab_count(data_path):
    vocab_count = defaultdict(Counter)

    line_count = 0
    with open(data_path) as data:
        for line in data:
            line = line.strip().lower()
            if line.startswith("#"):
                continue
            if not line:
                continue

            parts = line.split("\t")
            predicate = parts[0]

            vocab_count['predicate'][predicate] += 1

            arg_fields = parts[3:-1]

            for v in [arg_fields[x:x + num_arg_fields] for x in
                      range(0, len(arg_fields), num_arg_fields)]:
                syn_role, frame_role, entity, text, resolvable = v
                vocab_count['argument'][text] += 1
                if syn_role.startswith('prep'):
                    vocab_count['preposition'][syn_role] += 1

            line_count += 1

            if line_count % 10000 == 0:
                print('\rRead {} lines.'.format(line_count), end='')

    print('\n')

    return vocab_count


def main(argv):
    event_tsv = argv[1]
    vocab_dir = argv[2]

    if not os.path.exists(vocab_dir):
        os.makedirs(vocab_dir)

    # Count vocabulary first.
    if not os.path.exists(os.path.join(vocab_dir, 'predicate.vocab')):
        print("Loading vocabulary counts.")
        vocab_count = get_vocab_count(event_tsv)
        for key, counter in vocab_count.items():
            raw_vocab_path = os.path.join(vocab_dir, key + '.vocab')
            with open(raw_vocab_path, 'w') as out:
                for key, value in counter.most_common():
                    out.write('{}\t{}\n'.format(key, value))
        print("Done vocabulary counting.")

    # Now filter the vocabulary.
    print("Filtering vocabulary.")
    filter_vocab(vocab_dir)
    print("Done filtering.")


if __name__ == '__main__':
    import sys

    main(sys.argv)
