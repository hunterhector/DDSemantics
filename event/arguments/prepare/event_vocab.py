from collections import defaultdict, Counter
import os

num_arg_fields = 5


def select_predicate(predicate, predicate_vocab, oov):
    text = oov

    forms = [predicate]

    if predicate.startswith('not_'):
        # Add the non-negation form
        non_negation = predicate[4:]
        forms.append(non_negation)
        # Add the base form
        parts = non_negation.split('_')
        if len(parts) > 1:
            forms.append(parts[0])
    else:
        parts = predicate.split('_')
        if len(parts) > 1:
            forms.append(parts[0])

    for form in forms:
        if form in predicate_vocab:
            text = form
            break

    return text


def parse_doc(doc, output_path, vocab_sets, include_frame=False):
    # TODO:
    # 1. the original doc need to be fixed (lower-cased)
    # 2. think about how to get "prep_to" for non-verbal
    # 3. use the representative entity for the entity itself.

    sentence = []
    line_count = 0

    oov = 'UNK'

    with open(doc) as data, open(output_path, 'w') as out:
        for line in data:
            line = line.strip()
            if line.startswith("#"):
                continue

            if not line:
                out.write(" ".join(sentence))
                out.write("\n")
                sentence.clear()
                continue

            parts = line.split("\t")
            pred = parts[0]
            frame = parts[2]
            arg_fields = parts[3:-1]

            pred = select_predicate(pred, vocab_sets['predicate'], oov)

            sentence.append(pred.lower() + "-pred")

            if include_frame:
                sentence.append(frame)

            for v in [arg_fields[x:x + num_arg_fields] for x in
                      range(0, len(arg_fields), num_arg_fields)]:
                syn_role, frame_role, entity, text, resolvable = v

                if syn_role == 'NA':
                    continue

                text = text.lower()
                arg_text = text if text in vocab_sets['argument'] else oov

                if syn_role == 'arg0':
                    syn_role = 'subj'
                if syn_role == 'arg1':
                    syn_role = 'obj'

                sentence.append(arg_text + "-" + syn_role)

            line_count += 1

            if line_count % 10000 == 0:
                print('\rRead {} lines.'.format(line_count), end='')

            if line_count > 1000:
                break
    print('\n')


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
    filtered_vocab = {
        'predicate_min_%d' % min_token_count: filter_by_count(
            os.path.join(vocab_dir, 'predicate.vocab'), min_token_count),
        'argument_min_%d' % min_token_count: filter_by_count(
            os.path.join(vocab_dir, 'argument.vocab'), min_token_count),
        'preposition_top_%d' % top_num_prep: filter_by_rank(
            os.path.join(vocab_dir, 'preposition.vocab'), top_num_prep)
    }

    for key, vocab in filtered_vocab.items():
        with open(os.path.join(vocab_dir, key + '.vocab'), 'w') as out:
            for token, count in vocab:
                out.write('{}\t{}\n'.format(token, count))

    return filtered_vocab


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
    data_out = argv[3]

    if not os.path.exists(vocab_dir):
        os.makedirs(vocab_dir)

    if not os.path.exists(data_out):
        os.makedirs(data_out)

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
    filtered_vocab = filter_vocab(vocab_dir)
    print("Done filtering.")

    vocab_sets = defaultdict(set)
    for key, vocab in filtered_vocab.items():
        key = key.split("_")[0]
        for token, count in vocab:
            vocab_sets[key].add(token)

    event_sentence_out = os.path.join(data_out, 'event_sentences.txt')
    print("Creating training data.")
    parse_doc(event_tsv, event_sentence_out, vocab_sets)


if __name__ == '__main__':
    import sys

    main(sys.argv)
