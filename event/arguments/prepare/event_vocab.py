from collections import defaultdict, Counter
import os
import gzip
import json
import pickle
from json.decoder import JSONDecodeError
import logging
from event.arguments import consts


def get_word(word, key, lookups, oovs):
    if not word:
        return oovs[key]

    if word in lookups[key]:
        return word
    else:
        return oovs[key]


def make_predicate(text):
    return text.lower() + "-pred"


def make_arg(text, role):
    if role == 'NA':
        return text + "-" + consts.unk_dep
    else:
        return text + "-" + role


def make_fe(frame, fe):
    return frame + ',' + fe


def create_sentences(doc, output_path, lookups, oovs, include_frame=False):
    if include_frame:
        print("Adding frames to sentences.")

    doc_count = 0
    event_count = 0

    with gzip.open(doc) as data, open(output_path, 'w') as out:
        for line in data:
            try:
                doc_info = json.loads(line)
            except JSONDecodeError:
                continue

            sentence = []

            represent_by_id = {}
            for entity in doc_info['entities']:
                eid = entity['entityId']
                represent = entity['representEntityHead']
                represent_by_id[eid] = represent

            for event in doc_info['events']:
                event_count += 1

                pred = get_word(event['predicate'], 'predicate', lookups, oovs)
                frame_name = event.get('frame')
                frame = get_word(frame_name, 'frame', lookups, oovs)

                sentence.append(make_predicate(pred))

                if include_frame:
                    sentence.append(frame)

                for arg in event['arguments']:
                    dep = arg['dep']
                    if (not include_frame) and dep == 'NA':
                        continue

                    fe = arg['feName']

                    eid = arg['entityId']
                    text = represent_by_id.get(eid, arg['text'])
                    arg_text = get_word(text, 'argument', lookups, oovs)

                    if dep.startswith('prep'):
                        dep = get_word(
                            dep, 'preposition', lookups, oovs)

                    arg_role = make_arg(arg_text, dep)

                    sentence.append(arg_role)

                    if include_frame:
                        fe_name = get_word(
                            make_fe(frame_name, fe), 'fe', lookups, oovs
                        )
                        sentence.append(fe_name)

            doc_count += 1

            out.write(' '.join(sentence) + '\n')

            if event_count % 1000 == 0:
                print('\rCreated sentences for {} documents, '
                      '{} events.'.format(doc_count, event_count), end='')

    print('\rCreated sentences for {} documents, '
          '{} events.\n'.format(doc_count, event_count), end='')


def filter_by_count(counter, min_count):
    return [(key, count) for key, count in counter.most_common() if
            count >= min_count]


def filter_vocab(vocab_dir, vocab_counters,
                 top_num_prep=150,
                 min_token_count=500,
                 min_fe_count=50):
    filtered_vocab = {
        'predicate_min_%d' % min_token_count:
            filter_by_count(vocab_counters['predicate'], min_token_count),
        'argument_min_%d' % min_token_count:
            filter_by_count(vocab_counters['argument'], min_token_count),
        'preposition_top_%d' % top_num_prep:
            vocab_counters['preposition'].most_common(top_num_prep),
        'fe_min_%d' % min_fe_count:
            filter_by_count(vocab_counters['fe'], min_fe_count),
    }

    # Frames are retained if their frame elements are retained.
    fe_counter = filtered_vocab['fe_min_%d' % min_fe_count]
    frame_counter = Counter()
    for full_fe, count in fe_counter:
        frame, fe = full_fe.split(',')
        frame_counter[frame] += count
    filtered_vocab['frame'] = frame_counter.most_common()

    lookups = {}
    oov_words = {}
    for key, counts in filtered_vocab.items():
        name = key.split('_')[0]

        oov = 'unk_' + name
        counts.insert(0, (oov, 0))

        lookups[name] = {}
        oov_words[name] = oov

        index = 0
        for term, _ in counts:
            lookups[name][term] = index
            index += 1

    for key, vocab in filtered_vocab.items():
        with open(os.path.join(vocab_dir, key + '.vocab'), 'w') as out:
            for token, count in vocab:
                out.write('{}\t{}\n'.format(token, count))

    return lookups, oov_words


def get_vocab_count(data_path):
    vocab_counters = defaultdict(Counter)

    doc_count = 0
    event_count = 0

    with gzip.open(data_path) as data:
        for line in data:
            doc_info = json.loads(line)

            represent_by_id = {}
            for entity in doc_info['entities']:
                eid = entity['entityId']
                represent = entity['representEntityHead']
                represent_by_id[eid] = represent

            for event in doc_info['events']:
                event_count += 1

                predicate = event['predicate']
                vocab_counters['predicate'][predicate] += 1

                for arg in event['arguments']:
                    fe_name = arg['feName']
                    syn_role = arg['dep']
                    eid = arg['entityId']
                    arg_text = represent_by_id.get(eid, arg['text'])

                    vocab_counters['argument'][arg_text] += 1

                    if not fe_name == 'NA':
                        vocab_counters['fe'][
                            make_fe(event['frame'], fe_name)
                        ] += 1

                    if syn_role.startswith('prep'):
                        vocab_counters['preposition'][syn_role] += 1

            doc_count += 1
            if doc_count % 1000 == 0:
                print('\rCounted vocab for {} events in '
                      '{} docs.'.format(event_count, doc_count), end='')

    print('\n')

    return vocab_counters


def load_vocab(vocab_dir):
    lookups = {}
    oov_words = {}

    for f in os.listdir(vocab_dir):
        if '_' in f and f.endswith('.vocab'):
            vocab_type = f.split('_')[0]
        elif f == 'frame.vocab':
            vocab_type = 'frame'
        else:
            continue

        lookups[vocab_type] = {}
        oov_words[vocab_type] = 'unk_' + vocab_type

        with open(os.path.join(vocab_dir, f)) as vocab_file:
            index = 0
            for line in vocab_file:
                word, count = line.strip().split('\t')
                lookups[vocab_type][word] = index
                index += 1

        logging.info("Loaded {} types for {}".format(len(lookups[vocab_type]),
                                                     vocab_type))
    return lookups, oov_words


def main(event_data, vocab_dir, sent_out, add_frame_word):
    if not os.path.exists(vocab_dir):
        os.makedirs(vocab_dir)

    sent_out_par = os.path.dirname(sent_out)
    if not os.path.exists(sent_out_par):
        print("Making diretory", sent_out_par)
        os.makedirs(sent_out_par)

    if not os.path.exists(os.path.join(vocab_dir, 'predicate.vocab')):
        print("Counting vocabulary.")
        vocab_counters = get_vocab_count(event_data)
        for key, counter in vocab_counters.items():
            raw_vocab_path = os.path.join(vocab_dir, key + '.vocab')
            with open(raw_vocab_path, 'w') as out:
                for key, value in counter.most_common():
                    out.write('{}\t{}\n'.format(key, value))
        print("Done vocabulary counting.")

        # Now filter the vocabulary.
        print("Filtering vocabulary.")
        lookups, oovs = filter_vocab(vocab_dir, vocab_counters)
        with open(os.path.join(vocab_dir, 'lookups.pickle'), 'wb') as out:
            pickle.dump(lookups, out)
        print("Done filtering.")
    else:
        print("Will not overwrite vocabulary, using existing.")
        lookups, oovs = load_vocab(vocab_dir)
        print("Done loading.")

    print("Creating event sentences")
    if not os.path.exists(sent_out):
        create_sentences(event_data, sent_out, lookups, oovs,
                         include_frame=add_frame_word)
    else:
        print("Sentence frame file exists, skipping.")


if __name__ == '__main__':
    from event.util import OptionPerLineParser

    parser = OptionPerLineParser(description='Event Vocabulary.',
                                 fromfile_prefix_chars='@')
    parser.add_argument('--vocab_dir', type=str, help='Vocabulary direcotry.')
    parser.add_argument('--input_data', type=str, help='Input data.')
    parser.add_argument('--sent_out', type=str, help='Sentence out file.')
    parser.add_argument('--add_frame_word', action='store_true',
                        help='Add frame word.')

    args = parser.parse_args()
    main(
        args.input_data, args.vocab_dir, args.sent_out, args.add_frame_word
    )
