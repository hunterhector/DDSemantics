from collections import defaultdict, Counter
import os
import gzip
import json
import pickle
from gensim.models.word2vec import Word2Vec


def get_word(word, key, lookups, oovs):
    if not word:
        return oovs[key]

    if word in lookups[key]:
        return word
    else:
        return oovs[key]


def create_sentences(doc, output_path, lookups, oovs, include_frame=False):
    doc_count = 0
    event_count = 0

    with gzip.open(doc) as data, open(output_path, 'w') as out:
        for line in data:
            doc_info = json.loads(line)

            sentence = []

            for event in doc_info['events']:
                event_count += 1

                pred = get_word(event['predicate'], 'predicate', lookups, oovs)
                frame_name = event.get('frame', None)
                frame = get_word(frame_name, 'frame', lookups, oovs)

                sentence.append(pred.lower() + "-pred")

                if include_frame:
                    sentence.append(frame)

                for arg in event['arguments']:
                    syn_role = arg['dep']
                    text = arg['representText']

                    arg_text = get_word(text, 'argument', lookups, oovs)

                    if not syn_role == 'NA':
                        sentence.append(arg_text + "-" + syn_role)

                    if include_frame and frame_name:
                        fe_name = get_word(frame_name + ',' + arg['feName'],
                                           'fe', lookups, oovs)
                        sentence.append(fe_name)

            doc_count += 1

            out.write(' '.join(sentence) + '\n')

            if event_count % 1000 == 0:
                print('\rCreated sentences for {} documents, '
                      '{} events.'.format(doc_count, event_count), end='')
    print('\n')


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

    limits = [8000000, 40000000, 100000000]

    with gzip.open(data_path) as data:
        for line in data:
            doc_info = json.loads(line)

            for event in doc_info['events']:
                event_count += 1

                predicate = event['predicate']
                vocab_counters['predicate'][predicate] += 1

                for arg in event['arguments']:
                    fe_name = arg['feName']
                    syn_role = arg['dep']
                    arg_text = arg['representText']

                    vocab_counters['argument'][arg_text] += 1

                    if not fe_name == 'NA':
                        vocab_counters['fe'][
                            event['frame'] + ',' + fe_name] += 1

                    if syn_role.startswith('prep'):
                        vocab_counters['preposition'][syn_role] += 1

            doc_count += 1
            if doc_count % 1000 == 0:
                print('\rCounted vocab for {} events in '
                      '{} docs.'.format(event_count, doc_count), end='')

            if event_count >= limits[0]:
                yield event_count, vocab_counters

    print('\n')

    return vocab_counters


def train_event_vectors(sentence_file, vector_out_base):
    class Sentences():
        def __init__(self, sent_file):
            self.sent_file = sent_file

        def __iter__(self):
            with open(self.sent_file) as doc:
                for line in doc:
                    yield line.split()

    model = Word2Vec(Sentences(sentence_file), workers=10)
    model.save(vector_out_base + '.pickle')
    model.wv.save_word2vec_format(vector_out_base + '.vectors',
                                  fvocab=vector_out_base + '.vocab')


def main(argv):
    event_data = argv[1]
    vocab_dir = argv[2]
    data_out = argv[3]

    rebuild = len(argv) > 4

    if not os.path.exists(data_out):
        os.makedirs(data_out)

    event_sentence_out = os.path.join(data_out, 'event_sentences.txt')
    frame_sentence_out = os.path.join(data_out, 'event_frame_sentences.txt')

    # Count vocabulary first.
    if rebuild:
        if not os.path.exists(vocab_dir):
            os.makedirs(vocab_dir)
        print("Loading vocabulary counts.")
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

        print("Creating event sentences without frames.")
        create_sentences(event_data, event_sentence_out, lookups, oovs)

        print("Creating event sentences with frames.")
        create_sentences(event_data, frame_sentence_out, lookups, oovs,
                         include_frame=True)

    train_event_vectors(event_sentence_out,
                        os.path.join(vocab_dir, 'event_embeddings'))
    train_event_vectors(frame_sentence_out,
                        os.path.join(vocab_dir, 'event_frame_embeddings'))


if __name__ == '__main__':
    import sys

    main(sys.argv)
