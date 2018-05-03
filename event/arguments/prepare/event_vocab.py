from collections import defaultdict, Counter
import os
import gzip
import json
import pickle
from gensim.models.word2vec import Word2Vec
from json.decoder import JSONDecodeError


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
    if not role == 'NA':
        return text + "-" + role
    return None


def make_fe(frame, fe):
    return frame + ',' + fe


def create_sentences(doc, output_path, lookups, oovs, include_frame=False):
    doc_count = 0
    event_count = 0

    with gzip.open(doc) as data, open(output_path, 'w') as out:
        for line in data:
            try:
                doc_info = json.loads(line)
            except JSONDecodeError:
                continue

            sentence = []

            for event in doc_info['events']:
                event_count += 1

                pred = get_word(event['predicate'], 'predicate', lookups, oovs)
                frame_name = event.get('frame')
                frame = get_word(frame_name, 'frame', lookups, oovs)

                sentence.append(make_predicate(pred))

                if include_frame:
                    sentence.append(frame)

                for arg in event['arguments']:
                    syn_role = arg['dep']
                    fe = arg['feName']

                    text = arg['representText']

                    arg_text = get_word(text, 'argument', lookups, oovs)

                    arg_role = make_arg(arg_text, syn_role)
                    if arg_role is not None:
                        sentence.append(arg_text + "-" + syn_role)

                    if include_frame and not fe == 'NA':
                        fe_name = get_word(make_fe(frame_name, fe),
                                           'fe', lookups, oovs)
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
                    if eid in represent_by_id:
                        arg_text = represent_by_id[eid]
                    else:
                        arg_text = arg['text']

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


def train_event_vectors(sentence_file, vector_out_base, window_size):
    class Sentences():
        def __init__(self, sent_file):
            self.sent_file = sent_file

        def __iter__(self):
            with open(self.sent_file) as doc:
                for line in doc:
                    yield line.split()

    model = Word2Vec(Sentences(sentence_file), workers=10, size=300,
                     window=window_size, sample=1e-4, negative=10)
    model.save(vector_out_base + '.pickle')
    model.wv.save_word2vec_format(vector_out_base + '.vectors',
                                  fvocab=vector_out_base + '.voc')


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

        print("Loaded {} types for {}".format(len(lookups[vocab_type]),
                                              vocab_type))
    return lookups, oov_words


def main(argv):
    event_data = argv[1]
    vocab_dir = argv[2]
    data_out = argv[3]
    embedding_dir = argv[4]

    if not os.path.exists(data_out):
        os.makedirs(data_out)

    event_sentence_out = os.path.join(data_out, 'event_sentences.txt')
    frame_sentence_out = os.path.join(data_out, 'event_frame_sentences.txt')

    if not os.path.exists(vocab_dir):
        os.makedirs(vocab_dir)
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

    if not os.path.exists(event_sentence_out):
        print("Creating event sentences without frames.")
        create_sentences(event_data, event_sentence_out, lookups, oovs)

    if not os.path.exists(frame_sentence_out):
        print("Creating event sentences with frames.")
        create_sentences(event_data, frame_sentence_out, lookups, oovs,
                         include_frame=True)

    if not os.path.exists(embedding_dir):
        os.makedirs(embedding_dir)
    event_emb_out = os.path.join(embedding_dir, 'event_embeddings')
    if not os.path.exists(event_emb_out):
        print("Training embedding for event sentences.")
        train_event_vectors(event_sentence_out, event_emb_out, window_size=10)

    event_frame_emb_out = os.path.join(embedding_dir, 'event_frame_embeddings')
    if not os.path.exists(event_frame_emb_out):
        print("Training embedding for frame event sentences.")
        # Use a larger window for longer frame sentences.
        train_event_vectors(frame_sentence_out, event_frame_emb_out,
                            window_size=15)


if __name__ == '__main__':
    import sys

    main(sys.argv)
