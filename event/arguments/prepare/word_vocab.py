import gzip
import json
from collections import Counter
import os
from gensim.models.word2vec import Word2Vec
from event.arguments import consts


def main(input_data, vocab_dir, embedding_dir):
    count_vocab(input_data, vocab_dir)
    embed(input_data, vocab_dir, embedding_dir)


def embed(input_data, vocab_dir, embedding_dir, min_count=50):
    if not os.path.exists(embedding_dir):
        os.makedirs(embedding_dir)

    raw_vocab = Counter()
    with open(os.path.join(vocab_dir, 'word_counts.vocab')) as counts:
        for line in counts:
            word, count = line.split('\t')
            raw_vocab[word] = int(count)

    print("Filtering.")
    filter_out = os.path.join(vocab_dir, 'word_counts_min_%d.vocab' % min_count)
    kept_words = set()

    with open(filter_out, 'w') as out:
        for word, count in raw_vocab.most_common():
            if count >= min_count:
                kept_words.add(word)
                out.write('{}\t{}\n'.format(word, count))

    class Data:
        def __init__(self, in_file):
            self.in_file = in_file

        def __iter__(self):
            with gzip.open(self.in_file) as doc:
                for line in doc:
                    for sent in json.loads(line)['sentences']:
                        words = sent.split()
                        yield [w if w in kept_words else consts.unk_word for w
                               in words]

    print("Start training embeddings.")
    emb_out_base = os.path.join(embedding_dir, 'word_embeddings')
    model = Word2Vec(Data(input_data), workers=10, size=300)
    model.save(emb_out_base + '.pickle')
    model.wv.save_word2vec_format(emb_out_base + '.vectors',
                                  fvocab=emb_out_base + '.voc')
    print("Done.")


def count_vocab(input_data, vocab_dir):
    if not os.path.exists(vocab_dir):
        os.makedirs(vocab_dir)

    out_file = os.path.join(vocab_dir, 'word_counts.vocab')
    if os.path.exists(out_file):
        print("Not counting, vocab file exists.")
        return

    counter = Counter()
    doc_count = 0
    word_count = 0
    with gzip.open(input_data) as data:
        for line in data:
            doc_info = json.loads(line)

            for sent in doc_info['sentences']:
                for word in sent.split():
                    counter[word] += 1
                    word_count += 1

            doc_count += 1
            if doc_count % 1000 == 0:
                print('\rCounted vocab for {} words in '
                      '{} docs.'.format(word_count, doc_count), end='')
    print('\nTotally {} words and {} documents.'.format(word_count, doc_count))

    with open(os.path.join(vocab_dir, 'word_counts.vocab'), 'w') as out:
        for word, count in counter.items():
            out.write('{}\t{}\n'.format(word, count))


if __name__ == '__main__':
    from event.util import OptionPerLineParser

    parser = OptionPerLineParser(description='Word Vocabulary.',
                                 fromfile_prefix_chars='@')
    parser.add_argument('--vocab_dir', type=str, help='Vocabulary direcotry.')
    parser.add_argument('--embedding_dir', type=str, help='Event Embedding.')
    parser.add_argument('--input_data', type=str, help='Input data.')
    args = parser.parse_args()
    main(args.input_data, args.vocab_dir, args.embedding_dir)
