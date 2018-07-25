import os
from gensim.models.word2vec import Word2Vec
import glob


def train_event_vectors(sentence_dir, vector_out_base, window_size):
    class Sentences():
        def __init__(self, sentence_dir):
            self.sentence_dir = sentence_dir

        def __iter__(self):
            for sent_file in glob.glob(self.sentence_dir + '/*.txt'):
                with open(sent_file) as doc:
                    # print("Processing ", sent_file)
                    for line in doc:
                        yield line.split()

    model = Word2Vec(Sentences(sentence_dir), workers=10, size=300,
                     window=window_size, sample=1e-4, negative=10)
    model.save(vector_out_base + '.pickle')
    model.wv.save_word2vec_format(vector_out_base + '.vectors',
                                  fvocab=vector_out_base + '.voc')


def main(event_sentence_dir, event_emb_out_base):
    if not os.path.exists(event_emb_out_base + '.vectors'):
        print("Training embeddings.")

        embedding_dir = os.path.dirname(event_emb_out_base)
        if not os.path.exists(embedding_dir):
            os.makedirs(embedding_dir)

        train_event_vectors(event_sentence_dir, event_emb_out_base, window_size=10)


if __name__ == '__main__':
    import sys

    main(sys.argv[1], sys.argv[2])
