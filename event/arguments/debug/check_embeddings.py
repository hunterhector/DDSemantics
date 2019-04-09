from traitlets import (
    Unicode
)
from traitlets.config import Configurable
from event import util

import sys
import numpy as np
from scipy.spatial.distance import cosine
import heapq
from collections import defaultdict
from gensim.models import KeyedVectors


def load_embedding(vocab_path, wv_path):
    vocab = {}
    inverted = []
    with open(vocab_path) as vocab_file:
        index = 0
        for line in vocab_file:
            w, _ = line.split(' ')
            vocab[w] = index
            inverted.append(w)
            index += 1

    wv = []
    if wv_path.endswith('.bin'):
        # Load word2vec model.
        wv_gensim = KeyedVectors.load_word2vec_format(wv_path, binary=True)
        for _, w in enumerate(inverted):
            wv.append(wv_gensim[w])
    else:
        # Load regular model.
        wv = np.load(wv_path)

    return vocab, inverted, wv


def check_embeddings(vocab_path, wv_path):
    vocab, inverted, wv = load_embedding(vocab_path, wv_path)

    while True:
        word1 = input("Input 1:")
        word2 = input("Input 2:")

        try:
            index1 = vocab[word1]
            index2 = vocab[word2]

            v1 = wv[index1]
            v2 = wv[index2]

            print(word1, index1)
            print(word2, index2)

            print("Similarity between is %.5f." % (1 - cosine(v1, v2)))

            v1_most_ty_type = most_similar(v1, wv, inverted)
            print("Most similar for ", word1)
            for t, v1_most in v1_most_ty_type.items():
                print("Type: %s" % t)
                for score, word in v1_most:
                    print(word, 1 - score)
            print("")

            v2_most_by_type = most_similar(v2, wv, inverted)
            print("Most similar for ", word2)
            for t, v2_most in v2_most_by_type.items():
                print("Type: %s" % t)
                for score, word in v2_most:
                    print(word, 1 - score)
            print("")

        except KeyError:
            print("Words not found")


def most_similar(vector1, all_vector, inverted, k=10):
    heap_by_type = defaultdict(list)

    for index, v in enumerate(all_vector):
        term = inverted[index].replace("-lrb-", '(').replace("-rrb-", ')')
        word_parts = term.split('-')

        if len(word_parts) > 1:
            word = '-'.join(word_parts[:-1])
            t = word_parts[-1].lower()
            if t.startswith('prep_'):
                t = 'prep'
        else:
            word = word_parts[0]
            t = 'frame'

        score = cosine(vector1, v)
        heapq.heappush(heap_by_type[t], (score, word))

    most_sim_by_type = defaultdict(list)
    for t, h in heap_by_type.items():
        if len(h) < k:
            continue

        for _ in range(k):
            most_sim_by_type[t].append(heapq.heappop(h))

    return most_sim_by_type


if __name__ == '__main__':
    class Debug(Configurable):
        wv_path = Unicode(help='Saved Embedding Vectors').tag(config=True)
        vocab_path = Unicode(help='Saved Embedding Vocab').tag(config=True)


    conf = util.load_command_line_config(sys.argv[1:])
    para = Debug(config=conf)

    check_embeddings(para.vocab_path, para.wv_path)
