from traitlets import (
    Unicode
)
from traitlets.config import Configurable
from event import util

import sys
import numpy as np
from scipy.spatial.distance import cosine
import heapq


def check_embeddings(vocab_path, wv_path):
    wv = np.load(wv_path)

    vocab = {}
    inverted = []
    with open(vocab_path) as vocab_file:
        index = 0
        for line in vocab_file:
            w, _ = line.split(' ')
            vocab[w] = index
            inverted.append(w)
            index += 1

    while True:
        word1 = input("Input 1:")
        word2 = input("Input 2:")

        try:
            index1 = vocab[word1]
            index2 = vocab[word2]

            v1 = wv[index1]
            v2 = wv[index2]

            print("Similarity between is %.5f." % (1 - cosine(v1, v2)))

            v1_most = most_similar(v1, wv)
            print("Most similar for ", word1)
            for score, index in v1_most:
                print(inverted[index], 1 - score)
            print("")

            v2_most = most_similar(v2, wv)
            print("Most similar for ", word2)
            for score, index in v2_most:
                print(inverted[index], 1 - score)
            print("")

        except KeyError:
            print("Words not found")


def most_similar(vector1, all_vector, k=10):
    h = []
    for index, v in enumerate(all_vector):
        score = cosine(vector1, v)
        if score == 0:
            continue
        heapq.heappush(h, (score, index))

    close = []
    for _ in range(k):
        close.append(heapq.heappop(h))

    return close


if __name__ == '__main__':
    class Debug(Configurable):
        wv_path = Unicode(help='Saved Embedding Vectors').tag(config=True)
        vocab_path = Unicode(help='Saved Embedding Vocab').tag(config=True)


    conf = util.load_command_line_config(sys.argv[1:])
    para = Debug(config=conf)

    check_embeddings(para.vocab_path, para.wv_path)
