import sys
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors


def pad_2d_list(in_list, pad_to_length, axis=0, pad_value=0):
    if axis == 0:
        pad_len = pad_to_length - len(in_list)
        pads = [pad_value] * len(in_list[0])
        return in_list + [pads for _ in range(pad_len)]
    elif axis == 1:
        return [l + [pad_value] * (pad_to_length - len(l)) for l in in_list]
    else:
        raise ValueError("Invalid axis {} for 2D list".format(axis))


def pad_last_axis(data, dim, pad_to_length, pad_value=0):
    if dim == 2:
        for l in data:
            l.extend([pad_value] * (pad_to_length - len(l)))
    elif dim == 3:
        for l in data:
            for ll in data:
                l.extend([pad_value] * (pad_to_length - len(ll)))


def read_glove_vectors(glove_path):
    glove_file = datapath(glove_path)
    tmp_file = get_tmpfile("word2vec.txt")
    glove2word2vec(glove_file, tmp_file)
    model = KeyedVectors.load_word2vec_format(tmp_file)
    return model
