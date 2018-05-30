import argparse
import logging
import sys
import os
import gzip
from traitlets.config import Configurable
from traitlets import (
    Unicode,
    Bool,
    Float,
    Integer,
    List
)


class OptionPerLineParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        if arg_line.startswith("#"):
            return []
        return arg_line.split()


def smart_open(path):
    if path.endswith('.gz'):
        return gzip.open(path)
    else:
        return open(path)


def ensure_dir(p):
    parent = os.path.dirname(p)
    if not os.path.exists(parent):
        os.makedirs(parent)


def tokens_to_sent(tokens, sent_start):
    sent = ""

    for token, span in tokens:
        if span[0] > len(sent) + sent_start:
            padding = ' ' * (span[0] - len(sent) - sent_start)
            sent += padding
        sent += token
    return sent


class DetectionParams(Configurable):
    model_name = Unicode(help='').tag(config=True)

    experiment_folder = Unicode(help='').tag(config=True)
    model_dir = Unicode(help='').tag(config=True)

    train_files = List(help='').tag(config=True)
    dev_files = List(help='').tag(config=True)
    test_folder = Unicode(help='').tag(config=True)

    output = Unicode(help='').tag(config=True)

    word_embedding = Unicode(help='').tag(config=True)
    word_embedding_dim = Integer(help='', default=300).tag(config=True)

    position_embedding_dim = Integer(help='', default=50).tag(config=True)

    tag_list = Unicode(help='').tag(config=True)
    tag_embedding_dim = Integer(help='', default_value=50).tag(config=True)

    dropout = Float(help='', default=0.5).tag(config=True)
    context_size = Integer(help='', default=30).tag(config=True)
    window_sizes = List(help='', default=[2, 3, 4, 5]).tag(config=True)
    filter_num = Integer(help='', default=100).tag(config=True)
    fix_embedding = Bool(help='', default=False).tag(config=True)

    batch_size = Integer(help='').tag(config=True)

    format = Unicode(help='').tag(config=True)
    no_punct = Bool(help='', default=False).tag(config=True)
    no_sentence = Bool(help='', default=False).tag(config=True)

    # Frame based detector.
    frame_lexicon = Unicode(help='').tag(config=True)
    event_list = Unicode(help='').tag(config=True)
    entity_list = Unicode(help='').tag(config=True)
    relation_list = Unicode(help='Lexicon for relations', ).tag(config=True)

    language = Unicode(help='').tag(config=True)


def set_basic_log(log_level=logging.INFO):
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=log_level, format=log_format)
