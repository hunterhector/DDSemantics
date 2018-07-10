import argparse
import logging
import os
import gzip
import unicodedata
import sys

from traitlets.config.loader import KeyValueConfigLoader


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


def find_by_id(folder, docid):
    for filename in os.listdir(folder):
        if filename.startswith(docid):
            return os.path.join(folder, filename)


def rm_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def set_basic_log(log_level=logging.INFO):
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=log_level, format=log_format)


def load_command_line_config(args):
    cl_loader = KeyValueConfigLoader()
    return cl_loader.load_config(args)


tbl = dict.fromkeys(
    i for i in range(sys.maxunicode) if
    unicodedata.category(chr(i)).startswith('P')
)


def remove_punctuation(text):
    return text.translate(tbl)

