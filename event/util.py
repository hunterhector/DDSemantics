import argparse
import logging
import os
import gzip
import unicodedata
import sys

from traitlets.config.loader import KeyValueConfigLoader
from traitlets.config.loader import PyFileConfigLoader
import hashlib


class OptionPerLineParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        if arg_line.startswith("#"):
            return []
        return arg_line.split()


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


def set_file_log(log_file, log_level=logging.INFO):
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=log_level, format=log_format, filename=log_file)


def basic_console_log(log_level=logging.INFO):
    root = logging.getLogger()
    root.setLevel(log_level)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)


def load_command_line_config(args):
    cl_loader = KeyValueConfigLoader()
    return cl_loader.load_config(args)


def load_file_config(config_path):
    loader = PyFileConfigLoader(config_path)
    conf = loader.load_config()
    return conf


def load_config_with_cmd(args):
    file_conf = load_file_config(args[1])

    if len(args) > 1:
        cl_conf = load_command_line_config(args[2:])
        file_conf.merge(cl_conf)

    return file_conf


def load_with_sub_config(args):
    """
    This method try to mimics the behavior of the sub_config. It currently only
    take one base and one main.
    :param args:
    :return:
    """
    base_conf = args[1]
    main_conf = args[2]

    loader = PyFileConfigLoader(base_conf)
    loader.load_config()

    # Since subconfig will be merged to and override the base.
    loader.load_subconfig(main_conf)

    all_conf = loader.config

    if len(args) > 2:
        cl_conf = load_command_line_config(args[3:])
        all_conf.merge(cl_conf)

    return all_conf


def file_md5(file):
    hashlib.md5(open(file, 'rb').read()).hexdigest()


tbl = dict.fromkeys(
    i for i in range(sys.maxunicode) if
    unicodedata.category(chr(i)).startswith('P')
)


def remove_punctuation(text):
    return text.translate(tbl)


def get_env(var_name):
    if var_name not in os.environ:
        raise KeyError("Please supply the directory as environment "
                       "variable: {}".format(var_name))
    else:
        return os.environ[var_name]
