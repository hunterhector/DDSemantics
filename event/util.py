import argparse
import hashlib
import logging
import os
import sys
import unicodedata

from traitlets.config.loader import KeyValueConfigLoader
from traitlets.config.loader import PyFileConfigLoader


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


def load_mixed_configs():
    file_confs = [a for a in sys.argv[1:] if a.endswith('.py')]
    arg_confs = [a for a in sys.argv[1:] if a.startswith('--')]
    return load_multi_configs(file_confs, arg_confs)


def load_multi_configs(file_args, cmd_args):
    """This method try to mimics the behavior of the sub_config. It currently
    only take one base and one main.

    Args:
      file_args:
      cmd_args: 

    Returns:

    """
    cl_conf = load_command_line_config(cmd_args)

    if len(file_args) > 0:
        base_conf = file_args[0]

        loader = PyFileConfigLoader(base_conf)
        loader.load_config()

        for conf in file_args[1:]:
            # Since subconfig will be merged to and override the base.
            loader.load_subconfig(conf)

        all_conf = loader.config
        all_conf.merge(cl_conf)
        return all_conf
    else:
        return cl_conf


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

def append_num_to_path(file_path, suffix=0):
    if os.path.exists(file_path):
        new_path = f'{file_path}_{suffix}'
        if os.path.exists(new_path):
            append_num_to_path(file_path, suffix + 1)
        else:
            os.rename(file_path, new_path)
