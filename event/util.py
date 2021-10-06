import argparse
import gc
import hashlib
import logging
import os
import sys
import unicodedata
from collections import Counter
from time import strftime, localtime
from datetime import datetime
import psutil
from hurry.filesize import size

import numpy as np
import torch

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
            padding = " " * (span[0] - len(sent) - sent_start)
            sent += padding
        sent += token
    return sent


def find_by_id(folder, docid):
    for filename in os.listdir(folder):
        if filename.startswith(docid):
            return os.path.join(folder, filename)


def rm_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def get_date_stamp():
    return datetime.today().strftime("%Y%m%d")


def set_basic_log(log_level=logging.INFO):
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=log_level, format=log_format)


def set_file_log(log_file, log_level=logging.INFO):
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=log_level, format=log_format, filename=log_file)


def basic_console_log(log_level=logging.INFO):
    root = logging.getLogger()
    root.setLevel(log_level)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
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
    file_confs = [a for a in sys.argv[1:] if a.endswith(".py")]
    arg_confs = [a for a in sys.argv[1:] if a.startswith("--")]
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
    hashlib.md5(open(file, "rb").read()).hexdigest()


tbl = dict.fromkeys(
    i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")
)


def remove_punctuation(text):
    return text.translate(tbl)


def get_env(var_name):
    if var_name not in os.environ:
        raise KeyError(
            "Please supply the directory as environment "
            "variable: {}".format(var_name)
        )
    else:
        return os.environ[var_name]


def append_num_to_path(file_path, suffix=0):
    if os.path.exists(file_path):
        new_path = f"{file_path}_{suffix}"
        if os.path.exists(new_path):
            append_num_to_path(file_path, suffix + 1)
        else:
            os.rename(file_path, new_path)


def batch_combine(l_data):
    data = torch.cat([torch.unsqueeze(d, 0) for d in l_data], dim=0)
    return data


def to_torch(data, data_type):
    return torch.from_numpy(np.asarray(data, data_type))


def remove_neg(raw_predicate):
    # Frames of verb with or without negation should be the same.

    neg = "not_"
    if raw_predicate.startswith(neg):
        return raw_predicate[len(neg) :]

    return raw_predicate


def get_time():
    return strftime("%Y-%m-%d %H:%M:%S", localtime())


def show_tensors():
    num_allocated = 0
    cell_sum = Counter()

    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            # print(type(obj), obj.size(), obj.type())
            num_allocated += 1

            cell_count = 1
            for e in obj.size():
                cell_count *= e
            cell_sum[obj.type()] += cell_count

    print("Number of tensors: [%d]." % num_allocated)
    print("Cell by type")
    for key, num in cell_sum.items():
        print("\t", key, num)


def gpu_mem_report():
    print("Allocated memory ", size(torch.cuda.memory_allocated()))


def cpu_stats():
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2.0 ** 30  # memory use in GB...I think
    print("memory GB:", memoryUse)


def make_2d_one_hot(batched_indices, max_length, device):
    b, l = batched_indices.shape
    data = batched_indices.unsqueeze(-1)
    one_hot = torch.zeros([b, l, max_length], dtype=torch.float32).to(device)
    one_hot.scatter_(2, data, 1)
    return one_hot


def make_one_hot(labels, C=2):
    """Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Args:
      labels(torch.autograd.Variable of torch.cuda.LongTensor): N x 1 x H x W,
      where N is batch size. Each value is an integer representing correct
      classification.
      C(integer., optional): number of classes in labels. (Default value = 2)

    Returns:


    """
    one_hot = torch.FloatTensor(
        labels.size(0), C, labels.size(2), labels.size(3)
    ).zero_()
    target = one_hot.scatter_(1, labels.data, 1)

    return target


def topk_with_fill(data, k, dimension, largest, dtype=torch.int32, filler=0):
    if data.shape[dimension] >= k:
        res, _ = data.topk(k, dimension, largest=largest)
    else:
        pad_len = k - data.shape[dimension]
        l_pad_shape = []

        for index, s in data.shape:
            if index == dimension:
                l_pad_shape.append(pad_len)
            else:
                l_pad_shape.append(s)

        pad_shape = tuple(l_pad_shape)

        if filler == 1:
            padding = torch.ones(pad_shape, dtype=dtype)
        else:
            padding = torch.zeros(pad_shape, dtype=dtype)
            if not filler == 0:
                padding.fill_(filler)

        res = torch.cat((data, padding), -1)

    return res
