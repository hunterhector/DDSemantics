from time import gmtime, strftime
import torch
import numpy as np


def batch_combine(l_data, device):
    data = torch.cat(
        [torch.unsqueeze(d, 0) for d in l_data], dim=0
    ).to(device)
    return data


def to_torch(data, data_type):
    return torch.from_numpy(np.asarray(data, data_type))


def remove_neg(raw_predicate):
    # Frames of verb with or without negation should be the same.

    neg = 'not_'
    if raw_predicate.startswith(neg):
        return raw_predicate[len(neg):]

    return raw_predicate


def get_time():
    return strftime("%Y-%m-%d %H:%M:%S", gmtime())
