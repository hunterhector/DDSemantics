from time import localtime, strftime
import torch
import numpy as np
import random
import math


class ClozeSampler:
    def __init__(self, sample_pred_threshold=10e-5, seed=None):
        self.sample_pred_threshold = sample_pred_threshold
        self.provided_seed = seed
        random.seed(self.provided_seed)

    def reset(self):
        random.seed(self.provided_seed)

    def sample_cross(self, arg_entities, evm_id, ent_id):
        remaining = []
        for arg_entity_info in arg_entities:
            if not (arg_entity_info['event_index'] == evm_id
                    or arg_entity_info['entity_id'] == ent_id):
                remaining.append(arg_entity_info)

        if len(remaining) > 0:
            return random.choice(remaining)
        else:
            return None

    def sample_ignore_item(self, data, ignored_item):
        """
        Sample one item in the list, but ignore a provided one. If the list
        contains less than 2 elements, nothing will be sampled.
        :param data:
        :param ignored_item:
        :return:
        """
        if len(data) <= 1:
            return None
        while True:
            sampled_item = random.choice(data)
            if not sampled_item == ignored_item:
                break
        return sampled_item

    def subsample_pred(self, pred_tf, freq):
        if freq > self.sample_pred_threshold:
            if pred_tf > self.sample_pred_threshold:
                rate = self.sample_pred_threshold / freq
                if random.random() < 1 - rate - math.sqrt(rate):
                    return False
        return True


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
    return strftime("%Y-%m-%d %H:%M:%S", localtime())
