from collections import defaultdict
import pdb
from typing import Dict

import numpy as np

from event.arguments.data.cloze_instance import ClozeInstances
from event.util import batch_combine, to_torch
from event.io.io_utils import pad_2d_list, pad_last_axis


class ClozeBatcher:
    data_types = {
        'context_event_component': np.int64,
        'context_slot': np.int64,
        'context_slot_value': np.int64,
        'context_slot_length': np.int64,
        'context_predicate': np.int64,
        'event_indices': np.int64,
        'cross_event_indices': np.int64,
        'inside_event_indices': np.int64,
        'slot_indicators': np.int64,
        'cross_slot_indicators': np.int64,
        'inside_slot_indicators': np.int64,
        'event_component': np.int64,
        'slot': np.int64,
        'slot_value': np.int64,
        'slot_length': np.int64,
        'predicate': np.int64,
        'distances': np.float32,
        'features': np.float32,
    }

    data_dim = {
        'context_event_component': 2,
        'context_slot': 2,
        'context_slot_value': 2,
        'context_slot_length': 1,
        # TODO: what is the dim for predicate? 1 or 2, make it consistent.
        'context_predicate': 2,
        'event_indices': 1,
        'cross_event_indices': 1,
        'inside_event_indices': 1,
        'slot_indicators': 1,
        'cross_slot_indicators': 1,
        'inside_slot_indicators': 1,
        'event_component': 2,
        'slot': 2,
        'slot_value': 2,
        'slot_length': 1,
        'predicate': 2,
        'distances': 2,
        'features': 2,
    }

    # Keep track of the slot keys, since the size here might be different.
    slot_keys = {'slot', 'slot_value', 'context_slot', 'context_slot_value'}

    def __init__(self, batch_size, device):
        self.batch_size = batch_size
        self.device = device

        self.b_common_data = defaultdict(list)
        self.b_instance_data = defaultdict(list)
        self.b_meta_data = defaultdict(list)
        self.b_labels = []

        self.max_context_size = 0
        self.max_instance_size = 0
        self.doc_count = 0

        self.max_num_slots = 0
        self.max_c_num_slots = 0

    def clear(self):
        self.b_common_data.clear()
        self.b_instance_data.clear()
        self.b_labels.clear()
        self.b_meta_data.clear()

        self.max_context_size = 0
        self.max_instance_size = 0

        self.max_num_slots = 0
        self.max_c_num_slots = 0

        self.doc_count = 0

    def __var_pad(self, key, data, pad_length):
        pad_last_axis(data, self.data_dim[key], pad_length)

    def __batch_pad(self, key, data, pad_size):
        dim = self.data_dim[key]

        if dim == 2:
            return [pad_2d_list(v, pad_size) for v in data]
        elif dim == 1:
            return pad_2d_list(data, pad_size, axis=1)
        else:
            raise ValueError("Dimension unsupported %d" % dim)

    def get_batch(self, instances: ClozeInstances, common_data: Dict,
                  meta: Dict = None):
        instance_data = instances.data
        labels = instances.label

        for key, value in common_data.items():
            self.b_common_data[key].append(value)

            if key.startswith('context_'):
                if len(value) > self.max_context_size:
                    self.max_context_size = len(value)
            if key == 'event_indices':
                if len(value) > self.max_instance_size:
                    self.max_instance_size = len(value)
            if key == 'context_slot':
                self.max_c_num_slots = max(
                    list(len(l) for l in value) + [self.max_c_num_slots]
                )

        for key, value in instance_data.items():
            if key == 'slot':
                self.max_num_slots = max(
                    list(len(l) for l in value) + [self.max_num_slots]
                )
            self.b_instance_data[key].append(value)

        if meta is not None:
            for key, value in meta.items():
                self.b_meta_data[key].append(value)

        self.b_labels.append(labels)
        self.doc_count += 1

        # Each document is computed as a whole.
        if self.doc_count % self.batch_size == 0:
            yield self.create_batch()
            self.clear()

    def flush(self):
        if len(self.b_common_data) > 0:
            return self.create_batch()

    def create_batch(self):
        instance_data = {}
        common_data = {}
        # The actual instance lengths of in each batch.
        data_len = []
        sizes = {}

        for key, value in self.b_common_data.items():
            if key in self.slot_keys:
                [self.__var_pad(key, v, self.max_c_num_slots) for v in value]

            if key.startswith('context_'):
                padded = self.__batch_pad(key, value, self.max_context_size)
                try:
                    vectorized = to_torch(padded, self.data_types[key])
                except Exception:
                    pdb.set_trace()
                common_data[key] = batch_combine(vectorized, self.device)
            else:
                padded = self.__batch_pad(key, value, self.max_instance_size)
                vectorized = to_torch(padded, self.data_types[key])
                common_data[key] = batch_combine(vectorized, self.device)

            sizes[key] = len(padded)

        for key, value in self.b_instance_data.items():
            if key in self.slot_keys:
                [self.__var_pad(key, v, self.max_num_slots) for v in value]

            data_len = [len(v) for v in value]
            padded = self.__batch_pad(key, value, self.max_instance_size)

            vectorized = to_torch(padded, self.data_types[key])
            instance_data[key] = batch_combine(vectorized, self.device)

            sizes[key] = len(padded)

        labels = to_torch(
            pad_2d_list(self.b_labels, self.max_instance_size, axis=1),
            np.float32).to(self.device)

        data_size = -1
        for key, s in sizes.items():
            if data_size < 0:
                data_size = s
            else:
                assert s == data_size
        assert data_size > 0

        mask = np.zeros([data_size, self.max_instance_size], dtype=int)
        for i, l in enumerate(data_len):
            mask[i][0: l] = 1

        ins_mask = to_torch(mask, np.float32).to(self.device)

        return (
            labels, instance_data, common_data, data_size, ins_mask,
            self.b_meta_data
        )
