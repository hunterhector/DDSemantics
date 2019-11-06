from collections import defaultdict

import numpy as np
import torch
from event.io.io_utils import pad_2d_list, pad_last_axis
from event.arguments.util import (batch_combine, to_torch)


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

    def __init__(self, batch_size, device):
        self.batch_size = batch_size
        self.device = device

        self.common_data = defaultdict(list)
        self.instance_data = defaultdict(list)
        self.max_context_size = 0
        self.max_instance_size = 0
        self.max_num_slots = 0
        self.doc_count = 0

    def clear(self):
        self.common_data.clear()
        self.instance_data.clear()

        self.max_context_size = 0
        self.max_instance_size = 0
        self.max_num_slots = 0
        self.doc_count = 0

    def __var_pad(self, key, data, pad_length):
        pad_last_axis(data, self.data_dim[key], pad_length)

    def __batch_pad(self, key, data, pad_size):
        dim = self.data_dim[key]

        if dim == 2:
            print(key)
            return [pad_2d_list(v, pad_size) for v in data]
        elif dim == 1:
            return pad_2d_list(data, pad_size, axis=1)
        else:
            raise ValueError("Dimension unsupported %d" % dim)

    def read_data(self, instance_data, common_data):
        for key, value in common_data.items():
            self.common_data[key].append(value)
            if key.startswith('context_'):
                if len(value) > self.max_context_size:
                    self.max_context_size = len(value)
            if key == 'cross_slot_indicators':
                if len(value) > self.max_instance_size:
                    self.max_instance_size = len(value)
            if key == 'slot':
                if len(value) > 0 and len(value[0]) > self.max_num_slots:
                    self.max_num_slots = len(value[0])

        for ins_type, ins_data in instance_data.items():
            for key, value in ins_data.items():
                self.instance_data[ins_type][key].append(value)

        self.doc_count += 1

        # Each document is computed as a whole.
        if self.doc_count % self.batch_size == 0:
            debug_data = {
            }

            train_batch = self.create_batch(
                self.common_data, self.instance_data, self.max_num_slots,
                self.max_context_size, self.max_instance_size
            )

            yield train_batch, debug_data
            self.clear()

    def flush(self):
        if len(self.common_data) > 0:
            debug_data = {}
            train_batch = self.create_batch(
                self.common_data, self.instance_data, self.max_num_slots,
                self.max_context_size, self.max_instance_size)
            return train_batch, debug_data

    def create_batch(self, b_common_data, b_instance_data, max_num_slots,
                     max_context_size, max_instance_size):
        instance_data = {}
        common_data = {}
        # The actual instance lengths of in each batch.
        data_len = []
        sizes = {}

        for key, value in b_common_data.items():
            if key.startswith('context_'):
                padded = self.__batch_pad(key, value, max_context_size)
                vectorized = to_torch(padded, self.data_types[key])
                common_data[key] = batch_combine(vectorized, self.device)
            else:
                padded = self.__batch_pad(key, value, max_instance_size)
                vectorized = to_torch(padded, self.data_types[key])
                common_data[key] = batch_combine(vectorized, self.device)

            sizes[key] = len(padded)

        for ins_type, ins_data in b_instance_data.items():
            instance_data[ins_type] = {}
            for key, value in ins_data.items():
                data_len = [len(v) for v in value]
                padded = self.__batch_pad(key, value, max_instance_size)
                vectorized = to_torch(padded, self.data_types[key])

                instance_data[ins_type][key] = batch_combine(
                    vectorized, self.device
                )
                sizes[key] = len(padded)

        f_size = -1
        for key, s in sizes.items():
            if f_size < 0:
                f_size = s
            else:
                assert s == f_size

        assert f_size > 0

        mask = np.zeros([f_size, max_instance_size], dtype=int)
        for i, l in enumerate(data_len):
            mask[i][0: l] = 1

        ins_mask = to_torch(mask, np.uint8).to(self.device)

        # TODO: check if we need to include context mask?
        return instance_data, common_data, f_size, ins_mask
