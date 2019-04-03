import gc
import psutil
import sys
import torch
import os
import torch.cuda as cutorch
from hurry.filesize import size
from collections import Counter


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
        print('\t', key, num)


def gpu_mem_report():
    print("Allocated memory ", size(torch.cuda.memory_allocated()))


def cpu_stats():
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory GB:', memoryUse)


def make_one_hot(labels, C=2):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.cuda.FloatTensor N x C x H x W, where C is class number.
    One-hot encoded.
    '''
    one_hot = torch.cuda.FloatTensor(labels.size(0), C, labels.size(2),
                                     labels.size(3)).zero_()
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
