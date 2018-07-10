import gc
import psutil
import sys
import torch
import os
import torch.cuda as cutorch
from hurry.filesize import size


def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())


def gpuMemReport():
    print("Allocated memory ", size(torch.cuda.memory_allocated()))


def cpuStats():
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory GB:', memoryUse)
