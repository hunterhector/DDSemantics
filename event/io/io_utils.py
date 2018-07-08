import sys


def pad_2d_list(in_list, pad_to_length, dim=0, pad_value=0):
    if dim == 0:
        pad_len = pad_to_length - len(in_list)
        pads = [pad_value] * len(in_list[0])
        return in_list + [pads for _ in range(pad_len)]
    elif dim == 1:
        return [l + [pad_value] * (pad_to_length - len(l)) for l in in_list]
    else:
        raise ValueError("Invalid dimension {} for 2D list".format(dim))
