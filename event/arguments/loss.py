import torch


def cross_entropy(y_hat, y):
    print(y_hat)
    print(y)
    if y_hat == 1:
        return -torch.log(y)
    else:
        return -torch.log(1 - y)


def hinge(y_hat, y):
    return np.max(0, 1 - y_hat * y)
