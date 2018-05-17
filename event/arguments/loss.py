def cross_entropy(y_hat, y):
    if y_hat == 1:
        return -log(y)
    else:
        return -log(1 - y)


def hinge(y_hat, y):
    return np.max(0, 1 - y_hat * y)
