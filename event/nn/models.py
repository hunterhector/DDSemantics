from torch import nn
import torch
import json
import logging
from torch.nn.parameter import Parameter


class KernelPooling(nn.Module):
    """kernel pooling layer
    init:
        v_mu: a 1-d dimension of mu's
        sigma: the sigma
    input:
        similar to Linear()
            a n-D tensor, last dimension is the one to enforce kernel pooling
    output:
        n-K tensor, K is the v_mu.size(), number of kernels

    Args:

    Returns:

    """

    def __init__(self, l_mu=None, l_sigma=None):
        super(KernelPooling, self).__init__()
        if l_mu is None:
            l_mu = [1, 0.9, 0.7, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5, -0.7, -0.9]
        self.v_mu = Parameter(torch.FloatTensor(l_mu))
        self.K = len(l_mu)
        if l_sigma is None:
            l_sigma = [1e-3] + [0.1] * (self.v_mu.size()[-1] - 1)
        self.v_sigma = Parameter(torch.FloatTensor(l_sigma))

        logging.info(
            '[%d] pooling kernels: %s', self.K,
            json.dumps(list(zip(l_mu, l_sigma)))
        )
        return

    def forward(self, in_tensor, mtx_score=None):
        import pdb
        pdb.set_trace()
        in_tensor = in_tensor.unsqueeze(-1)
        in_tensor = in_tensor.expand(in_tensor.size()[:-1] + (self.K,))
        score = -(in_tensor - self.v_mu) * (in_tensor - self.v_mu)
        kernel_value = torch.exp(score / (2.0 * self.v_sigma * self.v_sigma))

        if mtx_score is not None:
            mtx_score = mtx_score.unsqueeze(-1).unsqueeze(1)
            mtx_score = mtx_score.expand_as(kernel_value)
            weighted_kernel_value = kernel_value * mtx_score
        else:
            weighted_kernel_value = kernel_value

        # TODO: The sum here causes nan.
        sum_kernel_value = torch.sum(weighted_kernel_value, dim=-2).clamp(
            min=1e-10)  # add freq/weight
        sum_kernel_value = torch.log(sum_kernel_value)
        return sum_kernel_value
