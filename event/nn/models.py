from torch import nn
from torch.autograd import Variable
import torch
import json
import logging

use_cuda = False


class KernelPooling(nn.Module):
    """
    kernel pooling layer
    init:
        v_mu: a 1-d dimension of mu's
        sigma: the sigma
    input:
        similar to Linear()
            a n-D tensor, last dimension is the one to enforce kernel pooling
    output:
        n-K tensor, K is the v_mu.size(), number of kernels
    """

    def __init__(self, l_mu=None, l_sigma=None):
        super(KernelPooling, self).__init__()
        if l_mu is None:
            l_mu = [1, 0.9, 0.7, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5, -0.7, -0.9]
        self.v_mu = Variable(torch.FloatTensor(l_mu), requires_grad=False)
        self.K = len(l_mu)
        if l_sigma is None:
            l_sigma = [1e-3] + [0.1] * (self.v_mu.size()[-1] - 1)
        self.v_sigma = Variable(torch.FloatTensor(l_sigma), requires_grad=False)
        if use_cuda:
            self.v_mu = self.v_mu.cuda()
            self.v_sigma = self.v_sigma.cuda()
        logging.info('[%d] pooling kernels: %s',
                     self.K, json.dumps(list(zip(l_mu, l_sigma)))
                     )
        return

    def forward(self, in_tensor, mtx_score=None):
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

        sum_kernel_value = torch.sum(weighted_kernel_value, dim=-2).clamp(
            min=1e-10)  # add freq/weight
        sum_kernel_value = torch.log(sum_kernel_value)
        return sum_kernel_value
