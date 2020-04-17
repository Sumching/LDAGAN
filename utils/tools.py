import torch
import numpy as np
from torch import autograd


def kl_loss(mu, logstd):
    # mu and logstd are b x k x d x d
    # make them into b*d*d x k

    dim = mu.shape[1]
    mu = mu.permute(0, 2, 3, 1).contiguous()
    logstd = logstd.permute(0, 2, 3, 1).contiguous()
    mu = mu.view(-1, dim)
    logstd = logstd.view(-1, dim)

    std = torch.exp(logstd)
    kl = torch.sum(-logstd + 0.5 * (std**2 + mu**2), dim=-1) - (0.5 * dim)

    return kl


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg

class alpha_Adam():
    """Adam optimizer for optimizing the Dirichlet parameter"""
    def __init__(self,
                 beta1 = 0.0,
                 beta2 = 0.9,
                 ep = 1e-8,
                 ):
        self.beta1 = beta1
        self.beta2 = beta2
        self.ep = ep
        self.m = 0
        self.v = 0
        self.t = 0

    def step(self, derv, alpha, lr):
        self.t += 1
        self.m = self.beta1*self.m + (1-self.beta1)*derv
        self.v = self.beta2*self.v + (1-self.beta2)*np.square(derv)
        m_hat = self.m/(1 - np.power(self.beta1, self.t))
        v_hat = self.v/(1 - np.power(self.beta2, self.t))
        alpha_new = alpha + lr * m_hat / (np.sqrt(v_hat)+self.ep)

        return alpha_new