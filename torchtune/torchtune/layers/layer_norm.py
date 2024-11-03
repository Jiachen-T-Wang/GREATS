import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

        self.pre_activation = None
        self.layer_input = None

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        self.layer_input = (x - mean) / (std + self.eps) 
        self.pre_activation = self.weight * self.layer_input + self.bias

        return self.pre_activation

    def per_example_gradient(self, deriv_pre_activ):
        is_2d = self.layer_input.dim() == 2

        dLdZ = deriv_pre_activ.permute(1, 0, 2)
        dLdZ *= dLdZ.size(0)
        Z = self.layer_input

        pe_grad_weight = (dLdZ * Z.transpose(0, 1)).sum(dim=1)
        pe_grad_bias = dLdZ.sum(dim=1)

        return pe_grad_weight, pe_grad_bias

    def pe_grad_sqnorm(self, deriv_pre_activ):
        pe_grad_weight, pe_grad_bias = self.per_example_gradient(deriv_pre_activ)

        return pe_grad_weight.pow(2).sum(dim=1) + pe_grad_bias.pow(2).sum(dim=1)

