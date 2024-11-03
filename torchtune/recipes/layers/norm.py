import torch
import torch.nn as nn


class InstanceNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(InstanceNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.Tensor(1, num_channels, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, x):
        N, C, H, W = x.shape

        x = x.view(N, C, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        normalized = (x - mean) / (var+self.eps).sqrt()
        self.layer_input = normalized.view(N, C, H, W)
        self.pre_activation = x * self.weight + bias

        return self.pre_activation

    def per_example_gradient(self, deriv_pre_activ):
        N, C = deriv_pre_activ.size(0), deriv_pre_activ.size(1)

        dLdZ = deriv_pre_activ
        dLdZ *= N
        X = self.layer_input

        pe_grad_weight = (dLdZ * X).view(N, C, -1).sum(-1)
        pe_grad_bias = dLdZ.view(N, C, -1).sum(-1)

        return pe_grad_weight, pe_grad_bias

    def pe_grad_sqnorm(self, deriv_pre_activ):
        pe_grad_weight, pe_grad_bias = self.per_example_gradient(deriv_pre_activ)

        return pe_grad_weight.pow(2).sum(dim=1) + pe_grad_bias.pow(2).sum(dim=1)
    

class GroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):        
        super(GroupNorm, self).__init__()
        self.pre_activation = None
        self.layer_input = None

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(1, num_groups, 1, 1))
            self.bias = nn.Parameter(torch.Tensor(1, num_groups, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, x):
        N, C, H, W = x.shape
        G = self.num_groups

        x = x.view(N, G, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        normalized = (x - mean) / (var + self.eps).sqrt()
        self.layer_input = normalized.view(N, C, H, W)
        self.pre_activation = self.weight * self.layer_input + self.bias

        return self.pre_activation

    def per_example_gradient(self, deriv_pre_activ):
        N = deriv_pre_activ.size(0)
        G = self.num_groups

        dLdZ = deriv_pre_activ
        dLdZ *= N
        X = self.layer_input

        pe_grad_weight = (dLdZ * X).view(N, G, -1).sum(-1)
        pe_grad_bias = dLdZ.view(N, G, -1).sum(-1)

        return pe_grad_weight, pe_grad_bias

    def pe_grad_sqnorm(self, deriv_pre_activ):
        pe_grad_weight, pe_grad_bias = self.per_example_gradient(deriv_pre_activ)

        return pe_grad_weight.pow(2).sum(dim=1) + pe_grad_bias.pow(2).sum(dim=1)

