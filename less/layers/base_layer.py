import torch
import torch.nn as nn


class BasePGradLayer(nn.Module):
    def __init__(self, module=None, pe_grad=True):
        super(BasePGradLayer, self).__init__()

        self.module = module
        self.pre_activ = None
        self.layer_input = None
        self.deriv_pre_activ = None

        if module and pe_grad:
            self.module.register_forward_hook(self.save_pre_activs)

    def save_pre_activs(self, module, input, output):
        # for per-example gradient calculation
        self.pre_activ = output
        self.layer_input = input[0]
        # self.pre_activ.register_hook(self.save_grad)

    def save_grad(self, grad):
        self.deriv_pre_activ = grad

    def forward(self, input):
        out = self.module(input)

        return out

    def per_example_gradient(self):
        is_2d = self.layer_input.dim() == 2
        Z = self.layer_input

        if is_2d:
            batch_size = self.deriv_pre_activ.size(0)
            dLdZ = self.deriv_pre_activ * batch_size

            pe_grad_weight = torch.bmm(dLdZ.view(batch_size, -1, 1),
                                       Z.view(batch_size, 1, -1))
            pe_grad_bias = dLdZ
        else:
            dLdZ = self.deriv_pre_activ.permute(1, 2, 0)
            dLdZ *= dLdZ.size(0)    
            pe_grad_weight = torch.bmm(dLdZ,
                                       Z.transpose(0, 1))
            pe_grad_bias = dLdZ.sum(dim=-1)

        return pe_grad_weight, pe_grad_bias
