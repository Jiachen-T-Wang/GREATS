import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .base_layer import BasePGradLayer
from fastgc.common.im2col import im2col_indices
from fastgc.util import conv_outsize

import sys


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=False, padding_mode='zeros'):        
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias, padding_mode)
        self.pre_activation = None
        self.layer_input = None

        self.name = 'conv'

    def forward(self, input):
        self.layer_input = input
        out = F.conv2d(input, self.weight, self.bias, self.stride, self.padding)
        self.pre_activation = out
        
        return out

    def per_example_gradient(self, deriv_pre_activ):
        dLdZ = deriv_pre_activ
        H = self.layer_input

        print('ConvShape={}'.format(dLdZ.shape))
        batch_size, n_filter = dLdZ.shape[0], dLdZ.shape[1]

        per_grad_bias = None
        if self.bias is not None:
            per_grad_bias = dLdZ.view(batch_size, n_filter, -1).sum(2)  # bias
            
        k1, k2 = self.kernel_size
        C_in = H.shape[1]

        dLdZ_reshaped = dLdZ.view(batch_size, n_filter, -1)
        padding = self.padding[0]
        stride = self.stride[0]

        h_col = im2col_indices(H, k1, k2, padding=padding, stride=stride)
        per_grad_weight = torch.bmm(dLdZ_reshaped, h_col.transpose(1, 2))

        return per_grad_weight, per_grad_bias
    
    def pe_grad_sqnorm(self, deriv_pre_activ):
        batch_size = deriv_pre_activ.shape[0]
        
        pe_grad_weight, pe_grad_bias = self.per_example_gradient(deriv_pre_activ)
        sq_norm_weight = pe_grad_weight.pow(2).view(batch_size, -1).sum(1)

        if self.bias is not None:
            sq_norm_bias = pe_grad_bias.pow(2).view(batch_size, -1).sum(1)
            return sq_norm_weight + sq_norm_bias
        else:
            return sq_norm_weight
        
    def pe_grad_gradcomp(self, deriv_pre_activ, per_sample=True):

        batch_size = deriv_pre_activ.shape[0]

        dLdZ = deriv_pre_activ
        H = self.layer_input

        # print('dLdZ_Shape={}'.format(dLdZ.shape))
        # print('H_Shape={}'.format(H.shape))

        batch_size, n_filter = dLdZ.shape[0], dLdZ.shape[1]
        per_grad_bias = None
        
        if self.bias is not None:
            per_grad_bias = dLdZ.view(batch_size, n_filter, -1).sum(2)  # bias
            
        k1, k2 = self.kernel_size
        C_in = H.shape[1]

        dLdZ_reshaped = dLdZ.view(batch_size, n_filter, -1)
        padding = self.padding[0]
        stride = self.stride[0]

        h_col = im2col_indices(H, k1, k2, padding=padding, stride=stride)

        # print('dLdZ_reshaped.shape={}, h_col.transpose(1, 2).shape={}'.format(dLdZ_reshaped.shape, (h_col.transpose(1, 2)).shape))
        per_grad_weight = torch.bmm(dLdZ_reshaped, h_col.transpose(1, 2))
        # print('per_grad_weight.shape={}'.format(per_grad_weight.shape))

        if self.bias is not None:
            per_grad_bias_expanded = per_grad_bias.unsqueeze(-1)  # This makes its shape [1000, 16, 1]
            per_grad_weight = torch.cat((per_grad_weight, per_grad_bias_expanded), dim=-1)  # This should result in shape [1000, 16, 28]

        per_grad_weight = per_grad_weight.view(batch_size, -1)
        dummy = torch.ones((batch_size, 1), device=H.device)

        per_grad_weight *= batch_size

        return per_grad_weight, dummy

