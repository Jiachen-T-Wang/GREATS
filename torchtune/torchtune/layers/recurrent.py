import torch
import torch.nn as nn
from torch.nn import Parameter
from .base_layer import BasePGradLayer


class RNNModule(nn.RNN):
    def __init__(self, input_size, hidden_size):
        super(RNNModule, self).__init__(input_size, hidden_size)

        self.pre_activation = None
        self.layer_input = None
        self.layer_hidden = None

    def forward(self, input, h0):
        out, hn = super(RNNModule, self).forward(input, h0)

        # save input and hiddens for per-example gradient calculation
        self.pre_activation = out
        self.layer_input = input
        self.layer_hidden = torch.cat([h0, out[:-1]], dim=0)

        return out, hn

    def per_example_gradient(self, deriv_pre_activ):
        batch_size = deriv_pre_activ[0].size(0)

        pe_grad_ih = []
        pe_grad_hh = []

        for X, H, H_1, dLdZ in zip(self.layer_input, self.layer_hidden,
                                   self.pre_activation, deriv_pre_activ):
            dLdZ *= batch_size
            dLdZ *= (1.0 - H_1.pow(2))

            pe_grad_ih.append(torch.bmm(dLdZ.view(batch_size, -1, 1),
                                        X.view(batch_size, 1, -1)))
            pe_grad_hh.append(torch.bmm(dLdZ.view(batch_size, -1, 1),
                                        H.view(batch_size, 1, -1)))

        pe_grad_weight_ih = torch.stack(pe_grad_ih, dim=0).sum(dim=0)
        pe_grad_weight_hh = torch.stack(pe_grad_hh, dim=0).sum(dim=0)
        pe_grad_bias = deriv_pre_activ.sum(dim=0)

        return pe_grad_weight_ih, pe_grad_weight_hh, pe_grad_bias

    def pe_grad_sqnorm(self, deriv_pre_activ):
        batch_size = deriv_pre_activ[0].size(0)
        g_weight_ih, g_weight_hh, g_bias = self.per_example_gradient(deriv_pre_activ)

        sqnorm = g_weight_ih.pow(2).view(batch_size, -1).sum(1)
        sqnorm += g_weight_hh.pow(2).view(batch_size, -1).sum(1)
        sqnorm += g_bias.pow(2).view(batch_size, -1).sum(1)
        
        return sqnorm            


class RNNCell(nn.RNNCell):
    def __init__(self, input_size, hidden_size):
        super(RNNCell, self).__init__(input_size, hidden_size, bias=True)

        self.pre_activation = []
        self.layer_input = []
        self.layer_hidden = []

    def reset_pgrad(self):
        self.pre_activation.clear()
        self.layer_input.clear()
        self.layer_hidden.clear()

    def forward(self, input, hx):
        out = super(RNNCell, self).forward(input, hx)

        self.pre_activation.append(out)
        self.layer_input.append(input)
        self.layer_hidden.append(hx)

        return out        

    def per_example_gradient(self, deriv_pre_activ):
        batch_size = deriv_pre_activ[0].size(0)

        pe_grad_ih = []
        pe_grad_hh = []
        
        for X, H, H_1, dLdZ in zip(self.layer_input, self.layer_hidden,
                                   self.pre_activation, deriv_pre_activ):
            dLdZ *= batch_size
            dLdZ *= (1.0 - H_1.pow(2))

            pe_grad_ih.append(torch.bmm(dLdZ.view(batch_size, -1, 1),
                                        X.view(batch_size, 1, -1)))
            pe_grad_hh.append(torch.bmm(dLdZ.view(batch_size, -1, 1),
                                        H.view(batch_size, 1, -1)))

        pe_grad_weight_ih = torch.stack(pe_grad_ih, dim=0).sum(dim=0)
        pe_grad_weight_hh = torch.stack(pe_grad_hh, dim=0).sum(dim=0)
        pe_grad_bias = torch.stack(deriv_pre_activ, dim=0).sum(dim=0)

        return pe_grad_weight_ih, pe_grad_weight_hh, pe_grad_bias

    def pe_grad_sqnorm(self, deriv_pre_activ):
        batch_size = deriv_pre_activ[0].size(0)
        g_weight_ih, g_weight_hh, g_bias = self.per_example_gradient(deriv_pre_activ)

        sqnorm = g_weight_ih.pow(2).view(batch_size, -1).sum(1)
        sqnorm += g_weight_hh.pow(2).view(batch_size, -1).sum(1)
        sqnorm += g_bias.pow(2).view(batch_size, -1).sum(1)
        
        return sqnorm            
    

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        gate_size = 4 * hidden_size

        self.ih = nn.Linear(input_size, gate_size)
        self.hh = nn.Linear(hidden_size, gate_size)
        # self.weight_ih = Parameter(torch.randn(4*hidden_size, input_size))
        # self.weight_hh = Parameter(torch.randn(4*hidden_size, hidden_size))
        # self.bias_ih = Parameter(torch.randn(4*hidden_size))
        # self.bias_hh = Parameter(torch.randn(4*hidden_size))

        self.pre_activation = []
        self.layer_input = []
        self.layer_hidden = []

    def reset_pgrad(self):
        self.pre_activation.clear()
        self.layer_input.clear()
        self.layer_hidden.clear()

    # @torch.jit.script_method
    def forward(self, x_t, hx, cx):
        gates = self.ih(x_t) + self.hh(hx)
        # gates = (torch.mm(x_t, self.weight_ih.t()) + self.bias_ih +
        #          torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        self.pre_activation.append(gates)
        self.layer_input.append(x_t)
        self.layer_hidden.append(hx)  # h_t

        return hy, cy

    def per_example_gradient(self, deriv_pre_activ):
        batch_size = deriv_pre_activ[0].size(0)

        pe_grad_ih = []
        pe_grad_hh = []
        
        for X, H, dLdZ in zip(self.layer_input, self.layer_hidden,
                              deriv_pre_activ):
            dLdZ *= batch_size

            pe_grad_ih.append(torch.bmm(dLdZ.view(batch_size, -1, 1),
                                        X.view(batch_size, 1, -1)))
            pe_grad_hh.append(torch.bmm(dLdZ.view(batch_size, -1, 1),
                                        H.view(batch_size, 1, -1)))

        pe_grad_weight_ih = torch.stack(pe_grad_ih, dim=0).sum(dim=0)
        pe_grad_weight_hh = torch.stack(pe_grad_hh, dim=0).sum(dim=0)
        pe_grad_bias = torch.stack(deriv_pre_activ, dim=0).sum(dim=0)

        return pe_grad_weight_ih, pe_grad_weight_hh, pe_grad_bias

    def pe_grad_sqnorm(self, deriv_pre_activ):
        batch_size = deriv_pre_activ[0].size(0)
        g_weight_ih, g_weight_hh, g_bias = self.per_example_gradient(deriv_pre_activ)

        sq_norm = g_weight_ih.pow(2).view(batch_size, -1).sum(1)
        sq_norm += g_weight_hh.pow(2).view(batch_size, -1).sum(1)
        sq_norm += g_bias.pow(2).view(batch_size, -1).sum(1)

        return sq_norm
