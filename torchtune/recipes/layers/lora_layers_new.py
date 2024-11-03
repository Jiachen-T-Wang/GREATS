# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import math
from typing import List

import torch.nn.functional as F

from torch import nn, Tensor

from torchao.dtypes.nf4tensor import linear_nf4, to_nf4
from torchtune.modules.peft.peft_utils import AdapterModule
from torchtune.utils import _register_nf4_dispatch_ops  # noqa: F401


class LoRALinear(nn.Module, AdapterModule):
    """LoRA linear layer as introduced in `LoRA: Low-Rank Adaptation of Large Language Models <https://arxiv.org/abs/2106.09685>`_.

    LoRA perturbs a given layer via a low-rank approximation where only
    the rank decomposition matrices are trainable. In a linear layer instead of
    :math:`x \\mapsto W_0x` a LoRALinear layer is defined as
    :math:`x \\mapsto W_0x + (\\alpha / r)BAx`, where :math:`r` is the rank of
    the matrices :math:`A` and :math:`B` and :math:`\\alpha` is a scaling factor.
    As in the original implementation, we support dropout before multiplication
    by the low-rank matrices.

    Args:
        in_dim (int): input dimension
        out_dim (int): output dimension
        rank (int): rank of the low-rank approximation
        alpha (float): scaling factor for the low-rank approximation
        dropout (float): dropout probability. Default: 0.0
        use_bias (bool): whether to include bias in the original linear layer.
            Default: False
        quantize_base (bool): Whether to quantize base linear weight or not.
            Default: False
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        use_bias: bool = False,
        quantize_base: bool = False
    ):
        super().__init__()
        self.in_dim = in_dim
        self.rank = rank
        self.alpha = alpha
        self.out_dim = out_dim
        self.use_bias = use_bias
        self._quantize_base = quantize_base
        weight, bias = self._create_weight_and_bias()
        weight = weight.to("cuda")
        if bias is not None:
            bias = bias.to("cuda")
        # 'self.disabled' is a flag showing whether to turn off LoRA adapters,
        # this can be used in DPO for treating the lora adapters as the policy model
        # and disabling it to treat the base model as the reference model
        self.disabled = False
        self.register_parameter("weight", nn.Parameter(weight))
        self.register_parameter(
            "bias", nn.Parameter(bias) if bias is not None else None
        )
        self.dropout = nn.Dropout(p=dropout)
        self.lora_a = nn.Linear(in_features=in_dim, out_features=rank, bias=False).to("cuda")
        self.lora_b = nn.Linear(in_features=rank, out_features=out_dim, bias=False).to("cuda")
        self.merged = False
        # Note: FSDP's meta device initialization contract assumes that a module's
        # reset_parameters method only initializes its own parameters (i.e. no child
        # params are initialized, as is done in initialize_parameters below).
        # For that reason, we patch reset_parameters directly on lora_a and lora_b submodules
        # when using meta device. This is done in
        # torchtune.utils.prepare_model_for_fsdp_with_meta_device.
        # See this issue for more details: https://github.com/pytorch/pytorch/issues/104187.
        # Without meta device, we only need the following:
        self.initialize_parameters()

    def initialize_parameters(self):
        # Initialize as in
        # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L119
        _lora_a_init_params(self.lora_a)
        _lora_b_init_params(self.lora_b)

    def _create_weight_and_bias(self):
        """
        Creates a linear weight and bias tensor, using NF4 dtype if we're quantizing
        (indicated via quantize_base=True).
        """
        in_dim, out_dim, use_bias = self.in_dim, self.out_dim, self.use_bias
        linear = nn.Linear(in_features=in_dim, out_features=out_dim, bias=use_bias)
        weight = linear.weight if not self._quantize_base else to_nf4(linear.weight)
        bias = None
        if self.use_bias:
            if self._quantize_base:
                raise NotImplementedError(
                    "Quantized LoRALinear does not support bias at the moment."
                )
            bias = linear.bias
        return weight, bias

    def adapter_params(self) -> List[str]:
        """
        Return lora_a.weight and lora_b.weight as adapter params.
        If bias is enabled, also return lora_a.bias and lora_b.bias.
        """
        # NOTE: this function has to be updated if the names of "lora_a" and "lora_b"
        # in this module change.
        adapter_params = ["lora_a.weight", "lora_b.weight"]
        return adapter_params

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape ``(..., in_dim)``

        Returns:
            Tensor: output tensor with shape ``(..., out_dim)``

        """
        if self._quantize_base:
            out = linear_nf4(input=x, weight=self.weight)
        else:
            self.weight.to("cuda").to(x.dtype)
            if self.use_bias:
                self.bias.to("cuda").to(x.dtype)
            out = F.linear(x, self.weight, self.bias)
        if self.disabled:
            return out
        lora_out = self.lora_a(self.dropout(x))
        lora_out = (self.alpha / self.rank) * self.lora_b(lora_out)
        return out + lora_out


def _lora_a_init_params(x: nn.Linear) -> None:
    """
    Initialize LoRA A weight to Kaiming uniform.
    """
    nn.init.kaiming_uniform_(x.weight, a=math.sqrt(5))


def _lora_b_init_params(x: nn.Linear) -> None:
    """
    Initialize LoRA B weight to zeros.
    """
    nn.init.zeros_(x.weight)



 
class GCLoRALinear(LoRALinear):
    def __init__(self, in_features, out_features, r=0, lora_alpha=1, lora_dropout=0.,  **kwargs):
        super(GCLoRALinear, self).__init__(in_features, out_features, rank=r, alpha=lora_alpha,
                                           dropout=lora_dropout, **kwargs)

        self.layer_type = 'GC_Linear_LoRA'
        self.register_forward_hook(self.capture_hook)

    def capture_hook(self, module, input, output):
        self.layer_input = input[0]  # input is a tuple
        self.pre_activation = output

    def pe_grad_gradcomp(self, deriv_pre_activ, per_sample=True):
        is_2d = self.layer_input.dim() == 2

        a = self.layer_input

        batch_size = deriv_pre_activ.shape[0]

        dLdO = deriv_pre_activ * batch_size

        dLdO = dLdO.to(self.lora_B.dtype)
        a = a.to(self.lora_A.dtype)

        dLdO_B = torch.matmul(dLdO, self.lora_B)
        a_A = torch.matmul(a, self.lora_A.T)

        decompose_pair_A = (dLdO_B, a)
        decompose_pair_B = (dLdO, a_A)

        return [decompose_pair_A, decompose_pair_B]
