import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from fastgc.layers.layer_norm import LayerNorm
from fastgc.layers.linear import Linear


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


## Note: key, value doesn't seem to be used in this function
def multi_head_attention_forward(query,                           # type: Tensor
                                 key,                             # type: Tensor
                                 value,                           # type: Tensor
                                 embed_dim_to_check,              # type: int
                                 num_heads,                       # type: int
                                 in_proj,                    # type: Tensor
                                 dropout_p,                       # type: float
                                 out_proj,                   # type: Tensor
                                 training=True,                   # type: bool
                                 key_padding_mask=None,           # type: Optional[Tensor]
                                 need_weights=True,               # type: bool
                                 attn_mask=None,                  # type: Optional[Tensor]
                                 use_separate_proj_weight=False,  # type: bool
                                 q_proj_weight=None,              # type: Optional[Tensor]
                                 k_proj_weight=None,              # type: Optional[Tensor]
                                 v_proj_weight=None,              # type: Optional[Tensor]
                                 static_k=None,                   # type: Optional[Tensor]
                                 static_v=None                    # type: Optional[Tensor]
                                 ):
    qkv_same = torch.equal(query, key) and torch.equal(key, value)
    kv_same = torch.equal(key, value)

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert list(query.size()) == [tgt_len, bsz, embed_dim]
    assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    # self-attention
    P = in_proj(query)
    # P = F.linear(query, in_proj_weight, in_proj_bias)
    q, k, v = P.chunk(3, dim=-1)

    q = q * scaling

    # head_dim = d_k in the paper
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(0)
        attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

    H = torch.bmm(attn_output_weights, v)
    assert list(H.size()) == [bsz * num_heads, tgt_len, head_dim]
    H = H.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = out_proj(H)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False): 
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj = Linear(embed_dim, 3*embed_dim)
        # self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))

        # if bias:
        #     self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        # else:
        #     self.register_parameter('in_proj_bias', None)
        self.out_proj = Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj.weight)

        if self.in_proj.bias is not None:
            constant_(self.in_proj.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):

        attn_out, _ = multi_head_attention_forward(query, key, value,
                                                   self.embed_dim, self.num_heads,
                                                   self.in_proj, self.dropout, self.out_proj,
                                                   training=self.training,
                                                   key_padding_mask=key_padding_mask,
                                                   need_weights=need_weights,
                                                   attn_mask=attn_mask)

        return attn_out

    def per_example_gradient(self, deriv_pre_activ_in, deriv_pre_activ_out):
        pe_grad_weight_in, pe_grad_bias_in = \
            self.in_proj.per_example_gradient(deriv_pre_activ_in)
        pe_grad_weight_out, pe_grad_bias_out = \
            self.out_proj.per_example_gradient(deriv_pre_activ_out)

        return (pe_grad_weight_in, pe_grad_bias_in, pe_grad_weight_out, pe_grad_bias_out)

    def pe_grad_sqnorm(self, deriv_pre_activ):
        grads = self.per_example_gradient(*deriv_pre_activ)
        batch_size = grads[0].size(0)

        grad_sq_norm = torch.zeros(batch_size, device=grads[0].device)
        for grad in grads:
            grad_sq_norm.add_(grad.pow(2).view(batch_size, -1).sum(1))

        return grad_sq_norm

    def collect_preactivations(self):        
        return (self.in_proj.pre_activation, self.out_proj.pre_activation)

    def collect_layers(self):        
        return (self.in_proj, self.out_proj)

    def pe_grad_gradcomp(self, deriv_pre_activ, per_sample=True):
        deriv_pre_activ_in, deriv_pre_activ_out = deriv_pre_activ
        return [self.in_proj.pe_grad_gradcomp(deriv_pre_activ_in), self.out_proj.pe_grad_gradcomp(deriv_pre_activ_out)]




class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 pe_grad=True):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # self._pe_modules = [self.self_attn, self.linear1, self.linear2,
        #                     self.norm1, self.norm2]
        self._pe_modules = [self.self_attn, self.linear1, self.linear2]


    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.self_attn(src, src, src, attn_mask=src_mask,
                             key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src)
        src = self.norm1(src)        
        
        if hasattr(self, "activation"):
            src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:  # for backward compatibility
            src = self.linear2(self.dropout(F.relu(self.linear1(src))))
        out = src + self.dropout2(src)
        out = self.norm2(out)

        return out

    def per_example_gradient(self):
        grads = []

        for m in self._pe_modules:
            grads.extend(m.per_example_gradient())

        return grads

    def pe_grad_sqnorm(self, deriv_pre_activ):
        batch_size = deriv_pre_activ[0].size(1)
        device = deriv_pre_activ[0].device
        grad_sq_norm = torch.zeros(batch_size, device=device)

        grad_sq_norm.add_(self.self_attn.pe_grad_sqnorm(deriv_pre_activ[:2]))
        grad_sq_norm.add_(self.linear1.pe_grad_sqnorm(deriv_pre_activ[2]))
        grad_sq_norm.add_(self.linear2.pe_grad_sqnorm(deriv_pre_activ[3]))
        grad_sq_norm.add_(self.norm1.pe_grad_sqnorm(deriv_pre_activ[4]))
        grad_sq_norm.add_(self.norm2.pe_grad_sqnorm(deriv_pre_activ[5]))

        return grad_sq_norm

    def collect_preactivations(self):
        pre_acts = []

        pre_acts.extend(self.self_attn.collect_preactivations())
        pre_acts.append(self.linear1.pre_activation)
        pre_acts.append(self.linear2.pre_activation)
        # pre_acts.append(self.norm1.pre_activation)                        
        # pre_acts.append(self.norm2.pre_activation)

        return pre_acts

    def collect_layers(self):
        layer_lst = []
        layer_lst.extend( self.self_attn.collect_layers() )
        layer_lst.append( self.linear1 )
        layer_lst.append( self.linear2 )
        return layer_lst
    
    def pe_grad_gradcomp(self, deriv_pre_activ, per_sample=True):
        result_attn = self.self_attn.pe_grad_gradcomp(deriv_pre_activ[:2])
        result_linear1 = self.linear1.pe_grad_gradcomp(deriv_pre_activ[2])
        result_linear2 = self.linear1.pe_grad_gradcomp(deriv_pre_activ[3])
        return result_attn + [result_linear1] + [result_linear2]



class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src

        for i in range(self.num_layers):
            output = self.layers[i](output, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            output = self.norm(output)

        return output

    def per_example_gradient(self):
        grads = []

        for i in range(self.num_layers):
            grads.extend(self.layers[i].per_example_gradient())

        return grads

    def pe_grad_sqnorm(self, deriv_pre_activ):
        batch_size = deriv_pre_activ[0].size(1)
        device = deriv_pre_activ[0].device
        grad_sq_norm = torch.zeros(batch_size, device=device)

        n_derivs = len(deriv_pre_activ)
        chunk_size = n_derivs // self.num_layers
        chunks = [deriv_pre_activ[i:i+chunk_size] for i in range(0, n_derivs, chunk_size)]

        for i, chunk in enumerate(chunks):
            grad_sq_norm.add_(self.layers[i].pe_grad_sqnorm(chunk))

        return grad_sq_norm

    def collect_preactivations(self):
        pre_acts = []

        for i in range(self.num_layers):
            pre_acts.extend(self.layers[i].collect_preactivations())

        return pre_acts

    def collect_layers(self):
        layer_lst = []

        for i in range(self.num_layers):
            layer_lst.extend(self.layers[i].collect_layers())

        return layer_lst
