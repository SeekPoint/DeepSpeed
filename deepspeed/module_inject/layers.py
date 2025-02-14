# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed import comm as dist
from torch import nn
from torch.nn import functional as F

from torch.nn.parameter import Parameter
from deepspeed.accelerator import get_accelerator
from pydebug import debuginfo

class LinearAllreduce(nn.Module):

    def __init__(self, weight, bias=None, mp_group=None):
        debuginfo(prj='ds', info='LinearAllreduce init')
        super(LinearAllreduce, self).__init__()
        self.weight = weight
        self.bias = bias
        self.mp_group = mp_group

    def forward(self, input):
        output = torch.matmul(input, self.weight.transpose(-1, -2))
        if self.mp_group is not None:
<<<<<<< HEAD
            debuginfo(prj='ds')
            dist.all_reduce(output, group=self.mp_group)
=======
            dist.inference_all_reduce(output, group=self.mp_group)
>>>>>>> 388c84834fca87465aff8bb8f6d85be88fa82ba6
        if self.bias is not None:
            debuginfo(prj='ds')
            output += self.bias
        return output


class LinearLayer(nn.Module):

    def __init__(self, weight_shape=None, dtype=torch.half, weight=None, bias=None):
        super(LinearLayer, self).__init__()
        if weight is not None:
            debuginfo(prj='ds')
            self.weight = weight
            self.bias = bias
        else:
            debuginfo(prj='ds')
            self.weight = Parameter(
                torch.empty(weight_shape, dtype=dtype, device=get_accelerator().current_device_name()))

            self.bias = Parameter(
                torch.empty(weight_shape[0],
                            dtype=dtype,
                            device=get_accelerator().current_device_name())) \
                if bias is not None else None

    def forward(self, input):
        debuginfo(prj='ds')
        output = torch.matmul(input, self.weight.transpose(-1, -2))
        if self.bias is not None:
            debuginfo(prj='ds')
            output += self.bias
        return output


class Normalize(nn.Module):

    def __init__(self, dim=None, dtype=torch.float, eps=1e-5, weight=None, bias=None):
        super(Normalize, self).__init__()
        if weight is not None:
            debuginfo(prj='ds')
            self.weight = weight
            self.bias = bias
        else:
            debuginfo(prj='ds')
            self.norm = nn.LayerNorm(dim, eps=eps).to(dtype).to(get_accelerator().current_device_name())
            self.weight = self.norm.weight
            self.bias = self.norm.bias

        self.eps = eps

    def forward(self, input):
        debuginfo(prj='ds')
        return nn.functional.layer_norm(input, input.shape[-1:], self.weight, self.bias, eps=self.eps)


class EmbeddingLayer(nn.Module):

    def __init__(self, weight_shape=None, dtype=torch.half, weight=None, bias=None):
        super(EmbeddingLayer, self).__init__()
        if weight is None:
            debuginfo(prj='ds')
            self.weight = Parameter(
                torch.empty(weight_shape[0],
                            weight_shape[1],
                            dtype=dtype,
                            device=get_accelerator().current_device_name()))
        else:
            debuginfo(prj='ds')
            self.weight = weight

    def forward(self, input):
        debuginfo(prj='ds')
        return F.embedding(input, self.weight)


class OPTEmbedding(EmbeddingLayer):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, weight_shape=None, weight=None, bias=None):
        debuginfo(prj='ds')
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(weight_shape, weight=weight)

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):
        debuginfo(prj='ds')
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()

        # create positions depending on attention_mask
        positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1

        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length:]

        return super().forward(positions + self.offset)


class RMSNormalize(nn.Module):

    def __init__(self, dim=None, dtype=torch.float, eps=1e-5, weight=None):
        super(RMSNormalize, self).__init__()
        if weight is not None:
            debuginfo(prj='ds')
            self.weight = weight
        else:
            debuginfo(prj='ds')
            self.weight = nn.Parameter(torch.ones(dim, dtype=dtype, device=get_accelerator().current_device_name()))

        self.eps = eps

    def forward(self, hidden_states):
        debuginfo(prj='ds')
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return hidden_states * self.weight
