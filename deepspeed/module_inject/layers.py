# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed import comm as dist
from torch import nn
from torch.nn import functional as F

from torch.nn.parameter import Parameter
from deepspeed.accelerator import get_accelerator
from pydebug import gd, infoTensor

class LinearAllreduce(nn.Module):

    def __init__(self, weight, bias=None, mp_group=None):
        gd.debuginfo(prj='ds', info=f"C:{self.__class__.__name__}")
        super(LinearAllreduce, self).__init__()
        self.weight = weight
        self.bias = bias
        self.mp_group = mp_group

    def forward(self, input):
        output = torch.matmul(input, self.weight.transpose(-1, -2))
        if self.mp_group is not None:
            gd.debuginfo(prj="ds")
            dist.all_reduce(output, group=self.mp_group)
        if self.bias is not None:
            gd.debuginfo(prj="ds")
            output += self.bias
        gd.debuginfo(prj='ds', info=f"output={output}")
        return output


class LinearLayer(nn.Module):

    def __init__(self, weight_shape=None, dtype=torch.half, weight=None, bias=None):
        super(LinearLayer, self).__init__()
        if weight is not None:
            gd.debuginfo(prj="ds")
            self.weight = weight
            self.bias = bias
        else:
            gd.debuginfo(prj="ds")
            self.weight = Parameter(
                torch.empty(weight_shape, dtype=dtype, device=get_accelerator().current_device_name()))

            self.bias = Parameter(
                torch.empty(weight_shape[0],
                            dtype=dtype,
                            device=get_accelerator().current_device_name())) \
                if bias is not None else None

    def forward(self, input):
        output = torch.matmul(input, self.weight.transpose(-1, -2))
        if self.bias is not None:
            gd.debuginfo(prj="ds")
            output += self.bias
        gd.debuginfo(prj='ds', info=f"output={infoTensor(output)}")
        return output


class Normalize(nn.Module):

    def __init__(self, dim=None, dtype=torch.float, eps=1e-5, weight=None, bias=None):
        super(Normalize, self).__init__()
        if weight is not None:
            gd.debuginfo(prj="ds")
            self.weight = weight
            self.bias = bias
        else:
            gd.debuginfo(prj="ds")
            self.norm = nn.LayerNorm(dim, eps=eps).to(dtype).to(get_accelerator().current_device_name())
            self.weight = self.norm.weight
            self.bias = self.norm.bias

        self.eps = eps

    def forward(self, input):
        tmp = nn.functional.layer_norm(input, input.shape[-1:], self.weight, self.bias, eps=self.eps)
        gd.debuginfo(prj="ds", info=f"T:tmp-A={infoTensor(tmp)}")
        return tmp


class EmbeddingLayer(nn.Module):

    def __init__(self, weight_shape=None, dtype=torch.half, weight=None, bias=None):
        super(EmbeddingLayer, self).__init__()
        if weight is None:
            gd.debuginfo(prj="ds")
            self.weight = Parameter(
                torch.empty(weight_shape[0],
                            weight_shape[1],
                            dtype=dtype,
                            device=get_accelerator().current_device_name()))
        else:
            gd.debuginfo(prj="ds")
            self.weight = weight

    def forward(self, input):
        tmp = F.embedding(input, self.weight)
        gd.debuginfo(prj="ds", info=f"T:tmp-B={infoTensor(tmp)}")
        return tmp


class OPTEmbedding(EmbeddingLayer):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, weight_shape=None, weight=None, bias=None):
        gd.debuginfo(prj="ds")
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(weight_shape, weight=weight)

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):

        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()

        # create positions depending on attention_mask
        positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1

        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length:]
        gd.debuginfo(prj="ds", info=f"positions={infoTensor(positions)}")

        tmp = positions + self.offset
        gd.debuginfo(prj="ds", info=f"T:tmp-C={infoTensor(tmp)}")

        return super().forward(tmp)


class RMSNormalize(nn.Module):

    def __init__(self, dim=None, dtype=torch.float, eps=1e-5, weight=None):
        super(RMSNormalize, self).__init__()
        if weight is not None:
            self.weight = weight
            gd.debuginfo(prj="ds", info=f"self.weight={self.weight}")
        else:
            self.weight = nn.Parameter(torch.ones(dim, dtype=dtype, device=get_accelerator().current_device_name()))
            gd.debuginfo(prj="ds", info=f"self.weight={self.weight}")

        self.eps = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        # print(self.weight)
        gd.debuginfo(prj="ds", info=f"self.weight={self.weight}")
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        tmp = hidden_states * self.weight
        gd.debuginfo(prj="ds", info=f"tmp={tmp}")

        return tmp
