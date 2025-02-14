# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from torch.nn.parameter import Parameter

from ..policy import DSPolicy
from ...model_implementations.diffusers.unet import DSUNet
from pydebug import debuginfo

class UNetPolicy(DSPolicy):

    def __init__(self):
        debuginfo(prj='ds', info='UNetPolicy init')
        super().__init__()
        try:
            debuginfo(prj='ds')
            import diffusers
            self._orig_layer_class = diffusers.models.unet_2d_condition.UNet2DConditionModel
        except ImportError:
            debuginfo(prj='ds')
            self._orig_layer_class = None

    def match(self, module):
        debuginfo(prj='ds')
        return isinstance(module, self._orig_layer_class)

    def match_replaced(self, module):
        debuginfo(prj='ds')
        return isinstance(module, DSUNet)

    def apply(self, module, enable_cuda_graph=True):
        debuginfo(prj='ds')
        # TODO(cmikeh2): Enable cuda graph should be an inference configuration
        return DSUNet(module, enable_cuda_graph=enable_cuda_graph)

    def attention(self, client_module):
        debuginfo(prj='ds')
        qw = client_module.to_q.weight
        kw = client_module.to_k.weight
        vw = client_module.to_v.weight

        if qw.shape[1] == kw.shape[1]:
            debuginfo(prj='ds')
            qkvw = Parameter(torch.cat((qw, kw, vw), dim=0), requires_grad=False)

            return qkvw, \
                   client_module.to_out[0].weight, \
                   client_module.to_out[0].bias, \
                   qw.shape[-1], \
                   client_module.heads
        else:
            debuginfo(prj='ds')
            #return None
            #kvw = Parameter(torch.cat((kw, vw), dim=0), requires_grad=False)
            return qw, \
                   kw, vw, \
                   client_module.to_out[0].weight, \
                   client_module.to_out[0].bias, \
                   qw.shape[-1], \
                   client_module.heads
