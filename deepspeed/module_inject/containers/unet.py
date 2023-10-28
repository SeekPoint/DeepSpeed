# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from torch.nn.parameter import Parameter

from ..policy import DSPolicy
from ...model_implementations.diffusers.unet import DSUNet
from pydebug import gd, infoTensor

class UNetPolicy(DSPolicy):

    def __init__(self):
        gd.debuginfo(prj='ds', info=f"C:{self.__class__.__name__}")
        super().__init__()
        try:
            gd.debuginfo(prj="ds")
            import diffusers
            self._orig_layer_class = diffusers.models.unet_2d_condition.UNet2DConditionModel
        except ImportError:
            gd.debuginfo(prj="ds")
            self._orig_layer_class = None

    def match(self, module):
        gd.debuginfo(prj="ds")
        return isinstance(module, self._orig_layer_class)

    def match_replaced(self, module):
        gd.debuginfo(prj="ds")
        return isinstance(module, DSUNet)

    def apply(self, module, enable_cuda_graph=True):
        gd.debuginfo(prj="ds")
        # TODO(cmikeh2): Enable cuda graph should be an inference configuration
        return DSUNet(module, enable_cuda_graph=enable_cuda_graph)

    def attention(self, client_module):
        gd.debuginfo(prj="ds")
        qw = client_module.to_q.weight
        kw = client_module.to_k.weight
        vw = client_module.to_v.weight

        if qw.shape[1] == kw.shape[1]:
            gd.debuginfo(prj="ds")
            qkvw = Parameter(torch.cat((qw, kw, vw), dim=0), requires_grad=False)

            return qkvw, \
                   client_module.to_out[0].weight, \
                   client_module.to_out[0].bias, \
                   qw.shape[-1], \
                   client_module.heads
        else:
            gd.debuginfo(prj="ds")
            #return None
            #kvw = Parameter(torch.cat((kw, vw), dim=0), requires_grad=False)
            return qw, \
                   kw, vw, \
                   client_module.to_out[0].weight, \
                   client_module.to_out[0].bias, \
                   qw.shape[-1], \
                   client_module.heads
