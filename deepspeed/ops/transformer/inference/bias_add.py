# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Optional
import torch
from deepspeed.ops.op_builder import SpatialInferenceBuilder

spatial_cuda_module = None


def nhwc_bias_add(activation: torch.Tensor,
                  bias: torch.Tensor,
                  other: Optional[torch.Tensor] = None,
                  other_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    global spatial_cuda_module
    if spatial_cuda_module is None:
        debuginfo(prj='ds')
        spatial_cuda_module = SpatialInferenceBuilder().load()

    if other is None:
        debuginfo(prj='ds')
        return spatial_cuda_module.nhwc_bias_add(activation, bias)
    elif other_bias is None:
        debuginfo(prj='ds')
        return spatial_cuda_module.nhwc_bias_add_add(activation, bias, other)
    else:
        debuginfo(prj='ds')
        return spatial_cuda_module.nhwc_bias_add_bias_add(activation, bias, other, other_bias)
