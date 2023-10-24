# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from deepspeed.ops.op_builder import QuantizerBuilder
from pydebug import gd, infoTensor
# Cuda modules will be imported if needed
quantizer_cuda_module = None


def ds_quantizer(input, groups=1, bit_num=8, sr=False, asym=False):
    # Load cuda modules if needed
    global quantizer_cuda_module
    if quantizer_cuda_module is None:
        gd.debuginfo(prj="ds")
        quantizer_cuda_module = QuantizerBuilder().load()
    if sr:
        if asym:
            gd.debuginfo(prj="ds")
            quantize_func = quantizer_cuda_module.ds_sr_quantize_asym_fp16 if input.dtype == torch.half else quantizer_cuda_module.ds_sr_quantize_asym_fp32
        else:
            gd.debuginfo(prj="ds")
            quantize_func = quantizer_cuda_module.ds_sr_quantize_fp16 if input.dtype == torch.half else quantizer_cuda_module.ds_sr_quantize_fp32
    else:
        if asym:
            gd.debuginfo(prj="ds")
            quantize_func = quantizer_cuda_module.ds_quantize_asym_fp16 if input.dtype == torch.half else quantizer_cuda_module.ds_quantize_asym_fp32
        else:
            gd.debuginfo(prj="ds")
            quantize_func = quantizer_cuda_module.ds_quantize_fp16 if input.dtype == torch.half else quantizer_cuda_module.ds_quantize_fp32
    return quantize_func(input, groups, bit_num)
