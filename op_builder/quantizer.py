# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import CUDAOpBuilder
from pydebug import gd, infoTensor

class QuantizerBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_QUANTIZER"
    NAME = "quantizer"

    def __init__(self, name=None):
        gd.debuginfo(prj="ds")
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        gd.debuginfo(prj="ds")
        return f'deepspeed.ops.quantizer.{self.NAME}_op'

    def sources(self):
        gd.debuginfo(prj="ds")
        return [
            'csrc/quantization/pt_binding.cpp',
            'csrc/quantization/fake_quantizer.cu',
            'csrc/quantization/quantize.cu',
            'csrc/quantization/dequantize.cu',
            'csrc/quantization/swizzled_quantize.cu',
            'csrc/quantization/quant_reduce.cu',
        ]

    def include_paths(self):
        gd.debuginfo(prj="ds")
        return ['csrc/includes']

    def extra_ldflags(self):
        gd.debuginfo(prj="ds")
        return ['-lcurand']
