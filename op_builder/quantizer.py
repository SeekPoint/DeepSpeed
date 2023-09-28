# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import CUDAOpBuilder
from pydebug import debuginfo

class QuantizerBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_QUANTIZER"
    NAME = "quantizer"

    def __init__(self, name=None):
        debuginfo(prj='ds', info='QuantizerBuilder init')
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        debuginfo(prj='ds')
        return f'deepspeed.ops.quantizer.{self.NAME}_op'

    def sources(self):
        debuginfo(prj='ds')
        return [
            'csrc/quantization/pt_binding.cpp',
            'csrc/quantization/fake_quantizer.cu',
            'csrc/quantization/quantize.cu',
            'csrc/quantization/quantize_int4.cu',
            'csrc/quantization/dequantize.cu',
            'csrc/quantization/swizzled_quantize.cu',
            'csrc/quantization/quant_reduce.cu',
        ]

    def include_paths(self):
        debuginfo(prj='ds')
        return ['csrc/includes']

    def extra_ldflags(self):
        debuginfo(prj='ds')
        return ['-lcurand']
