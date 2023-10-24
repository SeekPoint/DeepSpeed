# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import CUDAOpBuilder

import sys
from pydebug import gd, infoTensor

class FusedAdamBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_FUSED_ADAM"
    NAME = "fused_adam"

    def __init__(self):
        gd.debuginfo(prj="ds", info=self.__class__.__name__, level = 2)
        super().__init__(name=self.NAME)

    def absolute_name(self):
        gd.debuginfo(prj="ds", info = f'abs name is: deepspeed.ops.adam.{self.NAME}_op', level = 2)
        return f'deepspeed.ops.adam.{self.NAME}_op'

    def sources(self):
        gd.debuginfo(prj="ds", info=f"source files: 'csrc/adam/fused_adam_frontend.cpp,\
                                                      'csrc/adam/multi_tensor_adam.cu", level = 2)
        return ['csrc/adam/fused_adam_frontend.cpp', 'csrc/adam/multi_tensor_adam.cu']

    def include_paths(self):
        gd.debuginfo(prj="ds", info=self.__class__.__name__)
        return ['csrc/includes', 'csrc/adam']

    def cxx_args(self):
        gd.debuginfo(prj="ds", info=self.__class__.__name__)
        args = super().cxx_args()
        return args + self.version_dependent_macros()

    def nvcc_args(self):
        nvcc_flags = ['-O3'] + self.version_dependent_macros()
        if not self.is_rocm_pytorch():
            nvcc_flags.extend(
                ['-allow-unsupported-compiler' if sys.platform == "win32" else '', '-lineinfo', '--use_fast_math'] +
                self.compute_capability_args())
        gd.debuginfo(prj='ds', info=' nvcc_flags: ' + str(nvcc_flags))
        return nvcc_flags
