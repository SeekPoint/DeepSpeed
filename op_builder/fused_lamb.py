# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import CUDAOpBuilder

import sys
from pydebug import gd, infoTensor

class FusedLambBuilder(CUDAOpBuilder):
    BUILD_VAR = 'DS_BUILD_FUSED_LAMB'
    NAME = "fused_lamb"

    def __init__(self):
        gd.debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
        super().__init__(name=self.NAME)

    def absolute_name(self):
        gd.debuginfo(prj='ds', info=f'deepspeed.ops.lamb.{self.NAME}_op')
        return f'deepspeed.ops.lamb.{self.NAME}_op'

    def sources(self):
        gd.debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
        return ['csrc/lamb/fused_lamb_cuda.cpp', 'csrc/lamb/fused_lamb_cuda_kernel.cu']

    def include_paths(self):
        gd.debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
        return ['csrc/includes']

    def cxx_args(self):
        gd.debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
        args = super().cxx_args()
        return args + self.version_dependent_macros()

    def nvcc_args(self):
        nvcc_flags = ['-O3'] + self.version_dependent_macros()
        if self.is_rocm_pytorch():
            gd.debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
            ROCM_MAJOR, ROCM_MINOR = self.installed_rocm_version()
            nvcc_flags += ['-DROCM_VERSION_MAJOR=%s' % ROCM_MAJOR, '-DROCM_VERSION_MINOR=%s' % ROCM_MINOR]
        else:
            gd.debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
            nvcc_flags.extend(
                ['-allow-unsupported-compiler' if sys.platform == "win32" else '', '-lineinfo', '--use_fast_math'] +
                self.compute_capability_args())
        gd.debuginfo(prj='ds', info=' nvcc_flags: ' + str(nvcc_flags))
        return nvcc_flags
