# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
from .builder import TorchCPUOpBuilder

from pydebug import gd, infoTensor
class CPUAdagradBuilder(TorchCPUOpBuilder):
    BUILD_VAR = "DS_BUILD_CPU_ADAGRAD"
    NAME = "cpu_adagrad"

    def __init__(self):
        gd.debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
        super().__init__(name=self.NAME)

    def absolute_name(self):
        gd.debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
        return f'deepspeed.ops.adagrad.{self.NAME}_op'

    def sources(self):
        if self.build_for_cpu:
            gd.debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
            return ['csrc/adagrad/cpu_adagrad.cpp']
        
        gd.debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')

        return ['csrc/adagrad/cpu_adagrad.cpp', 'csrc/common/custom_cuda_kernel.cu']

    def libraries_args(self):
        gd.debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
        args = super().libraries_args()
        if self.build_for_cpu:
            gd.debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
            return args

        if not self.is_rocm_pytorch():
            gd.debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
            args += ['curand']

        gd.debuginfo(prj='ds', info=' args: ' + str(args))
        
        return args

    def include_paths(self):
        import torch
        if self.build_for_cpu:
            gd.debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
            CUDA_INCLUDE = []
        elif not self.is_rocm_pytorch():
            gd.debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
            CUDA_INCLUDE = [os.path.join(torch.utils.cpp_extension.CUDA_HOME, "include")]
        else:
            gd.debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
            CUDA_INCLUDE = [
                os.path.join(torch.utils.cpp_extension.ROCM_HOME, "include"),
                os.path.join(torch.utils.cpp_extension.ROCM_HOME, "include", "rocrand"),
                os.path.join(torch.utils.cpp_extension.ROCM_HOME, "include", "hiprand"),
            ]
        return ['csrc/includes'] + CUDA_INCLUDE
