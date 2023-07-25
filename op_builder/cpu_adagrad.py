# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
from .builder import TorchCPUOpBuilder

from pydebug import debuginfo
class CPUAdagradBuilder(TorchCPUOpBuilder):
    BUILD_VAR = "DS_BUILD_CPU_ADAGRAD"
    NAME = "cpu_adagrad"

    def __init__(self):
        debuginfo(prj='ds', info='CPUAdagradBuilder init')
        super().__init__(name=self.NAME)

    def absolute_name(self):
        debuginfo(prj='ds')
        return f'deepspeed.ops.adagrad.{self.NAME}_op'

    def sources(self):
        if self.build_for_cpu:
            debuginfo(prj='ds')
            return ['csrc/adagrad/cpu_adagrad.cpp']
        
        debuginfo(prj='ds')

        return ['csrc/adagrad/cpu_adagrad.cpp', 'csrc/common/custom_cuda_kernel.cu']

    def libraries_args(self):
        debuginfo(prj='ds')
        args = super().libraries_args()
        if self.build_for_cpu:
            debuginfo(prj='ds')
            return args

        if not self.is_rocm_pytorch():
            debuginfo(prj='ds')
            args += ['curand']

        debuginfo(prj='ds', info=' args: ' + str(args))
        
        return args

    def include_paths(self):
        import torch
        if self.build_for_cpu:
            debuginfo(prj='ds')
            CUDA_INCLUDE = []
        elif not self.is_rocm_pytorch():
            debuginfo(prj='ds')
            CUDA_INCLUDE = [os.path.join(torch.utils.cpp_extension.CUDA_HOME, "include")]
        else:
            debuginfo(prj='ds')
            CUDA_INCLUDE = [
                os.path.join(torch.utils.cpp_extension.ROCM_HOME, "include"),
                os.path.join(torch.utils.cpp_extension.ROCM_HOME, "include", "rocrand"),
                os.path.join(torch.utils.cpp_extension.ROCM_HOME, "include", "hiprand"),
            ]
        return ['csrc/includes'] + CUDA_INCLUDE
