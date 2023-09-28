# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
from .builder import TorchCPUOpBuilder
from pydebug import debuginfo

class CPUAdamBuilder(TorchCPUOpBuilder):
    BUILD_VAR = "DS_BUILD_CPU_ADAM"
    NAME = "cpu_adam"

    def __init__(self):
        debuginfo(prj='ds', info = " CPUAdamBuilder init")
        super().__init__(name=self.NAME)

    def absolute_name(self):
        debuginfo(prj='ds', info = f'deepspeed.ops.adam.{self.NAME}_op')
        return f'deepspeed.ops.adam.{self.NAME}_op'

    def sources(self):
        if self.build_for_cpu:
<<<<<<< HEAD
            debuginfo(prj='ds')
            return ['csrc/adam/cpu_adam.cpp']
        
        debuginfo(prj='ds')
=======
            return ['csrc/adam/cpu_adam.cpp', 'csrc/adam/cpu_adam_impl.cpp']
>>>>>>> 388c84834fca87465aff8bb8f6d85be88fa82ba6

        return ['csrc/adam/cpu_adam.cpp', 'csrc/adam/cpu_adam_impl.cpp', 'csrc/common/custom_cuda_kernel.cu']

    def libraries_args(self):
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
