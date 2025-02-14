# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import CUDAOpBuilder
from pydebug import debuginfo

class TransformerBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_TRANSFORMER"
    NAME = "transformer"

    def __init__(self, name=None):
        debuginfo(prj='ds', info='TransformerBuilder init')
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        debuginfo(prj='ds')
        return f'deepspeed.ops.transformer.{self.NAME}_op'

    def extra_ldflags(self):
        if not self.is_rocm_pytorch():
            debuginfo(prj='ds')
            return ['-lcurand']
        else:
            debuginfo(prj='ds')
            return []

    def sources(self):
        debuginfo(prj='ds')
        return [
            'csrc/transformer/ds_transformer_cuda.cpp', 'csrc/transformer/cublas_wrappers.cu',
            'csrc/transformer/transform_kernels.cu', 'csrc/transformer/gelu_kernels.cu',
            'csrc/transformer/dropout_kernels.cu', 'csrc/transformer/normalize_kernels.cu',
            'csrc/transformer/softmax_kernels.cu', 'csrc/transformer/general_kernels.cu'
        ]

    def include_paths(self):
        debuginfo(prj='ds')
        includes = ['csrc/includes']
        if self.is_rocm_pytorch():
            debuginfo(prj='ds')
            from torch.utils.cpp_extension import ROCM_HOME
            includes += ['{}/hiprand/include'.format(ROCM_HOME), '{}/rocrand/include'.format(ROCM_HOME)]
        return includes
