# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .builder import CUDAOpBuilder

from pydebug import debuginfo, infoTensor
class RandomLTDBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_RANDOM_LTD"
    NAME = "random_ltd"

    def __init__(self, name=None):
        debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
        return f'deepspeed.ops.{self.NAME}_op'

    def extra_ldflags(self):
        if not self.is_rocm_pytorch():
            debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
            return ['-lcurand']
        else:
            debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
            return []

    def sources(self):
        debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
        return [
            'csrc/random_ltd/pt_binding.cpp', 'csrc/random_ltd/gather_scatter.cu',
            'csrc/random_ltd/slice_attn_masks.cu', 'csrc/random_ltd/token_sort.cu'
        ]

    def include_paths(self):
        includes = ['csrc/includes']
        if self.is_rocm_pytorch():
            debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
            from torch.utils.cpp_extension import ROCM_HOME
            includes += ['{}/hiprand/include'.format(ROCM_HOME), '{}/rocrand/include'.format(ROCM_HOME)]
        debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
        return includes
