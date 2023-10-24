# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

try:
    # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
    # if successful this also means we're doing a local install and not JIT compile path
    from op_builder import __deepspeed__  # noqa: F401
    from op_builder.builder import OpBuilder
except ImportError:
    from deepspeed.ops.op_builder.builder import OpBuilder

from pydebug import gd, infoTensor
class CPUOpBuilder(OpBuilder):

    def builder(self):

        from torch.utils.cpp_extension import CppExtension as ExtensionBuilder

        compile_args = {'cxx': self.strip_empty_entries(self.cxx_args())}

        gd.debuginfo(prj='ds',
                     info=f'self.absolute_name() = {self.absolute_name()}')
        gd.debuginfo(prj='ds',
                     info=f'self.strip_empty_entries(self.sources()) = {self.strip_empty_entries(self.sources())}')
        gd.debuginfo(prj='ds',
                     info=f'self.strip_empty_entries(self.include_paths()) = {self.strip_empty_entries(self.include_paths())}')
        gd.debuginfo(prj='ds',
                     info=f'self.strip_empty_entries(self.libraries_args()) = {self.strip_empty_entries(self.libraries_args())}')
        gd.debuginfo(prj='ds', info=f'compile_args = {compile_args}')
        gd.debuginfo(prj='ds',
                     info=f'self.strip_empty_entries(self.extra_ldflags()) = {self.strip_empty_entries(self.extra_ldflags())}')

        cpp_ext = ExtensionBuilder(name=self.absolute_name(),
                                   sources=self.strip_empty_entries(self.sources()),
                                   include_dirs=self.strip_empty_entries(self.include_paths()),
                                   libraries=self.strip_empty_entries(self.libraries_args()),
                                   extra_compile_args=compile_args)

        return cpp_ext

    def cxx_args(self):
        gd.debuginfo(prj="ds")
        return ['-O3', '-g', '-Wno-reorder']

    def libraries_args(self):
        gd.debuginfo(prj="ds")
        return []
