# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
from .builder import CPUOpBuilder
from pydebug import debuginfo, infoTensor

class CCLCommBuilder(CPUOpBuilder):
    BUILD_VAR = "DS_BUILD_CCL_COMM"
    NAME = "deepspeed_ccl_comm"

    def __init__(self, name=None):
        debuginfo(prj='ds', info='CCLCommBuilder init' + str(self.name))
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        debuginfo(prj='ds', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
        return f'deepspeed.ops.comm.{self.NAME}_op'

    def sources(self):
        debuginfo(prj='ds', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
        return ['csrc/cpu/comm/ccl.cpp']

    def include_paths(self):
        debuginfo(prj='ds', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
        includes = ['csrc/cpu/includes']
        return includes

    def is_compatible(self, verbose=True):
        debuginfo(prj='ds', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
        # TODO: add soft compatibility check for private binary release.
        #  a soft check, as in we know it can be trivially changed.
        return super().is_compatible(verbose)

    def extra_ldflags(self):
        ccl_root_path = os.environ.get("CCL_ROOT")
        if ccl_root_path == None:
            raise ValueError(
                "Didn't find CCL_ROOT, install oneCCL from https://github.com/oneapi-src/oneCCL and source its environment variable"
            )
            return []
        else:
            debuginfo(prj='ds', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
            return ['-lccl', f'-L{ccl_root_path}/lib']
