# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import distutils.spawn
import subprocess

from .builder import OpBuilder
from pydebug import debuginfo

class AsyncIOBuilder(OpBuilder):
    BUILD_VAR = "DS_BUILD_AIO"
    NAME = "async_io"

    def __init__(self):
        debuginfo(prj='ds', info='AsyncIOBuilder init')
        super().__init__(name=self.NAME)

    def absolute_name(self):
        debuginfo(prj='ds')
        return f'deepspeed.ops.aio.{self.NAME}_op'

    def sources(self):
        debuginfo(prj='ds')
        return [
            'csrc/aio/py_lib/deepspeed_py_copy.cpp', 'csrc/aio/py_lib/py_ds_aio.cpp',
            'csrc/aio/py_lib/deepspeed_py_aio.cpp', 'csrc/aio/py_lib/deepspeed_py_aio_handle.cpp',
            'csrc/aio/py_lib/deepspeed_aio_thread.cpp', 'csrc/aio/common/deepspeed_aio_utils.cpp',
            'csrc/aio/common/deepspeed_aio_common.cpp', 'csrc/aio/common/deepspeed_aio_types.cpp',
            'csrc/aio/py_lib/deepspeed_pin_tensor.cpp'
        ]

    def include_paths(self):
        debuginfo(prj='ds')
        return ['csrc/aio/py_lib', 'csrc/aio/common']

    def cxx_args(self):
        debuginfo(prj='ds')
        # -O0 for improved debugging, since performance is bound by I/O
        CPU_ARCH = self.cpu_arch()
        SIMD_WIDTH = self.simd_width()
        return [
            '-g',
            '-Wall',
            '-O0',
            '-std=c++14',
            '-shared',
            '-fPIC',
            '-Wno-reorder',
            CPU_ARCH,
            '-fopenmp',
            SIMD_WIDTH,
            '-laio',
        ]

    def extra_ldflags(self):
        debuginfo(prj='ds')
        return ['-laio']

    def check_for_libaio_pkg(self):
        debuginfo(prj='ds')
        libs = dict(
            dpkg=["-l", "libaio-dev", "apt"],
            pacman=["-Q", "libaio", "pacman"],
            rpm=["-q", "libaio-devel", "yum"],
        )

        found = False
        for pkgmgr, data in libs.items():
            flag, lib, tool = data
            debuginfo(prj='ds', info= ' flag: ' + str(flag) + ' lib: ' + str(lib) + ' tool: ' + str(tool))
            path = distutils.spawn.find_executable(pkgmgr)
            debuginfo(prj='ds', info= ' path: ' + str(path))
            if path is not None:
                cmd = f"{pkgmgr} {flag} {lib}"
                result = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                if result.wait() == 0:
                    found = True
                else:
                    self.warning(f"{self.NAME}: please install the {lib} package with {tool}")
                break
        return found

    def is_compatible(self, verbose=True):
        debuginfo(prj='ds')
        # Check for the existence of libaio by using distutils
        # to compile and link a test program that calls io_submit,
        # which is a function provided by libaio that is used in the async_io op.
        # If needed, one can define -I and -L entries in CFLAGS and LDFLAGS
        # respectively to specify the directories for libaio.h and libaio.so.
        aio_compatible = self.has_function('io_submit', ('aio', ))
        if verbose and not aio_compatible:
            self.warning(f"{self.NAME} requires the dev libaio .so object and headers but these were not found.")

            # Check for the libaio package via known package managers
            # to print suggestions on which package to install.
            self.check_for_libaio_pkg()

            self.warning(
                "If libaio is already installed (perhaps from source), try setting the CFLAGS and LDFLAGS environment variables to where it can be found."
            )
        return super().is_compatible(verbose) and aio_compatible
