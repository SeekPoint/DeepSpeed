# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Copyright NVIDIA/apex
This file is adapted from NVIDIA/apex, commit a109f85
"""
from pydebug import debuginfo, infoTensor

class MultiTensorApply(object):

    def __init__(self, chunk_size):
        debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
        self.chunk_size = chunk_size

    def __call__(self, op, noop_flag_buffer, tensor_lists, *args):
        debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
        return op(self.chunk_size, noop_flag_buffer, tensor_lists, *args)
