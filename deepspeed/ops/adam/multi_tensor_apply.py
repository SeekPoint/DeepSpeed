# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Copyright NVIDIA/apex
This file is adapted from NVIDIA/apex, commit a109f85
"""
from pydebug import gd, infoTensor

class MultiTensorApply(object):

    def __init__(self, chunk_size):
        # gd.debuginfo(prj="ds", info=f"C:{self.__class__.__name__}")
        self.chunk_size = chunk_size

    def __call__(self, op, noop_flag_buffer, tensor_lists, *args):
        # gd.debuginfo(prj="ds", info=self.__class__.__name__)
        return op(self.chunk_size, noop_flag_buffer, tensor_lists, *args)
