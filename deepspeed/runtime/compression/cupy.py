# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import cupy
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
from pydebug import debuginfo

class CupyBackend(object):

    def __init__(self):
        debuginfo(prj='ds', info='CupyBackend init')
        pass

    def torch2cupy(self, tensor):
        debuginfo(prj='ds')
        return cupy.fromDlpack(to_dlpack(tensor))

    def cupy2torch(self, cupy_tensor):
        debuginfo(prj='ds')
        return from_dlpack(cupy_tensor.toDlpack())

    def compress_by_chunk(self, cupy_bool_tensor, num_chunks):
        debuginfo(prj='ds')
        packed_sign = cupy.packbits(cupy_bool_tensor)
        sign_list_packed = cupy.split(packed_sign, num_chunks)
        cupy.cuda.get_current_stream().synchronize()
        return sign_list_packed
