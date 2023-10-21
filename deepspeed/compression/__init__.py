# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .compress import init_compression, redundancy_clean
from .scheduler import compression_scheduler
from .helper import convert_conv1d_to_linear
from pydebug import gd, infoTensor
gd.debuginfo(prj='ds', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')