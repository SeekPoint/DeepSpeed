# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .fused_lamb import FusedLamb
from pydebug import gd, infoTensor
gd.debuginfo(prj="ds")