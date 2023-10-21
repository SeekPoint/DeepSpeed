# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .cpu_adam import DeepSpeedCPUAdam
from .fused_adam import FusedAdam
from pydebug import gd, infoTensor
gd.debuginfo(prj='ds', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')