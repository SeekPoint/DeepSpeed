# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .cpu_adam import DeepSpeedCPUAdam
from .fused_adam import FusedAdam
from pydebug import debuginfo
debuginfo(prj='ds')