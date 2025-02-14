# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .abstract_accelerator import DeepSpeedAccelerator
from .real_accelerator import get_accelerator, set_accelerator
from pydebug import debuginfo

debuginfo(prj='ds', info='accelerate __init__')  