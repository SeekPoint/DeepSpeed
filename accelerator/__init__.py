# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .abstract_accelerator import DeepSpeedAccelerator
from .real_accelerator import get_accelerator, set_accelerator
from pydebug import debuginfo, infoTensor

gd.debuginfo(prj='ds', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')