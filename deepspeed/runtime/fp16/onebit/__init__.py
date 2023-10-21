# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .adam import OnebitAdam
from .lamb import OnebitLamb
from .zoadam import ZeroOneAdam
from pydebug import gd, infoTensor
gd.debuginfo(prj='ds', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')