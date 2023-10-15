# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .cpu_adagrad import DeepSpeedCPUAdagrad

from pydebug import debuginfo, infoTensor

debuginfo(prj='ds', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')