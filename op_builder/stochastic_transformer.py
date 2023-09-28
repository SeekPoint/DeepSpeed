# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .transformer import TransformerBuilder
from pydebug import debuginfo

class StochasticTransformerBuilder(TransformerBuilder):
    BUILD_VAR = "DS_BUILD_STOCHASTIC_TRANSFORMER"
    NAME = "stochastic_transformer"

    def __init__(self):
        debuginfo(prj='ds', info='StochasticTransformerBuilder init')
        super().__init__(name=self.NAME)

    def absolute_name(self):
        debuginfo(prj='ds')
        return f'deepspeed.ops.transformer.{self.NAME}_op'

    def nvcc_args(self):
        debuginfo(prj='ds')
        args = super().nvcc_args()
        args.append('-D__STOCHASTIC_MODE__')
        debuginfo(prj='ds', info=' args: ' + str(args))
        return args
