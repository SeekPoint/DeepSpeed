# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from ..config import DeepSpeedInferenceConfig

from deepspeed.ops.op_builder import InferenceBuilder
from pydebug import gd, infoTensor

class BaseOp(torch.nn.Module):
    inference_module = None

    def __init__(self, config: DeepSpeedInferenceConfig):
        gd.debuginfo(prj='ds', info=f"C:{self.__class__.__name__}")
        super(BaseOp, self).__init__()
        self.config = config
        if BaseOp.inference_module is None:
            builder = InferenceBuilder()
            BaseOp.inference_module = builder.load()
