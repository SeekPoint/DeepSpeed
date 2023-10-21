# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from pydebug import debuginfo, infoTensor

class Diffusers2DTransformerConfig():

    def __init__(self, int8_quantization=False):
        gd.debuginfo(prj='ds', info=self.__class__.__name__)
        self.int8_quantization = int8_quantization
