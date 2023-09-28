# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from pydebug import debuginfo

class Diffusers2DTransformerConfig():

    def __init__(self, int8_quantization=False):
        debuginfo(prj='ds', info='Diffusers2DTransformerConfig init')
        self.int8_quantization = int8_quantization
