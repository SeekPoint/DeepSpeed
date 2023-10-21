# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.model_implementations.transformers.ds_transformer import DeepSpeedTransformerInference

from pydebug import debuginfo, infoTensor
class DeepSpeedGPTInference(DeepSpeedTransformerInference):
    """Initialize the DeepSpeed GPT Transformer Layer.
    """

    def __init__(self,
                 config,
                 mp_group=None,
                 quantize_scales=None,
                 quantize_groups=1,
                 merge_count=1,
                 mlp_extra_grouping=False):
        gd.debuginfo(prj='ds', info=self.__class__.__name__)
        super().__init__(config, mp_group, quantize_scales, quantize_groups, merge_count, mlp_extra_grouping)
