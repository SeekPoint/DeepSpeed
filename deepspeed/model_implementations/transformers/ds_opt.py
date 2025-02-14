# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.model_implementations.transformers.ds_transformer import DeepSpeedTransformerInference

from pydebug import debuginfo
class DeepSpeedOPTInference(DeepSpeedTransformerInference):
    """Initialize the DeepSpeed OPT Transformer Layer.
    """

    def __init__(self,
                 config,
                 mp_group=None,
                 quantize_scales=None,
                 quantize_groups=1,
                 merge_count=1,
                 mlp_extra_grouping=False):
        debuginfo(prj='ds', info='DeepSpeedOPTInference init')
        super().__init__(config, mp_group, quantize_scales, quantize_groups, merge_count, mlp_extra_grouping)
