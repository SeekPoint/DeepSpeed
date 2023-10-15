# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .dropping_utils import gpt_sample_tokens, bert_sample_tokens, GatherTokens, ScatterTokens
from pydebug import debuginfo, infoTensor
debuginfo(prj='ds', info='random_ltd __init__')
