# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .containers import HFGPT2LayerPolicy
from .containers import HFBertLayerPolicy
from .containers import BLOOMLayerPolicy
from .containers import HFGPTJLayerPolicy
from .containers import HFGPTNEOLayerPolicy
from .containers import GPTNEOXLayerPolicy
from .containers import HFOPTLayerPolicy
from .containers import MegatronLayerPolicy
from .containers import HFDistilBertLayerPolicy
from .containers import HFCLIPLayerPolicy
from .containers import LLAMALayerPolicy
from .containers import UNetPolicy
from .containers import VAEPolicy
<<<<<<< HEAD
from pydebug import debuginfo
debuginfo(prj='ds')
=======
from .containers import LLAMA2LayerPolicy
from .containers import InternLMLayerPolicy

>>>>>>> 388c84834fca87465aff8bb8f6d85be88fa82ba6
# transformer-based policies
replace_policies = [
    HFBertLayerPolicy, HFGPTNEOLayerPolicy, GPTNEOXLayerPolicy, HFGPTJLayerPolicy, MegatronLayerPolicy,
    HFGPT2LayerPolicy, BLOOMLayerPolicy, HFOPTLayerPolicy, HFCLIPLayerPolicy, HFDistilBertLayerPolicy,
    LLAMALayerPolicy, LLAMA2LayerPolicy, InternLMLayerPolicy
]

# non-transformer-based policies
generic_policies = [UNetPolicy, VAEPolicy]
