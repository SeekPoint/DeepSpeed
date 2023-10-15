# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from ..runtime.pipe import PipelineModule, LayerSpec, TiedLayerSpec
from pydebug import debuginfo, infoTensor

debuginfo(prj='ds', info='pipe __init__')
