# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from transformers import AutoModelForCausalLM
import deepspeed
import argparse
from deepspeed.accelerator import get_accelerator
from pydebug import debuginfo
deepspeed.runtime.utils.see_memory_usage('pre test', force=True)

model = AutoModelForCausalLM.from_pretrained('/home/amd00/hf_model/opt-350m/').half().to(get_accelerator().device_name())
parser = argparse.ArgumentParser()
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

deepspeed.runtime.utils.see_memory_usage('post test', force=True)

m, _, _, _ = deepspeed.initialize(model=model, args=args)

m.eval()
input = torch.ones(1, 16, device='cuda', dtype=torch.long)
out = m(input)

m.train()
out = m(input)
print(out['logits'], out['logits'].norm())


'''

必须指定配置！！！！

ds P: 25887 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: put L#: 74
(ds_chat_py39) amd00@MZ32-00:~/yk_repo/ds/DeepSpeed/tests/hybrid_engine$ pytest hybrid_engine_test.py --deepspeed_config hybrid_engine_config.json
ERROR: usage: pytest [options] [file_or_dir] [file_or_dir] [...]
pytest: error: unrecognized arguments: --deepspeed_config hybrid_engine_config.json
  inifile: /home/amd00/yk_repo/ds/DeepSpeed/tests/pytest.ini
  rootdir: /home/amd00/yk_repo/ds/DeepSpeed/tests

(ds_chat_py39) amd00@MZ32-00:~/yk_repo/ds/DeepSpeed/tests/hybrid_engine$


mpi4py install:
先conda install gcc_linux-64
然后python3 -m pip install mpi4py
可能需要conda install -c conda-forge gxx_linux-64
'''
