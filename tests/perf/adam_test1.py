# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed.ops.adam import DeepSpeedCPUAdam
import time
from deepspeed.accelerator import get_accelerator
from pydebug import debuginfo
device = 'cpu'
model_size = 1 * 1024**3
param = torch.nn.Parameter(torch.ones(model_size, device=device))
param_fp16 = torch.nn.Parameter(torch.ones(model_size, dtype=torch.half, device=get_accelerator().device_name(0)))

optimizer = DeepSpeedCPUAdam([param])
#torch.set_num_threads(128)
param.grad = torch.ones(model_size, device=device)
avg = 0
for i in range(100):
    start = time.time()
    optimizer.step(fp16_param_groups=[param_fp16])
    stop = time.time()
    avg += (stop - start)
    param.grad = torch.ones(model_size, device=device) * 2
print("Elapsed Time is ", avg / 100)


'''

ds P: 74044 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: __init__ L#: 64
ds P: 74044 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: put L#: 74
ds P: 74044 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/adam/cpu_adam.py f: __del__ L#: 102
Exception ignored in: <function DeepSpeedCPUAdam.__del__ at 0x7f771de950d0>
Traceback (most recent call last):
  File "/home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/adam/cpu_adam.py", line 105, in __del__
TypeError: 'NoneType' object is not callable

'''