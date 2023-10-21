# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import torch
import deepspeed
from deepspeed.accelerator import get_accelerator

from pydebug import debuginfo, infoTensor
class OneLayerNet(torch.nn.Module):

    def __init__(self, D_in, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        gd.debuginfo(prj='dsUT', info='C:' + self.__class__.__name__)
        super(OneLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        gd.debuginfo(prj='dsUT', info='C:' + self.__class__.__name__)
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear1(h_relu)
        return y_pred


def test_literal_device():
    gd.debuginfo(prj='dsUT')
    model = OneLayerNet(128, 128)

    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8088'
    os.environ['LOCAL_RANK'] = '0'
    deepspeed.init_distributed(get_accelerator().communication_backend_name())
    deepspeed.initialize(model=model, config='ds_config.json')
    string = get_accelerator().device_name()  #'xpu' or 'cuda'
    string0 = get_accelerator().device_name(0)  #'xpu:0' or 'cuda:0'
    string1 = get_accelerator().device_name(1)  #'xpu:1' or 'cuda:1'
    assert string == 'xpu' or string == 'cuda'
    assert string0 == 'xpu:0' or string0 == 'cuda:0'
    assert string1 == 'xpu:1' or string1 == 'cuda:1'


'''

(ds_chat_py39) amd00@MZ32-00:~/yk_repo/ds/DeepSpeed/tests/accelerator$


accelerator/test_ds_init.py::test_literal_device
  /home/amd00/anaconda3/envs/ds_chat_py39/lib/python3.9/site-packages/pkg_resources/__init__.py:2871: DeprecationWarning: Deprecated call to `pkg_resources.declare_namespace('sphinxcontrib')`.
  Implementing implicit namespace packages (as specified in PEP 420) is preferred to `pkg_resources.declare_namespace`. See https://setuptools.pypa.io/en/latest/references/keywords.html#keyword-namespace-packages
    declare_namespace(pkg)

accelerator/test_ds_init.py::test_literal_device
  /home/amd00/anaconda3/envs/ds_chat_py39/lib/python3.9/site-packages/pkg_resources/__init__.py:2871: DeprecationWarning: Deprecated call to `pkg_resources.declare_namespace('zope')`.
  Implementing implicit namespace packages (as specified in PEP 420) is preferred to `pkg_resources.declare_namespace`. See https://setuptools.pypa.io/en/latest/references/keywords.html#keyword-namespace-packages
    declare_namespace(pkg)

accelerator/test_ds_init.py::test_literal_device
  /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/adam/fused_adam.py:97: UserWarning: The torch.cuda.*DtypeTensor constructors are no longer recommended. It's best to use methods such as torch.tensor(data, dtype=*, device='cuda') to create tensors. (Triggered internally at ../torch/csrc/tensor/python_tensor.cpp:83.)
    self._dummy_overflow_buf = get_accelerator().IntTensor([0])

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=================================================================== slowest durations ====================================================================
3.90s call     accelerator/test_ds_init.py::test_literal_device

(2 durations < 1s hidden.  Use -vv to show these durations.)
============================================================ 1 passed, 14 warnings in 30.33s =============================================================
ds P: 25529 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: _default_cache_dir L#: 21
ds P: 25529 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: __init__ L#: 61
ds P: 25529 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: __init__ L#: 64
ds P: 25529 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: load L#: 83
ds P: 25529 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: _default_cache_dir L#: 21
ds P: 25529 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: __init__ L#: 61
ds P: 25529 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: __init__ L#: 64
ds P: 25529 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: put L#: 74
ds P: 25529 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: _default_cache_dir L#: 21
ds P: 25529 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: __init__ L#: 61
ds P: 25529 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: __init__ L#: 64
ds P: 25529 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: load L#: 83
ds P: 25529 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: _default_cache_dir L#: 21
ds P: 25529 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: __init__ L#: 61
ds P: 25529 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: __init__ L#: 64
ds P: 25529 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: put L#: 74
(ds_chat_py39) amd00@MZ32-00:~/yk_repo/ds/DeepSpeed/tests/accelerator$



'''