# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed

from unit.common import DistributedTest
from unit.simple_model import *

from unit.checkpoint.common import checkpoint_correctness_verification
from pydebug import debuginfo

class TestLatestCheckpoint(DistributedTest):
    world_size = 1

    def test_existing_latest(self, tmpdir):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            }
        }
        hidden_dim = 10
        models = [SimpleModel(hidden_dim=hidden_dim) for _ in range(2)]
        checkpoint_correctness_verification(config_dict=config_dict,
                                            models=models,
                                            hidden_dim=hidden_dim,
                                            tmpdir=tmpdir,
                                            load_optimizer_states=True,
                                            load_lr_scheduler_states=False,
                                            fp16=False,
                                            empty_tag=True)

    def test_missing_latest(self, tmpdir):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            }
        }
        hidden_dim = 10
        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        # should be no-op, since latest doesn't exist
        model.load_checkpoint(tmpdir)

'''

(ds_chat_py39) amd00@MZ32-00:~/yk_repo/ds/DeepSpeed/tests/unit/checkpoint$ pytest test_latest_checkpoint.py
================================================================== test session starts ===================================================================
platform linux -- Python 3.9.18, pytest-7.4.0, pluggy-1.0.0 -- /home/amd00/anaconda3/envs/ds_chat_py39/bin/python
cachedir: .pytest_cache
rootdir: /home/amd00/yk_repo/ds/DeepSpeed/tests
configfile: pytest.ini
plugins: anyio-3.5.0
collected 2 items

test_latest_checkpoint.py::TestLatestCheckpoint::test_existing_latest PASSED                                                                       [ 50%]
test_latest_checkpoint.py::TestLatestCheckpoint::test_missing_latest PASSED                                                                        [100%]

==================================================================== warnings summary ====================================================================
<string>:8
  <string>:8: PytestDeprecationWarning: A private pytest class or function was used.

unit/checkpoint/test_latest_checkpoint.py::TestLatestCheckpoint::test_existing_latest
  /home/amd00/yk_repo/ds/DeepSpeed/tests/conftest.py:47: UserWarning: Running test without verifying torch version, please provide an expected torch version with --torch_ver
    warnings.warn(

unit/checkpoint/test_latest_checkpoint.py::TestLatestCheckpoint::test_existing_latest
  /home/amd00/yk_repo/ds/DeepSpeed/tests/conftest.py:54: UserWarning: Running test without verifying cuda version, please provide an expected cuda version with --cuda_ver
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=================================================================== slowest durations ====================================================================
46.77s call     unit/checkpoint/test_latest_checkpoint.py::TestLatestCheckpoint::test_existing_latest
34.08s call     unit/checkpoint/test_latest_checkpoint.py::TestLatestCheckpoint::test_missing_latest

(4 durations < 1s hidden.  Use -vv to show these durations.)
======================================================= 2 passed, 3 warnings in 107.20s (0:01:47) ========================================================
ds P: 17926 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: _default_cache_dir L#: 21
ds P: 17926 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: __init__ L#: 61
ds P: 17926 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: __init__ L#: 64
ds P: 17926 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: load L#: 83
ds P: 17926 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: _default_cache_dir L#: 21
ds P: 17926 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: __init__ L#: 61
ds P: 17926 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: __init__ L#: 64
ds P: 17926 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: put L#: 74
ds P: 17926 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: _default_cache_dir L#: 21
ds P: 17926 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: __init__ L#: 61
ds P: 17926 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: __init__ L#: 64
ds P: 17926 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: load L#: 83
ds P: 17926 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: _default_cache_dir L#: 21
ds P: 17926 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: __init__ L#: 61
ds P: 17926 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: __init__ L#: 64
ds P: 17926 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: put L#: 74
(ds_chat_py39) amd00@MZ32-00:~/yk_repo/ds/DeepSpeed/tests/unit/checkpoint$
'''