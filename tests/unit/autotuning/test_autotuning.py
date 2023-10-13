# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import pytest
from unit.simple_model import create_config_from_dict
from deepspeed.launcher import runner as dsrun
from deepspeed.autotuning.autotuner import Autotuner
from deepspeed.autotuning.scheduler import ResourceManager

RUN_OPTION = 'run'
TUNE_OPTION = 'tune'
from pydebug import debuginfo

def test_command_line():
    '''Validate handling of command line arguments'''
    for opt in [RUN_OPTION, TUNE_OPTION]:
        dsrun.parse_args(args=f"--num_nodes 1 --num_gpus 1 --autotuning {opt} foo.py".split())

    for error_opts in [
            "--autotuning --num_nodes 1 --num_gpus 1 foo.py".split(),
            "--autotuning test --num_nodes 1 -- num_gpus 1 foo.py".split(), "--autotuning".split()
    ]:
        with pytest.raises(SystemExit):
            dsrun.parse_args(args=error_opts)


@pytest.mark.parametrize("arg_mappings",
                        [
                            None,
                            {
                            },
                            {
                                "train_micro_batch_size_per_gpu": "--per_device_train_batch_size"
                            },
                            {
                                "train_micro_batch_size_per_gpu": "--per_device_train_batch_size",
                                "gradient_accumulation_steps": "--gradient_accumulation_steps"
                            },
                            {
                                "train_batch_size": "-tbs"
                            }
                        ]) # yapf: disable
def test_resource_manager_arg_mappings(arg_mappings):
    rm = ResourceManager(args=None,
                         hosts="worker-0, worker-1",
                         num_gpus_per_node=4,
                         results_dir=None,
                         exps_dir=None,
                         arg_mappings=arg_mappings)

    if arg_mappings is not None:
        for k, v in arg_mappings.items():
            assert k.strip() in rm.arg_mappings.keys()
            assert arg_mappings[k.strip()].strip() == rm.arg_mappings[k.strip()]


@pytest.mark.parametrize("active_resources",
                        [
                           {"worker-0": [0, 1, 2, 3]},
                           {"worker-0": [0, 1, 2, 3], "worker-1": [0, 1, 2, 3]},
                           {"worker-0": [0], "worker-1": [0, 1, 2], "worker-2": [0, 1, 2]},
                           {"worker-0": [0, 1], "worker-2": [4, 5]}
                        ]
                        ) # yapf: disable
def test_autotuner_resources(tmpdir, active_resources):
    config_dict = {"autotuning": {"enabled": True, "exps_dir": os.path.join(tmpdir, 'exps_dir'), "arg_mappings": {}}}
    config_path = create_config_from_dict(tmpdir, config_dict)
    args = dsrun.parse_args(args=f'--autotuning {TUNE_OPTION} foo.py --deepspeed_config {config_path}'.split())
    tuner = Autotuner(args=args, active_resources=active_resources)

    expected_num_nodes = len(list(active_resources.keys()))
    assert expected_num_nodes == tuner.exp_num_nodes

    expected_num_gpus = min([len(v) for v in active_resources.values()])
    assert expected_num_gpus == tuner.exp_num_gpus


'''

(ds_chat_py39) amd00@MZ32-00:~/yk_repo/ds/DeepSpeed/tests/unit/autotuning$ pytest
================================================================== test session starts ===================================================================
platform linux -- Python 3.9.18, pytest-7.4.0, pluggy-1.0.0 -- /home/amd00/anaconda3/envs/ds_chat_py39/bin/python
cachedir: .pytest_cache
rootdir: /home/amd00/yk_repo/ds/DeepSpeed/tests
configfile: pytest.ini
plugins: anyio-3.5.0
collected 10 items

test_autotuning.py::test_command_line PASSED                                                                                                       [ 10%]
test_autotuning.py::test_resource_manager_arg_mappings[None] PASSED                                                                                [ 20%]
test_autotuning.py::test_resource_manager_arg_mappings[arg_mappings1] PASSED                                                                       [ 30%]
test_autotuning.py::test_resource_manager_arg_mappings[arg_mappings2] PASSED                                                                       [ 40%]
test_autotuning.py::test_resource_manager_arg_mappings[arg_mappings3] PASSED                                                                       [ 50%]
test_autotuning.py::test_resource_manager_arg_mappings[arg_mappings4] PASSED                                                                       [ 60%]
test_autotuning.py::test_autotuner_resources[active_resources0] PASSED                                                                             [ 70%]
test_autotuning.py::test_autotuner_resources[active_resources1] PASSED                                                                             [ 80%]
test_autotuning.py::test_autotuner_resources[active_resources2] PASSED                                                                             [ 90%]
test_autotuning.py::test_autotuner_resources[active_resources3] PASSED                                                                             [100%]

==================================================================== warnings summary ====================================================================
unit/autotuning/test_autotuning.py::test_command_line
  /home/amd00/yk_repo/ds/DeepSpeed/tests/conftest.py:47: UserWarning: Running test without verifying torch version, please provide an expected torch version with --torch_ver
    warnings.warn(

unit/autotuning/test_autotuning.py::test_command_line
  /home/amd00/yk_repo/ds/DeepSpeed/tests/conftest.py:54: UserWarning: Running test without verifying cuda version, please provide an expected cuda version with --cuda_ver
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=================================================================== slowest durations ====================================================================

(30 durations < 1s hidden.  Use -vv to show these durations.)
============================================================ 10 passed, 2 warnings in 31.08s =============================================================
ds P: 17536 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: _default_cache_dir L#: 21
ds P: 17536 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: __init__ L#: 61
ds P: 17536 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: __init__ L#: 64
ds P: 17536 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: load L#: 83
ds P: 17536 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: _default_cache_dir L#: 21
ds P: 17536 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: __init__ L#: 61
ds P: 17536 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: __init__ L#: 64
ds P: 17536 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: put L#: 74
ds P: 17536 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: _default_cache_dir L#: 21
ds P: 17536 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: __init__ L#: 61
ds P: 17536 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: __init__ L#: 64
ds P: 17536 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: load L#: 83
ds P: 17536 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: _default_cache_dir L#: 21
ds P: 17536 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: __init__ L#: 61
ds P: 17536 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: __init__ L#: 64
ds P: 17536 at MZ32-00 F: /home/amd00/yk_repo/ds/DeepSpeed/deepspeed/ops/transformer/inference/triton/matmul_ext.py f: put L#: 74
(ds_chat_py39) amd00@MZ32-00:~/yk_repo/ds/DeepSpeed/tests/unit/autotuning$
'''