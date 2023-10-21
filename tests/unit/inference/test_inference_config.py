# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import deepspeed
from unit.common import DistributedTest
from unit.simple_model import create_config_from_dict

from pydebug import debuginfo, infoTensor
@pytest.mark.inference
class TestInferenceConfig(DistributedTest):
    world_size = 1

    def test_overlap_kwargs(self):
        gd.debuginfo(prj='dsUT', info='C:' + self.__class__.__name__)
        config = {"replace_with_kernel_inject": True}
        kwargs = {"replace_with_kernel_inject": True}

        engine = deepspeed.init_inference(torch.nn.Module(), config=config, **kwargs)
        assert engine._config.replace_with_kernel_inject

    def test_overlap_kwargs_conflict(self):
        gd.debuginfo(prj='dsUT', info='C:' + self.__class__.__name__)
        config = {"replace_with_kernel_inject": True}
        kwargs = {"replace_with_kernel_inject": False}

        with pytest.raises(ValueError):
            engine = deepspeed.init_inference(torch.nn.Module(), config=config, **kwargs)

    def test_kwargs_and_config(self):
        gd.debuginfo(prj='dsUT', info='C:' + self.__class__.__name__)
        config = {"replace_with_kernel_inject": True}
        kwargs = {"dtype": torch.float32}

        engine = deepspeed.init_inference(torch.nn.Module(), config=config, **kwargs)
        assert engine._config.replace_with_kernel_inject
        assert engine._config.dtype == kwargs["dtype"]

    def test_json_config(self, tmpdir):
        gd.debuginfo(prj='dsUT', info='C:' + self.__class__.__name__)
        config = {"replace_with_kernel_inject": True}
        config_json = create_config_from_dict(tmpdir, config)

        engine = deepspeed.init_inference(torch.nn.Module(), config=config_json)
        assert engine._config.replace_with_kernel_inject
