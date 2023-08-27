# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import pytest
import deepspeed.comm as dist
from torch.nn import Module

from unit.common import DistributedTest
from unit.simple_model import random_dataloader

import deepspeed

from deepspeed.runtime.zero.config import DeepSpeedZeroConfig

import torch.nn as nn


# 首先，定义一个神经网络模型 NNModel，它是由多个全连接层和一个交叉熵损失函数组成。
class NNModel(nn.Module):

    def __init__(self, h_dim=1024, n_layers=2):
        super(NNModel, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(h_dim, h_dim) for i in range(n_layers)])
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, x, y):
        for layer in self.layers:
            x = layer(x)
        return self.cross_entropy_loss(x, y)

# 在函数中，首先创建一个 DeepSpeedZeroConfig 实例 config。
# 创建实例时，使用了 Python 的解包运算符 ** 将一个字典作为参数传入。
# 字典中有一个键值对 "zero_hpz_partition_size": 4，
# 这表示设置 ZeRO++ 的hpz划分大小为 4。
# 这个测试函数的目的是验证 DeepSpeedZeroConfig 的 zero_hpz_partition_size 属性是否能正确设置和获取。
def test_zero_hpz_partition_size_config():
    config = DeepSpeedZeroConfig(**{"zero_hpz_partition_size": 4})
    assert config.zero_hpz_partition_size == 4

# 函数内部，遍历了 model 的所有命名参数。model.named_parameters() 返回一个迭代器，
# 每次迭代返回一个元组，包含参数的名字和参数对象。
# 然后，对于每个参数对象 param，使用 assert 关键字进行断言。
# 断言 param.ds_secondary_tensor is None 检查 param 的 ds_secondary_tensor 属性是否为 None。
# ds_secondary_tensor 是 DeepSpeed ZeRO-3 中的一个属性，表示这个参数的第二存储。
# 如果这个属性为 None，说明没有为这个参数分配第二存储，这可能意味着 ZeRO-3 没有正确应用到这个参数。
def _assert_no_secondary_tensor_group(model: Module) -> None:
    for _, param in model.named_parameters():
        assert param.ds_secondary_tensor is None
        assert param.ds_zero_param_process_group is None

# 函数内部，遍历了 model 的所有命名参数。model.named_parameters() 返回一个迭代器，
# 每次迭代返回一个元组，包含参数的名字和参数对象。
# 然后，对于每个参数对象 param，使用 assert 关键字进行断言。
# 断言 param.ds_secondary_tensor is not None 检查 param 的 ds_secondary_tensor 属性是否为 None。
# ds_secondary_tensor 是 DeepSpeed ZeRO-3 中的一个属性，表示这个参数的第二存储。如果这个属性为 None，
# 说明没有为这个参数分配第二存储，即所有的参数都存储在 ds_tensor 中。
def _assert_secondary_tensor_size(model: Module) -> None:
    for _, param in model.named_parameters():
        assert param.ds_secondary_tensor is not None
        assert param.ds_secondary_tensor.size()[0] % param.ds_tensor.size()[0] == 0


# 这段代码定义了一个使用 PyTest 框架的单元测试类 TestZeroPPConfigSweep，
# 用于测试 DeepSpeed 的 zero3++ 优化的配置。特别地，它测试了不同的隐藏维度 h_dim、
# 层数 n_layers 和 ZeRO-3++ zero_hpz_partition_size(zpg) 大小对模型的影响。
#Large sweep along hidden dim, num_layers, and zpg of different sizes
#Assert when zpg=1 that secondary group and tensors are invalid
@pytest.mark.sequential
@pytest.mark.parametrize("h_dim", [1024])
@pytest.mark.parametrize("n_layers", [4, 9])
@pytest.mark.parametrize("zpg", [1, 2, 4])
class TestZeroPPConfigSweep(DistributedTest):
    world_size = 4

    def test(self, h_dim: int, n_layers: int, zpg: int) -> None:
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": 3,
                "stage3_max_reuse_distance": 0,
                "zero_hpz_partition_size": zpg,
                "zero_quantized_weights": True,
                "zero_quantized_gradients": True,
                "contiguous_gradients": True,
                "overlap_comm": True,
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1.
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 1.,
            }
        }

        model = NNModel(h_dim, n_layers)
        model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        data_loader = random_dataloader(model=model, total_samples=20, hidden_dim=h_dim, device=model.device)
        dist.barrier()
        if zpg == 1:
            _assert_no_secondary_tensor_group(model)

        for n, batch in enumerate(data_loader):
            if n == 0 and zpg != 1:
                _assert_secondary_tensor_size(model)
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

'''
这个不仅是zero++的测试脚本，也给我们指出了如何使用zero++提供的qgZ，qwZ，hpZ的feature。

0x4. 总结
这篇文章从DeepSpeed Zero++的博客出发，做了zero++博客的翻译和原理理解。然后深入到zero++的代码实现，将核心的2个cuda kerne：swizzled_quant_kernel和dequant_reduce进行了解析。接着，基于这两个kernel导出的pybind接口对上层的qgZ，qwZ，hpZ的python实现调用链进行了跟踪和代码解析。在解析qwZ 和 hpZ的时候明白了对zero3切分权重的核心实现是ds_secondary_tensor和ds_tensor。最后对qgZ，qwZ，hpZ的测试脚本进行了解析且这个脚本也指出了要启用qgZ，qwZ，hpZ的使用方法。大家也可以参考zero++的paper和deepspeed源码来获得更多细节。

0x5. 参考文献
https://zhuanlan.zhihu.com/p/641297077
https://zhuanlan.zhihu.com/p/639002087
deepspeed论文：https://www.microsoft.com/en-us/research/publication/zero-extremely-efficient-collective-communication-for-giant-model-training/

'''