// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team
#include "../cppdebug.h"
#include <torch/extension.h>

// CUDA版本的多张量Adam优化器
// chunk_size：每个块的大小
// noop_flag：一个标志张量，用于控制是否进行操作
// tensor_lists：一个包含多个张量列表的列表，每个列表表示一个参数的不同部分
// lr：学习率
// beta1：Adam优化器的第一个参数
// beta2：Adam优化器的第二个参数
// epsilon：Adam优化器的epsilon参数，用于防止除以零
// step：当前训练步数
// mode：优化器的模式
// bias_correction：是否进行偏差校正
// weight_decay：权重衰减系数
void multi_tensor_adam_cuda(int chunk_size,
                            at::Tensor noop_flag,
                            std::vector<std::vector<at::Tensor>> tensor_lists,
                            const float lr,
                            const float beta1,
                            const float beta2,
                            const float epsilon,
                            const int step,
                            const int mode,
                            const int bias_correction,
                            const float weight_decay);

// Pybind11模块定义
// "multi_tensor_adam"：函数名
// &multi_tensor_adam_cuda：函数实现
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("multi_tensor_adam",
          &multi_tensor_adam_cuda,
          "Compute and apply gradient update to parameters for Adam optimizer");
}
