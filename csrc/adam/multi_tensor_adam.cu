// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Copyright NVIDIA/apex
This file is adapted from fused adam in NVIDIA/apex, commit a109f85
*/
#include "../cppdebug.h"
#include "../cudebug.cuh"
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
// Another possibility:
// #include <torch/all.h>

#include <assert.h>

// 引入多张量处理库，这个就在下面的代码文件中讲到
#include "multi_tensor_apply.cuh"
#include "type_shim.h"

#define BLOCK_SIZE 512
#define ILP 4

// 枚举类型，定义Adam优化器的模式
typedef enum {
    ADAM_MODE_0 = 0,  // L2 regularization mode  L2正则化模式
    ADAM_MODE_1 = 1   // Decoupled weight decay mode(AdamW) 分离的权重衰减模式(AdamW)
} adamMode_t;

using MATH_T = float;

// 定义AdamFunctor模板类，实现自定义的操作
template <typename T>
struct AdamFunctor {
    // 在CUDA设备上定义一个操作函数
    __device__ __forceinline__ void operator()(int chunk_size,
                                               volatile int* noop_gmem,
                                               TensorListMetadata<4>& tl,
                                               const float beta1,
                                               const float beta2,
                                               const float beta1_correction,
                                               const float beta2_correction,
                                               const float epsilon,
                                               const float lr,
                                               adamMode_t mode,
                                               const float decay)
    {
         // 主要包含Adam优化器在CUDA中的并行计算步骤
        //debuginfo(); ///home/amd00/yk_repo/ds/DeepSpeed/csrc/adam/multi_tensor_adam.cu(48)-<operator()>:

        // I'd like this kernel to propagate infs/nans.
        // if(*noop_gmem == 1)
        //   return;

        // 定义校正模式的索引
        int tensor_loc = tl.block_to_tensor[blockIdx.x];

        // potentially use to pass in list of scalar
        // int tensor_num = tl.start_tensor_this_launch + tensor_loc;

        // 定义chunk的索引
        int chunk_idx = tl.block_to_chunk[blockIdx.x];

        // 当前tensor的大小
        int n = tl.sizes[tensor_loc];

        // 以下四个指针分别是梯度、参数、一阶矩和二阶矩的开始地址
        T* g = (T*)tl.addresses[0][tensor_loc];
        g += chunk_idx * chunk_size;

        T* p = (T*)tl.addresses[1][tensor_loc];
        p += chunk_idx * chunk_size;

        T* m = (T*)tl.addresses[2][tensor_loc];
        m += chunk_idx * chunk_size;

        T* v = (T*)tl.addresses[3][tensor_loc];
        v += chunk_idx * chunk_size;

        // 调整tensor的大小以适应chunk的大小
        n -= chunk_idx * chunk_size;

        // see note in multi_tensor_scale_kernel.cu
        // 以ILP（Instruction Level Parallelism）为步长并行处理每个tensor
        for (int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * ILP) {
            MATH_T r_g[ILP];
            MATH_T r_p[ILP];
            MATH_T r_m[ILP];
            MATH_T r_v[ILP];
            // 加载每个tensor的梯度、参数、一阶矩和二阶矩
#pragma unroll
            for (int ii = 0; ii < ILP; ii++) {
                int i = i_start + threadIdx.x + ii * blockDim.x;
                if (i < n && i < chunk_size) {
                    r_g[ii] = g[i];
                    r_p[ii] = p[i];
                    r_m[ii] = m[i];
                    r_v[ii] = v[i];
                } else {
                    // 如果超过tensor的大小，用0填充
                    r_g[ii] = MATH_T(0);
                    r_p[ii] = MATH_T(0);
                    r_m[ii] = MATH_T(0);
                    r_v[ii] = MATH_T(0);
                }
            }
            // 根据Adam优化器的计算公式更新参数
#pragma unroll
            for (int ii = 0; ii < ILP; ii++) {
                if (mode == ADAM_MODE_0) {
                    // L2 // L2正则化模式
                    r_g[ii] = r_g[ii] + (decay * r_p[ii]);
                    r_m[ii] = beta1 * r_m[ii] + (1 - beta1) * r_g[ii];
                    r_v[ii] = beta2 * r_v[ii] + (1 - beta2) * r_g[ii] * r_g[ii];
                    MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
                    MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
                    MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
                    MATH_T update = next_m_unbiased / denom;
                    r_p[ii] = r_p[ii] - (lr * update);
                } else {
                    // weight decay  // 分离的权重衰减模式（AdamW）
                    r_m[ii] = beta1 * r_m[ii] + (1 - beta1) * r_g[ii];
                    r_v[ii] = beta2 * r_v[ii] + (1 - beta2) * r_g[ii] * r_g[ii];
                    MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
                    MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
                    MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
                    MATH_T update = (next_m_unbiased / denom) + (decay * r_p[ii]);
                    r_p[ii] = r_p[ii] - (lr * update);
                }
            }
            // 更新原始tensor的参数、一阶矩和二阶矩
#pragma unroll
            for (int ii = 0; ii < ILP; ii++) {
                int i = i_start + threadIdx.x + ii * blockDim.x;
                if (i < n && i < chunk_size) {
                    p[i] = r_p[ii];
                    m[i] = r_m[ii];
                    v[i] = r_v[ii];
                }
            }
        }
    }
};

// 多张量Adam优化器的CUDA实现
// 各参数的含义
// chunk_size：每个块的大小，用于并行计算。
// noop_flag：一个标志，如果为真，那么这个函数将不会进行任何操作。
// tensor_lists：一个嵌套的向量，包含了所有需要更新的张量列表，每个列表包括参数张量（p）、梯度张量（g）、一阶矩张量（m1）和二阶矩张量（m2）。
// lr：学习率。
// beta1、beta2：Adam优化器中的超参数。
// epsilon：一个极小的常数，用于防止除以零的错误。
// step：当前的优化步骤。
// mode：优化模式，可以选择L2正则化模式或AdamW模式。
// bias_correction：是否进行偏差修正，如果为1，则进行修正。
// weight_decay：权重衰减系数。

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
                            const float weight_decay)
{
    using namespace at;

    // debuginfo();

    // Handle bias correction mode
    // 处理偏差校正模式
    float bias_correction1 = 1.0f, bias_correction2 = 1.0f;
    if (bias_correction == 1) {
        // 如果启用了偏差修正（bias_correction=1），则根据当前步骤计算修正因子
        bias_correction1 = 1 - std::pow(beta1, step);
        bias_correction2 = 1 - std::pow(beta2, step);
    }

    // 假设p,g,m1,m2现在都是同一种类型
    // Assume single type across p,g,m1,m2 now
    // DISPATCH_DOUBLE_FLOAT_AND_HALF是一个宏,决定了应该使用哪种数据类型的AdamFunctor，
    // 这个AdamFunctor被用于multi_tensor_apply函数中，进行实际的优化操作。
    DISPATCH_DOUBLE_FLOAT_AND_HALF(tensor_lists[0][0].scalar_type(),
                                   0,
                                   "adam",
                                   // 使用多张量处理库函数multi_tensor_apply
                                   multi_tensor_apply<4>(BLOCK_SIZE,
                                                         chunk_size,
                                                         noop_flag,
                                                         tensor_lists,
                                                         AdamFunctor<scalar_t_0>(),
                                                         beta1,
                                                         beta2,
                                                         bias_correction1,
                                                         bias_correction2,
                                                         epsilon,
                                                         lr,
                                                         (adamMode_t)mode,
                                                         weight_decay);)
    // 检查CUDA是否有错误
    AT_CUDA_CHECK(cudaGetLastError());
}
