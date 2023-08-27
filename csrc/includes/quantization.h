// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <cuda_fp16.h>
#include "ds_kernel_utils.h"

namespace quantize {

enum class Type { Symmetric, Asymmetric };

struct PackedInt4 {
    int8_t high : 4;
    int8_t low : 4;
};

DS_HD_INLINE bool requires_offset(Type qType) { return qType == Type::Asymmetric; }

}  // namespace quantize

void launch_quant(int8_t* output_data,
                  float* params,
                  const __half* input_data,
                  const int groups,
                  const int elems_per_group,
                  const int num_bits,
                  const quantize::Type quant_type,
                  cudaStream_t stream);

template <typename T>
void launch_dequantize_kernel(T* dequant_data,
                              const int8_t* q_data,
                              const float* q_params,
                              quantize::Type q_type,
                              int num_bits,
                              int elems_per_group,
                              int total_elems,
                              cudaStream_t stream);
// 在GPU上进行并行化量化操作
void launch_swizzled_quant(int8_t* q_data,// 量化后的数据的存储位置。
                           float* q_scales, //量化比例因子的存储位置。
                           const __half* input_data, //输入数据，这些数据将被量化。
                           int num_bits, //量化的位数。
                           quantize::Type q_type, //量化的类型，可能包含不同的量化策略。
                           int groups, //数据将被分割成的组数。
                           int elems_per_group, //每组元素的数量。
                           int pipelining, //是否使用流水线并行化。
                           int nodes, //计算节点数量。
                           int devices_per_node, //每个节点上设备的数量。
                           cudaStream_t stream); //CUDA流，用于在GPU上异步并行执行操作。

// GPU上进行并行化的反量化并执行reduce操作
void launch_dequant_reduce(int8_t* reduced_data, //reduce后的数据的存储位置。
                           float* reduced_scales, //reduce后的量化比例因子的存储位置。
                           const int8_t* input_data, // 输入的量化数据。
                           const float* input_scales, //  输入的量化比例因子。
                           int num_gpus, // 用于计算的GPU数量。
                           int num_bits, //  量化的位数。
                           quantize::Type quant_type, // 量化的类型，可能包含不同的量化策略。
                           int out_groups, // 输出数据将被分割成的组数。
                           int elems_per_out_group, // 每组输出元素的数量。
                           int elems_per_in_tensor, // 每个输入张量的元素数量。
                           int groups_per_in_tensor, // 每个输入张量被分割成的组数。
                           int elems_per_in_group, // 每个输入组的元素数量。
                           cudaStream_t stream);//CUDA流，用于在GPU上异步并行执行操作。

template <typename T>
void launch_fake_quantize_kernel(T* vals,
                                 int total_count,
                                 int group_num,
                                 int num_bits,
                                 cudaStream_t stream);
template <typename T>
void launch_sr_fake_quantize_kernel(T* vals,
                                    int total_count,
                                    int group_num,
                                    int num_bits,
                                    cudaStream_t stream);
template <typename T>
void launch_fake_quantize_kernel_asym(T* vals,
                                      int total_count,
                                      int group_num,
                                      int num_bits,
                                      cudaStream_t stream);
template <typename T>
void launch_sr_fake_quantize_kernel_asym(T* vals,
                                         int total_count,
                                         int group_num,
                                         int num_bits,
                                         cudaStream_t stream);
