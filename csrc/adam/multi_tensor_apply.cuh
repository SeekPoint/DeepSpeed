// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

/*
Copyright NVIDIA/apex
This file is adapted from fused adam in NVIDIA/apex, commit a109f85
*/

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include "compat.h"

#include <assert.h>

// #include <iostream>

// This header is the one-stop shop for all your multi-tensor apply needs.

// TODO:  Kernel arg size limit may be <4KB for some other cards (ie Jetson)
// 为了处理多张量应用这个需求，这个头文件是你的一站式解决方案。（DS团队的话，直译）
// TODO: 对于其他一些设备（例如：Jetson），内核参数大小限制可能小于4KB （DS团队的TODO，还没有实现的）
constexpr int depth_to_max_tensors[5] = {110, 64, 48, 36, 30};
constexpr int depth_to_max_blocks[5] = {320, 320, 320, 320, 320};

// 用于存储元数据的结构，包含了地址、大小、块到张量的映射、块到块的映射以及启动张量的索引。
template <int n>
struct TensorListMetadata {
    void* addresses[n][depth_to_max_tensors[n - 1]];
    int sizes[depth_to_max_tensors[n - 1]];
    unsigned char block_to_tensor[depth_to_max_blocks[n - 1]];
    // 这可能需要是一个完整的int.（原文，可能正在开发中）
    int block_to_chunk[depth_to_max_blocks[n - 1]];  // I fear this needs to be a full int.
    int start_tensor_this_launch;
};

// 定义了一个全局的CUDA内核函数，它将块信息传递给用户提供的函数进行处理。
template <typename T, typename U, typename... ArgTypes>
__global__ void multi_tensor_apply_kernel(int chunk_size,
                                          volatile int* noop_flag,
                                          T tl,
                                          U callable,
                                          ArgTypes... args)
{
    // Hand the chunk information to the user-supplied functor to process however it likes.
    callable(chunk_size, noop_flag, tl, args...);
}

// 这个函数用于处理多张量应用，它将块大小、块数量以及张量列表等信息传递给CUDA内核函数进行处理。
template <int depth, typename T, typename... ArgTypes>
void multi_tensor_apply(int block_size,
                        int chunk_size,
                        const at::Tensor& noop_flag,
                        const std::vector<std::vector<at::Tensor>>& tensor_lists,
                        T callable,
                        ArgTypes... args)
{
    // 检查张量列表的深度是否正确
    TORCH_CHECK(tensor_lists.size() == depth, "tensor_lists.size() != depth");

    // 检查张量列表的大小是否大于0
    int len0 = tensor_lists[0].size();
    TORCH_CHECK(len0 > 0, "tensor_lists[0].size() is not > 0");

    // 获取参考设备并检查是否为CUDA设备
    auto ref_device = tensor_lists[0][0].device();
    TORCH_CHECK(ref_device.type() == at::kCUDA, "expected input to be on cuda");

    // 检查每个张量列表的大小、设备以及元素数量是否与第一个张量列表一致
    for (int l = 0; l < tensor_lists.size(); l++)  // No range-based for because I need indices
    {
        TORCH_CHECK(tensor_lists[l].size() == len0, "Size mismatch among tensor lists");
        for (int t = 0; t < tensor_lists[l].size(); t++) {
            // TODO:  Print which tensor fails.
            bool contiguous_memory = tensor_lists[l][t].is_contiguous();
#ifdef VERSION_GE_1_5
            contiguous_memory = (contiguous_memory ||
                                 tensor_lists[l][t].is_contiguous(at::MemoryFormat::ChannelsLast));
#endif
            TORCH_CHECK(contiguous_memory, "A tensor was not contiguous.");
            TORCH_CHECK(tensor_lists[l][t].device() == ref_device,
                        "A tensor was not on the same device as the first tensor");
            TORCH_CHECK(tensor_lists[l][t].numel() == tensor_lists[0][t].numel(), "Size mismatch");
        }
    }

    int ntensors = tensor_lists[0].size();

    // 创建一个TensorListMetadata实例，用于保存张量元数据
    TensorListMetadata<depth> tl;

    // 设备保护，确保所有的操作都在张量所在的设备上进行
    const at::cuda::OptionalCUDAGuard device_guard(device_of(tensor_lists[0][0]));

    // 获取当前的CUDA流
    auto stream = at::cuda::getCurrentCUDAStream();

    tl.start_tensor_this_launch = 0;
    int loc_block_info = 0;
    int loc_tensor_info = 0;

    // 遍历每一个张量
    for (int t = 0; t < ntensors; t++) {
        // 将当前张量的元素数量保存到元数据中
        tl.sizes[loc_tensor_info] = tensor_lists[0][t].numel();

        // 遍历每一个深度，将每个深度的张量数据地址保存到元数据中
        for (int d = 0; d < depth; d++)
            tl.addresses[d][loc_tensor_info] = tensor_lists[d][t].data_ptr();
        loc_tensor_info++;

        // 计算当前张量需要的块数量
        int chunks_this_tensor = (tensor_lists[0][t].numel() + chunk_size - 1) / chunk_size;

        // 遍历每一个块
        for (int chunk = 0; chunk < chunks_this_tensor; chunk++) {
            // std::cout << chunks_this_tensor << std::endl;
            // 将块到张量的映射和块到块的映射保存到元数据中
            tl.block_to_tensor[loc_block_info] = loc_tensor_info - 1;
            tl.block_to_chunk[loc_block_info] = chunk;
            loc_block_info++;

            // 检查是否触及到张量的最大数量，块的最大数量或者已经是最后一个块
            bool tensors_full = (loc_tensor_info == depth_to_max_tensors[depth - 1] &&
                                 chunk == chunks_this_tensor - 1);
            bool blocks_full = (loc_block_info == depth_to_max_blocks[depth - 1]);
            bool last_chunk = (t == ntensors - 1 && chunk == chunks_this_tensor - 1);

            // 如果满足以上任意一个条件，那么就启动CUDA内核函数进行处理
            if (tensors_full || blocks_full || last_chunk) {
                // using accscalar_t = acc_type<scalar_t, true>;
                multi_tensor_apply_kernel<<<loc_block_info, block_size, 0, stream>>>(
                    chunk_size, noop_flag.DATA_PTR<int>(), tl, callable, args...);

                // 检查CUDA内核函数是否有错误发生
                AT_CUDA_CHECK(cudaGetLastError());

                // Reset.  The control flow possibilities here make my brain hurt.
                // 重置块信息的位置
                loc_block_info = 0;

                // 如果是当前张量的最后一个块，那么重置张量信息的位置，并更新启动张量的索引
                if (chunk == chunks_this_tensor - 1) {
                    // std::cout << "Hit case 1 " << cond1 << " " << cond2 << " " << cond3 <<
                    // std::endl;
                    loc_tensor_info = 0;
                    tl.start_tensor_this_launch = t + 1;
                } else {
                    // std::cout << "Hit case 2 " << cond1 << " " << cond2 << " " << cond3 <<
                    // std::endl;
                    // 如果不是当前张量的最后一个块，那么将当前张量的元数据移动到元数据的首部，并更新启动张量的索引
                    tl.sizes[0] = tl.sizes[loc_tensor_info - 1];
                    for (int d = 0; d < depth; d++)
                        tl.addresses[d][0] = tl.addresses[d][loc_tensor_info - 1];
                    loc_tensor_info = 1;
                    tl.start_tensor_this_launch = t;
                }
            }
        }
    }
}
