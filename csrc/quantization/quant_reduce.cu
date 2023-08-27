// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <cstdio>
#include "dequantization_utils.h"
#include "ds_kernel_utils.h"
#include "memory_access_utils.h"
#include "quantization_utils.h"
#include "reduction_utils.h"

using rop = reduce::ROpType;

/*
TODO(cmikeh2): Add implementation that better handles larger nodes. It would like make sense
to leverage some parallel reductions here to improve performance.
*/
// 这段 CUDA kernel 是用于将一些输入数据进行反量化和归约操作的。它的功能是将输入的量化数据（int8类型）
// 转换回浮点数据（__half2类型，也就是半精度浮点数），然后进行一些归约操作，并再次量化数据并输出。
// 这是一个模板函数，可以通过模板参数调整数据位宽（numBits）、张量数量（numTensors）
// 、需要处理的数据块的数量（totalChunks）、以及量化类型（quantType）：
template <int numBits, int numTensors, int totalChunks, quantize::Type quantType>
// 该 CUDA kernel 配置了一些输入和输出参数，包括输入和输出的数据和缩放因子、每个输出组的元素数量、
// 每个输入张量的元素数量、每个输入张量的组数量、每个输入组的元素数量，以及张量的总数：
__global__ void __launch_bounds__(1024) dequant_reduce(int8_t* reduced_data,
                                                       float* reduced_scales,
                                                       const int8_t* input_data,
                                                       const float* input_scales,
                                                       int elems_per_out_group,
                                                       int elems_per_in_tensor,
                                                       int groups_per_in_tensor,
                                                       int elems_per_in_group,
                                                       int num_tensors)
{
    // 这段代码首先获取了当前的线程块（tb）和线程块内的一个 warp（warp）：
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    // NOTE(cmikeh2): This probably could be hardcoded to a larger number,
    // but that means even stronger restrictions on the number of elements per group
    // A performance analysis here might be beneficial
    // 根据模板参数 numBits，这段代码确定了每次内存加载的元素数量（elems_per_load）和用于存储的值的数量（storage_values）：
    constexpr int mem_granularity = (numBits == 8) ? 8 : 4;
    constexpr int elems_per_load = mem_granularity / sizeof(int8_t);  // div by 1
    constexpr int storage_values = 16 / sizeof(__half2);

    // 然后，这段代码计算了每个线程块和每个线程的偏移量，以及每次迭代的步长
    const int block_offset = tb.group_index().x * elems_per_out_group;
    const int elem_offset = tb.thread_index().x * elems_per_load;
    const int base_offset = block_offset + elem_offset;
    const int stride = tb.group_dim().x * elems_per_load;

    // 接下来，这段代码为每个线程分配了一个本地缓冲区，并初始化了一个统计对象：
    __half2 local_buffer[totalChunks * storage_values];

    quantize::GroupStats<quantType> stats;

    // 这段代码是在一个更大的循环中，其中 i 是从 0 到 totalChunks 的索引。
    // 这个循环处理的每一个“块”都包含了 storage_values 的元素。
    // #pragma unroll 是一个编译器指令，意思是编译器应该将循环展开，以减少循环的开销。
#pragma unroll
    for (int i = 0; i < totalChunks; i++) {
        // 在每个块中，首先获取一个指向当前块在 local_buffer 中开始位置的指针 iteration_buffer
        __half2* iteration_buffer = local_buffer + i * storage_values;

        // 然后，初始化 iteration_buffer 的每一个元素。reduce::init<rop::Add, __half2>()
        // 是一个模板函数，根据给定的类型和运算，返回相应的初始值。这里，初始值是加法操作的中性元素，对于加法来说，就是0。
#pragma unroll
        for (int j = 0; j < storage_values; j++) {
            iteration_buffer[j] = reduce::init<rop::Add, __half2>();
        }

        // 接着，计算了一些用于后续操作的参数：
        const int iter_offset = i * stride + base_offset;
        const int iter_scale_idx = iter_offset / elems_per_in_group;
        bool do_loads = i * stride + elem_offset < elems_per_out_group;

        // 根据 numTensors 是否大于 0，执行不同的操作。如果 numTensors 大于 0，那么对每个张量执行以下操作：
        if (numTensors > 0) {
#pragma unroll
            for (int j = 0; j < numTensors; j++) {
                // 如果 do_loads 为真，从全局内存加载数据到 load_buffer；
                if (do_loads) {
                    int8_t load_buffer[elems_per_load];

                    mem_access::load_global<mem_granularity>(
                        load_buffer, input_data + j * elems_per_in_tensor + iter_offset);

                    // 创建一个参数对象 params，用于后续的反量化操作；
                    quantize::Params<quantType, numBits> params(
                        input_scales + j * groups_per_in_tensor, iter_scale_idx);

                    __half2 dequant_buffer[storage_values];
                    // 将 load_buffer 中的数据反量化，并将结果存储到 dequant_buffer；
                    dequantize::chunk<numBits, quantType>(dequant_buffer, load_buffer, params);

#pragma unroll
                    // 将 dequant_buffer 中的每个元素添加到 iteration_buffer 对应的元素。
                    // 这里的 #pragma unroll 指令又告诉编译器将内部的循环展开。
                    for (int k = 0; k < storage_values; k++) {
                        iteration_buffer[k] =
                            reduce::element<rop::Add>(iteration_buffer[k], dequant_buffer[k]);
                    }
                }
            }
        } else {
            // 如果 numTensors 不大于 0，那么对 num_tensors 个张量执行类似的操作。这里的 #pragma unroll 4 指令告诉编译器将内部的循环展开4次。
#pragma unroll 4
            for (int j = 0; j < num_tensors; j++) {
                if (do_loads) {
                    int8_t load_buffer[elems_per_load];

                    mem_access::load_global<mem_granularity>(
                        load_buffer, input_data + j * elems_per_in_tensor + iter_offset);

                    quantize::Params<quantType, numBits> params(
                        input_scales + j * groups_per_in_tensor, iter_scale_idx);

                    __half2 dequant_buffer[storage_values];
                    dequantize::chunk<numBits, quantType>(dequant_buffer, load_buffer, params);

#pragma unroll
                    for (int k = 0; k < storage_values; k++) {
                        iteration_buffer[k] =
                            reduce::element<rop::Add>(iteration_buffer[k], dequant_buffer[k]);
                    }
                }
            }
        }
        // 最后，将 iteration_buffer 中的每个元素更新到 stats 对象。
#pragma unroll
        for (int j = 0; j < storage_values; j++) { stats.update(iteration_buffer[j]); }
    }

    // stats 是一个 quantize::GroupStats<quantType> 类型的对象，其中 quantType 是模板参数。
    // get_params 是这个类的成员函数，接收两个参数，分别是当前线程块 tb 和 warp warp，
    // 并且有两个模板参数 numBits 和 threads_per_group(1024)。
    // 这个函数的返回值是一种参数类型的对象，具体的类型取决于 quantize::GroupStats<quantType> 的定义。
    auto params = stats.template get_params<numBits, 1024>(tb, warp);

    // 然后，如果当前线程是线程块的第一个线程，那么将参数存储到 reduced_scales 中，索引是线程块的索引：
    if (tb.thread_index().x == 0) { params.store(reduced_scales, tb.group_index().x); }

    // 接下来，这段代码再次进行多次循环，每次处理一个数据块。在每个数据块内，如果条件满足，
    // 那么将本地缓冲区的数据进行量化操作，并将结果存储到输出数据：
#pragma unroll
    for (int i = 0; i < totalChunks; i++) {
        const int iter_offset = i * stride + base_offset;
        if (i * stride + elem_offset < elems_per_out_group) {
            int8_t local_output[elems_per_load];
            // 这里的 quantize::_chunk 是一个模板函数，接收三个参数，分别是存储位置 local_output、
            // 输入数据 local_buffer + i * storage_values 和参数 params，
            // 并且有两个模板参数 numBits 和 quantType。这个函数的功能是将输入数据进行量化操作，并将结果存储到
            // local_output。之后，mem_access::store_global 将 local_output 中的数据存储到 reduced_data + iter_offset。
            quantize::_chunk<numBits, quantType>(
                local_output, local_buffer + i * storage_values, params);
            mem_access::store_global<mem_granularity>(reduced_data + iter_offset, local_output);
        }
    }
}

// 这是一个C++模板函数，名称为pow2_round，它接受一个模板参数Power，并有一个整型参数raw_value。
// 这个函数的功能是将输入的raw_value向上取到最接近的2的Power次方的倍数。
// 如果Power为2（即，我们希望结果是4的倍数），且raw_value为6，那么这个函数会返回8，因为8是最接近6的4的倍数。
template <int Power>
// int32_t pow2_round(int32_t raw_value)：这是函数的定义，函数名为pow2_round，
// 返回类型为int32_t，并接受一个类型为int32_t的参数raw_value。
int32_t pow2_round(int32_t raw_value)
{
    // 首先，raw_value - 1是将raw_value减1，然后>> Power是将结果右移Power位，
    // 这相当于除以2^Power。之后，+ 1是将结果加1，这实现了向上取整。
    // 最后，<< Power是将结果左移Power位，这相当于乘以2^Power，这样就得到了最接近的2的Power次方的倍数。
    return (((raw_value - 1) >> Power) + 1) << Power;
}

#define LAUNCH_DEQUANT_REDUCE(num_chunks)                      \
    dequant_reduce<numBits, numTensors, num_chunks, quantType> \
        <<<grid, block, 0, stream>>>(reduced_data,             \
                                     reduced_scales,           \
                                     input_data,               \
                                     input_scales,             \
                                     elems_per_out_group,      \
                                     elems_per_in_tensor,      \
                                     groups_per_in_tensor,     \
                                     elems_per_in_group,       \
                                     num_tensors);

// 这个C++模板函数 launch_dequant_reduce_impl 是用于启动反量化和数据规约的CUDA kernel。
// 该函数包含三个模板参数，numBits，numTensors和quantType，这些参数在编译时必须被确定。
template <int numBits, int numTensors, quantize::Type quantType>
void launch_dequant_reduce_impl(int8_t* reduced_data,
                                float* reduced_scales,
                                const int8_t* input_data,
                                const float* input_scales,
                                int out_groups,
                                int elems_per_out_group,
                                int elems_per_in_tensor,
                                int groups_per_in_tensor,
                                int elems_per_in_group,
                                int num_tensors,
                                cudaStream_t stream)
{
    // This is a coincidence. This is derived by 8 halves per 16 bytes with 2-way packing for int4
    // 定义了每个线程需要处理的元素数量，这个值与numBits（模板参数）相同。
    constexpr int elems_per_thread = numBits;
    // 计算处理一组输出元素需要的线程数，这个值取决于每个线程处理的元素数量和每个输出组的元素数量。
    // next_pow2函数计算最接近且大于等于其参数的2的幂。
    const int one_step_threads =
        next_pow2((elems_per_out_group + elems_per_thread - 1) / (elems_per_thread));
    // TODO(cmikeh2): Tune this
    // 确定线程数，如果一步所需的线程数小于1024，则使用这个值，否则使用1024。
    const int threads = (one_step_threads < 1024) ? one_step_threads : 1024;

    // 设置CUDA网格和块的维度。每个块中有threads个线程，而网格中有out_groups个块。
    dim3 block(threads);
    dim3 grid(out_groups);

    // 计算每步要处理的元素数量，这取决于线程数和每个线程处理的元素数。
    const int elems_per_step = threads * elems_per_thread;
    // 计算unroll需要多少步，取决于每个输出组中的元素数量和每一步要处理的元素数量
    const int unroll_raw = (elems_per_out_group + elems_per_step - 1) / elems_per_step;

    // 如果原始值大于等于4，那么就用2的幂进行近似，否则保持不变。
    const int unroll = (unroll_raw >= 4) ? pow2_round<1>(unroll_raw) : unroll_raw;

    // 根据优化后的unroll，调用不同的反量化和数据规约kernel。
    if (unroll == 1) {
        // 0-4096 elems
        LAUNCH_DEQUANT_REDUCE(1);
    } else if (unroll == 2) {
        // 4097-8192 etc...
        LAUNCH_DEQUANT_REDUCE(2);
    } else if (unroll == 3) {
        LAUNCH_DEQUANT_REDUCE(3);
    } else if (unroll == 4) {
        LAUNCH_DEQUANT_REDUCE(4);
    } else if (unroll == 6) {
        LAUNCH_DEQUANT_REDUCE(6);
    } else if (unroll == 8) {
        LAUNCH_DEQUANT_REDUCE(8);
    } else if (unroll == 10) {
        LAUNCH_DEQUANT_REDUCE(10);
    } else if (unroll == 12) {
        // 48k limit
        LAUNCH_DEQUANT_REDUCE(12);
    } else {
        assert(false);
    }
}

// 这是一个C++预处理器宏定义。预处理器宏是在编译时，即在源代码被转换为机器语言之前进行替换的一种机制。
// 在这个宏定义中，LAUNCH_DEQUANT_REDUCE_IMPL是宏名称，而NUM_BITS，NUM_GPUS和QUANT_TYPE是宏参数。
// 宏定义的主体是一个函数调用 launch_dequant_reduce_impl<NUM_BITS, NUM_GPUS, QUANT_TYPE>。
// 该函数是模板函数，NUM_BITS，NUM_GPUS和QUANT_TYPE是模板参数。
// 当这个宏在源代码中被使用时，例如LAUNCH_DEQUANT_REDUCE_IMPL(4, 8, quantize::Type::Symmetric)，
// 预处理器会将这个宏调用替换为launch_dequant_reduce_impl<4, 8, quantize::Type::Symmetric>
// 的函数调用，并将后面的参数列表插入到这个函数调用中。
#define LAUNCH_DEQUANT_REDUCE_IMPL(NUM_BITS, NUM_GPUS, QUANT_TYPE)                   \
    launch_dequant_reduce_impl<NUM_BITS, NUM_GPUS, QUANT_TYPE>(reduced_data,         \
                                                               reduced_scales,       \
                                                               input_data,           \
                                                               input_scales,         \
                                                               out_groups,           \
                                                               elems_per_out_group,  \
                                                               elems_per_in_tensor,  \
                                                               groups_per_in_tensor, \
                                                               elems_per_in_group,   \
                                                               num_gpus,             \
                                                               stream);
// 这个函数的作用是处理量化后的数据，将它们"反量化"并进行规约。
void launch_dequant_reduce(int8_t* reduced_data, //这是一个指针，指向存储反量化和reduce操作后的结果的内存位置。
                           float* reduced_scales, //这是一个指针，指向存储缩放因子的内存位置，这些缩放因子应用于反量化操作。
                           const int8_t* input_data, // 这是一个指向输入数据（已经量化）的常量指针。
                           const float* input_scales, // 这是一个指向输入数据量化时使用的缩放因子的常量指针。
                           int num_gpus, // 指示执行此操作的GPU数量。
                           int num_bits, // 指示用于量化操作的位数（4或8位）。
                           quantize::Type quant_type, // 指定了量化操作的类型（对称或非对称）。
                           int out_groups, // 这些是与输入数据和输出数据的维度或组相关的参数。
                           int elems_per_out_group,
                           int elems_per_in_tensor,
                           int groups_per_in_tensor,
                           int elems_per_in_group,
                           cudaStream_t stream)
{
    // 根据量化类型（对称或非对称）和位数（4或8），对应的反量化和reduce的实现(LAUNCH_DEQUANT_REDUCE_IMPL)被调用。
    // 这个实现可能会根据不同的配置优化计算过程，例如对于8个GPU和16个GPU的情况。
    if (quant_type == quantize::Type::Symmetric) {
        if (num_bits == 4) {
            if (num_gpus == 8) {
                LAUNCH_DEQUANT_REDUCE_IMPL(4, 8, quantize::Type::Symmetric);
            } else if (num_gpus == 16) {
                LAUNCH_DEQUANT_REDUCE_IMPL(4, 16, quantize::Type::Symmetric);
            } else {
                LAUNCH_DEQUANT_REDUCE_IMPL(4, -1, quantize::Type::Symmetric);
            }
        } else if (num_bits == 8) {
            if (num_gpus == 8) {
                LAUNCH_DEQUANT_REDUCE_IMPL(8, 8, quantize::Type::Symmetric);
            } else if (num_gpus == 16) {
                LAUNCH_DEQUANT_REDUCE_IMPL(8, 16, quantize::Type::Symmetric);
            } else {
                LAUNCH_DEQUANT_REDUCE_IMPL(8, -1, quantize::Type::Symmetric);
            }
        }
    } else if (quant_type == quantize::Type::Asymmetric) {
        if (num_bits == 4) {
            if (num_gpus == 8) {
                LAUNCH_DEQUANT_REDUCE_IMPL(4, 8, quantize::Type::Asymmetric);
            } else if (num_gpus == 16) {
                LAUNCH_DEQUANT_REDUCE_IMPL(4, 16, quantize::Type::Asymmetric);
            } else {
                LAUNCH_DEQUANT_REDUCE_IMPL(4, -1, quantize::Type::Asymmetric);
            }
        } else if (num_bits == 8) {
            if (num_gpus == 8) {
                LAUNCH_DEQUANT_REDUCE_IMPL(8, 8, quantize::Type::Asymmetric);
            } else if (num_gpus == 16) {
                LAUNCH_DEQUANT_REDUCE_IMPL(8, 16, quantize::Type::Asymmetric);
            } else {
                LAUNCH_DEQUANT_REDUCE_IMPL(8, -1, quantize::Type::Asymmetric);
            }
        }
    }
}