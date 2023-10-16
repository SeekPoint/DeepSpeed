// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team
#include "../cppdebug.h"
#include "../cudebug.cuh"
#include "memory_access_utils.h"
#include "quantization_utils.h"
#include "reduction_utils.h"

using rop = reduce::ROpType;

namespace swiz_quant {
// swiz_quant命名空间内定义了一些常量，包括最大线程数、最小线程数、
// 步长粒度以及每步处理的元素数量。这些值都是在量化过程中使用的。
constexpr int max_threads = 512;
constexpr int min_threads = 32;

constexpr int step_granularity = 2;
constexpr int h_per_step = step_granularity * quantize::h_per_load;
}  // namespace swiz_quant


// swizzled_quant_kernel是一个模板函数，它的模板参数包括：量化位数numBits、
// 总块数totalChunks、线程数threads、以及量化类型quantType。
//它接受的参数包括量化后的数据、量化比例尺、未压缩的数据、每个分组的元素数、节点数、每个节点的设备数。
template <int numBits, int totalChunks, int threads, quantize::Type quantType>
__global__ void swizzled_quant_kernel(int8_t* quantized_data,
                                      float* quantized_scales,
                                      const __half* uncompressed_data,
                                      int elems_per_group,
                                      int nodes,
                                      int devices_per_node)
{
    debuginfo();

    // 获取当前的线程块对象（thread block）。hw_warp_size是一个常量32
    cg::thread_block tb = cg::this_thread_block();
    // 从线程块中划分一个大小为硬件warp大小的分区（warp）。
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    // Indexing offsets, same as normal quantization for in-case
    // 计算线程块在网格中的全局排序（rank）。这里网格是3维的，每个维度可能包含多个线程块
    const int block_rank = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    // 根据线程块的全局排序和每组的元素数量来计算偏移量。
    const int block_offset = block_rank * elems_per_group;
    // quantize::h_per_load 的定义在 `DeepSpeed/csrc/includes/quantization_utils.h` 中的：
    // constexpr int granularity = 16;
    // constexpr int h_per_load = granularity / sizeof(__half);
    // 计算在一个线程块中的线程的偏移量。这里假设一个线程将加载quantize::h_per_load个元素。
    const int elem_offset = tb.thread_index().x * quantize::h_per_load;
    // 计算基础偏移量，即线程块偏移量和线程偏移量的和。
    const int base_offset = block_offset + elem_offset;
    // 计算步长。步长是一个线程块的大小乘以每个线程加载的元素数量。
    const int stride = tb.size() * quantize::h_per_load;
    // 根据基础偏移量获取未压缩数据的指针。
    const __half* input_base = uncompressed_data + base_offset;

    // Local buffer
    // 在本地声明一个缓冲区，用来存储加载的数据。这里__half2是CUDA中用于表示半精度浮点数的类型。
    __half2 local_buffer[totalChunks * quantize::h2_per_load];

    quantize::GroupStats<quantType> stats; // 声明一个GroupStats对象，用来存储统计信息。
#pragma unroll // 是一个编译指令，它告诉编译器展开接下来的循环，可以提高代码的执行效率。
    // 然后是一个循环，读取全局内存的数据并存储到本地缓冲区，然后更新统计信息。
    for (int i = 0; i < totalChunks; i++) {
        __half2* iteration_buffer = local_buffer + i * quantize::h2_per_load;

        mem_access::load_global<quantize::granularity>(
            iteration_buffer, input_base + i * stride, elem_offset + i * stride < elems_per_group);

#pragma unroll
        for (int j = 0; j < quantize::h2_per_load; j++) { stats.update(iteration_buffer[j]); }
    }

    // 调用get_params函数从统计对象（stats）中获取量化参数。这些参数包括每个矢量的缩放因子和零点。
    // 此行中numBits和threads是模板参数，分别表示量化的位数和线程数量。同时，tb和warp分别表示线程块和线程束的对象。
    auto params = stats.template get_params<numBits, threads>(tb, warp);

    // 设置partition_id为z方向的block索引。
    const int partition_id = blockIdx.z;
    // 计算每个节点的设备偏移，即当前分区ID除以每个节点的设备数。
    const int partition_offset = partition_id / devices_per_node;
    // 计算分区基数，即当前分区ID除以每个节点的设备数的余数乘以节点数。
    const int partition_base = (partition_id % devices_per_node) * nodes;
    // 计算流水线偏移，即y方向的block索引乘以设备总数。
    const int pipelining_offset = blockIdx.y * (devices_per_node * nodes);
    // 计算输出分区，即流水线偏移加上分区基数和设备偏移。
    const int output_partition = (pipelining_offset + partition_base + partition_offset);

    // 计算输出标量效应，即每个字节可以包含的元素数量。
    constexpr int out_scalar_effect = 8 / numBits;
    // 计算输出block的排名，即输出分区乘以x方向的grid大小加上x方向的block索引。
    const int out_block_rank = output_partition * gridDim.x + blockIdx.x;
    // 计算输出block的偏移，即输出block的排名乘以每个组的元素数除以输出标量效应。
    const int out_block_offset = out_block_rank * elems_per_group / out_scalar_effect;
    // 计算输出基础偏移，即输出block的偏移加上元素偏移除以输出标量效应。
    const int out_base_offset = out_block_offset + elem_offset / out_scalar_effect;
    // 计算输出基地址，即量化数据加上输出基础偏移。
    int8_t* out_base = quantized_data + out_base_offset;

    // 计算输出步长，即步长除以输出标量效应。
    const int out_stride = stride / out_scalar_effect;
    // 计算每次输出的int8数目，即每次加载的半精度浮点数数量除以输出标量效应。
    constexpr int num_int8_out = quantize::h_per_load / out_scalar_effect;

    // 如果当前线程是线程块中的第一个线程，那么将参数存储到指定的位置。
    if (tb.thread_index().x == 0) { params.store(quantized_scales, out_block_rank); }

#pragma unroll
    // 对每个块进行循环。
    for (int i = 0; i < totalChunks; i++) {
        // 如果当前元素在有效范围内，则执行以下操作：
        if (i * stride + elem_offset < elems_per_group) {
            // 定义一个本地输出数组，用于临时存储量化的结果。
            int8_t local_output[quantize::h_per_load / out_scalar_effect];
            // 进行量化操作，结果存储在local_output中。
            quantize::_chunk<numBits, quantType>(
                local_output, local_buffer + i * quantize::h2_per_load, params);
            // 将本地的量化结果存储到全局内存中。
            mem_access::store_global<num_int8_out>(out_base + i * out_stride, local_output);
        }
    }
}

#define LAUNCH_SWIZZLE_QUANT(total_chunks, threads)                                           \
    swizzled_quant_kernel<numBits, total_chunks, threads, qType><<<grid, block, 0, stream>>>( \
        q_data, q_scales, input_data, elems_per_group, nodes, devices_per_node);
// 这里解释了 "Swizzled quantization"（交错量化）是如何工作的。
// 这种方法主要是为了优化多节点多设备的并行计算中的通信效率。
// 这里给出了一个在两个节点，每个节点上有四个设备的情况下的划分示例。
// 原始的数据划分可能是线性的，比如0-7每个数代表一组数据，且数据在设备上的存储是连续的：
//  --- --- --- --- --- --- --- ---
// | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
//  --- --- --- --- --- --- --- ---
// 在交错量化中，数据会被重新组织，变成如下形式：
//  --- --- --- --- --- --- --- ---
// | 0 | 4 | 1 | 5 | 2 | 6 | 3 | 7 |
// --- --- --- --- --- --- --- ---
// 此处，每个数字代表一组数据，你可以看到原本连续存储的数据被"交错"了。
// 在这个例子中，0和4可能在同一个节点的不同设备上，1和5在另一个节点的不同设备上。
// 通过这种方式，我们可以在进行节点间的通信时，同时从每个节点的多个设备中获取数据，这样可以提高通信效率。
// 还提到了一个"分片"的概念，比如说二分分片。在这种情况下，每个分区的前半部分数据会被连接在一起，
// 这样可以为后续的流水线操作提供更好的支持。


// 这段代码是一个模板函数，实现了"Swizzled quantization"的过程。
// 主要参数包括量化数据q_data，量化比例尺q_scales，输入数据input_data，分组数量groups，
// 每组元素数量elems_per_group，流水线大小pipelining，节点数nodes
// 和每个节点上的设备数devices_per_node。最后一个参数stream是用于CUDA的异步并行执行的流。
/*
Swizzled quantization reorganizes the quantized groups in order to better facilitate
communication. As an example of the partitioning scheme we have the following example
of 2 node, 4 device swizzling:

 --- --- --- --- --- --- --- ---
| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
 --- --- --- --- --- --- --- ---
becomes
 --- --- --- --- --- --- --- ---
| 0 | 4 | 1 | 5 | 2 | 6 | 3 | 7 |
 --- --- --- --- --- --- --- ---

Multiple quantization groups may be mapped into a single partition. In order to better support
later pipelining, we may also perform an additional slicing. In two-way slicing, for instance,
the first halves of each partition are concatenated.
*/
template <int numBits, quantize::Type qType>
void launch_swizzled_quant_impl(int8_t* q_data,
                                float* q_scales,
                                const __half* input_data,
                                int groups,
                                int elems_per_group,
                                int pipelining,
                                int nodes,
                                int devices_per_node,
                                cudaStream_t stream)
{
    debuginfo();

    // 函数首先计算一步操作中需要的线程数one_step_threads。
    // 这是基于elems_per_group和固定步长swiz_quant::h_per_step计算得出的。
    // next_pow2函数将输入值向上取到最近的2的幂。这是为了优化线程分配，因为GPU在处理2的幂次数的线程块时，效率最高。
    const int one_step_threads =
        next_pow2((elems_per_group + swiz_quant::h_per_step - 1) / swiz_quant::h_per_step);
    // 之后，它计算最大线程数max_threads，
    // 这个值是one_step_threads和预设的最大线程数swiz_quant::max_threads中的较小值。
    const int max_threads = (one_step_threads < swiz_quant::max_threads) ? one_step_threads
                                                                         : swiz_quant::max_threads;
    // 然后，它计算实际线程数threads，这个值是max_threads和预设的最小线程数swiz_quant::min_threads中的较大值。
    const int threads = (max_threads < swiz_quant::min_threads) ? swiz_quant::min_threads
                                                                : max_threads;
    // 下一步是设置CUDA的block和grid维度。block的维度是threads，
    // grid的维度则是基于分组数量，节点数和设备数计算出的。
    // 这里，每个分区的分组数groups_per_partition是总分组数groups除以总设备数
    // （节点数nodes乘以每节点设备数devices_per_node）。
    // 接着，它断言分区中的分组数可以被流水线大小pipelining整除，得到连续分组数contiguous_groups。
    // 最后，设定grid的维度，每个维度代表一个不同的并行度。
    dim3 block(threads);
    const int groups_per_partition = groups / (nodes * devices_per_node);
    assert(groups_per_partition % pipelining == 0);
    const int contiguous_groups = groups_per_partition / pipelining;
    const int partitions = nodes * devices_per_node;
    dim3 grid(contiguous_groups, pipelining, partitions);

    // elems_per_step和total_unroll是关于处理步长和展开程度的参数，它们影响kernel的并行性和计算复杂度。
    const int elems_per_step = threads * swiz_quant::h_per_step;
    const int external_unroll = ((elems_per_group + elems_per_step - 1) / elems_per_step);
    const int total_unroll = external_unroll * swiz_quant::step_granularity;

    // 接下来的一系列判断和宏调用LAUNCH_SWIZZLE_QUANT，就是基于不同的线程数和展开程度，
    // 选择并启动相应的量化kernel。不同的量化kernel在执行效率和处理数据规模方面有各自的优化。
    assert(total_unroll % 2 == 0);

    if (threads == 32) {
        LAUNCH_SWIZZLE_QUANT(2, 32);
    } else if (threads == 64) {
        LAUNCH_SWIZZLE_QUANT(2, 64);
    } else if (threads == 128) {
        LAUNCH_SWIZZLE_QUANT(2, 128);
    } else if (threads == 256) {
        LAUNCH_SWIZZLE_QUANT(2, 256);
    } else if (threads == 512) {
        if (total_unroll == 2) {
            LAUNCH_SWIZZLE_QUANT(2, 512);
        } else if (total_unroll == 4) {
            LAUNCH_SWIZZLE_QUANT(4, 512);
        } else if (total_unroll == 6) {
            LAUNCH_SWIZZLE_QUANT(6, 512);
        } else if (total_unroll == 8) {
            LAUNCH_SWIZZLE_QUANT(8, 512);
        } else if (total_unroll == 10) {
            LAUNCH_SWIZZLE_QUANT(10, 512);
        }
    }
}

// DISPATCH_SWIZZLE_QUANT宏接收两个参数num_bits和qtype，并调用了一个模板函数launch_swizzled_quant_impl，
// 这个模板函数的模板参数为num_bits和qtype，函数参数为一系列传入的值。
#define DISPATCH_SWIZZLE_QUANT(num_bits, qtype)                   \
    launch_swizzled_quant_impl<num_bits, qtype>(q_data,           \
                                                q_scales,         \
                                                input_data,       \
                                                groups,           \
                                                elems_per_group,  \
                                                pipelining,       \
                                                nodes,            \
                                                devices_per_node, \
                                                stream);
// 这个函数主要是用来根据量化的位数num_bits和量化类型q_type来调用相应的模板函数。
// 函数的参数列表包含了数据指针q_data, q_scales和input_data，这些都是在GPU内存上的数据。
// 其它的参数如groups, elems_per_group, pipelining, nodes,
// devices_per_node, stream都是用来控制量化操作的参数
void launch_swizzled_quant(int8_t* q_data,
                           float* q_scales,
                           const __half* input_data,
                           int num_bits,
                           quantize::Type q_type,
                           int groups,
                           int elems_per_group,
                           int pipelining,
                           int nodes,
                           int devices_per_node,
                           cudaStream_t stream)
{
    debuginfo();

    // 如果num_bits等于4，那么就会进入第一个if分支；如果num_bits等于8，就会进入第二个if分支。
    // 在每个if分支中，都会再根据q_type的值来调用不同的模板函数。
    if (num_bits == 4) {
     // 如果q_type等于quantize::Type::Asymmetric，那么就会调用launch_swizzled_quant_impl
     // 模板函数并将模板参数设置为4和quantize::Type::Asymmetric
        if (q_type == quantize::Type::Asymmetric) {
            DISPATCH_SWIZZLE_QUANT(4, quantize::Type::Asymmetric);
        }
    // 如果q_type等于quantize::Type::Symmetric，那么就会调用launch_swizzled_quant_impl
    // 模板函数并将模板参数设置为4和quantize::Type::Symmetric。
    else if (q_type == quantize::Type::Symmetric) {
            DISPATCH_SWIZZLE_QUANT(4, quantize::Type::Symmetric);
        }
    } else if (num_bits == 8) {
        if (q_type == quantize::Type::Asymmetric) {
            DISPATCH_SWIZZLE_QUANT(8, quantize::Type::Asymmetric);
        } else if (q_type == quantize::Type::Symmetric) {
            DISPATCH_SWIZZLE_QUANT(8, quantize::Type::Symmetric);
        }
    }
}
