# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
batched collective operations for overhead amortization and better
bandwidth utilization
"""
import math
from typing import List
import torch
from torch import Tensor
from deepspeed import comm as dist
# NOTE: Use torch.distributed's ProcessGroup class until we have our own.
# 这行代码从PyTorch的torch.distributed模块中引入了ProcessGroup和all_to_all_single。
# ProcessGroup是PyTorch分布式计算的一个基础抽象，表示一个可以进行集合操作（如集合同步、集合计算等）的进程组。
# all_to_all_single函数用于在多个进程之间执行“all-to-all”操作，即每个进程可以发送和接收来自所有其他进程的不同数据。
from torch.distributed import ProcessGroup, all_to_all_single
# get_accelerator函数的功能是获取当前DeepSpeed运行的硬件加速器（通常是GPU）。
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import instrument_w_nvtx
from deepspeed.ops import op_builder
from pydebug import debuginfo, infoTensor

def _torch_reduce_scatter_fn(input_tensor: Tensor, output_tensor: Tensor, group=None, async_op=False, prof=False):
    gd.debuginfo(prj='ds', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
    return instrument_w_nvtx(dist.reduce_scatter_fn)(output_tensor, input_tensor, group=group, async_op=False)


# 这行代码定义了一个名为quantizer_module的变量，并将其初始化为None
quantizer_module = None


# 这是一个通过量化操作来减少网络通信量的函数，主要用于分布式训练环境中。
# 函数的名字all_to_all_quant_reduce是指所有节点之间进行量化的通信和信息聚合。
# 函数的输入参数是一个tensor列表，每个tensor表示不同节点上的数据，还有一个groups字典，表示不同的通信组。
@instrument_w_nvtx
@torch.no_grad()
def all_to_all_quant_reduce(tensors: List[Tensor], groups: {}) -> List[Tensor]:
    gd.debuginfo(prj='ds', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
    # quantizer_module是一个全局的量化模块对象，主要用于执行量化和反量化的操作。
    global quantizer_module
    # 如果量化模块未初始化，则使用QuantizerBuilder对象加载一个量化模块。
    if quantizer_module is None:
        gd.debuginfo(prj='ds', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
        quantizer_module = op_builder.QuantizerBuilder().load()
    # 获取当前节点（服务器）的设备数量。
    local_world_size = get_accelerator().device_count()
    # 获取全局的设备数量，这个数量是所有节点的设备数量之和。
    global_world_size = dist.get_world_size()
    # 计算节点数量，即全局设备数量除以每个节点的设备数量。
    num_nodes = global_world_size // local_world_size
    # 获取当前设备在全局设备中的排名。
    this_rank = dist.get_rank()
    # 计算节点内部的索引，即当前设备在本地节点中的排名。
    intra_idx = int(this_rank / local_world_size)
    # 计算节点间的索引，即当前节点在所有节点中的排名。
    inter_idx = this_rank % local_world_size
    # 初始化输出tensor列表，列表的长度等于输入tensor列表的长度，初始值设为None。
    output_lst: List[Tensor] = [None] * len(tensors)
    # 对于输入的每个tensor，进行以下操作：
    for idx, tensor in enumerate(tensors):
        # 如果tensor的维度是1，进行以下操作：
        if tensor.dim() == 1:
            # 设置量化组的大小为全局设备数量。
            intra_quant_group = global_world_size
            # 执行reduce和scatter操作，并将结果存储在输出列表对应的位置上。
            output_lst[idx] = reduce_scatter_coalesced([tensor])[0]
            continue
        # 如果tensor的维度不是1，进行以下操作：
        else:
            # 设置量化组的大小为tensor的第0维，第1维和全局设备数量中的最大值。
            intra_quant_group = max(tensor.shape[0], tensor.shape[1], global_world_size)

            # 计算节点间的量化组的大小。
            inter_quant_group = intra_quant_group // local_world_size
            # 对tensor执行量化操作，得到量化的结果和比例因子。
            intra_quant_int4, intra_q_scales = quantizer_module.swizzle_quant(tensor, intra_quant_group, 4,
                                                                              quantizer_module.Symmetric, 1, num_nodes,
                                                                              local_world_size)
            # 创建两个与量化结果和比例因子形状相同的tensor，用于存储后续操作的结果。
            local_output = torch.empty_like(intra_quant_int4)
            scale_output = torch.empty_like(intra_q_scales)
            # 执行all-to-all操作，将所有设备的数据聚合到每个设备上。
            all_to_all_single(local_output, intra_quant_int4, group=groups[f'local_{intra_idx}'])
            all_to_all_single(scale_output, intra_q_scales, group=groups[f'local_{intra_idx}'])
            # 对所有设备上的数据执行量化的归约操作，得到全局的输入tensor和全局的比例因子。
            global_input_tensor, global_scales = quantizer_module.quantized_reduction(
                local_output, scale_output, intra_quant_group, inter_quant_group, 4, quantizer_module.Symmetric)
            # 创建两个与全局输入tensor和全局比例因子形状相同的tensor，用于存储后续操作的结果
            global_output = torch.empty_like(global_input_tensor)
            global_scale_output = torch.empty_like(global_scales)
            # 执行all-to-all操作，将所有节点的数据聚合到每个节点上。
            all_to_all_single(global_output, global_input_tensor, group=groups[f'global_{inter_idx}'])
            all_to_all_single(global_scale_output, global_scales, group=groups[f'global_{inter_idx}'])
            # 对聚合后的数据执行反量化操作，得到最终的输出。
            final_output = quantizer_module.dequantize(global_output, global_scale_output, global_scale_output.numel(),
                                                       4, quantizer_module.Symmetric)
            # 将最终的输出按节点数量切分，计算每个部分的和，然后取平均值，得到最终的结果，结果的形状是一维的。
            output_lst[idx] = (sum(list(final_output.chunk(num_nodes))) / num_nodes).view(-1)
    return output_lst
'''
然后在https://github.com/microsoft/DeepSpeed/pull/3784/files#diff-1ad5daa1b31aa5573616024068d646f0c38e88d4d3a71d3d0e4bc352ea232178R1188-R1194 调用了这个all_to_all_quant_reduce实现。

对于qwZ 和 hpZ 的实现则在：https://github.com/microsoft/DeepSpeed/pull/3784/files#diff-bc45426db58250294594100cfdf3d73ecb653d879cabee404e38edc4eb4c9ecbR1051-R1164 。从源码来看qwZ和hpZ的实现并没有使用基于Block的quantization，而是普通的量化方法。
'''

@instrument_w_nvtx
@torch.no_grad()
def reduce_scatter_coalesced(
    tensors: List[Tensor],
    group: ProcessGroup = None,
) -> List[Tensor]:
    """simultaneously reduce-scatter a list of tensors - this can be done more
    efficiently than individual reduce scatter calls
    TODO. see if PyTorch team wants a c++ version of this for ProcessGroupNCCL
    """
    this_rank = dist.get_rank(group)
    world_sz = dist.get_world_size(group)

    partition_lst_for_each_tensor = [None] * len(tensors)
    for tensor_idx, tensor in enumerate(tensors):
        flattened_tensor = tensor.view(-1)
        chunk_sz = math.ceil(tensor.numel() / world_sz)
        partition_lst_for_each_tensor[tensor_idx] = [
            flattened_tensor[rank * chunk_sz:rank * chunk_sz + chunk_sz] for rank in range(0, world_sz)
        ]

    padded_partition_sz_for_each_tensor = tuple(math.ceil(t.numel() / world_sz) for t in tensors)

    if len(tensors) == 1 and tensors[0].numel() % world_sz == 0:
        gd.debuginfo(prj='ds', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
        # if there's only one tensor being reduced and we don't need to pad
        # we have an opportunity to avoid a memory allocation
        tensor_partition_flat_buffer = tensors[0].view(-1)
    else:
        gd.debuginfo(prj='ds', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
        # interleave tensor partitions such that the correct reduced partitions of each tensor
        # end up at each rank
        tensor_partitions_lst_with_padding = []
        for rank in range(world_sz):
            for tensor_idx in range(len(tensors)):
                # add tensor content
                tensor_chunk = partition_lst_for_each_tensor[tensor_idx][rank]
                tensor_partitions_lst_with_padding.append(tensor_chunk)

                # add padding if necessary
                padding_sz = padded_partition_sz_for_each_tensor[tensor_idx] - tensor_chunk.numel()
                if padding_sz > 0:
                    tensor_partitions_lst_with_padding.append(
                        torch.empty(padding_sz, dtype=tensor_chunk.dtype, device=tensor_chunk.device))

        tensor_partition_flat_buffer = instrument_w_nvtx(torch.cat)(tensor_partitions_lst_with_padding)

    tensor_partition_flat_buffer.div_(world_sz)  # pre-divide
    tensor_partition_buffer_for_each_rank: List[Tensor] = torch.chunk(tensor_partition_flat_buffer, world_sz)

    # batched reduce-scatter call
    _torch_reduce_scatter_fn(tensor_partition_flat_buffer,
                             tensor_partition_buffer_for_each_rank[this_rank],
                             group=group)

    # reverse procedure of the interleaving done previously, done on the
    # result of the batched reduce-scatter
    output_lst: List[Tensor] = [None] * len(tensors)
    offset = 0
    for tensor_idx in range(len(tensors)):
        output_lst[tensor_idx] = tensor_partition_buffer_for_each_rank[this_rank].narrow(
            0, offset, partition_lst_for_each_tensor[tensor_idx][this_rank].numel())

        offset += padded_partition_sz_for_each_tensor[tensor_idx]
    return output_lst
