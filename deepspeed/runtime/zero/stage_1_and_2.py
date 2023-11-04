# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import os
from deepspeed import comm as dist
from packaging import version as pkg_version
from collections import OrderedDict
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from deepspeed.runtime import ZeROOptimizer
from deepspeed.runtime.fp16.loss_scaler import CreateLossScaler
from deepspeed.runtime.utils import (bwc_tensor_model_parallel_rank, get_global_norm, empty_cache, see_memory_usage,
                                     inf, is_model_parallel_parameter, align_dense_tensors, all_gather_dp_groups)

from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.utils import logger
from deepspeed.moe.utils import is_moe_param
from deepspeed.git_version_info import version

from deepspeed.runtime.constants import PIPE_REPLICATED
from deepspeed.accelerator import get_accelerator

from deepspeed.checkpoint.constants import (DS_VERSION, GROUP_PADDINGS, PARTITION_COUNT,
                                            SINGLE_PARTITION_OF_FP32_GROUPS, BASE_OPTIMIZER_STATE, CLIP_GRAD,
                                            ZERO_STAGE, PARAM_SLICE_MAPPINGS)
from deepspeed.utils import link_hp_params
from deepspeed.checkpoint import enable_universal_checkpoint

# Toggle this to true to enable correctness test
# with gradient partitioning and without
pg_correctness_test = False

from pydebug import gd, infoTensor

def input(msg):
    return


def split_half_float_double(tensors):
    device_type = get_accelerator().device_name()
    dtypes = [
        "torch.{}.HalfTensor".format(device_type), "torch.{}.FloatTensor".format(device_type),
        "torch.{}.DoubleTensor".format(device_type), "torch.{}.BFloat16Tensor".format(device_type)
    ]
    gd.debuginfo(prj="ds", info=f'dtypes={dtypes}')
    buckets = []
    for i, dtype in enumerate(dtypes):
        bucket = [t for t in tensors if t.type() == dtype]
        if bucket:
            buckets.append(bucket)
    gd.debuginfo(prj="ds", info=f'buckets={buckets}')
    return buckets


def isclose(a, b, rtol=1e-09, atol=0.0):
    return abs(a - b) <= max(rtol * max(abs(a), abs(b)), atol)


def lcm(x, y):
    from fractions import gcd  # or can import gcd from `math` in Python 3
    return x * y // gcd(x, y)

# 获得对齐所需的填充个数
def get_alignment_padding(tensor_list, alignment):
    # gd.debuginfo(prj="ds")
    num_elements = sum([tensor.numel() for tensor in tensor_list])
    remainder = num_elements % alignment
    return (alignment - remainder) if remainder else remainder


def move_to_cpu(tensor_list):
    for tensor in tensor_list:
        tensor.data = tensor.data.cpu()


# def print_rank_msg(msg):
#     print(f"rank {dist.get_rank()} - {msg}")
def print_rank_0(message):
    if dist.get_rank() == 0:
        gd.debuginfo(prj='ds', info=message, level = 2)

# 返回填充后的tensor
def _get_padded_tensor(src_tensor, size):
    if src_tensor.numel() >= size: #大于指定size，所以无需填充
        return src_tensor
    padded_tensor = torch.zeros(size, dtype=src_tensor.dtype, device=src_tensor.device)
    slice_tensor = torch.narrow(padded_tensor, 0, 0, src_tensor.numel())
    slice_tensor.data.copy_(src_tensor.data)
    return padded_tensor

'''
3. stage2 - 初始化
从入口那里已经看到，当 stage 1,2 时， 会创建优化器 DeepSpeedZeroOptimizer 代替原来的优化器， 
stage 1,2 的特性都在这个优化器中实现。

Stage 1,2 的核心功能就是对参数的梯度和优化器的状态进行分割， 每个进程（GPU，rank）只保留一部分，减少对显存的消耗。 
新版本中，这部分也支持 cpu offload 的功能。

核心思路也简单，就是对基础（原始）优化器内的 params_group 进行处理， 
只保留属于当前进程（GPU，rank）的参数，其它的都从优化器中删除， 这样就只会计算保留部分的梯度，以及只有保留部分的优化器状态。

对于优化器的分割初始化功能的实现都写在类的 __init__ 方法里了， 
省略一些吐槽。导致这个 __init__ 方法代码很长，我们把它分拆了讲解。
'''
class DeepSpeedZeroOptimizer(ZeROOptimizer):
    """
    DeepSpeedZeroOptimizer designed to reduce the memory footprint
    required for training large deep learning models.

    For more details please see ZeRO: Memory Optimization Towards Training A Trillion Parameter Models
    https://arxiv.org/abs/1910.02054

    For usage examples, refer to TODO: DeepSpeed Tutorial

    """

    def __init__(self,
                 init_optimizer,  # 初始化优化器
                 param_names,  # 参数名称
                 timers,  # 计时器，用于测量代码的运行时间
                 static_loss_scale=1.0,  # 静态损失缩放，用于控制损失的缩放比例
                 dynamic_loss_scale=False,  # 是否使用动态损失缩放，当为True时，损失会根据训练过程动态调整
                 dynamic_loss_args=None,  # 动态损失缩放的参数
                 verbose=True,  # 是否打印详细的日志信息
                 contiguous_gradients=True,  # 是否连续存储梯度，有助于提高内存效率
                 reduce_bucket_size=500000000,  # reduce操作的bucket大小，用于梯度聚合
                 allgather_bucket_size=5000000000,  # allgather操作的bucket大小，用于同步所有节点的梯度
                 dp_process_group=None,  # 数据并行的进程组
                 expert_parallel_group=None,  # 专家并行的进程组
                 expert_data_parallel_group=None,  # 专家数据并行的进程组
                 reduce_scatter=True,  # 是否使用reduce_scatter进行梯度聚合，可以减少通信次数，提高效率
                 overlap_comm=False,  # 是否重叠通信和计算，当为True时，可以在计算梯度的同时进行梯度通信，提高效率
                 cpu_offload=False,
                 mpu = None,  # 模型并行单元，用于处理模型并行的相关操作
                 clip_grad = 0.0,  # 梯度裁剪值，用于防止梯度爆炸
                 communication_data_type = torch.float16,  # 通信时的数据类型，使用float16可以减少通信带宽，提高效率
                 postscale_gradients = True,  # 是否在计算完梯度后进行缩放，可以防止数值溢出
                 gradient_predivide_factor = 1.0,  # 在梯度累积前的预缩放因子
                 gradient_accumulation_steps = 1,  # 梯度累积步数，通过累积梯度可以模拟大批量训练，提高训练稳定性
                 ignore_unused_parameters = True,  # 是否忽略未使用的参数，当为True时，未使用的参数不会被优化器更新
                 partition_grads = True,  # 是否分区梯度，当为True时，梯度会被分区存储，可以节省内存
                 round_robin_gradients = False,  # 是否进行轮询梯度，当为True时，各个设备会轮流进行梯度计算，可以平衡设备负载
                 has_moe_layers = False,  # 是否包含moe层，moe层是一种用于大规模模型训练的技术
                 fp16_master_weights_and_gradients = False,  # 是否使用fp16存储主权重和梯度，可以节省内存
                 elastic_checkpoint = False):  # 是否使用弹性检查点，当为True时，可以在训练过程中动态保存和加载模型，提高训练的容错性

        # 如果当前是主节点，打印一些设置的日志信息
        if dist.get_rank() == 0:
            gd.debuginfo(prj='ds', info=f"Reduce bucket size {reduce_bucket_size}")
            gd.debuginfo(prj='ds', info=f"Allgather bucket size {allgather_bucket_size}")
            gd.debuginfo(prj='ds', info=f"CPU Offload: {cpu_offload}")
            gd.debuginfo(prj='ds', info=f'Round robin gradient partitioning: {round_robin_gradients}')

        # The fused optimizer does all the work. We need this layer for two reason:
        # 1. maintain same user API from apex.fp16_utils
        # 2. keep common stuff here in case we need to add ne552w fused optimizer later

        # 设置一些属性
        self.elastic_checkpoint = elastic_checkpoint
        self.param_names = param_names
        self.mpu = mpu

        # differences from apex.fp16_utils:
        # - assume all model params in fp16
        # - assume all params requires grad
        # - flat by groups, not keeping state. TODO: remove state explicitly?
        # - master grad and unflat master weight never exist. TODO: a way to save out unflat master?
        # 如果没有检测到计算加速器，则抛出异常
        if not get_accelerator().is_available():
            raise SystemError("Cannot use fp16 without accelerator.")

        # 基础优化器
        self.optimizer = init_optimizer

        # Use torch (un)flatten ops
        # 把张量打开扁平化的方法，这两个方法调用的是 torch 的方法， 设置参数展平和反展平的函数
        self.flatten = _flatten_dense_tensors
        self.unflatten = _unflatten_dense_tensors

        # ZeRO stage 1 (False) or 2 (True)
        #  是否启用梯度分割
        self.partition_gradients = partition_grads  # type: bool

        # stage 阶段
        self.zero_stage_string = "ZeRO-2" if partition_grads else "ZeRO-1"

        self.timers = timers

        self.reduce_scatter = reduce_scatter

        # 配置项 默认为 False
        # 尝试将梯度缩减与逆向计算相重叠
        self.overlap_comm = overlap_comm  # type: bool

        self.cpu_offload = cpu_offload

        self.deepspeed_adam_offload = cpu_offload

        # 获取当前设备，如果开启了CPU offload，那么设备为cpu，否则为当前设备
        self.device = get_accelerator().current_device_name() if not self.cpu_offload else 'cpu'

        # 所属的并行进程组
        self.dp_process_group = dp_process_group

        #  专家并行所属的组  expert parallel group
        self.ep_process_group = expert_parallel_group

        #data parallel group for experts
        # 专家数据并行组  data parallel group for experts
        self.expert_dp_process_group = expert_data_parallel_group

        #data parallel size for non-experts # 数据并行的大小
        dp_size = dist.get_world_size(group=self.dp_process_group)

        #For MoE models this maybe different for different param group
        #It will be modified during MoE setup later in the init
        # 对于MoE模型，这可能对于不同的参数组是不同的
        # 它将在init中的MoE设置过程中被修改
        self.real_dp_process_group = [dp_process_group for i in range(len(self.optimizer.param_groups))]
        self.partition_count = [dp_size for i in range(len(self.optimizer.param_groups))]
        gd.debuginfo(prj='ds', info=f"partition_count={self.partition_count}")
        gd.debuginfo(prj='ds', info=f"real_dp_process_group={self.real_dp_process_group}")

        self.is_gradient_accumulation_boundary = True

        # CPU-Offload requires contiguous gradients
        # 在生成梯度时将其复制到连续的缓冲区中。避免了后向传递过程中的内存碎片。
        self.contiguous_gradients = contiguous_gradients or cpu_offload # type: bool

        # 是否有 moe 层
        self.has_moe_layers = has_moe_layers
        if self.has_moe_layers:
            gd.debuginfo(prj="ds")
            self._configure_moe_settings()
        self._global_grad_norm = 0.

        if mpu is None:
            gd.debuginfo(prj="ds")
            self.model_parallel_group = None
            self.model_parallel_world_size = 1
            self.model_parallel_rank = 0
        else:
            gd.debuginfo(prj="ds")
            self.model_parallel_group = mpu.get_model_parallel_group()
            self.model_parallel_world_size = mpu.get_model_parallel_world_size()
            self.model_parallel_rank = bwc_tensor_model_parallel_rank(mpu)

        # 初始化一些参数
        self.overflow = False  # 是否溢出
        self.clip_grad = clip_grad  # 是否剪辑梯度
        self.communication_data_type = communication_data_type  # 通信数据类型
        self.gradient_predivide_factor = gradient_predivide_factor  # 梯度预分割因子
        self.postscale_gradients = postscale_gradients  # 是否在梯度后缩放
        self.gradient_accumulation_steps = gradient_accumulation_steps  # 梯度累积步数
        self.micro_step_id = 0  # 微步id
        self.ignore_unused_parameters = ignore_unused_parameters  # 是否忽略未使用的参数
        self.round_robin_gradients = round_robin_gradients  # 是否使用循环梯度

        self.extra_large_param_to_reduce = None  # 需要减小的额外大参数
        self.fp16_master_weights_and_gradients = fp16_master_weights_and_gradients  # 是否使用fp16主权重和梯度

        # 如果使用fp16主权重和梯度，检查cpu负载和优化器类型是否满足要求
        if self.fp16_master_weights_and_gradients:
            assert self.cpu_offload and type(self.optimizer) in [DeepSpeedCPUAdam], \
            f"fp16_master_and_gradients requires optimizer to support keeping fp16 master and gradients while keeping the optimizer states in fp32."\
            f"Currently only supported using ZeRO-Offload with DeepSpeedCPUAdam. But current setting is ZeRO-Offload:{self.cpu_offload} and optimizer type {type(self.optimizer)}." \
            f"Either disable fp16_master_weights_and_gradients or enable {self.zero_stage_string} Offload with DeepSpeedCPUAdam."

        # 如果支持reduce scatter，检查通信数据类型、梯度预分割因子和梯度预缩放是否满足要求
        if self.reduce_scatter:
            valid_reduce_scatter_dtypes = (torch.float16, torch.bfloat16, torch.float32)
            assert self.communication_data_type in valid_reduce_scatter_dtypes, f"{self.zero_stage_string} supports {valid_reduce_scatter_dtypes} communication_data_type with reduce scatter enabled. Got: '{self.communication_data_type}'"
            assert self.gradient_predivide_factor == 1.0, "gradient_predivide_factor != 1.0 is not yet supported with {self.zero_stage_string} with reduce scatter enabled"
            assert self.postscale_gradients, "pre-scale gradients is not yet supported with {self.zero_stage_string} with reduce scatter enabled"

        # param flattened by groups
        self.bit16_groups = []  # 按组划分的参数
        self.bit16_groups_flat = []  # 扁平化的参数组

        # param partitioned by data parallel degree， 并行划分的参数组
        # this will contain a list of equal sized tensors ， 它包含等尺寸的张量的列表
        # each of which will be updated by a different process， 每一个由不同进程更新
        self.parallel_partitioned_bit16_groups = []

        # a single 32-bit partition of the parallel partitioned parameters
        # that this process will update
        # 由本进程将更新的单个32位参数部分
        self.single_partition_of_fp32_groups = []

        # param partition info

        # These are the parameters in each group that will not be updated by this process directly
        # 不会由此进程直接更新的参数
        self.params_not_in_partition = []

        # These are the parameters that will be updated by this process directly
        # 将由此进程直接更新的参数  ？？区别 single_partition_of_fp32_groups
        self.params_in_partition = []

        # Offset from the first parameter in the the self.params_in_partition
        # the parameter boundaries may not align with partition boundaries
        # so we need to keep track of the offset
        # 第一个参数的偏移量
        self.first_offset = []

        # number of elements per partition in each group
        # 每个组的分区元素数量
        self.partition_size = []

        # align nccl all-gather send buffers to 4-byte boundary
        # NCCL all-gather发送缓冲区的4字节对齐
        self.nccl_start_alignment_factor = 2  # 4-byte alignment/sizeof(fp16) = 2

        assert (
            allgather_bucket_size % self.nccl_start_alignment_factor == 0
        ), f"allgather_bucket_size must be a multiple of nccl_start_alignment_factor, {self.nccl_start_alignment_factor} "

        self.all_reduce_print = False   # 是否打印all_reduce的输出
        self.dtype = self.optimizer.param_groups[0]['params'][0].dtype  # 参数数据类型

        self.round_robin_bit16_groups = []  # 初始化空的round_robin_bit16_groups
        self.round_robin_bit16_indices = []  # 初始化空的round_robin_bit16_indices

        # Use different parallel to do all_to_all_reduce related things
        # padding on each partition for alignment purposes
        # 用于对齐分区的填充
        self.groups_padding = []

        '''
        3.2. 参数分割
        接下来是一个大循环，循环处理 self.optimizer.param_groups 每个参数组，
        这里先回顾一下 optimizer.param_groups 是什么。
        
        首先 self.optimizer 是原来的基础优化器，它是 torch.optim.Optimizer 的（兼容）实例。 
        在创建 torch.optim.Optimizer 时，可以对模型参数进行分组，每组使用不同的学习率和更新参数， 
        这个 optimizer.param_groups: List[Dict] 是存储这个组的。其本身是一个 list，每个元素是一个 dict， 
        每个 dict 的key 是 dict_keys(['params', 'lr', 'betas', 'eps', 'weight_decay', 'amsgrad'])
        
        ‘params’ ：需要梯度更新的模型参数。
        
        ‘lr’, ‘betas’, ‘eps’, ‘weight_decay’, ‘amsgrad’ ： 本组参数学习率相关的配置项，可以不用管。
        '''
        # loop to deal with groups
        # 在创建 optimizer 时，可以对模型参数进行分组，每组使用不同的 学习率和更新参数
        # 这个 self.optimizer.param_groups 是存储这个组的
        # 其本身是一个 list，每个元素是一个 dict
        # self.optimizer.param_groups : List[Dict]
        # 每个 dict 的key 是 dict_keys(['params', 'lr', 'betas', 'eps', 'weight_decay', 'amsgrad'])
        # 'params' ：需要梯度更新的模型参数
        # 'lr', 'betas', 'eps', 'weight_decay', 'amsgrad' ： 本组参数学习率相关的配置项

        # 遍历优化器的参数组
        for i, param_group in enumerate(self.optimizer.param_groups):
            # 每组参数分开处理， 获取当前分区的id
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])

            gd.debuginfo(prj="ds", info=f'{i}th partition_id={partition_id}')

            # push this group to list before modify
            # TODO: Explore simplification that avoids the extra book-keeping by pushing the reordered group
            # 存储需要训练的参数
            trainable_parameters = [param for param in param_group['params'] if param.requires_grad]

            for index, param in enumerate(trainable_parameters):
                gd.debuginfo(prj="ds", info=f'{index}th param={infoTensor(param)} ')

            # 当前 param_group 中需要梯度更新，也就是需要训练的参数列表
            # 后续的分割都是针对他们的
            self.bit16_groups.append(trainable_parameters)

            # not sure why apex was cloning the weights before flattening
            # removing cloning here

            see_memory_usage(f"Before moving param group {i} to CPU")

            # move all the parameters to cpu to free up GPU space for creating flat buffer
            # 先转移到 cpu 内存，在 cpu 内存中进行处理 移动所有参数到cpu，以释放GPU空间，用于创建平坦缓冲区
            move_to_cpu(self.bit16_groups[i])
            empty_cache()

            see_memory_usage(f"After moving param group {i} to CPU", force=False)

            # Reorder group parameters for load balancing of gradient partitioning during backward among ranks.
            # This ensures that gradients are reduced in a fashion such that ownership round robins among the ranks.
            # For example, rather than 3 gradients (g_n+2, g_n+1, g_n) that are reduced consecutively belonging
            # to the same rank, instead they will belong to 3 ranks (r_m+2, r_m+1, r_m).
            # 我们要把参数分配到不同的 rank，然后每个 rank 负责部分参数的梯度计算
            # 可以先不用理具体怎么分的，反正就是按照组内进程（GPU）数量进行划分，
            # 对组参数进行重排序，以实现梯度分区在backward过程中的负载平衡
            # 通过这种方式，可以确保梯度的减少方式使所有权在等级之间轮流进行
            if self.round_robin_gradients:
                # 为了能尽量的均匀分配，这里采用循环分配（round_robin 方法）
                round_robin_tensors, round_robin_indices = self._round_robin_reorder(
                    self.bit16_groups[i], dist.get_world_size(group=self.real_dp_process_group[i]))
                gd.debuginfo(prj="ds", info=f'round_robin_indices={round_robin_indices}, \
                            round_robin_tensors={infoTensor(round_robin_tensors)}')
            else:
                round_robin_tensors = self.bit16_groups[i]
                round_robin_indices = list(range(len(self.bit16_groups[i])))
                gd.debuginfo(prj="ds", info=f'round_robin_indices={round_robin_indices}, \
                            round_robin_tensors={infoTensor(round_robin_tensors)}')

            self.round_robin_bit16_groups.append(round_robin_tensors)
            self.round_robin_bit16_indices.append(round_robin_indices)

            # create flat buffer in CPU and move to GPU
            # 将参数列表打平放到一个一维连续空间中 在CPU中创建平坦缓冲区并移动到GPU
            self.bit16_groups_flat.append(
                self.flatten_dense_tensors_aligned(
                    self.round_robin_bit16_groups[i],
                    self.nccl_start_alignment_factor * dist.get_world_size(group=self.real_dp_process_group[i])).to(
                        get_accelerator().current_device_name()))
            see_memory_usage(f"After flattening and moving param group {i} to GPU", force=True)

            # Record padding required for alignment
            # 上面在打平的时候，可能在尾部添加了padding，这里要记录一下padding的个数
            if partition_id == dist.get_world_size(group=self.real_dp_process_group[i]) - 1:
                # 如果是最后一个分区，计算填充值
                padding = self.bit16_groups_flat[i].numel() - sum(
                    [t.numel() for t in self.round_robin_bit16_groups[i]])
                gd.debuginfo(prj="ds", info=f'padding={padding}')
            else:
                # 否则，填充为0,就是不用填充
                padding = 0
            self.groups_padding.append(padding)

            if dist.get_rank(group=self.real_dp_process_group[i]) == 0:
                see_memory_usage(f"After Flattening and after emptying param group {i} cache", force=True)

            # set model bit16 weight to slices of flattened buffer
            # 更新模型的bit16权重
            self._update_model_bit16_weights(i)

            # divide the flat weights into near equal partition equal to the data parallel degree
            # each process will compute on a different part of the partition
            # data_parallel_partitions 是分割好的结果
            # data_parallel_partitions 是一个字典类型,key 是 rank ，value 是分号的参数
            # 将平坦权重划分为近等的分区，等于数据并行度
            # 每个进程将在分区的不同部分进行计算
            data_parallel_partitions = self.get_data_parallel_partitions(self.bit16_groups_flat[i], i)
            gd.debuginfo(prj="ds", info=f'data_parallel_partitions={data_parallel_partitions}')
            self.parallel_partitioned_bit16_groups.append(data_parallel_partitions)

            # verify that data partition start locations are 4-byte aligned
            # 验证数据分区起始位置是否为4字节对齐
            for partitioned_data in data_parallel_partitions:
                assert (partitioned_data.data_ptr() % (2 * self.nccl_start_alignment_factor) == 0)

            # A partition of the fp32 master weights that will be updated by this process.
            # Note that the params in single_partition_of_fp32_groups is cloned and detached
            # from the origin params of the model.
            # 把属于当前进程（rank）的参数移动到指定设备，然后创建一个副本
            # 这个副本用于累积梯度进行参数更新，根据配置，可以是 单精度（float32）也可以是半精度（float16）
            # 注意这个副本 detach 操作
            # 返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置, 不同之处只是requires_grad为false，
            # 得到的这个tensor永远不需要计算其梯度，不具有grad
            # 创建一个fp32主权重的分区，这个分区会被这个进程更新。
            # 注意，single_partition_of_fp32_groups中的参数是从模型的原始参数中克隆和分离出来的。
            if not fp16_master_weights_and_gradients:
                self.single_partition_of_fp32_groups.append(self.parallel_partitioned_bit16_groups[i][partition_id].to(
                    self.device).clone().float().detach())
            else:
                self.single_partition_of_fp32_groups.append(self.parallel_partitioned_bit16_groups[i][partition_id].to(
                    self.device).clone().half().detach())
            # self.single_partition_of_fp32_groups 中只包含属于当前进程（rank）的参数
            for index, pati in enumerate(self.single_partition_of_fp32_groups):
                gd.debuginfo(prj="ds", info=f'{index}th pati={infoTensor(pati)} ')

            # Set local optimizer to have flat params of its own partition.
            # After this, the local optimizer will only contain its own partition of params.
            # In that case, the local optimizer only saves the states(momentum, variance, etc.) related to its partition's params(zero stage1).
            # todo :  这里没理解,按照 detach 的说明，即使赋予 requires_grad = True 也不会计算梯度
            # 将本地优化器设置为拥有自己分区的扁平参数。
            # 之后，本地优化器将只包含其自己分区的参数。
            # 在这种情况下，本地优化器只保存与其分区参数有关的状态（动量、方差等）
            self.single_partition_of_fp32_groups[
                i].requires_grad = True  # keep this in case internal optimizer uses it # 保留这个，以防内部优化器使用它

            # 重置了优化器的 param_group，仅包含分给当前进程（rank）的参数
            param_group['params'] = [self.single_partition_of_fp32_groups[i]]

            # 计算分区大小和分区内的参数信息
            partition_size = len(self.bit16_groups_flat[i]) / dist.get_world_size(group=self.real_dp_process_group[i])
            params_in_partition, params_not_in_partition, first_offset = self.get_partition_info(
                self.round_robin_bit16_groups[i], partition_size, partition_id)

            # 存储分区大小和参数信息
            self.partition_size.append(partition_size)
            self.params_in_partition.append(params_in_partition)
            self.params_not_in_partition.append(params_not_in_partition)
            self.first_offset.append(first_offset)

        for rank in range(dist.get_world_size()):
            if dist.get_rank() == rank:
                gd.debuginfo(prj="ds", info=
                    f"Rank: {rank} partition count {self.partition_count} and sizes{[(p.numel(), self.is_moe_param_group[i] if hasattr(self, 'is_moe_param_group') else False) for i,p in enumerate(self.single_partition_of_fp32_groups)]} "
                )
                dist.barrier()

        # 设置一些基本参数和流
        self.reduce_bucket_size = int(reduce_bucket_size)
        self.allgather_bucket_size = int(allgather_bucket_size)
        self.reduction_event = get_accelerator().Event(enable_timing=False, blocking=False)
        self.reduction_stream = get_accelerator().Stream()
        self.cpu_computation_stream = get_accelerator().Stream()
        self.copy_grad_stream = get_accelerator().Stream()

        # 初始化一些参数和缓存列表
        self.callback_queued = False

        self.param_dict = {}

        # map between param_id and bool to specify if a param is in this partition
        self.is_param_in_current_partition = {}

        self.grads_in_ipg_bucket = []
        self.params_in_ipg_bucket = []
        self.elements_in_ipg_bucket = 0
        self.params_already_reduced = []
        self._release_ipg_buffers()
        self.previous_reduced_grads = None
        self.ipg_bucket_has_moe_params = False

        # simplified param id  # 简化参数id
        self.param_id = {}

        #interesting code: unique ids being assigned to individual parameters
        # 对每个参数进行唯一标识
        largest_param_numel = 0
        count = 0
        for i, params_group in enumerate(self.bit16_groups):
            for param in params_group:
                unique_id = id(param)
                self.param_id[unique_id] = count
                self.param_dict[count] = param
                self.params_already_reduced.append(False)
                if param.numel() > largest_param_numel:
                    largest_param_numel = param.numel()
                count = count + 1

        # 如果开启了CPU offload的功能
        # 标记参数是否在当前分区
        for param_group in self.params_in_partition:
            for param in param_group:
                self.is_param_in_current_partition[self.get_param_id(param)] = True

        for param_group in self.params_not_in_partition:
            for param in param_group:
                self.is_param_in_current_partition[self.get_param_id(param)] = False

        if self.cpu_offload:
            # 初始化一些参数，用来在CPU和GPU之间进行梯度转移
            self.accumulated_grads_in_cpu = {}  # 在CPU中累积的梯度
            self.norm_for_param_grads = {}  # 每个参数梯度的规范
            self.local_overflow = False  # 是否发生了溢出
            self.grad_position = {}  # 梯度的位置

            # 在设备上创造一个全零的tensor，用于CPU offload
            # 如果启用了内存锁页，就将数据固定在内存中，防止被交换到硬盘，加快数据传输速度
            self.temp_grad_buffer_for_cpu_offload = get_accelerator().pin_memory(
                torch.zeros(largest_param_numel, device=self.device, dtype=self.dtype))

            # 在设备上创造一个全零的tensor，用于GPU offload
            self.temp_grad_buffer_for_gpu_offload = torch.zeros(largest_param_numel,
                                                                device=get_accelerator().current_device_name(),
                                                                dtype=self.dtype)
            for i, params_group in enumerate(self.bit16_groups):
                self.get_grad_position(i, self.params_in_partition[i], self.first_offset[i], self.partition_size[i])

        # mapping from parameter to partition that it belongs to
        # 参数到它所在分区的映射
        self.param_to_partition_ids = {}

        # stores if a partition has been reduced in this step
        # 存储分区是否已经进行了reduce操作
        self.is_partition_reduced = {}

        # number of grads in partition that still need to be computed
        # 分区内还需要计算的梯度数量
        self.remaining_grads_in_partition = {}

        # total number of grads in partition
        # 分区内总共的梯度数量
        self.total_grads_in_partition = {}

        # stores if a grad in a partition has been computed or not
        # 存储分区内的梯度是否已被计算
        self.is_grad_computed = {}

        # stores the offset at which a parameter gradient needs to be inserted in a partition
        # 存储在分区中插入梯度所需要的偏移量
        self.grad_partition_insertion_offset = {}

        # the offset in the gradient at which it must be inserted at the beginning of the partition
        # 存储在分区开始的地方插入梯度所需要的偏移量
        self.grad_start_offset = {}

        # will store the averaged gradients required by this partition
        # 存储分区所需的平均梯度
        self.averaged_gradients = {}

        # For cpu_offload, will store the averaged gradients required by this partition
        # 用于CPU offload，存储分区所需的平均梯度
        self.offload_gradient_dict = {}

        # store index of first parameter in each partition
        # 存储分区中第一个参数的索引
        self.first_param_index_in_partition = {}

        # initializes all data structures for implementing gradient partitioning
        # 初始化实现梯度分区的所有数据结构
        self.initialize_gradient_partitioning_data_structures()

        # resets the data structure value for the next backward propagation
        # 重置数据结构的值以便下一次反向传播
        self.reset_partition_gradient_structures()

        # creates backward hooks for gradient partitioning
        # 如果启用了梯度分区或者通信重叠，创建后向钩子
        if self.partition_gradients or self.overlap_comm: # z1 false, z2 true
            self.create_reduce_and_remove_grad_hooks()

        self.custom_loss_scaler = False
        self.external_loss_scale = None

        # we may have a way of fusing dynamic scale. Do not support for now
        # 创建损失缩放器，可能是静态或者动态的
        self.loss_scaler = CreateLossScaler(dtype=self.dtype,
                                            static_loss_scale=static_loss_scale,
                                            dynamic_scaling=dynamic_loss_scale,
                                            dynamic_loss_args=dynamic_loss_args)
        self.dynamic_loss_scale = self.loss_scaler.dynamic
        gd.debuginfo(prj="ds", info=f"self.loss_scaler={self.loss_scaler}")
        gd.debuginfo(prj="ds", info=f"self.dynamic_loss_scale={self.dynamic_loss_scale}")

        # 只有当数据类型为float16时，才会使用动态损失缩放
        if self.dtype != torch.float16:
            # Only fp16 should use dynamic loss scaling
            assert self.loss_scaler.cur_scale == 1.0
            assert not self.dynamic_loss_scale

        see_memory_usage("Before initializing optimizer states", force=True)
        self.initialize_optimizer_states()
        see_memory_usage("After initializing optimizer states", force=True)

        # 如果是主节点，则打印优化器状态初始化成功的信息
        if dist.get_rank() == 0:
            gd.debuginfo(prj="ds", info=f"optimizer state initialized")

        # 如果是数据并行处理组的主节点，打印ZeRO优化器初始化后的内存使用情况
        if dist.get_rank(group=self.dp_process_group) == 0:
            see_memory_usage(f"After initializing ZeRO optimizer", force=True)

        # 链接所有超参数
        self._link_all_hp_params()

        # 启用通用检查点
        self._enable_universal_checkpoint()

        # 创建参数映射
        self._param_slice_mappings = self._create_param_mapping()


        gd.printall(prj='ds', cname=self)

    # 检查点启用  用于开启通用的模型检查点。
    def _enable_universal_checkpoint(self):
        # 遍历bit16_groups中的所有参数组
        for lp_param_group in self.bit16_groups:
            gd.debuginfo(prj="ds", info=f'lp_param_group={lp_param_group}')
            # 对每个参数组启用通用检查点
            enable_universal_checkpoint(param_list=lp_param_group)

    # 用于创建参数映射
    def _create_param_mapping(self):
        # 初始化一个空列表，用于保存参数映射
        param_mapping = []
        # 使用枚举函数遍历优化器的参数组，i是索引，_是参数组的内容（这里我们不需要使用内容，因此使用_作为占位符）
        for i, _ in enumerate(self.optimizer.param_groups):
            # 对于每一个参数组，我们使用一个有序字典来保存该组的参数映射
            param_mapping_per_group = OrderedDict()
            # 遍历bit16_groups中的每一个元素，这里的lp代表一个模型的层或参数
            for lp in self.bit16_groups[i]:
                # 检查lp是否有_hp_mapping属性，如果有，说明它有一些需要映射的超参数
                if lp._hp_mapping is not None:
                    # 获取该层或参数的名称
                    lp_name = self.param_names[lp]
                    # 在有序字典中添加一个新的键值对，键是层或参数的名称，值是超参数的映射地址
                    param_mapping_per_group[lp_name] = lp._hp_mapping.get_hp_fragment_address()
                    gd.printall(prj='ds', info=f'param_mapping_per_group[{lp_name}]={param_mapping_per_group[lp_name]}')
            # 将该参数组的映射添加到整体的参数映射列表中
            param_mapping.append(param_mapping_per_group)

        gd.printall(prj='ds', info=f'len of param_mapping={len(param_mapping)}')
        # 返回参数映射列表
        return param_mapping

    # 用于链接所有的超参数。这个函数的目标看起来是链接所有的半精度（16位）参数和单精度（32位）参数。
    # 它主要用于分布式训练，特别是在使用CPU offload和数据并行性（Data Parallelism）时
    def _link_all_hp_params(self):
        # 获取分布式处理过程中的世界大小
        dp_world_size = dist.get_world_size(group=self.dp_process_group)

        # 如果启用了CPU卸载，获取卸载梯度字典
        if self.cpu_offload:
            # gd.debuginfo(prj="ds")
            self.# gd.debuginfo(()

        # gd.debuginfo(prj="ds", info=f'self.optimizer.param_groups={self.optimizer.param_groups}')
        '''
        =[
        {'params': [tensor([ 0.1150, -0.1438,  0.0555,  ...,  0.0183,  0.0088,  0.0258],
        device='cuda:0', requires_grad=True)], 'weight_decay': 0.0, 'lr': 0.0, 'bias_correction': True, 'betas': (0.9, 0.95), 'eps': 1e-08, 'initial_lr': 9.65e-06, 'step': 0}, 
、       ...
        device='cuda:0', requires_grad=True)], 'weight_decay': 0.0, 'lr': 0.0, 'bias_correction': True, 'betas': (0.9, 0.95), 'eps': 1e-08, 'initial_lr': 9.65e-06, 'step': 0}
        ]
        '''
        # 遍历优化器的参数组
        for i, _ in enumerate(self.optimizer.param_groups):
            # Link bit16 and fp32 params in partition
            # 在分区中链接bit16和fp32参数, 获取实际分布式处理过程组的排名作为分区id
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])

            # 计算分区大小，即bit16群组中的元素数量除以世界大小
            partition_size = self.bit16_groups_flat[i].numel() // dp_world_size

            # 获取fp32群组的单个分区
            flat_hp_partition = self.single_partition_of_fp32_groups[i]

            # lp_param_list={self.bit16_groups[i]  tensor 的列表

            gd.debuginfo(prj='ds', info=f'flat_hp_partition={infoTensor(flat_hp_partition)}=='
                           f'gradient_dict={self.averaged_gradients}=='
                           f'offload_gradient_dict={self.offload_gradient_dict}=='
                           f'use_offload={self.cpu_offload}=='
                           f'param_group_index={i}=='
                           f'partition_start={partition_id * partition_size}=='
                           f'partition_size={partition_size}=='
                           f'partition_optimizer_state={self.optimizer.state[flat_hp_partition]}=='
                           f'dp_group={self.real_dp_process_group[i]}')

            # 链接超参数(params)
            link_hp_params(lp_param_list=self.bit16_groups[i],      # bit16参数列表
                           flat_hp_partition=flat_hp_partition,     # fp32参数的单个分区
                           gradient_dict=self.averaged_gradients,   # 平均梯度字典
                           offload_gradient_dict=self.offload_gradient_dict,  # 卸载梯度字典
                           use_offload=self.cpu_offload,            # 是否使用CPU卸载
                           param_group_index=i,                     # 参数组索引
                           partition_start=partition_id * partition_size,  # 分区开始的位置，由分区id乘以分区大小得出
                           partition_size=partition_size,           # 分区大小
                           partition_optimizer_state=self.optimizer.state[flat_hp_partition],   # 分区优化器状态，由优化器的状态和fp32参数的单个分区确定
                           dp_group=self.real_dp_process_group[i])  # 分布式处理过程组

    # 检查是否为MOE（Mixture of Experts）组
    def is_moe_group(self, group):
        # gd.debuginfo(prj="ds")
        return 'moe' in group and group['moe']

    # 用于配置MOE设置检查
    def _configure_moe_settings(self):
        gd.debuginfo(prj="ds")
        # if we're using ZeRO stage 2, ensure contiguous gradients are used
        # 如果我们使用的是ZeRO阶段2，确保使用连续梯度
        if self.partition_gradients:
            assert self.contiguous_gradients, "Contiguous Gradients in ZeRO Stage 2 must be set to True for MoE. Other code paths are not tested with MoE"

        # NOTE: To run ZeRO stage 1 with MoE, we need to set self.contiguous_gradients to True or ignore the assertion
        # 注意：要运行ZeRO阶段1与MoE，我们需要设置self.contiguous_gradients为True或忽略此断言
        if not self.partition_gradients and not self.contiguous_gradients:
            # ZeRO阶段1与MoE的配合尚未得到充分测试。这个配置仍然是实验性的
            gd.debuginfo(prj="ds", info=f"ZeRO Stage 1 has not been thoroughly tested with MoE. This configuration is still experimental.")

        assert self.reduce_scatter, "Reduce Scatter in ZeRO Stage 2 must be set to True for MoE. Other code paths are not tested with MoE"

        #  模型具有MoE层，但是没有参数组被标记为MoE。在创建优化器之前，创建一个参数组，其中'moe'键设置为True
        assert any(
            [self.is_moe_group(group) for group in self.optimizer.param_groups]
        ), "The model has moe layers, but None of the param groups are marked as MoE. Create a param group with 'moe' key set to True before creating optimizer"

        self.is_moe_param_group = []

        for i, group in enumerate(self.optimizer.param_groups):
            # 如果这是一个MoE参数组
            if self.is_moe_group(group):
                # MoE组中的所有参数都必须是MoE参数
                assert all([is_moe_param(param)
                            for param in group['params']]), "All params in MoE group must be MoE params"

                # 设置真实的数据并行进程组
                self.real_dp_process_group[i] = self.expert_dp_process_group[group['name']]

                # 设置分区数量
                self.partition_count[i] = dist.get_world_size(group=self.expert_dp_process_group[group['name']])

                # 标记为MoE参数组
                self.is_moe_param_group.append(True)
            else:
                # 不是MoE参数组
                self.is_moe_param_group.append(False)

        # 专家数据并行组应当在MoE中配置
        assert self.expert_dp_process_group is not None, "Expert data parallel group should be configured with MoE"

        # 专家并行组应当在MoE中配置
        assert self.ep_process_group is not None, "Expert parallel group should be configured with MoE"

    #更新16位浮点数权重的模型。
    def _update_model_bit16_weights(self, group_index):
        # 解压缩16位小组的数据
        updated_params = self.unflatten(self.bit16_groups_flat[group_index],
                                        self.round_robin_bit16_groups[group_index])
        gd.debuginfo(prj="ds", info=f'T: updated_params={infoTensor(updated_params)}')

        # 遍历原始小组和更新的参数，用更新的参数来更新原始参数
        for p, q in zip(self.round_robin_bit16_groups[group_index], updated_params):
            p.data = q.data

        # set model fp16 weight to slices of reordered flattened buffer
        # 将模型的16位权重设置为重新排序的扁平缓冲区的切片
        for param_index, param in enumerate(self.bit16_groups[group_index]):
            # 获取新的索引
            new_index = self.round_robin_bit16_indices[group_index][param_index]
            # 使用新的索引更新参数数据
            param.data = self.round_robin_bit16_groups[group_index][new_index].data

    # 用于在多个设备间重新排序数据。
    def _round_robin_reorder(self, tensor_list, num_partitions):
        # disable round robin if need to debug something
        # 如果需要调试某个问题，可以禁用round robin
        # return tensor_list, list(range(len(tensor_list)))

        # 创建一个字典来存储每个分区的张量
        partition_tensors = {}

        # 遍历张量列表，按照round-robin算法将张量分配到各个分区
        for i, tensor in enumerate(tensor_list):
            gd.debuginfo(prj="ds", info=f'T: {i}th tensor={infoTensor(tensor)}')

            # 计算当前张量应该分配到哪个分区
            j = i % num_partitions

            # 如果该分区还没有被分配过张量，就在字典中为这个分区创建一个空列表
            if not j in partition_tensors:
                partition_tensors[j] = []

            # 将当前张量添加到对应分区的列表中
            partition_tensors[j].append((i, tensor))

            gd.debuginfo(prj="ds", info=f'i={i}, j={j}, T: tensor={infoTensor(tensor)}')

        # 创建一个列表来存储重排序后的张量
        reordered_tensors = []
        # 创建一个字典来存储重排序后的索引
        reordered_indices = {}

        # 遍历每个分区
        for partition_index in partition_tensors.keys():
            # 遍历该分区的所有张量
            for i, (original_index, tensor) in enumerate(partition_tensors[partition_index]):
                gd.debuginfo(prj="ds", info=f'i={i},original_index={original_index}, T: tensor={infoTensor(tensor)}')

                # 记录当前张量的新索引
                reordered_indices[original_index] = len(reordered_tensors)

                # 将当前张量添加到重排序后的张量列表中
                reordered_tensors.append(tensor)

        # 返回重排序后的张量列表和索引字典
        return reordered_tensors, reordered_indices

    # 释放一些用于异步梯度累积的缓冲区
    def _release_ipg_buffers(self):
        # 如果梯度是连续的
        if self.contiguous_gradients:
            gd.debuginfo(prj="ds")
            # 释放IPG缓冲区，这个缓冲区一般用于存储临时的梯度数据
            self.ipg_buffer = None
            # 释放分区中的梯度，这个一般用于存储在模型分区中的梯度数据
            self.grads_in_partition = None
            # 重置分区中的梯度偏移量，这个一般用于记录当前处理到哪个梯度
            self.grads_in_partition_offset = 0

    # 初始化优化器状态
    def initialize_optimizer_states(self):
        gd.debuginfo(prj="ds")
        # 遍历 bit16_groups，i 是索引，group 是组
        for i, group in enumerate(self.bit16_groups):
            # 创建一个全零的张量，大小等于 partition_size[i]，数据类型和 device 都与 single_partition_of_fp32_groups[i]
            single_grad_partition = torch.zeros(int(self.partition_size[i]),
                                                dtype=self.single_partition_of_fp32_groups[i].dtype,
                                                device=self.device)
            gd.debuginfo(prj="ds", info=f'i={i},T: single_grad_partition={infoTensor(single_grad_partition)}')

            self.single_partition_of_fp32_groups[i].grad = get_accelerator().pin_memory(
                single_grad_partition) if self.cpu_offload else single_grad_partition

        # Initialize the optimizer states with the flattened fp32 partition.
        # State initialization for the Adagrad optimizer occurs at construction as opposed to other optimizers
        # which do lazy initialization of the state at the first call to step.
        # 如果优化器是 Adagrad，那么就用 single_partition_of_fp32_groups 来初始化它
        # 这是因为 Adagrad 在创建时就会初始化状态，而其他优化器则在第一次调用 step 方法时才会初始化状态
        if isinstance(self.optimizer, torch.optim.Adagrad):
            # gd.debuginfo(prj="ds")
            self.optimizer = torch.optim.Adagrad(self.single_partition_of_fp32_groups, **self.optimizer.defaults)
        else:
            # gd.debuginfo(prj="ds")
            # 其他类型的优化器则直接调用 step 方法
            self.optimizer.step()

        # 如果不进行 cpu_offload，那么就将 single_partition_of_fp32_groups 中的每个组的梯度设置为 None
        if not self.cpu_offload:
            for group in self.single_partition_of_fp32_groups:
                group.grad = None  #class init  # 初始化类

        return

    #########################################################################
    #################### ZeRO Stage 1 - reduce gradients ####################
    #########################################################################
    # reduce - 梯度。
    def reduce_gradients(self, pipeline_parallel=False):
        gd.debuginfo(prj="ds")
        # 获取集群中的计算节点总数
        world_size = dist.get_world_size(self.dp_process_group)
        # 获取当前计算节点的编号
        my_rank = dist.get_rank(self.dp_process_group)

        # 如果使用pipeline并行并且使用连续的梯度，我们需要创建ipg缓冲区，因为在这种情况下，反向传播是在zero之外处理的
        # with PP we must create ipg buffer, since backward is handled outside zero
        if pipeline_parallel and self.contiguous_gradients:
            # 创建ipg缓冲区
            self.ipg_buffer = []

            # 创建一个空的tensor，大小等于reduce_bucket_size，数据类型为self.dtype，设备为当前设备
            buf_0 = torch.empty(int(self.reduce_bucket_size),
                                dtype=self.dtype,
                                device=get_accelerator().current_device_name())

            # 将这个tensor添加到ipg缓冲区中
            self.ipg_buffer.append(buf_0)

            # 设置ipg索引为0
            self.ipg_index = 0

        gd.debuginfo(prj="ds", info=f'len of self.ipg_buffer={len(self.ipg_buffer)},\
                    T: one self.ipg_buffer={infoTensor(self.ipg_buffer[0])}')

        # 如果不使用通信重叠
        if not self.overlap_comm:
            # 遍历bit16组
            for i, group in enumerate(self.bit16_groups):
                # 遍历组内的每个参数
                for param in group:
                    gd.debuginfo(prj="ds", info=f'i={i},T: param={infoTensor(param)}')
                    # 如果梯度不为空
                    if param.grad is not None:
                        # reduce准备好的分区并移除梯度
                        self.reduce_ready_partitions_and_remove_grads(param, i)

        # reduce any pending grads in either hook/non-hook case
        # 在hook或non-hook情况下，reduce任何待处理的梯度
        self.overlapping_partition_gradients_reduce_epilogue()

    #########################################################################
    #########################ZeRO Partition Gradients########################
    #########################################################################
    # 获取第一个参数的索引
    def get_first_param_index(self, group_id, param_group, partition_id):
        # gd.debuginfo(prj="ds")
        # 遍历参数组中的每一个参数
        for index, param in enumerate(param_group):
            # 获取参数的ID
            param_id = self.get_param_id(param)
            gd.debuginfo(prj="ds", info=f'index={index}, param_id={param_id}, T: param={infoTensor(param)}')

            # 检查当前的参数ID是否在指定的分区ID内
            # 如果在，就返回当前的索引
            if partition_id in self.param_to_partition_ids[group_id][param_id]:
                return index

        # 如果没有找到满足条件的参数，就返回None
        return None

    # 初始化梯度分区的数据结构
    def initialize_gradient_partitioning_data_structures(self):
        # gd.debuginfo(prj="ds")
        # 遍历所有的参数组
        for i, param_group in enumerate(self.round_robin_bit16_groups):
            # 获取分区的总数，也就是分布式处理组的大小
            total_partitions = dist.get_world_size(group=self.real_dp_process_group[i])

            # 为每个参数组初始化相关的数据结构
            self.param_to_partition_ids[i] = {}
            self.is_partition_reduced[i] = {}
            self.total_grads_in_partition[i] = {}
            self.remaining_grads_in_partition[i] = {}
            self.is_grad_computed[i] = {}
            self.grad_partition_insertion_offset[i] = {}
            self.grad_start_offset[i] = {}
            self.first_param_index_in_partition[i] = {}

            # 为每个分区初始化相关的数据结构
            for partition_id in range(total_partitions):
                self.is_grad_computed[i][partition_id] = {}
                self.grad_partition_insertion_offset[i][partition_id] = {}
                self.grad_start_offset[i][partition_id] = {}
                # 初始时每个分区中的梯度总数为0
                self.total_grads_in_partition[i][partition_id] = 0
                # 初始化每个分区的梯度值
                self.initialize_gradient_partition(i, param_group, partition_id)
                # 初始化每个分区的缩减状态为False
                self.is_partition_reduced[i][partition_id] = False
                # 获取并存储每个分区的第一个参数的索引
                self.first_param_index_in_partition[i][partition_id] = self.get_first_param_index(
                    i, param_group, partition_id)
                gd.debuginfo(prj='ds',
                             info=f'first_param_index_in_partition[{i}][{partition_id}]={self.first_param_index_in_partition[i][partition_id]}')

    def independent_gradient_partition_epilogue(self):
        # gd.debuginfo(prj="ds")
        # 报告在执行梯度reduce之前的内存使用情况
        self.report_ipg_memory_usage(f"In ipg_epilogue before reduce_ipg_grads", 0)

        # 执行梯度reduce
        self.reduce_ipg_grads()

        # 报告在执行梯度缩减之后的内存使用情况
        self.report_ipg_memory_usage(f"In ipg_epilogue after reduce_ipg_grads", 0)

        if dist.get_rank() == 0:
           gd.debuginfo(prj="ds", info="Params already reduced ={self.params_already_reduced}")

        # 将所有参数的已缩减标记设置为False
        for i in range(len(self.params_already_reduced)):
            self.params_already_reduced[i] = False

        # 如果开启了通信的并行处理
        if self.overlap_comm:
            gd.debuginfo(prj="ds")
            # 等待所有的运算都完成
            get_accelerator().synchronize()

            # It is safe to clear previously reduced grads of other partitions
            # 清理之前已经缩减过的梯度的数据
            self._clear_previous_reduced_grads()

        # 如果不进行CPU卸载
        if self.cpu_offload is False:
            gd.debuginfo(prj="ds")
            # 对于所有的参数组
            for i, _ in enumerate(self.bit16_groups):
                # 如果该参数组还没有平均梯度，就计算并存储
                if not i in self.averaged_gradients or self.averaged_gradients[i] is None:
                    self.averaged_gradients[i] = self.get_flat_partition(
                        self.params_in_partition[i],
                        self.first_offset[i],
                        self.partition_size[i],
                        dtype=self.dtype,
                        device=get_accelerator().current_device_name(),
                        return_tensor_list=True)
                else:
                    # 如果该参数组已经有平均梯度，就将新的平均梯度添加到已有的梯度上
                    avg_new = self.get_flat_partition(self.params_in_partition[i],
                                                      self.first_offset[i],
                                                      self.partition_size[i],
                                                      dtype=self.dtype,
                                                      device=get_accelerator().current_device_name(),
                                                      return_tensor_list=True)

                    for accumulated_grad, new_avg_grad in zip(self.averaged_gradients[i], avg_new):
                        accumulated_grad.add_(new_avg_grad)

        # 释放independent parameter gradient (ipg)相关的内存
        self._release_ipg_buffers()

        # No need to keep the gradients anymore.
        # All gradients required by the step are in self.averaged_gradients
        # 没有必要再保留梯度信息了，因为所有步骤所需的梯度都已存储在self.averaged_gradients中。清空所有的梯度
        self.zero_grad(set_to_none=True)

        # 报告在执行完independent_gradient_partition_epilogue函数后的内存使用情况
        see_memory_usage(f"End ipg_epilogue")

    # resets all partition to no reduced
    # sets remaining grads to the total number of grads in each partition
    # set is grad computed to false for all grads in partition
    # 重置与每个分区相关的梯度结构
    def reset_partition_gradient_structures(self): #也就是init执行了一次
        # gd.debuginfo(prj="ds")
        # 遍历所有的参数组
        for i, _ in enumerate(self.bit16_groups):
            # 获取分区的总数，这是通过获取分布式处理组的大小来决定的
            total_partitions = dist.get_world_size(group=self.real_dp_process_group[i])

            # 遍历所有的分区
            for partition_id in range(total_partitions):
                gd.debuginfo(prj="ds", info=f'i={i}, partition_id={partition_id}')

                # 将每个分区的缩减状态设为False
                self.is_partition_reduced[i][partition_id] = False

                # 将每个分区剩余的梯度数量设置为每个分区的梯度总数
                self.remaining_grads_in_partition[i][partition_id] = self.total_grads_in_partition[i][partition_id]

                # 遍历分区中每个参数的梯度计算状态
                for param_id in self.is_grad_computed[i][partition_id]:
                    gd.debuginfo(prj="ds", info=f'i={i}, partition_id={partition_id}, param_id={param_id}')
                    # 将每个参数的梯度计算状态设为False
                    self.is_grad_computed[i][partition_id][param_id] = False

    # 初始化reduce分区
    def initialize_gradient_partition(self, i, param_group, partition_id):

        gd.debuginfo(prj="ds", info=f'i={i}+++partition_id={partition_id}')
        # param_group={infoTensor(param_group)} 是一个tensor的列表，太大
        print(f"len of param_group={len(param_group)}")  # 防止消失，直接打印
        for index, v in enumerate(param_group):
            gd.debuginfo(prj='ds', info=f"param_group[{index}]={infoTensor(param_group[index])}")

        # 如果key在里面，那么把value加入key到对应的列表中，否则新建立一个key->[value]  ??为什么不用defautdict(list)
        # 定义一个函数，用于在字典中设置键值对，如果键已经存在，就在值列表中添加新值，否则创建一个新列表
        def set_key_value_list(dictionary, key, value):
            if key in dictionary:
                # gd.debuginfo(prj="ds")
                dictionary[key].append(value)
            else:
                # gd.debuginfo(prj="ds")
                dictionary[key] = [value]

        # 如果key在里面，那么把对应的value加1，否则新建立一个key->1 ??为什么不用defautdict(1)
        # 定义一个函数，用于在字典中递增相应键的值，如果键不存在，就设置为1
        def increment_value(dictionary, key):
            if key in dictionary:
                # gd.debuginfo(prj="ds")
                dictionary[key] += 1
            else:
                # gd.debuginfo(prj="ds")
                dictionary[key] = 1

        # 获取分区大小
        partition_size = self.partition_size[i]

        # 计算分区的起始和结束索引
        start_index = partition_size * partition_id
        end_index = partition_size * (partition_id + 1)

        gd.debuginfo(prj="ds", info=f'start_index={start_index}+++end_index={end_index}+++partition_size={partition_size}')

        current_index = 0
        first_offset = 0

        # 遍历参数组
        for param in param_group:
            # 获取参数的大小和ID
            param_size = param.numel()
            param_id = self.get_param_id(param)
            gd.debuginfo(prj="ds", info=f'param={infoTensor(param)}+++param_size={param_size}+++param_id={param_id}')

            if start_index <= current_index < end_index:
                gd.debuginfo(prj="ds", info=f'当前索引在分区范围内:'
                                            f'start_index={start_index}, '
                                            f'current_index={current_index}, '
                                            f'end_index={end_index}')
                # 更新各种字典和列表
                set_key_value_list(self.param_to_partition_ids[i], param_id, partition_id)
                increment_value(self.total_grads_in_partition[i], partition_id)

                self.is_grad_computed[i][partition_id][param_id] = False

                self.grad_partition_insertion_offset[i][partition_id][param_id] = current_index - start_index
                self.grad_start_offset[i][partition_id][param_id] = 0

            elif current_index < start_index < (current_index + param_size):
                gd.debuginfo(prj="ds",
                             info=f'# 当前参数跨越了分区的起始边界:current_index={current_index}, '
                                  f'start_index={start_index}, '
                                  f'current_index + param_size={current_index + param_size}')

                #"这个情况应该只会发生一次，因为这必须是分区中的第一个张量"
                assert (first_offset == 0
                        ), "This can happen either zero or only once as this must be the first tensor in the partition"
                first_offset = start_index - current_index

                set_key_value_list(self.param_to_partition_ids[i], param_id, partition_id)
                increment_value(self.total_grads_in_partition[i], partition_id)

                self.is_grad_computed[i][partition_id][param_id] = False

                self.grad_partition_insertion_offset[i][partition_id][param_id] = 0
                self.grad_start_offset[i][partition_id][param_id] = first_offset

            # 更新当前索引
            current_index = current_index + param_size

    # 调用IGP的梯度reduce操作
    def overlapping_partition_gradients_reduce_epilogue(self):
        # gd.debuginfo(prj="ds")
        self.independent_gradient_partition_epilogue()

    # 创建并删除梯度钩子
    def create_reduce_and_remove_grad_hooks(self):
        # gd.debuginfo(prj="ds")
        # 初始化一个用于存储梯度累积函数的列表
        self.grad_accs = []

        # 遍历所有的16位分组
        for i, param_group in enumerate(self.bit16_groups):
            # 在每个分组中遍历所有的参数
            for param in param_group:
                gd.debuginfo(prj="ds", info=f'i={i}, param={infoTensor(param)}')
                # 如果参数需要计算梯度
                if param.requires_grad:
                    # 定义一个闭包函数，用于注册梯度钩子
                    def wrapper(param, i):
                        # 创建一个与参数形状相同的临时参数
                        param_tmp = param.expand_as(param)
                        gd.debuginfo(prj="ds", info=f'param_tmp={infoTensor(param_tmp)}')

                        # 获取梯度函数的下一个函数
                        grad_acc = param_tmp.grad_fn.next_functions[0][0]
                        gd.debuginfo(prj="ds", info=f'grad_acc={infoTensor(grad_acc)}')

                        # 定义一个函数，用于在需要的时候减少分区并移除梯度
                        def reduce_partition_and_remove_grads(*notneeded):
                            self.reduce_ready_partitions_and_remove_grads(param, i)

                        # 为梯度函数注册一个钩子，当梯度计算完成时会自动调用这个钩子
                        grad_acc.register_hook(reduce_partition_and_remove_grads)

                        # 将梯度函数添加到列表中
                        self.grad_accs.append(grad_acc)

                    # 调用闭包函数，注册梯度钩子
                    wrapper(param, i)

    # 获取参数ID。
    def get_param_id(self, param):
        # gd.debuginfo(prj="ds")
        # 使用python内置的id函数获取param的唯一id。
        # id函数返回对象的唯一标识符，此标识符是在该对象的生命周期内恒定的。
        unique_id = id(param)

        # 从self.param_id字典中获取对应的值。
        # self.param_id应该是一个字典，存储了参数（param）的唯一id和对应的值。
        # 此行代码返回的是与param的唯一id对应的值。
        return self.param_id[unique_id]

    # 报告IPG（Independent Parallel Gradient）内存使用情况
    def report_ipg_memory_usage(self, tag, param_elems):
        # 计算当前在 IPG 桶中的元素数量加上传入的 param_elems 参数的总数
        elem_count = self.elements_in_ipg_bucket + param_elems
        gd.debuginfo(prj="ds", info=f'param_elems={param_elems}')

        # 计算 elem_count 占总桶大小的百分比
        percent_of_bucket_size = (100.0 * elem_count) // self.reduce_bucket_size

        # 字符串格式化，包含了标签，IPG桶中元素的数量，参数元素的数量和占总桶大小的百分比
        see_memory_usage(
            f"{tag}: elems in_bucket {self.elements_in_ipg_bucket} param {param_elems} max_percent {percent_of_bucket_size}"
        )

    # create a flat tensor aligned at the alignment boundary
    # 按照指定的对齐方式首先进行对齐，然后再将对齐后的张量扁平化
    def flatten_dense_tensors_aligned(self, tensor_list, alignment):
        # 这个函数接受两个参数，一个是 tensor_list，是一个包含多个张量（tensor）的列表，另一个是 alignment，表示对齐方式
        # 这个函数的目标是将 tensor_list 中的所有张量首先进行对齐，然后再进行扁平化处理

        # 调用 align_dense_tensors 函数，对 tensor_list 中的每一个张量进行对齐，
        # align_dense_tensors 函数的返回值是一个新的张量列表，其中的每个张量都已经根据 alignment 对齐
        aligned_tensors = align_dense_tensors(tensor_list, alignment)

        # 调用 flatten 函数，对经过对齐处理的张量列表进行扁平化处理
        # flatten 函数的返回值是一个新的扁平化后的张量
        flattened_tensor = self.flatten(aligned_tensors)

        gd.debuginfo(prj="ds", info=f'T: aligned_tensors={infoTensor(aligned_tensors)}, flattened_tensor={infoTensor(flattened_tensor)}')

        # 返回扁平化处理后的张量
        return flattened_tensor


    ############### Independent Partition Gradient ########################
    # reduce ipg的梯度桶并删除梯度
    def reduce_independent_p_g_buckets_and_remove_grads(self, param, i):
        gd.debuginfo(prj="ds")

        # 如果当前bucket中的元素数量加上参数的元素数量超过了bucket的大小
        if self.elements_in_ipg_bucket + param.numel() > self.reduce_bucket_size:
            gd.debuginfo(prj="ds")

            # 报告当前的内存使用情况
            self.report_ipg_memory_usage("In ipg_remove_grads before reduce_ipg_grads", param.numel())

            # 减少ipg的梯度
            self.reduce_ipg_grads()

            # 如果开启了连续的梯度并且开启了通信重叠，那么交换ipg_index的值
            if self.contiguous_gradients and self.overlap_comm:
                # Swap ipg_index between 0 and 1
                self.ipg_index = 1 - self.ipg_index

            # 报告当前的内存使用情况
            self.report_ipg_memory_usage("In ipg_remove_grads after reduce_ipg_grads", param.numel())

        # 获取参数的id
        param_id = self.get_param_id(param)

        # 检查该参数是否已经被减少，如果已经被减少，则抛出异常
        assert self.params_already_reduced[param_id] == False, \
            f"The parameter {param_id} has already been reduced. \
            Gradient computed twice for this partition. \
            Multiple gradient reduction is currently not supported"

        # 如果开启了连续的梯度
        if self.contiguous_gradients:
            gd.debuginfo(prj="ds")
            if param.numel() > self.reduce_bucket_size:
                # 如果参数的元素数量大于bucket的大小，那么将该参数设置为待减少的参数
                gd.debuginfo(prj="ds")
                self.extra_large_param_to_reduce = param
            else:
                # 保持梯度连续，以防止内存碎片化，并且避免展平
                gd.debuginfo(prj="ds")
                # keeping the gradients contiguous to prevent memory fragmentation, and avoid flattening
                new_grad_tensor = self.ipg_buffer[self.ipg_index].narrow(0, self.elements_in_ipg_bucket, param.numel())
                new_grad_tensor.copy_(param.grad.view(-1))
                param.grad.data = new_grad_tensor.data.view_as(param.grad)

        # 更新bucket中的元素数量
        self.elements_in_ipg_bucket += param.numel()

        # 检查用于减少的梯度是否为None，如果为None则抛出异常
        assert param.grad is not None, f"rank {dist.get_rank()} - Invalid to reduce Param {param_id} with None gradient"

        # 将用于减少的梯度添加到bucket中
        self.grads_in_ipg_bucket.append(param.grad)

        # 将参数添加到bucket中
        self.params_in_ipg_bucket.append((i, param, param_id))

        # make sure the average tensor function knows how to average the gradients
        # 确保平均张量函数知道如何平均梯度
        if is_moe_param(param):
            gd.debuginfo(prj="ds")
            self.ipg_bucket_has_moe_params = True

        # 报告当前的内存使用情况
        self.report_ipg_memory_usage("End ipg_remove_grads", 0)

    # def print_rank_0(self, message):
    # `dist.get_rank()`是获取当前进程的标识
    # 当标识为0时，代表这是主进程
    #     if dist.get_rank() == 0:
    #         logger.info(message)

    def print_rank_0(self, message):
        if dist.get_rank() == 0:
            gd.debuginfo(prj='ds', info=message, level=2)

    # 实现了分布式训练中的梯度同步，确保每个进程的模型参数更新是一致的
    def gradient_reduction_w_predivide(self, tensor):
        # 获取当前分布式处理组的大小
        dp_world_size = dist.get_world_size(group=self.dp_process_group)

        gd.debuginfo(prj="ds", info=f'T: tensor={infoTensor(tensor)}')

        tensor_to_allreduce = tensor
        # 如果通信数据类型与张量数据类型不同，将张量转换为通信数据类型
        if self.communication_data_type != tensor.dtype:
            tensor_to_allreduce = tensor.to(self.communication_data_type)
            gd.debuginfo(prj="ds", info=f'T: tensor_to_allreduce={infoTensor(tensor_to_allreduce)}')

        if self.postscale_gradients:
            # 如果存在预分割因子，将张量乘以预分割因子的倒数
            if self.gradient_predivide_factor != 1.0:
                tensor_to_allreduce.mul_(1. / self.gradient_predivide_factor)
                gd.debuginfo(prj="ds", info=f'T: tensor_to_allreduce={infoTensor(tensor_to_allreduce)}')

            # 在所有设备上执行全部规约操作，即在所有设备上求和
            dist.all_reduce(tensor_to_allreduce, group=self.dp_process_group)

            # 如果预分割因子不等于处理组的大小，将张量乘以预分割因子与处理组大小的比值
            if self.gradient_predivide_factor != dp_world_size:
                tensor_to_allreduce.mul_(self.gradient_predivide_factor / dp_world_size)
                gd.debuginfo(prj="ds", info=f'T: tensor_to_allreduce={infoTensor(tensor_to_allreduce)}')
        else:
            # 如果不进行后缩放梯度，将张量除以处理组的大小
            tensor_to_allreduce.div_(dp_world_size)
            dist.all_reduce(tensor_to_allreduce, group=self.dp_process_group)
            gd.debuginfo(prj="ds", info=f'T: tensor_to_allreduce={infoTensor(tensor_to_allreduce)}')

        # 如果通信数据类型与张量数据类型不同，并且张量与待规约张量不是同一个，将待规约张量的值复制到原始张量
        if self.communication_data_type != tensor.dtype and tensor is not tensor_to_allreduce:
            tensor.copy_(tensor_to_allreduce)
            gd.debuginfo(prj="ds", info=f'T: tensor={infoTensor(tensor)}')

        return tensor

    # 计算张量的平均值 , only call by reduce_ipg_grads
    def average_tensor(self, tensor):
        # 如果允许重叠通信
        if self.overlap_comm:
            gd.debuginfo(prj="ds")
            # 等待当前设备完成任务
            stream = self.reduction_stream
            stream.wait_stream(get_accelerator().current_stream())
        else:
            gd.debuginfo(prj="ds")
            # 如果不允许重叠通信，使用当前设备的流
            stream = get_accelerator().current_stream()

        # 设置当前设备的流
        with get_accelerator().stream(stream):
            # 如果不进行reduce scatter操作，直接对tensor进行梯度减少操作，然后返回
            if not self.reduce_scatter:
                self.gradient_reduction_w_predivide(tensor)
                return

            # Accumulate destination ranks and bucket offsets for each gradient slice.
            # Note: potential future optimization, record access pattern of parameters
            # in backward pass and partition gradients w.r.t. access pattern so that our
            # bucket is guaranteed to be contiguous w.r.t. ranks
            # 进行reduce scatter操作的逻辑
            rank_and_offsets = []
            real_dp_process_group = []
            curr_size = 0
            prev_id, prev_process_group = -1, None

            process_group = self.dp_process_group
            # count = 0
            for i, param, param_id in self.params_in_ipg_bucket:
                # 获取梯度进行减少操作
                process_group = self.dp_process_group

                #Averages gradients at parameter level if ipg has a moe param
                #Otherwise averaging is done at the entire buffer level at the end of the loop
                # MoE param have different groups

                if self.ipg_bucket_has_moe_params:
                    # 如果参数是moe类型，需要特别处理
                    process_group = self.expert_dp_process_group[param.group_name] if is_moe_param(
                        param) else self.dp_process_group

                    # 对梯度进行平均
                    param.grad.data.div_(dist.get_world_size(group=process_group))

                partition_ids = self.param_to_partition_ids[i][param_id]
                assert all([p_id < dist.get_world_size(group=process_group) for p_id in partition_ids
                            ]), f"world size {dist.get_world_size(group=process_group)} and p_ids: {partition_ids}"
                partition_size = self.partition_size[i]
                # Get all partition ids + their offsets
                partition_ids_w_offsets = []

                for partition_id in partition_ids:
                    offset = self.grad_start_offset[i][partition_id][param_id]
                    partition_ids_w_offsets.append((partition_id, offset))

                # 根据offset对partition排序
                partition_ids_w_offsets.sort(key=lambda t: t[1])

                # Calculate rank and offsets for grad slices
                for idx in range(len(partition_ids_w_offsets)):
                    # 获取当前的分区ID和偏移量
                    partition_id, offset = partition_ids_w_offsets[idx]

                    # if dist.get_rank() == 0 and count < 100:
                    #     print(f"Rank {dist.get_rank()} rank offset id {idx} calculated dp size {dist.get_world_size(group=process_group)} real dp size {dist.get_world_size(self.real_dp_process_group[i])} and dst: {partition_id}")
                    # count += 1

                    # Calculate numel for grad slice depending on partition location
                    # 如果是最后一个分区，那么计算的元素数量为参数总元素数量减去偏移量
                    if idx == len(partition_ids_w_offsets) - 1:
                        # Last partition_id uses its own offset
                        numel = param.numel() - offset
                    else:
                        # Set numel to next partition's offset
                        # 如果不是最后一个分区，那么计算的元素数量为下一个分区的偏移量减去当前分区的偏移量
                        numel = partition_ids_w_offsets[idx + 1][1] - offset

                    # Merge bucket ranges if they belong to the same rank
                    # 如果当前的分区ID和上一个分区ID相同，并且处理组也相同
                    if partition_id == prev_id and process_group == prev_process_group:
                        # 获取上一个分区的ID，大小和元素数量
                        prev_pid, prev_size, prev_numel = rank_and_offsets[-1]

                        # 更新上一个分区的元素数量，加上当前分区的元素数量
                        rank_and_offsets[-1] = (prev_pid, prev_size, prev_numel + numel)
                    else:
                        # 如果分区ID或处理组和上一个不同，那么将当前的分区ID，当前的大小和元素数量添加到名次和偏移量的列表中
                        rank_and_offsets.append((partition_id, curr_size, numel))

                        # 将当前的处理组添加到真实的dp处理组列表中
                        real_dp_process_group.append(process_group)

                    # 更新当前的大小，加上当前的元素数量
                    curr_size += numel

                    # 更新上一个分区的ID和处理组为当前的分区ID和处理组
                    prev_id, prev_process_group = partition_id, process_group

            if not self.ipg_bucket_has_moe_params:
                # 对tensor进行平均操作
                tensor.div_(dist.get_world_size(group=self.dp_process_group))

            tensor_to_reduce = tensor
            if self.communication_data_type != tensor.dtype:
                # 如果通信数据类型和tensor数据类型不一致，进行转换
                tensor_to_reduce = tensor.to(self.communication_data_type)

            # 进行reduce操作，将结果分散存储在不同的节点上
            async_handles = []
            for i, (dst, bucket_offset, numel) in enumerate(rank_and_offsets):
                # 对需要减少的张量进行切片，获取需要进行reduce操作的部分
                grad_slice = tensor_to_reduce.narrow(0, int(bucket_offset), int(numel))

                # if dist.get_rank() == 0:
                #     print(f"Rank {dist.get_rank()} rank offset id {i} real dp size {dist.get_world_size(group=real_dp_process_group[i])} and dst: {dst}")
                # dist.barrier()
                #dist.barrier()

                # 获取目标节点的全局rank
                dst_rank = dist.get_global_rank(real_dp_process_group[i], dst)

                # 异步进行reduce操作，将grad_slice的数据减少到目标节点上，这是一个非阻塞操作，会立即返回一个句柄
                async_handle = dist.reduce(grad_slice, dst=dst_rank, group=real_dp_process_group[i], async_op=True)

                # 将异步操作的句柄添加到列表中，以便后续等待所有的reduce操作都完成
                async_handles.append(async_handle)

            # 等待所有的reduce操作完成
            for handle in async_handles:
                handle.wait()

            # 如果通信数据类型和tensor数据类型不一致，将tensor_to_reduce的数据复制到tensor中
            if self.communication_data_type != tensor.dtype:
                tensor.copy_(tensor_to_reduce)

    ##############################################################################
    ############################# CPU Offload Methods#############################
    ##############################################################################
    # 获取梯度位置, only call in init
    def get_grad_position(self, group_id, tensor_list, first_offset, partition_size):
        # 当前已处理的元素偏移量
        current_offset = 0

        # 遍历所有张量
        for i, tensor in enumerate(tensor_list):
            # 获取当前张量的ID
            param_id = self.get_param_id(tensor)

            # 设定当前张量的起始偏移量
            param_start_offset = 0

            # 获取当前张量的元素总数
            num_elements = tensor.numel()

            gd.debuginfo(prj="ds", info=f'group_id={group_id}, i={i}, tensor={infoTensor(tensor)}')
            gd.debuginfo(prj="ds", info=f'param_id={param_id}, param_start_offset={param_start_offset}, num_elements={num_elements}')

            # we need to offset to get to the right element
            # 我们需要偏移以获取到正确的元素
            # 如果是第一个张量且first_offset大于0
            if i == 0 and first_offset > 0:
                # 张量偏移量为first_offset
                tensor_offset = first_offset

                # 张量元素总数要减去偏移量
                num_elements = num_elements - tensor_offset

                # 当前张量的起始偏移量为first_offset
                param_start_offset = first_offset

            # we dont need all elements of the tensor
            # 我们不需要张量的所有元素
            # 如果当前张量的元素总数大于分区剩余空间
            if num_elements > (partition_size - current_offset):
                # 张量元素总数为分区剩余空间
                num_elements = partition_size - current_offset

            # 记录当前张量的信息到grad_position中
            # 其中包括：组ID，起始偏移量，当前偏移量，元素总数
            self.grad_position[param_id] = [
                int(group_id), int(param_start_offset),
                int(current_offset), int(num_elements)
            ]

            gd.debuginfo(prj="ds", info=f'self.grad_position[{param_id}]={param_id}++'
                                        f'int(group_id)={int(group_id)}, int(param_start_offset)={int(param_start_offset)}, int(current_offset)={int(current_offset)}, int(num_elements)={int(num_elements)}')

            # 更新当前已处理的元素偏移量
            current_offset += num_elements

    # 更新参数梯度的溢出跟踪器, only call in copy_grads_in_partition
    def update_overflow_tracker_for_param_grad(self, param):
        # gd.debuginfo(prj="ds")
        # 如果梯度不为空并且数据中存在无穷或NaN值
        if param.grad is not None and self._has_inf_or_nan(param.grad.data):
            # gd.debuginfo(prj="ds")
            self.local_overflow = True

    # only call in _link_all_hp_params  获取卸载梯度的字典
    def _get_offload_gradient_dict(self):
        # 遍历优化器的所有参数组
        for param_group_index, _ in enumerate(self.optimizer.param_groups):
            # 初始化当前参数组的梯度字典
            self.offload_gradient_dict[param_group_index] = []

            # 遍历当前参数组的所有参数
            for lp_param in self.params_in_partition[param_group_index]:
                # 获取参数的ID
                param_id = self.get_param_id(lp_param)

                # 获取参数的梯度位置信息
                [_, _, dest_offset, num_elements] = self.grad_position[param_id]

                # 根据位置信息，从单个分区的fp32组中获取对应的梯度张量
                dest_tensor = self.single_partition_of_fp32_groups[param_group_index].grad.view(-1).narrow(
                    0, dest_offset, num_elements)

                gd.debuginfo(prj="ds", info=f'param_group_index={param_group_index}, param_id={param_id}, num_elements={num_elements}, dest_offset={dest_offset}, dest_tensor={infoTensor(dest_tensor)}')

                # 将梯度张量添加到当前参数组的梯度字典中
                self.offload_gradient_dict[param_group_index].append(dest_tensor)

    # 通过GPU在CPU上异步累积梯度， only call in copy_grads_in_partition
    def async_accumulate_grad_in_cpu_via_gpu(self, param):
        # gd.debuginfo(prj="ds")
        # 获取参数的ID
        param_id = self.get_param_id(param)

        # 获取梯度的位置信息
        [i, source_offset, dest_offset, num_elements] = self.grad_position[param_id]
        gd.debuginfo(prj="ds", info=f'i={i}, source_offset={source_offset}, dest_offset={dest_offset}, num_elements={num_elements}, param_id={param_id}')

        # copy to a preexisiting buffer to avoid memory allocation penalty
        dest_buffer = self.temp_grad_buffer_for_gpu_offload.view(-1).narrow(0, 0, param.numel())
        gd.debuginfo(prj="ds", info=f'dest_buffer={dest_buffer}')

        #buffer for storing gradients for this parameter in CPU
        # 为在CPU中存储此参数的梯度创建缓冲区
        def buffer_to_accumulate_to_in_cpu():
            if not self.fp16_master_weights_and_gradients:
                # 如果不是使用16位浮点数的权重和梯度，则创建一个缓冲区，大小为参数元素的数量，数据类型为参数的数据类型，设备为当前设备
                gd.debuginfo(prj="ds")
                return get_accelerator().pin_memory(torch.zeros(param.numel(), dtype=param.dtype, device=self.device))
            else:
                gd.debuginfo(prj="ds")
                # 如果是使用16位浮点数的权重和梯度，则返回属于这个分区的fp32组的梯度，视图改为1维，并且截取从dest_offset开始，长度为num_elements的部分
                return self.single_partition_of_fp32_groups[i].grad.view(-1).narrow(0, dest_offset, num_elements)

        #accumulate gradients into param.grad or parts of it that belongs to this partition
        # 将梯度累积到param.grad_accum或者属于这个分区的部分
        def accumulate_gradients():
            if not self.fp16_master_weights_and_gradients:
                gd.debuginfo(prj="ds")

                # 如果不是使用16位浮点数的权重和梯度
                # 那么将CPU中已累积的梯度（将其视图改为1维）复制到目标缓冲区中，这个过程是非阻塞的
                dest_buffer.copy_(self.accumulated_grads_in_cpu[param_id].view(-1), non_blocking=True)

                # 然后将目标缓冲区中的内容添加到参数梯度的数据中（将其视图改为1维）
                param.grad.data.view(-1).add_(dest_buffer)
            else:
                gd.debuginfo(prj="ds")
                # 如果是使用16位浮点数的权重和梯度
                # 那么将CPU中已累积的梯度（将其视图改为1维）复制到目标缓冲区的指定部分中，这个过程是非阻塞的

                dest_buffer.narrow(0, source_offset,
                                   num_elements).copy_(self.accumulated_grads_in_cpu[param_id].view(-1),
                                                       non_blocking=True)
                # 然后将目标缓冲区中的指定部分的内容添加到参数梯度的数据的指定部分中
                param.grad.data.view(-1).narrow(0, source_offset,
                                                num_elements).add_(dest_buffer.narrow(0, source_offset, num_elements))

        # move accumulated gradients back to CPU
        # 将累积的梯度移回到CPU
        def copy_gradients_to_cpu():
            if not self.fp16_master_weights_and_gradients:
                gd.debuginfo(prj="ds")
                # 如果不是使用16位浮点数的权重和梯度，则将参数梯度的数据（将其视图改为1维）复制到CPU中已累积的梯度中，这个过程是非阻塞的
                self.accumulated_grads_in_cpu[param_id].data.copy_(param.grad.data.view(-1), non_blocking=True)
            else:
                gd.debuginfo(prj="ds")
                # 如果是使用16位浮点数的权重和梯度，则将参数梯度的数据的指定部分（将其视图改为1维并截取指定部分）复制到CPU中已累积的梯度中，这个过程是非阻塞的
                self.accumulated_grads_in_cpu[param_id].data.copy_(param.grad.data.view(-1).narrow(
                    0, source_offset, num_elements),
                                                                   non_blocking=True)

        if param_id not in self.accumulated_grads_in_cpu:
            gd.debuginfo(prj="ds")
            # 如果CPU中还没有当前参数的已累积梯度，则创建一个缓冲区用于在CPU中累积梯度
            self.accumulated_grads_in_cpu[param_id] = buffer_to_accumulate_to_in_cpu()

        if self.micro_step_id > 0:
            gd.debuginfo(prj="ds")
            # 如果微步长大于0，则累积梯度
            accumulate_gradients()

        # at the boundary we will send 32bit directly
        if not self.is_gradient_accumulation_boundary:
            gd.debuginfo(prj="ds")
            # 在边界处，我们将直接发送32位
            copy_gradients_to_cpu()

    # 为参数梯度设置范数  没有被触发？？？？
    def set_norm_for_param_grad(self, param):
        gd.debuginfo(prj="ds")
        # 获取参数的ID
        param_id = self.get_param_id(param)

        # 根据梯度累积步骤的数量，来决定使用哪种方式来获取累积梯度
        # 如果梯度累积步骤大于1，则从存储在CPU中的累积梯度列表中获取，否则直接使用grad_accum
        accumulated_grad = self.accumulated_grads_in_cpu[
            param_id] if self.gradient_accumulation_steps > 1 else param.grad

        # 从梯度位置列表中获取相关信息
        [i, source_offset, dest_offset, num_elements] = self.grad_position[param_id]

        # 定义开始位置
        start = source_offset

        # 对累积梯度进行调整，首先将其视图（view）调整为1维，然后从start开始，获取num_elements个元素
        accumulated_grad = accumulated_grad.view(-1).narrow(0, start, num_elements)

        # 计算调整后的累积梯度的2范数（即欧几里得范数，或者叫做欧氏距离），并将其设置为对应参数梯度的范数
        self.norm_for_param_grads[param_id] = accumulated_grad.data.double().norm(2)

    # 在GPU中为参数梯度设置范数
    def set_norm_for_param_grad_in_gpu(self, param):
        gd.debuginfo(prj="ds")

        # 获取参数的ID
        param_id = self.get_param_id(param)

        accumulated_grad = param.grad

        # 从梯度位置列表中获取相关信息
        [i, source_offset, dest_offset, num_elements] = self.grad_position[param_id]

        # 定义开始位置
        start = source_offset

        # 对累积梯度进行调整，首先将其视图（view）调整为1维，然后从start开始，获取num_elements个元素
        accumulated_grad = accumulated_grad.view(-1).narrow(0, start, num_elements)

        # 计算调整后的累积梯度的2范数（即欧几里得范数，或者叫做欧氏距离），并将其设置为对应参数梯度的范数
        self.norm_for_param_grads[param_id] = accumulated_grad.data.double().norm(2)

    # 从GPU异步复制梯度到FP32缓冲区， only call in copy_grads_in_partition
    def async_inplace_copy_grad_to_fp32_buffer_from_gpu(self, param):
        gd.debuginfo(prj="ds")
        # 根据参数获取其id
        param_id = self.get_param_id(param)

        # 从grad_position字典中获取参数的一些属性，包括所在的组i，源偏移量，目标偏移量，元素数量
        [i, source_offset, dest_offset, num_elements] = self.grad_position[param_id]

        # 获取fp32_groups的第i组的梯度，然后调整形状为一维，然后从dest_offset开始，获取num_elements个元素，得到目标张量
        dest_tensor = self.single_partition_of_fp32_groups[i].grad.view(-1).narrow(0, dest_offset, num_elements)

        # 从source_offset开始，获取num_elements个元素，得到源张量
        src_tensor = param.grad.view(-1).narrow(0, source_offset, num_elements)
        if not self.fp16_master_weights_and_gradients:
            gd.debuginfo(prj="ds")
            # 如果没有启用fp16主权重和梯度，则将源张量转换为float类型
            src_tensor = src_tensor.float()

        # 将源张量的内容复制到目标张量中，此操作为非阻塞操作
        dest_tensor.copy_(src_tensor, non_blocking=True)

        # 将参数的梯度设为None，此步骤为了减少GPU上的内存使用
        param.grad = None  #offload only

    # 完成CPU卸载的梯度范数计算  only call by scaled_global_norm
    def complete_grad_norm_calculation_for_cpu_offload(self, params):
        gd.debuginfo(prj="ds")
        total_norm = 0.0  # 初始化总梯度范数为0
        norm_type = 2.0  # 设置范数类型为2，即L2范数

        for p in params:  # 遍历所有的参数
            # Pipeline parallelism may replicate parameters. Avoid multi-counting.
            # 流水线并行可能会复制参数，我们需要避免重复计数
            if hasattr(p, PIPE_REPLICATED) and p.ds_pipe_replicated:
                continue

            # 如果参数是并行模型的参数或模型并行等级为0
            if is_model_parallel_parameter(p) or (self.model_parallel_rank == 0):
                param_id = self.get_param_id(p)  # 获取参数的ID

                # as some model have trainable parameters but skipped in training,
                # their backward hooks in self.create_reduce_and_remove_grad_hooks() will not run,
                # so they have no norm_for_param_grads
                # 一些模型可能有可训练的参数但在训练中被跳过，
                # 它们在self.create_reduce_and_remove_grad_hooks()中的反向钩子将不会运行，
                # 所以它们没有norm_for_param_grads

                # 如果参数ID在norm_for_param_grads中
                if param_id in self.norm_for_param_grads:
                    # 获取参数的范数
                    param_norm = self.norm_for_param_grads[param_id]

                    # 将参数的范数的平方累加到总范数
                    total_norm += param_norm.item()**2
                else:
                    # As unused parameters in modules may not be expected sometimes,
                    # add an explicit error msg when it occurred and an option to
                    # avoid the error
                    # 有时，模块中未使用的参数可能是预料之外的，
                    # 当出现这种情况时，添加一个明确的错误消息，并提供一个选项来避免该错误
                    assert self.ignore_unused_parameters, """
                        This assert indicates that your module has parameters that
                        were not used in producing loss.
                        You can avoid this assert by
                        (1) enable ignore_unused_parameters option in zero_optimization config;
                        (2) making sure all trainable parameters and `forward` function
                            outputs participate in calculating loss.
                    """

        # Sum across all model parallel GPUs. # 跨所有模型并行的GPU进行求和
        total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)])
        dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.SUM, group=self.dp_process_group)

        self._model_parallel_all_reduce(tensor=total_norm_cuda, op=dist.ReduceOp.SUM)

        # 计算总范数的开方，即L2范数
        total_norm = total_norm_cuda[0].item()**(1. / norm_type)

        # 如果总范数为无穷或者不是数字
        if total_norm == float('inf') or total_norm == -float('inf') or total_norm != total_norm:
            # 将总范数设置为-1
            total_norm = -1

        # 返回总范数
        return total_norm

    ############################################################################################
    # 在分区中复制梯度， only call reduce_ipg_grads
    def copy_grads_in_partition(self, param):
        gd.debuginfo(prj="ds")

        # 检查是否启用了CPU offload
        if self.cpu_offload:
            gd.debuginfo(prj="ds")
            # 如果梯度累积步数大于1
            if self.gradient_accumulation_steps > 1:
                gd.debuginfo(prj="ds")
                # 从GPU异步地累积梯度到CPU
                self.async_accumulate_grad_in_cpu_via_gpu(param)

            # 如果当前步骤是梯度累积的边界
            if self.is_gradient_accumulation_boundary:
                gd.debuginfo(prj="ds")
                # 在GPU中设置参数梯度的范数
                self.set_norm_for_param_grad_in_gpu(param)

                # 更新参数梯度的溢出跟踪器
                self.update_overflow_tracker_for_param_grad(param)

                # 从GPU异步地复制梯度到fp32缓冲区
                self.async_inplace_copy_grad_to_fp32_buffer_from_gpu(param)

            return
        #print(f"ID {self.get_param_id(param)} grad norm {param.grad.norm()}")

        # 如果分区中没有梯度
        if self.grads_in_partition is None:
            gd.debuginfo(prj="ds")
            self.grads_in_partition_offset = 0
            total_size = 0

            # 计算分区中所有参数的总元素数量
            for group in self.params_in_partition:
                for param_in_partition in group:
                    total_size += param_in_partition.numel()
            # 打印复制梯度前的内存使用情况  # f"复制{total_size}个梯度到分区前的内存使用情况"
            see_memory_usage(f"before copying {total_size} gradients into partition")
            self.grads_in_partition = torch.empty(int(total_size),
                                                  dtype=self.dtype,
                                                  device=get_accelerator().current_device_name())

            # 打印复制梯度后的内存使用情况 see_memory_usage(f"复制{total_size}个梯度到分区后的内存使用情况")
            see_memory_usage(f"after copying {total_size} gradients into partition")

        # The allreduce buffer will be rewritten. Copy the gradients in partition to a new buffer
        # allreduce缓冲区将被重写，将分区中的梯度复制到新的缓冲区
        new_grad_tensor = self.grads_in_partition.view(-1).narrow(0, self.grads_in_partition_offset, param.numel())

        # 使用原始的梯度更新新的梯度tensor
        new_grad_tensor.copy_(param.grad.view(-1))

        # 更新待进行聚合操作的梯度的数据
        param.grad.data = new_grad_tensor.data.view_as(param.grad)

        # 更新分区的梯度偏移量
        #print(f"Grad norm after copy to contiguous_buffer {param.grad.data.norm()}")
        self.grads_in_partition_offset += param.numel()

    # reduce-IPG梯度
    def reduce_ipg_grads(self):
        # 如果梯度是连续的
        if self.contiguous_gradients:
            gd.debuginfo(prj="ds")
            # 如果存在超大参数需要进行梯度汇总
            if self.extra_large_param_to_reduce is not None:
                gd.debuginfo(prj="ds")
                # 确保只有一个参数在 ipg bucket 中，否则会出现问题
                assert len(self.params_in_ipg_bucket) == 1, "more than 1 param in ipg bucket, this shouldn't happen"

                # 获取该参数的id
                _, _, param_id = self.params_in_ipg_bucket[0]

                # 确保 ipg bucket 中的参数和 extra-large 参数匹配
                assert self.get_param_id(self.extra_large_param_to_reduce
                                         ) == param_id, "param in ipg bucket does not match extra-large param"

                # 对梯度进行平均处理
                self.average_tensor(self.extra_large_param_to_reduce.grad.view(-1))

                # 清空 extra_large_param_to_reduce
                self.extra_large_param_to_reduce = None
            else:
                gd.debuginfo(prj="ds")
                # 对 ipg buffer 的梯度进行平均处理
                self.average_tensor(self.ipg_buffer[self.ipg_index])
        else:
            gd.debuginfo(prj="ds")
            # fallback 策略，对 grads_in_ipg_bucket 进行汇总
            self.buffered_reduce_fallback(None,
                                          self.grads_in_ipg_bucket,
                                          elements_per_buffer=self.elements_in_ipg_bucket)

        # 根据是否开启 overlap_comm 和 cpu_offload 选择合适的 stream
        if self.overlap_comm:
            gd.debuginfo(prj="ds")
            stream = self.reduction_stream
        elif self.cpu_offload:
            gd.debuginfo(prj="ds")
            #  注意：copy_grad_stream 被禁用了，因为它会和 reduce 产生冲突，这会影响性能，应该修复这个问题
            #  TODO: copy_grad_stream is disabled because of race with reduce. This hurts perf and should be fixed.
            #            get_accelerator().synchronize()
            #            stream = self.copy_grad_stream
            stream = get_accelerator().current_stream()
        else:
            gd.debuginfo(prj="ds")
            stream = get_accelerator().current_stream()

        # 在选定的 stream 中执行以下操作
        with get_accelerator().stream(stream):
            for _, param, param_id in self.params_in_ipg_bucket:
                # 确保该参数没有被汇总过，因为当前不支持多次梯度汇总
                assert self.params_already_reduced[param_id] == False, \
                    f"The parameter {param_id} has already been reduced. \
                    Gradient computed twice for this partition. \
                    Multiple gradient reduction is currently not supported"
                # 标记该参数已经被汇总
                self.params_already_reduced[param_id] = True

                # 如果需要对梯度进行分区
                if self.partition_gradients:
                    if not self.is_param_in_current_partition[param_id]:
                        if self.overlap_comm and self.contiguous_gradients is False:
                            # 在下一次梯度汇总过程中清空其他分区的梯度
                            # 这样可以避免在汇总完成之前就清空他们
                            # Clear grads of other partitions during the next reduction
                            # to avoid clearing them before the reduction is complete.
                            if self.previous_reduced_grads is None:
                                self.previous_reduced_grads = []
                            self.previous_reduced_grads.append(param)
                        else:
                            # 清空该参数的梯度属性
                            param.grad = None  #only if self.partition_gradients
                    elif self.contiguous_gradients:
                        # 如果梯度是连续的，复制当前分区的梯度
                        self.copy_grads_in_partition(param)
                else:
                    # zero stage 1 - 只分区优化器状态
                    # zero stage 1 - partition only optimizer state
                    if self.contiguous_gradients and self.is_param_in_current_partition[param_id]:
                        # 如果梯度是连续的，复制当前分区的梯度
                        self.copy_grads_in_partition(param)

        # 清空 ipg_bucket 和相关信息
        self.grads_in_ipg_bucket = []
        self.params_in_ipg_bucket = []
        self.ipg_bucket_has_moe_params = False
        self.elements_in_ipg_bucket = 0
        #####################################################################

    # 减少已准备好的分区并删除梯度
    def reduce_ready_partitions_and_remove_grads(self, param, i):
        gd.debuginfo(prj="ds")
        # 如果满足以下两个条件之一，执行操作：
        # 1. 需要对梯度进行分区处理
        # 2. 当前处于梯度累积的边界
        # 该操作包括：
        # 对独立的参数梯度分区桶进行归约，并移除梯度
        if self.partition_gradients or self.is_gradient_accumulation_boundary:
            gd.debuginfo(prj="ds")
            self.reduce_independent_p_g_buckets_and_remove_grads(param, i)

    # 将减少的梯度设置为零
    def zero_reduced_gradients(self, partition_id, i):
        gd.debuginfo(prj="ds")

        # 定义一个函数，用于检查与当前参数相关的所有分区是否已经完成了梯度的计算
        def are_all_related_partitions_reduced(params_id):
            # 遍历所有与当前参数相关的分区ID
            gd.debuginfo(prj="ds")
            for partition_id in self.param_to_partition_ids[i][params_id]:
                if not self.is_partition_reduced[i][partition_id]:
                    return False
            # 如果所有相关分区都完成了计算，就返回True
            return True

        # 遍历当前分区中所有参数的ID
        for params_id in self.is_grad_computed[i][partition_id]:
            # 如果与当前参数ID相关的所有分区都已经完成了梯度的计算
            if are_all_related_partitions_reduced(params_id):
                # 将当前参数的梯度设为None，这样可以节省存储空间
                self.param_dict[params_id].grad = None  # dead code

    # 压平并打印, 没有被使用
    def flatten_and_print(self, message, tensors, start=0, n=5):
        # 首先，我们将输入的多维张量（tensors）压缩为一维
        flatten_tensor = self.flatten(tensors)

        # 定义一个函数，负责打印压缩后的张量的一部分
        def print_func():
            # 这里我们调整了张量的维度以便进行打印，然后使用narrow方法取出要打印的部分
            # narrow函数的第一个参数是压缩的维度（这里是0，即第一维），第二个参数是开始的索引（start），第三个参数是长度（n）
            gd.debuginfo(prj="ds", info=f"T: flatten_tensor={infoTensor(flatten_tensor.contiguous().view(-1).narrow(0, start, n))}")

        # 最后在一个序列执行环境中执行这个打印函数，并传入一个消息（message）
        # 这个消息通常用于指示这次打印的含义，例如"打印压缩后的张量的前5个元素"
        self.sequential_execution(print_func, message)

    # 获取要reduce的梯度
    def get_grads_to_reduce(self, i, partition_id):
        gd.debuginfo(prj="ds")

        # 定义一个函数，用于获取可以reduce的梯度部分
        def get_reducible_portion(key):
            gd.debuginfo(prj="ds")
            # 从参数字典中获取梯度
            grad = self.param_dict[key].grad

            # 获取梯度的总元素数量
            total_elements = grad.numel()

            # 获取开始的偏移量
            start = self.grad_start_offset[i][partition_id][key]

            # 计算元素数量
            num_elements = min(total_elements - start,
                               self.partition_size[i] - self.grad_partition_insertion_offset[i][partition_id][key])

            # 如果不进行正确性测试
            if not pg_correctness_test:
                # 如果元素数量等于总元素数量，返回梯度
                if num_elements == total_elements:
                    gd.debuginfo(prj="ds")
                    return grad
                else:
                    gd.debuginfo(prj="ds")
                    # 否则，返回指定范围内的梯度
                    return grad.contiguous().view(-1).narrow(0, int(start), int(num_elements))
            else:
                # 如果进行正确性测试
                if num_elements == total_elements:
                    gd.debuginfo(prj="ds")
                    return grad.clone()
                else:
                    # 返回克隆并指定范围内的梯度
                    gd.debuginfo(prj="ds")
                    return grad.clone().contiguous().view(-1).narrow(0, int(start), int(num_elements))

        # 创建一个空列表，用于存储要redeuce的梯度
        grads_to_reduce = []

        # 遍历已计算梯度的键
        for key in self.is_grad_computed[i][partition_id]:
            # 获取可以reduce的梯度部分
            grad = get_reducible_portion(key)
            # 将梯度添加到列表中
            grads_to_reduce.append(grad)

        # 返回reduce的梯度列表
        return grads_to_reduce

    # 顺序执行
    def sequential_execution(self, function, message, group=None):
        # 如果没有指定分组，使用当前对象的dp_process_group作为默认分组
        if group is None:
            gd.debuginfo(prj="ds")
            group = self.dp_process_group

        # 如果当前进程的等级(rank)是0，记录日志信息
        if dist.get_rank(group=group) == 0:
            gd.debuginfo(prj="ds", info=message)

        # 遍历当前分组中的每个进程
        for id in range(dist.get_world_size(group=group)):
            # 如果当前进程的等级(rank)等于循环变量id，执行传入的函数
            if id == dist.get_rank(group=group):
                function()
            # 确保所有进程都执行到这一点后才能继续往下执行
            dist.barrier(group=group)

    # 将无梯度设置为零， 没有触发
    def set_none_gradients_to_zero(self, i, partition_id):
        gd.debuginfo(prj="ds")
        # 遍历在指定分区中的所有参数ID
        for param_id in self.is_grad_computed[i][partition_id]:
            # 从字典中获取该ID对应的参数对象
            param = self.param_dict[param_id]

            # 如果该参数的导数（梯度）为None
            if param.grad is None:
                # 则将其设置为与参数相同形状的零张量
                param.grad = torch.zero_like(param)

    ######################Reduction Related Methods##############################
    # allreduce操作，是基于桶的--only call by  allreduce_and_copy
    def allreduce_bucket(self, bucket, rank=None, log=None):
        gd.debuginfo(prj="ds")
        rank = None

        # 将bucket中的tensor扁平化
        tensor = self.flatten(bucket)

        tensor_to_allreduce = tensor

        if pg_correctness_test:
            gd.debuginfo(prj="ds")
            communication_data_type = torch.float32
        else:
            gd.debuginfo(prj="ds")
            communication_data_type = self.communication_data_type

        if communication_data_type != tensor.dtype:
            gd.debuginfo(prj="ds")
            tensor_to_allreduce = tensor.to(communication_data_type)

        tensor_to_allreduce.div_(dist.get_world_size(group=self.dp_process_group))

        if rank is None:
            gd.debuginfo(prj="ds")
            #    "All Reducing"
            dist.all_reduce(tensor_to_allreduce, group=self.dp_process_group)
        else:
            gd.debuginfo(prj="ds")
            global_rank = dist.get_global_rank(self.dp_process_group, rank)
            dist.reduce(tensor_to_allreduce, global_rank, group=self.dp_process_group)

        if communication_data_type != tensor.dtype and tensor is not tensor_to_allreduce:
            if rank is None or rank == dist.get_rank(group=self.dp_process_group):
                tensor.copy_(tensor_to_allreduce)

        return tensor

    def _clear_previous_reduced_grads(self):
        gd.debuginfo(prj="ds")
        if self.previous_reduced_grads is not None:
            for param in self.previous_reduced_grads:
                param.grad = None  # overlap enabled
            self.previous_reduced_grads = None

    # if rank is specified do a reduction instead of an allreduce
    def allreduce_and_copy(self, small_bucket, rank=None, log=None):
        if self.overlap_comm:
            gd.debuginfo(prj="ds")
            get_accelerator().synchronize()
            # It is safe to clear the previously reduced grads of other partitions
            self._clear_previous_reduced_grads()
            stream = self.reduction_stream
        else:
            gd.debuginfo(prj="ds")
            stream = get_accelerator().current_stream()

        with get_accelerator().stream(stream):
            allreduced = self.allreduce_bucket(small_bucket, rank=rank, log=log)
            if rank is None or rank == dist.get_rank(group=self.dp_process_group):
                for buf, synced in zip(small_bucket, self.unflatten(allreduced, small_bucket)):
                    buf.copy_(synced)

    def allreduce_no_retain(self, bucket, numel_per_bucket=500000000, rank=None, log=None):
        gd.debuginfo(prj="ds")
        small_bucket = []
        numel = 0
        for tensor in bucket:
            small_bucket.append(tensor)
            numel = numel + tensor.numel()
            if numel > numel_per_bucket:
                self.allreduce_and_copy(small_bucket, rank=rank, log=None)
                small_bucket = []

        if len(small_bucket) > 0:
            self.allreduce_and_copy(small_bucket, rank=rank, log=log)

    # allows using reduction of gradients instead of using all_reduce

    def buffered_reduce_fallback(self, rank, grads, elements_per_buffer=500000000, log=None):
        gd.debuginfo(prj="ds")
        split_buckets = split_half_float_double(grads)

        for i, bucket in enumerate(split_buckets):
            self.allreduce_no_retain(bucket, numel_per_bucket=elements_per_buffer, rank=rank, log=log)

    #############################################################################
    #############################################################################
    #############################################################################

    # views the tensor as multiple partitions and returns
    # those partitions
    def get_data_parallel_partitions(self, tensor, group_id):
        gd.debuginfo(prj="ds")
        partitions = []

        dp = dist.get_world_size(group=self.real_dp_process_group[group_id])
        # dp_id = dist.get_rank(group=self.real_dp_process_group[group_id])

        total_num_elements = tensor.numel()

        base_size = total_num_elements // dp
        remaining = total_num_elements % dp

        start = 0
        for id in range(dp):
            partition_size = base_size
            if id < remaining:
                partition_size = partition_size + 1
            partitions.append(tensor.narrow(0, start, partition_size))
            start = start + partition_size
        return partitions

    def get_partition_info(self, tensor_list, partition_size, partition_id):
        gd.debuginfo(prj="ds")
        params_in_partition = []
        params_not_in_partition = []

        start_index = partition_size * partition_id
        end_index = partition_size * (partition_id + 1)

        current_index = 0
        first_offset = 0

        for tensor in tensor_list:

            tensor_size = tensor.numel()

            if start_index <= current_index < end_index:
                params_in_partition.append(tensor)

            elif current_index < start_index < (current_index + tensor_size):
                params_in_partition.append(tensor)

                assert (first_offset == 0
                        ), "This can happen either zero or only once as this must be the first tensor in the partition"
                first_offset = start_index - current_index

            else:
                params_not_in_partition.append(tensor)

            current_index = current_index + tensor_size

        return params_in_partition, params_not_in_partition, first_offset

    def zero_grad(self, set_to_none=False):
        gd.debuginfo(prj="ds")
        """
        Zero FP16 parameter grads.
        """
        # FP32 grad should never exist.
        # For speed, set model fp16 grad to None by default
        for group in self.bit16_groups:
            for p in group:
                if set_to_none:
                    p.grad = None  # epilogue and in step
                else:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

    def _model_parallel_all_reduce(self, tensor, op):
        """ Perform all reduce within model parallel group, if any.
        """
        # gd.debuginfo(prj="ds")
        if self.model_parallel_group is None or self.model_parallel_world_size == 1:
            pass
        else:
            # gd.debuginfo(prj="ds")
            dist.all_reduce(tensor=tensor, op=op, group=self.model_parallel_group)

    def get_grad_norm_direct(self, gradients, params, norm_type=2):
        """Clips gradient norm of an iterable of parameters.

        This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
        added functionality to handle model parallel parameters. Note that
        the gradients are modified in place.

        Arguments:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.

        Returns:
            Total norm of the parameters (viewed as a single vector).
        """
        norm_type = float(norm_type)
        if norm_type == inf:
            total_norm = max(g.data.abs().max() for g in gradients)
            gd.debuginfo(prj="ds", info=f"1-total_norm={total_norm}")

            total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)])
            gd.debuginfo(prj="ds", info=f"1-total_norm_cuda={infoTensor(total_norm_cuda)}")

            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.MAX, group=self.dp_process_group)

            # Take max across all GPUs.
            self._model_parallel_all_reduce(tensor=total_norm_cuda, op=dist.ReduceOp.MAX)
            total_norm = total_norm_cuda[0].item()
            gd.debuginfo(prj="ds", info=f"2-total_norm={total_norm}")
        else:
            total_norm = 0.0
            if dist.get_rank() == 0:
               gd.debuginfo(prj="ds", info=f"Total Norm beginning {total_norm}")

            for g, p in zip(gradients, params):
                # Pipeline parallelism may replicate parameters. Avoid multi-counting.
                if hasattr(p, PIPE_REPLICATED) and p.ds_pipe_replicated:
                    continue
                if is_model_parallel_parameter(p) or (self.model_parallel_rank == 0):
                    param_norm = g.data.double().norm(2)
                    total_norm += param_norm.item()**2

            gd.debuginfo(prj="ds", info=f"2-total_norm={total_norm}")

            # Sum across all model parallel GPUs.
            total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)])
            gd.debuginfo(prj="ds", info=f"2-total_norm_cuda={infoTensor(total_norm_cuda)}")

            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.SUM, group=self.dp_process_group)

            self._model_parallel_all_reduce(tensor=total_norm_cuda, op=dist.ReduceOp.SUM)

            total_norm = total_norm_cuda[0].item()**(1. / norm_type)
            gd.debuginfo(prj="ds", info=f"5-total_norm={total_norm}")

        if total_norm == float('inf') or total_norm == -float('inf') or total_norm != total_norm:
            gd.debuginfo(prj="ds", info=f"total_norm set to -1")
            total_norm = -1

        return total_norm

    # creates a flat fused tensor from the tensor list starting at the first_offset
    # in the first tensor of the list. If there are not enough elements in the tensor
    # list then the flat tensor will be padded with zeros
    def get_flat_partition(self, tensor_list, first_offset, partition_size, dtype, device, return_tensor_list=False):
        gd.debuginfo(prj="ds")
        flat_tensor_list = []
        current_size = 0
        for i, tensor in enumerate(tensor_list):
            gd.debuginfo(prj="ds", info=f'tensor={infoTensor(tensor)}')
            if tensor.grad is None:
                tensor.grad = torch.zeros_like(tensor)
            tensor = tensor.grad
            num_elements = tensor.numel()
            tensor_offset = 0

            # we need to offset to get to the right element
            if i == 0 and first_offset > 0:
                tensor_offset = first_offset
                num_elements = num_elements - tensor_offset

            # we dont need all elements of the tensor
            if num_elements > (partition_size - current_size):
                num_elements = partition_size - current_size
            gd.debuginfo(prj="ds", info=f'tensor_offset={tensor_offset}, num_elements={num_elements}')
            # we need a narrow view of the tensor based on the tensor offset and number of elements that
            # we need from this tensor
            if tensor_offset > 0 or num_elements < tensor.numel():
                tmp = tensor.contiguous().view(-1).narrow(0, int(tensor_offset), int(num_elements))
                gd.debuginfo(prj="ds", info=f'tmp={infoTensor(tmp)}')
                flat_tensor_list.append(tmp)
            else:
                flat_tensor_list.append(tensor)

            current_size = current_size + num_elements

        # this means its the last partition and does not align with the dp boundary. We need to pad before flattening
        if current_size < partition_size:
            tmp = torch.zeros(int(partition_size - current_size), dtype=dtype, device=device)
            gd.debuginfo(prj="ds", info=f'tmp={infoTensor(tmp)}')
            flat_tensor_list.append(tmp)

        if return_tensor_list:
            gd.debuginfo(prj="ds")
            return flat_tensor_list

        return self.flatten(flat_tensor_list)

    def free_grad_in_param_list(self, param_list):
        gd.debuginfo(prj="ds")
        for p in param_list:
            p.grad = None  # in step

    def reset_cpu_buffers(self):
        gd.debuginfo(prj="ds")
        self.norm_for_param_grads = {}
        self.local_overflow = False

    def log_timers(self, timer_names):
        gd.debuginfo(prj="ds")
        if self.timers is None:
            gd.debuginfo(prj="ds")
            return

        self.timers.log(names=list(timer_names))

    def start_timers(self, timer_names):
        gd.debuginfo(prj="ds")
        if self.timers is None:
            gd.debuginfo(prj="ds")
            return

        for name in timer_names:
            self.timers(name).start()

    def stop_timers(self, timer_names):
        gd.debuginfo(prj="ds")
        if self.timers is None:
            gd.debuginfo(prj="ds")
            return

        for name in timer_names:
            self.timers(name).stop()

    def set_lr(self, lr):
        gd.debuginfo(prj="ds")
        """Set the learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        gd.debuginfo(prj="ds")
        """Return the current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def override_loss_scale(self, loss_scale):
        gd.debuginfo(prj="ds")
        if loss_scale != self.external_loss_scale:
            gd.debuginfo(prj="ds", info=f'[deepspeed] setting loss scale from {self.external_loss_scale} -> {loss_scale}')
        self.custom_loss_scaler = True
        self.external_loss_scale = loss_scale

    def scaled_global_norm(self, norm_type=2):
        gd.debuginfo(prj="ds")
        assert norm_type == 2, "only L2 norm supported"
        norm_groups = []
        for i, group in enumerate(self.bit16_groups):
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])
            if self.cpu_offload:
                norm_groups.append(self.complete_grad_norm_calculation_for_cpu_offload(self.params_in_partition[i]))
                single_grad_partition = self.single_partition_of_fp32_groups[i].grad
            else:
                norm_groups.append(self.get_grad_norm_direct(self.averaged_gradients[i], self.params_in_partition[i]))

        if self.has_moe_layers:
            gd.debuginfo(prj="ds")
            self._average_expert_grad_norms(norm_groups)

        # note that the get_global_norm function only supports l2 norm
        return get_global_norm(norm_list=norm_groups)

    def get_bit16_param_group(self, group_no):
        gd.debuginfo(prj="ds")
        bit16_partitions = self.parallel_partitioned_bit16_groups[group_no]
        partition_id = dist.get_rank(group=self.real_dp_process_group[group_no])
        return [bit16_partitions[dist.get_rank(group=self.real_dp_process_group[group_no])]]

    def _optimizer_step(self, group_no):
        gd.debuginfo(prj="ds")
        original_param_groups = self.optimizer.param_groups
        self.optimizer.param_groups = [original_param_groups[group_no]]
        # Disabling this as the C++ side copy & synchronize is not working correctly
        #from deepspeed.ops.adam import DeepSpeedCPUAdam
        #if type(self.optimizer) == DeepSpeedCPUAdam and self.dtype == torch.half:
        #    self.optimizer.step(fp16_param_groups=[self.get_bit16_param_group(group_no)])
        #else:
        #    self.optimizer.step()
        self.optimizer.step()
        self.optimizer.param_groups = original_param_groups

    def step(self, closure=None):
        gd.debuginfo(prj="ds")
        """
        Not supporting closure.
        """
        self.micro_step_id = -1

        see_memory_usage(f"In step before checking overflow")

        # First compute norm for all group so we know if there is overflow
        self.check_overflow()
        OPTIMIZER_ALLGATHER = 'optimizer_allgather'
        OPTIMIZER_GRADIENTS = 'optimizer_gradients'
        OPTIMIZER_STEP = 'optimizer_step'
        timer_names = [OPTIMIZER_ALLGATHER, OPTIMIZER_GRADIENTS, OPTIMIZER_STEP]

        prev_scale = self.loss_scale
        self._update_scale(self.overflow)
        if self.overflow:
            see_memory_usage('After overflow before clearing gradients')
            self.zero_grad(set_to_none=True)
            if self.cpu_offload:
                gd.debuginfo(prj="ds")
                self.reset_cpu_buffers()
            else:
                gd.debuginfo(prj="ds")
                self.averaged_gradients = {}

            see_memory_usage('After overflow after clearing gradients')

            self.start_timers(timer_names)
            self.stop_timers(timer_names)
            return

        # Step 1:- Calculate gradient norm using bit-16 grads
        see_memory_usage('Before norm calculation')
        scaled_global_grad_norm = self.scaled_global_norm()
        self._global_grad_norm = scaled_global_grad_norm / prev_scale
        see_memory_usage('After norm before optimizer')

        # Step 2:- run optimizer and upscaling simultaneously
        for i, group in enumerate(self.bit16_groups):
            self.start_timers([OPTIMIZER_GRADIENTS])
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])
            if self.cpu_offload:
                single_grad_partition = self.single_partition_of_fp32_groups[i].grad
                self.unscale_and_clip_grads([single_grad_partition], scaled_global_grad_norm)

                self.stop_timers([OPTIMIZER_GRADIENTS])
                self.start_timers([OPTIMIZER_STEP])
                self._optimizer_step(i)

                # Disabled, this is not currently working
                #from deepspeed.ops.adam import DeepSpeedCPUAdam
                #if not (type(self.optimizer) == DeepSpeedCPUAdam and self.dtype == torch.half):
                #    bit16_partitions = self.parallel_partitioned_bit16_groups[i]
                #    fp32_partition = self.single_partition_of_fp32_groups[i]
                #    bit16_partitions[partition_id].data.copy_(fp32_partition.data)
                bit16_partitions = self.parallel_partitioned_bit16_groups[i]
                fp32_partition = self.single_partition_of_fp32_groups[i]
                bit16_partitions[partition_id].data.copy_(fp32_partition.data)

                self.stop_timers([OPTIMIZER_STEP])
            else:
                # free gradients for all the parameters that are not updated by this process(ZeRO stage2)
                self.free_grad_in_param_list(self.params_not_in_partition[i])

                # create a flat gradients for parameters updated by this process
                # If we are last partition, ensure we have same size grads and partition size, if not pad with zero tensors
                if partition_id == dist.get_world_size(group=self.real_dp_process_group[i]) - 1:
                    single_grad_partition = self.flatten_dense_tensors_aligned(
                        self.averaged_gradients[i],
                        int(self.partition_size[i])).to(self.single_partition_of_fp32_groups[i].dtype)
                else:
                    single_grad_partition = self.flatten(self.averaged_gradients[i]).to(
                        self.single_partition_of_fp32_groups[i].dtype)
                assert single_grad_partition.numel() == self.partition_size[i], \
                    "averaged gradients have different number of elements that partition size {} {} {} {}".format(
                        single_grad_partition.numel(), self.partition_size[i], i, partition_id)

                self.single_partition_of_fp32_groups[i].grad = single_grad_partition
                # release all the gradient since we have already created a necessary copy in dp_grad_partition(ZeRO stage2)
                self.free_grad_in_param_list(self.params_in_partition[i])

                self.averaged_gradients[i] = None

                self.unscale_and_clip_grads([single_grad_partition], scaled_global_grad_norm)

                self.stop_timers([OPTIMIZER_GRADIENTS])

                # Step 3:- run the optimizer if no offloading
                self.start_timers([OPTIMIZER_STEP])
                self._optimizer_step(i)
                # Step 4:- get rid of the fp32 gradients. Not needed anymore
                self.single_partition_of_fp32_groups[i].grad = None
                del single_grad_partition
                bit16_partitions = self.parallel_partitioned_bit16_groups[i]
                fp32_partition = self.single_partition_of_fp32_groups[i]
                bit16_partitions[partition_id].data.copy_(fp32_partition.data)
                self.stop_timers([OPTIMIZER_STEP])

        see_memory_usage('After optimizer before all-gather')
        if self.cpu_offload:
            gd.debuginfo(prj="ds")
            self.reset_cpu_buffers()

        self.start_timers([OPTIMIZER_ALLGATHER])
        # Gather the updated weights from everyone.
        # Then all partitions of the model parameters are updated and ready for next round forward.
        all_gather_dp_groups(partitioned_param_groups=self.parallel_partitioned_bit16_groups,
                             dp_process_group=self.real_dp_process_group,
                             start_alignment_factor=self.nccl_start_alignment_factor,
                             allgather_bucket_size=self.allgather_bucket_size)

        self.stop_timers([OPTIMIZER_ALLGATHER])

        # TODO: we probably don't need this? just to be safe
        for i in range(len(self.bit16_groups)):
            self._update_model_bit16_weights(i)

        self.log_timers(timer_names)
        see_memory_usage('After zero_optimizer step')

        return

    @torch.no_grad()
    def update_lp_params(self):
        gd.debuginfo(prj="ds")
        for i, (bit16_partitions, fp32_partition) in enumerate(
                zip(self.parallel_partitioned_bit16_groups, self.single_partition_of_fp32_groups)):
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])
            bit16_partitions[partition_id].data.copy_(fp32_partition.data)
            # print_rank_0(f'update_lp_params {i=} {partition_id=}', force=True)
            # if i == 0:
            #     print_rank_0(f'{fp32_partition[:10]=}', force=True)

        all_gather_dp_groups(partitioned_param_groups=self.parallel_partitioned_bit16_groups,
                             dp_process_group=self.real_dp_process_group,
                             start_alignment_factor=self.nccl_start_alignment_factor,
                             allgather_bucket_size=self.allgather_bucket_size)

    def _average_expert_grad_norms(self, norm_groups):
        gd.debuginfo(prj="ds")
        for i, norm in enumerate(norm_groups):
            if self.is_moe_param_group[i]:
                scaled_norm = norm * 1.0 / float(dist.get_world_size(group=self.real_dp_process_group[i]))
                scaled_norm_tensor = torch.tensor(scaled_norm,
                                                  device=get_accelerator().device_name(),
                                                  dtype=torch.float)
                dist.all_reduce(scaled_norm_tensor, group=self.real_dp_process_group[i])
                norm_groups[i] = scaled_norm_tensor.item()

    def unscale_and_clip_grads(self, grad_groups_flat, total_norm):
        gd.debuginfo(prj="ds")
        # compute combined scale factor for this group
        combined_scale = self.loss_scale
        if self.clip_grad > 0.:
            gd.debuginfo(prj="ds")
            # norm is in fact norm*scale
            clip = ((total_norm / self.loss_scale) + 1e-6) / self.clip_grad
            if clip > 1:
                gd.debuginfo(prj="ds")
                combined_scale = clip * self.loss_scale

        for grad in grad_groups_flat:
            if isinstance(grad, list):
                sub_partitions = grad
                for g in sub_partitions:
                    g.data.mul_(1. / combined_scale)
            else:
                grad.data.mul_(1. / combined_scale)

    def _check_overflow(self, partition_gradients=True):
        gd.debuginfo(prj="ds")
        self.overflow = self.has_overflow(partition_gradients)

    # `params` is a list / generator of torch.Variable
    def has_overflow_serial(self, params, is_grad_list=False):
        gd.debuginfo(prj="ds")
        for p in params:
            if p.grad is not None and self._has_inf_or_nan(p.grad.data):
                return True

        return False

    def has_overflow_partitioned_grads_serial(self):
        gd.debuginfo(prj="ds")
        for i in range(len(self.bit16_groups)):
            for j, grad in enumerate(self.averaged_gradients[i]):
                if grad is not None and self._has_inf_or_nan(grad.data, j):
                    return True
        return False

    def has_overflow(self, partition_gradients=True):
        if partition_gradients:
            gd.debuginfo(prj="ds")
            overflow = self.local_overflow if self.cpu_offload else self.has_overflow_partitioned_grads_serial()
            overflow_gpu = get_accelerator().ByteTensor([overflow])
            '''This will capture overflow across all data parallel and expert parallel process
            Since expert parallel process are a subset of data parallel process'''
            dist.all_reduce(overflow_gpu, op=dist.ReduceOp.MAX, group=self.dp_process_group)

        else:
            gd.debuginfo(prj="ds")
            params = []
            for group in self.bit16_groups:
                for param in group:
                    params.append(param)

            overflow = self.has_overflow_serial(params, is_grad_list=partition_gradients)
            overflow_gpu = get_accelerator().ByteTensor([overflow])

        # Since each model parallel GPU carries only part of the model,
        # make sure overflow flag is synced across all the model parallel GPUs
        self._model_parallel_all_reduce(tensor=overflow_gpu, op=dist.ReduceOp.MAX)

        overflow = overflow_gpu[0].item()
        return bool(overflow)

    # `x` is a torch.Tensor
    @staticmethod
    def _has_inf_or_nan(x, j=None):
        try:
            gd.debuginfo(prj="ds")
            # if x is half, the .float() incurs an additional deep copy, but it's necessary if
            # Pytorch's .sum() creates a one-element tensor of the same type as x
            # (which is true for some recent version of pytorch).
            cpu_sum = float(x.float().sum())
            # More efficient version that can be used if .sum() returns a Python scalar
            # cpu_sum = float(x.sum())
        except RuntimeError as instance:
            gd.debuginfo(prj="ds")
            # We want to check if inst is actually an overflow exception.
            # RuntimeError could come from a different error.
            # If so, we still want the exception to propagate.
            if "value cannot be converted" not in instance.args[0]:
                raise
            return True
        else:
            gd.debuginfo(prj="ds")
            if cpu_sum == float('inf') or cpu_sum == -float('inf') or cpu_sum != cpu_sum:
                gd.debuginfo(prj="ds")
                return True
            return False

    def backward(self, loss, retain_graph=False):
        gd.debuginfo(prj="ds")
        """
        :attr:`backward` performs the following steps:

        1. fp32_loss = loss.float()
        2. scaled_loss = fp32_loss*loss_scale
        3. scaled_loss.backward(), which accumulates scaled gradients into the ``.grad`` attributes of the model's fp16 leaves
        """
        self.micro_step_id += 1

        if self.contiguous_gradients:
            self.ipg_buffer = []
            buf_0 = torch.empty(int(self.reduce_bucket_size),
                                dtype=self.dtype,
                                device=get_accelerator().current_device_name())
            self.ipg_buffer.append(buf_0)

            # Use double buffers to avoid data access conflict when overlap_comm is enabled.
            if self.overlap_comm:
                buf_1 = torch.empty(int(self.reduce_bucket_size),
                                    dtype=self.dtype,
                                    device=get_accelerator().current_device_name())
                self.ipg_buffer.append(buf_1)
            self.ipg_index = 0

        if self.custom_loss_scaler:
            gd.debuginfo(prj="ds")
            scaled_loss = self.external_loss_scale * loss
            scaled_loss.backward()
        else:
            gd.debuginfo(prj="ds")
            self.loss_scaler.backward(loss.float(), retain_graph=retain_graph)

    def check_overflow(self, partition_gradients=True):
        gd.debuginfo(prj="ds")
        self._check_overflow(partition_gradients)

    def _update_scale(self, has_overflow=False):
        gd.debuginfo(prj="ds")
        self.loss_scaler.update_scale(has_overflow)

    # Promote state so it can be retrieved or set via "fp16_optimizer_instance.state"
    def _get_state(self):
        gd.debuginfo(prj="ds")
        return self.optimizer.state

    def _set_state(self, value):
        gd.debuginfo(prj="ds")
        self.optimizer.state = value

    state = property(_get_state, _set_state)

    # Promote param_groups so it can be retrieved or set via "fp16_optimizer_instance.param_groups"
    # (for example, to adjust the learning rate)
    def _get_param_groups(self):
        gd.debuginfo(prj="ds")
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        gd.debuginfo(prj="ds")
        self.optimizer.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)

    # Promote loss scale so it can be retrieved or set via "fp16_optimizer_instance.loss_scale"
    def _get_loss_scale(self):
        gd.debuginfo(prj="ds")
        if self.custom_loss_scaler:
            gd.debuginfo(prj="ds")
            return self.external_loss_scale
        else:
            gd.debuginfo(prj="ds")
            return self.loss_scaler.cur_scale

    def _set_loss_scale(self, value):
        gd.debuginfo(prj="ds")
        self.loss_scaler.cur_scale = value

    loss_scale = property(_get_loss_scale, _set_loss_scale)
    cur_scale = property(_get_loss_scale, _set_loss_scale)

    # Return group tensor after removing paddings that are added for alignment to DP world size.
    # This method works on the assumption that each group contains a single flattened tensor.
    def _get_groups_without_padding(self, groups_with_padding):
        gd.debuginfo(prj="ds")
        groups_without_padding = []
        for i, group in enumerate(groups_with_padding):
            lean_length = group.numel() - self.groups_padding[i]
            groups_without_padding.append(group[:lean_length])

        return groups_without_padding

    # Return optimizer state after removing paddings that are added for alignment.
    def _get_state_without_padding(self, state_with_padding, padding):
        gd.debuginfo(prj="ds")
        lean_state = {}
        for key, value in state_with_padding.items():
            if torch.is_tensor(value):
                lean_length = value.numel() - padding
                lean_state[key] = value[:lean_length]
            else:
                lean_state[key] = value

        return lean_state

    # Return base optimizer states.
    # This method assumes that each param group contains a single flattened tensor.
    def _get_base_optimizer_state(self):
        gd.debuginfo(prj="ds")
        optimizer_groups_state = []
        for i, group in enumerate(self.optimizer.param_groups):
            p = group['params'][0]
            lean_optimizer_state = self._get_state_without_padding(self.optimizer.state[p], self.groups_padding[i])
            optimizer_groups_state.append(lean_optimizer_state)

        return optimizer_groups_state

    def state_dict(self):
        """
        Returns a dict containing the current state of this :class:`FP16_Optimizer` instance.
        This dict contains attributes of :class:`FP16_Optimizer`, as well as the state_dict
        of the contained Pytorch optimizer.
        Example::
            checkpoint = {}
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            torch.save(checkpoint, "saved.pth")
        """
        state_dict = {}
        state_dict['loss_scaler'] = self.loss_scaler
        state_dict['dynamic_loss_scale'] = self.dynamic_loss_scale
        state_dict['overflow'] = self.overflow
        state_dict[CLIP_GRAD] = self.clip_grad

        if self.elastic_checkpoint:
            gd.debuginfo(prj="ds")
            state_dict[BASE_OPTIMIZER_STATE] = self._get_base_optimizer_state()
        else:
            gd.debuginfo(prj="ds")
            state_dict[BASE_OPTIMIZER_STATE] = self.optimizer.state_dict()

        # Remove paddings for DP alignment to enable loading for other alignment values
        fp32_groups_without_padding = self._get_groups_without_padding(self.single_partition_of_fp32_groups)
        state_dict[SINGLE_PARTITION_OF_FP32_GROUPS] = fp32_groups_without_padding

        state_dict[
            ZERO_STAGE] = ZeroStageEnum.gradients if self.partition_gradients else ZeroStageEnum.optimizer_states
        state_dict[GROUP_PADDINGS] = self.groups_padding
        state_dict[PARTITION_COUNT] = self.partition_count

        state_dict[DS_VERSION] = version
        state_dict[PARAM_SLICE_MAPPINGS] = self._param_slice_mappings

        return state_dict

    # Restore base optimizer fp32 weights from elastic checkpoint by:
    # 1) Merging fp32 weights from checkpoints of all partitions
    # 2) Extracting fp32 weights for current partition from merged weights
    # 3) Using extracted weights to update base optimizer weights directly.
    def _restore_from_elastic_fp32_weights(self, all_state_dict):
        gd.debuginfo(prj="ds")
        merged_single_partition_of_fp32_groups = []

        for i in range(len(self.single_partition_of_fp32_groups)):
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])
            merged_partitions = [sd[SINGLE_PARTITION_OF_FP32_GROUPS][i] for sd in all_state_dict]
            if self.is_moe_group(self.optimizer.param_groups[i]):
                ranks = self.get_ep_ranks(group_name=self.optimizer.param_groups[i]['name'])
                merged_partitions = [merged_partitions[i] for i in ranks]
            flat_merged_partitions = self.flatten_dense_tensors_aligned(
                merged_partitions,
                self.nccl_start_alignment_factor * dist.get_world_size(group=self.real_dp_process_group[i]))
            dp_partitions = self.get_data_parallel_partitions(flat_merged_partitions, i)
            merged_single_partition_of_fp32_groups.append(dp_partitions[partition_id])

        for current, saved in zip(self.single_partition_of_fp32_groups, merged_single_partition_of_fp32_groups):
            current.data.copy_(saved.data)

    # Restore base optimizer fp32 weights from ZeRO fp16 or bfloat16 weights
    def _restore_from_bit16_weights(self):
        gd.debuginfo(prj="ds")
        for group_id, (bit16_partitions, fp32_partition) in enumerate(
                zip(self.parallel_partitioned_bit16_groups, self.single_partition_of_fp32_groups)):
            partition_id = dist.get_rank(group=self.real_dp_process_group[group_id])
            fp32_partition.data.copy_(bit16_partitions[partition_id].data)

    # Refresh the fp32 master params from the fp16 or bfloat16 copies.
    def refresh_fp32_params(self):
        gd.debuginfo(prj="ds")
        self._restore_from_bit16_weights()

    # Extract optimizer state for current partition from merged states of all partitions
    def _partition_base_optimizer_state(self, state_key, all_partition_states, group_id):
        partition_id = dist.get_rank(group=self.real_dp_process_group[group_id])
        alignment = dist.get_world_size(group=self.real_dp_process_group[group_id])
        if torch.is_tensor(all_partition_states[0]):
            gd.debuginfo(prj="ds")
            flat_merged_partitions = self.flatten_dense_tensors_aligned(all_partition_states, alignment)
            dp_partitions = self.get_data_parallel_partitions(flat_merged_partitions, group_id)
            return dp_partitions[partition_id]
        else:
            gd.debuginfo(prj="ds")
            # Assume non-tensor states are not partitioned and equal across ranks, so return first one
            return all_partition_states[0]

    def _restore_base_optimizer_state(self, base_optimizer_group_states):
        gd.debuginfo(prj="ds")
        if type(base_optimizer_group_states) == dict:
            gd.debuginfo(prj="ds")
            base_optimizer_group_states = base_optimizer_group_states['state']
        for i, group in enumerate(self.optimizer.param_groups):
            p = group['params'][0]
            for key, saved in base_optimizer_group_states[i].items():
                if torch.is_tensor(self.optimizer.state[p][key]):
                    dst_tensor = self.optimizer.state[p][key]
                    src_tensor = _get_padded_tensor(saved, dst_tensor.numel())
                    self.optimizer.state[p][key].data.copy_(src_tensor.data)
                else:
                    self.optimizer.state[p][key] = saved

    def get_ep_ranks(self, rank=0, group_name=None):
        gd.debuginfo(prj="ds")
        from deepspeed.utils import groups
        expert_parallel_size_ = groups._get_expert_parallel_world_size(group_name)
        world_size = groups._get_data_parallel_world_size()
        rank = groups._get_expert_parallel_rank(group_name)
        ranks = range(rank, world_size, expert_parallel_size_)
        return list(ranks)

    # Restore base optimizer state from elastic checkpoint by
    # 1) Merging optimizer state from checkpoints of all partitions
    # 2) Extracting optimizer state for current partition from the merged state
    # 3) Using the extracted value to directly update the base optimizer.
    def _restore_elastic_base_optimizer_state(self, all_state_dict):
        gd.debuginfo(prj="ds")
        base_optimizer_group_states = []
        for i in range(len(self.optimizer.param_groups)):
            partition_states = {}
            all_partition_group_states = [sd[BASE_OPTIMIZER_STATE][i] for sd in all_state_dict]

            if self.is_moe_group(self.optimizer.param_groups[i]):
                ranks = self.get_ep_ranks(group_name=self.optimizer.param_groups[i]['name'])
                all_partition_group_states = [all_partition_group_states[i] for i in ranks]

            for key in all_partition_group_states[0].keys():
                all_partition_states = [all_states[key] for all_states in all_partition_group_states]
                partition_states[key] = self._partition_base_optimizer_state(key, all_partition_states, i)
            base_optimizer_group_states.append(partition_states)

        self._restore_base_optimizer_state(base_optimizer_group_states)

    def load_state_dict(self,
                        state_dict_list,
                        load_optimizer_states=True,
                        load_from_fp32_weights=False,
                        checkpoint_folder=None):
        if checkpoint_folder:
            gd.debuginfo(prj="ds")
            self._load_universal_checkpoint(checkpoint_folder, load_optimizer_states, load_from_fp32_weights)
        else:
            gd.debuginfo(prj="ds")
            self._load_legacy_checkpoint(state_dict_list, load_optimizer_states, load_from_fp32_weights)

    # 定义一个方法来加载通用检查点 用于加载通用的模型检查点
    def _load_universal_checkpoint(self, checkpoint_folder, load_optimizer_states, load_from_fp32_weights):
        # 从检查点文件夹中加载超参数检查点状态
        self._load_hp_checkpoint_state(checkpoint_folder)

    @property
    def param_groups(self):
        gd.debuginfo(prj="ds")
        """Forward the wrapped optimizer's parameters."""
        return self.optimizer.param_groups

    def _load_hp_checkpoint_state(self, checkpoint_dir):
        gd.debuginfo(prj="ds")
        checkpoint_dir = os.path.join(checkpoint_dir, "zero")
        tp_rank = bwc_tensor_model_parallel_rank(mpu=self.mpu)
        tp_world_size = self.mpu.get_slice_parallel_world_size()

        for i, _ in enumerate(self.optimizer.param_groups):
            for lp in self.bit16_groups[i]:
                if lp._hp_mapping is not None:
                    #print(f"Loading {self.param_names[lp]} {tp_rank=} {tp_world_size=}")
                    lp.load_hp_checkpoint_state(os.path.join(checkpoint_dir, self.param_names[lp]), tp_rank,
                                                tp_world_size)

    def _load_legacy_checkpoint(self, state_dict_list, load_optimizer_states=True, load_from_fp32_weights=False):
        gd.debuginfo(prj="ds")
        r"""Loading ZeRO checkpoint

        Arguments:
            state_dict_list: List of all saved ZeRO checkpoints, one for each saved partition.
                Note that the number of saved partitions may differ from number of loading partitions to support
                changing GPU count, specifically DP world size, between saving and loading checkpoints.
            load_optimizer_states: Boolean indicating whether or not to load base optimizer states
            load_from_fp32_weights: Boolean indicating whether to initialize fp32 master weights from fp32
            copies in checkpoints (no precision loss) or from model's fp16 copies (with precision loss).
        """
        """
        Loads a state_dict created by an earlier call to state_dict().
        If ``fp16_optimizer_instance`` was constructed from some ``init_optimizer``,
        whose parameters in turn came from ``model``, it is expected that the user
        will call ``model.load_state_dict()`` before
        ``fp16_optimizer_instance.load_state_dict()`` is called.
        Example::
            model = torch.nn.Linear(D_in, D_out).to(get_accelerator().device_name()).half()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            optimizer = FP16_Optimizer(optimizer, static_loss_scale = 128.0)
            ...
            checkpoint = torch.load("saved.pth")
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        """

        # I think it should actually be ok to reload the optimizer before the model.
        dp_rank = dist.get_rank(group=self.dp_process_group)
        current_rank_sd = state_dict_list[dp_rank]
        self.loss_scaler = current_rank_sd.get('loss_scaler', self.loss_scaler)
        self.dynamic_loss_scale = current_rank_sd.get('dynamic_loss_scale', self.dynamic_loss_scale)
        self.overflow = current_rank_sd.get('overflow', self.overflow)
        self.clip_grad = current_rank_sd.get(CLIP_GRAD, self.clip_grad)

        ckpt_version = current_rank_sd.get(DS_VERSION, False)
        assert ckpt_version, f"Empty ds_version in checkpoint, not clear how to proceed"
        ckpt_version = pkg_version.parse(ckpt_version)

        # zero stage 1 mode
        if not self.partition_gradients:
            required_version = pkg_version.parse("0.3.17")
            error_str = f"ZeRO stage 1 changed in {required_version} and is not backwards compatible " \
                "with older stage 1 checkpoints. If you'd like to load an old ZeRO-1 checkpoint " \
                "please use an older version of DeepSpeed (<= 0.5.8) and set 'legacy_stage1': true in your zero config json."
            assert required_version <= ckpt_version, f"Old version: {ckpt_version} {error_str}"

        ckpt_is_rigid = isinstance(current_rank_sd[BASE_OPTIMIZER_STATE], dict)

        # padding is always at the last rank/partition
        # if DP=1024 and param-group elems=16 -> padding will be 1024-16 across all but one rank
        # scenario-1 (shrink): saving w. 4 gpus -> loading w. 2 gpus
        # scenario-2 (expand): saving w. 2 gpus -> loading w. 4 gpus
        # if load_optimizer_states:
        #     if new_dp_size:
        #         self.strip_padding()
        #         self.add_padding_w_new_dp_size()
        #     self.optimizer.load_state_dict(current_rank_sd[BASE_OPTIMIZER_STATE])

        if load_optimizer_states:
            if ckpt_is_rigid:
                gd.debuginfo(prj="ds")
                # loading rigid ckpt into either rigid or elastic exec
                self.optimizer.load_state_dict(current_rank_sd[BASE_OPTIMIZER_STATE])
            else:
                if self.elastic_checkpoint:
                    gd.debuginfo(prj="ds")
                    # loading elastic into elastic exec
                    self._restore_elastic_base_optimizer_state(state_dict_list)
                else:
                    gd.debuginfo(prj="ds")
                    # loading an elastic checkpoint into rigid exec
                    self._restore_base_optimizer_state(current_rank_sd[BASE_OPTIMIZER_STATE])

        # At this point, the optimizer's references to the model's fp32 parameters are up to date.
        # The optimizer's hyperparameters and internal buffers are also up to date.
        # However, the fp32 master copies of the model's fp16 params stored by the optimizer are still
        # out of date.  There are two options.
        # 1:  Refresh the master params from the model's fp16 params.
        # This requires less storage but incurs precision loss.
        # 2:  Save and restore the fp32 master copies separately.
        # We choose option 1 if changing DP degree and option 2 otherwise.
        #
        # Pytorch Optimizer.load_state_dict casts saved buffers (e.g. momentum) to the type and device
        # of their associated parameters, because it's possible those buffers might not exist yet in
        # the current optimizer instance.  In our case, as long as the current FP16_Optimizer has been
        # constructed in the same way as the one whose state_dict we are loading, the same master params
        # are guaranteed to exist, so we can just copy_() from the saved master params.

        if load_from_fp32_weights:
            # option 2 from above
            if self.elastic_checkpoint and not ckpt_is_rigid:
                gd.debuginfo(prj="ds")
                self._restore_from_elastic_fp32_weights(state_dict_list)
            else:
                gd.debuginfo(prj="ds")
                # For non-elastic checkpoint, simply copying from saved weights of current rank is sufficient.
                for current, saved in zip(self.single_partition_of_fp32_groups,
                                          current_rank_sd[SINGLE_PARTITION_OF_FP32_GROUPS]):
                    src_tensor = _get_padded_tensor(saved, current.numel())
                    current.data.copy_(src_tensor.data)
        else:
            gd.debuginfo(prj="ds")
            # option 1 from above
            self._restore_from_bit16_weights()

        if load_optimizer_states:
            gd.debuginfo(prj="ds")
            self._link_all_hp_params()


def _handle_overflow(cpu_sum, x, i):
    gd.debuginfo(prj="ds")
    import math
    rank = dist.get_rank()
    if rank == 0:
        t_i = -1
        for v_i, v in enumerate(x.data.contiguous().view(-1)):
            if not math.isfinite(float(v)):
                t_i = v_i
                break
        gd.debuginfo(prj="ds", info=f"rank {rank} detected overflow {cpu_sum} in tensor {i}:{t_i} shape {x.shape}")


def estimate_zero2_model_states_mem_needs(total_params,
                                          num_gpus_per_node=1,
                                          num_nodes=1,
                                          cpu_offload=True,
                                          additional_buffer_factor=1.5):

    total_gpus = num_nodes * num_gpus_per_node

    if cpu_offload:
        gd.debuginfo(prj="ds")
        gpu_mem = 2 * total_params
        cpu_mem = total_params * max(4 * total_gpus, 16) * additional_buffer_factor
    else:
        gd.debuginfo(prj="ds")
        gpu_mem = 4 * total_params + int(16 * total_params / total_gpus)
        cpu_mem = total_params * 4 * num_gpus_per_node * additional_buffer_factor

    return int(cpu_mem), int(gpu_mem)

def model_to_params(model):
    gd.debuginfo(prj="ds")
    # shared params calculated only once
    total_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
    return total_params


def estimate_zero2_model_states_mem_needs_all_live(model,
                                                   num_gpus_per_node=1,
                                                   num_nodes=1,
                                                   additional_buffer_factor=1.5):
    """
    Print out estimates on memory usage requirements for ZeRO 2 params, optim states and gradients
    for a given ``model`` and hardware setup.

    If you have an actual model object, use this function and everything will be derived
    automatically.

    If it's a hypothetical model, use ``estimate_zero2_model_states_mem_needs_all_cold`` where you have to pass
    the ``total_params`` explicitly.

    Args:
        - ``model``: ``nn.Module`` object
        - ``num_gpus_per_node``: how many gpus per node (defaults to 1)
        - ``num_nodes``: how many nodes (defaults to 1),
        - ``additional_buffer_factor``: estimation factor (defaults to 1.5):

    """
    gd.debuginfo(prj="ds")

    total_params = model_to_params(model)

    estimate_zero2_model_states_mem_needs_all_cold(total_params=total_params,
                                                   num_gpus_per_node=num_gpus_per_node,
                                                   num_nodes=num_nodes,
                                                   additional_buffer_factor=additional_buffer_factor)


def estimate_zero2_model_states_mem_needs_all_cold(total_params,
                                                   num_gpus_per_node=1,
                                                   num_nodes=1,
                                                   additional_buffer_factor=1.5):
    """
    Print out estimates on memory usage requirements for ZeRO 2 params, optim states and gradients
    for a given ``model`` and hardware setup.

    If it's a hypothetical model, use this function where you have to pass
    the ``total_params`` and ``largest_layer_params`` explicitly.

    If you have an actual model object, use ``estimate_zero2_model_states_mem_needs_all_live`` and everything
    will be derived automatically.

    Args:
        - ``total_params``: total  model params
        - ``num_gpus_per_node``: how many gpus per node (defaults to 1)
        - ``num_nodes``: how many nodes (defaults to 1),
        - ``additional_buffer_factor``: estimation factor (defaults to 1.5):

    """

    def format_options(cpu_offload):
        gd.debuginfo(prj="ds")
        enabled = []
        device = f'{OffloadDeviceEnum.cpu:4}' if cpu_offload else "none"
        enabled.append(f"offload_optimizer={device}")
        return ", ".join(enabled)

    nodes_str = "nodes" if num_nodes > 1 else "node"
    gpus_str = "GPUs" if num_gpus_per_node > 1 else "GPU"
    print("Estimated memory needed for params, optim states and gradients for a:\n"
          f"HW: Setup with {num_nodes} {nodes_str}, {num_gpus_per_node} {gpus_str} per node.\n"
          f"SW: Model with {int(total_params/1e6)}M total params.")
    print("  per CPU  |  per GPU |   Options")
    for cpu_offload in [True, False]:
        cpu_mem, gpu_mem = estimate_zero2_model_states_mem_needs(total_params=total_params,
                                                                 num_gpus_per_node=num_gpus_per_node,
                                                                 num_nodes=num_nodes,
                                                                 cpu_offload=cpu_offload,
                                                                 additional_buffer_factor=additional_buffer_factor)

        options_str = format_options(cpu_offload=cpu_offload)
        print(f" {cpu_mem/2**30:7.2f}GB | {gpu_mem/2**30:6.2f}GB | {options_str}")
