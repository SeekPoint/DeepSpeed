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

# 就是把list中所有tensor放到cpu上去
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

        gd.debuginfo(prj='ds', info=f'C:{self.__class__.__name__} FUNC_IN')
        # gd.debuginfo(prj='ds', info=f'init_optimizer={init_optimizer.param_groups}')
        for gi, g in enumerate(init_optimizer.param_groups):
            for k, v in g.items():
                if isinstance(v, list):
                    gd.debuginfo(prj='ds', info=f'init_optimizer[{gi}][{k}]= len of items={len(v)}')
                    for index, val in enumerate(v):
                        gd.debuginfo(prj='ds', info=f'init_optimizer[{gi}][{k}][{index}]={infoTensor(val)}')
                else:
                    gd.debuginfo(prj='ds', info=f'init_optimizer[{gi}][{k}]={str(v)}')
        # 2卡3卡都一样， TBD 记录到文件

        # 如果当前是主节点，打印一些设置的日志信息
        if dist.get_rank() == 0:
            gd.debuginfo(prj='ds', info=f"Reduce bucket size {reduce_bucket_size}") # 500,000,000
            gd.debuginfo(prj='ds', info=f"Allgather bucket size {allgather_bucket_size}") # 500,000,000
            gd.debuginfo(prj='ds', info=f"CPU Offload: {cpu_offload}") # False
            gd.debuginfo(prj='ds', info=f'Round robin gradient partitioning: {round_robin_gradients}') # False

        # The fused optimizer does all the work. We need this layer for two reason:
        # 1. maintain same user API from apex.fp16_utils
        # 2. keep common stuff here in case we need to add ne552w fused optimizer later
        # 设置一些属性
        self.elastic_checkpoint = elastic_checkpoint
        self.param_names = param_names
        self.mpu = mpu  # 模型并行单元，用于处理模型并行的相关操作
        gd.debuginfo(prj='ds', info=f"self.elastic_checkpoint={self.elastic_checkpoint}") # False
        gd.debuginfo(prj='ds', info=f"len of self.param_names={len(self.param_names)}")
        for k, v in self.param_names.items():   # k 是tensor， v是字符串，张量的名称
            gd.debuginfo(prj='ds', info=f"self.param_names[{infoTensor(k)}]={v}")
        gd.debuginfo(prj='ds', info=f"self.mpu={self.mpu}")  # None

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
        gd.debuginfo(prj='ds', info=f"self.partition_gradients={self.partition_gradients}") # True

        # stage 阶段
        self.zero_stage_string = "ZeRO-2" if partition_grads else "ZeRO-1"
        gd.debuginfo(prj='ds', info=f"self.zero_stage_string={self.zero_stage_string}")  # "ZeRO-2"

        self.timers = timers

        self.reduce_scatter = reduce_scatter
        gd.debuginfo(prj='ds', info=f"self.reduce_scatter={self.reduce_scatter}") # True

        # 配置项 默认为 False
        # 尝试将梯度缩减与逆向计算相重叠
        self.overlap_comm = overlap_comm  # type: bool
        gd.debuginfo(prj='ds', info=f"self.overlap_comm={self.overlap_comm}")  # False TBD-True

        self.cpu_offload = cpu_offload  # False
        self.deepspeed_adam_offload = cpu_offload
        gd.debuginfo(prj='ds', info=f"self.cpu_offload={self.cpu_offload}") # False

        # 获取当前设备，如果开启了CPU offload，那么设备为cpu，否则为当前设备
        self.device = get_accelerator().current_device_name() if not self.cpu_offload else 'cpu'
        gd.debuginfo(prj='ds', info=f"self.device={self.device}")  # cuda:0

        # 所属的并行进程组
        self.dp_process_group = dp_process_group
        # gd.debuginfo(prj='ds', info=f"self.dp_process_group={self.dp_process_group}")
        # self.dp_process_group=<torch.distributed.distributed_c10d.ProcessGroup object at 0x7f2500342eb0>
        #  专家并行所属的组  expert parallel group  ，这是MoE的概念，ph1-z2没有用到
        self.ep_process_group = expert_parallel_group
        gd.debuginfo(prj='ds', info=f"self.ep_process_group={self.ep_process_group}") # None

        #data parallel group for experts
        # 专家数据并行组  data parallel group for experts  ，这是MoE的概念，ph1-z2没有用到
        self.expert_dp_process_group = expert_data_parallel_group
        gd.debuginfo(prj='ds', info=f"self.expert_dp_process_group={self.expert_dp_process_group}") # None

        #data parallel size for non-experts # 数据并行的大小
        dp_size = dist.get_world_size(group=self.dp_process_group)
        gd.debuginfo(prj='ds', info=f"dp_size={dp_size}") # 2，就是机器上显卡个数， 由pytorch返回， 可以CUDA_VISIBILE_DEVICE控制

        #For MoE models this maybe different for different param group
        #It will be modified during MoE setup later in the init
        # 对于MoE模型，这可能对于不同的参数组是不同的
        # 它将在init中的MoE设置过程中被修改
        self.real_dp_process_group = [dp_process_group for i in range(len(self.optimizer.param_groups))] # 也就是说每个参数组的进程组对象是一样的
        self.partition_count = [dp_size for i in range(len(self.optimizer.param_groups))] # 也就是说每个参数组的分区大小是一样的
        gd.debuginfo(prj='ds', info=f"partition_count={self.partition_count}")  #2张卡 [2, 2]没有lora, [2,2,2]有lora；3张卡[3,3]
        gd.debuginfo(prj='ds', info=f"number of real_dp_process_group={len(self.real_dp_process_group)}") # 2
        # real_dp_process_group=[<torch.distributed.distributed_c10d.ProcessGroup object at 0x7f75c439f2f0>, <torch
        
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
            gd.debuginfo(prj="ds")  # ph1-z2 没有使用模型并行！
            self.model_parallel_group = None
            self.model_parallel_world_size = 1
            self.model_parallel_rank = 0
        else:
            gd.debuginfo(prj="ds")
            self.model_parallel_group = mpu.get_model_parallel_group()
            self.model_parallel_world_size = mpu.get_model_parallel_world_size()
            self.model_parallel_rank = bwc_tensor_model_parallel_rank(mpu)

        gd.debuginfo(prj='ds', info=f'self.model_parallel_group={self.model_parallel_group}')
        gd.debuginfo(prj='ds', info=f'self.model_parallel_world_size={self.model_parallel_world_size}')
        gd.debuginfo(prj='ds', info=f'self.model_parallel_rank={self.model_parallel_rank}')

        gd.debuginfo(prj='ds', info=f'++++++++++++++sep 0+++++++++++++++')

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
        self.bit16_groups_flat = []  # 扁平化按组划分的参数

        # param partitioned by data parallel degree， 数据并行划分的参数组
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
        # self.params_in_partition中第一个参数的偏移量， 参数的边界可能和分区的边界不一致，需要跟踪offset
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
        gd.debuginfo(prj="ds", info=f'self.dtype={self.dtype}')
        self.round_robin_bit16_groups = []  # 初始化空的round_robin_bit16_groups
        self.round_robin_bit16_indices = []  # 初始化空的round_robin_bit16_indices

        # Use different parallel to do all_to_all_reduce related things
        # padding on each partition for alignment purposes all_to_all_reduce
        # 用于对齐每个分区的填充， 和使用不同的并行来做相关 ???, index是每个partition_ID, val是padding个数
        self.groups_padding = []

        gd.debuginfo(prj='ds', info=f'++++++++++++++sep 1+++++++++++++++')

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
        # 遍历优化器的参数组
        for i, param_group in enumerate(self.optimizer.param_groups):
            # 每组参数分开处理， 获取当前分区的id
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])

            gd.debuginfo(prj="ds", info=f'-----The {i}th param_group with partition_id={partition_id} start----------------')

            # push this group to list before modify
            # TODO: Explore simplification that avoids the extra book-keeping by pushing the reordered group
            # 存储需要训练的参数
            trainable_parameters = [param for param in param_group['params'] if param.requires_grad]
            gd.debuginfo(prj="ds", info=f'len of trainable_parameters={len(trainable_parameters)}')
            for index, param in enumerate(trainable_parameters):
                gd.debuginfo(prj="ds", info=f'{index}th param={infoTensor(param)} ')  # 99个
            # 上面2卡3卡都一样，TBD 记录到文件
            # 当前 param_group 中需要梯度更新，也就是需要训练的参数列表
            # 后续的分割都是针对他们的，这是针对每个参数group的
            self.bit16_groups.append(trainable_parameters)

            # not sure why apex was cloning the weights before flattening
            # removing cloning here

            see_memory_usage(f"Before moving param group {i} to CPU")

            # move all the parameters to cpu to free up GPU space for creating flat buffer
            # 先转移到 cpu 内存，在 cpu 内存中进行处理 移动所有参数到cpu，以释放GPU空间，用于创建平坦缓冲区
            move_to_cpu(self.bit16_groups[i])
            # 调用 pytorch/accelerator api
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
                gd.debuginfo(prj="ds")
                # 为了能尽量的均匀分配，这里采用循环分配（round_robin 方法）
                round_robin_tensors, round_robin_indices = self._round_robin_reorder(
                    self.bit16_groups[i], dist.get_world_size(group=self.real_dp_process_group[i]))
                # gd.debuginfo(prj="ds", info=f'round_robin_indices={round_robin_indices}, \
                #             round_robin_tensors={infoTensor(round_robin_tensors)}') ==不是tensor，而是tensor list
            else:
                gd.debuginfo(prj="ds")  # ph1-z2
                round_robin_tensors = self.bit16_groups[i]
                round_robin_indices = list(range(len(self.bit16_groups[i])))  # [0,1,2...len-1]
                # gd.debuginfo(prj="ds", info=f'round_robin_indices={round_robin_indices}, \
                #             round_robin_tensors={infoTensor(round_robin_tensors)}')

            # group级别的记录
            self.round_robin_bit16_groups.append(round_robin_tensors) #放到cpu后
            self.round_robin_bit16_indices.append(round_robin_indices)

            # create flat buffer in CPU and move to GPU
            # 将参数列表打平放到一个一维连续空间中 在CPU中创建平坦缓冲区并移动到GPU
            self.bit16_groups_flat.append(
                self.flatten_dense_tensors_aligned(
                    self.round_robin_bit16_groups[i],   # 下面是alignment，-单机上2卡是4,3卡是6
                    self.nccl_start_alignment_factor * dist.get_world_size(group=self.real_dp_process_group[i])).to(
                        get_accelerator().current_device_name()))

            see_memory_usage(f"After flattening and moving param group {i} to GPU", force=True)

            # Record padding required for alignment
            # 上面在打平的时候，可能在尾部添加了padding，这里要记录一下padding的个数
            if partition_id == dist.get_world_size(group=self.real_dp_process_group[i]) - 1:
                # 如果是最后一个分区，计算填充值 ???没什么一个都没有
                padding = self.bit16_groups_flat[i].numel() - sum(
                    [t.numel() for t in self.round_robin_bit16_groups[i]])
                gd.debuginfo(prj="ds", info=f'padding={padding} on partition_id={partition_id}')
            else:
                # 否则，填充为0,就是不用填充
                gd.debuginfo(prj="ds", info=f'padding is 0 on partition_id={partition_id}')
                padding = 0   # ph1-z2全在这里

            self.groups_padding.append(padding)

            if dist.get_rank(group=self.real_dp_process_group[i]) == 0:
                see_memory_usage(f"After Flattening and after emptying param group {i} cache", force=True)
            gd.debuginfo(prj="ds", info=f'++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

            # set model bit16 weight to slices of flattened buffer
            # 更新模型各个param组的bit16权重  ---init阶段时候有必要????
            self._update_model_bit16_weights(i)
            gd.debuginfo(prj="ds", info=f'++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            # divide the flat weights into near equal partition equal to the data parallel degree
            # each process will compute on a different part of the partition
            # data_parallel_partitions 是分割好的结果
            # data_parallel_partitions 是一个字典类型,key 是 rank ，value 是分号的参数
            # 将平坦权重划分为近等的分区，等于数据并行度，每个进程将在分区的不同部分进行计算
            # 具体内容在里面打印
            data_parallel_partitions = self.get_data_parallel_partitions(self.bit16_groups_flat[i], i) #data_parallel_partitions=[tensor([ 0.1150, -0.1438,  0.0555,  ...,  0.0183,  0.0088,  0.0258],device='cuda:0', dtype=torch.float16, grad_fn=<SliceBackward0>), tensor([0.0698, 0.0299, 0.0293,  ..., 0.3955, 0.3677, 0.5161], device='cuda:0', dtype=torch.float16, grad_fn=<SliceBackward0>)]
            gd.debuginfo(prj="ds", info=f'len of data_parallel_partitions={len(data_parallel_partitions)}')

            self.parallel_partitioned_bit16_groups.append(data_parallel_partitions)
            # verify that data partition start locations are 4-byte aligned
            # 验证数据分区起始位置是否为4字节对齐
            # torch.Tensor.data_ptr = Returns the address of the first element of self tensor
            # 返回tensor首元素的内存地址， 常用来判断两个Tensor是不是共享内存
            for partitioned_data in data_parallel_partitions:
                assert (partitioned_data.data_ptr() % (2 * self.nccl_start_alignment_factor) == 0)

            # A partition of the fp32 master weights that will be updated by this process.
            # Note that the params in single_partition_of_fp32_groups is cloned and detached
            # from the origin params of the model.
            # 把属于当前进程（rank）的参数移动到指定设备，然后创建一个副本
            # 这个副本用于累积梯度进行参数更新，根据配置，可以是 单精度（float32）也可以是半精度（float16）
            # 注意这个副本 detach 操作，
            # 返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置, 不同之处只是requires_grad为false，
            # 得到的这个tensor永远不需要计算其梯度，不具有grad
            # 创建一个fp32主权重的分区，这个分区会被这个进程更新。
            # 注意，single_partition_of_fp32_groups中的参数是从模型的原始参数中克隆和分离出来的。
            if not fp16_master_weights_and_gradients:
                tmp = self.parallel_partitioned_bit16_groups[i][partition_id].to(self.device).clone().float().detach()
                gd.debuginfo(prj="ds", info=f'self.parallel_partitioned_bit16_groups[{i}][{partition_id}]={tmp}')  # ph1-z2
                self.single_partition_of_fp32_groups.append(tmp)
            else:
                tmp = self.parallel_partitioned_bit16_groups[i][partition_id].to(self.device).clone().half().detach()
                gd.debuginfo(prj="ds", info=f'self.parallel_partitioned_bit16_groups[{i}][{partition_id}]={tmp}')
                self.single_partition_of_fp32_groups.append(tmp)
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
            for index, val in enumerate(param_group['params']):
                gd.debuginfo(prj="ds", info=f"param_group['params'][{index}]={infoTensor(val)}")

            # 计算分区大小和分区内的参数信息
            partition_size = len(self.bit16_groups_flat[i]) / dist.get_world_size(group=self.real_dp_process_group[i])
            params_in_partition, params_not_in_partition, first_offset = self.get_partition_info(
                self.round_robin_bit16_groups[i], partition_size, partition_id)

            gd.debuginfo(prj="ds", info=f"partition_size={partition_size}") # partition_size=62568576.0
            for index, val in enumerate(params_in_partition):
                gd.debuginfo(prj="ds", info=f"params_in_partition[{index}]={infoTensor(val)}")
            gd.debuginfo(prj="ds", info=f"-----------------------------------------------------")
            for index, val in enumerate(params_not_in_partition):
                gd.debuginfo(prj="ds", info=f"params_not_in_partition[{index}]={infoTensor(val)}")
            gd.debuginfo(prj="ds", info=f"first_offset={first_offset}")

            # 存储分区大小和参数信息， index就是param_group ID
            self.partition_size.append(partition_size)
            self.params_in_partition.append(params_in_partition)
            self.params_not_in_partition.append(params_not_in_partition)
            self.first_offset.append(first_offset)

            gd.debuginfo(prj="ds", info=f'-----------{i}th partition_id={partition_id} end----------------')

        gd.debuginfo(prj="ds", info=f"lens of self.partition_size={len(self.partition_size)}")
        gd.debuginfo(prj="ds", info=f"lens of self.params_in_partition={len(self.params_in_partition)}")
        gd.debuginfo(prj="ds", info=f"lens of self.params_not_in_partition={len(self.params_not_in_partition)}")
        gd.debuginfo(prj="ds", info=f"lens of self.first_offset={len(self.first_offset)}")

        for rank in range(dist.get_world_size()):
            if dist.get_rank() == rank:
                gd.debuginfo(prj="ds", info=f"Rank: {rank} partition count {self.partition_count} and "
                                            f"sizes{[(p.numel(), self.is_moe_param_group[i] if hasattr(self, 'is_moe_param_group') else False) for i,p in enumerate(self.single_partition_of_fp32_groups)]} ")
                dist.barrier()

        gd.debuginfo(prj='ds', info=f'++++++++++++++sep 2+++++++++++++++')

        # 设置一些基本参数和流
        self.reduce_bucket_size = int(reduce_bucket_size)
        self.allgather_bucket_size = int(allgather_bucket_size)
        self.reduction_event = get_accelerator().Event(enable_timing=False, blocking=False)
        self.reduction_stream = get_accelerator().Stream()
        self.cpu_computation_stream = get_accelerator().Stream()
        self.copy_grad_stream = get_accelerator().Stream()

        gd.debuginfo(prj='ds', info=f'self.reduce_bucket_size={self.reduce_bucket_size}') # 500000000 五亿
        gd.debuginfo(prj='ds', info=f'self.allgather_bucket_size={self.allgather_bucket_size}') # 500000000
        # gd.debuginfo(prj='ds', info=f'self.reduction_event={self.reduction_event}')
        # gd.debuginfo(prj='ds', info=f'self.reduction_stream={self.reduction_stream}')
        # gd.debuginfo(prj='ds', info=f'self.cpu_computation_stream={self.cpu_computation_stream}')
        # gd.debuginfo(prj='ds', info=f'self.copy_grad_stream={self.copy_grad_stream}')
        # self.reduction_event=<torch.cuda.Event uninitialized>
        # self.reduction_stream=<torch.cuda.Stream device=cuda:0 cuda_stream=0xab91a80>
        # self.cpu_computation_stream=<torch.cuda.Stream device=cuda:0 cuda_stream=0xab9d710>
        # self.copy_grad_stream=<torch.cuda.Stream device=cuda:0 cuda_stream=0xab9d990>

        # 初始化一些参数和缓存列表
        self.callback_queued = False

        # 从后面代码中看出key是参数ID(从零开始连续的值), 值是参数
        self.param_dict = {}

        # map between param_id and bool to specify if a param is in this partition
        # 用来确定一个param是否在分区中，key是param_id,值是bool
        self.is_param_in_current_partition = {}

        # TBD
        self.grads_in_ipg_bucket = []
        self.params_in_ipg_bucket = []
        self.elements_in_ipg_bucket = 0

        # 以param_id为下标，记录该param是否已经reduce，
        self.params_already_reduced = []

        self._release_ipg_buffers()

        self.previous_reduced_grads = None
        self.ipg_bucket_has_moe_params = False

        gd.debuginfo(prj='ds', info=f'++++++++++++++sep 3+++++++++++++++')

        # simplified param id  # 简化参数id
        # key是 用id函数得到的唯一标识号，val是连续的整数值，目的是简化
        self.param_id = {}

        #interesting code: unique ids being assigned to individual parameters
        # 对每个参数进行唯一标识
        largest_param_numel = 0 #所有param中tensor个数最大值
        count = 0 # 简化的param_id从0开始的连续整数
        for i, params_group in enumerate(self.bit16_groups):
            for param in params_group:
                unique_id = id(param)
                gd.debuginfo(prj='ds', info=f'params_group={i}, count={count}, unique_id={unique_id}, param={infoTensor(param)}')

                self.param_id[unique_id] = count
                self.param_dict[count] = param
                self.params_already_reduced.append(False)
                gd.debuginfo(prj='ds', info=f'param.numel()={param.numel()}, largest_param_numel={largest_param_numel}')

                if param.numel() > largest_param_numel:
                    largest_param_numel = param.numel()
                count = count + 1

            gd.debuginfo(prj='ds', info=f'------------------i={i}, count={count}-------------------')

        # 标记参数是否在当前分区, index是param ID--简化的，从0开始的整数
        for index, param_group in enumerate(self.params_in_partition):
            for param in param_group:
                gd.debuginfo(prj='ds', info=f'index={index}, param={infoTensor(param)}, self.get_param_id(param)={self.get_param_id(param)}')
                self.is_param_in_current_partition[self.get_param_id(param)] = True
        gd.debuginfo(prj='ds', info=f'********************************************************************************')
        for index, param_group in enumerate(self.params_not_in_partition):
            for param in param_group:
                gd.debuginfo(prj='ds', info=f'index={index}, param={infoTensor(param)}, self.get_param_id(param)={self.get_param_id(param)}')
                self.is_param_in_current_partition[self.get_param_id(param)] = False

        # 如果开启了CPU offload的功能
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
            gd.debuginfo(prj='ds', info=f'self.temp_grad_buffer_for_cpu_offload={infoTensor(self.temp_grad_buffer_for_cpu_offload)}')

            # 在设备上创造一个全零的tensor，用于GPU offload
            self.temp_grad_buffer_for_gpu_offload = torch.zeros(largest_param_numel,
                                                                device=get_accelerator().current_device_name(),
                                                                dtype=self.dtype)
            gd.debuginfo(prj='ds', info=f'self.temp_grad_buffer_for_gpu_offload={infoTensor(self.temp_grad_buffer_for_gpu_offload)}')

            for i, params_group in enumerate(self.bit16_groups):
                self.get_grad_position(i, self.params_in_partition[i], self.first_offset[i], self.partition_size[i])

        # 初始化一些用于梯度分区的参数 === z2

        # mapping from parameter to partition that it belongs to
        # 参数到它所在分区的映射 是二维的[group_id][param_id]
        self.param_to_partition_ids = {}

        # stores if a partition has been reduced in this step
        # 存储分区是否已经进行了reduce操作 是二维的[group_id][param_id]
        self.is_partition_reduced = {}

        # number of grads in partition that still need to be computed
        # 分区内还需要计算的梯度数量 是二维的[group_id][param_id]
        self.remaining_grads_in_partition = {}

        # total number of grads in partition
        # 分区内总共的梯度数量  是二维的[group_id][param_id]
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
        gd.debuginfo(prj='ds', info=f'++++++++++++++sep 4+++++++++++++++')
        self.initialize_gradient_partitioning_data_structures()

        # resets the data structure value for the next backward propagation
        # 重置数据结构的值以便下一次反向传播
        gd.debuginfo(prj='ds', info=f'++++++++++++++sep 5+++++++++++++++')
        self.reset_partition_gradient_structures()

        gd.debuginfo(prj='ds', info=f'++++++++++++++sep 6+++++++++++++++')

        # creates backward hooks for gradient partitioning
        # 如果启用了梯度分区或者通信重叠，创建后向钩子
        if self.partition_gradients or self.overlap_comm: # z1 false, z2 true
            gd.debuginfo(prj="ds") # ph1-z2
            self.create_reduce_and_remove_grad_hooks()

        self.custom_loss_scaler = False
        self.external_loss_scale = None

        gd.debuginfo(prj='ds', info=f'++++++++++++++sep 7+++++++++++++++')

        # we may have a way of fusing dynamic scale. Do not support for now
        # 创建损失缩放器，可能是静态或者动态的
        self.loss_scaler = CreateLossScaler(dtype=self.dtype,
                                            static_loss_scale=static_loss_scale,
                                            dynamic_scaling=dynamic_loss_scale,
                                            dynamic_loss_args=dynamic_loss_args)
        self.dynamic_loss_scale = self.loss_scaler.dynamic
        gd.debuginfo(prj="ds", info=f"self.loss_scaler={self.loss_scaler}") # self.loss_scaler=<deepspeed.runtime.fp16.loss_scaler.DynamicLossScaler object at 0x7f25109e7310>
        gd.debuginfo(prj="ds", info=f"self.dynamic_loss_scale={self.dynamic_loss_scale}")  # True

        # 只有当数据类型为float16时，才会使用动态损失缩放
        if self.dtype != torch.float16:
            gd.debuginfo(prj='ds')
            # Only fp16 should use dynamic loss scaling
            assert self.loss_scaler.cur_scale == 1.0
            assert not self.dynamic_loss_scale
        gd.debuginfo(prj='ds', info=f'++++++++++++++sep 8+++++++++++++++')

        see_memory_usage("Before initializing optimizer states", force=True)
        self.initialize_optimizer_states()
        gd.debuginfo(prj='ds', info=f'self.optimizer.param_groups={self.optimizer.param_groups}')
        see_memory_usage("After initializing optimizer states", force=True)

        gd.debuginfo(prj='ds', info=f'++++++++++++++sep 9+++++++++++++++')
        # 如果是主节点，则打印优化器状态初始化成功的信息
        if dist.get_rank() == 0:
            gd.debuginfo(prj="ds", info=f"optimizer state initialized")

        # 如果是数据并行处理组的主节点，打印ZeRO优化器初始化后的内存使用情况
        if dist.get_rank(group=self.dp_process_group) == 0:
            see_memory_usage(f"After initializing ZeRO optimizer", force=True)

        gd.debuginfo(prj='ds', info=f'++++++++++++++sep A+++++++++++++++')
        # 链接所有超参数
        self._link_all_hp_params()

        gd.debuginfo(prj='ds', info=f'++++++++++++++sep B+++++++++++++++')

        # 启用通用检查点
        self._enable_universal_checkpoint()

        gd.debuginfo(prj='ds', info=f'++++++++++++++sep C+++++++++++++++')

        # 创建参数映射
        self._param_slice_mappings = self._create_param_mapping()
        gd.debuginfo(prj='ds', info=f'++++++++++++++sep D+++++++++++++++')


        gd.printall(prj='ds', cname=self)
        gd.debuginfo(prj='ds', info=f'C:{self.__class__.__name__} FUNC_OUT')

    # 检查点启用  用于开启通用的模型检查点。
    def _enable_universal_checkpoint(self):
        # 遍历bit16_groups中的所有参数组
        for index, lp_param_group in enumerate(self.bit16_groups):
            gd.debuginfo(prj="ds", info=f'index={index}') #tensor的list
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
            for index, lp in enumerate(self.bit16_groups[i]):
                # 检查lp是否有_hp_mapping属性，如果有，说明它有一些需要映射的超参数
                if lp._hp_mapping is not None:
                    # 获取该层或参数的名称
                    lp_name = self.param_names[lp]

                    # 在有序字典中添加一个新的键值对，键是层或参数的名称，值是超参数的映射地址
                    param_mapping_per_group[lp_name] = lp._hp_mapping.get_hp_fragment_address()

                    gd.debuginfo(prj='ds', info=f'On bit16_groups[{i}][{index}]:, '
                                                f'param_mapping_per_group[{lp_name}]={param_mapping_per_group[lp_name]}')
                else:
                    gd.debuginfo(prj='ds', info=f'On bit16_groups[{i}][{index}] no action!')
            # 将该参数组的映射添加到整体的参数映射列表中
            param_mapping.append(param_mapping_per_group)
            gd.debuginfo(prj='ds', info=f'===============================================')

        gd.debuginfo(prj='ds', info=f'len of param_mapping={len(param_mapping)}')
        # 返回参数映射列表
        return param_mapping

    # 用于链接所有的超参数。这个函数的目标看起来是链接所有的半精度（16位）参数和单精度（32位）参数。
    # 它主要用于分布式训练，特别是在使用CPU offload和数据并行性（Data Parallelism）时
    # call in _load_legacy_checkpoint and init
    def _link_all_hp_params(self):
        # 获取分布式处理过程中的世界大小
        dp_world_size = dist.get_world_size(group=self.dp_process_group)
        gd.debuginfo(prj='ds', info=f'FUNC_IN, dp_world_size={dp_world_size}')

        # 如果启用了CPU卸载，获取卸载梯度字典
        if self.cpu_offload:
            # gd.debuginfo(prj="ds")
            self._get_offload_gradient_dict()

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

            gd.debuginfo(prj='ds', info=f'{i} param_groups, flat_hp_partition={infoTensor(flat_hp_partition)},'
                           f'gradient_dict={self.averaged_gradients},'
                           f'use_offload={self.cpu_offload},'
                           f'param_group_index={i},'
                           f'partition_start={partition_id * partition_size},'
                           f'partition_size={partition_size},'
                           f'partition_optimizer_state={self.optimizer.state[flat_hp_partition]},'
                           f'dp_group={self.real_dp_process_group[i]}')

            if self.cpu_offload:
                for k,v in self.offload_gradient_dict.items():
                    gd.debuginfo(prj='ds', info=f'self.offload_gradient_dict[{k}] has {len(v)} tensors')
                    for index, val in enumerate(v):
                        gd.debuginfo(prj='ds', info=f'self.offload_gradient_dict[{k}][index]={infoTensor(val)}')


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

        gd.debuginfo(prj='ds', info=f'FUNC_OUT')

    # 检查是否为MOE（Mixture of Experts）组
    def is_moe_group(self, group):
        # gd.debuginfo(prj="ds")
        return 'moe' in group and group['moe']

    # 用于配置MOE设置检查
    def _configure_moe_settings(self):
        gd.debuginfo(prj='ds', info=f'FUNC_IN')
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
        gd.debuginfo(prj='ds', info=f'FUNC_OUT')

    #更新16位浮点数权重的模型。
    def _update_model_bit16_weights(self, group_index):
        gd.debuginfo(prj="ds", info=f'group_index={group_index}  FUNC_IN')
        # 解压缩16位小组的数据 是pytorch函数
        updated_params = self.unflatten(self.bit16_groups_flat[group_index],
                                        self.round_robin_bit16_groups[group_index])

        #打印出来 gd.debuginfo(prj="ds", info=f'T: updated_params={infoTensor(updated_params)}') tensor list
        # 可能不相等 assert len(self.bit16_groups_flat[group_index]) == len(self.round_robin_bit16_groups[group_index])

        gd.debuginfo(prj="ds", info=f'len of self.bit16_groups_flat[{group_index}]={len(self.bit16_groups_flat[group_index])}')

        gd.debuginfo(prj="ds", info=f'len of self.round_robin_bit16_groups[{group_index}]={len(self.round_robin_bit16_groups[group_index])}')
        gd.debuginfo(prj="ds", info=f'len of updated_params={len(updated_params)}')
        assert len(updated_params) == len(self.round_robin_bit16_groups[group_index])
        gd.debuginfo(prj='ds', info=f'+++++++++++打印出来对比异同+++++++++++++++++++++++++++++++++++++++++')
        # 2卡3卡内容一样
        for index, (pu,pi) in enumerate(zip(updated_params, self.round_robin_bit16_groups[group_index])): # 99个
            gd.debuginfo(prj="ds", info=f'T: updated_params[{index}]={infoTensor(pu)}, '
                                        f'self.round_robin_bit16_groups[{group_index}][{index}]={infoTensor(pi)}')
        # 两个张量维度一样 updated_params[77]=_Size([768, 768])_float16_cuda:0_,
        # self.round_robin_bit16_groups[0][77]=_Size([768, 768])_float16_cpu_
        # 区别是 device！！  TBD 记录到文件
        gd.debuginfo(prj='ds', info=f'+++++++++++打印出来对比异同+++++++++++++++++++++++++++++++++++++++++')
        # for index, p in enumerate(updated_params): # 99个
        #     gd.debuginfo(prj="ds", info=f'T: updated_params[{index}]={infoTensor(updated_params[index])}')

        # 遍历原始小组和更新的参数，用更新的参数来更新原始参数
        for index, (p, q) in enumerate(zip(self.round_robin_bit16_groups[group_index], updated_params)):
            p.data = q.data  #下面2卡3卡都一样
            gd.debuginfo(prj="ds", info=f'T: index={index},'
                                        f'p={infoTensor(p)}, '
                                        f'q={infoTensor(q)}, '
                                        f'p.data={infoTensor(p.data)},'
                                        f'p.equal(q)={p.equal(q)}') # init中都是相等！
            # gd.debuginfo(prj="ds", info=f'T: p.grad={infoTensor(p.grad)}')  # None

        gd.debuginfo(prj='ds', info=f'+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        # set model fp16 weight to slices of reordered flattened buffer
        # 将模型的16位权重设置为重新排序的扁平缓冲区的切片
        for param_index, param in enumerate(self.bit16_groups[group_index]):
            # 获取新的索引
            new_index = self.round_robin_bit16_indices[group_index][param_index] # init时候param_index和new_index应该是一样的

            # 使用新的索引更新参数数据
            tmp = self.round_robin_bit16_groups[group_index][new_index]
            param.data = tmp.data  #2卡3卡都一样
            gd.debuginfo(prj="ds", info=f'T: new_index={new_index}, '
                                        f'param_index={param_index}, '
                                        f'param.data={infoTensor(param.data)}'
                                        f'param.equal(tmp)={param.equal(tmp)}')  # init中都是相等

        gd.debuginfo(prj='ds', info=f'FUNC_OUT')

    # 用于在多个设备间重新排序数据。
    def _round_robin_reorder(self, tensor_list, num_partitions):
        gd.debuginfo(prj='ds', info=f'FUNC_IN', num_partitions={num_partitions})
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

        gd.debuginfo(prj='ds', info=f'FUNC_OUT')
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
        gd.debuginfo(prj='ds', info=f'FUNC_IN')
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
            gd.debuginfo(prj="ds")
            self.optimizer = torch.optim.Adagrad(self.single_partition_of_fp32_groups, **self.optimizer.defaults)
        else:
            # gd.debuginfo(prj="ds")
            # 其他类型的优化器则直接调用 step 方法
            self.optimizer.step()

        # 如果不进行 cpu_offload，那么就将 single_partition_of_fp32_groups 中的每个组的梯度设置为 None
        if not self.cpu_offload:
            for group in self.single_partition_of_fp32_groups:
                group.grad = None  #class init  # 初始化类

        gd.debuginfo(prj='ds', info=f'FUNC_OUT')
        return

    #########################################################################
    #################### ZeRO Stage 1 - reduce gradients ####################
    #########################################################################
    # reduce - 梯度。
    def reduce_gradients(self, pipeline_parallel=False):
        gd.debuginfo(prj="ds", info=f'FUNC_IN')
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

        gd.debuginfo(prj="ds", info=f'FUNC_OUT')

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

            # 检查当前的参数ID是否在指定的分区ID内
            # 如果在，就返回当前的索引
            if partition_id in self.param_to_partition_ids[group_id][param_id]:
                gd.debuginfo(prj="ds", info=f'index={index}, param_id={param_id}, T: param={infoTensor(param)}, return {index}')
                return index

        gd.debuginfo(prj="ds", info=f'index={index}, param_id={param_id}, T: param={infoTensor(param)}, return {None}')

        # 如果没有找到满足条件的参数，就返回None
        return None

    # 初始化梯度分区的数据结构
    def initialize_gradient_partitioning_data_structures(self):
        gd.debuginfo(prj='ds', info=f'FUNC_IN')
        # 遍历所有的参数组
        for i, param_group in enumerate(self.round_robin_bit16_groups):
            # 获取分区的总数，也就是分布式处理组的大小
            total_partitions = dist.get_world_size(group=self.real_dp_process_group[i])
            gd.debuginfo(prj='ds', info=f'i={i} th param_group total_partitions={total_partitions}')

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
                gd.debuginfo(prj='ds', info=f'--sep--groupID={i}--partition_id={partition_id}--initialize_gradient_partition--start')
                self.initialize_gradient_partition(i, param_group, partition_id)
                gd.debuginfo(prj='ds', info=f'--sep--groupID={i}--partition_id={partition_id}--initialize_gradient_partition--end')

                # 初始化每个分区的缩减状态为False
                self.is_partition_reduced[i][partition_id] = False

                # 获取并存储每个分区的第一个参数的索引
                self.first_param_index_in_partition[i][partition_id] = self.get_first_param_index(i, param_group, partition_id)
                gd.debuginfo(prj='ds',info=f'self.first_param_index_in_partition[{i}][{partition_id}]={self.first_param_index_in_partition[i][partition_id]}')

            gd.debuginfo(prj='ds', info=f'self.param_to_partition_ids[{i}]={self.param_to_partition_ids[i]}')
            gd.debuginfo(prj='ds', info=f'self.is_partition_reduced[{i}]={self.is_partition_reduced[i]}')
            gd.debuginfo(prj='ds', info=f'self.total_grads_in_partition[{i}]={self.total_grads_in_partition[i]}')
            gd.debuginfo(prj='ds', info=f'self.remaining_grads_in_partition[{i}]={self.remaining_grads_in_partition[i]}')
            gd.debuginfo(prj='ds', info=f'self.is_grad_computed[{i}]={self.is_grad_computed[i]}')
            gd.debuginfo(prj='ds', info=f'self.grad_partition_insertion_offset[{i}]={self.grad_partition_insertion_offset[i]}')
            gd.debuginfo(prj='ds', info=f'self.grad_start_offset[{i}]={self.grad_start_offset[i]}')
            gd.debuginfo(prj='ds', info=f'self.first_param_index_in_partition[{i}]={self.first_param_index_in_partition[i]}')

        gd.debuginfo(prj='ds', info=f'FUNC_OUT')

    def independent_gradient_partition_epilogue(self):
        gd.debuginfo(prj='ds', info=f'FUNC_IN')
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

        gd.debuginfo(prj='ds', info=f'FUNC_OUT')

    # resets all partition to no reduced
    # sets remaining grads to the total number of grads in each partition
    # set is grad computed to false for all grads in partition
    # 重置与每个分区相关的梯度结构
    def reset_partition_gradient_structures(self): #也就是init执行了一次
        gd.debuginfo(prj="ds")
        # 遍历所有的参数组
        for i, _ in enumerate(self.bit16_groups):
            # 获取分区的总数，这是通过获取分布式处理组的大小来决定的
            total_partitions = dist.get_world_size(group=self.real_dp_process_group[i])

            # 遍历所有的分区
            for partition_id in range(total_partitions):
                # 将每个分区的缩减状态设为False
                self.is_partition_reduced[i][partition_id] = False

                # 将每个分区剩余的梯度数量设置为每个分区的梯度总数i
                self.remaining_grads_in_partition[i][partition_id] = self.total_grads_in_partition[i][partition_id]
                gd.debuginfo(prj="ds", info=f'remaining_grads_in_partition[{i}][{partition_id}]={self.remaining_grads_in_partition[i][partition_id]}')

                # 遍历分区中每个参数的梯度计算状态
                gd.debuginfo(prj="ds", info=f'len of self.is_grad_computed[{i}][{partition_id}]={len(self.is_grad_computed[i][partition_id])}')
                for param_id in self.is_grad_computed[i][partition_id]:
                    # gd.debuginfo(prj="ds", info=f'i={i}, partition_id={partition_id}, param_id={param_id}')
                    # 将每个参数的梯度计算状态设为False
                    self.is_grad_computed[i][partition_id][param_id] = False

    # 初始化reduce分区
    def initialize_gradient_partition(self, i, param_group, partition_id):

        gd.debuginfo(prj="ds", info=f'i={i}+++partition_id={partition_id} FUNC_IN')
        # param_group={infoTensor(param_group)} 是一个tensor的列表，太大
        gd.debuginfo(prj='ds', info=f"len of param_group={len(param_group)}")  # 防止消失，直接打印
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
            gd.debuginfo(prj="ds", info=f'{i}_param_group, param_id={param_id}, param={infoTensor(param)}, param_size={param_size}')

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

        gd.debuginfo(prj="ds", info=f'{i}_param_group, partition_id={partition_id}, FUNC_OUT')

    # 调用IGP的梯度reduce操作
    def overlapping_partition_gradients_reduce_epilogue(self):
        # gd.debuginfo(prj="ds")
        self.independent_gradient_partition_epilogue()

    # 创建并删除梯度钩子  本身只是init执行，
    # 注册的函数reduce_ready_partitions_and_remove_grads会在train的backward执行
    def create_reduce_and_remove_grad_hooks(self):
        gd.debuginfo(prj="ds", info=f'FUNC_IN')
        # 初始化一个用于存储梯度累积函数的列表
        self.grad_accs = []

        # 遍历所有的16位分组
        for i, param_group in enumerate(self.bit16_groups):
            gd.debuginfo(prj="ds", info=f'---START HOOK--------------------------')
            # 在每个分组中遍历所有的参数
            for param in param_group:
                # 如果参数需要计算梯度
                if param.requires_grad:
                    # 定义一个闭包函数，用于注册梯度钩子
                    def wrapper(param, i):
                        # 创建一个与参数形状相同的临时参数
                        param_tmp = param.expand_as(param)
                        gd.debuginfo(prj="ds", info=f'i={i}, param={infoTensor(param)} requires_grad')
                        # gd.debuginfo(prj="ds", info=f'param_tmp={infoTensor(param_tmp)}') 和上面完全一样

                        # 获取梯度函数的下一个函数
                        grad_acc = param_tmp.grad_fn.next_functions[0][0]
                        # gd.debuginfo(prj="ds", info=f'grad_acc={grad_acc}') #<AccumulateGrad object at 0x7f94d41260d0>

                        # 定义一个函数，用于在需要的时候减少分区并移除梯度
                        def reduce_partition_and_remove_grads(*notneeded):
                            self.reduce_ready_partitions_and_remove_grads(param, i)

                        # 为梯度函数注册一个钩子，当梯度计算完成时会自动调用这个钩子
                        grad_acc.register_hook(reduce_partition_and_remove_grads)

                        # 将梯度函数添加到列表中
                        self.grad_accs.append(grad_acc)

                    # 调用闭包函数，注册梯度钩子
                    wrapper(param, i)
                else:
                    gd.debuginfo(prj="ds", info=f'i={i}, param={infoTensor(param)} NOT requires_grad')
            gd.debuginfo(prj="ds", info=f'---END HOOK--------------------------')

        gd.debuginfo(prj="ds", info=f'FUNC_OUT')

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
        # for index, p in enumerate(tensor_list):
        #     gd.debuginfo(prj="ds", info=f'tensor_list[{index}]={infoTensor(p)}')
        gd.debuginfo(prj="ds", info=f'len of tensor_list= {len(tensor_list)}, alignment={alignment}')

        # 这个函数接受两个参数，一个是 tensor_list，是一个包含多个张量（tensor）的列表，另一个是 alignment，表示对齐方式
        # 这个函数的目标是将 tensor_list 中的所有张量的总长度计算出来后进行对齐，然后再进行扁平化处理
        # 不是把list每个tensor对齐，仅仅是list最后可能会加上一个padding tensor!!
        # 某些blog有严重错误理解！！！！
        # align_dense_tensors 函数的返回值是一个新的张量列表，
        aligned_tensors = align_dense_tensors(tensor_list, alignment)

        gd.debuginfo(prj="ds", info=f'len of aligned_tensors= {len(aligned_tensors)}')

        # 调用 flatten 函数，对经过对齐处理的张量列表进行扁平化处理
        # flatten 函数的返回值是一个新的扁平化后的张量
        flattened_tensor = self.flatten(aligned_tensors) # 直接使用了pytorch的函数
        # gd.debuginfo(prj="ds", info=f'T: aligned_tensors={infoTensor(aligned_tensors)}, flattened_tensor={infoTensor(flattened_tensor)}') tensor list

        # 数量过于巨大，百万级别, 所以只能计数器大概看看前100个
        # 如果不用rank保护，每个进程都会产生巨大内存消耗，三块卡就OOM
        if dist.get_rank() == 0:
            from collections import Counter
            tmpcnt = Counter()
            for _, p in enumerate(flattened_tensor):
                tmpcnt[infoTensor(p)] += 1
            gd.debuginfo(prj="ds", info=f'flattened_tensor most 100 freq:{tmpcnt.most_common(100)}')

            gd.debuginfo(prj="ds", info=f'len of flattened_tensor= {len(flattened_tensor)}')

        # 返回扁平化处理后的张量
        return flattened_tensor

    ############### Independent Partition Gradient ########################
    # reduce ipg的梯度桶并删除梯度
    def reduce_independent_p_g_buckets_and_remove_grads(self, param, i):
        gd.debuginfo(prj="ds",
                     info=f'self.elements_in_ipg_bucket={self.elements_in_ipg_bucket}, \
                     param.numel()={param.numel()}, \
                     self.reduce_bucket_size={self.reduce_bucket_size} FUNC_IN')
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
            if param.numel() > self.reduce_bucket_size:
                # 如果参数的元素数量大于bucket的大小，那么将该参数设置为待减少的参数
                gd.debuginfo(prj="ds", info=f'param={param}')
                self.extra_large_param_to_reduce = param
            else:
                # 保持梯度连续，以防止内存碎片化，并且避免展平
                # 都是有效输出
                # gd.debuginfo(prj="ds", info=f'1-param={infoTensor(param)}')
                # gd.debuginfo(prj="ds", info=f'1-param.data={infoTensor(param.data)}')
                # gd.debuginfo(prj="ds", info=f'1-param.grad={infoTensor(param.grad)}')
                # gd.debuginfo(prj="ds", info=f'1-param.grad.data={infoTensor(param.grad.data)}')
                # keeping the gradients contiguous to prevent memory fragmentation, and avoid flattening

                new_grad_tensor = self.ipg_buffer[self.ipg_index].narrow(0, self.elements_in_ipg_bucket, param.numel())
                #gd.debuginfo(prj="ds", info=f'1-new_grad_tensor={infoTensor(new_grad_tensor)}')

                new_grad_tensor.copy_(param.grad.view(-1))
                gd.debuginfo(prj="ds", info=f'2-new_grad_tensor={infoTensor(new_grad_tensor)}')

                param.grad.data = new_grad_tensor.data.view_as(param.grad)
                # gd.debuginfo(prj="ds", info=f'2-param={infoTensor(param)}')
                # gd.debuginfo(prj="ds", info=f'2-param.data={infoTensor(param.data)}')
                # gd.debuginfo(prj="ds", info=f'2-param.grad={infoTensor(param.grad)}')
                # gd.debuginfo(prj="ds", info=f'2-param.grad.data={infoTensor(param.grad.data)}')

        gd.debuginfo(prj="ds", info=f'param_id={param_id}, self.elements_in_ipg_bucket={self.elements_in_ipg_bucket}, param.numel()={param.numel()}')

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

        gd.debuginfo(prj="ds",
                     info=f'self.elements_in_ipg_bucket={self.elements_in_ipg_bucket}, \
                     param.numel()={param.numel()}, \
                     self.reduce_bucket_size={self.reduce_bucket_size} FUNC_OUT')

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
        gd.debuginfo(prj="ds", info=f'tensor={infoTensor(tensor)}, FUNC_IN')
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

        gd.debuginfo(prj="ds", info=f'tensor={infoTensor(tensor)}, FUNC_OUT')
        return tensor

    # 计算张量的平均值 , only call by reduce_ipg_grads
    def average_tensor(self, tensor):
        gd.debuginfo(prj="ds", info=f'tensor={infoTensor(tensor)}, FUNC_IN')
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

        gd.debuginfo(prj="ds", info=f'tensor={infoTensor(tensor)}, FUNC_OUT')

    ##############################################################################
    ############################# CPU Offload Methods#############################
    ##############################################################################
    # 获取梯度位置, only call in init
    def get_grad_position(self, group_id, tensor_list, first_offset, partition_size):
        gd.debuginfo(prj="ds", info=f'FUNC_IN')
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

        gd.debuginfo(prj="ds", info=f'FUNC_OUT')

    # 更新参数梯度的溢出跟踪器, only call in copy_grads_in_partition
    def update_overflow_tracker_for_param_grad(self, param):
        # gd.debuginfo(prj="ds")
        # 如果梯度不为空并且数据中存在无穷或NaN值
        if param.grad is not None and self._has_inf_or_nan(param.grad.data):
            # gd.debuginfo(prj="ds")
            self.local_overflow = True

    # only call in _link_all_hp_params  获取卸载梯度的字典
    def _get_offload_gradient_dict(self):
        gd.debuginfo(prj="ds", info=f'FUNC_IN')
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

        gd.debuginfo(prj="ds", info=f'FUNC_OUT')

    # 通过GPU在CPU上异步累积梯度， only call in copy_grads_in_partition
    def async_accumulate_grad_in_cpu_via_gpu(self, param):
        gd.debuginfo(prj="ds", info=f'FUNC_IN')
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

        gd.debuginfo(prj="ds", info=f'FUNC_OUT')

    # 为参数梯度设置范数  没有被触发？？？？
    def set_norm_for_param_grad(self, param):
        gd.debuginfo(prj="ds", info=f'FUNC_IN')
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

        gd.debuginfo(prj="ds", info=f'FUNC_OUT')

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
        gd.debuginfo(prj="ds", info=f'FUNC_IN')
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

        gd.debuginfo(prj="ds", info=f'FUNC_OUT')

        # 返回总范数
        return total_norm

    ############################################################################################
    # 在分区中复制梯度， only call in reduce_ipg_grads
    def copy_grads_in_partition(self, param):
        gd.debuginfo(prj="ds", info=f'FUNC_IN')

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
            gd.debuginfo(prj="ds", info=f'self.grads_in_partition={self.grads_in_partition}')

            # 打印复制梯度后的内存使用情况 see_memory_usage(f"复制{total_size}个梯度到分区后的内存使用情况")
            see_memory_usage(f"after copying {total_size} gradients into partition")

        # The allreduce buffer will be rewritten. Copy the gradients in partition to a new buffer
        # allreduce缓冲区将被重写，将分区中的梯度复制到新的缓冲区
        new_grad_tensor = self.grads_in_partition.view(-1).narrow(0, self.grads_in_partition_offset, param.numel())
        gd.debuginfo(prj="ds", info=f'new_grad_tensor={infoTensor(new_grad_tensor)}')
        # 使用原始的梯度更新新的梯度tensor
        new_grad_tensor.copy_(param.grad.view(-1))

        # 更新待进行聚合操作的梯度的数据
        param.grad.data = new_grad_tensor.data.view_as(param.grad)

        # 更新分区的梯度偏移量
        #print(f"Grad norm after copy to contiguous_buffer {param.grad.data.norm()}")
        self.grads_in_partition_offset += param.numel()

        gd.debuginfo(prj="ds", info=f'FUNC_OUT')

    # reduce-IPG梯度
    def reduce_ipg_grads(self):
        gd.debuginfo(prj="ds", info=f'FUNC_IN')

        # 如果梯度是连续的
        if self.contiguous_gradients:
            gd.debuginfo(prj="ds")
            # 如果存在超大参数需要进行梯度汇总
            if self.extra_large_param_to_reduce is not None:
                # 确保只有一个参数在 ipg bucket 中，否则会出现问题
                assert len(self.params_in_ipg_bucket) == 1, "more than 1 param in ipg bucket, this shouldn't happen"

                # 获取该参数的id
                a, b, param_id = self.params_in_ipg_bucket[0]

                # 确保 ipg bucket 中的参数和 extra-large 参数匹配
                assert self.get_param_id(self.extra_large_param_to_reduce
                                         ) == param_id, "param in ipg bucket does not match extra-large param"
                gd.debuginfo(prj="ds", info=f'param_id={param_id}, a={a}, b={b}')

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
            stream = self.reduction_stream
            gd.debuginfo(prj="ds", info=f'stream={stream}')
        elif self.cpu_offload:
            #  注意：copy_grad_stream 被禁用了，因为它会和 reduce 产生冲突，这会影响性能，应该修复这个问题
            #  TODO: copy_grad_stream is disabled because of race with reduce. This hurts perf and should be fixed.
            #            get_accelerator().synchronize()
            #            stream = self.copy_grad_stream
            stream = get_accelerator().current_stream()
            gd.debuginfo(prj="ds", info=f'stream={stream}')
        else:
            stream = get_accelerator().current_stream()
            gd.debuginfo(prj="ds", info=f'stream={stream}')

        # 在选定的 stream 中执行以下操作
        with get_accelerator().stream(stream):
            for tmp, param, param_id in self.params_in_ipg_bucket:
                gd.debuginfo(prj="ds", info=f'param_id={param_id}, param={infoTensor(param)}, tmp={tmp}')
                # 确保该参数没有被汇总过，因为当前不支持多次梯度汇总
                assert self.params_already_reduced[param_id] == False, \
                    f"The parameter {param_id} has already been reduced. \
                    Gradient computed twice for this partition. \
                    Multiple gradient reduction is currently not supported"
                # 标记该参数已经被汇总
                self.params_already_reduced[param_id] = True

                # 如果需要对梯度进行分区
                if self.partition_gradients:
                    gd.debuginfo(prj="ds")
                    if not self.is_param_in_current_partition[param_id]:
                        gd.debuginfo(prj="ds")
                        if self.overlap_comm and self.contiguous_gradients is False:
                            gd.debuginfo(prj="ds")
                            # 在下一次梯度汇总过程中清空其他分区的梯度
                            # 这样可以避免在汇总完成之前就清空他们
                            # Clear grads of other partitions during the next reduction
                            # to avoid clearing them before the reduction is complete.
                            if self.previous_reduced_grads is None:
                                gd.debuginfo(prj="ds")
                                self.previous_reduced_grads = []
                            self.previous_reduced_grads.append(param)
                        else:
                            gd.debuginfo(prj="ds")
                            # 清空该参数的梯度属性
                            param.grad = None  #only if self.partition_gradients
                    elif self.contiguous_gradients:
                        gd.debuginfo(prj="ds")
                        # 如果梯度是连续的，复制当前分区的梯度
                        self.copy_grads_in_partition(param)
                else:
                    # zero stage 1 - 只分区优化器状态
                    # zero stage 1 - partition only optimizer state
                    if self.contiguous_gradients and self.is_param_in_current_partition[param_id]:
                        # 如果梯度是连续的，复制当前分区的梯度
                        gd.debuginfo(prj="ds")
                        self.copy_grads_in_partition(param)

        # 清空 ipg_bucket 和相关信息
        self.grads_in_ipg_bucket = []
        self.params_in_ipg_bucket = []
        self.ipg_bucket_has_moe_params = False
        self.elements_in_ipg_bucket = 0

        gd.debuginfo(prj="ds", info=f'FUNC_OUT')
        #####################################################################

    # 减少已准备好的分区并删除梯度  #这是hook，由backward触发！
    def reduce_ready_partitions_and_remove_grads(self, param, i):
        gd.debuginfo(prj="ds", info=f'i={i}, param={infoTensor(param)},  FUNC_IN')
        # 如果满足以下两个条件之一，执行操作：
        # 1. 需要对梯度进行分区处理
        # 2. 当前处于梯度累积的边界
        # 该操作包括：
        # 对独立的参数梯度分区桶进行归约，并移除梯度
        if self.partition_gradients or self.is_gradient_accumulation_boundary:
            gd.debuginfo(prj="ds")
            self.reduce_independent_p_g_buckets_and_remove_grads(param, i)

        gd.debuginfo(prj="ds", info=f'i={i}, param={infoTensor(param)},  FUNC_OUT')

    # 将减少的梯度设置为零
    def zero_reduced_gradients(self, partition_id, i):
        gd.debuginfo(prj="ds", info=f'FUNC_IN')

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

        gd.debuginfo(prj="ds", info=f'FUNC_OUT')

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
        gd.debuginfo(prj="ds", info=f'FUNC_IN')

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

        gd.debuginfo(prj="ds", info=f'FUNC_OUT')

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

        # 待进行allreduce操作的tensor
        tensor_to_allreduce = tensor

        # 在进行pg_correctness_test的情况下，通信数据类型设置为float32
        # 否则，使用预设的通信数据类型
        if pg_correctness_test:
            gd.debuginfo(prj="ds")
            communication_data_type = torch.float32
        else:
            gd.debuginfo(prj="ds")
            communication_data_type = self.communication_data_type

        # 如果通信数据类型与tensor的数据类型不一致，将tensor转为通信数据类型
        if communication_data_type != tensor.dtype:
            gd.debuginfo(prj="ds")
            tensor_to_allreduce = tensor.to(communication_data_type)

        # 将待allreduce的tensor除以进程组的大小，以进行平均操作
        tensor_to_allreduce.div_(dist.get_world_size(group=self.dp_process_group))

        if rank is None:
            # 执行allreduce操作，所有进程共享数据
            gd.debuginfo(prj="ds")
            #    "All Reducing"
            dist.all_reduce(tensor_to_allreduce, group=self.dp_process_group)
        else:
            gd.debuginfo(prj="ds")
            # 获取全局rank
            global_rank = dist.get_global_rank(self.dp_process_group, rank)

            # 执行reduce操作，将数据发送到指定的进程
            dist.reduce(tensor_to_allreduce, global_rank, group=self.dp_process_group)

        # 如果通信数据类型与tensor的数据类型不一致，并且tensor不等于tensor_to_allreduce
        # 在rank为None或者等于当前进程的rank的情况下，将tensor_to_allreduce的值复制给tensor
        if communication_data_type != tensor.dtype and tensor is not tensor_to_allreduce:
            if rank is None or rank == dist.get_rank(group=self.dp_process_group):
                tensor.copy_(tensor_to_allreduce)

        # 返回处理后的tensor
        return tensor

    # 清除先前reduce的梯度
    def _clear_previous_reduced_grads(self):
        gd.debuginfo(prj="ds")
        # 如果之前的梯度不为None，即之前计算过梯度
        if self.previous_reduced_grads is not None:
            # 遍历每一个之前计算的梯度
            for param in self.previous_reduced_grads:
                # 清除对应的梯度信息
                param.grad = None  # overlap enabled

            # 清空之前的梯度列表，准备下一次计算
            self.previous_reduced_grads = None

    # allreduce操作和复制
    # if rank is specified do a reduction instead of an allreduce
    def allreduce_and_copy(self, small_bucket, rank=None, log=None):
        # 如果启用了overlap_comm，进行通信和计算的重叠
        if self.overlap_comm:
            gd.debuginfo(prj="ds")
            # 确保所有设备上的运行都已完成
            get_accelerator().synchronize()

            # It is safe to clear the previously reduced grads of other partitions
            # 清除其他分区之前reduce的梯度，这是安全的
            self._clear_previous_reduced_grads()

            # 使用专门的流进行reduce操作
            stream = self.reduction_stream
        else:
            # 如果没有启用overlap_comm，使用当前设备的当前流
            gd.debuginfo(prj="ds")
            stream = get_accelerator().current_stream()

        # 使用指定的流
        with get_accelerator().stream(stream):
            # 对small_bucket进行allreduce操作，然后返回结果
            allreduced = self.allreduce_bucket(small_bucket, rank=rank, log=log)

            # 如果rank是None（即没有指定），或者rank等于当前进程的rank
            # 就对small_bucket中的每个buf和对应的synced进行copy操作
            if rank is None or rank == dist.get_rank(group=self.dp_process_group):
                for buf, synced in zip(small_bucket, self.unflatten(allreduced, small_bucket)):
                    # 将synced的内容复制到buf中
                    buf.copy_(synced)

    # allreduce且不保留  only call by buffered_reduce_fallback
    def allreduce_no_retain(self, bucket, numel_per_bucket=500000000, rank=None, log=None):
        gd.debuginfo(prj="ds")
        # 初始化一个空的小桶
        small_bucket = []

        # 初始化元素数量为0
        numel = 0

        # 对bucket中的每一个tensor进行遍历
        for tensor in bucket:
            # 将当前tensor添加到小桶中
            small_bucket.append(tensor)

            # 计算小桶中所有tensor的元素总数
            numel = numel + tensor.numel()

            # 如果小桶中的元素总数超过了设定的阈值，那么就对小桶进行allreduce_and_copy操作
            # 然后清空小桶，为接下来的tensors做准备
            if numel > numel_per_bucket:
                self.allreduce_and_copy(small_bucket, rank=rank, log=None)
                small_bucket = []

        # 如果bucket中的所有tensors都已经被处理完，但是小桶中还剩下一些tensors没有被处理
        # 就对这些剩下的tensors进行allreduce_and_copy操作
        if len(small_bucket) > 0:
            self.allreduce_and_copy(small_bucket, rank=rank, log=log)

    # allows using reduction of gradients instead of using all_reduce
    # 缓冲reduce，fallback那个钩子（也许？fallback，回调，也许在某个地方调用了）
    def buffered_reduce_fallback(self, rank, grads, elements_per_buffer=500000000, log=None):
        gd.debuginfo(prj="ds")
        # 将 grads 分为半精度浮点数和双精度浮点数两部分
        split_buckets = split_half_float_double(grads)

        # 遍历每一个 bucket
        for i, bucket in enumerate(split_buckets):
            # 进行全局归约操作，这是一种并行算法，用于在所有进程中累加每个进程的输入值
            # 并将结果返回给所有进程
            self.allreduce_no_retain(bucket, numel_per_bucket=elements_per_buffer, rank=rank, log=log)

    #############################################################################
    #############################################################################
    #############################################################################

    # views the tensor as multiple partitions and returns
    # those partitions  获取数据并行分区
    def get_data_parallel_partitions(self, tensor, group_id):
        gd.debuginfo(prj="ds",info=f'group_id={group_id}, FUNC_IN')
        # 初始化一个空列表，用于存储每个分区的数据
        partitions = []

        # 获取分布式处理（dp）的大小，即有多少个处理单元参与分布式计算,也就是显卡数量
        dp = dist.get_world_size(group=self.real_dp_process_group[group_id])  # 单机就是GPU数量
        dp_id = dist.get_rank(group=self.real_dp_process_group[group_id])     # 没有到，打出来看看

        # 计算张量（tensor）中的总元素数量
        total_num_elements = tensor.numel()

        # 计算每个处理单元应分配的基本元素数量，这里使用了整数除法
        base_size = total_num_elements // dp

        # 计算不能被均匀分配的剩余元素数量
        remaining = total_num_elements % dp
        # 初始化起始索引为0，这个索引表示当前处理的张量部分的起始位置
        start = 0
        gd.debuginfo(prj="ds", info=f'dp={dp}, dp_id={dp_id}, '
                                    f'total_num_elements={total_num_elements}, '
                                    f'base_size={base_size}, '
                                    f'remaining={remaining}')
        # dp=2, dp_id=0, total_num_elements=125137152, base_size=62568576, remaining=0
        # dp=3, dp_id=0, total_num_elements=125137152, base_size=41712384, remaining=0
        # 遍历每个处理单元
        for id in range(dp):
            # 默认每个处理单元分配的元素数量为base_size
            partition_size = base_size

            if id < remaining:
                # 如果当前处理单元的id小于剩余元素数量remaining，那么就给这个处理单元多分配一个元素
                partition_size = partition_size + 1

            # 使用narrow函数从张量中抽出一部分数据，0表示要操作的维度（这里是第一维），
            # start表示开始的索引，partition_size表示长度
            # 抽出的部分数据作为一个分区，添加到partitions列表中
            tmp=tensor.narrow(0, start, partition_size) # https://pytorch.org/docs/stable/generated/torch.narrow.html
            partitions.append(tmp)
            gd.debuginfo(prj="ds", info=f'tensor={infoTensor(tensor)},'
                                        f'tmp={infoTensor(tmp)}')

            # 更新开始索引，准备处理下一个分区
            start = start + partition_size
            gd.debuginfo(prj="ds", info=f'id={id}, '
                                        f'base_size={base_size}, '
                                        f'partition_size={partition_size}, '
                                        f'start={start}')
        gd.debuginfo(prj="ds", info=f'group_id={group_id}, FUNC_OUT')
        # 返回分区列表，列表中的每个元素都是一个张量，表示一个分区的数据
        return partitions

    # 获取分区信息 only call in __init__
    def get_partition_info(self, tensor_list, partition_size, partition_id):
        gd.debuginfo(prj="ds")
        # 初始化两个列表，用于存储在分区内的参数和不在分区内的参数
        params_in_partition = []
        params_not_in_partition = []

        # 计算分区的起始和结束索引
        start_index = partition_size * partition_id
        end_index = partition_size * (partition_id + 1)

        # 初始化当前索引和第一个偏移值
        current_index = 0
        first_offset = 0

        # 遍历tensor列表
        for tensor in tensor_list:
            # 获取tensor的元素数量
            tensor_size = tensor.numel()

            # 如果当前索引在分区的范围内，将tensor添加到params_in_partition列表中
            if start_index <= current_index < end_index:
                params_in_partition.append(tensor)

            elif current_index < start_index < (current_index + tensor_size):
                # 如果当前索引小于分区起始索引，且分区起始索引在tensor范围内，将tensor添加到params_in_partition列表中
                params_in_partition.append(tensor)

                # 确保first_offset只被设置一次，因为这必须是分区中的第一个tensor
                assert (first_offset == 0
                        ), "This can happen either zero or only once as this must be the first tensor in the partition"
                first_offset = start_index - current_index

            else:
                # 否则，将tensor添加到params_not_in_partition列表中
                params_not_in_partition.append(tensor)

            # 更新当前索引
            current_index = current_index + tensor_size

        # 返回在分区内的参数、不在分区内的参数以及第一个偏移值
        return params_in_partition, params_not_in_partition, first_offset

    # 将所有模型参数的梯度设置为零
    def zero_grad(self, set_to_none=False):
        gd.debuginfo(prj="ds")
        """
        Zero FP16 parameter grads.
        清零 FP16 参数的梯度。
    
        # FP32 的梯度永远不应该存在。
        # 出于速度的考虑，默认情况下将模型 FP16 的梯度设置为 None
        # 清零所有指向梯度张量的指针
        """
        # FP32 grad should never exist.
        # For speed, set model fp16 grad to None by default
        for group in self.bit16_groups:
            for p in group:
                if set_to_none:
                    p.grad = None  # epilogue and in step  # 在尾声和步骤中
                else:
                    if p.grad is not None:
                        p.grad.detach_()  # 分离梯度
                        p.grad.zero_()  # 清零梯度

    # 执行模型并行的allreduce操作
    def _model_parallel_all_reduce(self, tensor, op):
        """ Perform all reduce within model parallel group, if any.
        在模型并行组内执行allreduce操作，如果有的话
        """
        # gd.debuginfo(prj="ds")

        if self.model_parallel_group is None or self.model_parallel_world_size == 1:
            pass # 如果模型并行组不存在或模型并行世界大小等于1 不执行任何操作
        else:
            # gd.debuginfo(prj="ds")
            # 否则，使用 dist.all_reduce 函数在模型并行组内对 tensor 进行allreduce操作
            dist.all_reduce(tensor=tensor, op=op, group=self.model_parallel_group)

    # 直接获取梯度的范数（L2范数）
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

        计算并裁剪参数的梯度范数。

        本函数是从torch.nn.utils.clip_grad.clip_grad_norm_中调整而来，
        其中加入了处理模型并行参数的功能。注意，梯度将在原地被修改。

        参数:
            gradients (Iterable[Tensor] or Tensor): 需要进行梯度范数计算的张量的迭代器或单个张量
            params (Iterable[Tensor] or Tensor): 需要进行梯度范数计算的参数的迭代器或单个张量
            norm_type (float or int): 使用的p-范数类型。可以是 ``'inf'``表示无穷范数。

        返回:
            参数的总体范数（视为单个向量）。

        """
        norm_type = float(norm_type)
        if norm_type == inf:
            # 找到所有梯度中绝对值最大的
            total_norm = max(g.data.abs().max() for g in gradients)
            gd.debuginfo(prj="ds", info=f"1-total_norm={total_norm}")

            # 将梯度的最大值转为张量
            total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)])
            gd.debuginfo(prj="ds", info=f"1-total_norm_cuda={infoTensor(total_norm_cuda)}")

            # 使用all_reduce进行跨设备的梯度同步，取最大值
            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.MAX, group=self.dp_process_group)

            # Take max across all GPUs.
            # 在所有GPUs之间取最大值。
            self._model_parallel_all_reduce(tensor=total_norm_cuda, op=dist.ReduceOp.MAX)
            total_norm = total_norm_cuda[0].item()
            gd.debuginfo(prj="ds", info=f"2-total_norm={total_norm}")
        else:
            total_norm = 0.0
            if dist.get_rank() == 0:
               gd.debuginfo(prj="ds", info=f"Total Norm beginning {total_norm}")

            for g, p in zip(gradients, params):
                # 管道并行化可能会复制参数。避免多次计数。
                # Pipeline parallelism may replicate parameters. Avoid multi-counting.
                if hasattr(p, PIPE_REPLICATED) and p.ds_pipe_replicated:
                    continue

                # 如果参数是模型并行的参数，或者当前设备是主设备，则计算该参数的梯度范数
                if is_model_parallel_parameter(p) or (self.model_parallel_rank == 0):
                    param_norm = g.data.double().norm(2)
                    total_norm += param_norm.item()**2

            gd.debuginfo(prj="ds", info=f"2-total_norm={total_norm}")

            # 将梯度的平方和转为张量
            # Sum across all model parallel GPUs.
            total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)])
            gd.debuginfo(prj="ds", info=f"2-total_norm_cuda={infoTensor(total_norm_cuda)}")

            # 使用all_reduce进行跨设备的梯度同步，取和
            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.SUM, group=self.dp_process_group)

            # 在所有模型并行的GPUs之间进行求和
            self._model_parallel_all_reduce(tensor=total_norm_cuda, op=dist.ReduceOp.SUM)

            # 计算梯度的总体范数
            total_norm = total_norm_cuda[0].item()**(1. / norm_type)
            gd.debuginfo(prj="ds", info=f"5-total_norm={total_norm}")

        # 如果总体范数是无穷大，负无穷大，或不是一个数，则将其设置为-1
        if total_norm == float('inf') or total_norm == -float('inf') or total_norm != total_norm:
            gd.debuginfo(prj="ds", info=f"total_norm set to -1")
            total_norm = -1

        return total_norm

    # creates a flat fused tensor from the tensor list starting at the first_offset
    # in the first tensor of the list. If there are not enough elements in the tensor
    # list then the flat tensor will be padded with zeros
    # 获取拉平的分区
    def get_flat_partition(self, tensor_list, first_offset, partition_size, dtype, device, return_tensor_list=False):
        gd.debuginfo(prj="ds")
        # 初始化一个空列表，用于存储处理后的tensor
        flat_tensor_list = []

        # 记录当前已处理tensor的元素数量
        current_size = 0

        # 遍历每个传入的tensor
        for i, tensor in enumerate(tensor_list):
            gd.debuginfo(prj="ds", info=f'tensor={infoTensor(tensor)}')

            # 如果没有梯度属性，用0填充
            if tensor.grad is None:
                tensor.grad = torch.zeros_like(tensor)

            # 将梯度属性作为处理的目标tensor
            tensor = tensor.grad

            # 获取tensor的元素数量
            num_elements = tensor.numel()
            tensor_offset = 0

            # 对于列表中的第一个tensor，如果有偏移量，根据偏移量调整元素数量和偏移值
            # we need to offset to get to the right element
            if i == 0 and first_offset > 0:
                tensor_offset = first_offset
                num_elements = num_elements - tensor_offset

            # we dont need all elements of the tensor
            # 如果当前tensor的元素数量超过了分区大小，调整元素数量以适应分区
            if num_elements > (partition_size - current_size):
                num_elements = partition_size - current_size
            gd.debuginfo(prj="ds", info=f'tensor_offset={tensor_offset}, num_elements={num_elements}')

            # we need a narrow view of the tensor based on the tensor offset and number of elements that
            # we need from this tensor

            # 根据偏移量和元素数量，获取tensor的视图并添加到列表
            if tensor_offset > 0 or num_elements < tensor.numel():
                tmp = tensor.contiguous().view(-1).narrow(0, int(tensor_offset), int(num_elements))
                gd.debuginfo(prj="ds", info=f'tmp={infoTensor(tmp)}')
                flat_tensor_list.append(tmp)
            else:
                flat_tensor_list.append(tensor)

            # 更新当前已处理的元素数量
            current_size = current_size + num_elements

        # this means its the last partition and does not align with the dp boundary. We need to pad before flattening
        # 如果当前处理的元素数量小于分区大小，需要填充0
        if current_size < partition_size:
            tmp = torch.zeros(int(partition_size - current_size), dtype=dtype, device=device)
            gd.debuginfo(prj="ds", info=f'tmp={infoTensor(tmp)}')
            flat_tensor_list.append(tmp)

        # 如果需要返回tensor列表，直接返回
        if return_tensor_list:
            gd.debuginfo(prj="ds")
            return flat_tensor_list

        # 否则，返回压平后的单个tensor
        return self.flatten(flat_tensor_list)

    # 释放参数列表中的梯度
    def free_grad_in_param_list(self, param_list):
        gd.debuginfo(prj="ds")
        # 遍历参数列表中的每一个参数
        for p in param_list:
            p.grad = None  # in step  # 将参数的梯度设置为None，表示清除该参数的梯度

    # 重置CPU缓冲区
    def reset_cpu_buffers(self):
        gd.debuginfo(prj="ds")
        self.norm_for_param_grads = {}
        # 用于标识是否发生了本地溢出
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

    # 设置学习率
    def set_lr(self, lr):
        gd.debuginfo(prj="ds")
        """Set the learning rate."""
        # 遍历优化器中的所有参数组
        for param_group in self.optimizer.param_groups:
            # 对每个参数组设置新的学习率
            param_group["lr"] = lr

    # 获取当前的学习率
    def get_lr(self):
        gd.debuginfo(prj="ds")
        """Return the current learning rate."""
        # 这个函数的目的是返回当前的学习率。
        # 从优化器的参数组中获取第一个参数组的学习率
        return self.optimizer.param_groups[0]["lr"]

    # 覆盖损失比例
    def override_loss_scale(self, loss_scale):
        gd.debuginfo(prj="ds")
        # 如果给定的loss_scale不等于当前的外部loss_scale，我们需要更新它
        if loss_scale != self.external_loss_scale:
            # 打印一条信息，说明正在从原来的外部loss_scale更新到新的loss_scale
            gd.debuginfo(prj="ds", info=f'[deepspeed] setting loss scale from {self.external_loss_scale} -> {loss_scale}')

        # 设置custom_loss_scaler为True，表示我们正在使用自定义的loss_scaler
        self.custom_loss_scaler = True

        # 更新外部loss_scale为给定的loss_scale
        self.external_loss_scale = loss_scale

    # 计算缩放全局范数
    def scaled_global_norm(self, norm_type=2):
        gd.debuginfo(prj="ds")
        # 断言：仅支持L2范数
        assert norm_type == 2, "only L2 norm supported"

        # 初始化范数组
        norm_groups = []

        # 遍历位组
        for i, group in enumerate(self.bit16_groups):
            # 获取分区ID
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])

            # 如果进行了CPU卸载
            if self.cpu_offload:
                # 计算并添加对应的梯度范数
                norm_groups.append(self.complete_grad_norm_calculation_for_cpu_offload(self.params_in_partition[i]))
                # 获取单个分区的梯度
                single_grad_partition = self.single_partition_of_fp32_groups[i].grad
            else:
                # 直接获取并添加对应的梯度范数
                norm_groups.append(self.get_grad_norm_direct(self.averaged_gradients[i], self.params_in_partition[i]))

        # 如果存在moe层
        if self.has_moe_layers:
            gd.debuginfo(prj="ds")
            # 平均专家梯度范数
            self._average_expert_grad_norms(norm_groups)

        # 注意，get_global_norm函数仅支持l2范数
        # 获取全局范数并返回
        # note that the get_global_norm function only supports l2 norm
        return get_global_norm(norm_list=norm_groups)

    # 获取16位参数组
    def get_bit16_param_group(self, group_no):
        gd.debuginfo(prj="ds")
        # 获取16位参数组中的某个组的分区
        bit16_partitions = self.parallel_partitioned_bit16_groups[group_no]

        # 获取当前分区ID，这是根据分布式处理组中的排名来获取的
        partition_id = dist.get_rank(group=self.real_dp_process_group[group_no])

        # 返回对应分区ID的参数组
        return [bit16_partitions[dist.get_rank(group=self.real_dp_process_group[group_no])]]

    # 执行优化器的步骤
    def _optimizer_step(self, group_no):
        gd.debuginfo(prj="ds")
        # 获取优化器的参数组
        original_param_groups = self.optimizer.param_groups

        # 将优化器的参数组设置为特定的参数组
        self.optimizer.param_groups = [original_param_groups[group_no]]

        # 这段代码被禁用，因为C++端的复制和同步功能无法正确工作
        # Disabling this as the C++ side copy & synchronize is not working correctly
        #from deepspeed.ops.adam import DeepSpeedCPUAdam
        #if type(self.optimizer) == DeepSpeedCPUAdam and self.dtype == torch.half:
        #    self.optimizer.step(fp16_param_groups=[self.get_bit16_param_group(group_no)])
        #else:
        #    self.optimizer.step()

        # 执行优化器的步进
        self.optimizer.step()

        # 将优化器的参数组还原为原始参数组
        self.optimizer.param_groups = original_param_groups

    # 优化器的step步骤，真正执行的方法体
    def step(self, closure=None):
        gd.debuginfo(prj="ds")
        """
        Not supporting closure.  不支持闭包。
        """
        self.micro_step_id = -1

        # 在检查溢出之前查看内存使用情况
        see_memory_usage(f"In step before checking overflow")

        # First compute norm for all group so we know if there is overflow
        # 首先计算所有组的规范，以便我们知道是否有溢出
        self.check_overflow()
        OPTIMIZER_ALLGATHER = 'optimizer_allgather'
        OPTIMIZER_GRADIENTS = 'optimizer_gradients'
        OPTIMIZER_STEP = 'optimizer_step'
        timer_names = [OPTIMIZER_ALLGATHER, OPTIMIZER_GRADIENTS, OPTIMIZER_STEP]

        prev_scale = self.loss_scale

        # 更新规模
        self._update_scale(self.overflow)
        if self.overflow:
            # 溢出后，清除梯度之前，查看内存使用情况
            see_memory_usage('After overflow before clearing gradients')

            # 清除梯度
            self.zero_grad(set_to_none=True)
            if self.cpu_offload:
                gd.debuginfo(prj="ds")
                # 重置CPU缓冲区
                self.reset_cpu_buffers()
            else:
                gd.debuginfo(prj="ds")
                self.averaged_gradients = {}

            # 溢出后，清除梯度之后，查看内存使用情况
            see_memory_usage('After overflow after clearing gradients')

            self.start_timers(timer_names)
            self.stop_timers(timer_names)
            return

        # Step 1:- Calculate gradient norm using bit-16 grads
        # 步骤 1：- 使用 bit-16 梯度计算梯度规范

        # 在计算规范之前查看内存使用情况
        see_memory_usage('Before norm calculation')

        scaled_global_grad_norm = self.scaled_global_norm()
        self._global_grad_norm = scaled_global_grad_norm / prev_scale

        # 在优化器之前，规范后查看内存使用情况
        see_memory_usage('After norm before optimizer')

        # Step 2:- run optimizer and upscaling simultaneously
        # 步骤 2：- 同时运行优化器和上升规模
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
                # 释放所有这个进程不更新的参数的梯度（ZeRO stage2）
                # free gradients for all the parameters that are not updated by this process(ZeRO stage2)
                self.free_grad_in_param_list(self.params_not_in_partition[i])

                # 创建一个为这个进程更新的参数的平坦梯度
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
                # 释放所有梯度，因为我们已经在dp_grad_partition中创建了必要的副本（ZeRO stage2）
                self.free_grad_in_param_list(self.params_in_partition[i])

                self.averaged_gradients[i] = None

                self.unscale_and_clip_grads([single_grad_partition], scaled_global_grad_norm)

                self.stop_timers([OPTIMIZER_GRADIENTS])

                # Step 3:- run the optimizer if no offloading
                # 步骤 3：- 运行优化器（如果没有卸载）
                self.start_timers([OPTIMIZER_STEP])
                self._optimizer_step(i)

                # Step 4:- get rid of the fp32 gradients. Not needed anymore
                # 第四步：去掉 fp32 梯度，不再需要它们
                self.single_partition_of_fp32_groups[i].grad = None
                del single_grad_partition
                bit16_partitions = self.parallel_partitioned_bit16_groups[i]
                fp32_partition = self.single_partition_of_fp32_groups[i]

                # 将fp32分区的数据拷贝到bit16分区
                bit16_partitions[partition_id].data.copy_(fp32_partition.data)

                # 停止定时器
                self.stop_timers([OPTIMIZER_STEP])

        # 查看内存使用情况
        see_memory_usage('After optimizer before all-gather')

        # 如果开启了CPU offload，重置CPU buffer
        if self.cpu_offload:
            gd.debuginfo(prj="ds")
            self.reset_cpu_buffers()

        # 启动定时器
        self.start_timers([OPTIMIZER_ALLGATHER])

        # Gather the updated weights from everyone.
        # Then all partitions of the model parameters are updated and ready for next round forward.
        # 从每个节点收集更新后的权重。
        # 然后，所有分区的模型参数都更新完成，准备好进行下一轮的前向传播。
        all_gather_dp_groups(partitioned_param_groups=self.parallel_partitioned_bit16_groups,
                             dp_process_group=self.real_dp_process_group,
                             start_alignment_factor=self.nccl_start_alignment_factor,
                             allgather_bucket_size=self.allgather_bucket_size)

        # 停止定时器
        self.stop_timers([OPTIMIZER_ALLGATHER])

        # TODO: we probably don't need this? just to be safe
        # 循环更新模型的bit16权重（虽然可能并不需要，但为了保险起见）
        for i in range(len(self.bit16_groups)):
            self._update_model_bit16_weights(i)

        # 日志记录优化器的定时信息
        self.log_timers(timer_names)

        see_memory_usage('After zero_optimizer step')

        return

    # 更新LP参数
    @torch.no_grad()
    def update_lp_params(self):
        gd.debuginfo(prj="ds")
        # 遍历bit16和fp32的分区组
        for i, (bit16_partitions, fp32_partition) in enumerate(
                zip(self.parallel_partitioned_bit16_groups, self.single_partition_of_fp32_groups)):
            # 获取当前分区的ID
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])

            # 将fp32分区的数据复制到相应的bit16分区中
            bit16_partitions[partition_id].data.copy_(fp32_partition.data)

            # print_rank_0(f'update_lp_params {i=} {partition_id=}', force=True)
            # if i == 0:
            #     print_rank_0(f'{fp32_partition[:10]=}', force=True)

        # 将所有分区的数据集合在一起，以便在分布式处理过程中使用
        all_gather_dp_groups(partitioned_param_groups=self.parallel_partitioned_bit16_groups,
                             dp_process_group=self.real_dp_process_group,
                             start_alignment_factor=self.nccl_start_alignment_factor,
                             allgather_bucket_size=self.allgather_bucket_size)

    # 平均专家梯度范数
    def _average_expert_grad_norms(self, norm_groups):
        gd.debuginfo(prj="ds")
        # 遍历norm_groups中的每个元素，并获取其索引和值
        for i, norm in enumerate(norm_groups):
            # 如果当前索引对应的参数组是moe参数组
            if self.is_moe_param_group[i]:
                # 计算规模化的梯度norm，这是通过对原始norm除以分布式进程组的大小来实现的
                scaled_norm = norm * 1.0 / float(dist.get_world_size(group=self.real_dp_process_group[i]))

                # 将规模化的norm转换为tensor格式，并确保其在正确的设备上，并且具有正确的数据类型
                scaled_norm_tensor = torch.tensor(scaled_norm,
                                                  device=get_accelerator().device_name(),
                                                  dtype=torch.float)

                # 使用all_reduce进行跨所有设备的归约操作，所有设备上的scaled_norm_tensor将会被加起来
                dist.all_reduce(scaled_norm_tensor, group=self.real_dp_process_group[i])

                # 将归约后的结果（一个tensor）转换为Python数值，并存回norm_groups中对应的位置
                norm_groups[i] = scaled_norm_tensor.item()

    # 取消缩放并裁剪梯度
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

    # 检查是否有溢出
    def _check_overflow(self, partition_gradients=True):
        # gd.debuginfo(prj="ds")
        # 这个方法用于检查是否有溢出
        # partition_gradients 参数决定是否在不同的设备上分割梯度，默认为True
        # has_overflow 是一个方法，检查是否有任何梯度超出了表示范围，如果有，返回 True
        self.overflow = self.has_overflow(partition_gradients)

    # `params` is a list / generator of torch.Variable
    # 检查是否串行溢出
    def has_overflow_serial(self, params, is_grad_list=False):
        # gd.debuginfo(prj="ds")
        # 遍历传入的参数
        for p in params:
            # 如果参数的梯度不为None，且该梯度的数据包含无穷大或者NaN值
            if p.grad is not None and self._has_inf_or_nan(p.grad.data):
                # 返回True，表示有溢出
                return True

        # 如果遍历所有参数都没有发现溢出，返回False
        return False

    # 检查分区的梯度是否串行溢出
    def has_overflow_partitioned_grads_serial(self):
        gd.debuginfo(prj="ds")
        # 遍历每一个16位组
        for i in range(len(self.bit16_groups)):
            # 在每个组内遍历每一个平均梯度
            for j, grad in enumerate(self.averaged_gradients[i]):
                # 如果当前梯度不为空，并且梯度数据中存在无穷大或者NaN（不是一个数字）
                # 则返回True，表示存在溢出的分区梯度
                if grad is not None and self._has_inf_or_nan(grad.data, j):
                    return True

        # 如果所有的梯度都被检查过，并且没有发现溢出的情况，那么返回False
        return False

    # 检查是否溢出
    def has_overflow(self, partition_gradients=True):
        # 如果分区梯度为真
        if partition_gradients:
            gd.debuginfo(prj="ds")
            # 如果执行CPU offload，则获取本地溢出，否则获取分区梯度的溢出
            overflow = self.local_overflow if self.cpu_offload else self.has_overflow_partitioned_grads_serial()

            # 将溢出值转换为GPU可处理的字节张量
            overflow_gpu = get_accelerator().ByteTensor([overflow])

            '''
            This will capture overflow across all data parallel and expert parallel process
            Since expert parallel process are a subset of data parallel process
            这将捕获所有数据并行和专家并行过程中的溢出
            由于专家并行过程是数据并行过程的子集
            '''
            # 将溢出值在数据并行过程组中进行归约处理，获取最大值
            dist.all_reduce(overflow_gpu, op=dist.ReduceOp.MAX, group=self.dp_process_group)

        else:
            gd.debuginfo(prj="ds")
            # 初始化参数列表
            params = []

            # 对于每一个16位组
            for group in self.bit16_groups:
                # 对于组中的每一个参数
                for param in group:
                    # 将参数添加到参数列表中
                    params.append(param)

            # 获取参数列表中的溢出值
            overflow = self.has_overflow_serial(params, is_grad_list=partition_gradients)

            # 将溢出值转换为GPU可处理的字节张量
            overflow_gpu = get_accelerator().ByteTensor([overflow])


        # Since each model parallel GPU carries only part of the model,
        # make sure overflow flag is synced across all the model parallel GPUs
        # 由于每个模型并行的GPU只携带模型的一部分
        # 确保溢出标志在所有的模型并行GPU中同步
        self._model_parallel_all_reduce(tensor=overflow_gpu, op=dist.ReduceOp.MAX)

        # 获取字节张量中的溢出值
        overflow = overflow_gpu[0].item()

        # 返回溢出值的布尔值，如果溢出则返回True，否则返回False
        return bool(overflow)

    # `x` is a torch.Tensor  #检查是否有无穷大或NaN
    @staticmethod
    def _has_inf_or_nan(x, j=None):
        try:
            gd.debuginfo(prj="ds")
            # 如果 x 是半精度浮点数(half)，.float()会引发额外的深拷贝，但是如果
            # Pytorch的 .sum() 创建一个与x类型相同的单元素张量
            # （对于一些最近版本的pytorch是这样的），那么这个操作就是必要的。

            # if x is half, the .float() incurs an additional deep copy, but it's necessary if
            # Pytorch's .sum() creates a one-element tensor of the same type as x
            # (which is true for some recent version of pytorch).

            # 如果 .sum() 返回一个Python标量，可以使用更高效的版本
            # cpu_sum = float(x.sum())
            cpu_sum = float(x.float().sum())

            # More efficient version that can be used if .sum() returns a Python scalar
            # cpu_sum = float(x.sum())
        except RuntimeError as instance:
            gd.debuginfo(prj="ds")
            # We want to check if inst is actually an overflow exception.
            # RuntimeError could come from a different error.
            # If so, we still want the exception to propagate.

            # 我们想要检查这个异常是否真的是溢出异常。
            # RuntimeError 可能来自不同的错误。
            # 如果是这样，我们仍然希望异常能够传播。

            if "value cannot be converted" not in instance.args[0]:
                raise
            return True
        else:
            # 如果 cpu_sum 是正无穷、负无穷或者不是一个数字（NaN），
            # 那么我们就返回True，意味着x中含有无穷或者NaN
            gd.debuginfo(prj="ds")
            if cpu_sum == float('inf') or cpu_sum == -float('inf') or cpu_sum != cpu_sum:
                gd.debuginfo(prj="ds")
                return True

            # 否则我们返回False，意味着x中没有无穷或者NaN
            return False

    # 执行反向传播      ph1-z2 train1batch 入口  只有backward调用z2优化器
    def backward(self, loss, retain_graph=False):
        gd.debuginfo(prj='ds', info=f'C:{self.__class__.__name__} FUNC_IN, loss={infoTensor(loss)}')
        """
        :attr:`backward` 执行以下步骤:

        1. fp32_loss = loss.float()  # 将损失转换为浮点类型
        2. scaled_loss = fp32_loss*loss_scale  # 对损失进行缩放
        3. scaled_loss.backward()  # 对缩放后的损失进行反向传播，这会将缩放的梯度累积到模型的 fp16 叶子节点的 ``.grad`` 属性中

        :attr:`backward` performs the following steps:

        1. fp32_loss = loss.float()
        2. scaled_loss = fp32_loss*loss_scale
        3. scaled_loss.backward(), which accumulates scaled gradients into the ``.grad`` attributes of the model's fp16 leaves
        """
        # 微步进ID增加1
        self.micro_step_id += 1

        # 如果启用连续梯度
        if self.contiguous_gradients:
            # 初始化 ipg 缓冲区
            self.ipg_buffer = []

            # 创建一个大小为 reduce_bucket_size 的空tensor，数据类型为 self.dtype，设备为当前设备
            buf_0 = torch.empty(int(self.reduce_bucket_size),
                                dtype=self.dtype,
                                device=get_accelerator().current_device_name())
            gd.debuginfo(prj="ds", info=f'buf_0={infoTensor(buf_0)}')

            # 将 buf_0 添加到 ipg 缓冲区中
            self.ipg_buffer.append(buf_0)

            # Use double buffers to avoid data access conflict when overlap_comm is enabled.
            # 如果启用了 overlap_comm，使用双缓冲区以避免在启用 overlap_comm 时出现数据访问冲突
            if self.overlap_comm:
                buf_1 = torch.empty(int(self.reduce_bucket_size),
                                    dtype=self.dtype,
                                    device=get_accelerator().current_device_name())
                gd.debuginfo(prj="ds", info=f'buf_1={infoTensor(buf_1)}')
                # 将 buf_1 添加到 ipg 缓冲区中
                self.ipg_buffer.append(buf_1)

            # 初始化 ipg 索引
            self.ipg_index = 0

        # 如果使用自定义损失缩放器
        if self.custom_loss_scaler:
            # 将损失按照外部损失缩放因子进行缩放
            scaled_loss = self.external_loss_scale * loss
            gd.debuginfo(prj="ds", info=f'scaled_loss={infoTensor(scaled_loss)}')

            # 对缩放后的损失进行反向传播
            scaled_loss.backward()
        else:
            gd.debuginfo(prj="ds") #ph1-z2 train1batch
            # 如果没有使用自定义损失缩放器，使用 loss_scaler 对损失进行反向传播
            self.loss_scaler.backward(loss.float(), retain_graph=retain_graph)

        gd.debuginfo(prj='ds', info=f'C:{self.__class__.__name__} FUNC_OUT')

    # 检查是否溢出
    def check_overflow(self, partition_gradients=True):
        # gd.debuginfo(prj="ds")
        # 调用内部的 `_check_overflow` 方法
        # 该方法旨在检查计算过程中是否出现了溢出
        # `partition_gradients` 参数决定是否在不同设备上分割梯度计算
        self._check_overflow(partition_gradients)

    # 更新比例
    def _update_scale(self, has_overflow=False):
        # gd.debuginfo(prj="ds")
        # has_overflow 是一个布尔值，如果为True，表示在前向或者反向传播中发生了梯度溢出

        # self.loss_scaler 是一个梯度缩放器，它是用来防止在反向传播过程中梯度小到无法表示的情况
        # update_scale 是 loss_scaler 的一个方法，通过传入 has_overflow 参数，来更新当前的缩放因子

        # 如果发生了梯度溢出，就会减小缩放因子，否则就会增大缩放因子
        # 这样可以保证在训练过程中，梯度既不会太大导致溢出，也不会太小到无法表示
        self.loss_scaler.update_scale(has_overflow)

    # Promote state so it can be retrieved or set via "fp16_optimizer_instance.state"
    # 获得状态
    def _get_state(self):
        # gd.debuginfo(prj="ds")
        return self.optimizer.state

    # 获得状态
    def _set_state(self, value):
        # gd.debuginfo(prj="ds")
        self.optimizer.state = value

    state = property(_get_state, _set_state)  #????

    # Promote param_groups so it can be retrieved or set via "fp16_optimizer_instance.param_groups"
    # (for example, to adjust the learning rate)
    # 获取参数组
    def _get_param_groups(self):
        # gd.debuginfo(prj="ds")
        return self.optimizer.param_groups

    # 设置参数组
    def _set_param_groups(self, value):
        # gd.debuginfo(prj="ds")
        self.optimizer.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)

    # Promote loss scale so it can be retrieved or set via "fp16_optimizer_instance.loss_scale"
    def _get_loss_scale(self):
        """
        本函数的目标是获取损失缩放值。如果存在自定义的损失缩放器，
        则返回外部的损失缩放值。否则，返回当前的损失缩放值。
        """
        if self.custom_loss_scaler:
            # gd.debuginfo(prj="ds")
            # 存在自定义的损失缩放器时，返回外部的损失缩放值
            return self.external_loss_scale
        else:
            # gd.debuginfo(prj="ds")
            # 不存在自定义的损失缩放器时，返回当前的损失缩放值
            return self.loss_scaler.cur_scale

    # 设置损失比例
    def _set_loss_scale(self, value):
        # gd.debuginfo(prj="ds")
        self.loss_scaler.cur_scale = value

    loss_scale = property(_get_loss_scale, _set_loss_scale)
    cur_scale = property(_get_loss_scale, _set_loss_scale)

    # Return group tensor after removing paddings that are added for alignment to DP world size.
    # This method works on the assumption that each group contains a single flattened tensor.
    # 获取去除填充的组， 这个方法工作在假设每个组包含一个平坦的tensor之上
    def _get_groups_without_padding(self, groups_with_padding):
        gd.debuginfo(prj="ds")
        # 创建一个空列表，用于存储去除填充后的组
        groups_without_padding = []

        # 使用枚举函数对带填充的组进行迭代，获取每个组的索引和内容
        for i, group in enumerate(groups_with_padding):
            # 计算每个组真实的长度，即组的元素总数减去该组的填充数量
            lean_length = group.numel() - self.groups_padding[i]

            # 从每个组中提取出真实的元素（去除填充），并添加到新的列表中
            groups_without_padding.append(group[:lean_length])

        # 返回去除填充后的组列表
        return groups_without_padding

    # Return optimizer state after removing paddings that are added for alignment.
    # 获取没有填充的状态
    def _get_state_without_padding(self, state_with_padding, padding):
        gd.debuginfo(prj="ds")
        # 初始化一个空字典，用于存放没有padding的状态
        lean_state = {}
        # 遍历传入的状态字典
        for key, value in state_with_padding.items():
            # 如果状态的值是一个张量
            if torch.is_tensor(value):
                # 计算不包含padding的长度
                lean_length = value.numel() - padding

                # 截取原张量的前lean_length长度的部分，赋值给新的状态字典
                lean_state[key] = value[:lean_length]
            else:
                # 如果状态的值不是张量，直接赋值给新的状态字典
                lean_state[key] = value

        # 返回没有padding的状态字典
        return lean_state

    # Return base optimizer states.
    # This method assumes that each param group contains a single flattened tensor.
    # 获取基础优化器状态， 这个方法工作在假设每个组包含一个平坦的tensor之上 . only call in state_dict
    def _get_base_optimizer_state(self):
        # 初始化一个空列表用于存储优化器的状态
        optimizer_groups_state = []

        # 遍历优化器的参数组
        for i, group in enumerate(self.optimizer.param_groups):
            # 获取参数组中的第一个参数
            p = group['params'][0]

            # 调用函数_get_state_without_padding去掉参数的填充，并获取优化器状态
            # self.groups_padding[i]是获取当前参数组的填充
            lean_optimizer_state = self._get_state_without_padding(self.optimizer.state[p], self.groups_padding[i])
            gd.debuginfo(prj="ds", info=f'i={i}, lean_optimizer_state={lean_optimizer_state}')

            # 将优化器状态添加到列表中
            optimizer_groups_state.append(lean_optimizer_state)

        # 返回处理后的优化器状态列表
        return optimizer_groups_state

    # 返回优化器的状态字典
    def state_dict(self):
        """
        返回一个字典，包含当前FP16_Optimizer实例的状态。
        这个字典包含FP16_Optimizer的属性，以及包含的Pytorch优化器的state_dict。
        示例::
            checkpoint = {}
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            torch.save(checkpoint, "saved.pth")

        Returns a dict containing the current state of this :class:`FP16_Optimizer` instance.
        This dict contains attributes of :class:`FP16_Optimizer`, as well as the state_dict
        of the contained Pytorch optimizer.
        Example::
            checkpoint = {}
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            torch.save(checkpoint, "saved.pth")
        """
        # 初始化一个空字典用于存储状态
        state_dict = {}
        # 存储损失缩放器的状态
        state_dict['loss_scaler'] = self.loss_scaler
        # 存储动态损失缩放的状态
        state_dict['dynamic_loss_scale'] = self.dynamic_loss_scale
        # 存储溢出的状态
        state_dict['overflow'] = self.overflow
        # 存储梯度裁剪的状态
        state_dict[CLIP_GRAD] = self.clip_grad

        # 如果启用了弹性检查点
        if self.elastic_checkpoint:
            gd.debuginfo(prj="ds")
            # 获取基础优化器的状态
            state_dict[BASE_OPTIMIZER_STATE] = self._get_base_optimizer_state()
        else:
            # 存储优化器的状态
            gd.debuginfo(prj="ds")
            state_dict[BASE_OPTIMIZER_STATE] = self.optimizer.state_dict()

        # Remove paddings for DP alignment to enable loading for other alignment values
        # 移除DP对齐的填充，以便于加载其他对齐值
        fp32_groups_without_padding = self._get_groups_without_padding(self.single_partition_of_fp32_groups)
        state_dict[SINGLE_PARTITION_OF_FP32_GROUPS] = fp32_groups_without_padding

        # 存储ZeroStage（零阶段）的状态
        state_dict[
            ZERO_STAGE] = ZeroStageEnum.gradients if self.partition_gradients else ZeroStageEnum.optimizer_states

        # 存储分组填充的状态
        state_dict[GROUP_PADDINGS] = self.groups_padding

        # 存储分区计数的状态
        state_dict[PARTITION_COUNT] = self.partition_count

        # 存储DeepSpeed版本的状态
        state_dict[DS_VERSION] = version

        # 存储参数切片映射的状态
        state_dict[PARAM_SLICE_MAPPINGS] = self._param_slice_mappings

        # 返回状态字典
        return state_dict

    # Restore base optimizer fp32 weights from elastic checkpoint by:
    # 1) Merging fp32 weights from checkpoints of all partitions
    # 2) Extracting fp32 weights for current partition from merged weights
    # 3) Using extracted weights to update base optimizer weights directly.
    # only call in  _load_legacy_checkpoint
    def _restore_from_elastic_fp32_weights(self, all_state_dict):
        gd.debuginfo(prj="ds")
        # 初始化一个空列表用于存储FP32分区数据
        merged_single_partition_of_fp32_groups = []

        # 遍历 self 中的 FP32 分区组
        for i in range(len(self.single_partition_of_fp32_groups)):
            # 获取当前分区的ID，使用真实的数据并行过程组
            partition_id = dist.get_rank(group=self.real_dp_process_group[i])

            # 从所有的状态字典中获取对应的FP32分区
            merged_partitions = [sd[SINGLE_PARTITION_OF_FP32_GROUPS][i] for sd in all_state_dict]

            # 如果当前优化器的参数组是一个MOE组
            if self.is_moe_group(self.optimizer.param_groups[i]):
                # 获取EP的排名
                ranks = self.get_ep_ranks(group_name=self.optimizer.param_groups[i]['name'])
                # 从合并的分区中选择对应排名的部分
                merged_partitions = [merged_partitions[i] for i in ranks]

            # 将所有合并的分区数据进行平均，然后根据NCCL开始对齐因子和真实的数据并行过程组的大小进行对齐
            flat_merged_partitions = self.flatten_dense_tensors_aligned(
                merged_partitions,
                self.nccl_start_alignment_factor * dist.get_world_size(group=self.real_dp_process_group[i]))

            # 获取数据并行分区
            dp_partitions = self.get_data_parallel_partitions(flat_merged_partitions, i)

            # 将数据并行分区添加到最终的FP32分区组中
            merged_single_partition_of_fp32_groups.append(dp_partitions[partition_id])

        # 遍历当前的FP32分区组和合并后的FP32分区组
        for current, saved in zip(self.single_partition_of_fp32_groups, merged_single_partition_of_fp32_groups):
            # 将合并后的数据拷贝到当前的数据中
            current.data.copy_(saved.data)

    # Restore base optimizer fp32 weights from ZeRO fp16 or bfloat16 weights
    # 从16位权重恢复
    def _restore_from_bit16_weights(self):
        gd.debuginfo(prj="ds")
        # 遍历 bit16_partitions 和 fp32_partition
        for group_id, (bit16_partitions, fp32_partition) in enumerate(
                zip(self.parallel_partitioned_bit16_groups, self.single_partition_of_fp32_groups)):
            # 获取当前进程在进程组中的排名（ID）
            partition_id = dist.get_rank(group=self.real_dp_process_group[group_id])
            # 将bit16_partitions中对应ID的数据复制到fp32_partition中
            fp32_partition.data.copy_(bit16_partitions[partition_id].data)

    # Refresh the fp32 master params from the fp16 or bfloat16 copies.
    # 刷新FP32参数
    def refresh_fp32_params(self):
        # gd.debuginfo(prj="ds")
        # 调用内部函数_restore_from_bit16_weights，用于从16位浮点数权重恢复到32位浮点数
        self._restore_from_bit16_weights()

    # Extract optimizer state for current partition from merged states of all partitions
    # 分区基础优化器状态  only call by _restore_elastic_base_optimizer_state
    def _partition_base_optimizer_state(self, state_key, all_partition_states, group_id):
        # 获取当前进程在数据并行组中的排名
        partition_id = dist.get_rank(group=self.real_dp_process_group[group_id])

        # 获取数据并行组的总进程数
        alignment = dist.get_world_size(group=self.real_dp_process_group[group_id])

        gd.debuginfo(prj="ds", info=f'partition_id={partition_id}, alignment={alignment}')

        # 如果状态是一个张量
        if torch.is_tensor(all_partition_states[0]):
            gd.debuginfo(prj="ds", info=f'all_partition_states[0]={all_partition_states[0]}')
            # 将所有分区状态张量扁平化并对齐
            flat_merged_partitions = self.flatten_dense_tensors_aligned(all_partition_states, alignment)

            # 获取数据并行分区
            dp_partitions = self.get_data_parallel_partitions(flat_merged_partitions, group_id)

            # 返回当前进程对应的分区
            return dp_partitions[partition_id]
        else:
            gd.debuginfo(prj="ds")
            # Assume non-tensor states are not partitioned and equal across ranks, so return first one
            # 如果状态不是张量，假设它没有分区，并且在所有进程中都是相同的，因此返回第一个状态
            return all_partition_states[0]

    # 恢复基本优化器状态
    def _restore_base_optimizer_state(self, base_optimizer_group_states):
        # gd.debuginfo(prj="ds")
        # 如果 base_optimizer_group_states 是字典，我们取出其中的 'state' 键对应的值
        if type(base_optimizer_group_states) == dict:
            gd.debuginfo(prj="ds")
            base_optimizer_group_states = base_optimizer_group_states['state']

        # 遍历优化器中的参数组
        for i, group in enumerate(self.optimizer.param_groups):
            # 取出参数组中的第一个参数
            p = group['params'][0]

            # 遍历存储的优化器状态
            for key, saved in base_optimizer_group_states[i].items():
                # 如果优化器的状态是一个张量
                if torch.is_tensor(self.optimizer.state[p][key]):
                    # 获取目标张量
                    dst_tensor = self.optimizer.state[p][key]
                    # 通过_pad_tensor函数获取与目标张量等长的源张量
                    src_tensor = _get_padded_tensor(saved, dst_tensor.numel())

                    # 将源张量的数据复制到目标张量
                    self.optimizer.state[p][key].data.copy_(src_tensor.data)

                    gd.debuginfo(prj="ds", info=f'self.optimizer.state[{p}][{key}], src_tensor={infoTensor(src_tensor)}, dst_tensor={infoTensor(dst_tensor)}')
                else:
                    # 如果优化器的状态不是张量，直接赋值
                    gd.debuginfo(prj="ds", info=f'self.optimizer.state[{p}][{key}]={saved}')
                    self.optimizer.state[p][key] = saved

    # 获取EP等级
    def get_ep_ranks(self, rank=0, group_name=None):

        # 导入deepspeed库中的groups模块，该模块提供了并行计算组的管理功能
        from deepspeed.utils import groups

        # _get_expert_parallel_world_size函数返回专家并行世界的大小
        # 专家并行世界是指在专家并行（Expert Parallelism）策略下，所有专家（即模型的一部分）所在的并行计算环境
        expert_parallel_size_ = groups._get_expert_parallel_world_size(group_name)

        # _get_data_parallel_world_size函数返回数据并行世界的大小
        # 数据并行世界是指在数据并行（Data Parallelism）策略下，所有数据所在的并行计算环境
        world_size = groups._get_data_parallel_world_size()

        # _get_expert_parallel_rank函数获取当前处理器在专家并行世界中的排名
        # 排名决定了当前处理器在并行计算中的执行顺序
        rank = groups._get_expert_parallel_rank(group_name)

        # range函数生成一个序列，范围从当前处理器的排名开始，到数据并行世界的大小，步长为专家并行世界的大小
        # 这样可以确保每个处理器在其执行顺序上都是均匀分布的
        ranks = range(rank, world_size, expert_parallel_size_)

        gd.debuginfo(prj="ds", info=f'expert_parallel_size_={expert_parallel_size_}, world_size={world_size}, rank={rank}, ranks={ranks}')

        # 返回一个由处理器排名组成的列表，这个列表可以用于管理处理器的执行顺序
        return list(ranks)

    # Restore base optimizer state from elastic checkpoint by
    # 1) Merging optimizer state from checkpoints of all partitions
    # 2) Extracting optimizer state for current partition from the merged state
    # 3) Using the extracted value to directly update the base optimizer.
    #  恢复弹性基本优化器状态
    def _restore_elastic_base_optimizer_state(self, all_state_dict):
        gd.debuginfo(prj="ds")
        # 初始化一个空的列表，用于存储基础优化器的状态
        base_optimizer_group_states = []

        # 对优化器的参数组进行遍历
        for i in range(len(self.optimizer.param_groups)):
            # 初始化一个空字典，用于存储当前参数组的状态
            partition_states = {}

            # 从所有的状态字典中获取当前参数组的状态
            all_partition_group_states = [sd[BASE_OPTIMIZER_STATE][i] for sd in all_state_dict]

            gd.debuginfo(prj="ds", info=f'i={i}, all_partition_group_states={all_partition_group_states}')

            # 如果当前参数组是一个 MOE 组（Mixture of Experts，混合专家模型）
            if self.is_moe_group(self.optimizer.param_groups[i]):
                # 获取当前参数组在 EP 中的排名
                ranks = self.get_ep_ranks(group_name=self.optimizer.param_groups[i]['name'])
                # 根据排名，重新获取当前参数组的状态
                all_partition_group_states = [all_partition_group_states[i] for i in ranks]

            # 对所有的状态进行遍历，合并分区状态
            for key in all_partition_group_states[0].keys():
                # 获取所有分区的同一状态
                all_partition_states = [all_states[key] for all_states in all_partition_group_states]

                # 合并所有分区的状态
                partition_states[key] = self._partition_base_optimizer_state(key, all_partition_states, i)

            # 将合并后的状态添加到状态列表中
            base_optimizer_group_states.append(partition_states)

        # 通过状态列表，恢复基础优化器的状态
        self._restore_base_optimizer_state(base_optimizer_group_states)

    # 加载优化器状态字典
    def load_state_dict(self,
                        state_dict_list,              # 状态字典列表，一般用于存储模型的参数
                        load_optimizer_states=True,   # 是否加载优化器的状态
                        load_from_fp32_weights=False, # 是否从fp32权重加载
                        checkpoint_folder=None):      # 加载序列，默认为None
        # 如果提供了检查点文件夹
        if checkpoint_folder:
            # gd.debuginfo(prj="ds")
            # 从通用检查点加载模型和优化器状态
            self._load_universal_checkpoint(checkpoint_folder, load_optimizer_states, load_from_fp32_weights)
        else:
            # gd.debuginfo(prj="ds")
            # 从传统检查点加载模型和优化器状态
            self._load_legacy_checkpoint(state_dict_list, load_optimizer_states, load_from_fp32_weights)

    # 定义一个方法来加载通用检查点 用于加载通用的模型检查点
    def _load_universal_checkpoint(self, checkpoint_folder, load_optimizer_states, load_from_fp32_weights):
        # 从检查点文件夹中加载超参数检查点状态
        self._load_hp_checkpoint_state(checkpoint_folder)

    # 参数组
    @property
    def param_groups(self):
        # gd.debuginfo(prj="ds")
        """Forward the wrapped optimizer's parameters."""
        return self.optimizer.param_groups

    # 加载HP检查点状态
    def _load_hp_checkpoint_state(self, checkpoint_dir):
        # 将给定的路径后面添加 "zero" 子目录
        checkpoint_dir = os.path.join(checkpoint_dir, "zero")

        # 获取模型并行的排名
        tp_rank = bwc_tensor_model_parallel_rank(mpu=self.mpu)

        # 获取模型并行的世界大小
        tp_world_size = self.mpu.get_slice_parallel_world_size()

        gd.debuginfo(prj="ds",info=f'checkpoint_dir={checkpoint_dir}, tp_rank={tp_rank}, tp_world_size={tp_world_size}')

        # 遍历优化器的参数组
        for i, _ in enumerate(self.optimizer.param_groups):
            # 遍历16位的参数组
            for lp in self.bit16_groups[i]:
                # 如果参数有映射
                if lp._hp_mapping is not None:
                    gd.debuginfo(f"Loading {self.param_names[lp]}=tp_rank={tp_rank}, tp_world_size={tp_world_size}")
                    # 加载检查点状态
                    lp.load_hp_checkpoint_state(os.path.join(checkpoint_dir, self.param_names[lp]), tp_rank,
                                                tp_world_size)

    # 加载旧版检查点
    # 定义了一个名为 _load_legacy_checkpoint 的函数，这个函数的目的是从旧版的检查点中加载模型和优化器的状态
    def _load_legacy_checkpoint(self, state_dict_list, load_optimizer_states=True, load_from_fp32_weights=False):

        r"""Loading ZeRO checkpoint

        Arguments:
            state_dict_list: List of all saved ZeRO checkpoints, one for each saved partition.
                Note that the number of saved partitions may differ from number of loading partitions to support
                changing GPU count, specifically DP world size, between saving and loading checkpoints.
            load_optimizer_states: Boolean indicating whether or not to load base optimizer states
            load_from_fp32_weights: Boolean indicating whether to initialize fp32 master weights from fp32
            copies in checkpoints (no precision loss) or from model's fp16 copies (with precision loss).
            
        加载 ZeRO 检查点
        参数：
            state_dict_list: 所有保存的 ZeRO 检查点的列表，每个保存的分区一个。
                注意，保存的分区数量可能与加载的分区数量不同，以支持在保存和加载检查点之间更改 GPU 数量，特别是 DP 世界大小。
            load_optimizer_states: 布尔值，表示是否加载基础优化器状态
            load_from_fp32_weights: 布尔值，表示是否从检查点的 fp32 副本（无精度损失）或模型的 fp16 副本（有精度损失）初始化 fp32 主权重。

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

        # 获取当前进程的等级，这是在分布式训练中使用的
        dp_rank = dist.get_rank(group=self.dp_process_group)

        # 获取当前等级的状态字典
        current_rank_sd = state_dict_list[dp_rank]
        gd.debuginfo(prj="ds", info=f'dp_rank={dp_rank}, current_rank_sd={current_rank_sd}')

        # 获取状态信息，包括损失缩放器，动态损失缩放，溢出状态，梯度裁剪等
        self.loss_scaler = current_rank_sd.get('loss_scaler', self.loss_scaler)
        self.dynamic_loss_scale = current_rank_sd.get('dynamic_loss_scale', self.dynamic_loss_scale)
        self.overflow = current_rank_sd.get('overflow', self.overflow)
        self.clip_grad = current_rank_sd.get(CLIP_GRAD, self.clip_grad)
        gd.debuginfo(prj="ds", info=f'self.loss_scaler={self.loss_scaler}')
        gd.debuginfo(prj="ds", info=f'self.dynamic_loss_scale={self.dynamic_loss_scale}')
        gd.debuginfo(prj="ds", info=f'self.overflow={self.overflow}')
        gd.debuginfo(prj="ds", info=f'self.clip_grad={self.clip_grad}')

        # 获取检查点版本，版本信息在加载时需要进行检查以保证兼容性
        ckpt_version = current_rank_sd.get(DS_VERSION, False)
        gd.debuginfo(prj="ds", info=f'A-ckpt_version={ckpt_version}')

        assert ckpt_version, f"Empty ds_version in checkpoint, not clear how to proceed"

        ckpt_version = pkg_version.parse(ckpt_version)
        gd.debuginfo(prj="ds", info=f'B-ckpt_version={ckpt_version}')

        # zero stage 1 mode # 针对 zero stage 1 模式进行版本检查
        # 如果当前使用的是 ZeRO stage 1 模式，需要进行版本检查
        if not self.partition_gradients:
            required_version = pkg_version.parse("0.3.17")
            error_str = f"ZeRO stage 1 changed in {required_version} and is not backwards compatible " \
                "with older stage 1 checkpoints. If you'd like to load an old ZeRO-1 checkpoint " \
                "please use an older version of DeepSpeed (<= 0.5.8) and set 'legacy_stage1': true in your zero config json."
            assert required_version <= ckpt_version, f"Old version: {ckpt_version} {error_str}"

        # 检查状态字典中基础优化器状态的数据类型是否为字典
        ckpt_is_rigid = isinstance(current_rank_sd[BASE_OPTIMIZER_STATE], dict)
        gd.debuginfo(prj="ds", info=f'ckpt_is_rigid={ckpt_is_rigid}')

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
                # 如果状态字典中基础优化器状态的数据类型是字典，则直接加载状态
                self.optimizer.load_state_dict(current_rank_sd[BASE_OPTIMIZER_STATE])
            else:
                # 如果状态字典中基础优化器状态的数据类型不是字典，那么根据是否是弹性检查点来使用不同的恢复方法
                if self.elastic_checkpoint:
                    gd.debuginfo(prj="ds")  # 如果是弹性检查点，使用对应的恢复方法
                    # loading elastic into elastic exec
                    self._restore_elastic_base_optimizer_state(state_dict_list)
                else:
                    gd.debuginfo(prj="ds") # 如果是非弹性检查点，使用基础的恢复方法
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

        # 在这一点上，优化器对模型的 fp32 参数的引用是最新的。
        # 优化器的超参数和内部缓冲区也是最新的。
        # 然而，优化器存储的模型的 fp16 参数的 fp32 主副本仍然过时。有两个选择。
        # 1：从模型的 fp16 参数刷新主参数。
        #    这需要更少的存储但会导致精度损失。
        # 2：单独保存和恢复 fp32 主副本。
        #    如果改变 DP 度，我们选择选项 1，否则选择选项 2

        if load_from_fp32_weights:
            # option 2 from above
            # 选择方案2，如果是弹性检查点并且状态字典不是字典类型，使用对应的恢复方法
            if self.elastic_checkpoint and not ckpt_is_rigid:
                gd.debuginfo(prj="ds")
                self._restore_from_elastic_fp32_weights(state_dict_list)
            else:
                gd.debuginfo(prj="ds")
                # 对于非弹性检查点，简单地从当前等级的保存权重复制就足够了
                # For non-elastic checkpoint, simply copying from saved weights of current rank is sufficient.
                for current, saved in zip(self.single_partition_of_fp32_groups,
                                          current_rank_sd[SINGLE_PARTITION_OF_FP32_GROUPS]):

                    src_tensor = _get_padded_tensor(saved, current.numel())
                    gd.debuginfo(prj="ds", info=f'current={current}, saved={saved}, src_tensor={infoTensor(src_tensor)}')

                    current.data.copy_(src_tensor.data)
        else:
            gd.debuginfo(prj="ds")
            # option 1 from above
            self._restore_from_bit16_weights()

        if load_optimizer_states:
            gd.debuginfo(prj="ds")
            self._link_all_hp_params()


def _handle_overflow(cpu_sum, x, i):
    import math
    rank = dist.get_rank()
    gd.debuginfo(prj="ds", info=f'rank={rank}')
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
    # shared params calculated only once
    total_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
    gd.debuginfo(prj="ds", info=f'total_params={total_params}')
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
