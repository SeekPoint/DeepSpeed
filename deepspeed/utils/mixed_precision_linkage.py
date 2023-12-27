# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import types
from deepspeed.utils import get_full_hp_param, get_full_hp_grad, get_hp_fragment_mapping

from pydebug import gd, infoTensor

def link_hp_params(lp_param_list,
                   flat_hp_partition,
                   gradient_dict,
                   offload_gradient_dict,
                   use_offload,
                   param_group_index,
                   partition_start,
                   partition_size,
                   partition_optimizer_state,
                   dp_group):

    gd.debuginfo(prj='ds', info=f'__FUNC_IN_OUT__')

    local_lp_param_and_offset = _init_lp_to_hp_mapping(lp_param_list, partition_start, partition_size, dp_group)

    # 具体内容里面函数内部打印
    gd.debuginfo(prj='ds', info=f"len of output_tensor={len(local_lp_param_and_offset)}") #len of output_tensor=21479424

    gd.debuginfo(prj='ds', info=f".............................................................")
    
    for index, (lp_param, lp_start) in enumerate(local_lp_param_and_offset):
        gd.debuginfo(prj='ds', info=f"call {index} get_hp_fragment_mapping")
        lp_param._hp_mapping = get_hp_fragment_mapping(lp_param, lp_start, flat_hp_partition, gradient_dict,
                                                       offload_gradient_dict, use_offload, param_group_index,
                                                       partition_start, partition_size, partition_optimizer_state)
        # 太大 gd.debuginfo(prj="ds", info=f'lp_param._hp_mapping={lp_param._hp_mapping}')
        # 就是一个数据类的实例 tensor_fragment(lp_fragment=tensor([ 1.7517e-02,  8.3069e-02,  1.3237e-02,  3.7140e-02,  2.5650e-02,

    gd.debuginfo(prj='ds', info=f'__FUNC_IN_OUT__')

def _init_lp_to_hp_mapping(lp_param_list, partition_start, partition_size, dp_group):
    gd.debuginfo(prj='ds', info=f'__FUNC_IN_OUT__')
    current_offset = 0
    param_and_offset_list = []
    partition_end = partition_start + partition_size
    index_in_param_group = 0
    gd.debuginfo(prj='ds', info=f'partition_start={partition_start}, partition_end={partition_end}')
    for i, lp_param in enumerate(lp_param_list):
        lp_param._hp_mapping = None
        lp_param._dp_group = dp_group
        lp_param.get_full_hp_param = types.MethodType(get_full_hp_param, lp_param)
        lp_param.get_full_hp_grad = types.MethodType(get_full_hp_grad, lp_param)

        # lp_param overlaps with partition if both are true
        # 1) current_offset < partition_end,
        # 2) current_offset + lp_param.numel() >= partition_start
        lp_param_end = current_offset + lp_param.numel()

        gd.debuginfo(prj='ds', info=f'i={i}, lp_param_end={lp_param_end}')
        #  f'lp_param.get_full_hp_param={infoTensor(lp_param.get_full_hp_param)}, '
        #  f'lp_param.get_full_hp_grad={infoTensor(lp_param.get_full_hp_grad)}')
        #  lp_param.get_full_hp_grad=<bound method get_full_hp_grad of Parameter containing:
        #  tensor([-5.0316e-03,  4.3678e-03,  1.2326e-04,  6.3820e-03, -2.7637e-03,
        #          2.8276e-04, -3.2158e-03,  1.5438e-04, -5.0697e-03,  4.1847e-03,

        if current_offset < partition_end and lp_param_end > partition_start:
            param_and_offset_list.append((lp_param, current_offset)) #这是一个tuple
            gd.debuginfo(prj='ds', info=f"lp_param={infoTensor(lp_param)}, current_offset={current_offset}")
            lp_param._index_in_param_group = index_in_param_group
            # Indices for params in this partition/GPU
            index_in_param_group += 1
        current_offset += lp_param.numel()

    gd.debuginfo(prj='ds', info=f'__FUNC_IN_OUT__')
    return param_and_offset_list # local_lp_param_and_offset 其实一个tuple的list，每个tupple前面是param/tensor, 后面是offset


