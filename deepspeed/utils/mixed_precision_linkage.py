# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import types
from deepspeed.utils import get_full_hp_param, get_full_hp_grad, get_hp_fragment_mapping

from pydebug import gd, infoTensor

def link_hp_params(lp_param_list, flat_hp_partition, gradient_dict, offload_gradient_dict, use_offload,
                   param_group_index, partition_start, partition_size, partition_optimizer_state, dp_group):
    local_lp_param_and_offset = _init_lp_to_hp_mapping(lp_param_list, partition_start, partition_size, dp_group)
    # gd.debuginfo(prj="ds", info=f'local_lp_param_and_offset={local_lp_param_and_offset}')
    '''
    ds F# deepspeed/utils/mixed_precision_linkage.py f# link_hp_params L#: 14 I# local_lp_param_and_offset=[(Parameter containing:
tensor([[ 0.0771,  0.0131, -0.0334,  ..., -0.0536,  0.2452,  0.1958],
        ...,
        [-0.1482,  0.2419, -0.2327,  ..., -0.0477,  0.0116,  0.1234]],
       device='cuda:0', dtype=torch.float16, requires_grad=True), 0), (Parameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        ...
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', dtype=torch.float16,
       requires_grad=True), 12288),
    '''
    pretensor = ''
    print(f"len of output_tensor={len(local_lp_param_and_offset)}") #防止消失，直接打印 # len of output_tensor=21479424
    for i, _ in enumerate(local_lp_param_and_offset):
        for j, _ in enumerate(local_lp_param_and_offset[i]):
            tmpstr = infoTensor(local_lp_param_and_offset[i][j])
            if tmpstr != pretensor: #重复的不打印
                gd.debuginfo(prj='ds', info=f"local_lp_param_and_offset[{i}]={tmpstr}")
                pretensor = tmpstr

    for lp_param, lp_start in local_lp_param_and_offset:
        lp_param._hp_mapping = get_hp_fragment_mapping(lp_param, lp_start, flat_hp_partition, gradient_dict,
                                                       offload_gradient_dict, use_offload, param_group_index,
                                                       partition_start, partition_size, partition_optimizer_state)
        # 太大 gd.debuginfo(prj="ds", info=f'lp_param._hp_mapping={lp_param._hp_mapping}')
        # 就是一个类的实例 tensor_fragment(lp_fragment=tensor([ 1.7517e-02,  8.3069e-02,  1.3237e-02,  3.7140e-02,  2.5650e-02,

def _init_lp_to_hp_mapping(lp_param_list, partition_start, partition_size, dp_group):
    gd.debuginfo(prj="ds")
    current_offset = 0
    param_and_offset_list = []
    partition_end = partition_start + partition_size
    index_in_param_group = 0
    for i, lp_param in enumerate(lp_param_list):
        lp_param._hp_mapping = None
        lp_param._dp_group = dp_group
        lp_param.get_full_hp_param = types.MethodType(get_full_hp_param, lp_param)
        lp_param.get_full_hp_grad = types.MethodType(get_full_hp_grad, lp_param)

        # lp_param overlaps with partition if both are true
        # 1) current_offset < partition_end,
        # 2) current_offset + lp_param.numel() >= partition_start
        lp_param_end = current_offset + lp_param.numel()
        if current_offset < partition_end and lp_param_end > partition_start:
            param_and_offset_list.append((lp_param, current_offset))
            lp_param._index_in_param_group = index_in_param_group
            # Indices for params in this partition/GPU
            index_in_param_group += 1
        current_offset += lp_param.numel()
    #print("param_and_offset_list is:", param_and_offset_list)

    gd.debuginfo(prj='ds')
    return param_and_offset_list

'''
	param_and_offset_list is: [(Parameter containing:
tensor([[ 0.0029, -0.0250,  0.0119,  ..., -0.0127,  0.0176, -0.0144],
        [ 0.0281, -0.0600,  0.0175,  ..., -0.0219,  0.0013, -0.0139],
        [-0.0181, -0.0034,  0.0650,  ..., -0.0435, -0.0037, -0.0121],
        ...,
        [ 0.0414, -0.0235,  0.0003,  ..., -0.0149, -0.0077,  0.0040],
        [ 0.0204,  0.0079,  0.0196,  ...,  0.0430, -0.0216,  0.0076],
        [-0.0198,  0.0371, -0.0093,  ...,  0.0057,  0.0320,  0.0507]],
       device='cuda:1', dtype=torch.float16, requires_grad=True), 62012160), (Parameter containing:
tensor([[-0.0131, -0.0069,  0.0939,  ..., -0.0011, -0.0174, -0.0332],
        [ 0.0023, -0.0219, -0.0055,  ...,  0.0086, -0.0050, -0.0138],
        [-0.0438, -0.0695, -0.0432,  ..., -0.0086, -0.0497,  0.0034],
        ...,
        [ 0.1256, -0.0108,  0.0238,  ...,  0.0352,  0.0161, -0.0339],
        [ 0.0340,  0.0185,  0.0350,  ...,  0.0643,  0.2128, -0.0453],
        [ 0.0249,  0.0026,  0.0283,  ..., -0.0864,  0.0279,  0.0225]],
       device='cuda:1', dtype=torch.float16, requires_grad=True), 62601984), (Parameter containing:
tensor([[ 0.0149,  0.0278,  0.0088,  ...,  0.0258, -0.0109,  0.0139],
        [-0.0132,  0.0049,  0.0150,  ...,  0.0267,  0.0112, -0.0395],
        [-0.0118, -0.0175, -0.0068,  ..., -0.0039, -0.0072, -0.0114],
        ...,
        [ 0.0676, -0.0199,  0.0306,  ...,  0.0166, -0.0092, -0.0257],
        [-0.0414, -0.0378,  0.0101,  ..., -0.0247,  0.0074,  0.0183],
        [ 0.0100,  0.0086, -0.0207,  ..., -0.0069, -0.0046,  0.0101]],
       device='cuda:1', dtype=torch.float16, requires_grad=True), 63191808), (Parameter containing:
tensor([0.4119, 0.3984, 0.4307, 0.4651, 0.4661, 0.3464, 0.4761, 0.3530, 0.4272,
        0.3428, 0.3816, 0.4199, 0.4089, 0.4968, 0.3723, 0.4668, 0.4998, 0.3953,
        0.4421, 0.4229, 0.9312, 0.4124, 0.4570, 0.4321, 0.3259, 0.4626, 0.4431,

'''
