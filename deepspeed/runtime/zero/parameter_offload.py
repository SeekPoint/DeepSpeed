# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import sys
import torch
from collections import OrderedDict
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
from deepspeed.runtime.zero.partition_parameters import _init_external_params
from deepspeed.runtime.zero.partition_parameters import *
from deepspeed.runtime.zero.partitioned_param_coordinator import PartitionedParameterCoordinator, InflightParamRegistry, iter_params
from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from pydebug import gd, infoTensor
FWD_MODULE_STACK = list()


def is_builtin_type(obj):
    gd.debuginfo(prj="ds")
    # https://stackoverflow.com/a/17795199
    return obj.__class__.__module__ == '__builtin__' or obj.__class__.__module__ == "builtins"


def isinstance_namedtuple(obj: object) -> bool:
    """
    Is this an instance of namedtuple/NamedTuple?
    From: https://stackoverflow.com/a/62692640

    Args:
        obj (object): An object.

    Returns:
        bool: True if namedtuple/NamedTuple else False.
    """
    gd.debuginfo(prj="ds")
    return isinstance(obj, tuple) and hasattr(obj, '_asdict') and hasattr(obj, '_fields')


# ensure we only warn once, otherwise every iteration will trigger a warning
warned = False


def _apply_to_tensors_only(module, functional, backward_function, outputs):
    """
    Apply a torch.autograd.Function that calls a `backward_function` to every Tensor in `outputs`.

    Args:
        module (torch.nn.Module):  A torch module
        functional (Type[torch.autograd.Function]): The function class to apply.
        backward_function (Callable[[torch.nn.Module], None]): A backward_function to pass to
            `functional.apply`.
        outputs (Any): The output of `module`.

    Returns:
        Any: The output of `module`.
    """
    # gd.debuginfo(prj="ds")
    if isinstance(outputs, (tuple, list)):
        # gd.debuginfo(prj="ds")
        touched_outputs = []
        for output in outputs:
            touched_output = _apply_to_tensors_only(module, functional, backward_function, output)
            touched_outputs.append(touched_output)

        if isinstance_namedtuple(outputs):
            # namedtuples require a slightly different syntax.
            return outputs.__class__(*touched_outputs)

        return outputs.__class__(touched_outputs)
    elif isinstance(outputs, dict):
        # gd.debuginfo(prj="ds")
        # apply inplace to avoid recreating dict inherited objects
        for key in outputs.keys():
            outputs[key] = _apply_to_tensors_only(module, functional, backward_function, outputs[key])
        return outputs

    elif isinstance(outputs, torch.Tensor):
        # gd.debuginfo(prj="ds")
        # this also applies to torch.Tensor's subclasses like torch.nn.parameter.Parameter
        touched_outputs = functional.apply(module, backward_function, outputs)

        # restore zero param attributes if those get stripped by `backward_function`
        if not is_zero_param(touched_outputs) and is_zero_param(outputs):
            gd.debuginfo(prj="ds")
            touched_outputs.ds_param_alias = outputs
        return touched_outputs
    else:
        # gd.debuginfo(prj="ds")
        if not is_builtin_type(outputs):
            global warned
            if not warned and dist.get_rank() == 0:
                logger.warning(
                    f"A module has unknown inputs or outputs type ({type(outputs)}) and the tensors embedded in it cannot be detected. "
                    "The ZeRO-3 hooks designed to trigger before or after backward pass of the module relies on knowing the input and "
                    "output tensors and therefore may not get triggered properly.")
                warned = True
        return outputs


#for each tensor in outputs run the forward_function and register backward_function as hook
def _apply_forward_and_backward_to_tensors_only(module, forward_function, backward_function, outputs):
    if type(outputs) is tuple:
        gd.debuginfo(prj="ds")
        touched_outputs = []
        for output in outputs:
            touched_output = _apply_forward_and_backward_to_tensors_only(module, forward_function, backward_function,
                                                                         output)
            touched_outputs.append(touched_output)
        return tuple(touched_outputs)
    elif type(outputs) is torch.Tensor:
        gd.debuginfo(prj="ds")
        forward_function(outputs)
        if outputs.requires_grad:
            gd.debuginfo(prj="ds")
            outputs.register_hook(backward_function)
        return outputs
    else:
        gd.debuginfo(prj="ds")
        return outputs


class ZeROOrderedDict(OrderedDict):

    def __init__(self, parent_module, *args, **kwargs):
        """A replacement for ``collections.OrderedDict`` to detect external ZeRO params.

        Args:
            parent_module (``collections.OrderedDict``): the collection to replace
        """
        # gd.debuginfo(prj='ds', info=f"C:{self.__class__.__name__}")
        super().__init__(*args, **kwargs)
        self._parent_module = parent_module
        self._in_forward = False

    def __getitem__(self, key):
        param = super().__getitem__(key)

        # Params can be registered as None (e.g., bias)
        if param is None:
            # gd.debuginfo(prj="ds")
            return param

        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            # gd.debuginfo(prj="ds")
            if self._parent_module._parameters._in_forward:
                gd.debuginfo(prj="ds")
                register_external_parameter(FWD_MODULE_STACK[-1], param)
                param.all_gather()
                print_rank_0(f'Registering external parameter from getter {key} ds_id = {param.ds_id}', force=False)

        return param


def _inject_parameters(module, cls):
    # gd.debuginfo(prj="ds")
    for module in module.modules():
        if cls == ZeROOrderedDict:
            # gd.debuginfo(prj="ds")
            new_param = cls(parent_module=module)
        else:
            gd.debuginfo(prj="ds")
            new_param = cls()

        for key, param in module._parameters.items():
            new_param[key] = param
        module._parameters = new_param


class PreBackwardFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, module, pre_backward_function, outputs):
        gd.debuginfo(prj="ds")
        ctx.module = module
        ctx.pre_backward_function = pre_backward_function
        if not hasattr(module, "applied_pre_backward_ref_cnt"):
            module.applied_pre_backward_ref_cnt = 0
        module.applied_pre_backward_ref_cnt += 1
        #print(f"After Forward: {ctx.module.__class__.__name__}")
        outputs = outputs.detach()
        return outputs

    @staticmethod
    def backward(ctx, *args):
        gd.debuginfo(prj="ds")
        #print(f"Before Backward: {ctx.module.__class__.__name__}")
        ctx.pre_backward_function(ctx.module)
        return (None, None) + args


class PostBackwardFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, module, pre_backward_function, output):
        # gd.debuginfo(prj="ds")
        ctx.module = module
        if output.requires_grad:
            gd.debuginfo(prj="ds")
            #TODO SOME TIMES post backward does not seem to be triggered debug in detail
            #Should only cause increase in memory not correctness issue
            #if output.grad_fn.__class__.__name__ == 'ViewBackward':
            #    ctx.view=True
            #    print(f"Warning view tensor for input to module : {module.__class__.__name__}. Backward hooks may not trigger properly")
            #assert len(module.parameters(recurse=False)), "The input tensor to the module is a view, and autograd Function or register_hook is not triggered with view tensors."
            #if module.ds_grads_remaining == 0:
            #    print(f"Before Forward: {ctx.module.__class__.__name__}")
            module.ds_grads_remaining += 1
            ctx.pre_backward_function = pre_backward_function
        output = output.detach()
        return output

    @staticmethod
    def backward(ctx, *args):
        # gd.debuginfo(prj="ds")
        ctx.module.ds_grads_remaining = ctx.module.ds_grads_remaining - 1
        if ctx.module.ds_grads_remaining == 0:
            gd.debuginfo(prj="ds")
            ctx.pre_backward_function(ctx.module)
            #print(f"After Backward: {ctx.module.__class__.__name__}")
        return (None, None) + args

# 4.2. DeepSpeedZeRoOffload
# Stage3 中对参数划分的的核心逻辑在 类 DeepSpeedZeRoOffload 中实现，
# 这个类在文件 deepspeed.runtime.zero.parameter_offload 中， 同样，查看这个类的初始化方法。
class DeepSpeedZeRoOffload(object):

    def __init__(self,
                 module,
                 timers,
                 ds_config,
                 overlap_comm=True,
                 prefetch_bucket_size=50000000,
                 max_reuse_distance=1000000000,
                 max_live_parameters=1000000000,
                 param_persistence_threshold=100000,
                 model_persistence_threshold=sys.maxsize,
                 offload_param_config=None,
                 mpu=None,
                 zero_param_parallel_group=None,
                 zero_quantized_weights=False):
        # gd.debuginfo(prj="ds")

        see_memory_usage("DeepSpeedZeRoOffload initialize [begin]", force=True)

        print_rank_0(f"initialized {__class__.__name__} with args: {locals()}", force=False)

        self.module = module
        self.timers = timers
        self.dtype = list(module.parameters())[0].dtype
        self.offload_device = None
        self.offload_param_pin_memory = False
        self.zero_param_parallel_group = zero_param_parallel_group
        self.zero_quantized_weights = zero_quantized_weights

        if offload_param_config is not None and offload_param_config.device != OffloadDeviceEnum.none:
            gd.debuginfo(prj="ds")
            self.offload_device = offload_param_config.device
            self.offload_param_pin_memory = offload_param_config.pin_memory

        # 参数分割的逻辑
        self._convert_to_zero_parameters(ds_config, module, mpu)

        for m in module.modules():
            _init_external_params(m)

        _inject_parameters(module, ZeROOrderedDict)

        self.param_numel_persistence_threshold = int(param_persistence_threshold)
        self.model_persistence_threshold = int(model_persistence_threshold)
        self.persistent_parameters = self.mark_persistent_parameters(self.param_numel_persistence_threshold,
                                                                     self.model_persistence_threshold)

        self.param_coordinators = {}
        self._prefetch_bucket_sz = int(prefetch_bucket_size)
        self._max_reuse_distance_in_numel = int(max_reuse_distance)
        self._max_available_parameters_in_numel = int(max_live_parameters)
        self.__allgather_stream = get_accelerator().Stream() if overlap_comm else get_accelerator().default_stream()

        if not hasattr(module, "ds_inflight_param_registry"):
            gd.debuginfo(prj="ds")
            module.ds_inflight_param_registry = dict()
            # we need two registries, one for training and one for eval. They will be used when creating PartitionedParameterCoordinator
            module.ds_inflight_param_registry[True] = InflightParamRegistry()
            module.ds_inflight_param_registry[False] = InflightParamRegistry()
        self.__inflight_param_registry = module.ds_inflight_param_registry

        self.forward_hooks = []
        self.backward_hooks = []
        self.setup_zero_stage3_hooks()
        print_rank_0(
            f'Created module hooks: forward = {len(self.forward_hooks)}, backward = {len(self.backward_hooks)}',
            force=False)

        see_memory_usage("DeepSpeedZeRoOffload initialize [end]", force=True)

    @instrument_w_nvtx
    def partition_all_parameters(self):
        gd.debuginfo(prj="ds")
        """Partitioning Parameters that were not partitioned usually if parameters
        of modules whose input parameters do not require grad computation do not
        trigger post call and will therefore will remain unpartitioned"""
        self.get_param_coordinator(training=self.module.training).release_and_reset_all(self.module)
        for param in iter_params(self.module, recurse=True):
            if param.ds_status != ZeroParamStatus.NOT_AVAILABLE:
                raise RuntimeError(f"{param.ds_summary()} expected to be released")

    def get_param_coordinator(self, training):
        gd.debuginfo(prj="ds")
        if not training in self.param_coordinators:
            gd.debuginfo(prj="ds")
            self.param_coordinators[training] = PartitionedParameterCoordinator(
                prefetch_bucket_sz=self._prefetch_bucket_sz,
                max_reuse_distance_in_numel=self._max_reuse_distance_in_numel,
                max_available_parameters_in_numel=self._max_available_parameters_in_numel,
                allgather_stream=self.__allgather_stream,
                inflight_param_registry=self.__inflight_param_registry[training],
                prefetch_nvme=self.offload_device == OffloadDeviceEnum.nvme,
                timers=self.timers,
            )

        return self.param_coordinators[training]

    def empty_partition_cache(self):
        gd.debuginfo(prj="ds")
        self.partition_all_parameters()

    # __init__ 方法中调用了方法 _convert_to_zero_parameters ， 从方法名字也能看出就是我们想要的。
    def _convert_to_zero_parameters(self, ds_config, module, mpu):
        # 没有 ds_id 属性的参数，即还没有被 deepspeed 处理过的参数
        non_zero_params = [p for p in module.parameters() if not is_zero_param(p)]
        if non_zero_params: # 存在没有处理过的参数
            gd.debuginfo(prj="ds")
            zero_params = [p for p in module.parameters() if is_zero_param(p)]
            if zero_params: # 存已经处理过参数
                # 这意味着这个模型，之前被初始化过，现在额外多了一些未处理的新参数
                # 为了保持一致，调用原先的处理过程
                gd.debuginfo(prj="ds")
                zero_params[0].convert_to_zero_parameters(param_list=non_zero_params)
            else:  # 模型参数是首次初始化
                gd.debuginfo(prj="ds")
                group = None
                if mpu:
                    gd.debuginfo(prj="ds")
                    group = mpu.get_data_parallel_group()

                # 核心的参数分割过程在这里,来自 deepspeed.runtime.zero.partition_parameters.Init
                Init(module=module,
                     data_parallel_group=group,
                     dtype=self.dtype,
                     config_dict_or_path=ds_config,
                     remote_device=self.offload_device,
                     pin_memory=self.offload_param_pin_memory,
                     mpu=mpu,
                     zero_param_parallel_group=self.zero_param_parallel_group,
                     zero_quantized_weights=self.zero_quantized_weights)
                # 这个方法最终又跳到了 Init 类，创建了 Init 实例， 真实够绕的。

    def destroy(self):
        gd.debuginfo(prj="ds")
        self._remove_module_hooks()

    def _remove_module_hooks(self):
        gd.debuginfo(prj="ds")
        num_forward_hooks = len(self.forward_hooks)
        num_backward_hooks = len(self.backward_hooks)

        for hook in self.forward_hooks:
            gd.debuginfo(prj="ds")
            hook.remove()

        for hook in self.backward_hooks:
            gd.debuginfo(prj="ds")
            hook.remove()

        print_rank_0(f'Deleted module hooks: forward = {num_forward_hooks}, backward = {num_backward_hooks}',
                     force=False)

    '''
    5. Stage3 - hook 注册
    参数分割之后，在执行前向、后向之前，需要先把参数再还原回来。 
    同理，在执行前向后向之后，还要释放掉各自不需要的参数。 这里利用 pytorch 的 hook 功能在上述四个关键节点插入相关的动作。 
    pytorch 的 Module 类型提供了一系列 register_xxx_hook 方法来实现 hook 功能。
    
    deepspeed 的 hook 动作都在类 DeepSpeedZeRoOffload 中实现， 
    具体的在方法 DeepSpeedZeRoOffload::setup_zero_stage3_hooks 中。
    '''
    def setup_zero_stage3_hooks(self):
        gd.debuginfo(prj="ds")
        """
        注册 stage3 相关的hook函数
        Returns:

        """
        self.hierarchy = 0

        #reset step if in inference mode
        @instrument_w_nvtx
        def _end_of_forward_hook(module, *args):
            if not torch._C.is_grad_enabled():
                self.get_param_coordinator(training=False).reset_step()

        #likely one of them should be enough but just to be safe
        # 注册各种 钩子 hook ，
        # 包括 pre_forward、pre_backward、post_forward、post_backward
        self._register_hooks_recursively(self.module)
        self.module.register_forward_hook(_end_of_forward_hook)

        # Add top module to stack trace
        global FWD_MODULE_STACK
        FWD_MODULE_STACK.append(self.module)

    def mark_persistent_parameters(self, param_threshold, model_threshold):
        gd.debuginfo(prj="ds")
        persistent_params = []
        total_persistent_parameters = 0
        params_count = 0
        for name, param in self.module.named_parameters(recurse=True):
            if param.ds_numel + total_persistent_parameters > model_threshold:
                continue

            if param.ds_numel <= param_threshold:
                params_count += 1
                param.ds_persist = True
                persistent_params.append(param)
                total_persistent_parameters += param.ds_numel

        print_rank_0(
            f"Parameter Offload: Total persistent parameters: {total_persistent_parameters} in {params_count} params",
            force=True)

        return persistent_params

    def _register_hooks_recursively(self, module, count=[0]):
        # gd.debuginfo(prj="ds")
        """真正执行hook操作"""
        my_count = count[0]
        module.id = my_count

        #print(f"{module.__class__} : {module.id}")

        for child in module.children():
            count[0] = count[0] + 1
            self._register_hooks_recursively(child, count=count)

        @instrument_w_nvtx
        def _pre_forward_module_hook(module, *args):
            gd.debuginfo(prj="ds")
            self.pre_sub_module_forward_function(module)

        @instrument_w_nvtx
        def _post_forward_module_hook(module, input, output):
            global FWD_MODULE_STACK
            FWD_MODULE_STACK.pop()
            if output is None:
                gd.debuginfo(prj="ds")
                output = []
            elif not isinstance(output, (list, tuple)):
                if torch.is_tensor(output):
                    gd.debuginfo(prj="ds")
                    output = [output]
                else:
                    gd.debuginfo(prj="ds")
                    #print(f'got UNKNOWN type {type(output)}')
                    outputs = []
                    output = output if isinstance(output, dict) else vars(output)
                    for name, val in output.items():
                        if not name.startswith('__') and torch.is_tensor(val):
                            outputs.append(val)
                    output = outputs

            for item in filter(lambda item: is_zero_param(item) or hasattr(item, 'ds_param_alias'), output):
                key = id(item) if hasattr(item, 'ds_id') else id(item.ds_param_alias)
                actual_external_param = item if hasattr(item, 'ds_id') else item.ds_param_alias

                if not any(key in m._external_params for m in FWD_MODULE_STACK):
                    actual_external_param.is_external_param = True
                    module_to_register = FWD_MODULE_STACK[-1]
                    register_external_parameter(module_to_register, actual_external_param)
                    print_rank_0(
                        f'Registering dangling parameter for module {module_to_register.__class__.__name__}, ds_id = {actual_external_param.ds_id}.',
                        force=False)

                    # It's possible that the parameter was already external to the completed module. If so, remove it the
                    # registration as it will be covered by the outer module instead.
                    if key in module._external_params:
                        print_rank_0(
                            f'  Unregistering nested dangling parameter from module {module.__class__.__name__}, ds_id = {actual_external_param.ds_id}',
                            force=False)
                        unregister_external_parameter(module, actual_external_param)

                    actual_external_param.all_gather()

            self.post_sub_module_forward_function(module)

        def _pre_backward_module_hook(module, inputs, output):

            @instrument_w_nvtx
            def _run_before_backward_function(sub_module):
                gd.debuginfo(prj="ds")
                # some models (e.g. Albert) may run multiple forwards on the same layer in a loop
                # before doing backwards, so each backward will need a pre-fetch - using reference
                # counting to support this scenario
                #print(f"COUNTER before: {sub_module.applied_pre_backward_ref_cnt}")
                if sub_module.applied_pre_backward_ref_cnt > 0:
                    gd.debuginfo(prj="ds")
                    self.pre_sub_module_backward_function(sub_module)
                    sub_module.applied_pre_backward_ref_cnt -= 1
                #print(f"COUNTER after: {sub_module.applied_pre_backward_ref_cnt}")

            return _apply_to_tensors_only(module, PreBackwardFunction, _run_before_backward_function, output)

        #This is an alternate to doing _post_backward_module_hook
        #it uses tensor.register_hook instead of using torch.autograd.Function
        def _alternate_post_backward_module_hook(module, inputs):
            gd.debuginfo(prj="ds")
            module.ds_grads_remaining = 0

            #print(f"Before Forward {module.__class__.__name__}")

            def _run_after_backward_hook(*unused):
                gd.debuginfo(prj="ds")
                module.ds_grads_remaining = module.ds_grads_remaining - 1
                if module.ds_grads_remaining == 0:
                    gd.debuginfo(prj="ds")
                    #print(f"After backward {module.__class__.__name__}")
                    self.post_sub_module_backward_function(module)

            def _run_before_forward_function(input):
                gd.debuginfo(prj="ds")
                if input.requires_grad:
                    gd.debuginfo(prj="ds")
                    module.ds_grads_remaining += 1

            return _apply_forward_and_backward_to_tensors_only(module, _run_before_forward_function,
                                                               _run_after_backward_hook, inputs)

        def _post_backward_module_hook(module, inputs):
            gd.debuginfo(prj="ds")
            module.ds_grads_remaining = 0

            @instrument_w_nvtx
            def _run_after_backward_function(sub_module):
                # gd.debuginfo(prj="ds")
                if sub_module.ds_grads_remaining == 0:
                    # gd.debuginfo(prj="ds")
                    self.post_sub_module_backward_function(sub_module)

            return _apply_to_tensors_only(module, PostBackwardFunction, _run_after_backward_function, inputs)

        # Pre forward hook
        self.forward_hooks.append(module.register_forward_pre_hook(_pre_forward_module_hook))

        # Post forward hook
        self.forward_hooks.append(module.register_forward_hook(_post_forward_module_hook))

        # Pre backward hook
        self.backward_hooks.append(module.register_forward_hook(_pre_backward_module_hook))

        # post backward hook
        self.backward_hooks.append(module.register_forward_pre_hook(_post_backward_module_hook))

    '''
    6. Stage3 - 前后向过程
    在 Hook 章节（ 节 5 ） 我们看到对每个 Module 的前后向过程都进行了 hook ，
    在 Module::forward 前后都会执行特定的动作， 也就是在这里需要完成被分割参数的还原与重新释放。 
    并且关键的动作都在类 PartitionedParameterCoordinator 中， 本章我们研究前向过程。
    '''

    '''
    前向过程之前 pre_forward
    
    显然，在执行前向过程之前，我们需要 把被分割的参数还原回来，这里自然通过 AllGather 通信还原本层的参数。 
    这里通过 module.register_forward_pre_hook(_pre_forward_module_hook) 进行注册， 
    顺着 _pre_forward_module_hook 跟踪下去：
    '''

    '''
    6.1. 参数还原
    理论上，前后向过程前都要还原参数， 先把 hook 的函数贴过来，
    '''
    @torch.no_grad()
    def pre_sub_module_forward_function(self, sub_module):
        gd.debuginfo(prj="ds")
        see_memory_usage(f"Before sub module function {sub_module.__class__.__name__}", force=False)

        global FWD_MODULE_STACK
        FWD_MODULE_STACK.append(sub_module)

        param_coordinator = self.get_param_coordinator(training=sub_module.training)
        param_coordinator.trace_prologue(sub_module)
        if param_coordinator.is_record_trace():
            gd.debuginfo(prj="ds")
            param_coordinator.record_module(sub_module)

        # 真正的参数聚合动作是在这里
        # param_coordinator 的类型是 deepspeed.runtime.zero.PartitionedParameterCoordinator
        param_coordinator.fetch_sub_module(sub_module, forward=True)

        see_memory_usage(f"Before sub module function {sub_module.__class__.__name__} after fetch", force=False)

    '''
    6.2. 参数重新分割
    同样先把直接 hook 的函数贴过来，
    
    显然核心的逻辑在 param_coordinator.release_sub_module(sub_module, backward=False) ，接下来跟进去分析。
    '''
    # 前向过程之后 post_forward
    @torch.no_grad()
    def post_sub_module_forward_function(self, sub_module):
        gd.debuginfo(prj="ds")
        see_memory_usage(f"After sub module function {sub_module.__class__.__name__} {sub_module.id} before release",
                         force=False)
        # 重新释放参数
        param_coordinator = self.get_param_coordinator(training=sub_module.training)
        # 具体操作都在 deepspeed.runtime.zero.PartitionedParameterCoordinator::release_sub_module
        param_coordinator.release_sub_module(sub_module, backward=False)

        see_memory_usage(f"After sub module function {sub_module.__class__.__name__}  {sub_module.id} after release",
                         force=False)

    # 后向过程之前 pre_backward
    @torch.no_grad()
    def pre_sub_module_backward_function(self, sub_module):
        gd.debuginfo(prj="ds")
        assert sub_module.training, "backward pass is invalid for module in evaluation mode"
        param_coordinator = self.get_param_coordinator(training=True)
        param_coordinator.trace_prologue(sub_module)
        if param_coordinator.is_record_trace():
            gd.debuginfo(prj="ds")
            param_coordinator.record_module(sub_module)
        param_coordinator.fetch_sub_module(sub_module, forward=False)

    #后向过程之后 post_backward
    @torch.no_grad()
    def post_sub_module_backward_function(self, sub_module):
        gd.debuginfo(prj="ds")
        assert sub_module.training, "backward pass is invalid for module in evaluation mode"
        see_memory_usage(
            f"After sub module backward function {sub_module.__class__.__name__} {sub_module.id} before release",
            force=False)

        self.get_param_coordinator(training=True).release_sub_module(sub_module, backward=True)

        see_memory_usage(
            f"After sub module backward function {sub_module.__class__.__name__} {sub_module.id} after release",
            force=False)
'''
可以看到每一个节点的实现其实都在 deepspeed.runtime.zero.PartitionedParameterCoordinator 里面， 最终会跳转这个类里去执行。
'''