# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed.accelerator.abstract_accelerator import DeepSpeedAccelerator
import oneccl_bindings_for_pytorch  # noqa: F401 # type: ignore
import psutil
import os
from pydebug import debuginfo

# accelerator for Intel CPU
class CPU_Accelerator(DeepSpeedAccelerator):

    def __init__(self):
        debuginfo(prj='ds', info='CPU_Accelerator init')
        self._name = 'cpu'
        self._communication_backend_name = 'ccl'
        self.max_mem = psutil.Process().memory_info().rss

    def is_synchronized_device(self):
        return True

    # Device APIs
    def device_name(self, device_index=None):
        return 'cpu'

    def device(self, device_index=None):
        return None

    def set_device(self, device_index):
        return

    def current_device(self):
        return os.environ.get('LOCAL_RANK', 0)

    def current_device_name(self):
        return 'cpu'

    def device_count(self):
        device_count = int(os.environ.get('LOCAL_SIZE', 0))
        if device_count > 0:
            debuginfo(prj='ds')
            return device_count
        else:
            debuginfo(prj='ds')
            from deepspeed.utils.numa import get_numa_cores
            # Count NUMA node for number of cpu accelerators. On machine with HBM
            # In flat mode, HBM is in separate NUMA node with no cores on this node.
            # Ignore these NUMA nodes with no cores.
            numa_core_lists = get_numa_cores()
            numa_count = 0
            prev_core_list = []
            for core_list in numa_core_lists:
                if len(core_list) > 0 and core_list != prev_core_list:
                    numa_count += 1
                    prev_core_list = core_list
            return numa_count

    def synchronize(self, device_index=None):
        return

    # RNG APIs
    def random(self):
        return torch.random

    def set_rng_state(self, new_state, device_index=None):
        if device_index == None:
            return torch.set_rng_state(new_state)
        return torch.set_rng_state(new_state, device_index)

    def get_rng_state(self, device_index=None):
        return torch.get_rng_state()

    def manual_seed(self, seed):
        return torch.manual_seed(seed)

    def manual_seed_all(self, seed):
        return torch.manual_seed(seed)

    def initial_seed(self, seed):
        return torch.initial_seed(seed)

    def default_generator(self, device_index):
        return torch.default_generator

    # Streams/Events
    @property
    def Stream(self):
        return None

    def stream(self, stream):
<<<<<<< HEAD
        debuginfo(prj='ds')
        from deepspeed.runtime.utils import noop_decorator
        return noop_decorator
=======
        from deepspeed.runtime.utils import noop_context
        return noop_context()
>>>>>>> 388c84834fca87465aff8bb8f6d85be88fa82ba6

    def current_stream(self, device_index=None):
        return None

    def default_stream(self, device_index=None):
        return None

    @property
    def Event(self):
        return None

    # Memory management
    def empty_cache(self):
        return

    def get_rss(self):
        debuginfo(prj='ds')
        mem = psutil.Process().memory_info().rss
        if mem > self.max_mem:
            self.max_mem = mem
        return mem

    def reset_rss(self):
        debuginfo(prj='ds')
        mem = psutil.Process().memory_info().rss
        self.max_mem = mem
        return mem

    def memory_allocated(self, device_index=None):
        return self.get_rss()

    def max_memory_allocated(self, device_index=None):
        self.get_rss()
        return self.max_mem

    def reset_max_memory_allocated(self, device_index=None):
        self.reset_rss()
        return

    def memory_cached(self, device_index=None):
        return self.get_rss()

    def max_memory_cached(self, device_index=None):
        self.get_rss()
        return self.max_mem

    def reset_max_memory_cached(self, device_index=None):
        self.reset_rss()
        return

    def memory_stats(self, device_index=None):
        mem = self.get_rss()
        mem_stat = {}
        mem_stat['allocated_bytes.all.current'] = mem
        mem_stat['allocated_bytes.all.peak'] = self.max_mem
        return mem_stat

    def reset_peak_memory_stats(self, device_index=None):
        self.reset_rss()
        return

    def memory_reserved(self, device_index=None):
        return self.get_rss()

    def max_memory_reserved(self, device_index=None):
        self.get_rss()
        return self.max_mem

    def total_memory(self, device_index=None):
        return psutil.virtual_memory().total

    # Misc
    def amp(self):
        return torch.cpu.amp

    def is_available(self):
        return True

    def range_push(self, msg):
        # TODO itt is currently not supported yet
        # return torch.profiler.itt.range_push(msg)
        return

    def range_pop(self):
        # TODO itt is currently not supported yet
        # return torch.profiler.itt.range_pop()
        return

    def lazy_call(self, callback):
        return callback()

    def communication_backend_name(self):
        return self._communication_backend_name

    def is_triton_supported(self):
        return False

    # Data types
    def is_bf16_supported(self):
        return True

    def is_fp16_supported(self):
        return False

    def supported_dtypes(self):
        return [torch.float, torch.bfloat16]

    # Tensor operations

    @property
    def BFloat16Tensor(self):
        return torch.BFloat16Tensor

    @property
    def ByteTensor(self):
        return torch.ByteTensor

    @property
    def DoubleTensor(self):
        return torch.DoubleTensor

    @property
    def FloatTensor(self):
        return torch.FloatTensor

    @property
    def HalfTensor(self):
        return torch.HalfTensor

    @property
    def IntTensor(self):
        return torch.IntTensor

    @property
    def LongTensor(self):
        return torch.LongTensor

    def pin_memory(self, tensor):
        return tensor

    def op_builder_dir(self):
        debuginfo(prj='ds')
        try:
            # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
            # if successful this also means we're doing a local install and not JIT compile path
            from op_builder import __deepspeed__  # noqa: F401 # type: ignore
            return "op_builder.cpu"
        except ImportError:
            return "deepspeed.ops.op_builder.cpu"

    def on_accelerator(self, tensor):
        debuginfo(prj='ds')
        device_str = str(tensor.device)
        if device_str.startswith('cpu'):
            return True
        else:
            return False

    # create an instance of op builder and return, name specified by class_name
    def create_op_builder(self, op_name):
        debuginfo(prj='ds')
        builder_class = self.get_op_builder(op_name)
        if builder_class != None:
            return builder_class()
        return None

    # return an op builder class, name specified by class_name
    def get_op_builder(self, class_name):
        debuginfo(prj='ds')
        try:
            # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
            # if successful this also means we're doing a local install and not JIT compile path
            from op_builder import __deepspeed__  # noqa: F401 # type: ignore
            from op_builder.cpu import CCLCommBuilder, FusedAdamBuilder, CPUAdamBuilder, NotImplementedBuilder
        except ImportError:
            from deepspeed.ops.op_builder.cpu import CCLCommBuilder, FusedAdamBuilder, CPUAdamBuilder, NotImplementedBuilder

        if class_name == "CCLCommBuilder":
            return CCLCommBuilder
        elif class_name == "FusedAdamBuilder":
            return FusedAdamBuilder
        elif class_name == "CPUAdamBuilder":
            return CPUAdamBuilder
        else:
            # return a NotImplementedBuilder to avoid get NoneType[Name] in unit tests
            return NotImplementedBuilder

    def build_extension(self):
        debuginfo(prj='ds')
        from torch.utils.cpp_extension import BuildExtension
        return BuildExtension
