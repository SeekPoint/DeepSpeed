# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed import utils

from .utils import *
from .backend import *
from .comm import *
import os

DS_COMM_ALL_GATHER_OFF = False
DS_COMM_REDUCE_SCATTER_OFF = False
DS_COMM_BROADCAST_OFF = False
DS_COMM_ALL_REDUCE_OFF = False
DS_COMM_REDUCE_OFF = False
from pydebug import debuginfo

def is_torch_two():
    TORCH_MAJOR = int(torch.__version__.split('.')[0])
    if TORCH_MAJOR >= 2:
        return True
    else:
        return False


def torch_ver_ge_1_13():
    if is_torch_two():
        return True
    else:
        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        assert TORCH_MAJOR == 1

        TORCH_MINOR = int(torch.__version__.split('.')[1])
        return TORCH_MINOR >= 13


def has_coalescing_manager():
    debuginfo(prj='ds')
    has_c10d = hasattr(torch.distributed, 'distributed_c10d')
    return has_c10d and hasattr(torch.distributed.distributed_c10d, '_coalescing_manager')


def has_all_reduce_coalesced():
    debuginfo(prj='ds')
    return hasattr(torch.distributed, "all_reduce_coalesced") and torch_ver_ge_1_13()


def get_coalescing_manager(group, device, reqs):
    if is_torch_two():
        debuginfo(prj='ds')
        return torch.distributed.distributed_c10d._coalescing_manager(group, device=device, reqs=reqs)
    else:
        debuginfo(prj='ds')
        return torch.distributed.distributed_c10d._coalescing_manager(group, reqs)


##Utilities to turn comm off
##TODO: move to base comm (wrapper)
def all_gather_comm_off(flag=False):
    debuginfo(prj='ds')
    global DS_COMM_ALL_GATHER_OFF
    DS_COMM_ALL_GATHER_OFF = flag


def reduce_scatter_comm_off(flag=False):
    debuginfo(prj='ds')
    global DS_COMM_REDUCE_SCATTER_OFF
    DS_COMM_REDUCE_SCATTER_OFF = flag


def broadcast_comm_off(flag=False):
    debuginfo(prj='ds')
    global DS_COMM_BROADCAST_OFF
    DS_COMM_BROADCAST_OFF = flag


def all_reduce_comm_off(flag=False):
    debuginfo(prj='ds')
    global DS_COMM_ALL_REDUCE_OFF
    DS_COMM_ALL_REDUCE_OFF = flag


def reduce_comm_off(flag=False):
    debuginfo(prj='ds')
    global DS_COMM_REDUCE_OFF
    DS_COMM_REDUCE_OFF = flag


#assumption: all_gather and reduce scatter
## are what we care about
def backward_comm_off(flag=False):
    debuginfo(prj='ds')
    all_gather_comm_off(flag)
    reduce_scatter_comm_off(flag)


class Noop:

    def wait(self):
        return None


class TorchBackend(Backend):
    """
        A light-weight wrapper class for torch.distributed API.
        Only a subset of functions are wrapped. Once the init_process_group
        is initialized, standard torch.distributed.* can be used directly
        so no need to wrap all the functions. We can keep adding wrappers as
        needed.
    """

    def __init__(self, backend, timeout, init_method, rank=-1, world_size=-1, name='torch'):
        debuginfo(prj='ds', info='TorchBackend init')
        super(TorchBackend, self).__init__()
        self.has_all_reduce_coalesced = has_all_reduce_coalesced()
        self.has_coalescing_manager = has_coalescing_manager()
        self.all_gather_function = self.get_all_gather_function()
        self.reduce_scatter_function = self.get_reduce_scatter_function()
        self.initialized = True
        self.name = name
        # Future functionality to support ds.initialize() on a single GPU
        # The idea is to fake that dist backend is initialized even when
        # it is not so we can run on a single GPU without doing any init_process_group
        self.single_gpu_mode = True
        self.init_process_group(backend, timeout, init_method, rank, world_size)

    @classmethod
    def get_all_gather_function(self):
        if hasattr(torch.distributed, "all_gather_into_tensor"):
            debuginfo(prj='ds')
            return torch.distributed.all_gather_into_tensor
        elif hasattr(torch.distributed, "_all_gather_base"):
            debuginfo(prj='ds')
            return torch.distributed._all_gather_base
        return None

    @classmethod
    def get_reduce_scatter_function(self):
        if hasattr(torch.distributed, "reduce_scatter_tensor"):
            debuginfo(prj='ds')
            return torch.distributed.reduce_scatter_tensor
        elif hasattr(torch.distributed, "_reduce_scatter_base"):
            debuginfo(prj='ds')
            return torch.distributed._reduce_scatter_base
        return None

    def has_all_gather_into_tensor(self):
        debuginfo(prj='ds')
        return self.all_gather_function is not None

    def has_reduce_scatter_tensor(self):
        debuginfo(prj='ds')
        return self.reduce_scatter_function is not None

    def init_process_group(self, backend, timeout, init_method, rank, world_size):
        debuginfo(prj='ds')
        if not torch.distributed.is_initialized():
            debuginfo(prj='ds')
            torch.distributed.init_process_group(backend,
                                                 timeout=timeout,
                                                 init_method=init_method,
                                                 rank=rank,
                                                 world_size=world_size)
        self.using_mpi = torch.distributed.get_backend() == 'mpi'

    def all_reduce(self, tensor, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False):
        debuginfo(prj='ds')
        op = self._reduce_op(op)
        return torch.distributed.all_reduce(tensor=tensor, op=op, group=group, async_op=async_op)

    def all_reduce_coalesced(self, tensors, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False):
        """ proxy func to torch.distributed.all_reduce_coalesced,
        which is included in PyTorch 1.13 and above
        """
        debuginfo(prj='ds')
        if not self.has_all_reduce_coalesced:
            raise RuntimeError(f"Current torch version does not have all_reduce_coalesced "
                               f"api (torch.__version__: {torch.__version__})")
        op = self._reduce_op(op)
        return torch.distributed.all_reduce_coalesced(tensors=tensors, op=op, group=group, async_op=async_op)

    def reduce(self, tensor, dst, op=ReduceOp.SUM, group=None, async_op=False):
        debuginfo(prj='ds')
        if DS_COMM_REDUCE_OFF:
            if int(os.getenv('RANK', '0')) == 0:
                utils.logger.warning("REDUCE is OFF")
            return Noop()
        return torch.distributed.reduce(tensor=tensor, dst=dst, op=self._reduce_op(op), group=group, async_op=async_op)

    def reduce_scatter(self, output, input_list, op=ReduceOp.SUM, group=None, async_op=False):
        if DS_COMM_REDUCE_SCATTER_OFF:
            debuginfo(prj='ds')
            if int(os.getenv('RANK', '0')) == 0:
                utils.logger.warning("REDUCE SCATTER  is OFF")
            return Noop()
        else:
            debuginfo(prj='ds')
            return torch.distributed.reduce_scatter(output=output,
                                                    input_list=input_list,
                                                    op=self._reduce_op(op),
                                                    group=group,
                                                    async_op=async_op)

    def broadcast(self, tensor, src, group=None, async_op=False):
        if DS_COMM_BROADCAST_OFF:
            # debuginfo(prj='ds')
            if int(os.getenv('RANK', '0')) == 0:
                utils.logger.warning("BROADCAST  is OFF")
            return Noop()
        else:
            # debuginfo(prj='ds')
            return torch.distributed.broadcast(tensor=tensor, src=src, group=group, async_op=async_op)

    def all_gather(self, tensor_list, tensor, group=None, async_op=False):
        if DS_COMM_ALL_GATHER_OFF:
            debuginfo(prj='ds')
            if int(os.getenv('RANK', '0')) == 0:
                utils.logger.warning("All Gather is OFF")
            return Noop()
        else:
            debuginfo(prj='ds')
            return torch.distributed.all_gather(tensor_list=tensor_list, tensor=tensor, group=group, async_op=async_op)

    def all_gather_into_tensor(self, output_tensor, input_tensor, group=None, async_op=False):
        if self.has_all_gather_into_tensor():
            debuginfo(prj='ds')
            return self.all_gather_function(output_tensor=output_tensor,
                                            input_tensor=input_tensor,
                                            group=group,
                                            async_op=async_op)

    def all_gather_base(self, output_tensor, input_tensor, group=None, async_op=False):
        if DS_COMM_ALL_GATHER_OFF:
            debuginfo(prj='ds')
            if int(os.getenv('RANK', '0')) == 0:
                utils.logger.warning("All Gather is OFF")
            return Noop()
        else:
            if self.has_allgather_base:
                debuginfo(prj='ds')
                return torch.distributed.distributed_c10d._all_gather_base(output_tensor=output_tensor,
                                                                           input_tensor=input_tensor,
                                                                           group=group,
                                                                           async_op=async_op)
            else:
                utils.logger.warning("unable to find torch.distributed._all_gather_base. will fall back to "
                                     "torch.distributed.reduce_scatter which will result in suboptimal performance. "
                                     "please consider upgrading your pytorch installation.")
                pass

    def all_gather_coalesced(self, output_tensors, input_tensors, group=None, async_op=False):
        """"""
        assert len(output_tensors) == len(input_tensors), ""
        if hasattr(torch.distributed.distributed_c10d, '_all_gather_base_coalesced'):
            debuginfo(prj='ds')
            # customized PyTorch
            return torch.distributed.distributed_c10d._all_gather_base_coalesced(output_tensors,
                                                                                 input_tensors,
                                                                                 group=group,
                                                                                 async_op=async_op)
        elif has_coalescing_manager():
            debuginfo(prj='ds')
            reqs = []
            with get_coalescing_manager(group, input_tensors[0].device, reqs):
                for output, input in zip(output_tensors, input_tensors):
                    handle = torch.distributed.distributed_c10d.all_gather_into_tensor(output,
                                                                                       input,
                                                                                       group=group,
                                                                                       async_op=True)
                    reqs.append(handle)
            if async_op:
                debuginfo(prj='ds')
                return reqs[-1]
            else:
                debuginfo(prj='ds')
                reqs[-1].wait()

    def reduce_scatter_tensor(self, output_tensor, input_tensor, op=ReduceOp.SUM, group=None, async_op=False):
        if self.has_reduce_scatter_tensor():
            debuginfo(prj='ds')
            return self.reduce_scatter_function(output_tensor,
                                                input_tensor,
                                                op=self._reduce_op(op),
                                                group=group,
                                                async_op=async_op)
        else:
            utils.logger.warning("unable to find torch.distributed.reduce_scatter_tensor. will fall back to "
                                 "torch.distributed.reduce_scatter which will result in suboptimal performance. "
                                 "please consider upgrading your pytorch installation.")
            pass

    def all_to_all_single(self,
                          output,
                          input,
                          output_split_sizes=None,
                          input_split_sizes=None,
                          group=None,
                          async_op=False):
        debuginfo(prj='ds')
        return torch.distributed.all_to_all_single(output=output,
                                                   input=input,
                                                   output_split_sizes=output_split_sizes,
                                                   input_split_sizes=input_split_sizes,
                                                   group=group,
                                                   async_op=async_op)

    def send(self, tensor, dst, group=None, tag=0):
        debuginfo(prj='ds')
        return torch.distributed.send(tensor=tensor, dst=dst, group=group, tag=tag)

    def recv(self, tensor, src=None, group=None, tag=0):
        debuginfo(prj='ds')
        return torch.distributed.recv(tensor=tensor, src=src, group=group, tag=tag)

    def isend(self, tensor, dst, group=None, tag=0):
        debuginfo(prj='ds')
        return torch.distributed.isend(tensor=tensor, dst=dst, group=group, tag=tag)

    def irecv(self, tensor, src=None, group=None, tag=0):
        debuginfo(prj='ds')
        return torch.distributed.irecv(tensor=tensor, src=src, group=group, tag=tag)

    def gather(self, tensor, gather_list=None, dst=0, group=None, async_op=False):
        debuginfo(prj='ds')
        return torch.distributed.gather(tensor=tensor,
                                        gather_list=gather_list,
                                        dst=dst,
                                        group=group,
                                        async_op=async_op)

    def scatter(self, tensor, scatter_list=None, src=0, group=None, async_op=False):
        debuginfo(prj='ds')
        return torch.distributed.scatter(tensor=tensor,
                                         scatter_list=scatter_list,
                                         src=src,
                                         group=group,
                                         async_op=async_op)

    def barrier(self, group=torch.distributed.GroupMember.WORLD, async_op=False, device_ids=None):
        if group is None:
            debuginfo(prj='ds')
            group = torch.distributed.GroupMember.WORLD
        debuginfo(prj='ds')
        return torch.distributed.barrier(group=group, async_op=async_op, device_ids=device_ids)

    def monitored_barrier(self, group=torch.distributed.GroupMember.WORLD, timeout=None, wait_all_ranks=False):
        if group is None:
            debuginfo(prj='ds')
            group = torch.distributed.GroupMember.WORLD
        debuginfo(prj='ds')
        return torch.distributed.monitored_barrier(group=group, timeout=timeout, wait_all_ranks=wait_all_ranks)

    def get_rank(self, group=None):
        # debuginfo(prj='ds')
        return torch.distributed.get_rank(group=group)

    def get_world_size(self, group=None):
        # debuginfo(prj='ds')
        return torch.distributed.get_world_size(group=group)

    def is_initialized(self):
        # debuginfo(prj='ds')
        return torch.distributed.is_initialized()

    def get_backend(self, group=None):
        debuginfo(prj='ds')
        return torch.distributed.get_backend(group=group)

    def new_group(self, ranks):
        debuginfo(prj='ds')
        return torch.distributed.new_group(ranks)

    def get_global_rank(self, group, group_rank):
        if hasattr(torch.distributed.distributed_c10d, "get_global_rank"):
            # debuginfo(prj='ds')
            from torch.distributed.distributed_c10d import get_global_rank as _get_global_rank
        else:
            # debuginfo(prj='ds')
            from torch.distributed.distributed_c10d import _get_global_rank
        return _get_global_rank(group, group_rank)

    def get_world_group(self):
        debuginfo(prj='ds')
        return torch.distributed.group.WORLD

    def destroy_process_group(self, group=None):
        debuginfo(prj='ds')
        return torch.distributed.destroy_process_group(group=group)

    def _reduce_op(self, op):
        '''
            Helper function. If the op provided is not a torch.dist.ReduceOp, convert it and return
        '''
        if not isinstance(op, torch.distributed.ReduceOp):
            if op == ReduceOp.SUM:
                op = torch.distributed.ReduceOp.SUM
            elif op == ReduceOp.PRODUCT:
                op = torch.distributed.ReduceOp.PRODUCT
            elif op == ReduceOp.AVG:
                op = torch.distributed.ReduceOp.AVG
            elif op == ReduceOp.MIN:
                op = torch.distributed.ReduceOp.MIN
            elif op == ReduceOp.MAX:
                op = torch.distributed.ReduceOp.MAX
            elif op == ReduceOp.BAND:
                op = torch.distributed.ReduceOp.BAND
            elif op == ReduceOp.BOR:
                op = torch.distributed.ReduceOp.BOR
            elif op == ReduceOp.BXOR:
                op = torch.distributed.ReduceOp.BXOR
        debuginfo(prj='ds', info=f'op is {op}')
        return op


# This will become a light-weight wrapper around torch.distributed functions
# TODO: create some example to show how this wrapper can help profile communication
# TODO: make sure there is no performance regression with this approach
# TODO: explore monkey-patching if this does not work
