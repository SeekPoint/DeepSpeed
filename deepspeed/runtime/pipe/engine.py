# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from types import MethodType

import torch
from deepspeed import comm as dist

from deepspeed.utils import logger
from deepspeed.utils.timer import ThroughputTimer
from deepspeed.accelerator import get_accelerator

from ..engine import DeepSpeedEngine, MEMORY_OPT_ALLREDUCE_SIZE
from ..utils import PartitionedTensor
from ..dataloader import RepeatingLoader
from ..zero.config import ZeroStageEnum
from ..activation_checkpointing import checkpointing as ds_checkpointing

from .module import PipelineModule, PipelineError
from . import p2p
from . import schedule

TARGET_ID = -2
LOG_STAGE = -2
DATA_PARALLEL_ID = -2


def is_even(number):
    return number % 2 == 0


mem_alloced = 0
mem_cached = 0
from pydebug import gd, infoTensor

def _tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


class PipelineEngine(DeepSpeedEngine):
    """ A training engine hybrid pipeline, data, and model parallel training.

    This engine is created by ``deepspeed.initialize()`` when a :class:`PipelineModule`
    is provided.
    """
    ID_TO_DTYPE = [
        torch.float32, torch.float64, torch.complex64, torch.complex128, torch.float16, torch.bfloat16, torch.uint8,
        torch.int8, torch.int16, torch.int32, torch.int64, torch.bool
    ]
    DTYPE_TO_ID = {dtype: id_ for id_, dtype in enumerate(ID_TO_DTYPE)}

    def __init__(self, has_bool_tensors=False, *super_args, **super_kwargs):
        gd.debuginfo(prj="ds", info=f'__FUNC_IN_OUT__') # ppl
        super().__init__(*super_args, **super_kwargs)
        assert isinstance(self.module, PipelineModule), "model must base PipelineModule"

        assert self.zero_optimization_stage() < 2, "ZeRO-2 and ZeRO-3 are incompatible with pipeline parallelism"

        # We schedule the all-reduces, so disable it in super().backward()
        self.enable_backward_allreduce = False
        self.has_bool_tensors = has_bool_tensors
        self.eval_return_logits = False
        self.outputs = None

        # used to disable the pipeline all-reduce when used with 1-bit Adam/1-bit LAMB
        self.pipeline_enable_backward_allreduce = True

        if self.elasticity_enabled():
            if not self.is_elastic_model_parallel_supported():
                assert not self.elasticity_enabled(), "Elasticity is not currently supported" \
                " with pipeline parallelism."

        # pipeline step for logging
        self.log_batch_step_id = -1

        self.micro_batch_size = self.train_micro_batch_size_per_gpu()
        self.micro_batches = self.gradient_accumulation_steps()

        # Set Grid and Communication Groups
        self.grid = self.module._grid
        # if self.grid.get_global_rank() == 0:

        self.global_rank = self.grid.get_global_rank()
        gd.debuginfo(prj="ds", info=f'self.micro_batches={self.micro_batches}, '
                                    f'self.micro_batch_size={self.micro_batch_size}, '
                                    f'self.grid={self.grid}, '
                                    f'self.global_rank={self.global_rank}')

        assert self.dp_world_size == self.grid.data_parallel_size
        assert self.train_batch_size() == \
            self.micro_batch_size * self.micro_batches * self.grid.data_parallel_size

        #  Set Stage Inf
        self.num_stages = self.grid.pipe_parallel_size
        self.stage_id = self.grid.get_stage_id()
        self.prev_stage = self.stage_id - 1
        self.next_stage = self.stage_id + 1

        gd.debuginfo(prj="ds", info=f'self.num_stages={self.num_stages}, '
                                    f'self.stage_id={self.stage_id}, '
                                    f'self.prev_stage={self.prev_stage}, '
                                    f'self.next_stage={self.next_stage}')

        self.data_iterator = None
        self.batch_fn = None

        self._force_grad_boundary = False

        self.batch_timer = ThroughputTimer(batch_size=self.train_batch_size(),
                                           logging_fn=self.tput_log,
                                           monitor_memory=False,
                                           steps_per_output=self.steps_per_print())

        # PipelineEngine needs to handle data loading specially due to only the first
        # and last stages loading inputs/labels. We construct a sampler that uses
        if self.training_data:
            gd.debuginfo(prj="ds")
            self._build_data_iter(self.training_data)

        self.is_pipe_parallel = self.grid.pipe_parallel_size > 1
        self.is_data_parallel = self.grid.data_parallel_size > 1
        self.is_model_parallel = self.grid.model_parallel_size > 1

        gd.debuginfo(prj="ds", info=f'self.is_pipe_parallel={self.is_pipe_parallel}, '
                                    f'self.is_data_parallel={self.is_data_parallel}, '
                                    f'self.is_model_parallel={self.is_model_parallel}')

        # Partition input/output buffers
        # XXX temporarily disable while I revert some partition hacks.
        self.is_pipe_partitioned = self.is_model_parallel
        self.is_grad_partitioned = self.is_model_parallel

        gd.debuginfo(prj="ds", info=f'self.is_pipe_partitioned={self.is_pipe_partitioned}, '
                                    f'self.is_grad_partitioned={self.is_grad_partitioned}')

        model_parameters = filter(lambda p: p.requires_grad, self.module.parameters())
        num_params = sum([p.numel() for p in model_parameters])

        gd.debuginfo(prj="ds", info=f'model_parameters={model_parameters}')
        gd.debuginfo(prj="ds", info=f'num_params={num_params}')

        unique_params = num_params
        # Subtract tied parameters if we don't own them
        if self.module.tied_comms:
            tied_params = 0
            for key, d in self.module.tied_comms.items():
                if self.global_rank != min(d['ranks']):
                    tied_params += sum(p.numel() for p in d['module'].parameters())
            unique_params -= tied_params

        params_tensor = torch.LongTensor(data=[num_params, unique_params]).to(self.device)
        gd.debuginfo(prj="ds", info=f'params_tensor={infoTensor(params_tensor)}')

        tmp = self.grid.get_model_parallel_group()
        gd.debuginfo(prj="ds", info=f'tmp={tmp}')
        dist.all_reduce(params_tensor, group=tmp)

        params_tensor = params_tensor.tolist()
        gd.debuginfo(prj="ds", info=f'params_tensor={params_tensor}')

        total_params = params_tensor[0]
        unique_params = params_tensor[1]
        gd.debuginfo(prj="ds", info=f'total_params={total_params}, unique_params={unique_params}')

        # if self.grid.data_parallel_id == 0:
        gd.debuginfo(prj="ds", info=f'RANK={self.global_rank} '
                    f'STAGE={self.stage_id} '
                    f'LAYERS={self.module._local_stop - self.module._local_start} '
                    f'[{self.module._local_start}, {self.module._local_stop}) '
                    f'STAGE_PARAMS={num_params} ({num_params/1e6:0.3f}M) '
                    f'TOTAL_PARAMS={total_params} ({total_params/1e6:0.3f}M) '
                    f'UNIQUE_PARAMS={unique_params} ({unique_params/1e6:0.3f}M)')

        #initialize peer-2-peer communication and allreduce groups
        if self.is_pipe_parallel:
            p2p.init_process_groups(self.grid)

        # Pipeline buffers
        self.num_pipe_buffers = 0
        self.pipe_buffers = {
            'inputs': [],  # batch input and received activations
            'labels': [],  # labels from batch input
            'outputs': [],  # activations
            'output_tensors': [],  # tensor object to preserve backward graph
        }
        self.pipe_recv_buf = None
        self.grad_layer = None

        self.meta_buffer = None

        self.first_output_send = True
        self.first_gradient_send = True

        #stores the loss for the current micro batch being processed
        self.loss = torch.tensor(0.0).to(self.device)

        #stores the loss for the entire batch
        self.total_loss = None

        self.agg_loss = torch.tensor(0.0, requires_grad=False).to(self.device)
        gd.debuginfo(prj="ds", info=f'self.agg_loss={infoTensor(self.agg_loss)}')

        self.dp_group_loss = torch.tensor(0.0, requires_grad=False).to(self.device)
        gd.debuginfo(prj="ds", info=f'self.dp_group_loss={infoTensor(self.dp_group_loss)}')

        gd.debuginfo(prj="ds", info=f'self._config={self._config}') # _config来自基类成员

        if self._config.pipeline['activation_checkpoint_interval'] > 0:
            gd.debuginfo(prj="ds")
            self.module.activation_checkpoint_interval = self._config.pipeline['activation_checkpoint_interval']

        self.module.checkpoint_parallel_write_pipeline = self._config.checkpoint_parallel_write_pipeline

        if self.is_last_stage():
            gd.debuginfo(prj="ds")
            self.loss_model = self.module.loss_fn

        self.has_attention_mask = self.module.__class__.__name__ == 'GPT2ModelPipe'
        gd.debuginfo(prj="ds", info=f'self.has_attention_mask={self.has_attention_mask}')

        # Initialize pipeline communicators. Just send a 0.
        if is_even(self.stage_id):
            if not self.is_last_stage():
                gd.debuginfo(prj="ds")
                p2p.send(self.loss, self.next_stage)
            if not self.is_first_stage():
                gd.debuginfo(prj="ds")
                p2p.recv(self.loss, self.prev_stage)
        else:
            if not self.is_first_stage():
                gd.debuginfo(prj="ds")
                p2p.recv(self.loss, self.prev_stage)
            if not self.is_last_stage():
                gd.debuginfo(prj="ds")
                p2p.send(self.loss, self.next_stage)

        # XXX look into timer reporting timing
        # Initialize some timers because of early weirdness.
        if self.wall_clock_breakdown():
            gd.debuginfo(prj="ds")
            self.timers('forward_microstep').start()
            self.timers('forward_microstep').stop()
            self.timers('backward_microstep').start()
            self.timers('backward_microstep').stop()
            self.timers('backward_inner_microstep').start()
            self.timers('backward_inner_microstep').stop()
            self.timers('backward_allreduce_microstep').start()
            self.timers('backward_allreduce_microstep').stop()
            self.timers('backward_allreduce').start()
            self.timers('backward_allreduce').stop()
            self.timers('step_microstep').start()
            self.timers('step_microstep').stop()

        gd.debuginfo(prj="ds", info=f'__FUNC_IN_OUT__')

    def set_has_attention_mask(self, value):
        gd.debuginfo(prj="ds")
        assert isinstance(value, bool)
        self.has_attention_mask = value

    def _build_data_iter(self, dataset):

        sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                  num_replicas=self.dp_world_size,
                                                                  rank=self.mpu.get_data_parallel_rank(),
                                                                  shuffle=False)
        # Build a loader and make it repeating.
        pipe_dataloader = self.deepspeed_io(dataset, data_sampler=sampler)
        pipe_dataloader = RepeatingLoader(pipe_dataloader)

        gd.debuginfo(prj="ds", info=f'sampler={sampler}')
        gd.debuginfo(prj="ds", info=f'pipe_dataloader={pipe_dataloader}')
        gd.debuginfo(prj="ds", info=f'pipe_dataloader={pipe_dataloader}')

        self.set_dataloader(pipe_dataloader)

    def _exec_reduce_tied_grads(self):

        # We need to run this first to write to self.averaged_gradients;
        # since this class turns `enable_backward_allreduce` off,
        # `self.overlapping_partition_gradients_reduce_epilogue()` defined in the DeepSpeedEngine
        # never actually runs. I suspect this is because of efficiency problems; get_flat_partition in
        # stage2.py might do something expensive; someone will have to look into that later. But
        # in the meantime, this fixes ZeRO2 + Pipelining enough to run a demo. Further profiling
        # needed to decide if it actually breaks everything.
        # (see https://github.com/EleutherAI/gpt-neox/issues/62#issuecomment-761471944)
        if self.zero_optimization_partition_gradients():
            gd.debuginfo(prj="ds")
            self.optimizer.overlapping_partition_gradients_reduce_epilogue()

        weight_group_list = self.module.get_tied_weights_and_groups()
        gd.debuginfo(prj="ds", info=f'weight_group_list={weight_group_list}')

        for weight, group in weight_group_list:
            gd.debuginfo(prj="ds", info=f'weight={weight}, group={group}')
            grad = weight._hp_grad if self.bfloat16_enabled() else weight.grad
            dist.all_reduce(grad, group=group)

    def _exec_reduce_grads(self):
        gd.debuginfo(prj="ds")
        self._force_grad_boundary = True
        if self.pipeline_enable_backward_allreduce:
            if self.bfloat16_enabled():
                if self.zero_optimization_stage() < ZeroStageEnum.gradients:
                    gd.debuginfo(prj="ds")
                    self._bf16_reduce_grads()
                else:
                    raise NotImplementedError("PP+BF16 only work for ZeRO Stage 1")
            else:
                gd.debuginfo(prj="ds")
                self.allreduce_gradients(bucket_size=MEMORY_OPT_ALLREDUCE_SIZE)
        self._force_grad_boundary = False

    def _bf16_reduce_grads(self):
        gd.debuginfo(prj="ds")
        # Make our own list of gradients from the optimizer's FP32 grads
        grads = []
        self.buffered_allreduce_fallback(grads=self.optimizer.get_grads_for_reduction(),
                                         elements_per_buffer=MEMORY_OPT_ALLREDUCE_SIZE)

    def _reserve_pipe_buffers(self, num_buffers):
        """Ensure that each pipeline buffer has at least ``num_buffers`` slots.

        This method only reserves slots and does not allocate tensors.

        Args:
            num_buffers (int): The number of buffers to reserve.
        """
        if self.num_pipe_buffers >= num_buffers:
            gd.debuginfo(prj="ds")
            return

        num_added = num_buffers - self.num_pipe_buffers
        gd.debuginfo(prj="ds", info=f'num_added={num_added}')

        for key in self.pipe_buffers:
            gd.debuginfo(prj="ds", info=f'key={key}')
            self.pipe_buffers[key].extend([None] * num_added)
        self.num_pipe_buffers = num_buffers

    def reset_activation_shape(self):
        """Reset the buffers when the shape of activation and gradient change.
        For example, for curriculum learning that changes the seqlen of each
        sample, we need to call this whenever the seqlen is going to change.
        """
        gd.debuginfo(prj="ds")
        self.first_output_send = True
        self.pipe_recv_buf = None
        self.grad_layer = None
        self.meta_buffer = None

    def train_batch(self, data_iter=None):
        """Progress the pipeline to train the next batch of data. The engine will ingest
        ``self.train_batch_size()`` total samples collectively across all workers.


        An iterator that over training data should be provided as an argument
        unless ``deepspeed.initialize()`` was provided a training set. In that event,
        the training data will automatically be read.


        .. warning::
            A total of ``self.gradient_accumulation_steps()`` entries will be pulled
            from ``data_iter`` by each pipeline. There must be sufficient
            data left in ``data_iter`` or else a ``StopIteration`` will halt training.

            DeepSpeed provides a convenience class :class:`deepspeed.utils.RepeatingLoader`
            that wraps data loaders to automatically restart upon a ``StopIteration``.

        Args:
            data_iter (Iterator, optional): Iterator of training data.

        Returns:
            The arithmetic mean of the losses computed this batch.
        """
        if not torch._C.is_grad_enabled():
            raise RuntimeError(f'train_batch() requires gradients enabled. Use eval_batch() instead.')

        # Curriculum learning could change activation shape
        if self.curriculum_enabled_legacy():
            new_difficulty = self.curriculum_scheduler_legacy.update_difficulty(self.global_steps + 1)
            gd.debuginfo(prj="ds", info=f'new_difficulty={new_difficulty}')

            if self.global_steps == 0 or self.curriculum_scheduler_legacy.first_step:
                gd.debuginfo(prj="ds")
                self.reset_activation_shape()
                self.curriculum_scheduler_legacy.first_step = False
            elif new_difficulty != self.curriculum_scheduler_legacy.get_difficulty(self.global_steps):
                gd.debuginfo(prj="ds")
                self.reset_activation_shape()

        if data_iter:
            gd.debuginfo(prj="ds")
            self.set_dataiterator(data_iter)

        gd.debuginfo(prj="ds", info='---------------1------------------')
        self.module.train()
        self.total_loss = None
        self._compute_loss = True

        gd.debuginfo(prj="ds", info='---------------2------------------')

        # Do the work
        self.timers('train_batch').start()
        sched = schedule.TrainSchedule(micro_batches=self.micro_batches,
                                       stages=self.num_stages,
                                       stage_id=self.stage_id)
        '''
        1.2.4.1 schedule.TrainSchedule(micro_batches=self.micro_batches..) runtime schedule 将会自动调用 steps 函数
        首先根据传入的 micro batch size 和 batch size 得到 num micro batch，既可以得到每个 stage 有多少个 micro batch
        根据计算的 num micro batch 和 1F1B 策略得到一个 stage 的步数，在此例子中，每个stage 的步数为 6.
        为每个 stage 的步，标一个 index，为 micro batch id，并为这些 id 计算出 cmds。
        cmds 是计算出的每一步需要做的函数。
        例如 stage 0 的 step 0 ，需要 load data 和 进行第一步 forward。
        
        接下来每个 stage 将跟据生成的 cmds，在各自的 rank 里串行的执行每一步。然后每个 stage 有事并行执行的。
        前面的计算就是为了不同 stage 之间可以同步进行，
        例如 stage 0 的 step 1 send activation 后，
        stage 1 step 2 需要执行玩 step 0 的 load 和 forward，
        并几乎没有延时地执行 receive activation。这样的话，就可以有效率的run 整个网络。
        '''
        gd.debuginfo(prj="ds", info=f'sched={sched}')

        gd.debuginfo(prj="ds", info='---------------3------------------')
        self._exec_schedule(sched)
        gd.debuginfo(prj="ds", info='---------------4------------------')
        self.agg_train_loss = self._aggregate_total_loss()
        gd.debuginfo(prj="ds", info=f'self.agg_train_loss={self.agg_train_loss}')

        self.timers('train_batch').stop()

        if self.global_steps % self.steps_per_print() == 0:
            if self.global_rank == 0:
                elapsed = self.timers('train_batch').elapsed(reset=True) / 1000.0
                iter_time = elapsed / self.steps_per_print()
                tput = self.train_batch_size() / iter_time
                gd.debuginfo(prj="ds", info=f'steps: {self.global_steps}, '
                                            f'loss: {self.agg_train_loss:0.4f} '
                                            f'iter time (s): {iter_time:0.3f} '
                                            f'samples/sec: {tput:0.3f}')

        # Monitoring
        if self.global_rank == 0 and self.monitor.enabled:
            self.summary_events = [(f'Train/Samples/train_loss', self.agg_train_loss.mean().item(),
                                    self.global_samples)]
            self.monitor.write_events(self.summary_events)

        if self.wall_clock_breakdown() and self.global_steps % self.steps_per_print() == 0:
            self.timers.log(['pipe_send_output', 'pipe_send_grad', 'pipe_recv_input', 'pipe_recv_grad'])

        # TODO: should return precisely what loss returned and allow others to be queried?
        return self.agg_train_loss

    def eval_batch(self, data_iter, return_logits=False, compute_loss=True, reduce_output='avg'):
        """Evaluate the pipeline on a batch of data from ``data_iter``. The
        engine will evaluate ``self.train_batch_size()`` total samples
        collectively across all workers.

        This method is equivalent to:

        .. code-block:: python

            module.eval()
            with torch.no_grad():
                output = module(batch)

        .. warning::
            A total of ``self.gradient_accumulation_steps()`` entries will be pulled
            from ``data_iter`` by each pipeline. There must be sufficient
            data left in ``data_iter`` or else a ``StopIteration`` will halt training.

            DeepSpeed provides a convenience class :class:`deepspeed.utils.RepeatingLoader`
            that wraps data loaders to automatically restart upon a ``StopIteration``.

        Args:
            data_iter (Iterator): Iterator of data to evaluate.

        Returns:
            The arithmetic mean of the losses computed this batch.
        """
        self.eval_return_logits = return_logits
        gd.debuginfo(prj="ds", info=f'self.eval_return_logits={self.eval_return_logits}')
        self.module.eval()
        gd.debuginfo(prj="ds", info=f'++++++++++++++++++++1++++++++++++++++++++++++++')

        # Curriculum learning could change activation shape
        if self.curriculum_enabled_legacy():
            new_difficulty = self.curriculum_scheduler_legacy.update_difficulty(self.global_steps + 1)
            gd.debuginfo(prj="ds", info=f'new_difficulty={new_difficulty}')
            if self.global_steps == 0 or self.curriculum_scheduler_legacy.first_step:
                gd.debuginfo(prj="ds")
                self.reset_activation_shape()
                self.curriculum_scheduler_legacy.first_step = False
            elif new_difficulty != self.curriculum_scheduler_legacy.get_difficulty(self.global_steps):
                gd.debuginfo(prj="ds")
                self.reset_activation_shape()

        eval_output = None

        self._compute_loss = compute_loss
        gd.debuginfo(prj="ds", info=f'self._compute_loss={compute_loss}')

        # Use the provided data iterator
        train_iterator = self.data_iterator
        self.set_dataiterator(data_iter)
        gd.debuginfo(prj="ds", info=f'++++++++++++++++++++2++++++++++++++++++++++++++')

        # Do the work
        sched = schedule.InferenceSchedule(micro_batches=self.micro_batches,
                                           stages=self.num_stages,
                                           stage_id=self.stage_id)
        gd.debuginfo(prj="ds", info=f'sched={sched}')

        # prevent dead-lock with multiple evals sequence
        dist.barrier()

        gd.debuginfo(prj="ds", info=f'++++++++++++++++++++3++++++++++++++++++++++++++')

        with torch.no_grad():
            self._exec_schedule(sched)

        gd.debuginfo(prj="ds", info=f'++++++++++++++++++++4++++++++++++++++++++++++++')

        if self.is_last_stage():
            eval_output = self._reduce_outputs(self.fwd_outputs, reduce=reduce_output)
            gd.debuginfo(prj="ds", info=f'A-eval_output={eval_output}')

        gd.debuginfo(prj="ds", info=f'++++++++++++++++++++5++++++++++++++++++++++++++')

        if compute_loss:
            eval_output = self._bcast_pipe_scalar(eval_output)
            gd.debuginfo(prj="ds", info=f'B-eval_output={eval_output}')

        gd.debuginfo(prj="ds", info=f'++++++++++++++++++++6++++++++++++++++++++++++++')

        if self.global_rank == 0 and self.monitor.enabled:
            self.summary_events = [(f'Train/Samples/eval_loss', eval_output.mean().item(), self.global_samples)]
            self.monitor.write_events(self.summary_events)

        # Restore the training iterator
        self.set_dataiterator(train_iterator)

        gd.debuginfo(prj="ds", info=f'++++++++++++++++++++7++++++++++++++++++++++++++')

        # Reset any buffers that may have been populated during the forward passes.
        #ds_checkpointing.reset()
        self.eval_return_logits = False
        if return_logits:
            outputs = self.outputs
            self.outputs = None
            gd.debuginfo(prj="ds", info=f'outputs={outputs}, eval_output={eval_output}')
            return eval_output, outputs

        gd.debuginfo(prj="ds", info=f'eval_output={eval_output}')

        return eval_output

    def set_train_batch_size(self, train_batch_size):
        """Adjust the global batch size by increasing or decreasing the number of
        micro-batches (i.e., gradient accumulation steps). The size of each micro-batch
        (i.e., ``train_micro_batch_size_per_gpu``) is not changed.
        Args:
            train_batch_size (int): The new global batch size for training.
        Raises:
            ValueError: if ``train_batch_size`` is not divisible by the
                configured micro-batch size and data parallelism.
        """
        gd.debuginfo(prj="ds")
        super().set_train_batch_size(train_batch_size)
        gd.debuginfo(prj="ds", info=f'++++++++++++++++++++8++++++++++++++++++++++++++')
        self.micro_batches = self.gradient_accumulation_steps()
        gd.debuginfo(prj="ds", info=f'self.micro_batches={self.micro_batches}')

    def is_first_stage(self):
        """True if this process is in the first stage in the pipeline."""
        gd.debuginfo(prj="ds", info=f'self.stage_id={self.stage_id == 0}')
        return self.stage_id == 0

    def is_last_stage(self):
        """True if this process is in the last stage in the pipeline."""
        gd.debuginfo(prj="ds", info=f'self.stage_id={self.num_stages - 1}')
        return self.stage_id == self.num_stages - 1

    def _reduce_outputs(self, outputs, reduce='avg', reduce_dp=True):
        if reduce is None:
            gd.debuginfo(prj="ds")
            return outputs

        if reduce.lower() == 'avg':
            # first sum over all microbatches
            if torch.is_tensor(outputs[0]):
                reduced = sum(outputs)
                gd.debuginfo(prj="ds", info=f'A-reduced={reduced}')
            else:
                assert isinstance(outputs, (list, tuple))
                reduced = [torch.zeros_like(o) for o in outputs[0]]
                gd.debuginfo(prj="ds", info=f'B-reduced={reduced}')
                for idx, out in outputs:
                    reduced[idx] += out
                    gd.debuginfo(prj="ds", info=f'out={out}, reduced[{idx}]={reduced[idx]}')

            # Average over the microbatches
            reduced = self._scale_loss_by_gas(reduced)
            gd.debuginfo(prj="ds", info=f'C-reduced={reduced}')

            # Average over DP groups
            if reduce_dp and self.is_data_parallel:
                if torch.is_tensor(reduced):
                    dist.all_reduce(reduced, group=self.mpu.get_data_parallel_group())
                    reduced /= self.dp_world_size
                    gd.debuginfo(prj="ds", info=f'D-reduced={reduced}')
                else:
                    for idx in range(len(reduced)):
                        dist.all_reduce(reduced[idx], group=self.mpu.get_data_parallel_group())
                        reduced[idx] /= self.dp_world_size
                        gd.debuginfo(prj="ds", info=f'reduced[{idx}]={reduced[idx]}')

            gd.debuginfo(prj="ds", info=f'return reduced={reduced}')
            return reduced
        else:
            raise NotImplementedError(f'reduction type {reduce} not supported.')

    def _bcast_pipe_scalar(self, data, src_rank=None, dtype=torch.float32):
        # Default to last stage (e.g., for broadcasting loss)
        if src_rank is None:
            src_rank = self.grid.stage_to_global(self.num_stages - 1)
            gd.debuginfo(prj="ds", info=f'src_rank={src_rank}')
        assert src_rank in self.grid.pp_group

        if self.global_rank == src_rank:
            result = data.clone().detach().type(dtype).to(self.device)
            gd.debuginfo(prj="ds", info=f'result={infoTensor(result)}')
        else:
            result = torch.Tensor([0.]).type(dtype).to(self.device)
            gd.debuginfo(prj="ds", info=f'result={infoTensor(result)}')

        dist.broadcast(tensor=result, src=src_rank, group=self.mpu.get_pipe_parallel_group())
        gd.debuginfo(prj="ds", info=f'++++++++++++++++++++9++++++++++++++++++++++++++')

        return result

    def _aggregate_total_loss(self):
        # Scale loss, average among DP ranks, and bcast loss to the rest of my DP group
        if self.is_last_stage():
            loss = self._scale_loss_by_gas(self.total_loss)
            self.dp_group_loss = loss.clone().detach()

            ## Average loss across all data-parallel groups
            agg_loss = self.dp_group_loss.clone().detach()

            gd.debuginfo(prj="ds", info=f'loss={loss}')
            gd.debuginfo(prj="ds", info=f'self.dp_group_loss={self.dp_group_loss}')
            gd.debuginfo(prj="ds", info=f'A-agg_loss={agg_loss}')

            gd.debuginfo(prj="ds", info=f'RANK={self.global_rank} bcast '
                                        f'SENDER src={self.global_rank} ,'
                                        f'group={self.grid.pp_group}')
            if self.is_data_parallel:
                dist.all_reduce(agg_loss, group=self.mpu.get_data_parallel_group())
                agg_loss /= self.dp_world_size
                gd.debuginfo(prj="ds", info=f'B-agg_loss={agg_loss}')

            assert self.global_rank in self.grid.pp_group
            losses = torch.Tensor([self.dp_group_loss, agg_loss]).to(self.device)
            gd.debuginfo(prj="ds", info=f'losses={infoTensor(losses)}')

            if self.is_pipe_parallel:
                gd.debuginfo(prj="ds", info=f'++++++++++++++++++++10++++++++++++++++++++++++++')
                dist.broadcast(tensor=losses, src=self.global_rank, group=self.mpu.get_pipe_parallel_group())
        else:
            # Get loss from last stage
            src_rank = self.grid.stage_to_global(self.num_stages - 1)
            gd.debuginfo(prj="ds", info=f'src_rank={src_rank}')

            assert src_rank in self.grid.pp_group
            losses = torch.Tensor([0., 0.]).to(self.device)
            gd.debuginfo(prj="ds", info=f'losses={infoTensor(losses)}')

            dist.broadcast(tensor=losses, src=src_rank, group=self.grid.get_pipe_parallel_group())
            gd.debuginfo(prj="ds", info=f'++++++++++++++++++++11++++++++++++++++++++++++++')

            self.dp_group_loss = losses[0].clone().detach()
            agg_loss = losses[1].clone().detach()
            gd.debuginfo(prj="ds", info=f'self.dp_group_loss={self.dp_group_loss}')
            gd.debuginfo(prj="ds", info=f'agg_loss={agg_loss}')

        return agg_loss

    def set_dataloader(self, loader):
        """"""
        gd.debuginfo(prj="ds")
        if self.is_first_stage() or self.is_last_stage():
            gd.debuginfo(prj="ds")
            self.training_dataloader = loader
            self.data_iterator = iter(self.training_dataloader)

    def set_dataiterator(self, iterator):
        """ Store an iterator to sample for training data. """
        gd.debuginfo(prj="ds")
        if self.is_first_stage() or self.is_last_stage():
            gd.debuginfo(prj="ds")
            self.training_dataloader = None
            self.data_iterator = iterator

    def set_batch_fn(self, fn):
        """Execute a post-processing function on input data.

        Args:
            fn (function): The function to run.
        """
        # gd.debuginfo(prj="ds")
        self.batch_fn = fn

    def is_gradient_accumulation_boundary(self):
        """True if the engine is executing a gradient reduction or optimizer step instruction.

        This is overridden from :class:`DeepSpeedEngine` to force reductions
        and steps when the pipeline engine is instructed to do so.

        Returns:
            bool: whether reductions and optimizer steps should occur.
        """
        gd.debuginfo(prj="ds", info=f'return {self._force_grad_boundary}')
        return self._force_grad_boundary

    def log_for_device(self, *msg):
        gd.debuginfo(prj="ds")
        if LOG_STAGE == self.stage_id or LOG_STAGE == -1:
            if DATA_PARALLEL_ID == self.grid.data_parallel_id or DATA_PARALLEL_ID == -1:
                gd.debuginfo(prj="ds", info=f'RANK={dist.get_rank()} '
                    f'PIPE-ID={self.stage_id} '
                    f'DATA-ID={self.grid.data_parallel_id} '
                    f'MBATCH-ID={self.microbatch_id} '
                    f'STEP-ID={self.log_batch_step_id} '
                    '::',
                    *msg)

    def tput_log(self, *msg):
        if self.global_rank == 0 and self.global_steps % self.steps_per_print() == 0:
            print(*msg)

    def _next_batch(self):
        # If using 3D parallelism, only some first-stage ranks may do IO
        batch = None
        if self.data_iterator is not None:
            batch = next(self.data_iterator)
            gd.debuginfo(prj="ds", info=f'A-len of batch={len(batch)}')

        # Any post-processing, like broadcasting across a slice-parallel group.
        if self.batch_fn:
            batch = self.batch_fn(batch)
            gd.debuginfo(prj="ds", info=f'B-batch={batch}')

        return batch

    def _exec_forward_pass(self, buffer_id):
        self.tput_timer.start()
        self.mem_status('BEFORE FWD', reset_max=True)

        if isinstance(self.pipe_buffers['inputs'][buffer_id], tuple):
            gd.debuginfo(prj="ds")
            inputs = tuple(t.clone() for t in self.pipe_buffers['inputs'][buffer_id])
        else:
            gd.debuginfo(prj="ds")
            inputs = self.pipe_buffers['inputs'][buffer_id].clone()

        gd.debuginfo(prj='ds', info=str(input))

        # collect the partitioned input from the previous stage
        if self.is_pipe_partitioned and not self.is_first_stage():

            part_input = PartitionedTensor.from_meta(meta=inputs[0],
                                                     local_part=inputs[1],
                                                     group=self.grid.get_slice_parallel_group())

            gd.debuginfo(prj="ds", info=f'part_input={part_input}')

            inputs = (part_input.full(), *inputs[2:])
            gd.debuginfo(prj="ds", info=f'A-inputs={inputs}')

            inputs[0].requires_grad = True
            # skip mask
            #inputs[1].requires_grad = True
            part_input = None
            inputs = inputs[0] if len(inputs) == 1 else inputs
            gd.debuginfo(prj="ds", info=f'B-inputs={inputs}')

            self.pipe_buffers['inputs'][buffer_id] = inputs

        # Zero out the gradients each time we use the tensor because only the data in
        # tensor changes across batches

        gd.debuginfo(prj="ds", info=f'#######1###########')
        self._zero_grads(inputs)
        gd.debuginfo(prj="ds", info=f'#######2###########')

        outputs = super().forward(inputs)
        gd.debuginfo(prj="ds", info=f'#######3###########')

        # Reset activation checkpointing buffers.
        # Need to call this between evaluation iterations
        if not self.module.training:
            gd.debuginfo(prj="ds")
            ds_checkpointing.reset()

        # Partition the outputs if we are not the last stage
        if self.is_pipe_partitioned and not self.is_last_stage():
            if isinstance(outputs, tuple):
                first_output = outputs[0]
                # TODO: Improve pipe partitioning to pass multiple tensors that require grads
                assert all([torch.is_tensor(elt) and elt.requires_grad is False for elt in outputs[1:]])
                outputs_tail = outputs[1:]
                gd.debuginfo(prj="ds", info=f'A-first_output={first_output}, outputs_tail={outputs_tail}')
            elif torch.is_tensor(outputs):
                first_output = outputs
                outputs_tail = []
                gd.debuginfo(prj="ds", info=f'B-first_output={first_output}')
            else:
                raise ValueError("expecting a tensor or a tuple of tensors")

            part = PartitionedTensor(tensor=first_output, group=self.grid.get_slice_parallel_group())
            gd.debuginfo(prj="ds", info=f'part={part}')

            # Clear the large output data, but save the computation graph
            first_output.data = torch.zeros(1)
            gd.debuginfo(prj="ds", info=f'first_output.data={infoTensor(first_output.data)}')

            self.pipe_buffers['output_tensors'][buffer_id] = first_output
            gd.debuginfo(prj="ds", info=f"self.pipe_buffers['output_tensors'][{buffer_id}]={infoTensor(first_output)}")

            # Inject the partitioned tensor into the output before sending
            outputs = (part.to_meta(), part.data(), *outputs_tail)
            part = None
            gd.debuginfo(prj="ds", info=f'part.to_meta()={infoTensor(part.to_meta())}')
            gd.debuginfo(prj="ds", info=f'part.data()={infoTensor(part.data())}')
            gd.debuginfo(prj="ds", info=f'*outputs_tail={infoTensor(*outputs_tail)}')

        self.pipe_buffers['outputs'][buffer_id] = outputs

        # Optionally compute loss on the last device
        if self.is_last_stage():
            if self._compute_loss and self.module.loss_fn is not None:
                labels = self.pipe_buffers['labels'][buffer_id]
                self.loss = self.module.loss_fn(outputs, labels)
                gd.debuginfo(prj="ds", info=f'labels={labels}, self.loss={self.loss}')
            else:
                # Some models just return loss from forward()
                self.loss = outputs
                gd.debuginfo(prj="ds", info=f'self.loss={self.loss}')
            if self.eval_return_logits:
                self.outputs = outputs
                gd.debuginfo(prj="ds", info=f'self.outputs={self.outputs}')
            if isinstance(self.loss, torch.Tensor):
                tmp = self.loss.detach()
                self.fwd_outputs.append(tmp)
                gd.debuginfo(prj="ds", info=f'A-self.loss.detach()={infoTensor(tmp)}')

                if self.total_loss is None:
                    self.total_loss = torch.zeros_like(self.loss)
                    gd.debuginfo(prj="ds", info=f'self.total_loss={infoTensor(self.total_loss)}')

                self.total_loss += self.loss.detach()
                gd.debuginfo(prj="ds", info=f'self.total_loss={infoTensor(self.total_loss)}')
            else:
                tmp = [l.detach() for l in self.loss]
                self.fwd_outputs.append(tmp)
                gd.debuginfo(prj="ds", info=f'tmp={infoTensor(tmp)}')

                if self.total_loss is None:
                    self.total_loss = [torch.zeros_like(l) for l in self.loss]
                    gd.debuginfo(prj="ds", info=f'self.total_loss={infoTensor(self.total_loss)}')
                for idx, l in enumerate(self.loss):
                    gd.debuginfo(prj="ds", info=f'idx={idx}, l={infoTensor(l)}')
                    self.total_loss[idx] += l.detach()

    def _exec_backward_pass(self, buffer_id):
        assert self.optimizer is not None, "must provide optimizer during " \
                                           "init in order to use backward"

        self.mem_status('BEFORE BWD', reset_max=True)

        # The last stage just runs backward on the loss using DeepSpeed's typical
        # mechanisms.
        if self.is_last_stage():
            gd.debuginfo(prj="ds")
            super().backward(self.loss)
            self.mem_status('AFTER BWD')
            return

        outputs = self.pipe_buffers['outputs'][buffer_id]
        gd.debuginfo(prj="ds", info=f"self.pipe_buffers['outputs'][{buffer_id}]={infoTensor(outputs)}")

        if self.wall_clock_breakdown():
            gd.debuginfo(prj="ds")
            self.timers('backward_microstep').start()
            self.timers('backward').start()
            self.timers('backward_inner_microstep').start()
            self.timers('backward_inner').start()

        # Reconstruct if we previously partitioned the output. We must be
        # careful to also restore the computational graph of the tensors we partitioned.
        if self.is_pipe_partitioned:
            if self.is_grad_partitioned:
                part_output = PartitionedTensor.from_meta(meta=outputs[0],
                                                          local_part=outputs[1],
                                                          group=self.grid.get_slice_parallel_group())
                gd.debuginfo(prj="ds", info=f"part_output={part_output}")
                self.pipe_buffers['output_tensors'][buffer_id].data = part_output.full()

                outputs = (self.pipe_buffers['output_tensors'][buffer_id], *outputs[2:])
                gd.debuginfo(prj="ds", info=f"self.pipe_buffers['output_tensors'][{buffer_id}]={infoTensor(outputs)}")
            else:
                # Already restored from partition
                self.pipe_buffers['output_tensors'][buffer_id].data = outputs[0]
                outputs = (self.pipe_buffers['output_tensors'][buffer_id], *outputs[1:])
                gd.debuginfo(prj="ds", info=f"self.pipe_buffers['output_tensors'][{buffer_id}]={infoTensor(outputs)}")

        grad_tensors = self.grad_layer
        gd.debuginfo(prj="ds", info=f"grad_tensors={infoTensor(grad_tensors)}")

        if self.is_grad_partitioned:
            gd.debuginfo(prj="ds", info=f'RANK={self.global_rank} BEFORE-BWD '
                                        f'restoring grad={self.grad_layer[0].size()} {self.grad_layer[1].size()}')
            part_grad = PartitionedTensor.from_meta(meta=self.grad_layer[0],
                                                    local_part=self.grad_layer[1],
                                                    group=self.grid.get_slice_parallel_group())
            gd.debuginfo(prj="ds", info=f"part_grad={infoTensor(part_grad)}")

            grad_tensors = (part_grad.full(), *grad_tensors[2:])
            gd.debuginfo(prj="ds", info=f"grad_tensors={infoTensor(grad_tensors)}")

            part_grad = None
            gd.debuginfo(prj="ds", info=f'RANK={self.global_rank} BEFORE-BWD '
                                        f'restored grad={self.grad_layer[0].size()} {self.grad_layer[1].size()}')

        if self.bfloat16_enabled() and not self.is_last_stage():
            gd.debuginfo(prj="ds")
            # manually call because we don't call optimizer.backward()
            self.optimizer.clear_lp_grads()

        # This handles either a single tensor or tuple of tensors.
        if isinstance(outputs, tuple):
            gd.debuginfo(prj="ds")
            out_tensors = [t for t in outputs if t.is_floating_point()]
            assert len(out_tensors) == len(grad_tensors)
            torch.autograd.backward(tensors=out_tensors, grad_tensors=grad_tensors)
        else:
            gd.debuginfo(prj="ds")
            torch.autograd.backward(tensors=(outputs, ), grad_tensors=(grad_tensors, ))

        if self.bfloat16_enabled() and not self.is_last_stage():
            gd.debuginfo(prj="ds")
            # manually call because we don't call optimizer.backward()
            self.optimizer.update_hp_grads(clear_lp_grads=False)

        # Free up the memory from the output of forward()
        self.pipe_buffers['output_tensors'][buffer_id] = None
        self.pipe_buffers['outputs'][buffer_id] = None
        grad_tensors = None

        if self.wall_clock_breakdown():
            gd.debuginfo(prj="ds")
            self.timers('backward_inner').stop()
            self.timers('backward_inner_microstep').stop()
            self.timers('backward').stop()
            self.timers('backward_microstep').stop()

        self.mem_status('AFTER BWD')

    def _exec_load_micro_batch(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers('batch_input').start()

        batch = self._next_batch()
        gd.debuginfo(prj="ds", info=f'len of batch={len(batch)}')

        if self.is_first_stage():
            loaded = None
            if torch.is_tensor(batch[0]):
                gd.debuginfo(prj="ds")
                loaded = batch[0].clone().to(self.device).detach()
                loaded.requires_grad = loaded.is_floating_point()
            else:
                gd.debuginfo(prj="ds")
                assert isinstance(batch[0], (tuple, list))
                # Assume list or tuple
                loaded = []
                for x in batch[0]:
                    assert torch.is_tensor(x)
                    mine = x.clone().detach().to(self.device)
                    mine.requires_grad = mine.is_floating_point()
                    loaded.append(mine)
                loaded = tuple(loaded)

            self.pipe_buffers['inputs'][buffer_id] = loaded

        if self.is_last_stage():
            loaded = batch[1]
            if torch.is_tensor(batch[1]):
                gd.debuginfo(prj="ds")
                loaded = batch[1].to(self.device)
            elif isinstance(batch[1], tuple):
                gd.debuginfo(prj="ds")
                loaded = []
                for x in batch[1]:
                    assert torch.is_tensor(x)
                    x = x.to(self.device).detach()
                    loaded.append(x)
                loaded = tuple(loaded)

            self.pipe_buffers['labels'][buffer_id] = loaded

        if self.wall_clock_breakdown():
            gd.debuginfo(prj="ds")
            self.timers('batch_input').stop()

    def _send_tensor_meta(self, buffer, recv_stage):
        """ Communicate metadata about upcoming p2p transfers.

        Metadata is communicated in this order:
            * type (0: tensor, 1: list)
            * num_tensors if type=list
            foreach tensor in buffer:
                * ndims
                * shape
        """
        gd.debuginfo(prj="ds", info=f'recv_stage={infoTensor(recv_stage)}')
        send_bytes = 0
        if isinstance(buffer, torch.Tensor):
            type_tensor = torch.LongTensor(data=[0]).to(self.device)
            gd.debuginfo(prj="ds", info=f'type_tensor={infoTensor(type_tensor)}')
            p2p.send(type_tensor, recv_stage)

            send_shape = torch.LongTensor(data=buffer.size()).to(self.device)
            send_ndims = torch.LongTensor(data=[len(buffer.size())]).to(self.device)
            gd.debuginfo(prj="ds", info=f'send_shape={infoTensor(send_shape)}')
            gd.debuginfo(prj="ds", info=f'send_ndims={infoTensor(send_ndims)}')

            p2p.send(send_ndims, recv_stage)
            p2p.send(send_shape, recv_stage)

            send_bytes += _tensor_bytes(buffer)
            gd.debuginfo(prj="ds", info=f'send_bytes={infoTensor(send_bytes)}')
        elif isinstance(buffer, list):
            assert (False)
            type_tensor = torch.LongTensor(data=[1]).to(self.device)
            gd.debuginfo(prj="ds", info=f'type_tensor={infoTensor(type_tensor)}')
            p2p.send(type_tensor, recv_stage)

            count_tensor = torch.LongTensor(data=[len(buffer)]).to(self.device)
            p2p.send(count_tensor, recv_stage)

            for index, tensor in enumerate(buffer):
                gd.debuginfo(prj="ds", info=f'index={index}, tensor={infoTensor(tensor)}')

                assert isinstance(tensor, torch.Tensor)

                send_shape = torch.LongTensor(data=tensor.size()).to(self.device)
                send_ndims = torch.LongTensor(data=[len(tensor.size())]).to(self.device)
                gd.debuginfo(prj="ds", info=f'send_shape={infoTensor(send_shape)}')
                gd.debuginfo(prj="ds", info=f'send_ndims={infoTensor(send_ndims)}')

                p2p.send(send_ndims, recv_stage)
                p2p.send(send_shape, recv_stage)

                send_bytes += _tensor_bytes(tensor)
                gd.debuginfo(prj="ds", info=f'send_bytes={infoTensor(send_bytes)}')

        elif isinstance(buffer, tuple):
            type_tensor = torch.LongTensor(data=[2]).to(self.device)
            gd.debuginfo(prj="ds", info=f'type_tensor={infoTensor(type_tensor)}')
            p2p.send(type_tensor, recv_stage)

            count_tensor = torch.LongTensor(data=[len(buffer)]).to(self.device)
            gd.debuginfo(prj="ds", info=f'count_tensor={infoTensor(count_tensor)}')

            p2p.send(count_tensor, recv_stage)
            for idx, tensor in enumerate(buffer):
                gd.debuginfo(prj="ds", idx=f'index={idx}, tensor={infoTensor(tensor)}')
                assert isinstance(tensor, torch.Tensor)
                send_shape = torch.LongTensor(data=tensor.size()).to(self.device)
                send_ndims = torch.LongTensor(data=[len(tensor.size())]).to(self.device)
                send_dtype = torch.LongTensor(data=[self.DTYPE_TO_ID[tensor.dtype]]).to(self.device)
                gd.debuginfo(prj="ds", idx=f'index={idx}, send_shape={infoTensor(send_shape)}')
                gd.debuginfo(prj="ds", idx=f'index={idx}, send_ndims={infoTensor(send_ndims)}')
                gd.debuginfo(prj="ds", idx=f'index={idx}, send_dtype={infoTensor(send_dtype)}')
                p2p.send(send_dtype, recv_stage)
                p2p.send(send_ndims, recv_stage)
                p2p.send(send_shape, recv_stage)
                # Useful for performance debugging.
                '''
                new_bytes = _tensor_bytes(tensor)
                send_bytes += _tensor_bytes(tensor)
                # Useful for performance debugging.
                if self.grid.data_parallel_id == 0:
                    print(
                        f'STAGE={self.stage_id} pipe-send-volume[{idx}]: shape={send_shape} {new_bytes/1024**2:0.2f}MB'
                    )
                '''
        else:
            raise NotImplementedError(f'Could not send meta type {type(buffer)}')

        # Useful for performance debugging.

        # if self.grid.data_parallel_id == 0:
        gd.debuginfo(prj='ds', info=f'STAGE={self.stage_id} pipe-send-volume: {send_bytes/1024**2:0.2f}MB')


    def _recv_tensor_meta(self, send_stage):
        """Receive metadata about upcoming p2p transfers and return allocated buffers.

        Metadata is communicated in this order:
            * type (0: tensor, 1: list)
            * num_tensors if type=list
            foreach tensor in buffer:
                * ndims
                * shape

        Returns:
            Allocated buffer for receiving from send_stage.
        """

        type_tensor = torch.LongTensor(data=[0]).to(self.device)
        gd.debuginfo(prj="ds", info=f'type_tensor={infoTensor(type_tensor)}, send_stage={send_stage}')

        p2p.recv(type_tensor, send_stage)
        recv_type = type_tensor.item()
        gd.debuginfo(prj="ds", info=f'recv_type={recv_type}')

        # A single tensor will be sent.
        if recv_type == 0:
            recv_ndims = torch.LongTensor(data=[0]).to(self.device)
            gd.debuginfo(prj="ds", info=f'recv_ndims={infoTensor(recv_ndims)}')

            p2p.recv(recv_ndims, send_stage)

            recv_ndims = recv_ndims.item()
            recv_shape = torch.LongTensor([1] * recv_ndims).to(self.device)
            gd.debuginfo(prj="ds", info=f'recv_ndims={infoTensor(recv_ndims)}, recv_shape={infoTensor(recv_shape)}')

            p2p.recv(recv_shape, send_stage)
            recv_shape = recv_shape.tolist()
            return self._allocate_buffer(recv_shape, num_buffers=1)[0]

        # List or tuple of tensors
        elif recv_type == 1 or recv_type == 2:
            count_tensor = torch.LongTensor(data=[0]).to(self.device)
            gd.debuginfo(prj="ds", info=f'count_tensor={infoTensor(count_tensor)}')

            p2p.recv(count_tensor, send_stage)
            num_tensors = count_tensor.item()
            gd.debuginfo(prj="ds", info=f'num_tensors={infoTensor(num_tensors)}')

            recv_shapes_and_dtypes = []
            for idx in range(num_tensors):
                recv_dtype = torch.LongTensor(data=[0]).to(self.device)
                gd.debuginfo(prj="ds", info=f'recv_dtype={infoTensor(recv_dtype)}')
                p2p.recv(recv_dtype, send_stage)

                recv_dtype = self.ID_TO_DTYPE[recv_dtype.item()]
                recv_ndims = torch.LongTensor(data=[0]).to(self.device)
                gd.debuginfo(prj="ds", info=f'recv_dtype={infoTensor(recv_dtype)}')
                gd.debuginfo(prj="ds", info=f'recv_ndims={infoTensor(recv_ndims)}')
                p2p.recv(recv_ndims, send_stage)

                recv_ndims = recv_ndims.item()
                recv_shape = torch.LongTensor([1] * recv_ndims).to(self.device)
                gd.debuginfo(prj="ds", info=f'recv_ndims={infoTensor(recv_ndims)}')
                gd.debuginfo(prj="ds", info=f'recv_shape={infoTensor(recv_shape)}')
                p2p.recv(recv_shape, send_stage)

                recv_shapes_and_dtypes.append((recv_shape.tolist(), recv_dtype))

            gd.debuginfo(prj="ds", info=f'len of recv_shapes_and_dtypes={len(recv_shapes_and_dtypes)}')
            buffers = self._allocate_buffers(recv_shapes_and_dtypes, num_buffers=1)[0]
            # Convert to tuples if requested.
            if recv_type == 2:
                gd.debuginfo(prj="ds")
                buffers = tuple(buffers)
            return buffers

        else:
            raise NotImplementedError(f'Could not receive type {type(recv_type)}')

    def _exec_send_activations(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers('pipe_send_output').start()

        outputs = self.pipe_buffers['outputs'][buffer_id]

        # NCCL does not like to send torch.BoolTensor types, so cast the mask to half().
        # We could do char, but with half() we can eventually flatten with other fp16
        # messages (TODO)
        if self.has_attention_mask or self.has_bool_tensors:
            gd.debuginfo(prj="ds")
            outputs = list(outputs)
            outputs[-1] = outputs[-1].half()
            outputs = tuple(outputs)

        if self.first_output_send:
            self.first_output_send = False
            gd.debuginfo(prj="ds", info=f'self.next_stage={self.next_stage}, outputs={infoTensor(outputs)}')
            self._send_tensor_meta(outputs, self.next_stage)

        if isinstance(outputs, torch.Tensor):
            gd.debuginfo(prj="ds", info=f'self.next_stage={self.next_stage}, outputs={infoTensor(outputs)}')
            p2p.send(outputs, self.next_stage)
        elif isinstance(outputs, tuple):
            for idx, buffer in enumerate(outputs):
                gd.debuginfo(prj="ds", info=f'idx={idx}, buffer={infoTensor(buffer)}')
                p2p.send(buffer, self.next_stage)
        else:
            raise NotImplementedError('Could not send output of type '
                                      f'{type(outputs)}')

        # Restore the boolean tensor
        if self.has_attention_mask or self.has_bool_tensors:
            gd.debuginfo(prj="ds")
            outputs = list(outputs)
            outputs[-1] = outputs[-1].bool()
            outputs = tuple(outputs)

        if self.wall_clock_breakdown():
            gd.debuginfo(prj="ds")
            self.timers('pipe_send_output').stop()

    def _exec_send_grads(self, buffer_id):
        if self.wall_clock_breakdown():
            gd.debuginfo(prj="ds")
            self.timers('pipe_send_grad').start()

        inputs = self.pipe_buffers['inputs'][buffer_id]
        gd.debuginfo(prj="ds", info=f"self.pipe_buffers['inputs'][{buffer_id}]={infoTensor(inputs)}")

        # Partition the gradient
        if self.is_grad_partitioned:
            if isinstance(inputs, tuple):
                first_input = inputs[0]
                gd.debuginfo(prj="ds", info=f'first_input={infoTensor(first_input)}')

                assert all([torch.is_tensor(elt) for elt in inputs[1:]])
                inputs_grad_tail = [elt.grad for elt in inputs[1:] if elt.grad is not None]
                gd.debuginfo(prj="ds", info=f'len of inputs_grad_tail={len(inputs_grad_tail)}')

            elif torch.is_tensor(inputs):
                first_input = inputs
                inputs_grad_tail = []
                gd.debuginfo(prj="ds", info=f'first_input={infoTensor(first_input)}')
            else:
                raise ValueError("expecting a tensor or a tuple of tensors")

            assert torch.is_tensor(first_input)
            part = PartitionedTensor(tensor=first_input.grad, group=self.grid.get_slice_parallel_group())
            gd.debuginfo(prj="ds", info=f'part={part}')

            inputs = (part.to_meta(), part.data(), *inputs_grad_tail)
            gd.debuginfo(prj="ds", info=f'part.to_meta()={infoTensor(part.to_meta())}')
            gd.debuginfo(prj="ds", info=f'part.data()={infoTensor(part.data())}')
            gd.debuginfo(prj="ds", info=f'*inputs_grad_tail={inputs_grad_tail}')

        # XXX Terrible hack
        # Drop the attention mask from the input buffer here. It does not have
        # a grad that needs to be communicated. We free the buffer immediately
        # after, so no need to restore it. The receiver also has a hack that skips
        # the recv. This is because NCCL does not let us send torch.BoolTensor :-(.
        if self.has_attention_mask or self.has_bool_tensors:
            gd.debuginfo(prj="ds", info=f'inputs={inputs}')
            inputs = list(inputs)
            inputs.pop()
            inputs = tuple(inputs)
            gd.debuginfo(prj="ds", info=f'inputs={inputs}')

        if isinstance(inputs, torch.Tensor):
            gd.debuginfo(prj="ds", info=f'inputs.grad={infoTensor(inputs.grad)}, '
                                        f'self.prev_stage={self.prev_stage}')
            assert inputs.grad is not None
            p2p.send(inputs.grad, self.prev_stage)
        else:

            # XXX terrible hacky branch
            if self.is_grad_partitioned:
                gd.debuginfo(prj="ds", info=f'inputs[0]={infoTensor(inputs[0])}, '
                                            f'inputs[1]={infoTensor(inputs[1])}, '
                                            f'self.prev_stage={self.prev_stage}')
                # First two sends are partitioned gradient
                p2p.send(inputs[0], self.prev_stage)
                p2p.send(inputs[1], self.prev_stage)
            else:
                gd.debuginfo(prj="ds")
                for idx, buffer in enumerate(inputs):
                    gd.debuginfo(prj="ds", info=f"{idx}, "
                                                f"buffer={infoTensor(buffer)}, "
                                                f"self.prev_stage={self.prev_stage}")
                    # Skip tensors that will not produce a grad
                    if not buffer.is_floating_point():
                        assert buffer.grad is None
                        continue
                    assert buffer.grad is not None
                    p2p.send(buffer.grad, self.prev_stage)

        # We can free up the input buffer now
        self.pipe_buffers['inputs'][buffer_id] = None

        if self.wall_clock_breakdown():
            self.timers('pipe_send_grad').stop()

    def _exec_recv_activations(self, buffer_id):
        gd.debuginfo(prj="ds", info=f'buffer_id={buffer_id}')
        if self.wall_clock_breakdown():
            self.timers('pipe_recv_input').start()

        recvd = None

        # Allocate the buffer if necessary
        if self.pipe_recv_buf is None:
            self.pipe_recv_buf = self._recv_tensor_meta(self.prev_stage)
            gd.debuginfo(prj="ds", info=f'self.pipe_recv_buf={self.pipe_recv_buf}')

        if isinstance(self.pipe_recv_buf, torch.Tensor):
            p2p.recv(self.pipe_recv_buf, self.prev_stage)
            recvd = self.pipe_recv_buf.clone().detach()
            recvd.requires_grad = recvd.is_floating_point()
            gd.debuginfo(prj="ds", info=f'recvd={infoTensor(recvd)}')
        else:
            assert isinstance(self.pipe_recv_buf, tuple)
            recvd = [None] * len(self.pipe_recv_buf)
            gd.debuginfo(prj="ds", info=f"self.pipe_recv_buf={self.pipe_recv_buf}")
            for idx, buffer in enumerate(self.pipe_recv_buf):
                gd.debuginfo(prj="ds", info=f"idx={idx}, buffer={infoTensor(buffer)}")
                assert torch.is_tensor(buffer)

                # XXX hardcode meta type
                if self.is_pipe_partitioned and idx == 0 and buffer.dtype != torch.long:
                    if self.meta_buffer is None:
                        self.meta_buffer = torch.zeros(buffer.size(), dtype=torch.long, device=self.device)
                        gd.debuginfo(prj="ds", info=f"self.meta_buffer={infoTensor(self.meta_buffer)}")
                    buffer = self.meta_buffer

                p2p.recv(buffer, self.prev_stage)
                recvd[idx] = buffer.clone().detach()
                gd.debuginfo(prj="ds", info=f"recvd[{idx}]={infoTensor(recvd[idx])}")

            # NCCL does not like to send torch.BoolTensor types, so un-cast the
            # attention mask
            if self.has_attention_mask or self.has_bool_tensors:
                recvd[-1] = recvd[-1].bool()

            recvd = tuple(recvd)

            for index, buffer in enumerate(recvd):
                gd.debuginfo(prj="ds", info=f"recvd[{index}]={infoTensor(buffer)}")
                buffer.requires_grad = buffer.is_floating_point()

        self.pipe_buffers['inputs'][buffer_id] = recvd

        gd.debuginfo(prj="ds", info=f"self.pipe_buffers['inputs'][{buffer_id}]={infoTensor(recvd)}")

        if self.wall_clock_breakdown():
            self.timers('pipe_recv_input').stop()

    def _exec_recv_grads(self, buffer_id):
        gd.debuginfo(prj="ds")
        if self.wall_clock_breakdown():
            self.timers('pipe_recv_grad').start()

        outputs = self.pipe_buffers['outputs'][buffer_id]
        # XXX these shapes are hardcoded for Megatron
        # Restore partitioned output if it was partitioned and we are sending full gradients
        if self.is_pipe_partitioned and not self.is_grad_partitioned:
            part_output = PartitionedTensor.from_meta(meta=outputs[0],
                                                      local_part=outputs[1],
                                                      group=self.grid.get_slice_parallel_group())
            gd.debuginfo(prj="ds", info=f"part_output={infoTensor(part_output)}")

            outputs[0].data = part_output.full()
            outputs = (outputs[0], *outputs[2:])
            # save for backward
            self.pipe_buffers['outputs'][buffer_id] = outputs

            gd.debuginfo(prj="ds", info=f"self.pipe_buffers['outputs'][{buffer_id}]={infoTensor(outputs)}")

        # Allocate gradient if necessary
        if self.grad_layer is None:
            if isinstance(outputs, torch.Tensor):
                s = list(outputs.size())
                self.grad_layer = self._allocate_buffer(s, dtype=outputs.dtype, num_buffers=1)[0]
                gd.debuginfo(prj="ds", info=f'self.grad_layer={infoTensor(self.grad_layer)}')
            else:
                # XXX This is a HACK
                # When we exchange activations/gradients, the two pipe stages
                # need to issue the send/recv with the same buffer sizes or
                # else there is a deadlock. The is_floating_point() filter is
                # used to avoid sending gradients for tensors that do not
                # produce gradients. When TP>1, we partition the first
                # activations/gradients across TP ranks to save communication
                # volume and memory. That partitioned tensor is represented as
                # two tensors: a 1/TPth chunk of the original data and also a
                # small LongTensor storing the metadata used to reconstruct on
                # the other side. When combined, the floating point filter also
                # filtered out the metadata tensor. This quick (hacky) fix just
                # branches on is_grad_partitioned so we don't filter out the
                # metadata tensor.
                if self.is_grad_partitioned:
                    sizes_and_dtypes = [(list(t.size()), t.dtype)
                                        for t in outputs[:2]] + [(list(t.size()), t.dtype)
                                                                 for t in outputs[2:] if t.is_floating_point()]
                    gd.debuginfo(prj="ds", info=f'sizes_and_dtypes={sizes_and_dtypes}')
                else:
                    sizes_and_dtypes = [(list(t.size()), t.dtype) for t in outputs if t.is_floating_point()]
                    gd.debuginfo(prj="ds", info=f'sizes_and_dtypes={sizes_and_dtypes}')

                self.grad_layer = self._allocate_buffers(sizes_and_dtypes, num_buffers=1)[0]
                gd.debuginfo(prj="ds", info=f'self.grad_layer={infoTensor(self.grad_layer)}')

        if isinstance(self.grad_layer, torch.Tensor):
            gd.debuginfo(prj="ds")
            p2p.recv(self.grad_layer, self.next_stage)
        else:
            gd.debuginfo(prj="ds")
            assert isinstance(outputs, tuple)
            for idx, buffer in enumerate(self.grad_layer):
                gd.debuginfo(prj="ds", info=f'idx={idx}, buffer={buffer}')
                # XXX GPT-2 hack
                if self.is_grad_partitioned and idx == 0 and buffer.dtype != torch.long:
                    buffer.data = torch.zeros(buffer.size(), dtype=torch.long, device=self.device)
                    gd.debuginfo(prj="ds", info=f'buffer.data={infoTensor(buffer.data)}')
                p2p.recv(buffer, self.next_stage)

        if self.wall_clock_breakdown():
            self.timers('pipe_recv_grad').stop()

    def _exec_optimizer_step(self, lr_kwargs=None):
        gd.debuginfo(prj="ds")
        if self.wall_clock_breakdown():
            self.timers('step_microstep').start()
            self.timers('step').start()
        self.mem_status('BEFORE STEP', reset_max=True)

        self._force_grad_boundary = True
        self._take_model_step(lr_kwargs)
        self._force_grad_boundary = False

        self.mem_status('AFTER STEP')

        if self.global_rank == 0 and self.monitor.enabled:
            self.summary_events = [(f'Train/Samples/lr', self.get_lr()[0], self.global_samples)]
            if self.fp16_enabled() and hasattr(self.optimizer, 'cur_scale'):
                self.summary_events.append(
                    (f'Train/Samples/loss_scale', self.optimizer.cur_scale, self.global_samples))
            self.monitor.write_events(self.summary_events)

        if self.wall_clock_breakdown():
            self.timers('step_microstep').stop()
            self.timers('step').stop()
            if self.global_steps % self.steps_per_print() == 0:
                self.timers.log([
                    'batch_input', 'forward_microstep', 'backward_microstep', 'backward_inner_microstep',
                    'backward_allreduce_microstep', 'backward_tied_allreduce_microstep', 'step_microstep'
                ])
            if self.global_steps % self.steps_per_print() == 0:
                self.timers.log(['forward', 'backward', 'backward_inner', 'backward_allreduce', 'step'])

    def _zero_grads(self, inputs):
        if isinstance(inputs, torch.Tensor):
            gd.debuginfo(prj="ds")
            if inputs.grad is not None:
                gd.debuginfo(prj="ds")
                inputs.grad.data.zero_()
        else:
            gd.debuginfo(prj="ds")
            for t in inputs:
                gd.debuginfo(prj="ds")
                if t.grad is not None:
                    gd.debuginfo(prj="ds")
                    t.grad.data.zero_()

    def _allocate_zeros(self, shape, **kwargs):
        """ Allocate a tensor of zeros on the engine's device.

        Arguments:
            shape: the shape of the tensor to allocate
            kwargs: passed to torch.zeros()

        Returns:
            A tensor from torch.zeros() allocated on self.device.
        """
        gd.debuginfo(prj="ds")
        if "dtype" not in kwargs:
            if self.fp16_enabled():
                kwargs["dtype"] = torch.half
            if self.bfloat16_enabled():
                kwargs["dtype"] = torch.bfloat16

        return torch.zeros(shape, device=self.device, **kwargs)

    def _allocate_buffer(self, shape, num_buffers=-1, **kwargs):
        gd.debuginfo(prj="ds", info=f'shape={shape}, kwargs={kwargs}')
        buffers = []
        if num_buffers == -1:
            gd.debuginfo(prj="ds")
            num_buffers = self.num_pipe_buffers

        for i, count in enumerate(range(num_buffers)):
            buffers.append(self._allocate_zeros(shape, **kwargs))
            gd.debuginfo(prj="ds", info=f'{i}, count={count}')

        gd.debuginfo(prj="ds", info=f'len(buffers)={len(buffers)}')
        return buffers

    def _allocate_buffers(self, shapes_and_dtypes, requires_grad=False, num_buffers=-1):
        buffers = []
        if num_buffers == -1:
            gd.debuginfo(prj="ds")
            num_buffers = self.num_pipe_buffers

        for i, count in enumerate(range(num_buffers)):
            buffer = []
            for shape, dtype in shapes_and_dtypes:
                gd.debuginfo(prj="ds", info=f'count={count}, dtype={dtype}')
                buffer.append(self._allocate_zeros(shape, dtype=dtype, requires_grad=requires_grad))

            gd.debuginfo(prj="ds", info=f'{i}, count={count}, len(buffer)={len(buffer)}')
            buffers.append(buffer)

        gd.debuginfo(prj="ds", info=f'len(buffers)={len(buffers)}')
        return buffers

    def forward(self, *args, **kwargs):
        """Disabled for pipeline parallel training. See ``train_batch()``. """
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    def backward(self, *args, **kwargs):
        """Disabled for pipeline parallel training. See ``train_batch()``. """
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    def step(self, *args, **kwargs):
        """Disabled for pipeline parallel training. See ``train_batch()``. """
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    def mem_status(self, msg, print_rank=-1, reset_max=False):
        return
        global mem_alloced, mem_cached
        if not self.global_steps == 0 or not self.global_steps == 9:
            #return
            pass
        if self.mpu.get_data_parallel_rank() != 0:
            return

        if self.global_rank != 0:
            return

        rank = self.global_rank
        if print_rank != -1 and rank != print_rank:
            return

        get_accelerator().synchronize()

        if reset_max:
            get_accelerator().reset_max_memory_cached()
            get_accelerator().reset_max_memory_allocated()

        new_alloced = get_accelerator().memory_allocated()
        new_cached = get_accelerator().memory_cached()

        delta_alloced = new_alloced - mem_alloced
        delta_cached = new_cached - mem_cached

        mem_cached = new_cached
        mem_alloced = new_alloced

        max_alloced = get_accelerator().max_memory_allocated()
        max_cached = get_accelerator().max_memory_cached()

        # convert to GB for printing
        new_alloced /= 1024**3
        new_cached /= 1024**3
        delta_alloced /= 1024**3
        delta_cached /= 1024**3
        max_alloced /= 1024**3
        max_cached /= 1024**3

        gd.debuginfo(prj='ds', info=f'RANK={rank} STAGE={self.stage_id} STEP={self.global_steps} MEMSTATS {msg} '
                                    f'current alloc={new_alloced:0.4f}GB (delta={delta_alloced:0.4f}GB max={max_alloced:0.4f}GB) '
                                    f'current cache={new_cached:0.4f}GB (delta={delta_cached:0.4f}GB max={max_cached:0.4f}GB)')

    def module_state_dict(self):
        gd.debuginfo(prj='ds', info='save the dict!')
        """Override hack to save a pipe model and return the directory path of the save.

        This method should only be called by DeepSpeed's ``save_checkpoint()``. The
        recommended way of saving a ``PipelineModule`` outside of ``save_checkpoint()``
        is ``save_state_dict()``.

        Returns:
            None
        """
        assert isinstance(self.module, PipelineModule)
        assert self._curr_ckpt_path is not None, \
            "PipelineEngine expects module_state_dict() to be called from save_checkpoint()"

        self.module.save_state_dict(self._curr_ckpt_path, checkpoint_engine=self.checkpoint_engine)
        return None

    def load_module_state_dict(self, checkpoint, strict=True, custom_load_fn=None):
        """Override hack to instead use a directory path.

        This is important because pipeline models checkpoint by layer instead of rank.

        If ``state_dict`` is not ``None`` or a ``str``, we revert to ``super()`` expecting a ``dict``.

        Args:
            state_dict (str, None): unused
            strict (bool, optional): Strict state loading. Defaults to True.
        """
        gd.debuginfo(prj="ds")
        assert custom_load_fn is None, "custom_load_fn not supported w. pipeline parallelism"
        state_dict = checkpoint['module']
        if (state_dict is not None) and (not isinstance(state_dict, str)):
            super().load_module_state_dict(state_dict, strict)
            return

        self.module.load_state_dir(load_dir=self._curr_ckpt_path,
                                   strict=strict,
                                   checkpoint_engine=self.checkpoint_engine)

    # A map of PipeInstruction types to methods. Each method will be executed with the
    # kwargs provided to the PipeInstruction from the scheduler.
    _INSTRUCTION_MAP = {
        schedule.OptimizerStep: _exec_optimizer_step,
        schedule.ReduceGrads: _exec_reduce_grads,
        schedule.ReduceTiedGrads: _exec_reduce_tied_grads,
        schedule.LoadMicroBatch: _exec_load_micro_batch,
        schedule.ForwardPass: _exec_forward_pass,
        schedule.BackwardPass: _exec_backward_pass,
        schedule.SendActivation: _exec_send_activations,
        schedule.RecvActivation: _exec_recv_activations,
        schedule.SendGrad: _exec_send_grads,
        schedule.RecvGrad: _exec_recv_grads,
    }

    def _exec_schedule(self, pipe_schedule):
        gd.debuginfo(prj="ds", info=f'pipe_schedule={pipe_schedule}')
        # Reserve and reset buffers.
        self._reserve_pipe_buffers(pipe_schedule.num_pipe_buffers())
        self.fwd_outputs = []

        # For each step in the schedule
        for step_cmds in pipe_schedule:
            gd.debuginfo(prj="ds", info=f'step_cmds={step_cmds}')
            # For each instruction in the step
            for cmd in step_cmds:
                gd.debuginfo(prj="ds", info=f'cmd={cmd}')
                if type(cmd) not in self._INSTRUCTION_MAP:
                    raise RuntimeError(f'{self.__class__.__name__} does not understand instruction {repr(cmd)}')

                # Equivalent to: self._exec_forward_pass(buffer_id=0)
                # https://blog.csdn.net/shuibuzhaodeshiren/article/details/87115118
                # 2.1 MethodType 把方法绑定到类的实例中,此时方法在类实例间不共享
                self._exec_instr = MethodType(self._INSTRUCTION_MAP[type(cmd)], self)

                gd.debuginfo(prj="ds", info=f'=====## {self._exec_instr}/{self._INSTRUCTION_MAP[type(cmd)]} START======')
                self._exec_instr(**cmd.kwargs)
                gd.debuginfo(prj="ds", info=f'=====## {self._exec_instr}/{self._INSTRUCTION_MAP[type(cmd)]} END======')
