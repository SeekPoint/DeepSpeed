# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .utils import check_tb_availability
from .monitor import Monitor
import os

import deepspeed.comm as dist
from pydebug import debuginfo, infoTensor

class TensorBoardMonitor(Monitor):

    def __init__(self, tensorboard_config):
        gd.debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
        super().__init__(tensorboard_config)
        check_tb_availability()

        self.summary_writer = None
        self.enabled = tensorboard_config.enabled
        self.output_path = tensorboard_config.output_path
        self.job_name = tensorboard_config.job_name

        if self.enabled and dist.get_rank() == 0:
            gd.debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
            self.get_summary_writer()

    def get_summary_writer(self, base=os.path.join(os.path.expanduser("~"), "tensorboard")):
        if self.enabled and dist.get_rank() == 0:
            gd.debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
            from torch.utils.tensorboard import SummaryWriter
            if self.output_path is not None:
                gd.debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
                log_dir = os.path.join(self.output_path, self.job_name)
            # NOTE: This code path currently is never used since the default output_path is an empty string and not None. Saving it in case we want this functionality in the future.
            else:
                if "DLWS_JOB_ID" in os.environ:
                    gd.debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
                    infra_job_id = os.environ["DLWS_JOB_ID"]
                elif "DLTS_JOB_ID" in os.environ:
                    gd.debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
                    infra_job_id = os.environ["DLTS_JOB_ID"]
                else:
                    gd.debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
                    infra_job_id = "unknown-job-id"

                summary_writer_dir_name = os.path.join(infra_job_id, "logs")
                log_dir = os.path.join(base, summary_writer_dir_name, self.output_path)
            os.makedirs(log_dir, exist_ok=True)
            self.summary_writer = SummaryWriter(log_dir=log_dir)
        return self.summary_writer

    def write_events(self, event_list, flush=True):
        gd.debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
        if self.enabled and self.summary_writer is not None and dist.get_rank() == 0:
            gd.debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
            for event in event_list:
                gd.debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
                self.summary_writer.add_scalar(*event)
            if flush:
                gd.debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
                self.summary_writer.flush()

    def flush(self):
        gd.debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
        if self.enabled and self.summary_writer is not None and dist.get_rank() == 0:
            gd.debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
            self.summary_writer.flush()
