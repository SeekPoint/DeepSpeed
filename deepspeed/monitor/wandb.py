# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .utils import check_wandb_availability
from .monitor import Monitor

import deepspeed.comm as dist
from pydebug import debuginfo

class WandbMonitor(Monitor):

    def __init__(self, wandb_config):
        debuginfo(prj='ds', info='WandbMonitor init')
        super().__init__(wandb_config)
        check_wandb_availability()
        import wandb

        self.enabled = wandb_config.enabled
        self.group = wandb_config.group
        self.team = wandb_config.team
        self.project = wandb_config.project

        if self.enabled and dist.get_rank() == 0:
            debuginfo(prj='ds')
            wandb.init(project=self.project, group=self.group, entity=self.team)

    def log(self, data, step=None, commit=None, sync=None):
        if self.enabled and dist.get_rank() == 0:
            import wandb
            return wandb.log(data, step=step, commit=commit, sync=sync)

    def write_events(self, event_list):
        debuginfo(prj='ds')
        if self.enabled and dist.get_rank() == 0:
            debuginfo(prj='ds')
            for event in event_list:
                label = event[0]
                value = event[1]
                step = event[2]
                self.log({label: value}, step=step)
