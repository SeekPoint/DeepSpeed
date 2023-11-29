# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed.utils import logger, log_dist
from deepspeed.runtime.checkpoint_engine.checkpoint_engine import \
    CheckpointEngine
from pydebug import gd, infoTensor
gd.debuginfo(prj="ds")

class TorchCheckpointEngine(CheckpointEngine):

    def __init__(self, config_params=None):
        gd.debuginfo(prj="ds")
        super().__init__(config_params)

    def create(self, tag):
        gd.debuginfo(prj="ds", info=f"[Torch] Checkpoint {tag} is about to be saved!")

    def save(self, state_dict, path: str):
        gd.debuginfo(prj="ds", info=f"[Torch] Saving {path}...")
        torch.save(state_dict, path)
        gd.debuginfo(prj="ds", info=f"[Torch] Saved {path}.")
        return None

    def load(self, path: str, map_location=None):
        gd.debuginfo(prj="ds", info=f"[Torch] Loading checkpoint from {path}...")
        partition = torch.load(path, map_location=map_location)
        gd.debuginfo(prj="ds", info=f"[Torch] Loaded checkpoint from {path}.")
        return partition

    def commit(self, tag):
        gd.debuginfo(prj="ds", info=f"[Torch] Checkpoint {tag} is ready now!")
        return True
