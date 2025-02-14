# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
from pydebug import debuginfo

class CheckpointEngine(object):

    # init checkpoint engine for save/load
    def __init__(self, config_params=None):
        debuginfo(prj='ds', info='CheckpointEngine init')
        pass

    def create(self, tag):
        # create checkpoint on give tag for save/load.
        pass

    def makedirs(self, path, exist_ok=False):
        os.makedirs(path, exist_ok=exist_ok)

    def save(self, state_dict, path: str):
        pass

    def load(self, path: str, map_location=None):
        pass

    def commit(self, tag):
        # to tell checkpoint services if all files are ready.
        pass
