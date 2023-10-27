# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
from pydebug import gd, infoTensor

#cp基类引擎
class CheckpointEngine(object):

    # init checkpoint engine for save/load
    def __init__(self, config_params=None):
        gd.debuginfo(prj='ds', info=f"c:{self.__class__.__name__}")
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
