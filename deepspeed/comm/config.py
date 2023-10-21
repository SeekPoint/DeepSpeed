# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from pydantic import BaseModel
from .constants import *
from pydebug import debuginfo, infoTensor

class CommsConfig(BaseModel):

    class Config:
        gd.debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
        validate_all = True
        validate_assignment = True
        use_enum_values = True
        extra = 'forbid'


class CommsLoggerConfig(CommsConfig):
    gd.debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
    enabled: bool = COMMS_LOGGER_ENABLED_DEFAULT
    prof_all: bool = COMMS_LOGGER_PROF_ALL_DEFAULT
    prof_ops: list = COMMS_LOGGER_PROF_OPS_DEFAULT
    verbose: bool = COMMS_LOGGER_VERBOSE_DEFAULT
    debug: bool = COMMS_LOGGER_DEBUG_DEFAULT


class DeepSpeedCommsConfig:

    def __init__(self, ds_config):
        gd.debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
        self.comms_logger_enabled = 'comms_logger' in ds_config

        if self.comms_logger_enabled:
            self.comms_logger = CommsLoggerConfig(**ds_config['comms_logger'])
