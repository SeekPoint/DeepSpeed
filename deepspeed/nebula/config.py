# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.runtime.config_utils import get_scalar_param, DeepSpeedConfigObject
from deepspeed.nebula.constants import *
from pydebug import gd, infoTensor

class DeepSpeedNebulaConfig(DeepSpeedConfigObject):

    def __init__(self, param_dict):
        
        super(DeepSpeedNebulaConfig, self).__init__()

        self.enabled = None
        self.persistent_storage_path = None
        self.persistent_time_interval = None
        self.num_of_version_in_retention = None
        self.enable_nebula_load = None

        if NEBULA in param_dict.keys():
            gd.debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
            nebula_dict = param_dict[NEBULA]
        else:
            gd.debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
            nebula_dict = {}

        self._initialize(nebula_dict)

    def _initialize(self, nebula_dict):
        gd.debuginfo(prj='ds-chat', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
        self.enabled = get_scalar_param(nebula_dict, NEBULA_ENABLED, NEBULA_ENABLED_DEFAULT)

        self.load_path = get_scalar_param(nebula_dict, NEBULA_LOAD_PATH, NEBULA_LOAD_PATH_DEFAULT)

        self.enable_nebula_load = get_scalar_param(nebula_dict, NEBULA_ENABLE_NEBULA_LOAD,
                                                   NEBULA_ENABLE_NEBULA_LOAD_DEFAULT)

        self.persistent_storage_path = get_scalar_param(nebula_dict, NEBULA_PERSISTENT_STORAGE_PATH,
                                                        NEBULA_PERSISTENT_STORAGE_PATH_DEFAULT)

        self.persistent_time_interval = get_scalar_param(nebula_dict, NEBULA_PERSISTENT_TIME_INTERVAL,
                                                         NEBULA_PERSISTENT_TIME_INTERVAL_DEFAULT)

        self.num_of_version_in_retention = get_scalar_param(nebula_dict, NEBULA_NUM_OF_VERSION_IN_RETENTION,
                                                            NEBULA_NUM_OF_VERSION_IN_RETENTION_DEFAULT)
