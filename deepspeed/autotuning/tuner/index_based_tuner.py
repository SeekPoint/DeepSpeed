# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import random

from .base_tuner import BaseTuner

from pydebug import gd, infoTensor
class RandomTuner(BaseTuner):
    """Explore the search space in random order"""

    def __init__(self, exps: list, resource_manager, metric):
        gd.debuginfo(prj='ds', info=self.__class__.__name__)
        super().__init__(exps, resource_manager, metric)

    def next_batch(self, sample_size=1):
        gd.debuginfo(prj='ds', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
        if sample_size > len(self.all_exps):
            sample_size = len(self.all_exps)

        sampled_batch = random.sample(self.all_exps, sample_size)
        self.all_exps = [x for x in self.all_exps if x not in sampled_batch]

        return sampled_batch


class GridSearchTuner(BaseTuner):
    """Explore the search space in sequential order"""

    def __init__(self, exps: list, resource_manager, metric):
        gd.debuginfo(prj='ds', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
        super().__init__(exps, resource_manager, metric)

    def next_batch(self, sample_size=1):
        gd.debuginfo(prj='ds', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
        if sample_size > len(self.all_exps):
            sample_size = len(self.all_exps)

        sampled_batch = self.all_exps[0:sample_size]
        self.all_exps = [x for x in self.all_exps if x not in sampled_batch]

        return sampled_batch
