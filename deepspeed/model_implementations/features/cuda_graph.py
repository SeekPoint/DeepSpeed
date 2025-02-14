# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from abc import ABC, abstractmethod
from pydebug import debuginfo

class CUDAGraph(ABC):

    def __init__(self, enable_cuda_graph=False):
        debuginfo(prj='ds', info='CUDAGraph init')
        super().__init__()
        self.enable_cuda_graph = enable_cuda_graph

    @abstractmethod
    def _create_cuda_graph(self):
        """
        Create CUDA graph(s)
        """
        raise NotImplementedError

    @abstractmethod
    def _graph_replay(self):
        """
        Replay CUDA graph(s)
        """
        raise NotImplementedError
