# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import math
import numpy as np

from deepspeed.utils import logger
from .indexed_dataset import MMapIndexedDatasetBuilder
from pydebug import debuginfo, infoTensor

def find_fit_int_dtype(min_value, max_value):
    if min_value >= 0:
        if max_value <= 255:
            gd.debuginfo(prj='ds', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
            return np.uint8
        elif max_value <= 65535:
            gd.debuginfo(prj='ds', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
            return np.uint16
        elif max_value <= 4294967295:
            gd.debuginfo(prj='ds', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
            return np.uint32
        else:
            gd.debuginfo(prj='ds', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
            return np.uint64
    else:
        if max_value <= 127 and min_value >= -128:
            gd.debuginfo(prj='ds', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
            return np.int8
        elif max_value <= 32767 and min_value >= -32768:
            gd.debuginfo(prj='ds', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
            return np.int16
        elif max_value <= 2147483647 and min_value >= -2147483648:
            gd.debuginfo(prj='ds', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
            return np.int32
        else:
            gd.debuginfo(prj='ds', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
            return np.int64


def split_index(start_idx, end_idx, num_partitions):
    gd.debuginfo(prj='ds', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
    partition_size = math.ceil((end_idx - start_idx) / num_partitions)
    partitions = [[start_idx + x * partition_size,
                   min(end_idx, start_idx + (x + 1) * partition_size)] for x in range(num_partitions)]
    return partitions


def split_dataset(dataset, num_workers, worker_id, num_threads):
    gd.debuginfo(prj='ds', info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
    worker_splits = split_index(0, len(dataset), num_workers)
    thread_splits = split_index(worker_splits[worker_id][0], worker_splits[worker_id][1], num_threads)
    return worker_splits, thread_splits


def create_mmap_dataset_builder(fname, dtype):
    logger.info(f"Creating mmap dataset builder at {fname}.")
    return MMapIndexedDatasetBuilder(f"{fname}.bin", dtype=dtype)


def close_mmap_dataset_builder(builder, fname):
    builder.end_document()
    builder.finalize(f"{fname}.idx")
    logger.info(f"Finalized mmap dataset builder at {fname}.")
