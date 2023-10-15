# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import sys
import os
from pydebug import debuginfo, infoTensor

debuginfo(prj='ds', info='trsrc __init__')

def _build_file_index(directory, suffix='.tr'):
    """Build an index of source files and their basenames in a given directory.

    Args:
        directory (string): the directory to index
        suffix (string): index files with this suffix

    Returns:
        list: A list of tuples of the form [(basename, absolute path), ...]
    """

    index = []

    for fname in os.listdir(directory):
        debuginfo(prj='ds', info='1-fname is:', fname)
        if fname.endswith(suffix):
            basename = fname[:fname.rfind(suffix)]  # strip the suffix
            path = os.path.join(directory, fname)
            index.append((basename, path))
    
    debuginfo(prj='ds', info='index:', index)

    return index


# Go over all local source files and parse them as strings
_module = sys.modules[_build_file_index.__module__]
_directory = os.path.dirname(os.path.realpath(__file__))
for name, fname in _build_file_index(_directory):
    debuginfo(prj='ds', info='2-fname is:', fname)
    with open(fname, 'r') as fin:
        setattr(_module, name, fin.read())
