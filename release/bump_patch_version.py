# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from packaging import version as pkg_version

from pydebug import gd, infoTensor

with open('../version.txt') as fd:
    version = pkg_version.parse(fd.read())
    gd.debuginfo(prj='ds')

with open('../version.txt', 'w') as fd:
    fd.write(f'{version.major}.{version.minor}.{version.micro + 1}\n')
    gd.debuginfo(prj='ds')

print(f'{version} -> {version.major}.{version.minor}.{version.micro + 1}')
