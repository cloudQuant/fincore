#!/usr/bin/env python
#
# Copyright 2016 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from setuptools import setup, find_packages

DISTNAME = "fincore"
VERSION = "0.1.0"
DESCRIPTION = """fincore is a Python library for quantitative performance and risk analytics"""
LONG_DESCRIPTION = """fincore is a continuation of the empyrical analytics stack maintained by `cloudQuant`_.
It delivers more than 50 return, risk, attribution, timing, and streak statistics with
aligned rolling utilities, structured factor attribution helpers, and pandas/numpy friendly APIs.

For documentation, examples, and contribution guidelines see the project homepage:
`cloudQuant/fincore <https://github.com/cloudQuant/fincore>`_.

.. _cloudQuant: https://www.cloudquant.ai
"""
MAINTAINER = "cloudQuant"
MAINTAINER_EMAIL = "yunjinqi@gmail.com"
AUTHOR = "cloudQuant"
AUTHOR_EMAIL = "yunjinqi@gmail.com"
URL = "https://github.com/cloudQuant/fincore"
LICENSE = "Apache-2.0"

classifiers = [
    "Development Status :: 4 - Beta",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "License :: OSI Approved :: Apache Software License",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Operating System :: OS Independent"
]

install_requires = [
    "numpy>=1.17.0",
    "pandas>=0.25.0",
    "scipy>=1.3.0",
]

extras_require = {
    "dev": [
        "pytest>=6.0",
        "pytest-xdist>=2.0",
        "pytest-cov>=2.10",
        "pytest-benchmark>=3.2",
        "parameterized",
        "ruff>=0.4.0",
        "mypy>=1.5",
    ],
    "viz": [
        "matplotlib>=3.3",
        "seaborn>=0.11",
    ],
    "bayesian": [
        "pymc>=5.0",
    ],
    "datareader": [
        "pandas-datareader>=0.8.0",
    ],
    "all": [
        "fincore[viz,bayesian,datareader]",
    ],
}

if __name__ == "__main__":
    setup(
        name=DISTNAME,
        version=VERSION,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        classifiers=classifiers,
        install_requires=install_requires,
        extras_require=extras_require,
        python_requires=">=3.11",
    )
