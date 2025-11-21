#
# Copyright 2018 Quantopian, Inc.
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
from __future__ import division

import warnings
from time import time
import os
import shutil
# import pyfolio as pf  # Remove circular import
from fincore import empyrical as ep
try:
    from IPython.display import display, Markdown
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False
    # Define dummy functions for non-IPython environments
    def display(obj):
        print(obj)
    def Markdown(string):
        return string
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from . import _seaborn as sns
from . import capacity
from . import perf_attrib
from . import plotting
from . import pos
from . import risk
from . import round_trips
from . import timeseries
from . import txn
from . import utils
from .utils import make_timezone_aware

try:
    from . import bayesian

    have_bayesian = True
except ImportError:
    warnings.warn(
        "Could not import bayesian submodule due to missing pymc3 dependency.",
        ImportWarning)
    have_bayesian = False

import matplotlib
matplotlib.use('Agg')








