# import warnings

from . import utils
from . import timeseries
from . import pos
from . import txn
from . import interesting_periods
from . import capacity
from . import round_trips
from . import risk
from . import perf_attrib

from .tears import *  # noqa
from .plotting import *  # noqa


try:
    from . import bayesian
except ImportError:
    warnings.warn(
        "Could not import bayesian submodule due to missing pymc3 dependency.",
        ImportWarning)


__version__ = "0.9.6"

__all__ = ['utils', 'timeseries', 'pos', 'txn', 'bayesian',
           'interesting_periods', 'capacity', 'round_trips',
           'risk', 'perf_attrib']
