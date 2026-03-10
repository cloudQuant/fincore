"""Proxy classes for type checking in tests.

These proxy classes wrap empyrical functions to verify:
1. Return types match expected types
2. Inputs are not mutated by function calls

Split from test_2d_stats.py for maintainability.
"""

from __future__ import annotations

from copy import copy
from functools import wraps
from operator import attrgetter
from unittest import SkipTest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from pandas.core.generic import NDFrame

from fincore.empyrical import Empyrical

DECIMAL_PLACES = 8


class ReturnTypeEmpyricalProxy:
    """
    A wrapper around the empyrical module which, on each function call, asserts
    that the type of the return value is in a given set.

    Also asserts that inputs were not modified by the empyrical function call.

    Calling an instance with kwargs will return a new copy with those
    attributes overridden.
    """

    def __init__(self, test_case, return_types):
        self._test_case = test_case
        self._return_types = return_types

    def __call__(self, **kwargs):
        dupe = copy(self)

        for k, v in kwargs.items():
            attr = "_" + k
            if hasattr(dupe, attr):
                setattr(dupe, attr, v)

        return dupe

    def __copy__(self):
        newone = type(self).__new__(type(self))
        newone.__dict__.update(self.__dict__)
        return newone

    def __getattr__(self, item):
        emp = Empyrical()
        func = getattr(emp, item)
        return self._check_input_not_mutated(self._check_return_type(func))

    def _check_return_type(self, func):
        @wraps(func)
        def check_return_type(*args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(result, tuple):
                tuple_result = result
            else:
                tuple_result = (result,)

            for r in tuple_result:
                self._test_case.assertIsInstance(r, self._return_types)
            return result

        return check_return_type

    def _check_input_not_mutated(self, func):
        @wraps(func)
        def check_not_mutated(*args, **kwargs):
            # Copy inputs to compare them to originals later.
            arg_copies = [(i, arg.copy()) for i, arg in enumerate(args) if isinstance(arg, (NDFrame, np.ndarray))]
            kwarg_copies = {k: v.copy() for k, v in kwargs.items() if isinstance(v, (NDFrame, np.ndarray))}

            result = func(*args, **kwargs)

            # Check that inputs weren't mutated by func.
            for i, arg_copy in arg_copies:
                assert_allclose(
                    args[i],
                    arg_copy,
                    atol=0.5 * 10 ** (-DECIMAL_PLACES),
                    err_msg=f"Input 'arg {i}' mutated by {func.__name__}",
                )
            for kwarg_name, kwarg_copy in kwarg_copies.items():
                assert_allclose(
                    kwargs[kwarg_name],
                    kwarg_copy,
                    atol=0.5 * 10 ** (-DECIMAL_PLACES),
                    err_msg=f"Input '{kwarg_name}' mutated by {func.__name__}",
                )

            return result

        return check_not_mutated


class ConvertPandasEmpyricalProxy(ReturnTypeEmpyricalProxy):
    """
    A ReturnTypeEmpyricalProxy which also converts pandas NDFrame inputs to
    empyrical functions according to the given conversion method.

    Calling an instance with a truthy pandas_only will return a new instance
    which will skip the test when an empyrical function is called.
    """

    def __init__(self, test_case, return_types, convert, pandas_only=False):
        super().__init__(test_case, return_types)
        self._convert = convert
        self._pandas_only = pandas_only

    def __getattr__(self, item):
        if self._pandas_only:
            raise SkipTest("empyrical.%s expects pandas-only inputs that have dt indices/labels" % item)

        func = super().__getattr__(item)

        @wraps(func)
        def convert_args(*args, **kwargs):
            args = [self._convert(arg) if isinstance(arg, NDFrame) else arg for arg in args]
            kwargs = {k: self._convert(v) if isinstance(v, NDFrame) else v for k, v in kwargs.items()}
            return func(*args, **kwargs)

        return convert_args


class PassArraysEmpyricalProxy(ConvertPandasEmpyricalProxy):
    """
    A ConvertPandasEmpyricalProxy which converts NDFrame inputs to empyrical
    functions to numpy arrays.

    Calls the underlying
    empyrical.[alpha|beta|alpha_beta]_aligned functions directly, instead of
    the wrappers which align Series first.
    """

    def __init__(self, test_case, return_types):
        super().__init__(
            test_case,
            return_types,
            attrgetter("values"),
        )

    def __getattr__(self, item):
        if item in (
            "alpha",
            "beta",
            "alpha_beta",
            "beta_fragility_heuristic",
            "gpd_risk_estimates",
        ):
            item += "_aligned"

        return super().__getattr__(item)
