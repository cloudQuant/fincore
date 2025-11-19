import os
import numpy as np
import pandas as pd
import unittest
from fincore.empyrical import Empyrical


class PerfAttribTestCase(unittest.TestCase):

    def test_perf_attrib_simple(self):

        start_date = '2017-01-01'
        periods = 2
        dts = pd.date_range(start_date, periods=periods)
        dts.name = 'dt'

        tickers = ['stock1', 'stock2']
        styles = ['risk_factor1', 'risk_factor2']

        returns = pd.Series(data=[0.1, 0.1], index=dts)

        factor_returns = pd.DataFrame(
            columns=styles,
            index=dts,
            data={'risk_factor1': [.1, .1],
                  'risk_factor2': [.1, .1]}
        )

        index = pd.MultiIndex.from_product(
            [dts, tickers], names=['dt', 'ticker'])

        positions = pd.Series([0.2857142857142857, 0.7142857142857143,
                               0.2857142857142857, 0.7142857142857143],
                              index=index)

        factor_loadings = pd.DataFrame(
            columns=styles,
            index=index,
            data={'risk_factor1': [0.25, 0.25, 0.25, 0.25],
                  'risk_factor2': [0.25, 0.25, 0.25, 0.25]}
        )

        expected_perf_attrib_output = pd.DataFrame(
            index=dts,
            columns=['risk_factor1', 'risk_factor2', 'total_returns',
                     'common_returns', 'specific_returns',
                     'tilt_returns', 'timing_returns'],
            data={'risk_factor1': [0.025, 0.025],
                  'risk_factor2': [0.025, 0.025],
                  'common_returns': [0.05, 0.05],
                  'specific_returns': [0.05, 0.05],
                  'tilt_returns': [0.05, 0.05],
                  'timing_returns': [0.0, 0.0],
                  'total_returns': returns}
        )

        expected_exposures_portfolio = pd.DataFrame(
            index=dts,
            columns=['risk_factor1', 'risk_factor2'],
            data={'risk_factor1': [0.25, 0.25],
                  'risk_factor2': [0.25, 0.25]}
        )

        emp = Empyrical(
            returns=returns,
            positions=positions,
            factor_returns=factor_returns,
            factor_loadings=factor_loadings,
        )

        exposures_portfolio, perf_attrib_output = emp.perf_attrib()

        self.assertTrue(
            expected_perf_attrib_output.round(10).equals(
                perf_attrib_output.round(10)
            )
        )
        self.assertTrue(expected_exposures_portfolio.equals(exposures_portfolio))
        # test long and short positions
        positions = pd.Series([0.5, -0.5, 0.5, -0.5], index=index)

        emp = Empyrical(
            returns=returns,
            positions=positions,
            factor_returns=factor_returns,
            factor_loadings=factor_loadings,
        )

        exposures_portfolio, perf_attrib_output = emp.perf_attrib()

        expected_perf_attrib_output = pd.DataFrame(
            index=dts,
            columns=['risk_factor1', 'risk_factor2', 'total_returns',
                     'common_returns', 'specific_returns',
                     'tilt_returns', 'timing_returns'],
            data={'risk_factor1': [0.0, 0.0],
                  'risk_factor2': [0.0, 0.0],
                  'common_returns': [0.0, 0.0],
                  'specific_returns': [0.1, 0.1],
                  'tilt_returns': [0.0, 0.0],
                  'timing_returns': [0.0, 0.0],
                  'total_returns': returns}
        )

        expected_exposures_portfolio = pd.DataFrame(
            index=dts,
            columns=['risk_factor1', 'risk_factor2'],
            data={'risk_factor1': [0.0, 0.0],
                  'risk_factor2': [0.0, 0.0]}
        )

        self.assertTrue(
            expected_perf_attrib_output.round(10).equals(
                perf_attrib_output.round(10)
            )
        )
        self.assertTrue(expected_exposures_portfolio.equals(exposures_portfolio))
        # test long and short positions with tilt exposure
        positions = pd.Series([1.0, -0.5, 1.0, -0.5], index=index)

        emp = Empyrical(
            returns=returns,
            positions=positions,
            factor_returns=factor_returns,
            factor_loadings=factor_loadings,
        )

        exposures_portfolio, perf_attrib_output = emp.perf_attrib()

        expected_perf_attrib_output = pd.DataFrame(
            index=dts,
            columns=['risk_factor1', 'risk_factor2', 'total_returns',
                     'common_returns', 'specific_returns',
                     'tilt_returns', 'timing_returns'],
            data={'risk_factor1': [0.0125, 0.0125],
                  'risk_factor2': [0.0125, 0.0125],
                  'common_returns': [0.025, 0.025],
                  'specific_returns': [0.075, 0.075],
                  'tilt_returns': [0.025, 0.025],
                  'timing_returns': [0.0, 0.0],
                  'total_returns': returns}
        )

        expected_exposures_portfolio = pd.DataFrame(
            index=dts,
            columns=['risk_factor1', 'risk_factor2'],
            data={'risk_factor1': [0.125, 0.125],
                  'risk_factor2': [0.125, 0.125]}
        )

        self.assertTrue(
            expected_perf_attrib_output.round(10).equals(
                perf_attrib_output.round(10)
            )
        )
        self.assertTrue(expected_exposures_portfolio.equals(exposures_portfolio))

    def test_perf_attrib_regression(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        positions = pd.read_csv(os.path.join(current_dir, 'test_data', 'positions.csv'),
                                index_col=0, parse_dates=True)

        positions.columns = [int(col) if col != 'cash' else col
                             for col in positions.columns]

        positions = positions.divide(positions.sum(axis='columns'),
                                     axis='rows')
        positions = positions.drop('cash', axis='columns').stack()

        returns = pd.read_csv(os.path.join(current_dir, 'test_data', 'returns.csv'),
                              index_col=0, parse_dates=True,
                              header=None)
        if returns.shape[1] == 1:
            returns = returns.iloc[:, 0]

        # date_format parameter was added in pandas 2.0
        read_csv_kwargs = {'index_col': [0, 1], 'parse_dates': True}
        try:
            # Try with date_format for newer pandas
            factor_loadings = pd.read_csv(
                os.path.join(current_dir, 'test_data', 'factor_loadings.csv'),
                date_format='ISO8601', **read_csv_kwargs
            )
        except TypeError:
            # Fall back without date_format for older pandas
            factor_loadings = pd.read_csv(
                os.path.join(current_dir, 'test_data', 'factor_loadings.csv'),
                **read_csv_kwargs
            )

        factor_returns = pd.read_csv(
            os.path.join(current_dir, 'test_data', 'factor_returns.csv'),
            index_col=0, parse_dates=True
        )

        residuals = pd.read_csv(os.path.join(current_dir, 'test_data', 'residuals.csv'),
                                index_col=0, parse_dates=True)

        residuals.columns = [int(col) for col in residuals.columns]

        intercepts = pd.read_csv(os.path.join(current_dir, 'test_data', 'intercepts.csv'),
                                 index_col=0, header=None)
        if intercepts.shape[1] == 1:
            intercepts = intercepts.iloc[:, 0]

        emp = Empyrical(
            returns=returns,
            positions=positions,
            factor_returns=factor_returns,
            factor_loadings=factor_loadings,
        )

        risk_exposures_portfolio, perf_attrib_output = emp.perf_attrib()

        specific_returns = perf_attrib_output['specific_returns']
        common_returns = perf_attrib_output['common_returns']
        combined_returns = specific_returns + common_returns

        # since all returns are factor returns, common returns should be
        # equivalent to total returns, and specific returns should be 0
        pd.testing.assert_series_equal(returns,
                                       common_returns,
                                       check_names=False)

        self.assertTrue(np.isclose(specific_returns, 0).all())

        # specific and common returns combined should equal total returns
        pd.testing.assert_series_equal(returns,
                                       combined_returns,
                                       check_names=False)

        # check that residuals + intercepts = specific returns
        self.assertTrue(np.isclose((residuals + intercepts), 0).all())

        # check that exposure * factor returns = common returns
        expected_common_returns = risk_exposures_portfolio.multiply(
            factor_returns, axis='rows'
        ).sum(axis='columns')

        pd.testing.assert_series_equal(expected_common_returns,
                                       common_returns,
                                       check_names=False)

        # since factor loadings are ones, portfolio risk exposures
        # should be ones
        pd.testing.assert_frame_equal(
            risk_exposures_portfolio,
            pd.DataFrame(np.ones_like(risk_exposures_portfolio),
                         index=risk_exposures_portfolio.index,
                         columns=risk_exposures_portfolio.columns)
        )
