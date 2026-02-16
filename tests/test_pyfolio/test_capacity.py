from datetime import datetime
from unittest import TestCase

from pandas import DataFrame, Series, date_range
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal
from parameterized import parameterized

from fincore.empyrical import Empyrical


class CapacityTestCase(TestCase):
    dates = date_range(start="2015-01-01", freq="D", periods=3)

    positions = DataFrame([[1.0, 3.0, 0.0], [0.0, 1.0, 1.0], [3.0, 0.0, 1.0]], columns=["A", "B", "cash"], index=dates)

    transactions = DataFrame(
        data=[[1, 100000, 10, "A"]] * len(dates), columns=["sid", "amount", "price", "symbol"], index=dates
    )

    volume = DataFrame([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]], columns=["A", "B"], index=dates)
    volume.index.name = "dt"
    volume = volume * 1000000
    # volume['market_data'] = 'volume'
    price = DataFrame([[1.0, 1.0]] * len(dates), columns=["A", "B"], index=dates)
    price.index.name = "dt"
    # price['market_data'] = 'price'
    # market_data = concat([volume, price]).reset_index().set_index(
    #     ['dt', 'market_data'])
    market_data = {"price": price, "volume": volume}

    def test_days_to_liquidate_positions(self):
        dtlp = Empyrical.days_to_liquidate_positions(
            self.positions,
            self.market_data,
            max_bar_consumption=1,
            capital_base=1e6,
            mean_volume_window=1,
        )

        expected = DataFrame([[0.0, 0.5 / 3], [0.75 / 2, 0.0]], columns=["A", "B"], index=self.dates[1:])
        assert_frame_equal(dtlp, expected)

    def test_get_max_days_to_liquidate_by_ticker(self):
        mdtl = Empyrical.get_max_days_to_liquidate_by_ticker(
            self.positions,
            self.market_data,
            max_bar_consumption=1,
            capital_base=1e6,
            mean_volume_window=1,
        )

        expected = DataFrame(
            [[datetime(2015, 1, 3), 0.75 / 2, 75.0], [datetime(2015, 1, 2), 0.5 / 3, 50.0]],
            columns=["date", "days_to_liquidate", "pos_alloc_pct"],
            index=["A", "B"],
        )
        expected.index.name = "symbol"

        assert_frame_equal(mdtl, expected)

    @parameterized.expand(
        [
            (
                DataFrame(
                    [[datetime(2015, 1, 1), 100.0], [datetime(2015, 1, 2), 100]],
                    columns=["date", "max_pct_bar_consumed"],
                    index=["A", "B"],
                ),
                None,
            ),
            (
                DataFrame(
                    [[datetime(2015, 1, 3), (1 / 3) * 100.0]], columns=["date", "max_pct_bar_consumed"], index=["A"]
                ),
                1,
            ),
        ]
    )
    def test_get_low_liquidity_transactions(self, expected, last_n_days):
        txn_daily = DataFrame(
            data=[[1, 1000000, 1, "A"], [2, 2000000, 1, "B"], [1, 1000000, 1, "A"]],
            columns=["sid", "amount", "price", "symbol"],
            index=self.dates,
        )

        llt = Empyrical.get_low_liquidity_transactions(
            txn_daily,
            self.market_data,
            last_n_days=last_n_days,
        )
        expected.index.name = "symbol"
        assert_frame_equal(llt, expected)

    def test_daily_txns_with_bar_data(self):
        daily_txn = Empyrical.daily_txns_with_bar_data(
            self.transactions,
            self.market_data,
        )
        expected = DataFrame(
            data=[["A", 100000, 1.0, 1000000.0], ["A", 100000, 1.0, 2000000.0], ["A", 100000, 1.0, 3000000.0]],
            columns=["symbol", "amount", "price", "volume"],
            index=self.dates,
        )
        print("daily_txn = ", daily_txn)
        print("expected = ", expected)
        print("daily_txn.info()  = ", daily_txn.info())
        print("expected.info() = ", expected.info())
        # assert_frame_equal(daily_txn, expected, check_less_precise=True)
        # Ensure the indices match
        assert_index_equal(daily_txn.index, expected.index)
        # Set the frequency of the daily_txn index to match the expected index
        daily_txn.index.freq = expected.index.freq
        # daily_txn.index = daily_txn.index.asfreq('D')
        # Analyze differences
        # analyze_dataframe_differences(daily_txn, expected)
        # assert_frame_equal(daily_txn, expected, atol=1e-4)
        assert_frame_equal(daily_txn, expected)

    @parameterized.expand(
        [
            (1000000, 1, [0.9995, 0.9999375, 0.99998611]),
            (10000000, 1, [0.95, 0.99375, 0.998611]),
            (100000, 1, [0.999995, 0.999999375, 0.9999998611]),
            (1000000, 0.1, [0.99995, 0.99999375, 0.999998611]),
        ]
    )
    def test_apply_slippage_penalty(self, starting_base, impact, expected_adj_returns):
        returns = Series([1.0, 1.0, 1.0], index=self.dates)
        daily_txn = Empyrical.daily_txns_with_bar_data(
            self.transactions,
            self.market_data,
        )

        adj_returns = Empyrical.apply_slippage_penalty(
            returns,
            daily_txn,
            starting_base,
            1000000,
            impact=impact,
        )
        expected_adj_returns = Series(expected_adj_returns, index=self.dates)

        assert_series_equal(adj_returns, expected_adj_returns)
