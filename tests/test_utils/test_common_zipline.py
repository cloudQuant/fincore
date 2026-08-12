"""Tests for zipline-related utilities in fincore.utils.common_utils."""

from __future__ import annotations

import pandas as pd
import pytest

from fincore.utils import common_utils as cu

EXPECTED_TRANSACTION_COLUMNS = [
    "dt",
    "sid",
    "symbol",
    "amount",
    "price",
    "order_id",
    "commission",
    "txn_dollars",
]


def test_extract_rets_pos_txn_from_zipline_smoke(monkeypatch):
    """Test extract_rets_pos_txn_from_zipline basic functionality."""
    # Patch Empyrical helpers so we can run without zipline.
    from fincore.empyrical import Empyrical

    def fake_extract_pos(positions_df, ending_cash):
        # Minimal extract: pivot values by sid and join cash.
        df = positions_df.copy()
        df["values"] = df["amount"] * df["last_sale_price"]
        out = df.reset_index().pivot_table(index="index", columns="sid", values="values").fillna(0)
        cash = ending_cash.copy()
        cash.name = "cash"
        out = out.join(cash).fillna(0)
        out.columns.name = "sid"
        return out

    def fake_make_transaction_frame(txn):
        return pd.DataFrame(
            {"amount": [1], "price": [10.0], "symbol": ["A"]},
            index=pd.to_datetime(["2024-01-02 10:00"]),
        )

    monkeypatch.setattr(Empyrical, "extract_pos", staticmethod(fake_extract_pos))
    monkeypatch.setattr(Empyrical, "make_transaction_frame", staticmethod(fake_make_transaction_frame))

    class Backtest:
        def __init__(self) -> None:
            self.index = pd.date_range("2024-01-01", periods=2, freq="B")
            self.returns = pd.Series([0.0, 0.01], index=self.index)
            self.ending_cash = pd.Series([100.0, 101.0], index=self.index)
            self.positions = {
                self.index[0]: [{"sid": "A", "amount": 1.0, "last_sale_price": 10.0}],
                self.index[1]: [{"sid": "A", "amount": 2.0, "last_sale_price": 11.0}],
            }
            self.transactions = []

    bt = Backtest()
    rets, pos, txn = cu.extract_rets_pos_txn_from_zipline(bt)
    assert isinstance(rets, pd.Series)
    assert isinstance(pos, pd.DataFrame)
    assert isinstance(txn, pd.DataFrame)
    assert str(txn.index.tz) in {"UTC", "utc"}


def test_extract_rets_pos_txn_from_zipline_raises_when_no_positions():
    """Test extract_rets_pos_txn_from_zipline raises when no positions."""

    class Backtest:
        def __init__(self) -> None:
            self.index = pd.date_range("2024-01-01", periods=2, freq="B")
            self.returns = pd.Series([0.0, 0.01], index=self.index)
            self.ending_cash = pd.Series([100.0, 101.0], index=self.index)
            self.positions = {}
            self.transactions = []

    with pytest.raises(ValueError, match="does not have any positions"):
        cu.extract_rets_pos_txn_from_zipline(Backtest())


def test_extract_rets_pos_txn_from_zipline_runs_real_position_and_transaction_chain():
    """Exercise the real helpers, including Zipline's date-to-list protocol."""

    dates = pd.date_range("2024-08-01", periods=2, freq="B")
    first_txn_dt = pd.Timestamp("2024-08-01 14:30:00", tz="UTC")
    second_txn_dt = pd.Timestamp("2024-08-02 15:45:00", tz="UTC")
    backtest = pd.DataFrame(index=dates)
    backtest["returns"] = [0.0, 0.01]
    backtest["ending_cash"] = [90.0, 79.0]
    backtest["positions"] = [
        [{"sid": "AAA", "amount": 1.0, "last_sale_price": 10.0}],
        [{"sid": "AAA", "amount": 2.0, "last_sale_price": 11.0}],
    ]
    backtest["transactions"] = [
        [
            {
                "sid": {"sid": 101, "symbol": "AAA"},
                "amount": 1.0,
                "price": 10.0,
                "order_id": "order-a",
                "commission": 0.1,
                "dt": first_txn_dt,
            }
        ],
        [
            {
                "sid": {"sid": 101, "symbol": "AAA"},
                "amount": -1.0,
                "price": 11.0,
                "order_id": "order-b",
                "commission": 0.2,
                "dt": second_txn_dt,
            }
        ],
    ]

    returns, positions, transactions = cu.extract_rets_pos_txn_from_zipline(backtest)

    assert str(returns.index.tz).lower() == "utc"
    assert positions.columns.tolist() == ["AAA", "cash"]
    assert positions["AAA"].tolist() == [10.0, 22.0]
    assert positions["cash"].tolist() == [90.0, 79.0]
    assert transactions.columns.tolist() == EXPECTED_TRANSACTION_COLUMNS
    assert transactions["sid"].tolist() == [101, 101]
    assert transactions["symbol"].tolist() == ["AAA", "AAA"]
    assert transactions["order_id"].tolist() == ["order-a", "order-b"]
    assert transactions["commission"].tolist() == [0.1, 0.2]
    assert transactions["txn_dollars"].tolist() == [-10.0, 11.0]
    assert transactions.index.tolist() == [first_txn_dt, second_txn_dt]
