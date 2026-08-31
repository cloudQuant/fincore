"""Causal point-in-time factor materialization tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.factor_analysis.data import prepare_pit_factor_data
from fincore.factor_analysis.pit import PITPoint, materialize_pit_factor, validate_pit_alignment


def _observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asset": ["A", "B", "A", "B"],
            "as_of": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-03", "2024-01-03"], utc=True),
            "known_at": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-04", "2024-01-04"], utc=True),
            "effective_from": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-04", "2024-01-04"], utc=True),
            "value": [1.0, -1.0, 2.0, -2.0],
            "in_universe": [True, True, True, False],
        }
    )


class TestPITFactorMaterialization:
    def test_exposes_the_causal_path_from_its_owning_modules(self) -> None:
        assert PITPoint.__module__ == "fincore.factor_analysis.pit"
        assert materialize_pit_factor.__module__ == "fincore.factor_analysis.pit"
        assert prepare_pit_factor_data.__module__ == "fincore.factor_analysis.data"
        assert validate_pit_alignment.__module__ == "fincore.factor_analysis.pit"

    def test_pit_point_enforces_the_event_time_contract(self) -> None:
        as_of = pd.Timestamp("2024-01-01", tz="UTC")
        point = PITPoint(as_of, as_of, as_of, np.float64(1.5))

        assert point.value == 1.5
        validate_pit_alignment((point,))
        with pytest.raises(ValueError, match="effective_from"):
            PITPoint(as_of, as_of, as_of - pd.Timedelta(days=1), 1.0)

    def test_selects_only_values_known_and_effective_on_each_evaluation_date(self) -> None:
        evaluation_dates = pd.date_range("2024-01-02", periods=3, tz="UTC")

        factor = materialize_pit_factor(_observations(), evaluation_dates)

        expected = pd.Series(
            [1.0, -1.0, 1.0, -1.0, 2.0],
            index=pd.MultiIndex.from_tuples(
                [
                    (evaluation_dates[0], "A"),
                    (evaluation_dates[0], "B"),
                    (evaluation_dates[1], "A"),
                    (evaluation_dates[1], "B"),
                    (evaluation_dates[2], "A"),
                ],
                names=("date", "asset"),
            ),
            name="factor",
        )
        pd.testing.assert_series_equal(factor, expected)

    def test_future_observation_cannot_change_prior_materialized_factor_values(self) -> None:
        evaluation_dates = pd.date_range("2024-01-02", periods=4, tz="UTC")
        baseline = materialize_pit_factor(_observations(), evaluation_dates)
        future = pd.concat(
            (
                _observations(),
                pd.DataFrame(
                    {
                        "asset": ["A"],
                        "as_of": pd.to_datetime(["2024-01-05"], utc=True),
                        "known_at": pd.to_datetime(["2024-01-06"], utc=True),
                        "effective_from": pd.to_datetime(["2024-01-06"], utc=True),
                        "value": [999.0],
                        "in_universe": [True],
                    }
                ),
            ),
            ignore_index=True,
        )

        perturbed = materialize_pit_factor(future, evaluation_dates)

        pd.testing.assert_series_equal(perturbed, baseline)

    @pytest.mark.parametrize(
        "column,value,match",
        [
            ("known_at", pd.Timestamp("2023-12-31", tz="UTC"), "known_at"),
            ("effective_from", pd.Timestamp("2024-01-01", tz="UTC"), "effective_from"),
            ("value", np.inf, "finite"),
        ],
    )
    def test_rejects_invalid_causal_observation_contract(self, column: str, value: object, match: str) -> None:
        observations = _observations()
        observations.loc[0, column] = value

        with pytest.raises(ValueError, match=match):
            materialize_pit_factor(observations, pd.date_range("2024-01-02", periods=2, tz="UTC"))

    def test_prepare_pit_factor_data_uses_causal_materialization_and_forbids_global_zscore_filter(self) -> None:
        observations = _observations()
        prices = pd.DataFrame(
            {
                "A": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
                "B": [100.0, 99.0, 98.0, 97.0, 96.0, 95.0],
            },
            index=pd.date_range("2024-01-02", periods=6, tz="UTC"),
        )
        evaluation_dates = pd.date_range("2024-01-02", periods=2, tz="UTC")

        prepared = prepare_pit_factor_data(
            observations,
            prices,
            evaluation_dates,
            periods=(1,),
            quantiles=2,
            max_loss=1.0,
        )

        assert set(prepared.data.index.get_level_values("date")) <= set(evaluation_dates)
        assert prepared.data.loc[(evaluation_dates[0], "A"), "factor"] == 1.0
        with pytest.raises(ValueError, match="filter_zscore"):
            prepare_pit_factor_data(
                observations,
                prices,
                evaluation_dates,
                periods=(1,),
                filter_zscore=3.0,
            )

    def test_prepare_pit_factor_data_fails_closed_when_no_observation_is_yet_eligible(self) -> None:
        observations = _observations()
        prices = pd.DataFrame(
            {"A": [100.0, 101.0], "B": [100.0, 99.0]},
            index=pd.date_range("2024-01-01", periods=2, tz="UTC"),
        )

        with pytest.raises(ValueError, match="no eligible factor values"):
            prepare_pit_factor_data(
                observations,
                prices,
                pd.date_range("2024-01-01", periods=1, tz="UTC"),
                periods=(1,),
            )
