"""Annual alpha/beta common-year grouping behavior tests."""

import numpy as np
import pandas as pd

from fincore.metrics import alpha_beta


class TestAnnualAlphaLine557:
    """Annual alpha groups only the labels retained by alignment."""

    def test_annual_alpha_keeps_only_common_year(self):
        """Partial labels are aligned before calendar-year grouping."""
        returns = pd.Series(
            [0.01, 0.02, 0.015, 0.01],
            index=pd.to_datetime(["2020-01-01", "2021-01-01", "2021-01-02", "2021-01-03"]),
        )
        factor_returns = pd.Series(
            [0.005, 0.01, 0.008, 0.004],
            index=pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03", "2022-01-01"]),
        )

        result = alpha_beta.annual_alpha(returns, factor_returns)

        assert result.index.tolist() == [2021]
        assert np.isfinite(result.iloc[0])


class TestAnnualBetaLine610:
    """Annual beta groups only the labels retained by alignment."""

    def test_annual_beta_keeps_only_common_year(self):
        """Partial labels are aligned before calendar-year grouping."""
        returns = pd.Series(
            [0.01, 0.02, 0.015, 0.01],
            index=pd.to_datetime(["2020-01-01", "2021-01-01", "2021-01-02", "2021-01-03"]),
        )
        factor_returns = pd.Series(
            [0.005, 0.01, 0.008, 0.004],
            index=pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03", "2022-01-01"]),
        )

        result = alpha_beta.annual_beta(returns, factor_returns)

        assert result.index.tolist() == [2021]
        assert np.isfinite(result.iloc[0])
