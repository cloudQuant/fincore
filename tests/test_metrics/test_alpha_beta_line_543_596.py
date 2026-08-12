"""Annual alpha/beta empty-alignment behavior tests."""

import pandas as pd
import pytest

from fincore.metrics import alpha_beta


@pytest.mark.serial
class TestAnnualAlphaLine543:
    """Test to cover line 543 in alpha_beta.py.

    Line 543 is hit when after aligned_series, len(returns) < 1.
    """

    def test_annual_alpha_empty_after_alignment(self):
        """Disjoint labels produce an empty enhanced annual-alpha result."""
        returns = pd.Series(
            [0.01, 0.02, 0.015],
            index=pd.date_range("2020-01-01", periods=3),
        )
        factor_returns = pd.Series(
            [0.005, 0.01, 0.008],
            index=pd.date_range("2021-01-01", periods=3),
        )

        result = alpha_beta.annual_alpha(returns, factor_returns)

        assert isinstance(result, pd.Series)
        assert result.empty


@pytest.mark.serial
class TestAnnualBetaLine596:
    """Test to cover line 596 in alpha_beta.py (similar to line 543)."""

    def test_annual_beta_empty_after_alignment(self):
        """Disjoint labels produce an empty enhanced annual-beta result."""
        returns = pd.Series(
            [0.01, 0.02, 0.015],
            index=pd.date_range("2020-01-01", periods=3),
        )
        factor_returns = pd.Series(
            [0.005, 0.01, 0.008],
            index=pd.date_range("2021-01-01", periods=3),
        )

        result = alpha_beta.annual_beta(returns, factor_returns)

        assert isinstance(result, pd.Series)
        assert result.empty
