import pandas as pd
import pytest

from fincore.attribution import fama_french


@pytest.fixture(autouse=True)
def _reset_ff_provider() -> None:
    # Keep tests isolated: provider and cache are module-level globals.
    fama_french.set_ff_provider(None)
    yield
    fama_french.set_ff_provider(None)


def test_fetch_ff_factors_raises_without_provider() -> None:
    with pytest.raises(NotImplementedError):
        fama_french.fetch_ff_factors("2020-01-01", "2020-01-31", library="french")


def test_fetch_ff_factors_caches_module_provider_and_returns_copies() -> None:
    calls: dict[str, int] = {"n": 0}

    def provider(start: str, end: str, library: str) -> pd.DataFrame:
        calls["n"] += 1
        # Minimal deterministic payload; values matter for mutation/copy checks.
        return pd.DataFrame(
            {"MKT": [0.01, 0.02], "SMB": [0.0, 0.0], "HML": [0.0, 0.0]},
            index=pd.date_range("2020-01-01", periods=2, freq="D"),
        )

    fama_french.set_ff_provider(provider)

    df1 = fama_french.fetch_ff_factors("2020-01-01", "2020-01-31", library="french")
    df1.iloc[0, 0] = 999.0

    df2 = fama_french.fetch_ff_factors("2020-01-01", "2020-01-31", library="french")

    assert calls["n"] == 1
    assert float(df2.iloc[0, 0]) == 0.01
