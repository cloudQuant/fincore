import pandas as pd

import fincore.attribution.fama_french as fama_french


def test_fetch_ff_factors_uses_an_explicit_provider_and_returns_copies() -> None:
    calls: dict[str, int] = {"n": 0}
    source = pd.DataFrame(
        {"MKT": [0.01, 0.02], "SMB": [0.0, 0.0], "HML": [0.0, 0.0]},
        index=pd.date_range("2020-01-01", periods=2, freq="D"),
    )

    def provider(start: str, end: str, library: str) -> pd.DataFrame:
        calls["n"] += 1
        assert start == "2020-01-01"
        assert end == "2020-01-31"
        assert library == "french"
        return source

    df1 = fama_french.fetch_ff_factors("2020-01-01", "2020-01-31", library="french", provider=provider)
    df1.iloc[0, 0] = 999.0

    df2 = fama_french.fetch_ff_factors("2020-01-01", "2020-01-31", library="french", provider=provider)

    assert calls["n"] == 2
    assert float(source.iloc[0, 0]) == 0.01
    assert float(df2.iloc[0, 0]) == 0.01
