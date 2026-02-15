import pandas as pd

from fincore.report.render_html import generate_html


def test_generate_html_writes_file_and_contains_expected_sections(tmp_path) -> None:
    # Minimal deterministic returns series.
    idx = pd.date_range("2024-01-01", periods=60, freq="B", tz="UTC")
    # Avoid constant returns (can trigger scipy moment precision RuntimeWarnings which are treated as errors).
    values = [(0.001 if (i % 2 == 0) else -0.0007) + (i % 7) * 1e-5 for i in range(len(idx))]
    returns = pd.Series(values, index=idx, name="strategy")

    out = tmp_path / "report.html"
    title = "Unit Test Report"

    result = generate_html(
        returns,
        benchmark_rets=None,
        positions=None,
        transactions=None,
        trades=None,
        title=title,
        output=str(out),
        rolling_window=20,
    )

    assert result == str(out)
    assert out.exists()

    html = out.read_text(encoding="utf-8")
    assert title in html
    # Sidebar section anchors
    assert 'href="#overview"' in html
    assert 'href="#period"' in html
    assert 'href="#performance"' in html
    assert 'href="#returns"' in html
    assert 'href="#rolling"' in html
    assert 'href="#drawdown"' in html
    # Report chart payload marker
    assert "var D=" in html


def test_generate_html_with_benchmark_positions_transactions_and_trades(tmp_path) -> None:
    idx = pd.date_range("2024-01-01", periods=80, freq="B", tz="UTC")
    values = [(0.0012 if (i % 3 == 0) else -0.0005) + (i % 11) * 1e-5 for i in range(len(idx))]
    returns = pd.Series(values, index=idx, name="strategy")
    benchmark = pd.Series([v * 0.8 for v in values], index=idx, name="benchmark")

    positions = pd.DataFrame(
        {
            "AAA": 100.0,
            "BBB": 200.0,
            "cash": 50.0,
        },
        index=idx,
    )

    tx_idx = pd.to_datetime(
        [
            "2024-02-01 10:00:00",
            "2024-02-01 15:30:00",
            "2024-02-02 11:15:00",
        ],
        utc=True,
    )
    transactions = pd.DataFrame(
        {
            "amount": [10, -5, 7],
            "price": [10.0, 12.0, 11.0],
            "symbol": ["AAA", "AAA", "BBB"],
        },
        index=tx_idx,
    )

    trades = pd.DataFrame(
        {
            "pnlcomm": [10.0, -6.0, 3.0],
            "commission": [0.1, 0.1, 0.1],
            "long": [1, 0, 1],
            "barlen": [5, 3, 2],
        }
    )

    out = tmp_path / "report_full.html"
    result = generate_html(
        returns,
        benchmark_rets=benchmark,
        positions=positions,
        transactions=transactions,
        trades=trades,
        title="Full Inputs",
        output=str(out),
        rolling_window=20,
    )
    assert result == str(out)

    html = out.read_text(encoding="utf-8")
    assert 'href="#benchmark"' in html
    assert 'href="#positions"' in html
    assert 'href="#transactions"' in html
    assert 'href="#trades"' in html
