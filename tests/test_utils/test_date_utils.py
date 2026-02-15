import time

from fincore.utils.date_utils import timer


def test_timer_prints_and_returns_timestamp(capsys) -> None:
    prev = time.time() - 0.5
    now = timer("unit-test", prev)
    assert now >= prev

    out = capsys.readouterr().out
    assert "Finished unit-test" in out
