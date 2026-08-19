"""RequestPolicy contract tests."""

from __future__ import annotations

import pytest

from fincore.data.contracts import RequestPolicy


def test_request_policy_defaults() -> None:
    policy = RequestPolicy()
    assert policy.connect_timeout == 10.0
    assert policy.read_timeout == 30.0
    assert policy.total_timeout == 60.0
    assert policy.max_attempts == 3


def test_request_policy_accepts_custom_values() -> None:
    policy = RequestPolicy(
        connect_timeout=1.0,
        read_timeout=2.0,
        total_timeout=3.0,
        max_attempts=5,
    )
    assert policy.max_attempts == 5
    assert policy.connect_timeout == 1.0


@pytest.mark.parametrize("field", ["connect_timeout", "read_timeout", "total_timeout"])
def test_request_policy_rejects_nonpositive_timeout(field: str) -> None:
    with pytest.raises(ValueError, match="positive"):
        RequestPolicy(**{field: 0.0})


@pytest.mark.parametrize("field", ["connect_timeout", "read_timeout", "total_timeout"])
def test_request_policy_rejects_negative_timeout(field: str) -> None:
    with pytest.raises(ValueError, match="positive"):
        RequestPolicy(**{field: -1.0})


def test_request_policy_rejects_zero_attempts() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        RequestPolicy(max_attempts=0)


def test_request_policy_rejects_negative_attempts() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        RequestPolicy(max_attempts=-2)


@pytest.mark.parametrize("exc", [ConnectionError(), TimeoutError(), OSError()])
def test_should_retry_transient_errors(exc: BaseException) -> None:
    assert RequestPolicy().should_retry(exc) is True


@pytest.mark.parametrize(
    "exc",
    [ValueError("bad"), TypeError("bad"), KeyError("k"), RuntimeError("r")],
)
def test_should_retry_non_transient_errors(exc: BaseException) -> None:
    assert RequestPolicy().should_retry(exc) is False


def test_remaining_attempts() -> None:
    policy = RequestPolicy(max_attempts=3)
    assert policy.remaining_attempts(1) == 2
    assert policy.remaining_attempts(2) == 1
    assert policy.remaining_attempts(3) == 0


def test_remaining_attempts_clamps_at_zero() -> None:
    policy = RequestPolicy(max_attempts=3)
    assert policy.remaining_attempts(4) == 0
    assert policy.remaining_attempts(100) == 0
