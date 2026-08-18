"""Reproducible data-provider contracts.

These contracts make external-data access explicit: a bounded request policy
with deterministic retry classification, and typed provider protocols.  They
are import-light (no optional SDK is imported at module load) and do not ship
a default network fetcher.
"""

from __future__ import annotations

from dataclasses import dataclass

# Transient transport errors are retryable; caller validation errors are not.
_TRANSIENT_ERRORS = (ConnectionError, TimeoutError, OSError)


@dataclass(frozen=True)
class RequestPolicy:
    """A bounded, deterministic HTTP/data-access policy.

    ``should_retry`` classifies only transient transport errors as retryable.
    Caller validation errors (``ValueError``, ``TypeError``) are never retried,
    and credentials are never logged by the callers that honour this policy.
    """

    connect_timeout: float = 10.0
    read_timeout: float = 30.0
    total_timeout: float = 60.0
    max_attempts: int = 3

    def __post_init__(self) -> None:
        if self.connect_timeout <= 0 or self.read_timeout <= 0 or self.total_timeout <= 0:
            raise ValueError("timeouts must be positive")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

    def should_retry(self, exc: BaseException) -> bool:
        """Return True only for transient transport errors."""
        return isinstance(exc, _TRANSIENT_ERRORS)

    def remaining_attempts(self, attempt: int) -> int:
        """Return how many attempts remain for the given 1-based attempt."""
        return max(0, self.max_attempts - attempt)


__all__ = ["RequestPolicy"]
