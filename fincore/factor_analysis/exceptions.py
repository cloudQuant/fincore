"""Separate enhanced-kernel and strict-compatibility exception identities.

The enhanced kernel exposes ``ValueError``-derived validation errors.  The
pinned Alphalens facade, however, deliberately preserves its two direct
``Exception`` subclasses at the adapter boundary.
"""


class FactorDataError(ValueError):
    """Base class for invalid factor-analysis inputs."""


class EnhancedNonMatchingTimezoneError(FactorDataError):
    """Factor and price timestamps use different timezone conventions."""


class FactorLossExceededError(FactorDataError):
    """The enhanced kernel dropped more input rows than the accepted limit."""

    def __init__(self, message: str = "", report: object | None = None) -> None:
        super().__init__(message)
        self.report = report


class MaxLossExceededError(Exception):
    """Pinned strict identity for a rejected cleaning-loss budget."""

    def __init__(self, message: str = "", report: object | None = None) -> None:
        super().__init__(message)
        self.report = report


class NonMatchingTimezoneError(Exception):
    """Pinned strict identity for a factor/prices timezone mismatch."""


__all__ = [
    "EnhancedNonMatchingTimezoneError",
    "FactorDataError",
    "FactorLossExceededError",
    "MaxLossExceededError",
    "NonMatchingTimezoneError",
]
