"""Statistical inference for factor research.

Provides Fama-MacBeth cross-sectional regression, IC confidence intervals and
multiple-testing (Benjamini-Hochberg) correction so factor conclusions carry
inference, not just point estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "FDRResult",
    "ICInferenceResult",
    "benjamini_hochberg",
    "factor_model_inference",
    "fama_macbeth",
    "ic_confidence_interval",
    "ic_mean",
    "ic_t_stat",
    "information_coefficient_inference",
]


@dataclass(frozen=True, slots=True)
class FDRResult:
    """Audit-friendly result of a Benjamini-Hochberg correction.

    The three series retain the input factor labels. ``adjusted_p_values`` are
    the monotone BH q-values and ``rejected`` is the decision at ``alpha``.
    """

    alpha: float
    p_values: pd.Series
    adjusted_p_values: pd.Series
    rejected: pd.Series
    method: Literal["benjamini-hochberg"] = "benjamini-hochberg"

    @property
    def n_tests(self) -> int:
        """Number of hypotheses corrected together."""
        return len(self.p_values)


@dataclass(frozen=True, slots=True)
class ICInferenceResult:
    """Auditable two-sided IC tests with Benjamini-Hochberg correction.

    ``hypotheses`` is indexed by forward-return period and contains
    ``n_observations``, ``mean_ic``, ``t_stat``, ``p_value``,
    ``adjusted_p_value``, ``rejected``, and ``testable``. Untestable periods
    (fewer than two finite IC observations) retain their sample information
    but receive ``NaN`` p/q values and are never treated as discoveries.
    """

    alpha: float
    hypotheses: pd.DataFrame
    method: Literal["two-sided-student-t+benjamini-hochberg"] = "two-sided-student-t+benjamini-hochberg"

    def __post_init__(self) -> None:
        """Own a validated, schema-stable result table for downstream audit."""

        alpha = _normalize_alpha(self.alpha)
        if not isinstance(self.hypotheses, pd.DataFrame):
            raise TypeError("hypotheses must be a pandas DataFrame")
        expected_columns = (
            "n_observations",
            "mean_ic",
            "t_stat",
            "p_value",
            "adjusted_p_value",
            "rejected",
            "testable",
        )
        if tuple(self.hypotheses.columns) != expected_columns:
            raise ValueError(f"hypotheses must use columns {expected_columns!r}")
        if self.hypotheses.index.has_duplicates:
            raise ValueError("hypotheses index must contain unique forward-period labels")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "hypotheses", self.hypotheses.copy(deep=True))

    @property
    def n_hypotheses(self) -> int:
        """Number of forward-period hypotheses represented in the table."""

        return len(self.hypotheses)

    @property
    def n_tested(self) -> int:
        """Number of finite, two-sided tests included in the BH family."""

        return int(self.hypotheses["testable"].sum())


def _normalize_alpha(alpha: object) -> float:
    """Normalize an FDR target while rejecting boolean and non-finite inputs."""

    if isinstance(alpha, bool):
        raise ValueError("alpha must be a finite probability in (0, 1]")
    try:
        alpha_value = float(cast("Any", alpha))
    except (TypeError, ValueError) as error:
        raise ValueError("alpha must be a finite probability in (0, 1]") from error
    if not np.isfinite(alpha_value) or not 0.0 < alpha_value <= 1.0:
        raise ValueError("alpha must be a finite probability in (0, 1]")
    return alpha_value


def benjamini_hochberg(
    p_values: pd.Series | np.ndarray,
    *,
    alpha: float = 0.05,
) -> FDRResult:
    """Apply Benjamini-Hochberg false-discovery-rate correction.

    Parameters
    ----------
    p_values
        One-dimensional, finite p-values in ``[0, 1]``. A Series preserves
        its factor labels; an ndarray receives a RangeIndex.
    alpha
        Target false-discovery rate in ``(0, 1]``.

    Returns
    -------
    FDRResult
        Original p-values, adjusted BH q-values, and the step-up rejection
        decision. Inputs with duplicate Series labels are rejected because an
        audit result cannot identify a unique hypothesis.
    """
    alpha_value = _normalize_alpha(alpha)

    if isinstance(p_values, pd.Series):
        if p_values.index.has_duplicates:
            raise ValueError("p_values index must not contain duplicate hypothesis labels")
        source = p_values.astype(float).copy()
    elif isinstance(p_values, np.ndarray):
        try:
            values = np.asarray(p_values, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("p_values must be numeric") from error
        source = pd.Series(values, name="p_value")
    else:
        raise TypeError("p_values must be a pandas Series or numpy ndarray")

    values = source.to_numpy(dtype=float)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must be finite probabilities in [0, 1]")

    if not len(values):
        return FDRResult(
            alpha=alpha_value,
            p_values=source,
            adjusted_p_values=pd.Series([], index=source.index, dtype=float, name="adjusted_p_value"),
            rejected=pd.Series([], index=source.index, dtype=bool, name="rejected"),
        )

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.arange(1, len(values) + 1, dtype=float)
    thresholds = alpha_value * ranks / len(values)
    accepted = sorted_values <= thresholds
    sorted_rejected = np.zeros(len(values), dtype=bool)
    if accepted.any():
        sorted_rejected[: np.flatnonzero(accepted)[-1] + 1] = True

    sorted_adjusted = np.minimum.accumulate((len(values) * sorted_values / ranks)[::-1])[::-1]
    sorted_adjusted = np.clip(sorted_adjusted, 0.0, 1.0)
    adjusted = np.empty(len(values), dtype=float)
    rejected = np.empty(len(values), dtype=bool)
    adjusted[order] = sorted_adjusted
    rejected[order] = sorted_rejected

    return FDRResult(
        alpha=alpha_value,
        p_values=source,
        adjusted_p_values=pd.Series(adjusted, index=source.index, name="adjusted_p_value"),
        rejected=pd.Series(rejected, index=source.index, name="rejected"),
    )


def information_coefficient_inference(
    information_coefficient: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> ICInferenceResult:
    """Test per-period factor IC means and correct the resulting p-values.

    The input is the date-by-period IC snapshot produced by the enhanced
    factor workflow. Each period uses a two-sided Student t test of mean IC
    against zero under the explicit i.i.d. time-series assumption. Only
    periods with at least two finite observations enter the
    Benjamini-Hochberg family; untestable periods remain visible but cannot be
    reported as discoveries.
    """

    alpha_value = _normalize_alpha(alpha)
    if not isinstance(information_coefficient, pd.DataFrame):
        raise TypeError("information_coefficient must be a pandas DataFrame")
    if information_coefficient.columns.has_duplicates:
        raise ValueError("information_coefficient columns must not contain duplicate forward-period labels")

    periods = information_coefficient.columns.copy()
    records: list[dict[str, object]] = []
    p_values: dict[object, float] = {}
    for period in periods:
        observations = _clean_ic_observations(information_coefficient[period])
        n_observations = len(observations)
        mean_ic = float(np.mean(observations)) if n_observations else float("nan")
        if n_observations < 2:
            records.append(
                {
                    "n_observations": n_observations,
                    "mean_ic": mean_ic,
                    "t_stat": float("nan"),
                    "p_value": float("nan"),
                    "adjusted_p_value": float("nan"),
                    "rejected": False,
                    "testable": False,
                }
            )
            continue
        t_statistic = ic_t_stat(observations)
        p_value = _two_sided_student_t_p_value(t_statistic, n_observations)
        p_values[period] = p_value
        records.append(
            {
                "n_observations": n_observations,
                "mean_ic": mean_ic,
                "t_stat": t_statistic,
                "p_value": p_value,
                "adjusted_p_value": float("nan"),
                "rejected": False,
                "testable": True,
            }
        )

    hypotheses = pd.DataFrame(
        records,
        index=periods.rename(periods.name or "forward_period"),
        columns=(
            "n_observations",
            "mean_ic",
            "t_stat",
            "p_value",
            "adjusted_p_value",
            "rejected",
            "testable",
        ),
    )
    if p_values:
        correction = benjamini_hochberg(pd.Series(p_values, name="p_value"), alpha=alpha_value)
        hypotheses.loc[correction.p_values.index, "adjusted_p_value"] = correction.adjusted_p_values
        hypotheses.loc[correction.p_values.index, "rejected"] = correction.rejected
    hypotheses["n_observations"] = hypotheses["n_observations"].astype(int)
    hypotheses["rejected"] = hypotheses["rejected"].astype(bool)
    hypotheses["testable"] = hypotheses["testable"].astype(bool)
    return ICInferenceResult(alpha=alpha_value, hypotheses=hypotheses)


def factor_model_inference(model: object, *, alpha: float = 0.05) -> ICInferenceResult:
    """Run auditable IC/FDR inference on an enhanced ``FactorAnalysisModel``.

    This is an additive post-analysis step. It consumes the model's stored
    aggregate IC snapshot rather than recomputing factor returns, so a caller
    can retain a deterministic prepare → analyze → infer research trail.
    """

    from fincore.factor_analysis.models import FactorAnalysisModel

    if not isinstance(model, FactorAnalysisModel):
        raise TypeError("model must be a FactorAnalysisModel")
    return information_coefficient_inference(model.aggregate_information_coefficient, alpha=alpha)


def _two_sided_student_t_p_value(t_statistic: float, n_observations: int) -> float:
    """Return the exact two-sided Student-t tail probability for a finite sample."""

    if n_observations < 2:  # pragma: no cover - protected by the public boundary
        raise ValueError("at least two observations are required for a t-test")
    if np.isnan(t_statistic):  # pragma: no cover - protected by ic_t_stat contract
        raise ValueError("t-statistic must not be NaN for a testable IC series")
    p_value = float(2.0 * stats.t.sf(abs(t_statistic), df=n_observations - 1))
    if not np.isfinite(p_value):  # pragma: no cover - scipy distribution contract guard
        raise ValueError("Student-t p-value must be finite")
    return float(np.clip(p_value, 0.0, 1.0))


def _clean_ic_observations(ic_series: pd.Series | np.ndarray) -> np.ndarray:
    """Drop missing IC observations but reject unbounded inference inputs."""
    try:
        values = np.asarray(ic_series, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("IC observations must be numeric") from error
    if values.ndim != 1:
        raise ValueError("IC observations must be one-dimensional")
    if np.isinf(values).any():
        raise ValueError("IC observations must not contain infinite values")
    return cast("np.ndarray", values[~np.isnan(values)])


def ic_mean(ic_series: pd.Series | np.ndarray) -> float:
    """Mean information coefficient over time."""
    ic = _clean_ic_observations(ic_series)
    return float(np.mean(ic)) if len(ic) else float("nan")


def ic_t_stat(ic_series: pd.Series | np.ndarray) -> float:
    """t-statistic of the mean IC (Newey-West-free, i.i.d. assumption)."""
    ic = _clean_ic_observations(ic_series)
    n = len(ic)
    if n < 2:
        return float("nan")
    se = np.std(ic, ddof=1) / np.sqrt(n)
    if se < 1e-15:
        mean = float(np.mean(ic))
        if mean > 0.0:
            return float("inf")
        if mean < 0.0:
            return float("-inf")
        return 0.0
    return float(np.mean(ic) / se)


def ic_confidence_interval(ic_series: pd.Series | np.ndarray, *, z: float = 1.96) -> tuple[float, float]:
    """95% (by default) confidence interval for the mean IC."""
    if isinstance(z, bool):
        raise ValueError("z must be a finite positive multiplier")
    try:
        z_value = float(z)
    except (TypeError, ValueError) as error:
        raise ValueError("z must be a finite positive multiplier") from error
    if not np.isfinite(z_value) or z_value <= 0.0:
        raise ValueError("z must be a finite positive multiplier")
    ic = _clean_ic_observations(ic_series)
    n = len(ic)
    if n < 2:
        return (float("nan"), float("nan"))
    se = np.std(ic, ddof=1) / np.sqrt(n)
    mean = float(np.mean(ic))
    return (mean - z_value * se, mean + z_value * se)


def fama_macbeth(
    returns: pd.DataFrame,
    exposures: pd.DataFrame,
    *,
    covariance: Literal["iid", "newey-west"] = "iid",
    newey_west_lags: int = 1,
) -> pd.DataFrame:
    """Fama-MacBeth cross-sectional regression.

    ``returns`` is a ``(n_periods, n_assets)`` panel and ``exposures`` a panel
    of cross-sectional characteristics with asset labels matching
    ``returns.columns``. A single ``(n_assets,)`` exposure row is treated as a
    static cross-section and broadcast across all return dates. For a panel,
    unavailable exposure dates become missing rows and are skipped; values are
    always reindexed by *asset label*, never by input column position.

    For each usable period a cross-sectional regression ``R_i = alpha + beta
    * X_i`` is estimated. The reported coefficient is the time-series mean of
    those slopes with an i.i.d. time-series standard error by default. Set
    ``covariance="newey-west"`` to report Bartlett-kernel HAC standard errors
    over the chronologically ordered fitted cross-sections. ``newey_west_lags``
    is explicit and must be smaller than the number of fitted cross-sections;
    the returned DataFrame records both choices in ``.attrs``. This routine
    does not provide clustered standard errors.
    """
    if not isinstance(returns, pd.DataFrame) or not isinstance(exposures, pd.DataFrame):
        raise TypeError("returns and exposures must be pandas DataFrames")
    if covariance not in ("iid", "newey-west"):
        raise ValueError("covariance must be 'iid' or 'newey-west'")
    if not isinstance(newey_west_lags, int) or isinstance(newey_west_lags, bool) or newey_west_lags < 0:
        raise ValueError("newey_west_lags must be a non-negative integer")
    if returns.empty or returns.shape[1] < 2:
        raise ValueError("returns must contain at least two asset columns")
    if returns.index.has_duplicates or exposures.index.has_duplicates:
        raise ValueError("returns and exposures indices must not contain duplicates")
    if returns.columns.has_duplicates or exposures.columns.has_duplicates:
        raise ValueError("returns and exposures columns must not contain duplicates")
    if covariance == "newey-west" and not returns.index.is_monotonic_increasing:
        raise ValueError("newey-west covariance requires chronological returns index")

    missing_assets = returns.columns.difference(exposures.columns)
    if len(missing_assets):
        raise ValueError(f"exposures are missing return asset columns: {list(missing_assets)!r}")

    if len(exposures) == 1:
        static = exposures.iloc[0].reindex(returns.columns)
        aligned_exposures = pd.DataFrame(
            np.tile(static.to_numpy(dtype=float), (len(returns), 1)),
            index=returns.index,
            columns=returns.columns,
        )
    else:
        aligned_exposures = exposures.reindex(index=returns.index, columns=returns.columns)

    estimates: list[float] = []
    alphas: list[float] = []
    for t in returns.index:
        y = returns.loc[t].to_numpy(dtype=float)
        x = aligned_exposures.loc[t].to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 2 or np.std(x[mask]) < 1e-15:
            continue
        design = np.column_stack((np.ones(mask.sum()), x[mask]))
        coefficients, _, rank, _ = np.linalg.lstsq(design, y[mask], rcond=None)
        if rank < 2:  # pragma: no cover - protected by the variance guard
            continue
        alphas.append(float(coefficients[0]))
        estimates.append(float(coefficients[1]))

    if covariance == "newey-west" and newey_west_lags >= len(estimates):
        raise ValueError("newey_west_lags must be smaller than the fitted cross-section count")

    if not estimates:
        empty = pd.DataFrame(columns=["mean", "std_error", "t_stat"], index=["intercept", "exposure"])
        empty.attrs["covariance"] = covariance
        empty.attrs["newey_west_lags"] = newey_west_lags if covariance == "newey-west" else None
        empty.attrs["n_cross_sections"] = 0
        return empty

    slopes = np.asarray(estimates, dtype=float)
    intercepts = np.asarray(alphas, dtype=float)
    mean_slope = float(np.mean(slopes))
    mean_intercept = float(np.mean(intercepts))
    if covariance == "newey-west":
        se_slope = _newey_west_mean_standard_error(slopes, newey_west_lags)
        se_intercept = _newey_west_mean_standard_error(intercepts, newey_west_lags)
    else:
        se_slope = float(np.std(slopes, ddof=1) / np.sqrt(len(slopes))) if len(slopes) > 1 else float("nan")
        se_intercept = (
            float(np.std(intercepts, ddof=1) / np.sqrt(len(intercepts))) if len(intercepts) > 1 else float("nan")
        )

    def t_stat(mean: float, se: float) -> float:
        if not np.isfinite(se):
            return float("nan")
        if se > 1e-15:
            return mean / se
        if mean > 0.0:
            return float("inf")
        if mean < 0.0:
            return float("-inf")
        return 0.0

    result = pd.DataFrame(
        {
            "mean": [mean_intercept, mean_slope],
            "std_error": [se_intercept, se_slope],
            "t_stat": [t_stat(mean_intercept, se_intercept), t_stat(mean_slope, se_slope)],
        },
        index=["intercept", "exposure"],
    )
    result.attrs["covariance"] = covariance
    result.attrs["newey_west_lags"] = newey_west_lags if covariance == "newey-west" else None
    result.attrs["n_cross_sections"] = len(slopes)
    return result


def _newey_west_mean_standard_error(values: np.ndarray, nlags: int) -> float:
    """Return an uncorrected Bartlett Newey-West standard error for a mean.

    This is the intercept-only equivalent of
    ``statsmodels.OLS(values, ones).fit(cov_type="HAC")`` with its default
    uncorrected covariance. The separate implementation keeps the enhanced
    kernel free of a runtime statsmodels import while its numerical tests use
    statsmodels as the independent oracle.
    """

    observations = np.asarray(values, dtype=float)
    if observations.ndim != 1 or len(observations) < 2:  # pragma: no cover - public caller guards this
        return float("nan")
    if nlags < 0 or nlags >= len(observations):  # pragma: no cover - public caller guards this
        raise ValueError("newey_west_lags must be smaller than the fitted cross-section count")
    centered = observations - np.mean(observations)
    long_run_variance = float(centered @ centered)
    for lag in range(1, nlags + 1):
        bartlett_weight = 1.0 - lag / (nlags + 1.0)
        long_run_variance += float(2.0 * bartlett_weight * (centered[lag:] @ centered[:-lag]))
    variance_of_mean = long_run_variance / len(observations) ** 2
    if variance_of_mean < 0.0:
        if np.isclose(variance_of_mean, 0.0, rtol=0.0, atol=float(np.finfo(float).eps)):
            variance_of_mean = 0.0
        else:  # pragma: no cover - Bartlett HAC is positive semidefinite; defensive numerical guard
            raise ValueError("newey-west variance must be non-negative")
    return float(np.sqrt(variance_of_mean))
