"""Isolated runtime state and deterministic-operation cache ownership."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import TYPE_CHECKING, Any, Mapping

from .data import AnalysisSnapshot
from .engine import _mapping, _run_snapshot

if TYPE_CHECKING:
    from .catalog import OperationCatalog
    from .results import Result


@dataclass(frozen=True, slots=True)
class _CacheKey:
    """The stable identity of one cacheable operation execution."""

    operation_id: str
    input_digest: str
    config_digest: str
    catalog_digest: str


@dataclass(slots=True)
class AnalysisSession:
    """Own cache state for one fixed, immutable catalog snapshot.

    A session never observes later extension registration or catalog replacement.
    Only operations explicitly declared deterministic participate in its cache.
    """

    catalog: OperationCatalog
    _cache: dict[_CacheKey, Result] = field(default_factory=dict, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    @property
    def catalog_digest(self) -> str:
        """Return the catalog identity fixed when this session was constructed."""
        return self.catalog.digest

    @property
    def closed(self) -> bool:
        """Whether this session has released its owned cache."""
        return self._closed

    @property
    def cache_entries(self) -> int:
        """Return the number of deterministic results owned by this session."""
        with self._lock:
            return len(self._cache)

    def run(
        self,
        operation_id: str,
        inputs: Mapping[str, Any],
        config: Mapping[str, Any] | None = None,
    ) -> Result:
        """Run one catalog operation with session-local deterministic caching."""
        with self._lock:
            if self._closed:
                raise RuntimeError("analysis session is closed")

        spec = self.catalog.resolve(operation_id)
        snapshot = AnalysisSnapshot.from_inputs(_mapping(inputs, "inputs"))
        run_config = _mapping(config, "config") if config is not None else {}
        config_digest = AnalysisSnapshot.from_inputs({"config": run_config}).digest
        cache_key = _CacheKey(operation_id, snapshot.digest, config_digest, self.catalog_digest)

        if spec.deterministic:
            with self._lock:
                cached = self._cache.get(cache_key)
            if cached is not None:
                return cached.copy_for_consumer(cache="hit")

        result = _run_snapshot(
            spec,
            snapshot,
            run_config,
            catalog_digest=self.catalog_digest,
            cache="miss" if spec.deterministic else "disabled",
            config_digest=config_digest,
        )
        if spec.deterministic:
            try:
                cache_value = result.copy_for_consumer()
            except Exception:  # noqa: BLE001 - third-party domain values define their own copy protocol.
                return result.with_metadata(cache="disabled")
            with self._lock:
                if self._closed:
                    raise RuntimeError("analysis session is closed")
                existing = self._cache.setdefault(cache_key, cache_value)
            if existing is not cache_value:
                return existing.copy_for_consumer(cache="hit")
        return result

    def close(self) -> None:
        """Release cache ownership; safe to call repeatedly."""
        with self._lock:
            if self._closed:
                return
            self._cache.clear()
            self._closed = True

    def __enter__(self) -> AnalysisSession:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
