"""Visualization backends for fincore.

This package provides a pluggable visualization layer decoupled from
the computation engine.  The existing ``tearsheets/`` module is left
untouched; this new ``viz/`` package offers a lighter, backend-agnostic
alternative accessible via :class:`~fincore.core.context.AnalysisContext`.
"""
from fincore.viz.base import VizBackend, get_backend

__all__ = ['VizBackend', 'get_backend']
