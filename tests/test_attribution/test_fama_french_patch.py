"""Tests for fincore.attribution.fama_french_patch."""

from __future__ import annotations

import pytest

from fincore.attribution.fama_french_patch import FamaFrenchModelPatch


def test_fama_french_model_patch_sets_factors_3factor():
    m = FamaFrenchModelPatch(model_type="3factor", risk_free_rate=0.01)
    assert m.model_type == "3factor"
    assert m.risk_free_rate == 0.01
    assert m.factors == ["MKT", "SMB", "HML"]


def test_fama_french_model_patch_sets_factors_5factor():
    m = FamaFrenchModelPatch(model_type="5factor")
    assert m.factors == ["MKT", "SMB", "HML", "RMW", "CMA"]


def test_fama_french_model_patch_sets_factors_4factor_mom():
    m = FamaFrenchModelPatch(model_type="4factor_mom")
    assert m.factors == ["MKT", "SMB", "HML", "MOM"]


def test_fama_french_model_patch_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown model_type"):
        FamaFrenchModelPatch(model_type="nope")
