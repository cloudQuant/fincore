"""Test basic imports to debug CI issues for the refactored fincore package."""


def test_basic_imports():
    """Test that fincore and Pyfolio can be imported."""
    import fincore
    from fincore import Pyfolio

    assert hasattr(fincore, "__version__")
    assert Pyfolio is not None


def test_utils_import():
    """Test that utils/common_utils can be imported."""
    from fincore.utils import common_utils as utils

    assert hasattr(utils, "HAS_IPYTHON")


def test_ipython_optional():
    """Test that IPython-related helpers are optional but importable."""
    from fincore.utils import common_utils as utils

    # Whether or not IPython is available, import should work
    assert hasattr(utils, "display")
    assert hasattr(utils, "HTML")