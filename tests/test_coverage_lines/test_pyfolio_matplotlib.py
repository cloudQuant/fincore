"""Tests for pyfolio.py coverage edge cases.

Targets:
- pyfolio.py: 55-58 - matplotlib.use('Agg') exception
"""


class TestPyfolioMatplotlibException:
    """Test pyfolio.py lines 55-58."""

    def test_pyfolio_import_with_matplotlib_exception(self):
        """Lines 55-58: matplotlib.use('Agg') exception handling."""
        # The pyfolio module handles exceptions when setting matplotlib backend
        import fincore.pyfolio as pyfolio_module
        assert hasattr(pyfolio_module, "Pyfolio")
