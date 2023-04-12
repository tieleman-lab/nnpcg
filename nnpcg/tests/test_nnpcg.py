"""
Unit and regression test for the nnpcg package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import nnpcg


def test_nnpcg_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "nnpcg" in sys.modules
