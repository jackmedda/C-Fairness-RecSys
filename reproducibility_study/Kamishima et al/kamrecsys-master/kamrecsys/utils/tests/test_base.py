#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    print_function,
    division,
    absolute_import,
    unicode_literals)

# =============================================================================
# Imports
# =============================================================================

from numpy.testing import (
    run_module_suite,
    assert_)
import numpy as np

# =============================================================================
# Variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================


def test_is_binary_score():
    from kamrecsys.utils import is_binary_score

    assert_(is_binary_score([0, 1, 1, 0, 1]))
    assert_(is_binary_score(np.identity(3), allow_uniform=True))
    assert_(is_binary_score([0, 0, 0]))
    assert_(is_binary_score([1], allow_uniform=True))

    assert_(is_binary_score([0, 1, 1, 0, 1], allow_uniform=False))
    assert_(is_binary_score(np.identity(3), allow_uniform=False))
    assert_(not is_binary_score([0, 0, 0], allow_uniform=False))
    assert_(not is_binary_score([1], allow_uniform=False))


# =============================================================================
# Test Classes
# =============================================================================

# =============================================================================
# Main Routine
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
