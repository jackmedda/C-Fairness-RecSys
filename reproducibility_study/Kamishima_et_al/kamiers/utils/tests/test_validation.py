#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    print_function,
    division,
    absolute_import,
    unicode_literals)
from six.moves import xrange

# =============================================================================
# Imports
# =============================================================================

from numpy.testing import (
    TestCase,
    run_module_suite,
    assert_,
    assert_allclose,
    assert_array_almost_equal_nulp,
    assert_array_max_ulp,
    assert_array_equal,
    assert_array_less,
    assert_equal,
    assert_raises,
    assert_raises_regex,
    assert_warns,
    assert_string_equal)

import numpy as np
import scipy.sparse as sparse

from kamiers.utils import check_sensitive

# =============================================================================
# Variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Test Classes
# =============================================================================


class TestCheckSensitive(TestCase):

    def test_func(self):

        # dense matrix ###

        # 1-sensitive value
        a = np.ones(3, dtype=int)
        b = check_sensitive(3, a, dtype=int)
        assert_(b.dtype == int)
        assert_array_equal(b,  [1, 1, 1])
        a = np.ones((3, 1), dtype=int)
        b = check_sensitive(3, a, dtype=int)
        assert_(b.dtype == int)
        assert_array_equal(b,  [1, 1, 1])
        assert_(check_sensitive(3, a, dtype=np.float32).dtype == np.float32)
        assert_(check_sensitive(3, a, dtype=complex).dtype == complex)

        # 2-sensitive value2
        a = np.ones((3, 2), dtype=int)
        b = check_sensitive(3, a, dtype=int)
        assert_(b.dtype == int)
        assert_array_equal(b,  [[1, 1], [1, 1], [1, 1]])
        assert_(check_sensitive(3, a, dtype=np.float32).dtype == np.float32)
        assert_(check_sensitive(3, a, dtype=complex).dtype == complex)

        # dtype=binary
        a = np.ones(3, dtype=float)
        b = check_sensitive(3, a, dtype='binary')
        assert_(b.dtype == int)
        assert_array_equal(b, [1, 1, 1])
        a = np.zeros(3, dtype=float)
        b = check_sensitive(3, a, dtype='binary')
        assert_(b.dtype == int)
        assert_array_equal(b, [0, 0, 0])
        a = np.array([0, 1, 2])
        with assert_raises(ValueError):
            check_sensitive(3, a, dtype='binary')

        # dtype=numeric
        a = np.ones(3, dtype=int)
        assert_(check_sensitive(3, a, dtype='numeric').dtype == int)
        a = np.ones(3, dtype=float)
        assert_(check_sensitive(3, a, dtype='numeric').dtype == float)
        a = np.ones(3, dtype=complex)
        assert_(check_sensitive(3, a, dtype='numeric').dtype == complex)
        a = np.ones(3, dtype=np.uint)
        assert_(check_sensitive(3, a, dtype='numeric').dtype == np.uint)
        a = np.ones(3, dtype=np.float32)
        assert_(check_sensitive(3, a, dtype='numeric').dtype == np.float32)
        a = np.array(['a', 'b', 'c'])
        with assert_raises(ValueError):
            check_sensitive(3, a, dtype='numeric')
        a = np.array([True, False, True])
        assert_(check_sensitive(3, a, dtype='numeric').dtype == float)

        # sparse matrix ###

        # 2-sensitive values
        a = sparse.coo_matrix((3, 2), dtype=int)
        b = check_sensitive(3, a, dtype=None)
        assert_(sparse.issparse(b))
        assert_(b.dtype == int)
        assert_(check_sensitive(3, a, dtype=int).dtype == int)
        assert_(check_sensitive(3, a, dtype=float).dtype == float)
        assert_(check_sensitive(3, a, dtype=np.float32).dtype == np.float32)
        assert_(check_sensitive(3, a, dtype='numeric').dtype == int)

        # dtype=non-numeric
        a = sparse.coo_matrix((3, 2), dtype=np.dtype('S1'))
        b = check_sensitive(3, a, dtype=None)
        assert_(b.dtype == np.dtype('S1'))
        assert_(check_sensitive(3, a, dtype='numeric').dtype == float)

        # dtype=binary
        a = sparse.lil_matrix((3, 1), dtype=float)
        b = check_sensitive(3, a, accept_sparse=True)
        assert_(not sparse.issparse(b))
        a = sparse.lil_matrix((3, 2), dtype=float)
        b = check_sensitive(3, a, accept_sparse=True, dtype=float)
        assert_(sparse.issparse(b))
        b = check_sensitive(3, a, accept_sparse=True, dtype='binary')
        assert_array_equal(b, [[0, 0], [0, 0], [0, 0]])
        a[0, 0] = 2
        with assert_raises(ValueError):
            check_sensitive(3, a, accept_sparse=True, dtype='binary')
        a[0, 0] = 1
        b = check_sensitive(3, a, accept_sparse=True, dtype='binary')
        assert_(not sparse.issparse(b))

        # array-size consistency ###

        X = np.ones((3, 2), dtype=int)
        a = np.ones(3, dtype=int)
        assert_array_equal(check_sensitive(X, a), [1, 1, 1])
        a = np.ones(4, dtype=int)
        with assert_raises(ValueError):
            check_sensitive(X, a)


# =============================================================================
# Main Routine
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
