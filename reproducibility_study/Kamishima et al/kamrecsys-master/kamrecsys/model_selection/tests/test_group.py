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
    assert_array_equal,
    assert_equal,
    assert_raises)
import numpy as np

# =============================================================================
# Variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Test Classes
# =============================================================================


def test_ShuffleSplitWithinGroups():

    from kamrecsys.model_selection import ShuffleSplitWithinGroups

    groups = np.array([1, 0, 1, 1, 3, 1, 3, 0, 3, 3, 0, 1, 3])

    # error handling
    with assert_raises(ValueError):
        cv = ShuffleSplitWithinGroups(n_splits=3)
        next(cv.split(np.arange(10), groups=groups))

    with assert_raises(ValueError):
        cv = ShuffleSplitWithinGroups(n_splits=1, test_size=3)
        next(cv.split(np.arange(13), groups=groups))

    # function
    cv = ShuffleSplitWithinGroups(n_splits=1, random_state=1234)
    train_i, test_i = next(cv.split(
        np.arange(20), groups=np.r_[np.zeros(10), np.ones(10)]))
    assert_array_equal(
        train_i,
        [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19])
    assert_array_equal(test_i, [7, 17])

    cv = ShuffleSplitWithinGroups(n_splits=1, test_size=0.3, random_state=1234)
    train_i, test_i = next(cv.split(
        np.arange(20), groups=np.r_[np.zeros(10), np.ones(10)]))
    assert_array_equal(
        train_i, [0, 1, 3, 4, 5, 6, 8, 10, 11, 12, 14, 16, 18, 19])
    assert_array_equal(test_i, [2, 7, 9, 13, 15, 17])

    cv = ShuffleSplitWithinGroups(n_splits=1, test_size=2, random_state=1234)
    train_i, test_i = next(cv.split(
        np.arange(20), groups=np.r_[np.zeros(10), np.ones(10)]))
    assert_array_equal(
        train_i, [0, 1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 18, 19])
    assert_array_equal(test_i, [2, 7, 13, 17])

    cv = ShuffleSplitWithinGroups(
        n_splits=1, test_size=None, train_size=4, random_state=1234)
    train_i, test_i = next(cv.split(
        np.arange(20), groups=np.r_[np.zeros(10), np.ones(10)]))
    assert_array_equal(train_i, [3, 4, 5, 6, 10, 12, 16, 19])
    assert_array_equal(test_i, [0, 1, 2, 7, 8, 9, 11, 13, 14, 15, 17, 18])

    cv = ShuffleSplitWithinGroups(
        n_splits=1, test_size=None, train_size=0.6, random_state=1234)
    train_i, test_i = next(cv.split(
        np.arange(20), groups=np.r_[np.zeros(10), np.ones(10)]))
    assert_array_equal(train_i, [0, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 19])
    assert_array_equal(test_i, [1, 2, 7, 9, 11, 13, 15, 17])

    cv = ShuffleSplitWithinGroups(n_splits=2, test_size=0.3, random_state=1234)
    iter = cv.split(np.arange(13), groups=groups)
    train_i, test_i = next(iter)
    assert_array_equal(train_i, [0, 5, 6, 7, 9, 10, 11, 12])
    assert_array_equal(test_i, [1, 2, 3, 4, 8])
    train_i, test_i = next(iter)
    assert_array_equal(train_i, [0, 3, 4, 7, 8, 9, 10, 11])
    assert_array_equal(test_i, [1, 2, 5, 6, 12])
    with assert_raises(StopIteration):
        next(iter)


def test_GroupWiseKfold():

    from kamrecsys.model_selection import KFoldWithinGroups

    groups = np.array([1, 0, 1, 1, 3, 1, 3, 0, 3, 3, 0, 1, 3])

    # error handling
    with assert_raises(ValueError):
        KFoldWithinGroups(n_splits=1)

    with assert_raises(ValueError):
        cv = KFoldWithinGroups(n_splits=3)
        next(cv.split(np.arange(10), groups=np.zeros(9)))

    with assert_raises(ValueError):
        cv = KFoldWithinGroups(n_splits=4)
        next(cv.split(np.arange(13), groups=groups))

    # function
    test_fold = np.zeros(13, dtype=np.int)
    cv = KFoldWithinGroups(3)
    for i, g in enumerate(cv.split(np.arange(13), groups=groups)):
        test_fold[g[1]] = i
    assert_array_equal(
        test_fold, [0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 2, 2, 2])

    test_fold = np.zeros(13, dtype=np.int)
    cv = KFoldWithinGroups(5)
    for i, g in enumerate(cv.split(np.arange(13), groups=None)):
        test_fold[g[1]] = i
    assert_array_equal(
        test_fold, [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4])

    test_fold = np.zeros(13, dtype=np.int)
    cv = KFoldWithinGroups(3, shuffle=True, random_state=1234)
    for i, g in enumerate(cv.split(np.arange(13), groups=groups)):
        test_fold[g[1]] = i
    assert_array_equal(
        test_fold, [1, 0, 0, 0, 0, 1, 2, 1, 0, 1, 2, 2, 1])

    test_fold = np.zeros(13, dtype=np.int)
    cv = KFoldWithinGroups(5, shuffle=True, random_state=1234)
    for i, g in enumerate(cv.split(np.arange(13), groups=None)):
        test_fold[g[1]] = i
    assert_array_equal(
        test_fold, [0, 2, 1, 4, 3, 3, 4, 2, 2, 1, 0, 1, 0])


def test_InterlacedKFold():

    from kamrecsys.model_selection import InterlacedKFold

    with assert_raises(ValueError):
        InterlacedKFold(n_splits=1)

    with assert_raises(ValueError):
        cv = InterlacedKFold(n_splits=2)
        cv.split(np.zeros(1))

    with assert_raises(ValueError):
        cv = InterlacedKFold(n_splits=2)
        cv.split(np.zeros(3), np.zeros(2))

    test_fold = np.empty(7, dtype=int)
    cv = InterlacedKFold(n_splits=3)
    iter = cv.split(np.zeros(7))
    train_i, test_i = next(iter)
    assert_array_equal(train_i, [1, 2, 4, 5])
    assert_array_equal(test_i, [0, 3, 6])
    test_fold[test_i] = 0
    train_i, test_i = next(iter)
    test_fold[test_i] = 1
    train_i, test_i = next(iter)
    test_fold[test_i] = 2
    assert_array_equal(test_fold, [0, 1, 2, 0, 1, 2, 0])
    with assert_raises(StopIteration):
        next(iter)

    cv = InterlacedKFold(n_splits=3)
    assert_equal(cv.get_n_splits(), 3)


# =============================================================================
# Main Routine
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
