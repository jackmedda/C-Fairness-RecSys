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

# =============================================================================
# Variables
# =============================================================================

y_true = [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1]
y_pred = [
    3.57693181, 3.74035694, 4.05667626, 4.11208204, 3.90311352,
    3.45602985, 3.45850967, 3.62878891, 3.44663680, 3.90525169,
    3.83583531, 3.57535821, 3.36962744, 3.88604703, 3.64760914,
    4.28210103, 2.36221703, 3.57729091, 3.37095081, 3.80150211]
sen = [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1]

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Test Classes
# =============================================================================


def test_item_finder_and_single_categorical_report():
    from kamiers.metrics import item_finder_and_single_categorical_report

    stats = item_finder_and_single_categorical_report(
        y_true, y_pred, sen, disp=False)

    assert_allclose(stats['KS statistic'], 0.7142857142857143, rtol=1e-5)
    assert_allclose(stats['Gaussian NMI'], 0.20976290212377685, rtol=1e-5)


def test_item_finder_and_single_categorical_statistics():
    from kamiers.metrics import item_finder_and_single_categorical_statistics

    stats = item_finder_and_single_categorical_statistics(y_true, y_pred, sen)

    sub_stats = stats['KS statistic']
    assert_allclose(sub_stats['p'], 0.012910336137009088, rtol=1e-5)
    assert_allclose(sub_stats['statistic'], 0.7142857142857143, rtol=1e-5)

    sub_stats = stats['Gaussian NMI']
    assert_allclose(sub_stats['mi'], 0.20976290212377685, rtol=1e-5)
    assert_allclose(sub_stats['mi / ent_S'], 0.3433870688107866, rtol=1e-5)

    assert_equal(stats['sensitive'][0]['n_samples'], 6)
    assert_equal(stats['sensitive'][1]['n_samples'], 14)

    assert_('area_under_the_curve' not in stats['sensitive'][0])
    assert_allclose(
        stats['sensitive'][1]['area under the curve'],
        0.77083333333333337, rtol=1e-5)


# =============================================================================
# Main Routine
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
