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
    assert_,
    assert_allclose,
    assert_equal,
    assert_raises)
import numpy as np

# =============================================================================
# Variables
# =============================================================================

y_true = [1, 1, 1, 1, 1, 0, 1, 0, 1, 0]
y_pred = [3.96063305016, 3.16580296689, 4.17585047905, 4.08648849520,
          4.11381603218, 3.45056765134, 4.31221525136, 4.08790965172,
          4.01993828853, 4.56297459028]

# =============================================================================
# Functions
# =============================================================================


def test_item_finder_report():
    from kamrecsys.metrics import item_finder_report

    with assert_raises(ValueError):
        item_finder_report([2], [1])

    stats = item_finder_report(y_true, y_pred, disp=False)
    assert_allclose(stats['area under the curve'], 0.4285714285714286,
                    rtol=1e-5)

    assert_equal(stats['n_samples'], 10)
    assert_allclose(stats['true']['mean'], 0.7, rtol=1e-5)
    assert_allclose(stats['true']['stdev'], 0.45825756949558405, rtol=1e-5)
    assert_allclose(stats['predicted']['mean'], 3.99361964567, rtol=1e-5)
    assert_allclose(stats['predicted']['stdev'], 0.383771468193, rtol=1e-5)

    stats = item_finder_report(np.zeros(10), y_pred, disp=False)
    assert_('area_under_the_curve' not in stats)


def test_item_finder_statistics():
    from kamrecsys.metrics import item_finder_statistics

    with assert_raises(ValueError):
        item_finder_statistics([3], [1])

    stats = item_finder_statistics(y_true, y_pred)

    assert_equal(stats['n_samples'], 10)

    assert_allclose(stats['area under the curve'], 0.4285714285714286,
                    rtol=1e-5)

    assert_allclose(stats['true']['mean'], 0.7, rtol=1e-5)
    assert_allclose(stats['true']['stdev'], 0.45825756949558405, rtol=1e-5)
    assert_allclose(stats['predicted']['mean'], 3.99361964567, rtol=1e-5)
    assert_allclose(stats['predicted']['stdev'], 0.383771468193, rtol=1e-5)

    stats = item_finder_statistics(np.zeros(10), y_pred)
    assert_('area under the curve' not in stats)


# =============================================================================
# Test Classes
# =============================================================================

# =============================================================================
# Main Routine
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
