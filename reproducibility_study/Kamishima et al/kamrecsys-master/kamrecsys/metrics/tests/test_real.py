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
    TestCase,
    run_module_suite,
    assert_allclose,
    assert_array_equal)
import numpy as np

# =============================================================================
# Variables
# =============================================================================

y_true = [5.0, 5.0, 5.0, 5.0, 4.0, 3.0, 5.0, 2.0, 4.0, 3.0]
y_pred = [3.96063305016, 3.16580296689, 4.17585047905, 4.08648849520,
          4.11381603218, 3.45056765134, 4.31221525136, 4.08790965172,
          4.01993828853, 4.56297459028]

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Test Classes
# =============================================================================


class TestMeanAbsoluteError(TestCase):

    def test_class(self):
        from kamrecsys.metrics import mean_absolute_error

        mean, stdev = mean_absolute_error(y_true, y_pred)
        assert_allclose(mean, 0.9534215971390001, rtol=1e-5)
        assert_allclose(stdev, 0.6602899115612394, rtol=1e-5)


class TestMeanSquaredError(TestCase):

    def test_class(self):
        from kamrecsys.metrics import mean_squared_error

        rmse, mean, stdev = mean_squared_error(y_true, y_pred)

        assert_allclose(rmse, 1.1597394143516166, rtol=1e-5)
        assert_allclose(mean, 1.3449955092006309, rtol=1e-5)
        assert_allclose(stdev, 1.4418716080648177, rtol=1e-5)


class TestScoreHistogram(TestCase):

    def test_class(self):
        from kamrecsys.metrics import score_histogram

        hist, scores = score_histogram(y_pred)
        assert_array_equal(hist, [0, 0, 2, 7, 1])
        assert_array_equal(scores, [1, 2, 3, 4, 5])

        hist, scores = score_histogram(y_pred, score_domain=(3, 5, 2))
        assert_array_equal(hist, [3, 7])
        assert_array_equal(scores, [3, 5])

        hist, scores = score_histogram(
            np.linspace(0.0, 1.0, 21), score_domain=(0.2, 0.4, 0.2))
        assert_array_equal(hist, [6, 15])
        assert_array_equal(scores, [0.2, 0.4])


class TestVarianceWithGammaPrior(TestCase):

    def test_func(self):
        from kamrecsys.metrics import variance_with_gamma_prior as safe_var

        assert_allclose(safe_var(y_true), 1.08999999782)
        assert_allclose(safe_var([1], a=1, b=3), 2)
        assert_allclose(safe_var([1], a=1, b=3), 2)
        assert_allclose(safe_var([np.nan, 1, 2]), 0.2499999975)
        with self.assertRaises(ValueError):
            safe_var([np.nan, 1, 2], force_all_finite=True)
        with self.assertRaises(ValueError):
            safe_var([np.inf])
        with self.assertRaises(ValueError):
            safe_var([])
        with self.assertRaises(ValueError):
            safe_var([], force_all_finite=True)
        assert_allclose(
            safe_var([-np.inf, 3.0, 5.0, np.nan, 2.0, 4.0, 3.0, np.inf],
                     full_output=True),
            [1.03999999584, 5])


# =============================================================================
# Main Routine
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
