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
    assert_,
    assert_allclose,
    assert_array_equal,
    assert_equal)

from kamrecsys.datasets import load_movielens_mini
from kamrecsys.score_predictor.topic_model import MultinomialPLSA

# =============================================================================
# Module variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Test Classes
# =============================================================================


class TestMultinomialPLSA(TestCase):

    def test_class(self):
        import numpy as np

        data = load_movielens_mini()

        rec = MultinomialPLSA(k=2, random_state=1234, tol=1e-8)

        # import logging
        # logging.getLogger('kamrecsys').addHandler(logging.StreamHandler())
        rec.fit(data)

        assert_equal(rec.fit_results_['n_users'], 8)
        assert_equal(rec.fit_results_['n_items'], 10)
        assert_equal(rec.fit_results_['n_events'], 30)
        assert_(rec.fit_results_['success'])
        assert_equal(rec.fit_results_['status'], 0)
        assert_allclose(
            rec.fit_results_['initial_loss'], 5.41836900049, rtol=1e-5)
        assert_allclose(
            rec.fit_results_['final_loss'], 5.17361298499, rtol=1e-5)
        assert_equal(rec.fit_results_['n_iterations'], 38)
        assert_allclose(rec.score_levels, [1, 2, 3, 4, 5], rtol=1e-5)

        # output size
        assert_array_equal(
            rec.raw_predict(data.event[0].reshape(1, 2)).shape, (1,))
        assert_array_equal(rec.raw_predict(data.event).shape, (30,))
        rec.use_expectation = False
        assert_array_equal(
            rec.raw_predict(data.event[0].reshape(1, 2)).shape, (1,))
        assert_array_equal(rec.raw_predict(data.event).shape, (30,))
        rec.use_expectation = True

        # known user and item
        assert_allclose(
            rec.raw_predict(np.array([[0, 6]])), 3.64580117249, rtol=1e-5)
        assert_allclose(
            rec.raw_predict(np.array([[2, 8]])), 3.62184516985, rtol=1e-5)
        assert_allclose(rec.predict((1, 7)), 3.64580117249, rtol=1e-5)
        assert_allclose(rec.predict((1, 9)), 3.6587422493, rtol=1e-5)
        assert_allclose(rec.predict((5, 7)), 3.60707987724, rtol=1e-5)
        assert_allclose(rec.predict((5, 9)), 3.62184516985, rtol=1e-5)

        # known user and unknown item
        assert_allclose(
            rec.raw_predict(np.array([[2, 10]])), 3.62387542269, rtol=1e-5)
        assert_allclose(rec.predict((1, 11)), 3.66032199689, rtol=1e-5)
        assert_allclose(rec.predict((5, 12)), 3.62387542269, rtol=1e-5)

        # unknown user and known item
        assert_allclose(
            rec.raw_predict(np.array([[8, 6]])), 3.60821491793, rtol=1e-5)
        assert_allclose(rec.predict((3, 7)), 3.60821491793, rtol=1e-5)
        assert_allclose(rec.predict((11, 9)), 3.62304301551, rtol=1e-5)

        # unknown user and item
        assert_allclose(
            rec.raw_predict(np.array([[8, 10]])), 3.62507437787, rtol=1e-5)
        assert_allclose(rec.predict((3, 11)), 3.62507437787, rtol=1e-5)

        x = np.array([
            [1, 7], [1, 9], [1, 11],
            [3, 7], [3, 9], [3, 11],
            [5, 7], [5, 9], [5, 11]])
        assert_allclose(
            rec.predict(x),
            [3.64580117249, 3.6587422493, 3.66032199689,
             3.60821491793, 3.62304301551, 3.62507437787,
             3.60707987724, 3.62184516985, 3.62387542269],
            rtol=1e-5)

        rec.use_expectation = False
        assert_allclose(
            rec.predict(x),
            [4., 5., 5., 4., 4., 4., 4., 4., 4.],
            rtol=1e-5)


# =============================================================================
# Main Routines
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
