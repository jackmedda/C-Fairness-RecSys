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

import os

from numpy.testing import (
    TestCase,
    run_module_suite,
    assert_allclose)
import numpy as np

from kamrecsys.datasets import load_movielens_mini

# =============================================================================
# Variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Test Classes
# =============================================================================


class TestIndependentScorePredictor(TestCase):

    @staticmethod
    def load_data():
        from kamrecsys.datasets import load_event_with_score
        from kamiers.datasets import event_dtype_sensitive_and_timestamp

        infile = os.path.join(os.path.dirname(__file__), 'mlmini_t.event')
        data = load_event_with_score(
            infile, score_domain=(1, 5, 1),
            event_dtype=event_dtype_sensitive_and_timestamp)
        sen = data.event_feature['sensitive']
        return data, sen

    def test_single_mode(self):
        from kamrecsys.score_predictor import PMF
        from kamiers.sp_generic.standard import IndependentScorePredictor

        data = load_movielens_mini()

        base_esitormator = PMF(C=0.1, k=2)
        rec = IndependentScorePredictor(
            base_esitormator, multi_mode=False, random_state=1234)
        sen = np.r_[np.zeros(15, dtype=int), np.ones(15, dtype=int)]
        rec.fit(data, sen)

        assert_allclose(
            rec.fit_results_['initial_loss'], 8.543466258979379, rtol=1e-5)
        assert_allclose(
            rec.fit_results_['final_loss'], 0.683785876513324, rtol=1e-3)

        # single prediction
        assert_allclose(rec.predict((1, 7), 0), 3.99008787679166, rtol=1e-5)
        assert_allclose(rec.predict((1, 9), 0), 4.96434648861651, rtol=1e-5)
        assert_allclose(rec.predict((1, 11), 0), 3.64758345453257, rtol=1e-5)
        assert_allclose(rec.predict((3, 7), 1), 3.56014414944393, rtol=1e-5)
        assert_allclose(rec.predict((3, 9), 1), 4.25581462889725, rtol=1e-5)
        assert_allclose(rec.predict((3, 11), 1), 3.75170345345246, rtol=1e-5)
        assert_allclose(rec.predict((5, 7), 1), 3.26738103418439, rtol=1e-5)
        assert_allclose(rec.predict((5, 9), 1), 3.9921637922356, rtol=1e-5)
        assert_allclose(rec.predict((5, 11), 1), 3.45201827055917, rtol=1e-5)

        # multiple prediction
        x = np.array([
            [1, 7], [1, 9], [1, 11],
            [3, 7], [3, 9], [3, 11],
            [5, 7], [5, 9], [5, 11]])
        sen = [0, 0, 0, 1, 1, 1, 0, 0, 0]

        assert_allclose(
            rec.predict(x, sen),
            [3.9900878768, 4.9643464886, 3.6475834545,
             3.5601441494, 4.2558146289, 3.7517034535,
             3.2673810342, 3.9921637922, 3.4520182706],
            rtol=1e-5)

    def test_multi_mode(self):
        from kamrecsys.score_predictor import PMF
        from kamiers.sp_generic.standard import IndependentScorePredictor

        data, sen = self.load_data()

        base_estimator = PMF(C=0.01, k=2, tol=1e-03, maxiter=200)
        rec = IndependentScorePredictor(
            base_estimator, multi_mode=True, random_state=1234)
        rec.fit(data, sen)

        assert_allclose(
            rec.fit_results_['fit_results'][0]['initial_loss'],
            1.73031647683995, rtol=1e-5)
        assert_allclose(
            rec.fit_results_['fit_results'][1]['initial_loss'],
            14.3319776844221, rtol=1e-5)
        assert_allclose(
            rec.fit_results_['fit_results'][0]['final_loss'],
            0.0244318486610081, rtol=1e-3)
        assert_allclose(
            rec.fit_results_['fit_results'][1]['final_loss'],
            0.056865364252381, rtol=1e-3)

        # prediction
        assert_allclose(
            rec.raw_predict(np.array([[2, 1]]), np.array([0])),
            3.00463343162446, rtol=1e-3)
        assert_allclose(
            rec.raw_predict(
                np.array([[6, 0], [2, 0], [6, 3]]), np.array([1, 0, 0])),
            [3.997133999991, 3.999303316453, 3.999705758683], rtol=1e-3)

        assert_allclose(rec.predict((5, 2), 0), 3.00463343162446, rtol=1e-3)
        assert_allclose(
            rec.predict([[10, 7], [5, 1], [10, 4]], [1, 0, 0]),
            [3.997133999991, 3.999303316453, 3.999705758683],
            rtol=1e-3)


# =============================================================================
# Main Routine
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
