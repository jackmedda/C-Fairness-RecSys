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
        from kamiers.sp_generic.post_linear import IndependentScorePredictor

        data = load_movielens_mini()
        sen = np.r_[np.zeros(15, dtype=int), np.ones(15, dtype=int)]
        base_esitormator = PMF(C=0.1, k=2)

        rec = IndependentScorePredictor(
            base_esitormator, multi_mode=False, random_state=1234)
        rec.fit(data, sen)

        assert_allclose(rec.t_mean_, 3.83333316291949)
        assert_allclose(rec.t_std_, 0.973380806234203)
        assert_allclose(rec.g_mean_, [4.053428771205, 3.613237554634])
        assert_allclose(rec.g_std_, [0.752327885648, 1.109981525429])

        rec = IndependentScorePredictor(
            base_esitormator, multi_mode=False, use_predicted=False,
            random_state=1234)
        rec.fit(data, sen)

        assert_allclose(rec.t_mean_, 3.83333333333333)
        assert_allclose(rec.t_std_, 1.00277393043275)
        assert_allclose(rec.g_mean_, [4.066666666667, 3.6])
        assert_allclose(rec.g_std_, [0.771722460186, 1.143095213299])

    def test_multi_mode(self):
        from kamrecsys.score_predictor import PMF
        from kamiers.sp_generic.post_linear import IndependentScorePredictor

        data, sen = self.load_data()
        base_estimator = PMF(C=0.01, k=2, tol=1e-03, maxiter=200)

        # use_predicted == True
        rec = IndependentScorePredictor(
            base_estimator, multi_mode=True, random_state=1234)
        rec.fit(data, sen)

        assert_allclose(rec.t_mean_, 3.83330746497265)
        assert_allclose(rec.t_std_, 0.999259339009276)
        assert_allclose(rec.g_mean_, [3.999958089908, 3.687488168154])
        assert_allclose(rec.g_std_, [0.75265101673, 1.153683794088])

        # prediction
        assert_allclose(
            rec.raw_predict(np.array([[2, 1]]), np.array([0])),
            2.511861751051512, rtol=1e-3)
        assert_allclose(
            rec.raw_predict(
                np.array([[6, 0], [2, 0], [6, 3]]), np.array([1, 0, 0])),
            [4.101506161039, 3.832438153067, 3.832972456679], rtol=1e-3)

        assert_allclose(rec.predict((5, 2), 0), 2.51186175105151, rtol=1e-3)
        assert_allclose(
            rec.predict([[10, 7], [5, 1], [10, 4]], [1, 0, 0]),
            [4.101506161039, 3.832438153067, 3.832972456679],
            rtol=1e-3)

        rec = IndependentScorePredictor(
            base_estimator, multi_mode=True, use_predicted=False,
            random_state=1234)
        rec.fit(data, sen)

        assert_allclose(rec.t_mean_, 3.83333333333333)
        assert_allclose(rec.t_std_, 1.00277393043275)
        assert_allclose(rec.g_mean_, [4., 3.6875])
        assert_allclose(rec.g_std_, [0.755928946018, 1.157516198591])


# =============================================================================
# Main Routine
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
