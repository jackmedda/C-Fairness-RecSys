#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    print_function,
    division,
    absolute_import)

# =============================================================================
# Imports
# =============================================================================

from numpy.testing import (
    TestCase,
    run_module_suite,
    assert_allclose)

import os
import numpy as np

from kamiers.sp_pmf import BaseIndependentPMF
from kamiers.sp_pmf.mi_histogram import IndependentScorePredictor

# =============================================================================
# Module variables
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

    def test_loss(self):

        # setup data
        data, sen = self.load_data()
        rec = IndependentScorePredictor(C=0.01, k=2, eta=100.0)
        BaseIndependentPMF.fit(rec, data, sen, event_index=(0, 1))
        sev, ssc, n_events = rec.get_sensitive_divided_data()
        rec._init_coef(data, sev, ssc, rec.n_objects, tol=1000.0, maxiter=200)
        score_bins = np.array([-np.inf, 1.5, 2.5, 3.5, 4.5, np.inf])

        # set array's view
        mu = rec._coef.view(rec._dt)['mu']
        bu = rec._coef.view(rec._dt)['bu']
        bi = rec._coef.view(rec._dt)['bi']
        p = rec._coef.view(rec._dt)['p']
        q = rec._coef.view(rec._dt)['q']

        # all zero
        mu[:] = 0.0
        bu[:] = 0.0
        bi[:] = 0.0
        p[:, :] = 0.0
        q[:, :] = 0.0
        assert_allclose(
            rec.loss(rec._coef, sev, ssc, rec.n_objects, score_bins),
            235.5)

        # all one
        mu[:] = 1.0
        bu[:] = 1.0
        bi[:] = 1.0
        p[:, :] = 1.0
        q[:, :] = 1.0
        assert_allclose(
            rec.loss(rec._coef, sev, ssc, rec.n_objects, score_bins),
            36.04)

        mu[0] = 1.0
        mu[1] = 1.5
        bu[0][:] = np.arange(0.0, 0.8, 0.1)
        bu[1][:] = np.arange(0.2, 1.0, 0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) + 1.0
        bi[1][:] = np.arange(0.0, 1.0, 0.1) + 2.0
        p[0][:, 0] = 0.5
        p[0][:, 1] = 1.0
        p[1][:, 0] = 1.0
        p[1][:, 1] = 1.0
        q[0][:, 0] = 0.2
        q[0][:, 1] = 1.0
        q[1][:, 0] = 1.0
        q[1][:, 1] = 1.0
        assert_allclose(
            rec.loss(rec._coef, sev, ssc, rec.n_objects, score_bins),
            159.389030931382)

    def test_class(self):

        # load data
        data, sen = self.load_data()

        # training
        rec = IndependentScorePredictor(
            C=0.01, k=2, eta=100, random_state=1234, tol=1e-03)
        rec.fit(data, sen)

        # check loss
        assert_allclose(
            rec.fit_results_['initial_loss'], 8.05163757812418, rtol=1e-5)
        assert_allclose(
            rec.fit_results_['final_loss'], 1.88970187046955, rtol=1e-3)

        # prediction
        assert_allclose(
            rec.raw_predict(data.event, sen)[:5],
            [3.010545007542, 3.997104661842, 4.007377074691, 4.004752370733,
             4.984031165437], rtol=1e-3)

        assert_allclose(
            rec.predict((5, 2), 0), 3.01054500754, rtol=1e-3)
        assert_allclose(
            rec.predict([[10, 7], [5, 1], [10, 4]], [1, 0, 0]),
            [3.997104661842, 4.007377074691, 4.004752370733])


# =============================================================================
# Main Routines
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
