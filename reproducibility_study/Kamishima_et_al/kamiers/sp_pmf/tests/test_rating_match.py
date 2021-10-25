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
    assert_allclose)

import os
import numpy as np

from kamiers.sp_pmf import BaseIndependentPMF
from kamiers.sp_pmf.rating_match import IndependentScorePredictor

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
            rec.loss(rec._coef, sev, ssc, rec.n_objects),
            235.5)

        # all one
        mu[:] = 1.0
        bu[:] = 1.0
        bi[:] = 1.0
        p[:, :] = 1.0
        q[:, :] = 1.0
        assert_allclose(
            rec.loss(rec._coef, sev, ssc, rec.n_objects),
            36.04)

        mu[0] = 1.0
        mu[1] = 1.5
        bu[0][:] = np.arange(0.0, 0.8, 0.1)
        bu[1][:] = np.arange(0.2, 1.0, 0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) + 1.0
        bi[1][:] = np.arange(0.0, 1.0, 0.1) + 2.0
        p[0][:, 0] = 0.5
        p[1][:, 1] = 1.0
        q[0][:, 0] = 0.2
        q[1][:, 1] = 1.0
        assert_allclose(
            rec.loss(rec._coef, sev, ssc, rec.n_objects),
            428.29670000000004)

        mu[0] = 2.0
        mu[1] = 3.5
        bu[0][:] = np.arange(0.8, 0.0, -0.1)
        bu[1][:] = np.arange(0.0, 0.8, 0.1) * 2
        bi[0][:] = np.arange(0.0, 1.0, 0.1) * 1.5 + 1.0
        bi[1][:] = np.arange(0.0, 1.0, 0.1) + 2.0
        p[0][:, 0] = np.arange(0.0, 0.8, 0.1) * 0.8 + 3
        p[1][:, 1] = np.arange(0.8, 0.0, -0.1) * 0.5 + 1
        q[0][:, 0] = np.arange(1.0, 0.0, -0.1) * 0.3 + 2
        q[0][:, 1] = np.arange(0.0, 1.0, 0.1)
        q[1][:, 0] = np.arange(1.0, 0.0, -0.1)
        q[1][:, 1] = np.arange(0.0, 1.0, 0.1) * 0.4 - 1
        assert_allclose(
            rec.loss(rec._coef, sev, ssc, rec.n_objects),
            2026.9978794533336)

    def test_grad_loss(self):

        # setup data
        data, sen = self.load_data()
        rec = IndependentScorePredictor(C=0.01, k=2, eta=100.0)
        BaseIndependentPMF.fit(rec, data, sen, event_index=(0, 1))
        sev, ssc, n_events = rec.get_sensitive_divided_data()
        rec._init_coef(data, sev, ssc, rec.n_objects, tol=1000.0, maxiter=200)

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
        grad = rec.grad_loss(rec._coef, sev, ssc, rec.n_objects)
        assert_allclose(grad[0], -56.0, rtol=1e-5)
        assert_allclose(
            grad[1:5],
            [-23., -4., -7., -4.],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [0., 0., 0., 0.],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [0., 0., 0., 0.],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [0., 0., 0., 0.],
            rtol=1e-5)

        # all one
        mu[:] = 1.0
        bu[:] = 1.0
        bi[:] = 1.0
        p[:, :] = 1.0
        q[:, :] = 1.0
        grad = rec.grad_loss(rec._coef, sev, ssc, rec.n_objects)
        assert_allclose(grad[0], 14.0, rtol=1e-5)
        assert_allclose(
            grad[1:5],
            [7.01, 1.01, 3.01, 1.01],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [0.01, 0.01, 0.01, 0.01], rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [7.01, 7.01, 1.01, 1.01],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [2.01, 2.01, 6.01, 6.01],
            rtol=1e-5)

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
        grad = rec.grad_loss(rec._coef, sev, ssc, rec.n_objects)
        assert_allclose(grad[0], -266.7, rtol=1e-5)
        assert_allclose(
            grad[1:5],
            [-89.566666666667, -18.132333333333, -17.631333333333,
             -35.263666666667], rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [-51.984, -25.983, -34.648666666667, -25.981],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [-17.908333333333, -89.556666666667, -3.621666666667,
             -18.123333333333], rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [44.076666666667, 44.076666666667, 37.31, 37.31],
            rtol=1e-5)

        mu[0] = 2.0
        mu[1] = 3.5
        bu[0][:] = np.arange(0.8, 0.0, -0.1)
        bu[1][:] = np.arange(0.0, 0.8, 0.1) * 2
        bi[0][:] = np.arange(0.0, 1.0, 0.1) * 1.5 + 1.0
        bi[1][:] = np.arange(0.0, 1.0, 0.1) + 2.0
        p[0][:, 0] = np.arange(0.0, 0.8, 0.1) * 0.8 + 3
        p[0][:, 1] = 1.0
        p[1][:, 0] = 1.0
        p[1][:, 1] = np.arange(0.8, 0.0, -0.1) * 0.5 + 1
        q[0][:, 0] = np.arange(1.0, 0.0, -0.1) * 0.3 + 2
        q[0][:, 1] = np.arange(0.0, 1.0, 0.1)
        q[1][:, 0] = np.arange(1.0, 0.0, -0.1)
        q[1][:, 1] = np.arange(0.0, 1.0, 0.1) * 0.4 - 1
        grad = rec.grad_loss(rec._coef, sev, ssc, rec.n_objects)
        assert_allclose(grad[0], 663.737333333, rtol=1e-5)
        assert_allclose(
            grad[1:5],
            [245.874666666667, 46.125666666667, 50.3612, 82.179666666667],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [110.465666666667, 59.361833333333, 77.886, 62.556833333333],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [533.990333333333, 105.12, 100.373973333333, 19.1092],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [-70.394, -87.3894, -53.560333333333, -70.724333333333],
            rtol=1e-5)

    def test_fit(self):

        data, sen = self.load_data()

        rec = IndependentScorePredictor(
            C=0.01, k=2, random_state=1234, eta=10.0, tol=1e-03)
        rec.fit(data, sen)

        assert_allclose(
            rec.fit_results_['initial_loss'], 5.12063458333318, rtol=1e-5)
        assert_allclose(
            rec.fit_results_['final_loss'], 0.1414860125325519, rtol=1e-3)


# =============================================================================
# Main Routines
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
