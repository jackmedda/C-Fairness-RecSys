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
from kamiers.sp_pmf.mean_match import IndependentScorePredictor

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
        p[0][:, 1] = 1.0
        p[1][:, 0] = 1.0
        p[1][:, 1] = 1.0
        q[0][:, 0] = 0.2
        q[0][:, 1] = 1.0
        q[1][:, 0] = 1.0
        q[1][:, 1] = 1.0
        assert_allclose(
            rec.loss(rec._coef, sev, ssc, rec.n_objects),
            609.464469451531)

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
            [0.01, 0.01, 0.01, 0.01],
            rtol=1e-5)
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
        assert_allclose(grad[0], -328.932142857, rtol=1e-5)
        assert_allclose(
            grad[1:5],
            [-140.999489795918, -23.815581632653, -46.331163265306,
             -23.613581632653], rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [0.016, 0.017, 0.018, 0.019],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [-28.194897959184, -140.989489795918, -4.758316326531,
             -23.806581632653], rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [89.968035714286, 89.968035714286, 71.728526785714,
             71.728526785714], rtol=1e-5)

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
        assert_allclose(grad[0], 593.757357143, rtol=1e-5)
        assert_allclose(
            grad[1:5],
            [254.938867346939, 42.01281122449, 85.34082244898,
             42.17881122449], rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [0.019, 0.0205, 0.022, 0.0235],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [5.671521798469e+02, 6.407271683673e+01, 9.664416581633e+01,
             1.000000000000e-02], rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [-115.806339285714, -142.314515625, -83.484254464286,
             -109.879185044643], rtol=1e-5)

    def test_fit(self):

        data, sen = self.load_data()

        rec = IndependentScorePredictor(
            C=0.01, k=2, eta=10.0, tol=1e-03, random_state=1234)
        rec.fit(data, sen)

        assert_allclose(
            rec.fit_results_['initial_loss'], 0.569924791905653, rtol=1e-5)
        assert_allclose(
            rec.fit_results_['final_loss'], 0.290055339199279, rtol=1e-3)

        # prediction
        assert_allclose(
            rec.raw_predict(data.event, sen)[:5],
            [2.909051862799, 4.080407695792, 3.903904936907, 3.904992008106,
             4.901243803772], rtol=1e-3)

        assert_allclose(
            rec.predict((5, 2), 0), 2.9090518628, rtol=1e-5)
        assert_allclose(
            rec.predict([[10, 7], [5, 1], [10, 4]], [1, 0, 0]),
            [4.080407695792, 3.903904936907, 3.904992008106],
            rtol=1e-3)


# =============================================================================
# Main Routines
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
