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

from kamiers.if_lpmf import BaseIndependentLogisticPMF
from kamiers.if_lpmf.mean_match import IndependentItemFinder

# =============================================================================
# Module variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Test Classes
# =============================================================================


class TestIndependentItemFinder(TestCase):

    @staticmethod
    def load_data():
        from kamrecsys.datasets import load_event_with_score
        from kamiers.datasets import event_dtype_sensitive_and_timestamp

        infile = os.path.join(os.path.dirname(__file__), 'mlmini_t.event')
        data = load_event_with_score(
            infile, score_domain=(1, 5, 1),
            event_dtype=event_dtype_sensitive_and_timestamp)
        sen = data.event_feature['sensitive']
        data.binarize_score(3)
        return data, sen

    def test_loss(self):

        # setup data
        data, sen = self.load_data()
        rec = IndependentItemFinder(
            C=0.01, k=2, eta=100.0, tol=1000.0, maxiter=200)
        BaseIndependentLogisticPMF.fit(rec, data, sen, event_index=(0, 1))
        sev, ssc, n_events = rec.get_sensitive_divided_data()
        rec._init_coef(data, sev, ssc, rec.n_objects)

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
            20.7944154167984)

        # all one
        mu[:] = 1.0
        bu[:] = 1.0
        bi[:] = 1.0
        p[:, :] = 1.0
        q[:, :] = 1.0
        assert_allclose(
            rec.loss(rec._coef, sev, ssc, rec.n_objects),
            45.74146045467371)

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
            47.846191835552446)

    def test_grad_loss(self):

        # setup data
        data, sen = self.load_data()
        rec = IndependentItemFinder(
            C=0.01, k=2, eta=100.0, tol=1000.0, maxiter=200)
        BaseIndependentLogisticPMF.fit(rec, data, sen, event_index=(0, 1))
        sev, ssc, n_events = rec.get_sensitive_divided_data()
        rec._init_coef(data, sev, ssc, rec.n_objects)

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

        assert_allclose(grad[0], -3.0, rtol=1e-5)
        assert_allclose(
            grad[1:5],
            [0., -0.5, 0., -0.5],
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
        assert_allclose(grad[0], 3.90630008706, rtol=1e-5)
        assert_allclose(
            grad[1:5],
            [2.969842894454, 0.003307149076, 0.996614298151, 0.003307149076],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [0.01, 0.01, 0.01, 0.01],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [2.969842894454, 2.969842894454, 0.003307149076, 0.003307149076],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [-0.016771403697, -0.016771403697, 1.989921447227, 1.989921447227],
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
        assert_allclose(grad[0], 0.698977005912156, rtol=1e-5)
        assert_allclose(
            grad[1:5],
            [1.559319476685, -0.243937667343, 0.522589456936, -0.235067409245],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [0.016, 0.017, 0.018, 0.019],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [0.316863895337, 1.569319476685, -0.043987533469, -0.234937667343],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [0.725835287968, 0.725835287968, 2.546652384668, 2.546652384668],
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
        assert_allclose(grad[0], 4.21825630146929, rtol=1e-5)
        assert_allclose(
            grad[1:5],
            [3.10153227628, 0.022583666984, 1.037173413648, 0.020586872448],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [0.019, 0.0205, 0.022, 0.0235],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [6.898108017775, 0.833387392233, 0.066642434063, 0.01], rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [-0.060704139497, -0.084169233072, 1.952002906139, 2.678853924585],
            rtol=1e-5)

    def test_fit(self):

        data, sen = self.load_data()

        rec = IndependentItemFinder(
            C=0.01, k=2, eta=10.0, random_state=1234, tol=1e-03)
        rec.fit(data, sen)
        assert_allclose(
            rec.fit_results_['initial_loss'], 0.640430231326407, rtol=1e-5)
        assert_allclose(
            rec.fit_results_['final_loss'], 0.640430231326407, rtol=1e-3)

        assert_allclose(
            rec.raw_predict(data.event, sen)[:5],
            [0.009105487356, 0.997862361352, 0.998630214017,
             0.997383459726, 0.998465946466],
            rtol=1e-3)

        assert_allclose(
            rec.predict((5, 2), 0), 0.00910548735648, rtol=1e-3)
        assert_allclose(
            rec.predict([[10, 7], [5, 1], [10, 4]], [1, 0, 0]),
            [0.997862361352, 0.998630214017, 0.997383459726],
            rtol=1e-5)


# =============================================================================
# Main Routines
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
