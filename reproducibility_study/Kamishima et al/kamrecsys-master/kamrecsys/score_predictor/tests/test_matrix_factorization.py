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
import numpy as np

from sklearn.utils import check_random_state

from kamrecsys.datasets import load_movielens_mini
from kamrecsys.score_predictor import PMF

# =============================================================================
# Module variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Test Classes
# =============================================================================


class TestPMF(TestCase):

    def test_loss(self):

        # setup
        data = load_movielens_mini()
        rec = PMF(C=0.1, k=2, random_state=1234, tol=1e-03)

        rec._rng = check_random_state(rec.random_state)
        ev = data.event
        sc = data.score
        n_objects = data.n_objects
        rec._init_coef(ev, sc, n_objects)

        # set array's view
        mu = rec._coef.view(rec._dt)['mu']
        bu = rec._coef.view(rec._dt)['bu']
        bi = rec._coef.view(rec._dt)['bi']
        p = rec._coef.view(rec._dt)['p']
        q = rec._coef.view(rec._dt)['q']

        # initial parameters
        assert_allclose(
            rec.loss(rec._coef, ev, sc, n_objects), 11.6412258667768,
            rtol=1e-5)

        # all zero
        mu[0] = 0.0
        bu[0][:] = 0.0
        bi[0][:] = 0.0
        p[0][:, :] = 0.0
        q[0][:, :] = 0.0
        assert_allclose(
            rec.loss(rec._coef, ev, sc, n_objects), 235.5,
            rtol=1e-5)

        # all one
        mu[0] = 1.0
        bu[0][:] = 1.0
        bi[0][:] = 1.0
        p[0][:, :] = 1.0
        q[0][:, :] = 1.0
        assert_allclose(
            rec.loss(rec._coef, ev, sc, n_objects), 38.2,
            rtol=1e-5)

        mu[0] = 1.0
        bu[0][:] = np.arange(0.0, 0.8, 0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) + 1.0
        p[0][:, 0] = 0.5
        p[0][:, 1] = 1.0
        q[0][:, 0] = 0.2
        q[0][:, 1] = 1.0
        assert_allclose(
            rec.loss(rec._coef, ev, sc, n_objects), 18.7025,
            rtol=1e-5)

        mu[0] = 2.0
        bu[0][:] = np.arange(0.8, 0.0, -0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) * 1.5 + 1.0
        p[0][:, 0] = np.arange(0.0, 0.8, 0.1) * 0.8 + 3
        p[0][:, 1] = 1.0
        q[0][:, 0] = np.arange(1.0, 0.0, -0.1) * 0.3 + 2
        q[0][:, 1] = np.arange(0.0, 1.0, 0.1)
        assert_allclose(
            rec.loss(rec._coef, ev, sc, n_objects), 947.15920616,
            rtol=1e-5)

    def test_grad_loss(self):

        # setup
        data = load_movielens_mini()
        rec = PMF(C=0.1, k=2, random_state=1234, tol=1e-03)
        rec._rng = check_random_state(rec.random_state)
        ev = data.event
        sc = data.score
        n_objects = data.n_objects
        rec._init_coef(ev, sc, n_objects)

        # set array's view
        mu = rec._coef.view(rec._dt)['mu']
        bu = rec._coef.view(rec._dt)['bu']
        bi = rec._coef.view(rec._dt)['bi']
        p = rec._coef.view(rec._dt)['p']
        q = rec._coef.view(rec._dt)['q']

        # initial parameters
        grad = rec.grad_loss(rec._coef, ev, sc, n_objects)
        assert_allclose(grad[0], -0.6389059157188512, rtol=1e-5)
        assert_allclose(
            grad[1:5],
            [1.0788090999, 0.464982617, -0.7963940914, -0.0692397823],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [-2.2818579721, -0.7857900494, -1.1561738739, 0.7205513543],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [0.1565786903, -2.9713197014, 0.4173437636, -0.0944053917],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [-0.2978489043, 1.6373689599, 0.6585058144, -0.2955021786],
            rtol=1e-5)

        # all zero
        mu[0] = 0.0
        bu[0][:] = 0.0
        bi[0][:] = 0.0
        p[0][:, :] = 0.0
        q[0][:, :] = 0.0
        grad = rec.grad_loss(rec._coef, ev, sc, n_objects)
        assert_allclose(grad[0], -115, rtol=1e-5)
        assert_allclose(
            grad[1:5],
            [-36., -6., -7., -14.],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [-22., -10., -18., -9.],
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
        mu[0] = 1.0
        bu[0][:] = 1.0
        bi[0][:] = 1.0
        p[0][:, :] = 1.0
        q[0][:, :] = 1.0
        grad = rec.grad_loss(rec._coef, ev, sc, n_objects)
        assert_allclose(grad[0], 35.0, rtol=1e-5)
        assert_allclose(
            grad[1:5],
            [14.1, 4.1, 3.1, 6.1],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [8.1, 5.1, 2.1, 6.1],
        rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [14.1, 14.1, 4.1, 4.1],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [2.1, 2.1, 6.1, 6.1],
            rtol=1e-5)

        mu[0] = 1.0
        bu[0][:] = np.arange(0.0, 0.8, 0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) + 1.0
        p[0][:, 0] = 0.5
        p[0][:, 1] = 1.0
        q[0][:, 0] = 0.2
        q[0][:, 1] = 1.0
        grad = rec.grad_loss(rec._coef, ev, sc, n_objects)
        assert_allclose(grad[0], 0.6000000000000028, rtol=1e-5)
        assert_allclose(
            grad[1:5],
            [-0.5, 1.31, -0.28, 1.73],
            rtol=1e-4)
        assert_allclose(
            grad[15:19],
            [2.86, 2.27, -0.82, 3.69],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [-0.05, -0.4, 0.31, 1.4],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [-0.48, -0.9, 1.77, 3.6],
            rtol=1e-5)

        mu[0] = 2.0
        bu[0][:] = np.arange(0.8, 0.0, -0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) * 1.5 + 1.0
        p[0][:, 0] = np.arange(0.0, 0.8, 0.1) * 0.8 + 3
        p[0][:, 1] = 1.0
        q[0][:, 0] = np.arange(1.0, 0.0, -0.1) * 0.3 + 2
        q[0][:, 1] = np.arange(0.0, 1.0, 0.1)
        grad = rec.grad_loss(rec._coef, ev, sc, n_objects)
        assert_allclose(grad[0], 234.84160000000003, rtol=1e-5)
        assert_allclose(
            grad[1:5],
            [78.28, 17.0564, 14.9512, 33.0668],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [49.89, 26.1354, 31.0472, 27.967],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [168.997, 37.31, 36.622072, 9.28216],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [101.688816, 30.9072, 86.899864, 27.822],
            rtol=1e-5)

    def test_class(self):

        data = load_movielens_mini()

        rec = PMF(C=0.1, k=2, random_state=1234, tol=1e-03)
        rec.fit(data)

        assert_allclose(
            rec.fit_results_['initial_loss'], 11.6412258667768, rtol=1e-5)
        assert_allclose(
            rec.fit_results_['final_loss'], 0.683795023344032, rtol=1e-5)

        # raw_predict
        assert_allclose(
            rec.raw_predict(np.array([[0, 6]])), 3.98981537208499, rtol=1e-5)
        assert_allclose(
            rec.raw_predict(np.array([[8, 8]])), 4.25466828602487, rtol=1e-5)
        assert_allclose(
            rec.raw_predict(np.array([[2, 10]])), 3.45221824936513, rtol=1e-5)
        assert_allclose(
            rec.raw_predict(np.array([[0, 6], [8, 8], [2, 10]])),
            [3.9898153721, 4.254668286, 3.4522182494],
            rtol=1e-5)

        # single prediction
        assert_allclose(rec.predict((1, 7)), 3.98981537208499, rtol=1e-5)
        assert_allclose(rec.predict((1, 9)), 4.96457075887952, rtol=1e-5)
        assert_allclose(rec.predict((1, 11)), 3.64937487633011, rtol=1e-5)
        assert_allclose(rec.predict((3, 7)), 3.56199613750802, rtol=1e-5)
        assert_allclose(rec.predict((3, 9)), 4.25466828602487, rtol=1e-5)
        assert_allclose(rec.predict((3, 11)), 3.74991933999025, rtol=1e-5)
        assert_allclose(rec.predict((5, 7)), 3.27152744041802, rtol=1e-5)
        assert_allclose(rec.predict((5, 9)), 3.99212459353293, rtol=1e-5)
        assert_allclose(rec.predict((5, 11)), 3.45221824936513, rtol=1e-5)

        # multiple prediction
        x = np.array([
            [1, 7], [1, 9], [1, 11],
            [3, 7], [3, 9], [3, 11],
            [5, 7], [5, 9], [5, 11]])
        assert_allclose(
            rec.predict(x),
            [3.9898153721, 4.9645707589, 3.6493748763,
             3.5619961375, 4.254668286, 3.74991934,
             3.2715274404, 3.9921245935, 3.4522182494],
            rtol=1e-5)


# =============================================================================
# Main Routines
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
