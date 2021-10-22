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

from scipy import sparse as sparse
from sklearn.utils import check_random_state

from kamrecsys.datasets import load_movielens_mini
from kamrecsys.item_finder import LogisticPMF, ImplicitLogisticPMF

# =============================================================================
# Module variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Test Classes
# =============================================================================


class TestLogisticPMF(TestCase):

    def test_loss(self):

        # setup
        data = load_movielens_mini()
        data.binarize_score(3)
        rec = LogisticPMF(C=0.1, k=2, random_state=1234, tol=1e-03)

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
            rec.loss(rec._coef, ev, sc, n_objects), 23.7963271809711,
            rtol=1e-5)

        # all zero
        mu[0] = 0.0
        bu[0][:] = 0.0
        bi[0][:] = 0.0
        p[0][:, :] = 0.0
        q[0][:, :] = 0.0
        assert_allclose(
            rec.loss(rec._coef, ev, sc, n_objects), 20.7944154167983,
            rtol=1e-5)

        # all one
        mu[0] = 1.0
        bu[0][:] = 1.0
        bi[0][:] = 1.0
        p[0][:, :] = 1.0
        q[0][:, :] = 1.0
        assert_allclose(
            rec.loss(rec._coef, ev, sc, n_objects), 47.9014604546737,
            rtol=1e-5)

        mu[0] = 1.0
        bu[0][:] = np.arange(0.0, 0.8, 0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) + 1.0
        p[0][:, 0] = 0.5
        p[0][:, 1] = 1.0
        q[0][:, 0] = 0.2
        q[0][:, 1] = 1.0
        assert_allclose(
            rec.loss(rec._coef, ev, sc, n_objects), 36.467883082757,
            rtol=1e-5)

        mu[0] = 2.0
        bu[0][:] = np.arange(0.8, 0.0, -0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) * 1.5 + 1.0
        p[0][:, 0] = np.arange(0.0, 0.8, 0.1) * 0.8 + 3
        p[0][:, 1] = 1.0
        q[0][:, 0] = np.arange(1.0, 0.0, -0.1) * 0.3 + 2
        q[0][:, 1] = np.arange(0.0, 1.0, 0.1)
        assert_allclose(
            rec.loss(rec._coef, ev, sc, n_objects), 113.246948248756,
            rtol=1e-5)

    def test_grad_loss(self):

        # setup
        data = load_movielens_mini()
        data.binarize_score()
        rec = LogisticPMF(C=0.1, k=2, random_state=1234, tol=1e-03)
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
        assert_allclose(grad[0], -2.69422502892223, rtol=1e-5)
        assert_allclose(
            grad[1:5],
            [1.4860354254, 0.5813833517, -0.2378488265, -0.4549466355],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [-1.4192072182, -0.4856324281, -1.793371146, 1.1698467502],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [2.218012237, -2.8022576986, 0.7577517736, -0.189979212],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [-0.252934635, 2.569059726, 1.5578077993, -0.4423181343],
            rtol=1e-5)

        # all zero
        mu[0] = 0.0
        bu[0][:] = 0.0
        bi[0][:] = 0.0
        p[0][:, :] = 0.0
        q[0][:, :] = 0.0
        grad = rec.grad_loss(rec._coef, ev, sc, n_objects)
        assert_allclose(grad[0], -6.0, rtol=1e-5)
        assert_allclose(
            grad[1:5],
            [0., 0., 0., -1.],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [-1., -0.5, -2., 0.5],
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
        assert_allclose(grad[0], 8.799214472271457, rtol=1e-5)
        assert_allclose(
            grad[1:5],
            [5.0330714908, 1.0866142982, 1.0866142982, 1.0732285963],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [2.0598428945, 1.0799214472, 0.0732285963, 2.0799214472],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [5.0330714908, 5.0330714908, 1.0866142982, 1.0866142982],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [0.0732285963, 0.0732285963, 2.0799214472, 2.0799214472],
            rtol=1e-5)

        mu[0] = 1.0
        bu[0][:] = np.arange(0.0, 0.8, 0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) + 1.0
        p[0][:, 0] = 0.5
        p[0][:, 1] = 1.0
        q[0][:, 0] = 0.2
        q[0][:, 1] = 1.0
        grad = rec.grad_loss(rec._coef, ev, sc, n_objects)
        assert_allclose(grad[0], 8.323751110862577, rtol=1e-5)
        assert_allclose(
            grad[1:5],
            [4.7101163359, 0.9545317778, 0.952133346, 0.9486417943],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [2.0612948847, 1.117042198, 0.1220469429, 2.1435828557],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [0.9920232672, 4.8101163359, 0.2389063556, 1.0445317778],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [-0.0089765286, 0.0420469429, 0.9967914278, 2.0535828557],
            rtol=1e-5)

        mu[0] = 2.0
        bu[0][:] = np.arange(0.8, 0.0, -0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) * 1.5 + 1.0
        p[0][:, 0] = np.arange(0.0, 0.8, 0.1) * 0.8 + 3
        p[0][:, 1] = 1.0
        q[0][:, 0] = np.arange(1.0, 0.0, -0.1) * 0.3 + 2
        q[0][:, 1] = np.arange(0.0, 1.0, 0.1)
        grad = rec.grad_loss(rec._coef, ev, sc, n_objects)
        assert_allclose(grad[0], 8.999701753193309, rtol=1e-5)
        assert_allclose(
            grad[1:5],
            [5.0798783079, 1.0699742535, 1.0599646235, 1.0499642981],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [2.1899607439, 1.2049810091, 0.2199797513, 2.2349854785],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [11.0797319135, 2.4999606492, 2.3379421381, 0.9999954835],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [0.2059342372, 0.0799797513, 6.2829547023, 2.0899854785],
            rtol=1e-5)

    def test_class(self):

        data = load_movielens_mini()
        data.binarize_score()

        rec = LogisticPMF(C=0.1, k=2, random_state=1234, tol=1e-03)
        rec.fit(data)

        assert_allclose(
            rec.fit_results_['initial_loss'], 23.7963271809711, rtol=1e-5)
        assert_allclose(
            rec.fit_results_['final_loss'], 3.48393252113573, rtol=1e-5)

        # raw_predict
        assert_allclose(
            rec.raw_predict(np.array([[0, 6]])), 0.989201290366165, rtol=1e-5)
        assert_allclose(
            rec.raw_predict(np.array([[8, 8]])), 0.959375735295748, rtol=1e-5)
        assert_allclose(
            rec.raw_predict(np.array([[2, 10]])), 0.824034860051334, rtol=1e-5)
        assert_allclose(
            rec.raw_predict(np.array([[0, 6], [8, 8], [2, 10]])),
            [0.989201290366, 0.959375735296, 0.824034860051],
            rtol=1e-5)

        # single prediction
        assert_allclose(rec.predict((1, 7)), 0.989201290366165, rtol=1e-5)
        assert_allclose(rec.predict((1, 9)), 0.974756390738565, rtol=1e-5)
        assert_allclose(rec.predict((1, 11)), 0.816803762429987, rtol=1e-5)
        assert_allclose(rec.predict((3, 7)), 0.854574774389645, rtol=1e-5)
        assert_allclose(rec.predict((3, 9)), 0.959375735295748, rtol=1e-5)
        assert_allclose(rec.predict((3, 11)), 0.900999255222352, rtol=1e-5)
        assert_allclose(rec.predict((5, 7)), 0.888811789653219, rtol=1e-5)
        assert_allclose(rec.predict((5, 9)), 0.9634835048284, rtol=1e-5)
        assert_allclose(rec.predict((5, 11)), 0.824034860051334, rtol=1e-5)

        # multiple prediction
        x = np.array([
            [1, 7], [1, 9], [1, 11],
            [3, 7], [3, 9], [3, 11],
            [5, 7], [5, 9], [5, 11]])
        assert_allclose(
            rec.predict(x),
            [0.989201290366, 0.974756390739, 0.81680376243,
             0.85457477439, 0.959375735296, 0.900999255222,
             0.888811789653, 0.963483504828, 0.824034860051],
            rtol=1e-5)


class TestImplicitLogisticPMF(TestCase):

    def test_loss(self):

        # setup
        data = load_movielens_mini()
        rec = ImplicitLogisticPMF(C=0.1, k=2, random_state=1234, tol=1e-03)

        rec._rng = check_random_state(rec.random_state)
        n_objects = data.n_objects
        ev = sparse.coo_matrix(
            (np.ones(data.n_events, dtype=int),
             (data.event[:, 0], data.event[:, 1])), shape=n_objects)
        ev = ev.tocsr()
        rec._init_coef(ev, n_objects)

        # set array's view
        mu = rec._coef.view(rec._dt)['mu']
        bu = rec._coef.view(rec._dt)['bu']
        bi = rec._coef.view(rec._dt)['bi']
        p = rec._coef.view(rec._dt)['p']
        q = rec._coef.view(rec._dt)['q']

        # initial parameters
        assert_allclose(
            rec.loss(rec._coef, ev, n_objects), 103.98807319757,
            rtol=1e-5)

        # all zero
        mu[0] = 0.0
        bu[0][:] = 0.0
        bi[0][:] = 0.0
        p[0][:, :] = 0.0
        q[0][:, :] = 0.0
        assert_allclose(
            rec.loss(rec._coef, ev, n_objects), 55.4517744447956,
            rtol=1e-5)

        # all one
        mu[0] = 1.0
        bu[0][:] = 1.0
        bi[0][:] = 1.0
        p[0][:, :] = 1.0
        q[0][:, :] = 1.0
        assert_allclose(
            rec.loss(rec._coef, ev, n_objects), 253.23722787913,
            rtol=1e-5)

        mu[0] = 1.0
        bu[0][:] = np.arange(0.0, 0.8, 0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) + 1.0
        p[0][:, 0] = 0.5
        p[0][:, 1] = 1.0
        q[0][:, 0] = 0.2
        q[0][:, 1] = 1.0
        assert_allclose(
            rec.loss(rec._coef, ev, n_objects), 200.293169308728,
            rtol=1e-5)

        mu[0] = 2.0
        bu[0][:] = np.arange(0.8, 0.0, -0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) * 1.5 + 1.0
        p[0][:, 0] = np.arange(0.0, 0.8, 0.1) * 0.8 + 3
        p[0][:, 1] = 1.0
        q[0][:, 0] = np.arange(1.0, 0.0, -0.1) * 0.3 + 2
        q[0][:, 1] = np.arange(0.0, 1.0, 0.1)
        assert_allclose(
            rec.loss(rec._coef, ev, n_objects), 593.059407200938,
            rtol=1e-5)

    def test_grad_loss(self):

        # setup
        data = load_movielens_mini()
        rec = ImplicitLogisticPMF(C=0.1, k=2, random_state=1234, tol=1e-03)

        rec._rng = check_random_state(rec.random_state)
        n_objects = data.n_objects
        ev = sparse.coo_matrix(
            (np.ones(data.n_events, dtype=int),
             (data.event[:, 0], data.event[:, 1])), shape=n_objects)
        ev = ev.tocsr()
        rec._init_coef(ev, n_objects)

        # set array's view
        mu = rec._coef.view(rec._dt)['mu']
        bu = rec._coef.view(rec._dt)['bu']
        bi = rec._coef.view(rec._dt)['bi']
        p = rec._coef.view(rec._dt)['p']
        q = rec._coef.view(rec._dt)['q']

        # initial parameters
        grad = rec.grad_loss(rec._coef, ev, n_objects)
        assert_allclose(grad[0], 29.54759073801552, rtol=1e-5)
        assert_allclose(
            grad[1:5],
            [-1.3585150835, 6.1190706839, 3.9453452394, 4.1675236607],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [-0.6724531834, 2.3829809039, 1.7510280667, 3.5386563872],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [-0.3332963261, -0.7966485582, 3.730424055, -1.8144337114],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [2.2257145558, 2.5006539103, 1.8384101853, -0.4847956693],
            rtol=1e-5)

        # all zero
        mu[0] = 0.0
        bu[0][:] = 0.0
        bi[0][:] = 0.0
        p[0][:, :] = 0.0
        q[0][:, :] = 0.0
        grad = rec.grad_loss(rec._coef, ev, n_objects)
        assert_allclose(grad[0], 10.0, atol=1e-5)
        assert_allclose(
            grad[1:5],
            [-5., 3., 3., 1.],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [-2., 1., 0., 1.],
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
        grad = rec.grad_loss(rec._coef, ev, n_objects)
        assert_allclose(grad[0], 49.46457192605721, rtol=1e-5)
        assert_allclose(
            grad[1:5],
            [0.0330714908, 8.0330714908, 8.0330714908, 6.0330714908],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [2.0464571926, 5.0464571926, 4.0464571926, 5.0464571926],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [0.0330714908, 0.0330714908, 8.0330714908, 8.0330714908],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [4.0464571926, 4.0464571926, 5.0464571926, 5.0464571926],
            rtol=1e-5)

        mu[0] = 1.0
        bu[0][:] = np.arange(0.0, 0.8, 0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) + 1.0
        p[0][:, 0] = 0.5
        p[0][:, 1] = 1.0
        q[0][:, 0] = 0.2
        q[0][:, 1] = 1.0
        grad = rec.grad_loss(rec._coef, ev, n_objects)
        assert_allclose(grad[0], 48.309883569540034, rtol=1e-5)
        assert_allclose(
            grad[1:5],
            [-0.2898836641, 7.7469210915, 7.7813127826, 5.813497054],
            rtol=1e-4)
        assert_allclose(
            grad[15:19],
            [2.0195733081, 5.0427133869, 4.0646428559, 5.085469863],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [-0.0079767328, -0.1898836641, 1.5973842183, 7.8369210915],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [1.962321428, 3.9846428559, 2.4677349315, 4.995469863],
            rtol=1e-5)

        mu[0] = 2.0
        bu[0][:] = np.arange(0.8, 0.0, -0.1)
        bi[0][:] = np.arange(0.0, 1.0, 0.1) * 1.5 + 1.0
        p[0][:, 0] = np.arange(0.0, 0.8, 0.1) * 0.8 + 3
        p[0][:, 1] = 1.0
        q[0][:, 0] = np.arange(1.0, 0.0, -0.1) * 0.3 + 2
        q[0][:, 1] = np.arange(0.0, 1.0, 0.1)
        grad = rec.grad_loss(rec._coef, ev, n_objects)
        assert_allclose(grad[0], 49.999242803470324, rtol=1e-5)
        assert_allclose(
            grad[1:5],
            [0.0798783079, 8.0698872383, 8.0598955089, 6.0499031689],
            rtol=1e-5)
        assert_allclose(
            grad[15:19],
            [2.1899451815, 5.204952933, 4.2199595873, 5.2349652997],
            rtol=1e-5)
        assert_allclose(
            grad[19:23],
            [0.2997319135, 0.0999606492, 17.6277516462, 3.6999633395],
            rtol=1e-5)
        assert_allclose(
            grad[-4:],
            [13.3258685412, 4.0799595873, 17.0428870887, 5.0899652997],
            rtol=1e-5)

    def test_class(self):

        # setup
        data = load_movielens_mini()
        rec = ImplicitLogisticPMF(C=0.1, k=2, random_state=1234, tol=1e-03)

        rec.fit(data)
        assert_allclose(
            rec.fit_results_['initial_loss'], 103.98807319757, rtol=1e-5)
        assert_allclose(
            rec.fit_results_['final_loss'], 13.8730554159163, rtol=1e-5)

        # raw_predict
        assert_allclose(
            rec.raw_predict(np.array([[0, 6]])), 0.997028296278715, rtol=1e-5)
        assert_allclose(
            rec.raw_predict(np.array([[8, 8]])), 0.131044842032382, rtol=1e-5)
        assert_allclose(
            rec.raw_predict(np.array([[2, 10]])), 0.058370496087627, rtol=1e-5)
        assert_allclose(
            rec.raw_predict(np.array([[0, 6], [8, 8], [2, 10]])),
            [0.9970282963, 0.131044842, 0.0583704961],
            rtol=1e-5)

        # single prediction
        assert_allclose(rec.predict((1, 7)), 0.997028296278715, rtol=1e-5)
        assert_allclose(rec.predict((1, 9)), 0.944998324514573, rtol=1e-5)
        assert_allclose(rec.predict((1, 11)), 0.964923027550838, rtol=1e-5)
        assert_allclose(rec.predict((3, 7)), 0.624248406054208, rtol=1e-5)
        assert_allclose(rec.predict((3, 9)), 0.131044842032382, rtol=1e-5)
        assert_allclose(rec.predict((3, 11)), 0.174497501438718, rtol=1e-5)
        assert_allclose(rec.predict((5, 7)), 4.07951069077663e-05, rtol=1e-5)
        assert_allclose(rec.predict((5, 9)), 0.00230569601442113, rtol=1e-5)
        assert_allclose(rec.predict((5, 11)), 0.058370496087627, rtol=1e-5)

        # multiple prediction
        x = np.array([
            [1, 7], [1, 9], [1, 11],
            [3, 7], [3, 9], [3, 11],
            [5, 7], [5, 9], [5, 11]])
        assert_allclose(
            rec.predict(x),
            [9.9702829628e-01, 9.4499832451e-01, 9.6492302755e-01,
             6.2424840605e-01, 1.3104484203e-01, 1.7449750144e-01,
             4.0795106908e-05, 2.3056960144e-03, 5.8370496088e-02],
            rtol=1e-5)


# =============================================================================
# Main Routines
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
