#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Independence-enhanced kamrecsys.score_predictor.PMF
"""

from __future__ import (
    print_function,
    division,
    absolute_import,
    unicode_literals)
from six.moves import xrange

# =============================================================================
# Imports
# =============================================================================

import logging

import numpy as np

from . import BaseIndependentPMFWithOptimizer

# =============================================================================
# Module metadata variables
# =============================================================================

# =============================================================================
# Public symbols
# =============================================================================

__all__ = ['IndependentScorePredictor']

# =============================================================================
# Constants
# =============================================================================

# =============================================================================
# Module variables
# =============================================================================

# =============================================================================
# Classes
# =============================================================================


class IndependentScorePredictor(BaseIndependentPMFWithOptimizer):
    """
    Independence enhanced :class:`kamrecsys.score_predictor.PMF`.

    The independence term is directly implemented as mutual information
    between scores and targets. Probability distribution of scores given
    targets is modeled by a normal distribution model.

    For the details of parameters, see
    :class:`kamiers.sp_pmf.BaseIndependentPMFWithOptimizer` .

    Parameters
    ----------
    a : int or float
        parameter of gamma prior for a Gaussian's variance.
        It should be a << n_samples. (default=1e-8)
    b : int or float
        parameter of gamma prior for a Gaussian's variance.
        It should be b << var(data) n_samples . (default=1e-8)

    References
    ----------

    .. [1] T. Kamishima et al. "Recommendation Independence"
        Conference on Fairness, Accountability, and Transparency, 2017.
    """

    method_name = 'pmf_mi_normal'

    def __init__(self, C=1.0, eta=1.0, k=1, tol=1e-5, maxiter=200,
                 a=1e-8, b=1e-24,
                 random_state=None):
        super(IndependentScorePredictor,
              self).__init__(C=C, eta=eta, k=k, tol=tol, maxiter=maxiter,
                             random_state=random_state)

        self.a = a
        self.b = b

    def loss(self, coef, sev, ssc, n_objects):
        """
        loss function to optimize

        main loss function: same as the kamrecsys.mf.pmf.

        independence term:

        This independence term approximates mutual information. To estimate
        distribution of estimated scores, we adopt a single normal distribution
        model.

        .. math::

            - [H(D) - \sum_t \frac{N_t}{N} H(D_t)]

        where :math:`H(X)` is an entropy function of a Gaussian distribution:

        .. math::

            H(X) = 0.5 \log(2 \pi e V(X))

        and, other constants are defined as:

        .. math::

            V(D) = \frac{(Q_1 + Q_2) - (S_1 + S_2)^2 / N + 2 b}{N + 2 a}

            V(D_t) = \frac{Q_t - S_t^2 / N_t + b}{N_t + b}

            S_t = \sum_{x, y \in D_t} \hat{r}(x, y, t)

            Q_t = \sum_{x, y \in D_t} (\hat{r}(x, y, t))^2

        Parameters
        ----------
        coef : array_like, shape=(variable,)
            coefficients of this model
        sev : array_like, shape(n_events, 2), dtype=int
            user and item indexes
        ssc : array_like, shape(n_events,), dtype=float
            target scores
        n_objects : array_like, shape(2,), dtype=int
            numbers of users and items

        Returns
        -------
        loss : float
            value of loss function
        """

        # constants
        n_s_values = self.n_sensitive_values
        n_events = np.array([ev.shape[0] for ev in sev])
        n_tevents = np.sum(n_events)

        # set array's view
        mu = coef.view(self._dt)['mu']
        bu = coef.view(self._dt)['bu']
        bi = coef.view(self._dt)['bi']
        p = coef.view(self._dt)['p']
        q = coef.view(self._dt)['q']

        # basic stats
        esc = np.empty(n_s_values, dtype=object)
        for s in xrange(n_s_values):
            ev = sev[s]
            esc[s] = (mu[s][0] + bu[s][ev[:, 0]] + bi[s][ev[:, 1]] +
                      np.sum(p[s][ev[:, 0], :] * q[s][ev[:, 1], :], axis=1))

        # loss term #####
        loss = 0.0
        for s in xrange(n_s_values):
            loss += np.sum((ssc[s] - esc[s]) ** 2)

        # independence term #####

        # sum and squared sum
        sc_sum = np.array([np.sum(sc) for sc in esc])
        sc_qum = np.array([np.sum(sc ** 2) for sc in esc])

        # variances
        total_var = (
            (np.sum(sc_qum) - (np.sum(sc_sum) ** 2 / n_tevents) +
             2 * self.b) / (n_tevents + 2 * self.a))
        s_var = ((sc_qum - sc_sum ** 2 / n_events + 2 * self.b) /
                 (n_events + 2 * self.a))

        # entropy of Gaussian distribution: 0.5 log( 2 Pi E sigma^2 )
        def g_ent(variance):
            return 0.5 * np.log(2.0 * np.pi * np.e * variance)

        ent_x = g_ent(total_var)
        ent_xgs = np.dot(n_events, g_ent(s_var)) / n_tevents
        indep = np.max((0.0, ent_x - ent_xgs))

        # regularization term #####
        reg = 0.0
        for s in xrange(n_s_values):
            reg += (np.sum(bu[s] ** 2) + np.sum(bi[s] ** 2) +
                    np.sum(p[s] ** 2) + np.sum(q[s] ** 2))

        return 0.5 * loss + self.eta * indep + 0.5 * self.C * reg

    def grad_loss(self, coef, sev, ssc, n_objects):
        """
        gradient of loss function

        .. math::

            \left[\frac{1}{2(N+2a)V(D)}
            - \frac{N_t}{2 N (N_t+a) V(D_t)}\right]
            \frac{\partial Q_t}{\partial \theta_t}\\
            -\left[\frac{\sum_t S_t}{N(N+2a)V(D)}-
            \frac{S_t}{N(N+a)V(D_t)}\right]
            \frac{\partial S_t}{\partial \theta_t}

        and, other constants are defined as:

        .. math::

            V(D) = \frac{(Q_1 + Q_2) - (S_1 + S_2)^2 / N + 2 b}{N + 2 a}

            V(D_t) = \frac{Q_t - S_t^2 / N_t + b}{N_t + b}

            S_t = \sum_{x, y \in D_t} \hat{r}(x, y, t)

            Q_t = \sum_{x, y \in D_t} (\hat{r}(x, y, t))^2

            \frac{\partial S_t}{\partial\theta_t} =
            \sum_{x, y \in D_t}
            \frac{\partial}{\partial\theta_t} \hat{r}(x, y, t)

            \frac{\partial Q_t}{\partial\theta_t} =
            \sum_{x, y \in D_t}
            2 \hat{r}(x, y, t)
            \frac{\partial}{\partial\theta_t} \hat{r}(x, y, t)

        Parameters
        ----------
        coef : array_like, shape=(variable,)
            coefficients of this model
        sev : array_like, shape(n_events, 2), dtype=int
            user and item indexes
        ssc : array_like, shape(n_events,), dtype=float
            target scores
        n_objects : array_like, shape(2,), dtype=int
            numbers of users and items

        Returns
        -------
        grad : array_like, shape=coef.shape
            the first gradient of loss function by coef
        """

        # constants
        n_s_values = self.n_sensitive_values
        n_events = np.array([ev.shape[0] for ev in sev])
        n_tevents = np.sum(n_events)
        n_users = n_objects[0]
        n_items = n_objects[1]
        k = self.k

        # set array's view
        mu = coef.view(self._dt)['mu']
        bu = coef.view(self._dt)['bu']
        bi = coef.view(self._dt)['bi']
        p = coef.view(self._dt)['p']
        q = coef.view(self._dt)['q']

        # create empty gradient
        grad = np.zeros_like(coef)
        grad_mu = grad.view(self._dt)['mu']
        grad_bu = grad.view(self._dt)['bu']
        grad_bi = grad.view(self._dt)['bi']
        grad_p = grad.view(self._dt)['p']
        grad_q = grad.view(self._dt)['q']

        # basic stats
        esc = np.empty(n_s_values, dtype=object)
        for s in xrange(n_s_values):
            ev = sev[s]
            esc[s] = (mu[s][0] + bu[s][ev[:, 0]] + bi[s][ev[:, 1]] +
                      np.sum(p[s][ev[:, 0], :] * q[s][ev[:, 1], :], axis=1))

        # gradients of loss term #####
        for s in xrange(n_s_values):
            ev = sev[s]
            neg_residual = - (ssc[s] - esc[s])
            grad_mu[s] += np.sum(neg_residual)
            grad_bu[s][:] += np.bincount(
                ev[:, 0], weights=neg_residual, minlength=n_users)
            grad_bi[s][:] += np.bincount(
                ev[:, 1], weights=neg_residual, minlength=n_items)
            weights = neg_residual[:, np.newaxis] * q[s][ev[:, 1], :]
            for i in xrange(k):
                grad_p[s][:, i] += np.bincount(
                    ev[:, 0], weights=weights[:, i], minlength=n_users)
            weights = neg_residual[:, np.newaxis] * p[s][ev[:, 0], :]
            for i in xrange(k):
                grad_q[s][:, i] += np.bincount(
                    ev[:, 1], weights=weights[:, i], minlength=n_items)

        # gradients of independence term #####

        # sum and squared sum
        sc_sum = np.array([np.sum(sc) for sc in esc])
        sc_qum = np.array([np.sum(sc ** 2) for sc in esc])

        # variances
        total_var = (
            (np.sum(sc_qum) - (np.sum(sc_sum) ** 2 / n_tevents) +
             2 * self.b) / (n_tevents + 2 * self.a))
        s_var = ((sc_qum - (sc_sum ** 2 / n_events) + 2 * self.b) /
                 (n_events + 2 * self.a))

        # gradient of squared sum of predicted scores
        qum_grad = np.empty_like(coef)
        qum_grad_mu = qum_grad.view(self._dt)['mu']
        qum_grad_bu = qum_grad.view(self._dt)['bu']
        qum_grad_bi = qum_grad.view(self._dt)['bi']
        qum_grad_p = qum_grad.view(self._dt)['p']
        qum_grad_q = qum_grad.view(self._dt)['q']

        for s in xrange(n_s_values):
            ev = sev[s]
            grad_coef = 2 * esc[s]
            qum_grad_mu[s] = np.sum(grad_coef)
            qum_grad_bu[s][:] = np.bincount(
                ev[:, 0], weights=grad_coef, minlength=n_users)
            qum_grad_bi[s][:] = np.bincount(
                ev[:, 1], weights=grad_coef, minlength=n_items)
            weights = grad_coef[:, np.newaxis] * q[s][ev[:, 1], :]
            for i in xrange(k):
                qum_grad_p[s][:, i] = np.bincount(
                    ev[:, 0], weights=weights[:, i], minlength=n_users)
            weights = grad_coef[:, np.newaxis] * p[s][ev[:, 0], :]
            for i in xrange(k):
                qum_grad_q[s][:, i] = np.bincount(
                    ev[:, 1], weights=weights[:, i], minlength=n_items)

        # gradient of sum of predicted scores
        sum_grad = np.empty_like(coef)
        sum_grad_mu = sum_grad.view(self._dt)['mu']
        sum_grad_bu = sum_grad.view(self._dt)['bu']
        sum_grad_bi = sum_grad.view(self._dt)['bi']
        sum_grad_p = sum_grad.view(self._dt)['p']
        sum_grad_q = sum_grad.view(self._dt)['q']

        for s in xrange(n_s_values):
            ev = sev[s]
            sum_grad_mu[s] = n_events[s]
            sum_grad_bu[s][:] = np.bincount(ev[:, 0], minlength=n_users)
            sum_grad_bi[s][:] = np.bincount(ev[:, 1], minlength=n_items)
            weights = q[s][ev[:, 1], :]
            for i in xrange(k):
                sum_grad_p[s][:, i] = np.bincount(
                    ev[:, 0], weights=weights[:, i], minlength=n_users)
            weights = p[s][ev[:, 0], :]
            for i in xrange(k):
                sum_grad_q[s][:, i] = np.bincount(
                    ev[:, 1], weights=weights[:, i], minlength=n_items)

        # final computation of gradient of independence term
        coef_qum = ((1 /
                     (2 * (n_tevents + 2 * self.a) * total_var)) -
                    (n_events /
                     (2 * n_tevents * (n_events + 2 * self.a) * s_var)))
        coef_sum = ((np.sum(sc_sum) /
                     (n_tevents * (n_tevents + 2 * self.a) * total_var)) -
                    (sc_sum /
                     (n_tevents * (n_events + 2 * self.a) * s_var)))

        for s in xrange(n_s_values):
            grad_mu[s] += self.eta * (
                coef_qum[s] * qum_grad_mu[s] - coef_sum[s] * sum_grad_mu[s])
            grad_bu[s][:] += self.eta * (
                coef_qum[s] * qum_grad_bu[s] - coef_sum[s] * sum_grad_bu[s])
            grad_bi[s][:] += self.eta * (
                coef_qum[s] * qum_grad_bi[s] - coef_sum[s] * sum_grad_bi[s])
            grad_p[s][:, :] += self.eta * (
                coef_qum[s] * qum_grad_p[s] - coef_sum[s] * sum_grad_p[s])
            grad_q[s][:, :] += self.eta * (
                coef_qum[s] * qum_grad_q[s] - coef_sum[s] * sum_grad_q[s])

        # gradient of regularization term #####
        for s in xrange(n_s_values):
            grad_bu[s][:] += self.C * bu[s]
            grad_bi[s][:] += self.C * bi[s]
            grad_p[s][:, :] += self.C * p[s]
            grad_q[s][:, :] += self.C * q[s]

        return grad


# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Module initialization 
# =============================================================================

# init logging system
logger = logging.getLogger('kamiers')
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

# =============================================================================
# Test routine
# =============================================================================


def _test():
    """ test function for this module
    """

    # perform doctest
    import sys
    import doctest

    doctest.testmod()

    sys.exit(0)


# Check if this is call as command script -------------------------------------

if __name__ == '__main__':
    _test()
