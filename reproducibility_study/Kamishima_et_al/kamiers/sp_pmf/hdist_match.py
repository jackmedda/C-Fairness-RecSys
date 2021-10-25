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

    The independence term is implemented as Hellinger distance between scores
    under the two sensitive features. each target distribution is modeled by
    a normal distribution model. To minimize the Hellinger distance, the
    corresponding Bhattacharyya coefficient is maximized.

    For the details of parameters, see
    :class:`kamiers.sp_pmf.BaseIndependentPMFWithOptimizer` .

    Parameters
    ----------
    a : float
        the number of prior effective samples. This is the 1/2 of shape
        parameter of a prior Gamma distribution.
    b : float
        (b0 / a0) is the variance of prior effective samples. This is
        the 1/2 of rate parameter of a prior Gamma distribution.
    """

    method_name = 'pmf_bdist_match'

    def __init__(self, C, k, eta, a=1e-8, b=1e-8,
                 random_state=None, **optimizer_kwargs):
        super(
            IndependentScorePredictor, self).__init__(
            C=C, k=k, eta=eta, random_state=random_state, **optimizer_kwargs)

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

            BC = \int\sqrt{\Pr[D_0]\Pr[D_1]}dx

            - \log \int\sqrt{\Pr[D_0]\Pr[D_1]}dx

            = - \frac{1}{2}
            \log\left(\frac{2\sqrt{V(D_0)V(D_1)}}{V(D_0)+V(D_1)}\right)
            + \frac{(M(D_0) - M(D_1))^2}{4(V(D_0) + V(D_1))}

        and, other constants are defined as:

        .. math::

            V(D_t) = \frac{Q_t - S_t^2 / N_t + b}{N_t + b}

            M(D_t) = S_t / N_t

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
        a = self.a
        b = self.b

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
        s_var = (sc_qum - sc_sum ** 2 / n_events + b) / (n_events + a)

        # Bhattachryya distance
        mean_diff = sc_sum[0] / n_events[0] - sc_sum[1] / n_events[1]
        indep = (- 0.5 * np.log(2) - 0.25 * np.sum(
            np.log(s_var)) + 0.5 * np.log(np.sum(s_var)) +
                 0.25 * (mean_diff ** 2) / np.sum(s_var))
        indep = - np.exp(- indep)

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

            \frac{\partial}{\partial\theta_0}
            - \exp\left[- \left(- \log \sqrt{\Pr[D_0]\Pr[D_1]}\right)\right]
            \frac{\partial}{\partial\theta_0}
            \left[- \log \sqrt{\Pr[D_0]\Pr[D_1]}\right]

            \frac{\partial}{\partial\theta_0}
            \left[- \log \sqrt{\Pr[D_0]\Pr[D_1]}\right] =
            \frac{M(D_0) - M(D_1)}{2(V(D_0) - V(D_1))}
            frac{\partial M(D_0)}{\partial\theta_0} + \left[
            - \frac{1}{4V(D_0)}
            + \frac{1}{2(V(D_0) + V(D_1))}
            - \frac{1}{4}\left(\frac{M(D_0)-M(D_1)}{V(D_0)+V(D_1)}\right)^2
            \right] frac{\partial V(D_0)}{\partial\theta_0}

        and, other constants are defined as:

        .. math::

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
        n_users = n_objects[0]
        n_items = n_objects[1]
        k = self.k
        a = self.a
        b = self.b

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
        s_var = (sc_qum - (sc_sum ** 2 / n_events) + b) / (n_events + a)

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
        mean_diff = sc_sum[0] / n_events[0] - sc_sum[1] / n_events[1]
        coef_m = mean_diff / (2 * np.sum(s_var))
        coef_v = (- 1 / (4 * s_var) +
                  1 / (2 * np.sum(s_var)) -
                  (1 / 4) * ((mean_diff / np.sum(s_var)) ** 2))
        coef_sum = np.empty(n_s_values)
        coef_sum[0] = coef_m - 2 * sc_sum[0] * coef_v[0] / (n_events[0] + a)
        coef_sum[1] = - coef_m - 2 * sc_sum[1] * coef_v[1] / (n_events[1] + a)
        coef_sum = coef_sum / n_events
        coef_qum = coef_v / (n_events + a)

        # Bhattacharyya coefficient
        bc = np.exp(- (- 0.5 * np.log(2) -
                       0.25 * np.sum(np.log(s_var)) +
                       0.5 * np.log(np.sum(s_var)) +
                       0.25 * (mean_diff ** 2) / np.sum(s_var)))

        for s in xrange(n_s_values):
            grad_mu[s] += (
                self.eta * bc *
                (coef_qum[s] * qum_grad_mu[s] + coef_sum[s] * sum_grad_mu[s]))
            grad_bu[s][:] += (
                self.eta * bc *
                (coef_qum[s] * qum_grad_bu[s] + coef_sum[s] * sum_grad_bu[s]))
            grad_bi[s][:] += (
                self.eta * bc *
                (coef_qum[s] * qum_grad_bi[s] + coef_sum[s] * sum_grad_bi[s]))
            grad_p[s][:, :] += (
                self.eta * bc *
                (coef_qum[s] * qum_grad_p[s] + coef_sum[s] * sum_grad_p[s]))
            grad_q[s][:, :] += (
                self.eta * bc *
                (coef_qum[s] * qum_grad_q[s] + coef_sum[s] * sum_grad_q[s]))

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
