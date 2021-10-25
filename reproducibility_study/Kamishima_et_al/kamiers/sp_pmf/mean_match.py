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

    The independence term is designed to match Pr[s|t=0] and Pr[s|t=1]. For
    this purpose, this tries to match means of these two distributions.

    For the details of parameters, see
    :class:`kamiers.sp_pmf.BaseIndependentPMFWithOptimizer` .

    References
    ----------

    .. [1] Kamishima et al. "Efficiency Improvement of Neutrality-enhanced
        Recommendation" The 3rd Workshop on Human Decision Making in
        Recommender Systems, 2013.
    """

    method_name = 'pmf_mean_match'

    def loss(self, coef, sev, ssc, n_objects):
        """
        loss function to optimize

        main loss function: same as the PMF score_predictor in kamrecsys.

        independence term:

        This independence term enforces to match two conditional distributions
        of scores when trg == 0 and trg = 1. For this purpose,
        this term tries to match means of these two distributions.
        To estimate distribution of estimated scores, we adopt a histogram

        .. math::

            \frac{1}{2}
            (\frac{1}{N(t=0)} \sum_{(x,y)\in D s.t. t=0} f(s | x, y, t=0) -
            \frac{1}{N(t=1)} \sum_{(x,y)\in D s.t. t=0} f(s | x, y, t=1))^2

        where N(t=0) is the number of samples in D such that t=0 and f(s |
        x, y, t=0) denotes a function for predicting scores given x, y, and t.

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
        indep = 0.5 * (np.mean(esc[0]) - np.mean(esc[1])) ** 2

        # regularization term #####
        reg = 0.0
        for s in xrange(n_s_values):
            reg += (np.sum(bu[s] ** 2) + np.sum(bi[s] ** 2) +
                    np.sum(p[s] ** 2) + np.sum(q[s] ** 2))

        return 0.5 * loss + self.eta * indep + 0.5 * self.C * reg

    def grad_loss(self, coef, sev, ssc, n_objects):
        """
        gradient of loss function

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

        # difference of means of predicted ratings for the same sensitive value
        diff_mean = np.mean(esc[0]) - np.mean(esc[1])

        ev = sev[0]
        grad_mu[0] += self.eta * diff_mean
        grad_bu[0][:] += (
            self.eta * diff_mean *
            np.bincount(ev[:, 0], minlength=n_users) / n_events[0])
        grad_bi[0][:] += (
            self.eta * diff_mean *
            np.bincount(ev[:, 1], minlength=n_items) / n_events[0])
        weights = q[0][ev[:, 1], :]
        for i in xrange(k):
            grad_p[0][:, i] += (
                self.eta * diff_mean *
                np.bincount(
                    ev[:, 0], weights=weights[:, i], minlength=n_users) /
                n_events[0])

        weights = p[0][ev[:, 0], :]
        for i in xrange(k):
            grad_q[0][:, i] += (
                self.eta * diff_mean *
                np.bincount(
                    ev[:, 1], weights=weights[:, i], minlength=n_items) /
                n_events[0])

        ev = sev[1]
        grad_mu[1] += self.eta * (- diff_mean)
        grad_bu[1][:] += (
            self.eta * (- diff_mean) *
            np.bincount(ev[:, 0], minlength=n_users) / n_events[1])
        grad_bi[1][:] += (
            self.eta * (- diff_mean) *
            np.bincount(ev[:, 1], minlength=n_items) / n_events[1])
        weights = q[1][ev[:, 1], :]
        for i in xrange(k):
            grad_p[1][:, i] += (
                self.eta * (- diff_mean) *
                np.bincount(
                    ev[:, 0], weights=weights[:, i], minlength=n_users) /
                n_events[1])
        weights = p[1][ev[:, 0], :]
        for i in xrange(k):
            grad_q[1][:, i] += (
                self.eta * (- diff_mean) *
                np.bincount(
                    ev[:, 1], weights=weights[:, i], minlength=n_items) /
                n_events[1])

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
