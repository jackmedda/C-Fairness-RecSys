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

from kamrecsys.utils import safe_sigmoid as sigmoid

from . import BaseIndependentLogisticPMFWithOptimizer

# =============================================================================
# Module metadata variables
# =============================================================================

# =============================================================================
# Public symbols
# =============================================================================

__all__ = ['IndependentItemFinder']

# =============================================================================
# Constants
# =============================================================================

# =============================================================================
# Module variables
# =============================================================================

# =============================================================================
# Classes
# =============================================================================


class IndependentItemFinder(BaseIndependentLogisticPMFWithOptimizer):
    """
    Independence enhanced :class:`kamrecsys.item_finder.LogisticPMF`.

    The independence term is designed to match Pr[s|t=0] and Pr[s|t=1]. For
    this purpose, this tries to match means of these two distributions.
    """

    method_name = 'logistic_pmf_mean_match'

    def __init__(self, C=1.0, eta=1.0, k=1,
                 random_state=None, **optimizer_kwargs):
        super(IndependentItemFinder,
              self).__init__(
            C=C, eta=eta, k=k, random_state=random_state, **optimizer_kwargs)

    def loss(self, coef, sev, ssc, n_objects):
        """
        loss function to optimize

        main loss function: same as the logistic PMF item finder in kamrecsys.

        independence term:

        This independence term enforces to match two conditional distributions
        of scores when sen == 0 and s = 1. For this purpose,
        this term tries to match means of these two distributions.
        To estimate distribution of estimated scores, we adopt a histogram

        .. math::

          (\frac{1}{N(s=0)} \sum_{(x,y)\in D s.t. t=0} f(r | x, y, s=0) -
           \frac{1}{N(s=1)} \sum_{(x,y)\in D s.t. t=0} f(r | x, y, s=1))^2

        where N(s=0) is the number of samples in D such that s=0 and f(r |
        x, y, s) denotes a function for predicting scores given x, y, and s.

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
        n_events = np.array([ev.shape[0] for ev in sev])

        # set array's view
        mu = coef.view(self._dt)['mu']
        bu = coef.view(self._dt)['bu']
        bi = coef.view(self._dt)['bi']
        p = coef.view(self._dt)['p']
        q = coef.view(self._dt)['q']

        # basic stats
        esc = np.empty(self.n_sensitive_values, dtype=object)
        for s in xrange(self.n_sensitive_values):
            ev = sev[s]
            esc[s] = sigmoid(
                mu[s][0] + bu[s][ev[:, 0]] + bi[s][ev[:, 1]] +
                np.sum(p[s][ev[:, 0], :] * q[s][ev[:, 1], :], axis=1))

        # loss term and independence term
        loss = - np.sum([
            np.sum(ssc[s] * np.log(esc[s]) + (1 - ssc[s]) * np.log(1 - esc[s]))
            for s in xrange(self.n_sensitive_values)])

        # independence term #####
        indep = (np.mean(esc[0]) - np.mean(esc[1])) ** 2

        # regularization term
        reg = 0.0
        for s in xrange(self.n_sensitive_values):
            reg += (np.sum(bu[s] ** 2) + np.sum(bi[s] ** 2) +
                    np.sum(p[s] ** 2) + np.sum(q[s] ** 2))

        return loss + self.eta * indep + 0.5 * self.C * reg

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

        Note
        ----
        A constant factor 2 is ignored.
        """
        # constants
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
        esc = np.empty(self.n_sensitive_values, dtype=object)
        for s in xrange(self.n_sensitive_values):
            ev = sev[s]
            esc[s] = sigmoid(
                mu[s][0] + bu[s][ev[:, 0]] + bi[s][ev[:, 1]] +
                np.sum(p[s][ev[:, 0], :] * q[s][ev[:, 1], :], axis=1))

        # gradients of loss term
        for s in xrange(self.n_sensitive_values):
            ev = sev[s]
            common_term = esc[s] - ssc[s]
            grad_mu[s] += np.sum(common_term)
            grad_bu[s][:] += np.bincount(
                ev[:, 0], weights=common_term, minlength=n_users)
            grad_bi[s][:] += np.bincount(
                ev[:, 1], weights=common_term, minlength=n_items)
            weights = common_term[:, np.newaxis] * q[s][ev[:, 1], :]
            for i in xrange(k):
                grad_p[s][:, i] += np.bincount(
                    ev[:, 0], weights=weights[:, i], minlength=n_users)
            weights = common_term[:, np.newaxis] * p[s][ev[:, 0], :]
            for i in xrange(k):
                grad_q[s][:, i] += np.bincount(
                    ev[:, 1], weights=weights[:, i], minlength=n_items)

        # gradients of independence term #####
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
                np.bincount(ev[:, 0], weights=weights[:, i],
                            minlength=n_users) / n_events[0])
        weights = p[0][ev[:, 0], :]
        for i in xrange(k):
            grad_q[0][:, i] += (
                self.eta * diff_mean *
                np.bincount(ev[:, 1], weights=weights[:, i],
                            minlength=n_items) / n_events[0])

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
                np.bincount(ev[:, 0], weights=weights[:, i],
                            minlength=n_users) / n_events[1])
        weights = p[1][ev[:, 0], :]
        for i in xrange(k):
            grad_q[1][:, i] += (
                self.eta * (- diff_mean) *
                np.bincount(ev[:, 1], weights=weights[:, i],
                            minlength=n_items) / n_events[1])

        # gradient of regularization term
        for s in xrange(self.n_sensitive_values):
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

# init logging system ---------------------------------------------------------

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
