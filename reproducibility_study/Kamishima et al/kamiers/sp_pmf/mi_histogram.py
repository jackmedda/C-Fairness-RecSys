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
from scipy.optimize import minimize

from . import BaseIndependentPMF

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


class IndependentScorePredictor(BaseIndependentPMF):
    """
    Independence enhanced :class:`kamrecsys.score_predictor.PMF`.

    The independence term is directly implemented as mutual information
    between scores and targets. Probability distribution of scores given
    targets is modeled by a histogram model.

    For the details of parameters, see
    :class:`kamiers.sp_pmf.BaseIndependentPMF` .

    References
    ----------

    .. [1] T. Kamishima et al. "Enhancement of the Neutrality in
        Recommendation" The 2nd Workshop on Human Decision Making in
        Recommender Systems (2012)
    """

    method_name = 'pmf_mi_histogram'

    def __init__(
            self, C=1.0, k=1, eta=1.0, random_state=None, **optimizer_kwargs):
        super(IndependentScorePredictor, self).__init__(
            C=C, k=k, eta=eta, random_state=random_state)

        # optimizer parameter
        self.optimizer_kwargs = optimizer_kwargs
        optimizer_kwargs['options'] = optimizer_kwargs.get('options', {})
        optimizer_kwargs['options']['disp'] = (
            optimizer_kwargs['options'].get('disp', False))
        opt_maxiter = optimizer_kwargs.pop('maxiter', None)
        if opt_maxiter is not None:
            optimizer_kwargs['options']['maxiter'] = opt_maxiter

    def loss(self, coef, sev, ssc, n_objects, score_bins):
        """
        loss function to optimize

        main loss function: same as the kamrecsys.score_predictor.PMF.

        independence term:

        To estimate distribution of estimated scores, we adopt a histogram
        model. Estimated scores are first discretized into bins according to
        `score_bins`. Distributions are derived from the frequencies of
        events in these bins.

        .. math::

            \sum_{u, i, t} \sum_{s \in bins}
            I(bin(\hat{s}(u, i, t)) == s) log p[s | t] - log p[s]

        u, i, t are user_index, item_index, sensitive feature,
        respectively. \hat(s)(u, i, t) is estimated score, bin(s) is a
        function that returns index of the bin to which s belongs.
        Distribution p[y | t] is derived by generating histogram from a data
        set that is consists of events whose target values equal to t.
        p[s] = \sum_t \p[t] p[s | t].

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
        score_bins : array_like, shape=(variable,) dtype=float, optional
            thresholds to discretize scores.

        Returns
        -------
        loss : float
            value of loss function
        """

        # constants
        n_s_values = self.n_sensitive_values
        n_events = np.array([ev.shape[0] for ev in sev])

        # set array's view
        mu = coef.view(self._dt)['mu']
        bu = coef.view(self._dt)['bu']
        bi = coef.view(self._dt)['bi']
        p = coef.view(self._dt)['p']
        q = coef.view(self._dt)['q']

        # basic stats
        esc = np.empty(n_s_values, dtype=object)
        pesct = np.empty((
            n_s_values, len(score_bins) - 1), dtype=float)
        for s in xrange(n_s_values):
            ev = sev[s]
            esc[s] = (mu[s][0] + bu[s][ev[:, 0]] + bi[s][ev[:, 1]] +
                      np.sum(p[s][ev[:, 0], :] * q[s][ev[:, 1], :], axis=1))
            pesct[s, :] = np.histogram(esc[s], score_bins)[0]
        pesc = np.sum(pesct, axis=0) / np.sum(n_events)
        pesct = pesct / n_events[:, np.newaxis]

        # loss term #####
        loss = 0.0
        for s in xrange(n_s_values):
            loss += np.sum((ssc[s] - esc[s]) ** 2)

        # independence term #####
        indep = 0.0
        for s in xrange(n_s_values):
            # NOTE: to avoid 0 in log
            pos = np.nonzero(pesct[s, :] > 0.0)[0]
            indep += (
                np.dot(pesct[s, pos],
                       np.log(pesct[s, pos]) - np.log(pesc[pos])) *
                n_events[s])
        indep /= np.sum(n_events)

        # regularization term #####
        reg = 0.0
        for s in xrange(n_s_values):
            reg += (np.sum(bu[s] ** 2) + np.sum(bi[s] ** 2) +
                    np.sum(p[s] ** 2) + np.sum(q[s] ** 2))

        return 0.5 * loss + self.eta * indep + 0.5 * self.C * reg

    def fit(self, data, sen, event_index=(0, 1)):
        """
        fitting model

        Parameters
        ----------
        data : :class:`kamrecsys.data.EventWithScoreData`
            data to fit
        sen : array_like, shape=(n_events,)
            a variable to be independent in recommendation
        event_index : array_like, shape=(s_event,)
            a set of indexes to specify the elements in events that are used in
            a recommendation model
        """

        super(IndependentScorePredictor, self).fit(
            data, sen, event_index=event_index)

        # bins fof a score histogram
        if self.score_domain is None:
            score_bins = np.array([-np.inf, 1.5, 2.5, 3.5, 4.5, np.inf])
        else:
            score_bins = self.generate_score_bins()

        # setup input data
        sev, ssc, n_events = self.get_sensitive_divided_data()

        # initialize coefficients
        optimizer_kwargs = self.optimizer_kwargs.copy()
        optimizer_kwargs['method'] = 'CG'
        self._init_coef(
            data, sev, ssc, self.n_objects,  **optimizer_kwargs)

        # check optimization parameters
        optimizer_kwargs = self.optimizer_kwargs.copy()
        optimizer_method = optimizer_kwargs.pop('method', 'Powell')
        if optimizer_method != "Powell":
            logger.info("Optimizer is changed to a Powell method")
            optimizer_method = 'Powell'

        # get final loss
        self.fit_results_['initial_loss'] = self.loss(
            self._coef, sev, ssc, self.n_objects, score_bins)

        # optimize model
        res = minimize(
            fun=self.loss,
            x0=self._coef,
            args=(sev, ssc, self.n_objects, score_bins),
            method=optimizer_method,
            **optimizer_kwargs)

        # get parameters
        self._coef[:] = res.x

        # add parameters for unknown users and items
        self._add_fallback_parameters()

        # store fitting results
        self.fit_results_['n_users'] = self.n_objects[0]
        self.fit_results_['n_items'] = self.n_objects[1]
        self.fit_results_['n_events'] = self.n_events
        self.fit_results_['n_sensitives'] = self.n_sensitives
        self.fit_results_['n_sensitive_values'] = self.n_sensitive_values
        self.fit_results_['n_parameters'] = self._coef.size
        self.fit_results_['success'] = res.success
        self.fit_results_['status'] = res.status
        self.fit_results_['message'] = res.message
        self.fit_results_['final_loss'] = res.fun
        self.fit_results_['n_iterations'] = res.nit
        self.fit_results_['func_calls'] = res.nfev
        self.fit_results_['optimizer_method'] = optimizer_method
        self.fit_results_['optimizer_kwargs'] = optimizer_kwargs

        # clean up temporary instance variables
        self.remove_data()


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
