#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Independence enhanced kamrecsys.score_predictor.MultinomialPLSA
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

from kamrecsys.utils import get_fit_status_message

from . import BaseIndependentMultinomialPLSA

# =============================================================================
# Metadata variables
# =============================================================================

# =============================================================================
# Public symbols
# =============================================================================

__all__ = ['IndependentScorePredictor']

# =============================================================================
# Constants
# =============================================================================

# =============================================================================
# Variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Classes
# =============================================================================


class IndependentScorePredictor(BaseIndependentMultinomialPLSA):
    """
    Model-based multinomial pLSA recommender

    The model proposed in Figure 2, Type 2 in [1]_ . Variables for users and
    items depend on sensitive and latent variables. A latent variable depends
    on rating and sensitive variables.

    .. math::

        \Pr[x, y, r, s] = \sum_z
        \Pr[x | s, z] \Pr[y | s, z] \Pr[z | r, s] \Pr[r] \Pr[s]

    .. warning::

        Independence parameter :math:`\eta` is ignored.

    Parameters
    ----------
    k : int, optional
        the number of latent factors, default=1
    alpha : float
        Laplace smoothing parameter
    use_expectation : bool, default=True
        use expectation in prediction if True, use mode if False
    eta : float, optional
        independence parameter (= :math:`\eta`),
        default=1.0
    optimizer_kwargs : dict
        keyword arguments passed to optimizer

        * tol (float) : tolerance parameter of conversion, default=1e-10
        * maxiter (int) : maximum number of iterations, default=100

     Attributes
    ----------
    pZgRS_ : array, dtype=float, shape=(n_score_levels, n_s_values, k,)
        Latent distribution: Pr[Z]
    pS_ : array, dtype=float, shape=(n_s_values,)
        Latent distribution: Pr[S]
    pXgSZ_ : array, dtype=float, shape=(n_users, n_s_values, k)
        User distribution: Pr[X | S, Z]
    pYgSZ_ : array, dtype=float, shape=(n_items, n_s_values, k)
        Item distribution: Pr[Y | S, Z]
    pR_ : array, dtype=float, shape=(n_score_levels)
        Rating distribution: Pr[R]

    References
    ----------

    .. [1] T. Kamishima et al. "Model-Based Approaches for Independence-
        Enhanced Recommendation" ICDM Workshop, 2016.
    """

    method_name = 'plsam_model_based_2'

    def __init__(
            self, k=1, alpha=1.0, eta=1.0, random_state=None,
            **optimizer_kwargs):

        super(IndependentScorePredictor, self).__init__(
            k=k, alpha=alpha, use_expectation=True, eta=eta,
            random_state=random_state, **optimizer_kwargs)

        # attributes
        self.pZgRS_ = None
        self.pS_ = None
        self.pXgSZ_ = None
        self.pYgSZ_ = None
        self.pR_ = None

        # internal vars
        self._q = None  # p[z | x, y, r, s]

    def loss(self, sev, ssc):
        """
        negative log-likelihood divided by n_events

        Parameters
        ----------
        sev : array, shape(n_s_values), dtype=array[shape=(n_events/s, 2)]
            event data for each sensitive values
        ssc : array, shape(n_s_values), dtype=array[shape=(n_events/s)]
            digitized scores corresponding to events

        Returns
        -------
        likelihood : float
            negative log-likelihood of current model
        """

        loss = 0.0
        for s in xrange(self.n_sensitive_values):
            ev = sev[s]
            sc = ssc[s]
            prob = (
                self.pS_[s] * self.pR_[sc] *
                np.sum(self.pZgRS_[sc, s, :] * self.pXgSZ_[ev[:, 0], s, :] *
                       self.pYgSZ_[ev[:, 1], s, :], axis=1))
            loss += np.log(prob).sum()

        return -loss / self.n_events

    def _init_params(self, sev, ssc):
        """
        initialize latent variables

        Parameters
        ----------
        sev : array, shape=(2), dtype=array[shape=(n_events/2, 2)]
            event data
        ssc : array, shape(2), dtype=array[shape=(n_events/s)]
            digitized scores corresponding to events
        """

        # Dirichlet parameters
        a = np.empty((self.n_score_levels, self.k), dtype=float)
        for r in xrange(self.n_score_levels):
            for k in xrange(self.k):
                if (k % self.n_score_levels) == r:
                    a[r, k] = 1000.0
                else:
                    a[r, k] = 1.0

        # init latent variables
        self._q = np.empty(self.n_sensitive_values, dtype=object)
        for s in xrange(self.n_sensitive_values):
            ev = sev[s]
            sc = ssc[s]
            n_events = ev.shape[0]

            self._q[s] = np.empty((n_events, self.k), dtype=float)
            for i in xrange(n_events):
                self._q[s][i, :] = self._rng.dirichlet(alpha=a[sc[i], :])

    def maximization_step(self, sev, ssc):
        """
        maximization step

        Parameters
        ----------
        sev : array, shape=(2), dtype=array[shape=(n_events/2, 2)]
            event data
        ssc : array, shape(2), dtype=array[shape=(n_events/s)]
            digitized scores corresponding to events
        """

        self.pXgSZ_ = np.empty(
            (self.n_objects[0], self.n_sensitive_values, self.k), dtype=float)
        self.pYgSZ_ = np.empty(
            (self.n_objects[1], self.n_sensitive_values, self.k), dtype=float)
        self.pZgRS_ = np.empty(
            (self.n_score_levels, self.n_sensitive_values, self.k),
            dtype=float)

        for si in xrange(self.n_sensitive_values):
            ev = sev[si]
            sc = ssc[si]

            # p[x | s, z]
            self.pXgSZ_[:, si, :] = (
                np.array([
                             np.bincount(
                                 ev[:, 0],
                                 weights=self._q[si][:, k],
                                 minlength=self.n_objects[0]
                             ) for k in xrange(self.k)]).T +
                self.alpha)

            # p[y | s, z]
            self.pYgSZ_[:, si, :] = (
                np.array([
                             np.bincount(
                                 ev[:, 1],
                                 weights=self._q[si][:, k],
                                 minlength=self.n_objects[1]
                             ) for k in xrange(self.k)]).T +
                self.alpha)

            # p[z | r, s]
            self.pZgRS_[:, si, :] = (
                np.array([
                             np.bincount(
                                 sc,
                                 weights=self._q[si][:, k],
                                 minlength=self.n_score_levels,
                             ) for k in xrange(self.k)]).T +
                self.alpha)

        # normalize
        self.pXgSZ_ /= self.pXgSZ_.sum(axis=0, keepdims=True)
        self.pYgSZ_ /= self.pYgSZ_.sum(axis=0, keepdims=True)
        self.pZgRS_ /= self.pZgRS_.sum(axis=2, keepdims=True)

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

        # initialization #####
        super(IndependentScorePredictor, self).fit(
            data, sen, event_index=event_index)

        # setup input data
        sev, ssc, n_events = self.get_sensitive_divided_data()
        ssc = self.digitize_score(data, ssc)

        # calc constant parameter
        self.pS_ = n_events + self.alpha
        self.pS_ /= self.pS_.sum()

        self.pR_ = np.asarray([
            np.bincount(ssc[s], minlength=self.n_score_levels)
            for s in xrange(self.n_sensitive_values)]).sum(axis=0) + self.alpha
        self.pR_ /= self.pR_.sum()

        # first m-step
        self._init_params(sev, ssc)
        self.maximization_step(sev, ssc)

        pre_loss = self.loss(sev, ssc)
        self.fit_results_['initial_loss'] = pre_loss
        logger.info("initial: {:.15g}".format(pre_loss))
        self.fit_results_['status'] = 0

        # get optimizer parameters
        maxiter = self.optimizer_kwargs['maxiter']
        tol = self.optimizer_kwargs['tol']

        # main loop
        iter_no = 0
        cur_loss = np.inf
        for iter_no in xrange(maxiter):

            # E-Step
            # p[z | r, y, z]
            for s in xrange(self.n_sensitive_values):
                ev = sev[s]
                sc = ssc[s]
                self._q[s] = (
                    self.pZgRS_[sc, s, :] *
                    self.pXgSZ_[ev[:, 0], s, :] *
                    self.pYgSZ_[ev[:, 1], s, :])
                self._q[s] /= self._q[s].sum(axis=1, keepdims=True)

            # M-step
            self.maximization_step(sev, ssc)

            # check loss
            cur_loss = self.loss(sev, ssc)
            logger.info("iter {:d}: {:.15g}".format(iter_no + 1, cur_loss))
            precision = np.abs((cur_loss - pre_loss) / cur_loss)
            if precision < tol:
                logger.info(
                    "Reached to specified tolerance:"
                    " {:.15g}".format(precision))
                break
            pre_loss = cur_loss

        if iter_no >= maxiter - 1:
            self.fit_results_['status'] = 2
            logger.warning(
                "Exceeded the maximum number of iterations".format(maxiter))

        self.fit_results_['final_loss'] = cur_loss
        logger.info("final: {:.15g}".format(
            self.fit_results_['final_loss']))
        self.fit_results_['n_iterations'] = iter_no + 1
        logger.info("nos of iterations: {:d}".format(
            self.fit_results_['n_iterations']))
        self.fit_results_['n_users'] = self.n_objects[0]
        self.fit_results_['n_items'] = self.n_objects[1]
        self.fit_results_['n_events'] = self.n_events
        self.fit_results_['n_sensitives'] = self.n_sensitives
        self.fit_results_['n_sensitive_values'] = self.n_sensitive_values
        self.fit_results_['n_parameters'] = (
            self.pZgRS_.size + self.pS_.size + self.pXgSZ_.size +
            self.pYgSZ_.size + self.pR_.size)
        self.fit_results_['success'] = (self.fit_results_['status'] == 0)
        self.fit_results_['message'] = get_fit_status_message(
            self.fit_results_['status'])
        self.fit_results_['optimizer_kwargs'] = self.optimizer_kwargs

        # add parameters for unknown users and items
        self.pXgSZ_ = np.r_[
            self.pXgSZ_,
            np.ones((1, self.n_sensitive_values, self.k), dtype=float)]
        self.pYgSZ_ = np.r_[
            self.pYgSZ_,
            np.ones((1, self.n_sensitive_values, self.k), dtype=float)]

        # clean garbage variables
        self._q = None

    def raw_predict(self, ev, sen):
        """
        predict score of given one event represented by internal ids

        Parameters
        ----------
        ev : array_like
            a target user's and item's ids. unknown objects assumed to be
            represented by n_object[event_otype]
        sen : array_like, shape=(n_events,)
            a variable to be independent in recommendation

        Returns
        -------
        sc : float
            score for a target pair of user and item

        Raises
        ------
        TypeError
            shape of an input array is illegal
        """

        pRgXYS = np.sum(
            np.swapaxes(self.pZgRS_[:, sen, :], 0, 1) *
            self.pR_[np.newaxis, :, np.newaxis] *
            self.pXgSZ_[ev[:, 0], sen, :][:, np.newaxis, :] *
            self.pYgSZ_[ev[:, 1], sen, :][:, np.newaxis, :],
            axis=2)
        pRgXYS /= pRgXYS.sum(axis=1, keepdims=True)

        if self.use_expectation:
            sc = np.dot(pRgXYS, self.score_levels[:, np.newaxis])[:, 0]
        else:
            sc = self.score_levels[np.argmax(pRgXYS, axis=1)]

        return sc


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


# Check if this is call as command script

if __name__ == '__main__':
    _test()
