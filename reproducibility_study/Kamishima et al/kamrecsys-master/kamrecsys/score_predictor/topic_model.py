#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Topic Model: probabilistic latent semantic analysis
"""

from __future__ import (
    print_function,
    division,
    absolute_import,
    unicode_literals)
from six.moves import xrange

# =============================================================================
# Module metadata variables
# =============================================================================

# =============================================================================
# Imports
# =============================================================================

import logging
import sys
import numpy as np

from . import BaseScorePredictor
from ..utils import get_fit_status_message

# =============================================================================
# Public symbols
# =============================================================================

__all__ = []

# =============================================================================
# Constants
# =============================================================================

# =============================================================================
# Module variables
# =============================================================================

# =============================================================================
# Classes
# =============================================================================


class MultinomialPLSA(BaseScorePredictor):
    """
    A probabilistic latent semantic analysis model in [1]_ Figure 2(b).

    Parameters
    ----------
    k : int, default=1
        nos of latent factors
    alpha : float, default=1.0
        Laplace smoothing parameter
    use_expectation : bool, default=True
        use expectation in prediction if True, use mode if False
    optimizer_kwargs : dict
        keyword arguments passed to optimizer

        * tol (float) : tolerance parameter of conversion, default=1e-10
        * maxiter (int) : maximum number of iterations, default=100

    Attributes
    ----------
    pZ_ : array_like
        Latent distribution: Pr[Z]
    pXgZ_ : array_like
        User distribution: Pr[X | Z]
    pYgZ_ : array_like
        Item distribution: Pr[Y | Z]
    pRgZ_ : array_like
        Raring distribution: Pr[R | Z]

    Notes
    -----
    3-way topic model: user x item x rating

    .. math::

       \Pr[X, Y, R] = \sum_{Z} \Pr[X | Z] \Pr[Y | Z] \Pr[R | Z] \Pr[Z]

    References
    ----------
    .. [1] T. Hofmann and J. Puzicha. "Latent Class Models for Collaborative
        Filtering", IJCAI 1999
    """

    def __init__(
            self, k=1, alpha=1.0, use_expectation=True,
            random_state=None, **optimizer_kwargs):

        super(MultinomialPLSA, self).__init__(random_state=random_state)

        # parameters
        self.k = k
        self.alpha = alpha
        self.use_expectation = use_expectation
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_kwargs['maxiter'] = (
            self.optimizer_kwargs.get('maxiter', 100))
        self.optimizer_kwargs['tol'] = (
            self.optimizer_kwargs.get('tol', 1e-10))

        # attributes
        self.pZ_ = None
        self.pXgZ_ = None
        self.pYgZ_ = None
        self.pRgZ_ = None
        self.score_levels = None
        self.fit_results_ = {
            'initial_loss': np.inf,
            'final_loss': np.inf,
            'n_iterations': 0
        }

        # internal vars
        self._q = None  # p[z | x, y]

    def loss(self, ev, sc):
        """
        negative log-likelihood

        Parameters
        ----------
        ev : array, shape(n_events, 2)
            event data
        sc : array, shape(n_events,)
            digitized scores corresponding to events

        Returns
        -------
        likelihood : float
            negative log-likelihood of current model
        """

        loss = np.sum(
            self.pZ_[np.newaxis, :] *
            self.pRgZ_[sc, :] *
            self.pXgZ_[ev[:, 0], :] *
            self.pYgZ_[ev[:, 1], :], axis=1)
        loss = -np.sum(np.log(loss)) / self.n_events

        return loss

    def _init_params(self, sc):
        """
        initialize latent variables

        Parameters
        ----------
        sc : array, shape(n_events,)
            digitized scores corresponding to events
        """

        a = np.empty((self.n_score_levels, self.k), dtype=float)
        for r in xrange(self.n_score_levels):
            for k in xrange(self.k):
                if (k % self.n_score_levels) == r:
                    a[r, k] = 1000.0
                else:
                    a[r, k] = 1.0

        self._q = np.empty((self.n_events, self.k), dtype=float)
        for i in xrange(self.n_events):
            self._q[i, :] = self._rng.dirichlet(alpha=a[sc[i], :])

    def maximization_step(self, ev, sc, n_objects):
        """
        maximization step

        Parameters
        ----------
        ev : array, shape(n_events, 2)
            event data
        sc : array, shape(n_events,)
            digitized scores corresponding to events
        n_objects : array, dtype=int, shape=(2,)
            the numbers of users and items
        """

        # p[r | z]
        self.pRgZ_ = (
            np.array([
                         np.bincount(
                             sc,
                             weights=self._q[:, k],
                             minlength=self.n_score_levels
                         ) for k in xrange(self.k)]).T +
            self.alpha)
        self.pRgZ_ /= self.pRgZ_.sum(axis=0, keepdims=True)

        # p[x | z]
        self.pXgZ_ = (
            np.array([
                         np.bincount(
                             ev[:, 0],
                             weights=self._q[:, k],
                             minlength=n_objects[0]
                         ) for k in xrange(self.k)]).T +
            self.alpha)
        self.pXgZ_ /= self.pXgZ_.sum(axis=0, keepdims=True)

        # p[y | z]
        self.pYgZ_ = (
            np.array([
                         np.bincount(
                             ev[:, 1],
                             weights=self._q[:, k],
                             minlength=n_objects[1]
                         ) for k in xrange(self.k)]).T +
            self.alpha)
        self.pYgZ_ /= self.pYgZ_.sum(axis=0, keepdims=True)

        # p[z]
        self.pZ_ = np.sum(self._q, axis=0) + self.alpha
        self.pZ_ /= np.sum(self.pZ_)

    def fit(self, data, event_index=(0, 1)):
        """
        fitting model

        Parameters
        ----------
        data : :class:`kamrecsys.data.EventWithScoreData`
            data to fit
        event_index : optional, array-like, shape=(2,), dtype=int 
            Index to specify the column numbers specifing a user and an item
            in an event array 
            (default=(0, 1))

        Notes
        -----
        * output intermediate results, if the logging level is lower than INFO
        """

        # initialization #####
        super(MultinomialPLSA, self).fit(data, event_index)

        ev, n_objects = self.get_event()
        sc = self.get_score()
        self.score_levels = self.get_score_levels()

        sc = data.digitize_score(sc)
        self._init_params(sc)

        # first m-step
        self.maximization_step(ev, sc, n_objects)

        self.fit_results_['initial_loss'] = self.loss(ev, sc)
        self.fit_results_['status'] = 0
        logger.info("initial: {:.15g}".format(
            self.fit_results_['initial_loss']))
        pre_loss = self.fit_results_['initial_loss']

        # get optimizer parameters
        maxiter = self.optimizer_kwargs['maxiter']
        tol = self.optimizer_kwargs['tol']

        # main loop
        iter_no = 0
        cur_loss = np.inf
        for iter_no in xrange(maxiter):

            # E-Step

            # p[z | r, y, z]
            self._q = (
                self.pZ_[np.newaxis, :] *
                self.pRgZ_[sc, :] *
                self.pXgZ_[ev[:, 0], :] *
                self.pYgZ_[ev[:, 1], :])
            self._q /= (self._q.sum(axis=1, keepdims=True))

            # M-step
            self.maximization_step(ev, sc, n_objects)

            # check loss
            cur_loss = self.loss(ev, sc)
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
                "Exceeded the maximum number of iterations: {:d}".format(
                    maxiter))

        logger.info("final: {:.15g}".format(cur_loss))
        logger.info("nos of iterations: {:d}".format(iter_no + 1))

        # store fitting results
        self.fit_results_['final_loss'] = cur_loss
        self.fit_results_['n_iterations'] = iter_no + 1
        self.fit_results_['n_users'] = n_objects[0]
        self.fit_results_['n_items'] = n_objects[1]
        self.fit_results_['n_events'] = self.n_events
        self.fit_results_['n_parameters'] = (
            self.pZ_.size + self.pXgZ_.size +
            self.pYgZ_.size + self.pRgZ_.size)
        self.fit_results_['success'] = (self.fit_results_['status'] == 0)
        self.fit_results_['message'] = (
            get_fit_status_message(self.fit_results_['status']))
        self.fit_results_['optimizer_kwargs'] = self.optimizer_kwargs

        # add parameters for unknown users and items
        self.pXgZ_ = np.r_[self.pXgZ_, np.ones((1, self.k), dtype=float)]
        self.pYgZ_ = np.r_[self.pYgZ_, np.ones((1, self.k), dtype=float)]

        # clean garbage variables
        self.remove_data()
        del self._q

    def raw_predict(self, ev):
        """
        predict score of given one event represented by internal ids

        Parameters
        ----------
        ev : array_like
            a target user's and item's ids. unknown objects assumed to be
            represented by n_object[event_otype]

        Returns
        -------
        sc : float
            score for a target pair of user and item

        Raises
        ------
        TypeError
            shape of an input array is illegal
        """

        pRgXY = np.sum(
            self.pZ_[np.newaxis, np.newaxis, :] *
            self.pRgZ_[np.newaxis, :, :] *
            self.pXgZ_[ev[:, 0], np.newaxis, :] *
            self.pYgZ_[ev[:, 1], np.newaxis, :], axis=2)
        pRgXY /= pRgXY.sum(axis=1, keepdims=True)

        if self.use_expectation:
            sc = np.dot(pRgXY, self.score_levels[:, np.newaxis])[:, 0]
        else:
            sc = self.score_levels[np.argmax(pRgXY, axis=1)]

        return sc


# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Module initialization
# =============================================================================

# init logging system ---------------------------------------------------------
logger = logging.getLogger('kamrecsys')
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
