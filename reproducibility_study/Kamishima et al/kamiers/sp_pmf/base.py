#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base classes for Independence-enhanced kamrecsys.score_predictor.PMF
"""

from __future__ import (
    print_function,
    division,
    absolute_import)
from six.moves import xrange

# =============================================================================
# Imports
# =============================================================================

import logging
from abc import ABCMeta, abstractmethod
import copy

from six import with_metaclass
import numpy as np
from scipy.optimize import minimize

from kamrecsys.score_predictor import PMF

from ..base import BaseIndependentScorePredictorFromSingleBinarySensitive

# =============================================================================
# Module metadata variables
# =============================================================================

# =============================================================================
# Public symbols
# =============================================================================

# =============================================================================
# Constants
# =============================================================================

# =============================================================================
# Module variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Classes
# =============================================================================


class BaseIndependentPMF(
        with_metaclass(
            ABCMeta,
            BaseIndependentScorePredictorFromSingleBinarySensitive)):
    """
    Base class for an independence variant of
    :class:`kamrecsys.score_predictor.PMF` model.

    Parameters
    ----------
    C : float, optional
        regularization parameter (= :math:`\lambda`), default=1.0
    eta : float, optional
        parameter of independence term (= :math:`\eta`),
        default=1.0
    k : int, optional
        the number of latent factors (= sizes of :math:`\mathbf{p}_u` or
        :math:`\mathbf{q}_i`), default=1

    Attributes
    ----------
    mu_ : array_like
        global biases
    bu_ : array_like
        users' biases
    bi_ : array_like
        items' biases
    p_ : array_like
        latent factors of users
    q_ : array_like
        latent factors of items
    fit_results_ : dict
        Side information about results of fitting
    """

    method_name = 'pmf_base'

    def __init__(
            self, C=1.0, eta=1.0, k=1, random_state=None):
        super(BaseIndependentPMF, self).__init__(random_state=random_state)

        # model hyper-parameter
        self.C = float(C)
        self.k = int(k)
        self.eta = float(eta)

        # learned parameters
        self.mu_ = None
        self.bu_ = None
        self.bi_ = None
        self.p_ = None
        self.q_ = None

        self._coef = None
        self._dt = None

    def _init_coef(self, orig_data, sev, ssc, n_objects, **optimizer_kwargs):
        """
        Initialize model parameters

        Parameters
        ----------
        orig_data : kamrecsys.data.EventWithScoreData
            data before separated depending on values of sensitive variables
        sev : array, shape(n_events, 2)
            separated event data
        ssc : array, shape(n_events,)
            scores attached to separated events
        n_objects : array, shape(2,)
            the numbers of users and items
        optimizer_kwargs : keyword arguments, optional
            keyword arguments passed to optimizer
        """

        # constants
        n_users = n_objects[0]
        n_items = n_objects[1]
        k = self.k
        n_s_values = self.n_sensitive_values

        # define dtype for parameters
        self._dt = np.dtype([
            ('mu', float, (1,)),
            ('bu', float, (n_users,)),
            ('bi', float, (n_items,)),
            ('p', float, (n_users, k)),
            ('q', float, (n_items, k))
        ])
        coef_size = 1 + n_users + n_items + n_users * k + n_items * k

        # memory allocation
        self._coef = np.zeros(coef_size * n_s_values, dtype=float)

        # set array's view
        self.mu_ = self._coef.view(self._dt)['mu']
        self.bu_ = self._coef.view(self._dt)['bu']
        self.bi_ = self._coef.view(self._dt)['bi']
        self.p_ = self._coef.view(self._dt)['p']
        self.q_ = self._coef.view(self._dt)['q']

        # init model by normal recommenders
        data = copy.copy(orig_data)
        rec = PMF(
            C=self.C, k=self.k, random_state=self._rng, **optimizer_kwargs)

        for s in xrange(n_s_values):
            data.event = sev[s]
            data.n_events = sev[s].shape[0]
            data.score = ssc[s]
            rec.fit(data)

            self.mu_[s][:] = rec.mu_
            self.bu_[s][:] = rec.bu_[:n_users]
            self.bi_[s][:] = rec.bi_[:n_items]
            self.p_[s][:, :] = rec.p_[:n_users, :]
            self.q_[s][:, :] = rec.q_[:n_items, :]

    def remove_data(self):
        """
        Remove information related to a training dataset
        """
        super(BaseIndependentPMF, self).remove_data()

        self._coef = None
        self._dt = None

    def _add_fallback_parameters(self):
        """
        Add parameters for unknown users and items
        """
        self.bu_ = np.empty(self.n_sensitive_values, dtype=object)
        self.bi_ = np.empty(self.n_sensitive_values, dtype=object)
        self.p_ = np.empty(self.n_sensitive_values, dtype=object)
        self.q_ = np.empty(self.n_sensitive_values, dtype=object)
        for s in xrange(self.n_sensitive_values):
            self.bu_[s] = np.r_[self._coef.view(self._dt)['bu'][s], 0.0]
            self.bi_[s] = np.r_[self._coef.view(self._dt)['bi'][s], 0.0]
            self.p_[s] = np.r_[self._coef.view(self._dt)['p'][s],
                               np.zeros((1, self.k), dtype=float)]
            self.q_[s] = np.r_[self._coef.view(self._dt)['q'][s],
                               np.zeros((1, self.k), dtype=float)]

    def raw_predict(self, ev, sen):
        """
        predict score of given one event represented by internal ids

        Parameters
        ----------
        (user, item) : array_like
            a target user's and item's ids. unknown objects assumed to be
            represented by eid=n_object[event_otype]

        Returns
        -------
        ev : array_like, shape=(s_events,) or (variable, s_events)
            events for which scores are predicted
        sen : int or array_like, dtype=int
            sensitive values to enhance recommendation independence

        Raises
        ------
        TypeError
            shape of an input array is illegal
        """

        return np.array(
            [self.mu_[s][0] + self.bu_[s][ev[i, 0]] + self.bi_[s][ev[i, 1]] +
             np.dot(self.p_[s][ev[i, 0], :], self.q_[s][ev[i, 1], :])
             for i, s in enumerate(sen)])


class BaseIndependentPMFWithOptimizer(
        with_metaclass(ABCMeta, BaseIndependentPMF)):
    """
    A subclass of :class:`BaseIndependent` , which uses numeric optimizers.

    Parameters
    ----------
    C : float, optional
        regularization parameter (= :math:`\lambda`), default=1.0
    eta : float, optional
        parameter of independence term (= :math:`\eta`),
        default=1.0
    k : int, optional
        the number of latent factors (= sizes of :math:`\mathbf{p}_u` or
        :math:`\mathbf{q}_i`), default=1
    optimizer_kwargs : keyword arguments, optional
        keyword arguments passed to optimizer

    Attributes
    ----------
    mu_ : array_like
        global biases
    bu_ : array_like
        users' biases
    bi_ : array_like
        items' biases
    p_ : array_like
        latent factors of users
    q_ : array_like
        latent factors of items
    fit_results_ : dict
        Side information about results of fitting    """

    def __init__(
            self, C=1.0, k=1, eta=1.0, random_state=None, **optimizer_kwargs):
        super(BaseIndependentPMFWithOptimizer, self).__init__(
            C=C, k=k, eta=eta, random_state=random_state)

        # optimizer parameter
        self.optimizer_kwargs = optimizer_kwargs
        optimizer_kwargs['options'] = optimizer_kwargs.get('options', {})
        optimizer_kwargs['options']['disp'] = (
            optimizer_kwargs['options'].get('disp', False))
        opt_maxiter = optimizer_kwargs.pop('maxiter', None)
        if opt_maxiter is not None:
            optimizer_kwargs['options']['maxiter'] = opt_maxiter

    @abstractmethod
    def loss(self, coef, sev, ssc, n_objects):
        """
        loss function to optimize
        """

    @abstractmethod
    def grad_loss(self, coef, sev, ssc, n_objects):
        """
        gradient of loss function
        """

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

        # call super class
        super(BaseIndependentPMFWithOptimizer, self).fit(
            data, sen, event_index=event_index)

        # setup input data
        sev, ssc, n_events = self.get_sensitive_divided_data()

        # initialize coefficients
        self._init_coef(
            data, sev, ssc, self.n_objects, **self.optimizer_kwargs)

        # check optimization parameters
        optimizer_kwargs = self.optimizer_kwargs.copy()
        optimizer_method = optimizer_kwargs.pop('method', 'CG')

        # get final loss
        self.fit_results_['initial_loss'] = self.loss(
            self._coef, sev, ssc, self.n_objects)

        # optimize model
        # fmin_bfgs is slow for large data, maybe because due to the
        # computation cost for the Hessian matrices.
        res = minimize(
            fun=self.loss,
            x0=self._coef,
            args=(sev, ssc, self.n_objects),
            method=optimizer_method,
            jac=self.grad_loss,
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
        self.fit_results_['grad_calls'] = res.njev
        self.fit_results_['optimizer_method'] = optimizer_method
        self.fit_results_['optimizer_kwargs'] = optimizer_kwargs

        # clean up temporary instance variables
        self.remove_data()


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
