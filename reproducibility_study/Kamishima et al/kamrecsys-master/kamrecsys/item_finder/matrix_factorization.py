#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Matrix Factorization: logistic probabilistic matrix factorization model
"""

from __future__ import (
    print_function,
    division,
    absolute_import)
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
from scipy.optimize import minimize
from sklearn.utils import check_random_state

from . import BaseExplicitItemFinder, BaseImplicitItemFinder
from ..utils import safe_sigmoid as sigmoid

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


class LogisticPMF(BaseExplicitItemFinder):
    """
    A probabilistic matrix factorization model proposed in [1]_.
    A method of handling bias terms is defined by equation (5) in [2]_.
    However, to deal with implicit ratings, a sigmoid function is additionally
    used.
    This class accepts a data of :class:`kamrecsys.data.EventWithScoreData`
    whose scores are binary type.

    Parameters
    ----------
    C : float, optional
        regularization parameter (= :math:`\lambda`), default=1.0
    k : int, optional
        the number of latent factors (= sizes of :math:`\mathbf{p}_u` or
        :math:`\mathbf{q}_i`), default=1
    optimizer_kwargs : keyword arguments, optional
        keyword arguments passed to optimizer

    Attributes
    ----------
    mu_ : array_like
        global bias
    bu_ : array_like
        users' biases
    bi_ : array_like
        items' biases
    p_ : array_like
        latent factors of users
    q_ : array_like
        latent factors of items

    Notes
    -----
    Rating scores are modeled by the sum of bias terms and the cross
    product of users' and items' latent factors.
    To constrain that ratings so as to lie between zero to one, a sigmoid
    function is applied.

    .. math::

        \hat{r}_{xy} =  \sigma(
        \mu + b_x + c_y + \mathbf{p}_x^\top \mathbf{q}_y
        )

    Parameters of this model is estimated by optimizing a cross-entropy loss
    function with L2 regularizer

    .. math::

        \sum_{(x,y)}
        \frac{1}{N_x N_y}
        - \Big( r_{xy} \log \hat{r}_{xy} +
                (1 - r_{xy}) \log(1 - \hat{r}_{xy})\Big)
        + \lambda \Big(
        \|\mathbf{b}\|_2^2 + \|\mathbf{c}\|_2^2 +
        \|\mathbf{P}\|_2^2 + \|\mathbf{Q}\|_2^2
        \Big)

    For computational reasons, a loss term is scaled by the number of
    events, and a regularization term is scaled by the number of model
    parameters.

    References
    ----------
    .. [1] R. Salakhutdinov and A. Mnih. "Probabilistic matrix factorization"
        NIPS2007
    .. [2] Y. Koren, "Factorization Meets the Neighborhood: A Multifaceted
        Collaborative Filtering Model", KDD2008
    """

    def __init__(self, C=1.0, k=1, random_state=None, **optimizer_kwargs):

        super(LogisticPMF, self).__init__(random_state=random_state)

        # model hyper parameter
        self.C = float(C)
        self.k = int(k)

        # optimizer parameter
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_kwargs['options'] = (
            self.optimizer_kwargs.get('options', {}))
        self.optimizer_kwargs['options']['disp'] = (
            self.optimizer_kwargs['options'].get('disp', False))
        opt_maxiter = self.optimizer_kwargs.pop('maxiter', None)
        if opt_maxiter is not None:
            self.optimizer_kwargs['options']['maxiter'] = opt_maxiter

        # learned parameter
        self.mu_ = None
        self.bu_ = None
        self.bi_ = None
        self.p_ = None
        self.q_ = None
        self.fit_results_ = {
            'initial_loss': np.inf,
            'final_loss': np.inf,
        }

        # private instance variables
        self._coef = None
        self._dt = None

    def _init_coef(self, ev, sc, n_objects):
        """
        Initialize model parameters

        Parameters
        ----------
        ev : array, shape(n_events, 2)
            event data
        sc : array, shape(n_events,)
            scores attached to events
        n_objects : array, shape(2,)
            vector of numbers of objects
        """
        # constants
        n_events = ev.shape[0]
        n_users = n_objects[0]
        n_items = n_objects[1]
        k = self.k

        # define dtype for parameters
        self._dt = np.dtype([
            ('mu', float, (1,)),
            ('bu', float, n_users),
            ('bi', float, n_items),
            ('p', float, (n_users, k)),
            ('q', float, (n_items, k))
        ])
        coef_size = 1 + n_users + n_items + n_users * k + n_items * k

        # memory allocation
        self._coef = np.zeros(coef_size, dtype=float)

        # set array's view
        self.mu_ = self._coef.view(self._dt)['mu'][0]
        self.bu_ = self._coef.view(self._dt)['bu'][0]
        self.bi_ = self._coef.view(self._dt)['bi'][0]
        self.p_ = self._coef.view(self._dt)['p'][0]
        self.q_ = self._coef.view(self._dt)['q'][0]

        # set bias term
        self.mu_[0] = np.sum(sc) / n_events
        for i in xrange(n_users):
            j = np.nonzero(ev[:, 0] == i)[0]
            if len(j) > 0:
                self.bu_[i] = np.sum(sc[j] - self.mu_[0]) / len(j)
        for i in xrange(n_items):
            j = np.nonzero(ev[:, 1] == i)[0]
            if len(j) > 0:
                self.bi_[i] = (
                    np.sum(sc[j] - (self.mu_[0] + self.bu_[ev[j, 0]])) /
                    len(j))

        # fill cross terms by normal randoms
        mask = np.bincount(ev[:, 0], minlength=n_users).nonzero()[0]
        self.p_[mask, :] = self._rng.normal(0.0, 1.0, (len(mask), k))
        mask = np.bincount(ev[:, 1], minlength=n_items).nonzero()[0]
        self.q_[mask, :] = self._rng.normal(0.0, 1.0, (len(mask), k))

    def loss(self, coef, ev, sc, n_objects):
        """
        loss function to optimize

        Parameters
        ----------
        coef : array_like, shape=(variable,)
            coefficients of this model
        ev : array_like, shape(n_events, 2), dtype=int
            user and item indexes
        sc : array_like, shape(n_events,), dtype=float
            target scores
        n_objects : array_like, shape(2,), dtype=int
            numbers of users and items

        Returns
        -------
        loss : float
            value of loss function
        """
        # constants
        n_events = ev.shape[0]

        # set array's view
        mu = coef.view(self._dt)['mu'][0]
        bu = coef.view(self._dt)['bu'][0]
        bi = coef.view(self._dt)['bi'][0]
        p = coef.view(self._dt)['p'][0]
        q = coef.view(self._dt)['q'][0]

        # loss term
        esc = sigmoid(
            mu[0] + bu[ev[:, 0]] + bi[ev[:, 1]] +
            np.sum(p[ev[:, 0], :] * q[ev[:, 1], :], axis=1))
        loss = - np.sum(sc * np.log(esc) + (1 - sc) * np.log(1 - esc))

        # regularization term
        reg = (np.sum(bu**2) + np.sum(bi**2) + np.sum(p**2) + np.sum(q**2))

        return loss + 0.5 * self.C * reg

    def grad_loss(self, coef, ev, sc, n_objects):
        """
        gradient of loss function

        Parameters
        ----------
        coef : array_like, shape=(variable,)
            coefficients of this model
        ev : array_like, shape(n_events, 2), dtype=int
            user and item indexes
        sc : array_like, shape(n_events,), dtype=float
            target scores
        n_objects : array_like, shape(2,), dtype=int
            numbers of users and items

        Returns
        -------
        grad : array_like, shape=coef.shape
            the first gradient of loss function by coef
        """
        # constants
        n_events = ev.shape[0]
        n_users = n_objects[0]
        n_items = n_objects[1]

        # set input array's view
        mu = coef.view(self._dt)['mu'][0]
        bu = coef.view(self._dt)['bu'][0]
        bi = coef.view(self._dt)['bi'][0]
        p = coef.view(self._dt)['p'][0]
        q = coef.view(self._dt)['q'][0]

        # create empty gradient
        grad = np.zeros_like(coef)
        grad_mu = grad.view(self._dt)['mu'][0]
        grad_bu = grad.view(self._dt)['bu'][0]
        grad_bi = grad.view(self._dt)['bi'][0]
        grad_p = grad.view(self._dt)['p'][0]
        grad_q = grad.view(self._dt)['q'][0]

        # gradient of loss term
        esc = sigmoid(
            mu[0] + bu[ev[:, 0]] + bi[ev[:, 1]] +
            np.sum(p[ev[:, 0], :] * q[ev[:, 1], :], axis=1))
        common_term = esc - sc

        grad_mu[0] = np.sum(common_term)
        grad_bu[:] = np.bincount(
            ev[:, 0], weights=common_term, minlength=n_users)
        grad_bi[:] = np.bincount(
            ev[:, 1], weights=common_term, minlength=n_items)
        weights = common_term[:, np.newaxis] * q[ev[:, 1], :]
        for i in xrange(self.k):
            grad_p[:, i] = np.bincount(
                ev[:, 0], weights=weights[:, i], minlength=n_users)
        weights = common_term[:, np.newaxis] * p[ev[:, 0], :]
        for i in xrange(self.k):
            grad_q[:, i] = np.bincount(
                ev[:, 1], weights=weights[:, i], minlength=n_items)

        # gradient of regularization term
        grad_bu[:] += self.C * bu
        grad_bi[:] += self.C * bi
        grad_p[:, :] += self.C * p
        grad_q[:, :] += self.C * q

        return grad

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
        """

        # call super class
        super(LogisticPMF, self).fit(data, event_index)

        # get input data
        ev, n_objects = self.get_event()
        sc = self.get_score()

        # initialize coefficients
        self._init_coef(ev, sc, n_objects)

        # check optimization parameters
        optimizer_kwargs = self.optimizer_kwargs.copy()
        optimizer_method = optimizer_kwargs.pop('method', 'CG')

        # get initial loss
        self.fit_results_['initial_loss'] = self.loss(
            self._coef, ev, sc, n_objects)

        # optimize model
        # fmin_bfgs is slow for large data, maybe because due to the
        # computation cost for the Hessian matrices.
        res = minimize(
            fun=self.loss,
            x0=self._coef,
            args=(ev, sc, n_objects),
            method=optimizer_method,
            jac=self.grad_loss,
            **optimizer_kwargs)

        # get parameters
        self._coef[:] = res.x

        # add parameters for unknown users and items
        self.mu_ = self._coef.view(self._dt)['mu'][0].copy()
        self.bu_ = np.r_[self._coef.view(self._dt)['bu'][0], 0.0]
        self.bi_ = np.r_[self._coef.view(self._dt)['bi'][0], 0.0]
        self.p_ = np.r_[self._coef.view(self._dt)['p'][0],
                        np.zeros((1, self.k), dtype=float)]
        self.q_ = np.r_[self._coef.view(self._dt)['q'][0],
                        np.zeros((1, self.k), dtype=float)]

        # store fitting results
        self.fit_results_['n_users'] = n_objects[0]
        self.fit_results_['n_items'] = n_objects[1]
        self.fit_results_['n_events'] = self.n_events
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
        self._coef = None
        self._dt = None
        self._reg = 1.0

    def raw_predict(self, ev):
        """
        predict score of given one event represented by internal ids

        Parameters
        ----------
        (user, item) : array_like
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

        sc = sigmoid(
            self.mu_[0] + self.bu_[ev[:, 0]] + self.bi_[ev[:, 1]] +
            np.sum(self.p_[ev[:, 0], :] * self.q_[ev[:, 1], :], axis=1))

        return sc


class ImplicitLogisticPMF(BaseImplicitItemFinder):
    """
    A probabilistic matrix factorization model proposed in [1]_.
    A method of handling bias terms is defined by equation (5) in [2]_.
    However, to deal with implicit ratings, a sigmoid function is additionally
    used.
    Unlike :class:`kamrecsys.item_finder.LogisticPMF` , this class accepts a
    dataset containing only positive events.  All the other unseen events are
    treated as negative ones.

    Parameters
    ----------
    C : float, optional
        regularization parameter (= :math:`\lambda`), default=1.0
    k : int, optional
        the number of latent factors (= sizes of :math:`\mathbf{p}_u` or
        :math:`\mathbf{q}_i`), default=1
    optimizer_kwargs : keyword arguments, optional
        keyword arguments passed to optimizer

    Attributes
    ----------
    mu_ : array_like
        global bias
    bu_ : array_like
        users' biases
    bi_ : array_like
        items' biases
    p_ : array_like
        latent factors of users
    q_ : array_like
        latent factors of items

    Notes
    -----
    Rating scores are modeled by the sum of bias terms and the cross
    product of users' and items' latent factors.
    To constrain that ratings so as to lie between zero to one, a sigmoid
    function is applied.
    
    .. math::
    
        \hat{r}_{xy} =  \sigma(
        \mu + b_x + c_y + \mathbf{p}_x^\top \mathbf{q}_y
        )
       
    Parameters of this model is estimated by optimizing a cross-entropy loss
    function with L2 regularizer

    .. math::

        \sum_{(x,y)}
        -\frac{1}{N_x N_y}
        \Big( r_{xy} \log \hat{r}_{xy} +
                (1 - r_{xy}) \log(1 - \hat{r}_{xy})\Big) 
        + \lambda \Big(
        \|\mathbf{b}\|_2^2 + \|\mathbf{c}\|_2^2 +
        \|\mathbf{P}\|_2^2 + \|\mathbf{Q}\|_2^2
        \Big)

    For computational reasons, a loss term is scaled by the number of
    events, and a regularization term is scaled by the number of model
    parameters.

    References
    ----------
    .. [1] R. Salakhutdinov and A. Mnih. "Probabilistic matrix factorization"
        NIPS2007
    .. [2] Y. Koren, "Factorization Meets the Neighborhood: A Multifaceted
        Collaborative Filtering Model", KDD2008
    """

    def __init__(self, C=1.0, k=1, random_state=None, **optimizer_kwargs):
        super(ImplicitLogisticPMF, self).__init__(random_state=random_state)

        # model parameter
        self.C = float(C)
        self.k = int(k)

        # optimizer parameter
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_kwargs['options'] = (
            self.optimizer_kwargs.get('options', {}))
        self.optimizer_kwargs['options']['disp'] = (
            self.optimizer_kwargs['options'].get('disp', False))
        opt_maxiter = self.optimizer_kwargs.pop('maxiter', None)
        if opt_maxiter is not None:
            self.optimizer_kwargs['options']['maxiter'] = opt_maxiter

        # learned parameter
        self.mu_ = None
        self.bu_ = None
        self.bi_ = None
        self.p_ = None
        self.q_ = None
        self.fit_results_ = {
            'initial_loss': np.inf,
            'final_loss': np.inf,
        }

        # private instance variables
        self._coef = None
        self._dt = None

    def _init_coef(self, ev, n_objects):
        """
        Initialize model parameters

        Parameters
        ----------
        ev : array, shape(n_events, 2)
            event data
        n_objects : array, shape(2,)
            vector of numbers of objects
        """
        # constants
        n_positives = ev.count_nonzero()
        n_users = n_objects[0]
        n_items = n_objects[1]
        k = self.k

        # define dtype for parameters
        self._dt = np.dtype([
            ('mu', float, (1,)),
            ('bu', float, n_users),
            ('bi', float, n_items),
            ('p', float, (n_users, k)),
            ('q', float, (n_items, k))
        ])

        # memory allocation
        coef_size = 1 + n_users + n_items + n_users * k + n_items * k
        self._coef = np.zeros(coef_size, dtype=float)

        # set array's view
        self.mu_ = self._coef.view(self._dt)['mu'][0]
        self.bu_ = self._coef.view(self._dt)['bu'][0]
        self.bi_ = self._coef.view(self._dt)['bi'][0]
        self.p_ = self._coef.view(self._dt)['p'][0]
        self.q_ = self._coef.view(self._dt)['q'][0]

        # set bias term
        self.mu_[0] = n_positives / (n_users * n_items)
        self.bu_[:] = ev.sum(axis=1).ravel() / n_items
        self.bi_[:] = ev.sum(axis=0).ravel() / n_users

        # fill cross terms by normal randoms
        self.p_[0:n_users, :] = (self._rng.normal(0.0, 1.0, (n_users, k)))
        self.q_[0:n_items, :] = (self._rng.normal(0.0, 1.0, (n_items, k)))

    def loss(self, coef, ev, n_objects):
        """
        loss function to optimize

        Parameters
        ----------
        coef : array_like, shape=(variable,)
            coefficients of this model
        ev : array_like, shape(n_events, 2), dtype=int
            user and item indexes
        n_objects : array_like, shape(2,), dtype=int
            numbers of users and items

        Returns
        -------
        loss : float
            value of loss function
        """
        # constants
        n_users = n_objects[0]
        n_events = n_objects[0] * n_objects[1]

        # set array's view
        mu = coef.view(self._dt)['mu'][0]
        bu = coef.view(self._dt)['bu'][0]
        bi = coef.view(self._dt)['bi'][0]
        p = coef.view(self._dt)['p'][0]
        q = coef.view(self._dt)['q'][0]

        # loss term
        loss = 0.0
        for i in xrange(n_users):
            evi = ev.getrow(i).toarray().reshape(-1)
            esc = sigmoid(
                mu[0] + bu[i] + bi[:] +
                np.sum(p[i, :][np.newaxis, :] * q, axis=1))
            loss = loss - np.sum(
                evi * np.log(esc) + (1 - evi) * np.log(1. - esc))

        # regularization term
        reg = (np.sum(bu**2) + np.sum(bi**2) + np.sum(p**2) + np.sum(q**2))

        return loss + 0.5 * self.C * reg

    def grad_loss(self, coef, ev, n_objects):
        """
        gradient of loss function

        Parameters
        ----------
        coef : array_like, shape=(variable,)
            coefficients of this model
        ev : array_like, shape(n_events, 2), dtype=int
            user and item indexes
        n_objects : array_like, shape(2,), dtype=int
            numbers of users and items

        Returns
        -------
        grad : array_like, shape=coef.shape
            the first gradient of loss function by coef
        """
        # constants
        n_users = n_objects[0]
        n_events = n_objects[0] * n_objects[1]

        # set input array's view
        mu = coef.view(self._dt)['mu'][0]
        bu = coef.view(self._dt)['bu'][0]
        bi = coef.view(self._dt)['bi'][0]
        p = coef.view(self._dt)['p'][0]
        q = coef.view(self._dt)['q'][0]

        # create empty gradient
        grad = np.zeros_like(coef)
        grad_mu = grad.view(self._dt)['mu'][0]
        grad_bu = grad.view(self._dt)['bu'][0]
        grad_bi = grad.view(self._dt)['bi'][0]
        grad_p = grad.view(self._dt)['p'][0]
        grad_q = grad.view(self._dt)['q'][0]

        # gradient of loss term
        for i in xrange(n_users):
            evi = ev.getrow(i).toarray().reshape(-1)
            esc = sigmoid(
                mu[0] + bu[i] + bi[:] +
                np.sum(p[i, :][np.newaxis, :] * q, axis=1))
            common_term = esc - evi

            grad_mu[0] += np.sum(common_term)
            grad_bu[i] = np.sum(common_term)
            grad_bi[:] += common_term
            grad_p[i, :] = np.sum(common_term[:, np.newaxis] * q, axis=0)
            grad_q[:, :] += common_term[:, np.newaxis] * p[i, :][np.newaxis, :]

        # gradient of regularization term
        grad_bu[:] += self.C * bu
        grad_bi[:] += self.C * bi
        grad_p[:, :] += self.C * p
        grad_q[:, :] += self.C * q

        return grad

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
        """

        # call super class
        super(ImplicitLogisticPMF, self).fit(data, event_index)

        # get input data
        ev, n_objects = self.get_event_array('csr')

        # initialize coefficients
        self._init_coef(ev, n_objects)

        # check optimization parameters
        optimizer_kwargs = self.optimizer_kwargs.copy()
        optimizer_method = optimizer_kwargs.pop('method', 'CG')

        # get initial loss
        self.fit_results_['initial_loss'] = self.loss(
            self._coef, ev, n_objects)

        # optimize model
        # fmin_bfgs is slow for large data, maybe because due to the
        # computation cost for the Hessian matrices.
        res = minimize(
            fun=self.loss,
            x0=self._coef,
            args=(ev, n_objects),
            method=optimizer_method,
            jac=self.grad_loss,
            **optimizer_kwargs)

        # get parameters
        self._coef[:] = res.x

        # add parameters for unknown users and items
        self.mu_ = self._coef.view(self._dt)['mu'][0].copy()
        self.bu_ = np.r_[self._coef.view(self._dt)['bu'][0], 0.0]
        self.bi_ = np.r_[self._coef.view(self._dt)['bi'][0], 0.0]
        self.p_ = np.r_[self._coef.view(self._dt)['p'][0],
                        np.zeros((1, self.k), dtype=float)]
        self.q_ = np.r_[self._coef.view(self._dt)['q'][0],
                        np.zeros((1, self.k), dtype=float)]

        # store fitting results
        self.fit_results_['n_users'] = n_objects[0]
        self.fit_results_['n_items'] = n_objects[1]
        self.fit_results_['n_events'] = self.n_events
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
        self._coef = None
        self._dt = None
        self._reg = 1.0

    def raw_predict(self, ev):
        """
        predict score of given one event represented by internal ids

        Parameters
        ----------
        (user, item) : array_like
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

        sc = sigmoid(
            self.mu_[0] + self.bu_[ev[:, 0]] + self.bi_[ev[:, 1]] +
            np.sum(self.p_[ev[:, 0], :] * self.q_[ev[:, 1], :], axis=1))

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
