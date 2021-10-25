#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base module for Independent MultinomialPLSA
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
from six import with_metaclass
import copy
import numpy as np
from scipy.optimize import fmin_cg
from sklearn.utils import check_random_state

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


class BaseIndependentMultinomialPLSA(
        with_metaclass(
            ABCMeta, BaseIndependentScorePredictorFromSingleBinarySensitive)):
    """
    Base class for a probabilistic matrix factorization model

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
    score_levels : get 
    """

    # class specific constants
    method_name = 'plsam_base'

    def __init__(
            self, k=1, alpha=1.0, use_expectation=True, eta=1.0,
            random_state=None, **optimizer_kwargs):

        super(BaseIndependentMultinomialPLSA, self).__init__(
            random_state=random_state)

        # model hyper-parameter
        self.k = int(k)
        self.alpha = float(alpha)
        self.use_expectation = use_expectation
        self.eta = float(eta)

        # optimizer parameter
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_kwargs['maxiter'] = (
            self.optimizer_kwargs.get('maxiter', 100))
        self.optimizer_kwargs['tol'] = (
            self.optimizer_kwargs.get('tol', 1e-10))

        self.score_levels = None

    def digitize_score(self, data, ssc):
        """

        Parameters
        ----------
        data: class:`kamrecsys.data.EventWithScoreData`
            data to fit
        ssc : array_like, shape(n_s_values, object)
            array of score arrays divided by the corresponding sensitive values

        Returns
        -------
        ssc : array_like, shape(n_s_values, object)
            array of digitized score arrays
        """

        # digitize scores
        for s in xrange(self.n_sensitive_values):
            ssc[s] = data.digitize_score(ssc[s])

        return ssc

    def fit(self, data, sen, event_index=(0, 1)):
        """
        fitting model

        Parameters
        ----------
        data : :class:`kamrecsys.data.BaseData`
            input data
        sen : array-like, (n_events,)
            binary sensitive values
        event_index : array_like, shape=(variable,)
            a set of indexes to specify the elements in events that are used
            in a recommendation model
        """
        super(BaseIndependentMultinomialPLSA, self).fit(
            data, sen, event_index=event_index)

        self.score_levels = self.get_score_levels()

    @abstractmethod
    def raw_predict(self, ev, sensitive):
        pass


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
