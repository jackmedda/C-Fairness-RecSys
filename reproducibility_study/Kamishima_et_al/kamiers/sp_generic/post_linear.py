#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Post processing by a linear transformation.
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

from . import BaseIndependentScorePredictorFromSingleBinarySensitive

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

class IndependentScorePredictor(
        BaseIndependentScorePredictorFromSingleBinarySensitive):
    """
    Standard score predictors

    This merely clone the given base estimators.

    Parameters
    ----------
    use_predicted : bool
        if True , means and standard deviations over predicted ratings
        are used for score modification.  otherwise, those over true ratings
        in a dataset is used.

    Attributes
    ----------
    t_mean_ : float
        means over an an entire dataset
    t_std_ : float
        standard deviations over an entire dataset
    g_mean_ : array, dtype=float, shape=(n_s_values,)
        means over each sensitive group
    g_std_ : array, dtype=float, shape=(n_s_values,)
        standard deviations over each sensitive group

    See Also
    --------

    For the details of parameters, see
    :class:`kamiers.sp_generic.
    BaseIndependentScorePredictorFromSingleBinarySensitive` .
    """

    method_name = 'generic_post_linear'

    def __init__(
            self, base_estimator, use_predicted=True, multi_mode=True,
            random_state=None):

        # call super class
        super(IndependentScorePredictor, self).__init__(
            base_estimator=base_estimator, multi_mode=multi_mode,
            random_state=random_state)

        # attributes
        self.use_predicted = use_predicted
        self.t_mean_ = 0.0
        self.t_std_ = 1.0
        self.g_mean_ = None
        self.g_std_ = None

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

        # fit model
        if self.multi_mode:
            self._fit_multi(data, sen, event_index)
        else:
            self._fit_single(data, sen, event_index)

        # obtain an appropriate score
        if self.use_predicted:
            sc = super(IndependentScorePredictor, self).predict(
                data.to_eid_event(data.event), sen)
        else:
            sc = data.score

        # calculate means and standard deviations
        self.t_mean_ = np.mean(sc)
        self.t_std_ = np.std(sc)
        self.g_mean_ = np.array([
            np.mean(sc[sen == si]) for si in xrange(self.n_sensitive_values)])
        self.g_std_ = np.array([
            np.std(sc[sen == si]) for si in xrange(self.n_sensitive_values)])

        # too small standard deviations are ignored
        EPSILON = 1e-5
        self.t_std_ = -1. if self.t_std_ < EPSILON else self.t_std_
        np.where(self.g_std_ < EPSILON, -1., self.g_std_)


    def _linear_transform_score(self, sc, sen):
        """
        Linear transform scores by

        :math:

            \mathrm{total_SD}
            \frac{\mathrm{score} - \mathrm{group_mean}}{\mathrm{group_SD}} +
            \mathrm{total_mean}

        Parameters
        ----------
        sc : array, dtype=float, shape=(n_samples,)
            original scores
        sen : array, dtype=float, shape=(n_samples,)
            sensitive values

        Returns
        -------
        sc : array, dtype=float, shape=(n_samples,)
            transformed scores
        """

        for si in xrange(self.n_sensitive_values):

            t_mean = self.t_mean_
            t_std = self.t_std_
            g_mean = self.g_mean_[si]
            g_std = self.g_std_[si]
            mask = (sen == si)

            if t_std < 0. or g_std < 0.:
                sc[mask] = sc[mask] - g_mean + t_mean
            else:
                sc[mask] = (sc[mask] - g_mean) * t_std / g_std + t_mean

        return sc

    def predict(self, eev, sen):
        """
        predict score of given event represented by external ids

        Parameters
        ----------
        eev : array_like, shape=(s_event,) or (n_events, s_event)
            events represented by external id
        sen : int or array_like, dtype=int
            target values to enhance recommendation independence

        Returns
        -------
        sc : float or array_like, shape=(n_events,), dtype=float
            predicted scores for given inputs
        """

        # get raw scores
        sc = super(IndependentScorePredictor, self).predict(eev, sen)

        # transform
        sc = self._linear_transform_score(
            np.atleast_1d(sc), np.atleast_1d(sen))

        return np.squeeze(sc)

    def raw_predict(self, ev, sen):
        """
        predict score of given one event represented by internal ids
        """

        # get raw scores
        sc = super(IndependentScorePredictor, self).raw_predict(ev, sen)

        # transform
        sc = self._linear_transform_score(sc, sen)

        return np.squeeze(sc)


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
