#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base classes for Independent Generic Score Predictors
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
from abc import ABCMeta

from six import with_metaclass
import numpy as np
from sklearn.base import clone
from sklearn.utils import check_array

from kamrecsys.score_predictor import BaseScorePredictor

from ..base import (
    BaseIndependentScorePredictorFromSingleBinarySensitive
    as BaseIndependentGeneric)
from ..utils import check_sensitive

# =============================================================================
# Metadata variables
# =============================================================================

# =============================================================================
# Public symbols
# =============================================================================

__all__ = []

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


class BaseIndependentScorePredictorFromSingleBinarySensitive(
        with_metaclass(
            ABCMeta, BaseIndependentGeneric)):
    """
    Base class for Generic Score Predictors

    Parameters
    ----------
    base_estimator : :class:`kamrecsys.score_predictor.BaseScorePredictor`
        base estimator to be cloned
    multi_mode : bool
        if `True`, base esttimators are learned for each sub-dataset whose
        sensitive value is 0 and 1; otherwise a single base estimator
    random_state : int, RandomState instance or None, optional
        random seed,  default=None

    Attributes
    ----------
    recommenders_ : array, dtype=`kamrecsys.score_predictor.BaseScorePredictor`
        recommenders
    n_recommenders_ : int
        the number of recommenders
    """

    def __init__(self, base_estimator, multi_mode=True, random_state=None):

        super(
            BaseIndependentScorePredictorFromSingleBinarySensitive,
            self).__init__(random_state=random_state)

        if not isinstance(base_estimator, BaseScorePredictor):
            raise TypeError('Base estimator must be BaseScorePredictor.')

        # set parameters
        self.base_estimator_ = base_estimator
        self.multi_mode = multi_mode

        # set attributes
        self.recommenders_ = None
        self.n_recommenders_ = self.n_sensitive_values if multi_mode else 1

    def _make_estimator(self):
        """
        Make and configure a copy of the `base_estimator_` attribute.
        """

        estimator = clone(self.base_estimator_)

        # set random_seed
        if self.random_state is not None:
            max_rand_seed = np.iinfo(np.int32).max

            to_set = {}
            for key in estimator.get_params(deep=True):
                if key == 'random_state' or key.endswith('__random_state'):
                    to_set[key] = self._rng.randint(max_rand_seed)

            if to_set:
                estimator.set_params(**to_set)

        # copy keyword arguments
        if 'optimizer_kwargs' in dir(estimator):
            estimator.optimizer_kwargs = (
                self.base_estimator_.optimizer_kwargs)

        return estimator

    def _fit_multi(self, data, sen, event_index=(0, 1)):
        """
        fitting model in a multi mode.

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

        super(
            BaseIndependentScorePredictorFromSingleBinarySensitive,
            self).fit(data, sen, event_index=event_index)

        # learn recommender corresponding to each sen value
        self.recommender_ = np.empty(self.n_recommenders_, dtype=object)
        for si in xrange(self.n_recommenders_):
            self.recommender_[si] = self._make_estimator()
            self.recommender_[si].fit(
                data.filter_event(sen == si), event_index)

        # get fitting status
        self.fit_results_['fit_results'] = [
            self.recommender_[si].fit_results_
            for si in xrange(self.n_sensitive_values)]
        self.fit_results_['n_users'] = self.n_objects[0]
        self.fit_results_['n_items'] = self.n_objects[1]
        self.fit_results_['n_events'] = self.n_events
        self.fit_results_['n_sensitives'] = self.n_sensitives
        self.fit_results_['n_sensitive_values'] = self.n_sensitive_values
        self.fit_results_['n_recommenders'] = self.n_recommenders_

        # clean up temporary instance variables
        self.remove_data()

    def _fit_single(self, data, sen, event_index=(0, 1)):
        """
        fitting model in a single mode.

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

        super(
            BaseIndependentScorePredictorFromSingleBinarySensitive,
            self).fit(data, sen, event_index=event_index)

        # learn recommender corresponding to each sen value
        self.recommender_ = self._make_estimator()
        self.recommender_.fit(data, event_index)

        # get fitting status
        self.fit_results_ = self.recommender_.fit_results_.copy()
        self.fit_results_['multi_mode'] = False
        self.fit_results_['n_recommenders'] = self.n_recommenders_

        # clean up temporary instance variables
        self.remove_data()

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

        # check inputs
        eev = check_array(np.atleast_2d(eev), dtype=int)
        sen = check_sensitive(eev, sen, dtype='binary', accept_sparse=False)
        if sen.ndim != 1:
            raise ValueError('The number of sensitive feature must be one')

        # predict scores
        if self.multi_mode:
            sc = np.empty(eev.shape[0], dtype=float)

            for si in xrange(self.n_sensitive_values):
                sev = self.recommender_[si].to_iid_event(
                    np.atleast_2d(eev[sen == si, :]))
                sc[sen == si] = np.squeeze(
                    self.recommender_[si].raw_predict(sev))
        else:
            sc = self.recommender_.raw_predict(
                self.recommender_.to_iid_event(eev))

        return np.squeeze(sc)

    def raw_predict(self, ev, sen):
        """
        predict score of given one event represented by internal ids
        """

        if self.multi_mode:
            sc = np.empty(ev.shape[0], dtype=float)

            for si in xrange(self.n_sensitive_values):
                sev = np.atleast_2d(ev[sen == si, :])
                sc[sen == si] = np.squeeze(
                    self.recommender_[si].raw_predict(sev))
        else:
            sc = self.recommender_.raw_predict(ev)

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
