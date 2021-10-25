#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Common base classes for independent recommenders  
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
from sklearn.utils import check_array

from kamrecsys.item_finder import BaseExplicitItemFinder

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


class BaseIndependentExplicitItemFinderFromSingleBinarySensitive(
        with_metaclass(ABCMeta, BaseExplicitItemFinder)):
    """
    Recommenders to find good items from event data. The prediction is
    independent from a single binary sensitive feature.

    Attributes
    ----------
    sensitive : array, shape=(n_events,)
        sensitive value
    n_sensitives : int
        the size of vector of sensitive feature. fixed to 1.
    n_sensitive_values : int
        possible values of a sensitive feature. fixed to 2.
    """

    def __init__(self, random_state=None):
        super(
            BaseIndependentExplicitItemFinderFromSingleBinarySensitive,
            self).__init__(random_state=random_state)

        self.sensitive = None
        self.n_sensitives = 1
        self.n_sensitive_values = 2

    def remove_data(self):
        """
        Remove information related to a training dataset
        """
        super(
            BaseIndependentExplicitItemFinderFromSingleBinarySensitive,
            self).remove_data()
        self.sensitive = None

    def get_sensitive(self):
        """
        return sensitive information

        Returns
        -------
        sensitive : array, shape=(n_events,)
            sensitive value
        n_sensitives : int
            the size of vector of sensitive feature. fixed to 1.
        n_sensitive_values : int
            possible values of a sensitive feature. fixed to 2.
        """

        return self.sensitive, self.n_sensitives, self.n_sensitive_values

    def get_sensitive_divided_data(self):
        """
        divide data according to the corresponding sensitive values

        Returns
        -------
        sev : array_like, shape(n_s_values, object)
            array of event arrays divided by the corresponding target values.
        ssc : array_like, shape(n_s_values, object)
            array of score arrays divided by the corresponding target values.
        n_events : array, dtype=int, shape=(n_sensitives,)
            the numbers of events for each sensitive value
        """

        # divide events and scores according to the corresponding target
        # variables
        sev = np.empty(self.n_sensitive_values, dtype=np.object)
        ssc = np.empty(self.n_sensitive_values, dtype=np.object)
        for s in xrange(self.n_sensitive_values):
            sev[s] = self.event.compress(self.sensitive == s, axis=0)
            ssc[s] = self.score.compress(self.sensitive == s)
        n_events = np.array([ev.shape[0] for ev in sev])

        return sev, ssc, n_events

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

        eev = check_array(np.atleast_2d(eev), dtype=int)
        sen = check_sensitive(eev, sen, dtype='binary', accept_sparse=False)
        if sen.ndim != 1:
            raise ValueError('The number of sensitive feature must be one')

        return np.squeeze(self.raw_predict(self.to_iid_event(eev), sen))

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
        super(
            BaseIndependentExplicitItemFinderFromSingleBinarySensitive,
            self).fit(data=data, event_index=event_index)

        # check and arrange sensitive features
        sen = check_sensitive(
            self.event, sen, dtype='binary', accept_sparse=False)
        if sen.ndim != 1:
            raise ValueError('The number of sensitive feature must be one')

        # store sensitive information
        self.sensitive = sen
        self.n_sensitives = 1
        self.n_sensitive_values = 2


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
