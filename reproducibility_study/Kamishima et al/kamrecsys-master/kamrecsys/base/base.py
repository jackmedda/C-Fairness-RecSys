#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base Recommender Classes
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
from abc import ABCMeta, abstractmethod

import numpy as np
from six import with_metaclass
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, check_array

from ..data import BaseData, ObjectUtilMixin, EventData, EventUtilMixin

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
# Classes
# =============================================================================


class BaseRecommender(with_metaclass(ABCMeta, BaseEstimator, ObjectUtilMixin)):
    """
    Abstract class for all recommenders

    Attributes
    ----------
    n_otypes : int
        the number of object types, succeed from training data sets
    n_objects : array_like, shape=(n_otypes), dtype=int
        the number of different objects in each type, succeed from training
        data sets
    eid : array_like, shape=(n_otypes,), dtype=(array_like)
        conversion table to external ids, succeed from training data sets
    iid : dictionary
        conversion table to internal ids, succeed from training data sets
    random_state : RandomState or an int seed (None by default)
        A random number generator instance
    fit_results_ : dict
        Side information about results of fitting

    Raises
    ------
    ValueError
        if n_otypes < 1
    """

    def __init__(self, random_state=None):
        self.n_otypes = 0
        self.n_objects = None
        self.eid = None
        self.iid = None

        self.random_state = random_state

        self.fit_results_ = {}

        self._rng = None

    def fit(self, data):
        """
        fitting model

        Parameters
        ----------
        data : :class:`kamrecsys.data.BaseData`
            input data
        """

        # set random state
        self._rng = check_random_state(self.random_state)

        # set object information in data
        self._set_object_info(data)

    def remove_data(self):
        """
        Remove information related to a training dataset
        """
        self._rng = None

    @abstractmethod
    def predict(self, eev):
        """
        abstract method: predict score of given event represented by external
        ids

        Parameters
        ----------
        eev : array_like, shape=(s_event,) or (n_events, s_event)
            events represented by external id

        Returns
        -------
        sc : float or array_like, shape=(n_events,), dtype=float
            predicted scores for given inputs
        """
        pass


class BaseEventRecommender(
        with_metaclass(ABCMeta, BaseRecommender, EventUtilMixin)):
    """
    Recommenders using a data.EventData class or its subclasses
    
    Attributes
    ----------
    s_event : int
        the size of event, which is the number of object types to represent a
        rating event
    event_otypes : array_like, shape=(s_event,)
        see attribute event_otypes. as default, a type of the i-th element of
        each event is the i-th object type.
    event : array_like, shape=(n_events, s_event), dtype=int
        each row is a vector of internal ids that indicates the target of
        rating event
    event_feature : array_like, shape=(n_events, variable), dtype=variable
        i-the row contains the feature assigned to the i-th event
    event_index : array_like, shape=(s_event,)
            a set of indexes to specify the elements in events that are used in
            a recommendation model
    """

    def __init__(self, random_state=None):
        super(BaseEventRecommender, self).__init__(random_state=random_state)

        self.s_event = 0
        self.event_otypes = None
        self.n_events = 0
        self.event = None
        self.event_feature = None

        self.event_index = None

    def get_event(self):
        """
        Returns numbers of objects and an event array
    
        Returns
        -------
        ev : array_like, shape=(n_events, event_index.shape[0])
            an extracted set of events
        n_objects : array_like, shape=(event_index.shape[0],), dtype=int
            the number of objects corresponding to elements tof an extracted
            events
        """

        # get event data
        ev = np.atleast_2d(self.event)

        # get number of objects
        n_objects = self.n_objects

        return ev, n_objects

    def remove_data(self):
        """
        Remove information related to a training dataset
        """
        super(BaseEventRecommender, self).remove_data()

        self.event = None
        self.event_feature = None

    def fit(self, data, event_index=None):
        """
        fitting model

        Parameters
        ----------
        data : :class:`kamrecsys.data.BaseData`
            input data
        event_index : array_like, shape=(variable,)
            a set of indexes to specify the elements in events that are used
            in a recommendation model
        """
        super(BaseEventRecommender, self).fit(data)

        # set object information in data
        self._set_event_info(data)
        if event_index is None:
            self.event_index = np.arange(self.s_event, dtype=int)
        else:
            self.event_index = np.asanyarray(event_index, dtype=int)

        # select information about events used for training
        self.event = self.event.take(self.event_index, axis=1)
        self.n_objects = self.n_objects.take(self.event_otypes)
        self.n_objects = self.n_objects.take(self.event_index)

    @abstractmethod
    def raw_predict(self, ev):
        """
        abstract method: predict score of given one event represented by
        internal ids

        Parameters
        ----------
        ev : array_like, shape=(n_events, s_event)
            events represented by internal id

        Returns
        -------
        sc : float or array_like, shape=(n_events,), dtype=float
            predicted scores for given inputs
        """

    def predict(self, eev):
        """
        predict score of given event represented by external ids

        Parameters
        ----------
        eev : array_like, shape=(s_event,) or (n_events, s_event)
            events represented by external id

        Returns
        -------
        sc : float or array_like, shape=(n_events,), dtype=float
            predicted scores for given inputs
        """

        eev = check_array(np.atleast_2d(eev), dtype=int)[:, self.event_index]

        return np.squeeze(self.raw_predict(self.to_iid_event(eev)))


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
