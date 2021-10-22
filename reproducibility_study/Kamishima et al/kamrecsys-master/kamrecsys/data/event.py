#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data model: rating events
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
from copy import copy
from abc import ABCMeta
from six import with_metaclass

import numpy as np

from . import BaseData

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


class EventUtilMixin(with_metaclass(ABCMeta, object)):
    """
    Methods that are commonly used in data containers and recommenders for
    handling events.
    """

    def to_eid_event(self, data):
        """
        convert an event vector or array represented by internal ids to those
        by external ids.

        Parameters
        ----------
        data : array_like
            array whose elements are represented by external ids
        
        Returns
        -------
        new_data : array_like
            array whose elements are represented by external ids
        """
        if data.ndim == 1 and data.shape[0] == self.s_event:
            new_data = np.array(
                [self.eid[self.event_otypes[e]][data[e]]
                 for e in xrange(self.s_event)],
                dtype=self.eid[0].dtype)
        elif data.ndim == 2 and data.shape[1] == self.s_event:
            new_data = np.empty_like(data, dtype=self.eid[0].dtype)
            for e in xrange(self.s_event):
                new_data[:, e] = self.eid[self.event_otypes[e]][data[:, e]]
        else:
            raise TypeError("Shape of input is illegal")

        return new_data

    def to_iid_event(self, ev, missing_values=None):
        """
        convert an event vector or array represented by external ids to those
        by internal ids.

        Parameters
        ----------
        ev : array_like
            array whose elements are represented by external ids
        missing_values : optional, int or array_like, shape=(s_event,)
            if unknown external ids are detected, these will be converted to
            max_id. as default, numbers of possible objects are used.

        Returns
        -------
        new_ev : array_like
            array whose elements are represented by external ids

        Raises
        ------
        TypeError
            Shape of an input array is illegal
        """
        if missing_values is None:
            missing_values = self.n_objects[self.event_otypes]
        if ev.ndim == 1 and ev.shape[0] == self.s_event:
            new_ev = np.array(
                [self.iid[self.event_otypes[e]].get(ev[e], missing_values[e])
                 for e in xrange(self.s_event)], dtype=int)
        elif ev.ndim == 2 and ev.shape[1] == self.s_event:
            new_ev = np.empty_like(ev, dtype=int)
            for e in xrange(self.s_event):
                iid = self.iid[self.event_otypes[e]]
                new_ev[:, e] = [iid.get(i, missing_values[e])
                                for i in ev[:, e]]
        else:
            raise TypeError('The shape of an input is illegal')

        return new_ev

    def _set_event_info(self, data):
        """
        import event meta information of input data to recommenders

        Parameters
        ----------
        data : :class:`kamrecsys.data.EventData`
            input data

        Raises
        ------
        TypeError
            if input data is not :class:`kamrecsys.data.EventData` class
        """
        if not isinstance(data, EventData):
            raise TypeError("input data must data.EventData class")

        self.s_event = data.s_event
        self.event_otypes = data.event_otypes
        self.n_events = data.n_events
        self.event = data.event
        self.event_feature = data.event_feature


class EventData(BaseData, EventUtilMixin):
    """ Container of rating events and their associated features.

    Parameters
    ----------
    n_otypes : optional, int
        see attribute n_otypes (default=2)
    event_otypes : array_like, shape=(variable,), optional
        see attribute event_otypes. as default, a type of the i-th element of
        each event is the i-th object type.

    Attributes
    ----------
    s_event : int
        the size of event, which is the number of objects to represent a rating
        event
    n_events : int
        the number of events
    event : array_like, shape=(n_events, s_event), dtype=int
        each row is a vector of internal ids that indicates the target of
        rating event
    event_feature : array_like, shape=(n_events, variable), dtype=variable
        i-the row contains the feature assigned to the i-th event

    Raises
    ------
    ValueError
        if n_otypes < 1 or event_otypes is illegal.

    See Also
    --------
    :ref:`glossary`
    """

    def __init__(self, n_otypes=2, event_otypes=None):
        super(EventData, self).__init__(n_otypes=n_otypes)

        self.s_event = 0
        self.event_otypes = None
        self.n_events = 0
        self.event = None
        self.event_feature = None

        if event_otypes is None:
            self.s_event = n_otypes
            self.event_otypes = np.arange(self.s_event, dtype=int)
        else:
            event_otypes = np.array(event_otypes)
            if (event_otypes.ndim != 1 or np.min(event_otypes) < 0 or
                    np.max(event_otypes) >= n_otypes):
                raise ValueError("Illegal event_otypes specification")
            self.s_event = event_otypes.shape[0]
            self.event_otypes = event_otypes

    def set_event(self, event, event_feature=None):
        """Set event data from structured array.

        Parameters
        ----------
        event : array_like, shape=(n_events, s_event)
            each row corresponds to an event represented by a vector of object
            with external ids
        event_feature : optional, array_like, shape=(n_events, variable)
            feature of events
        """
        for otype in xrange(self.n_otypes):
            self.n_objects[otype], self.eid[otype], self.iid[otype] = (
                self._gen_id(event[:, self.event_otypes == otype]))

        self.event = np.empty_like(event, dtype=int)
        for e in xrange(self.s_event):
            iid = self.iid[self.event_otypes[e]]
            self.event[:, e] = [iid[i] for i in event[:, e]]

        self.n_events = self.event.shape[0]
        if event_feature is not None:
            self.event_feature = np.asarray(event_feature).copy()
        else:
            self.event_feature = None

    def filter_event(self, filter_cond):
        """
        Returns a copy of data whose events are filtered based on
        `filter_cond` .  Information about the objects that is not contained
        in a filtered event set are eliminated as well.

        Parameters
        ----------
        filter_cond : array, dtype=bool, shape=(n_events,)
            Boolean array that specifies whether each event should be included
            in a new event array.

        Returns
        -------
        data : :class:`kamrecsys.EventData`
            A copy of data whose events are filtered.
        """

        # check whether event info is available
        if self.event is None:
            return

        # re-arrange filter
        filter_cond = np.asarray(filter_cond)

        # copy data
        data = copy(self)

        # generate a copy of filtered events
        data.event = self.event[filter_cond, :].copy()
        data.n_events = data.event.shape[0]

        # update object info and iid's in an event set
        data.n_objects = self.n_objects.copy()
        data.eid = self.eid.copy()
        data.iid = self.iid.copy()
        data.feature = self.feature.copy()
        # data.event_otypes = self.event_otypes.copy()

        for otype in xrange(data.n_otypes):

            # indexes of objects contained in a filtered event set
            sub_index = np.unique(data.event[:, data.event_otypes == otype])

            # update iid's in an event set
            table = self._gen_id_substitution_table(self.eid[otype], sub_index)
            data.event[:, self.event_otypes == otype] = table[
                data.event[:, self.event_otypes == otype]]

            # filter object info
            data.eid[otype] = self.eid[otype][sub_index]
            data.iid[otype] = {k: i for (i, k) in enumerate(data.eid[otype])}
            data.n_objects[otype] = data.eid[otype].shape[0]
            if self.feature[otype] is not None:
                data.feature[otype] = self.feature[otype][sub_index]

        # filter out event features
        if self.event_feature is not None:
            data.event_feature = self.event_feature[filter_cond]

        return data


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
