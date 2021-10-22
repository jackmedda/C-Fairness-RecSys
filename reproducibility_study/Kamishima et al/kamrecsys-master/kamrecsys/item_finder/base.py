#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base Class for Item Finders
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

import numpy as np
from scipy import sparse as sparse
from six import with_metaclass

from ..base import BaseEventRecommender
from ..data import EventData, ScoreUtilMixin
from ..utils import is_binary_score

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


class BaseExplicitItemFinder(
        with_metaclass(ABCMeta, BaseEventRecommender, ScoreUtilMixin)):
    """
    Recommenders to predict preference scores from event data

    Attributes
    ----------
    score_domain : tuple, fixed to (0, 1, 1)
        a triple of the minimum, the maximum, and strides of the score
    score : array_like, shape=(n_events,)
        rating scores of each events.
    n_score_levels : int, fixed to 2
        the number of score levels
    """

    task_type = 'item_finder'
    explicit_ratings = True

    def __init__(self, random_state=None):
        super(BaseExplicitItemFinder, self).__init__(random_state=random_state)

        # set empty score information
        self.score_domain = np.array([0, 1, 1], dtype=int)
        self.score = None
        self.n_score_levels = 2

    def get_score(self):
        """
        return score information

        Returns
        -------
        sc : array_like, shape=(n_events,)
            scores for each event
        """

        return self.score

    def remove_data(self):
        """
        Remove information related to a training dataset
        """
        super(BaseExplicitItemFinder, self).remove_data()

        self.score = None

    def fit(self, data, event_index=(0, 1)):
        """
        fitting model

        Parameters
        ----------
        data : :class:`kamrecsys.data.BaseData`
            input data
        event_index : array_like, shape=(variable,)
            a set of indexes to specify the elements in events that are used
            in a recommendation model

        Raises
        ------
        ValueError
            `data.score`, whose domain must be (0, 1, 1), it must consists of
            only 0 or 1, at least one 0 or 1 must be contained, and the
            number of score levels are 2
        """
        super(BaseExplicitItemFinder, self).fit(data, event_index)

        # check whether scores are binary
        if (
                (not np.array_equal(data.score_domain, [0, 1, 1])) or
                (not is_binary_score(data.score, allow_uniform=False)) or
                (data.n_score_levels != 2)):
            raise ValueError('Scores are not binary type')

        # set object information in data
        self._set_score_info(data)


class BaseImplicitItemFinder(with_metaclass(ABCMeta, BaseEventRecommender)):
    """
    Recommenders to find good items from event data
    """

    task_type = 'item_finder'
    explicit_ratings = True

    def get_event_array(self, sparse_type='csr'):
        """
        Set statistics of input dataset, and generate a matrix representing
        implicit feedback.

        Parameters
        ----------
        sparse_type: str
            type of sparse matrix: 'csr', 'csc', 'lil', or 'array'
            default='csr'

        Returns
        -------
        ev : array, shape=(n_users, n_items), dtype=int
            return rating matrix that takes 1 if it is consumed, 0 otherwise.
            if event data are not available, return None
        n_objects : array_like, shape=(event_index.shape[0],), dtype=int
            the number of objects corresponding to elements tof an extracted
            events
        """

        # validity of arguments
        if sparse_type not in ['csr', 'csc', 'lil', 'array']:
            raise TypeError("illegal type of sparse matrices")

        # get number of objects
        n_objects = self.n_objects[self.event_otypes[self.event_index]]

        # get event data
        users = self.event[:, self.event_index[0]]
        items = self.event[:, self.event_index[1]]
        scores = np.ones_like(users, dtype=int)

        # generate array
        ev = sparse.coo_matrix((scores, (users, items)), shape=n_objects)
        if sparse_type == 'csc':
            ev = ev.tocsc()
        elif sparse_type == 'csr':
            ev = ev.tocsr()
        elif sparse_type == 'lil':
            ev = ev.tolil()
        else:
            ev = ev.toarray()

        return ev, n_objects


# =============================================================================
# Module initialization
# =============================================================================

# init logging system
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


# Check if this is call as command script

if __name__ == '__main__':
    _test()
