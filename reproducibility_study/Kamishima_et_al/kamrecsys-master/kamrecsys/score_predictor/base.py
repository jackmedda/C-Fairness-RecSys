#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base class for Score Predictors
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
from six import with_metaclass

from ..base import BaseEventRecommender
from ..data import EventWithScoreData, ScoreUtilMixin

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


class BaseScorePredictor(
        with_metaclass(ABCMeta, BaseEventRecommender, ScoreUtilMixin)):
    """
    Recommenders to predict preference scores from event data

    Attributes
    ----------
    score_domain : array-like, shape=(3,)
        a triple of the minimum, the maximum, and strides of the score
    score : array_like, shape=(n_events,)
        rating scores of each events.
    n_score_levels : int
        the number of score levels
    """

    task_type = 'score_predictor'
    explicit_ratings = True

    def __init__(self, random_state=None):
        super(BaseScorePredictor, self).__init__(
            random_state=random_state)

        # set empty score information
        self.score_domain = None
        self.score = None
        self.n_score_levels = None

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
        super(BaseScorePredictor, self).remove_data()

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
        """
        super(BaseScorePredictor, self).fit(data, event_index)

        # set object information in data
        self._set_score_info(data)


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
