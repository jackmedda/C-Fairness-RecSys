#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standard recommenders.
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


    For the details of parameters, see
    :class:`kamiers.sp_generic.
    BaseIndependentScorePredictorFromSingleBinarySensitive` .
    """

    method_name = 'generic_standard'

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

        if self.multi_mode:
            self._fit_multi(data, sen, event_index)
        else:
            self._fit_single(data, sen, event_index)


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
