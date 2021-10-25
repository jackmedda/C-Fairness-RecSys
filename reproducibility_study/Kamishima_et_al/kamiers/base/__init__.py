#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base classes for recommenders
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

from .score_predictor import (
    BaseIndependentScorePredictorFromSingleBinarySensitive)
from .item_finder import (
    BaseIndependentExplicitItemFinderFromSingleBinarySensitive)

# =============================================================================
# Metadata variables
# =============================================================================

# =============================================================================
# Public symbols
# =============================================================================

__all__ = [
    BaseIndependentScorePredictorFromSingleBinarySensitive,
    BaseIndependentExplicitItemFinderFromSingleBinarySensitive]

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
