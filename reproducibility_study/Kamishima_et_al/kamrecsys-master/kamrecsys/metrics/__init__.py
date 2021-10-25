#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation Metrics
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

from .base import (
    generate_score_bins,
    statistics_mean)
from .real import (
    mean_absolute_error,
    mean_squared_error,
    score_histogram,
    variance_with_gamma_prior)
from .score_predictor import (
    score_predictor_report,
    score_predictor_statistics)
from .item_finder import (
    item_finder_report,
    item_finder_statistics)

# =============================================================================
# Metadata variables
# =============================================================================

# =============================================================================
# Public symbols
# =============================================================================

__all__ = [
    'generate_score_bins',
    'statistics_mean',
    'mean_absolute_error',
    'mean_squared_error',
    'score_histogram',
    'variance_with_gamma_prior',
    'score_predictor_report',
    'score_predictor_statistics',
    'item_finder_report',
    'item_finder_statistics']

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
