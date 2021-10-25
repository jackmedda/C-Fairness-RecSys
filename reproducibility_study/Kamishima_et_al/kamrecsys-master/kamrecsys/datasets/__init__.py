#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sample Data Sets
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
    SAMPLE_PATH,
    event_dtype_timestamp,
    load_event,
    load_event_with_score)
from .flixster import (
    load_flixster_rating)
from .movielens import (
    MOVIELENS100K_INFO,
    load_movielens100k,
    load_movielens_mini,
    MOVIELENS1M_INFO,
    load_movielens1m)
from .others import (
    load_pci_sample)
from .sushi3 import (
    SUSHI3_INFO,
    load_sushi3b_score)

# =============================================================================
# Metadata variables
# =============================================================================

# =============================================================================
# Public symbols
# =============================================================================

__all__ = [
    'SAMPLE_PATH',
    'event_dtype_timestamp',
    'load_event',
    'load_event_with_score',
    'load_flixster_rating',
    'MOVIELENS100K_INFO',
    'load_movielens100k',
    'load_movielens_mini',
    'MOVIELENS1M_INFO',
    'load_movielens1m',
    'load_pci_sample',
    'SUSHI3_INFO',
    'load_sushi3b_score']

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
