#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data container
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
    ObjectUtilMixin,
    BaseData)
from .event import (
    EventUtilMixin,
    EventData)
from .event_with_score import (
    ScoreUtilMixin,
    EventWithScoreData)

# =============================================================================
# Metadata variables
# =============================================================================

# =============================================================================
# Public symbols
# =============================================================================

__all__ = [
    'BaseData',
    'ObjectUtilMixin',
    'EventData',
    'EventUtilMixin',
    'EventWithScoreData',
    'ScoreUtilMixin']

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
