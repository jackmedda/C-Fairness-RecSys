#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilities for Recommenders
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
    fit_status_message,
    get_fit_status_message,
    is_binary_score)
from .kammath import safe_sigmoid
from .kamexputils import (
    json_decodable,
    get_system_info,
    get_version_info)

# =============================================================================
# Metadata variables
# =============================================================================

# =============================================================================
# Public symbols
# =============================================================================

__all__ = [
    'fit_status_message',
    'get_fit_status_message',
    'is_binary_score',
    'safe_sigmoid',
    'json_decodable',
    'get_system_info',
    'get_version_info']

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
