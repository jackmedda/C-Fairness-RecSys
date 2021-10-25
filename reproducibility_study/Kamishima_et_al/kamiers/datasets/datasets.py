#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Load data sets with sensitive information
"""

from __future__ import (
    print_function,
    division,
    absolute_import)
from six.moves import xrange

# =============================================================================
# Imports
# =============================================================================

import logging
import numpy as np

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

# pre-defined event dtype definitions -----------------------------------------

# sensitive feature and timestamp
event_dtype_sensitive = np.dtype([('sensitive', int)])

# sensitive feature and timestamp
event_dtype_sensitive_and_timestamp = np.dtype(
    [('sensitive', int), ('timestamp', int)])

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
