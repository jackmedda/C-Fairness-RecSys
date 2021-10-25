#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Item Finders

Find a good items for a user.  Predict a preference score of an item.
A prediction model is trained from from a dataset implicitly rated by users.  
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

from .base import BaseImplicitItemFinder, BaseExplicitItemFinder
from .matrix_factorization import LogisticPMF, ImplicitLogisticPMF

# =============================================================================
# Metadata variables
# =============================================================================

# =============================================================================
# Public symbols
# =============================================================================

__all__ = [
    'BaseExplicitItemFinder',
    'BaseImplicitItemFinder',
    'LogisticPMF',
    'ImplicitLogisticPMF']

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
