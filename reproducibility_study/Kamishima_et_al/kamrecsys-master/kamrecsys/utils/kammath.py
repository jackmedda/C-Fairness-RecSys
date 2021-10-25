#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Summary of Mathematical Functions
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

from scipy.special import expit
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

# =============================================================================
# Variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================


def safe_sigmoid(x):
    """
    safe_sigmoid function

    To restrict the range of sigmoid function within [1e-15, 1 - 1e-15],
    domain of inputs is clipped into [-SIGMOID_DOM,+SIGMOID_DOM], where
    SIGMOID_DOM = :math:`log( (1 - 10^{-15}) / 10^{-15})` =
    34.538776394910684

    Parameters
    ----------
    x : array_like, shape=(n_data), dtype=float
        arguments of function

    Returns
    -------
    sig : array, shape=(n_data), dtype=float
        1.0 / (1.0 + exp(- x))
    """
    # import numpy as np
    # from scipy.special import expit

    x = np.clip(x, -34.538776394910684, 34.538776394910684)

    return expit(x)


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
