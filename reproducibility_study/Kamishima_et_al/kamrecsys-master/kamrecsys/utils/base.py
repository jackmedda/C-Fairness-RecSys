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

# fitting status message
# compatible with `message` in :class:`scipy.optmize.OptimizeResult`
fit_status_message = {
    'success': 'Fitting terminated successfully.',
    'maxfev': 'Maximum number of function evaluations has been exceeded.',
    'maxiter': 'Maximum number of iterations has been exceeded.',
    'pr_loss': 'Desired error not necessarily achieved due to precision loss.'}

# =============================================================================
# Variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================


def get_fit_status_message(status):
    """
    Status messages of fitting results. These are compatible with
    :class:`scipy.optmize.OptimizeResult`
    
    Parameters
    ----------
    status : int
        fitting status code        

    Returns
    -------
    message : str 
        Error message corresponding to a given status code
    """

    if status == 0:
        message = fit_status_message['success']
    elif status == 1:
        message = fit_status_message['maxfev']
    elif status == 2:
        message = fit_status_message['maxiter']
    elif status == 3:
        message = fit_status_message['pr_loss']
    else:
        message = 'Unknown status code.'

    return message


def is_binary_score(score, allow_uniform=True):
    """
    check

    Parameters
    ----------
    score : array
        array of scores
    allow_uniform : bool
        allow an array containing only one or zero (default=True)

    Returns
    -------
    is_binary
        True if scores consist of 0 and 1 and contain at least one 0 and 1.
    """
    elements = np.unique(score)

    if allow_uniform:
        is_binary = (
            np.array_equal(elements, [0, 1]) or
            np.array_equal(elements, [0]) or
            np.array_equal(elements, [1]))
    else:
        is_binary = np.array_equal(elements, [0, 1])

    return is_binary


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
