#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Load sample Flixster data sets
"""

from __future__ import (
    print_function,
    division,
    absolute_import)
from six.moves import xrange

# =============================================================================
# Imports
# =============================================================================

import sys
import os
import io
import logging
import numpy as np

from ..data import EventWithScoreData
from . import SAMPLE_PATH, load_event_with_score

# =============================================================================
# Public symbols
# =============================================================================

__all__ = []

# =============================================================================
# Constants
# =============================================================================

# =============================================================================
# Module variables
# =============================================================================

# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Functions
# =============================================================================


def load_flixster_rating(infile=None, event_dtype=None):
    """ load the sushi3b score data set

    An original data set is distributed at:
    `Mohsen Jamali <http://www.cs.ubc.ca/~jamalim/datasets/>`_.

    Parameters
    ----------
    infile : optional, file or str
        input file if specified; otherwise, read from default sample directory.
    event_dtype : np.dtype, default=None
        dtype of extra event features

    Returns
    -------
    data : :class:`kamrecsys.data.EventWithScoreData`
        sample data

    Notes
    -----
    Format of events:

    * each event consists of a vector whose format is [user, item].
    * 8,196,077 events in total
    * 147,612 users rate 48,794 items (=movies)
    * dtype=int

    Format of scores:

    * one score is given to each event
    * domain of score is {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0}
    * dtype=float
    """

    # load event file
    if infile is None:
        infile = os.path.join(SAMPLE_PATH, 'flixster.event')
    data = load_event_with_score(
        infile, n_otypes=2, event_otypes=(0, 1),
        score_domain=(0.5, 5.0, 0.5), event_dtype=event_dtype)

    return data


# =============================================================================
# Module initialization
# =============================================================================

# init logging system ---------------------------------------------------------
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
    import doctest

    doctest.testmod()

    sys.exit(0)


# Check if this is call as command script -------------------------------------

if __name__ == '__main__':
    _test()
