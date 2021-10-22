#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Load other sample data sets
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
import logging
import numpy as np

from ..data import EventWithScoreData
from . import SAMPLE_PATH

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


def load_pci_sample(infile=None):
    """ load sample data in "Programming Collective Intelligence"
    
    Parameters
    ----------
    infile : optional, file or str
        input file if specified; otherwise, read from default sample directory.

    Returns
    -------
    data : :class:`kamrecsys.data.EventWithScoreData`
        sample data
    
    Notes
    -----
    Format of events:
    
    * each event consists of a vector whose format is [user, item]
    * 7 users rate 6 items (=movies).
    * 35 events in total
    * dtype=np.dtype('S18')
    
    Format of scores:

    * one score is given to each event
    * domain of score is [1.0, 2.0, 3.0, 4.0, 5.0]
    * dtype=float
    """

    # load event file
    if infile is None:
        infile = os.path.join(SAMPLE_PATH, 'pci.event')
    dtype = np.dtype([('event', 'U18', 2), ('score', float)])
    x = np.genfromtxt(fname=infile, delimiter='\t', dtype=dtype)
    data = EventWithScoreData(n_otypes=2, event_otypes=np.array([0, 1]))
    data.set_event(x['event'], x['score'], score_domain=(1.0, 5.0, 0.5))
    del x

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
