#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Common definitions of datasets
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
import os

import numpy as np

from ..data import (
    EventData,
    EventWithScoreData)

# =============================================================================
# Public symbols
# =============================================================================

__all__ = []

# =============================================================================
# Constants
# =============================================================================

# path to the directory containing sample files
SAMPLE_PATH = os.path.join(os.path.dirname(__file__), 'data')

# pre-defined event dtype definitions -----------------------------------------

# timestamp
event_dtype_timestamp = np.dtype([('timestamp', int)])

# =============================================================================
# Module variables
# =============================================================================

# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Functions
# =============================================================================


def load_event(infile, n_otypes=2, event_otypes=None, event_dtype=None):
    """
    load event file

    Tab separated file.  The counts of columns are as follows:

    * the first s_events columns are sets of object IDs representing events 
    * the rest of columns corresponds to event features

    Parameters
    ----------
    infile : file or str
        input file if specified; otherwise, read from default sample directory.
    n_otypes : optional, int
        see attribute n_otypes (default=2)
    event_otypes : array_like, shape=(variable,), optional
        see attribute event_otypes. as default, a type of the i-th element of
        each event is the i-th object type.
    event_dtype : np.dtype, default=None
        dtype of extra event features

    Returns
    -------
    data : :class:`kamrecsys.data.EventWithScoreData`
        event data with score information

        event_dtype : np.dtype, default=None
    """

    s_events = n_otypes if event_otypes is None else len(event_otypes)
    if event_dtype is None:
        dtype = np.dtype([('event', int, s_events)])
    else:
        dtype = np.dtype([('event', int, s_events),
                          ('event_feature', event_dtype)])
    x = np.genfromtxt(fname=infile, delimiter='\t', dtype=dtype)

    data = EventData(n_otypes=n_otypes, event_otypes=event_otypes)
    if event_dtype is None:
        event_feature = None
    else:
        event_feature = x['event_feature']
    data.set_event(x['event'], event_feature=event_feature)

    return data


def load_event_with_score(
        infile, n_otypes=2, event_otypes=None, score_domain=(1, 5, 1),
        event_dtype=None):
    """
    load event file with rating score

    Tab separated file.  The counts of columns are as follows:
    
    * the first s_events columns are sets of object IDs representing events 
    * the subsequent one column is a set of scores
    * the rest of columns corresponds to event features
    
    Parameters
    ----------
    infile : file or str
        input file if specified; otherwise, read from default sample directory.
    n_otypes : optional, int
        see attribute n_otypes (default=2)
    event_otypes : array_like, shape=(variable,), optional
        see attribute event_otypes. as default, a type of the i-th element of
        each event is the i-th object type.
    score_domain : optional, tuple or 1d-array of tuple
        min and max of scores, and the interval between scores
    event_dtype : np.dtype, default=None
        dtype of extra event features

    Returns
    -------
    data : :class:`kamrecsys.data.EventWithScoreData`
        event data with score information
    
        event_dtype : np.dtype, default=None
    """

    s_events = n_otypes if event_otypes is None else len(event_otypes)
    if event_dtype is None:
        dtype = np.dtype([('event', int, s_events), ('score', float)])
    else:
        dtype = np.dtype([('event', int, s_events), ('score', float),
                          ('event_feature', event_dtype)])
    x = np.genfromtxt(fname=infile, delimiter='\t', dtype=dtype)

    data = EventWithScoreData(n_otypes=n_otypes, event_otypes=event_otypes)
    if event_dtype is None:
        event_feature = None
    else:
        event_feature = x['event_feature']
    data.set_event(
        x['event'], x['score'], score_domain=score_domain,
        event_feature=event_feature)

    return data


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
