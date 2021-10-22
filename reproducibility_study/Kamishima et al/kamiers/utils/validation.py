#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilities for input validation
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

from six import integer_types
import numpy as np
import scipy.sparse as sparse
from sklearn.utils import assert_all_finite

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


def check_sensitive(X, s, dtype=int, accept_sparse=True):
    """
    Check the validity of sensitive values
    
    Parameters
    ----------
    X : array, shape=(n_samples, *), OR int
        the array indicating the number of data; OR
        integer value indicating the number of data.
    s : array, shape=(n_samples,) OR shape=(n_samples, n_sensitives)
        array of sensitive values
    dtype : 'numeric' / 'binary', OR type, OR None; optional
        the type of sen_converted array; OR unchanged if None.
        (default=int)
    accept_sparse : bool; optional
        converted to dense array, if false. If the number of columns of sen is 
        1, the sen is forced to a dense and 1d-array. 
        (default=True)

    Returns
    -------
    s_converted : array
        sensitive information whose shape and dtype are converted as specified.
        if the ndim of sen is 0 or 1, the shape of output is (n_events,).

    Raises
    ------
    ValueError
        * Sizes of X and sen are inconsistent
        * Illegal shape of sen
        * Inconsistent numbers of data
        * Sensitive values are not binary
    """

    # get number of data
    if isinstance(X, int):
        n_samples = X
    elif hasattr(X, 'shape') and X.ndim > 0:
        n_samples = X.shape[0]
    else:
        raise ValueError('Sizes of X and s are inconsistent')

    # ? The s is a sparse matrix with one column, the s is forced to be
    # converted to a dense 1d-array.
    if sparse.issparse(s) and s.shape[1] == 1 and accept_sparse:
        logger.warning("utils.check_sensitive: Converted to a dense 1d-array")
        accept_sparse = False

    # ? If s is sparse and a binary dtype is specified, force to dense array.
    if sparse.issparse(s) and str(dtype) == 'binary':
        logger.warning("utils.check_sensitive: "
                       "Converted to a dense array in a binary-dtype mode")
        accept_sparse = False

    # get dtype of s
    sen_dtype = getattr(s, "dtype", None)
    if not hasattr(sen_dtype, 'kind'):
        # not a data type (e.g. a column named dtype in a pandas DataFrame)
        sen_dtype = None

    # copy dtype of s if dtype is not specified
    if dtype is None and sen_dtype is not None:
        dtype = sen_dtype

    # check binary type
    if str(dtype) == 'binary':
        dtype = int
        is_binary = True
    else:
        is_binary = False

    # check numeric type
    if str(dtype) == 'numeric':
        if sen_dtype is not None and np.issubdtype(sen_dtype, np.number):
            dtype = sen_dtype
        else:
            dtype = float

    # convert the s array
    if sparse.issparse(s):
        if accept_sparse:
            s = s.astype(dtype)
        else:
            s = s.toarray().astype(dtype)
            if s.shape[0] == 1 or s.shape[1] == 1:
                # convert if n_sensitives == 1
                s = s.ravel()
    else:
        s = np.asarray(s, dtype=dtype)
        if s.ndim <= 1:
            s = s.reshape(-1)
        elif s.ndim == 2:
            if s.shape[1] == 1:
                s = s.reshape(-1)
        else:
            raise ValueError('Illegal shape of s')

    # consistency of the number of data
    if s.shape[0] != n_samples:
        raise ValueError('Inconsistent numbers of data')

    # check binary
    if is_binary:
        sv = np.unique(s)
        if not ((sv.shape[0] == 1 and (sv[0] == 0 or sv[0] == 1)) or
                (sv.shape[0] == 2 and sv[0] == 0 and sv[1] == 1)):
            raise ValueError('Sensitive values are not binary')

    return s


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
