#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Summary of ModuleName
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

import sys
import logging
import json

import numpy as np

from sklearn.utils import (
    as_float_array, assert_all_finite, check_consistent_length)
import sklearn.metrics as skm

from . import mean_absolute_error
from ..utils import is_binary_score

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


def item_finder_report(y_true, y_pred, disp=True):
    """
    Report brief summary of prediction performance

    * AUC
    * number of data
    * mean and standard dev. of true scores
    * mean and standard dev. of predicted scores

    Parameters
    ----------
    y_true : array, shape(n_samples,)
        Ground truth scores
    y_pred : array, shape(n_samples,)
        Predicted scores
    disp : bool, optional, default=True
        if True, print report

    Returns
    -------
    stats : dict
        belief summary of prediction performance
    """

    # check inputs
    assert_all_finite(y_true)
    if not is_binary_score(y_true):
        raise ValueError('True scores must be binary')
    y_true = as_float_array(y_true)
    assert_all_finite(y_pred)
    y_pred = as_float_array(y_pred)
    check_consistent_length(y_true, y_pred)

    # calc statistics
    stats = {
        'n_samples': y_true.size,
        'true': {'mean': np.mean(y_true), 'stdev': np.std(y_true)},
        'predicted': {'mean': np.mean(y_pred), 'stdev': np.std(y_pred)}}

    # statistics at least 0 and 1 must be contained in a score array
    if is_binary_score(y_true, allow_uniform=False):
        stats['area under the curve'] = skm.roc_auc_score(y_true, y_pred)

    # display statistics
    if disp:
        print(
            json.dumps(
                stats, sort_keys=True, indent=4, separators=(',', ': '),
                ensure_ascii=False), file=sys.stderr)

    return stats


def item_finder_statistics(y_true, y_pred):
    """
    Full Statistics of prediction performance

    * n_samples
    * mean_absolute_error: mean, stdev
    * mean_squared_error: mean, rmse, stdev
    * predicted: mean, stdev
    * true: mean, stdev

    Parameters
    ----------
    y_true : array, shape=(n_samples,)
        Ground truth scores
    y_pred : array, shape=(n_samples,)
        Predicted scores

    Returns
    -------
    stats : dict
        Full statistics of prediction performance
    """

    # check inputs
    assert_all_finite(y_true)
    if not is_binary_score(y_true):
        raise ValueError('True scores must be binary')
    y_true = as_float_array(y_true)
    assert_all_finite(y_pred)
    y_pred = as_float_array(y_pred)
    check_consistent_length(y_true, y_pred)

    # calc statistics
    stats = {}

    # dataset size
    stats['n_samples'] = y_true.size

    # descriptive statistics of ground truth scores
    stats['true'] = {'mean': np.mean(y_true), 'stdev': np.std(y_true)}

    # descriptive statistics of ground predicted scores
    stats['predicted'] = {'mean': np.mean(y_pred), 'stdev': np.std(y_pred)}

    # statistics at least 0 and 1 must be contained in a score array
    if is_binary_score(y_true, allow_uniform=False):

        # AUC (area under the curve)
        stats['area under the curve'] = skm.roc_auc_score(y_true, y_pred)

    return stats


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
