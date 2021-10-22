#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Summarizers for Item Finders with Sensitive Features
"""

from __future__ import (
    print_function,
    division,
    absolute_import,
    unicode_literals)

# =============================================================================
# Imports
# =============================================================================

import logging
import json
import sys

import numpy as np
from sklearn.utils import (
    as_float_array, assert_all_finite, check_consistent_length)

from kamrecsys.metrics import (
    item_finder_report, item_finder_statistics)

from . import (
    KS_statistic,
    Gaussian_normalized_mutual_information)
from ..utils import check_sensitive

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


def item_finder_and_single_categorical_report(y_true, y_pred, sen, disp=True):
    """
    Belief summary of performance indexes for item finders with a single
    categorical sensitive feature.

    Parameters
    ----------
    y_true : array, shape(n_samples,)
        Ground truth scores
    y_pred : array, shape(n_samples,)
        Predicted scores
    sen : array, shape=(n_samples,)
        sensitive
    disp : bool, optional, default=True
        if True, print report

    Returns
    -------
    stats : dict
        belief summary of prediction performance
    """

    # check inputs
    assert_all_finite(y_true)
    y_true = as_float_array(y_true)
    assert_all_finite(y_pred)
    y_pred = as_float_array(y_pred)
    check_consistent_length(y_true, y_pred)

    sen = check_sensitive(y_true, sen, dtype=int)
    sensitive_values = np.unique(sen)

    # descriptive statistics and accuracy measures
    stats = item_finder_report(y_true, y_pred, disp=False)

    # fairness measures
    stats['KS statistic'] = KS_statistic(y_pred, sen)
    stats['Gaussian NMI'] = Gaussian_normalized_mutual_information(
        y_pred, sen, sensitive_values=sensitive_values)

    # display statistics
    if disp:
        print(
            json.dumps(stats, sort_keys=True, indent=4, separators=(',', ': '),
                       ensure_ascii=False), file=sys.stderr)

    return stats


def item_finder_and_single_categorical_statistics(y_true, y_pred, sen):
    """
    Full summary of performance indexes for item finders with a single
    categorical sensitive feature.

    Parameters
    ----------
    y_true : array, shape=(n_samples,)
        Ground truth scores
    y_pred : array, shape=(n_samples,)
        Predicted scores
    sen : array, shape=(n_samples,)
        sensitive

    Returns
    -------
    stats : dict
        Full statistics of prediction and fairness performance
    """

    # check inputs
    assert_all_finite(y_true)
    y_true = as_float_array(y_true)
    assert_all_finite(y_pred)
    y_pred = as_float_array(y_pred)
    check_consistent_length(y_true, y_pred)

    sen = check_sensitive(y_true, sen, dtype=int)
    sensitive_values = np.unique(sen)

    # descriptive statistics and accuracy measures #####

    # whole dataset
    stats = item_finder_statistics(y_true, y_pred)

    # each sensitive group
    stats['sensitive'] = {}
    for s in sensitive_values:
        stats['sensitive'][s] = item_finder_statistics(
            y_true[sen == s], y_pred[sen == s])

    # Fairness measures #####
    statistic, p_value = KS_statistic(y_pred, sen, full_output=True)
    stats['KS statistic'] = {'statistic': statistic, 'p': p_value}

    mi, mi_p_hs = Gaussian_normalized_mutual_information(
        y_pred, sen, sensitive_values=sensitive_values, full_output=True)
    stats['Gaussian NMI'] = {'mi': mi, 'mi / ent_S': mi_p_hs}

    return stats


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
