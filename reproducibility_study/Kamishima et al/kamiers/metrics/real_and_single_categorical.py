#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation metrics for real target values and a single categorical sensitive
feature
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
from scipy.stats import (
    gmean, hmean, entropy,
    ks_2samp, chi2_contingency)
from sklearn.utils import as_float_array, assert_all_finite

from kamrecsys.metrics import generate_score_bins
from kamrecsys.metrics import variance_with_gamma_prior as safe_var

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


def KS_statistic(
        y_pred, sen, sensitive_index=(0, 1), full_output=False):
    """
    Kolmogorov-Smirnov statistic on 2 samples to test the contingency of
    distributions between a pair of sensitive groups.

    Fairness metrics for real target variable and a binary sensitive feature.

    Parameters
    ----------
    y_pred : array, shape=(n_samples,)
        predicted values
    sen : array, shape=(n_samples,)
        sensitive 
    sensitive_index : array-like, shape=(2,)
        a pair of sensitive values for specifying sensitive groups to compare.  
        default=(0, 1).
    full_output : bool
        if True, optional statistics are additionally returned

    Returns
    -------
    statistic : float
        Kolmogorov-Smirnov statistic
    p_value : float, optional
        p-value for the hypothesis that two sensitive groups follow the same
        distribution
    """

    # check inputs
    assert_all_finite(y_pred)
    y_pred = as_float_array(y_pred)
    sen = check_sensitive(y_pred, sen, dtype=int)

    statistic, p_value = ks_2samp(
        y_pred[sen == sensitive_index[0]],
        y_pred[sen == sensitive_index[1]])

    if full_output:
        return statistic, p_value
    else:
        return statistic


def CDF_difference(
        y_pred, sen, sensitive_index=(0, 1), full_output=False):
    """
    The area of the difference between cumulative distributions of predicted
    ratings for the events having different sensitive values.

    Fairness metrics for real target variable and a binary sensitive feature.

    Parameters
    ----------
    y_pred : array, shape=(n_samples,)
        predicted values
    sen : array, shape=(n_samples,)
        sensitive
    sensitive_index : array-like, shape=(2,)
        a pair of sensitive values for specifying sensitive groups to compare.
        default=(0, 1).
    full_output : bool
        if True, optional statistics are additionally returned

    Returns
    -------
    statistic : float
        the area of difference between two CDFs. this is normalzed by the
        (max - min) of y_pred
    area : float, optional
        the area of difference between two CDFs
    width : float, optional
        (max - min) of y_pred
    """

    # check inputs
    assert_all_finite(y_pred)
    y_pred = as_float_array(y_pred)
    sen = check_sensitive(y_pred, sen, dtype=int)

    # setup data
    data1 = np.sort(y_pred[sen == sensitive_index[0]])
    data2 = np.sort(y_pred[sen == sensitive_index[1]])
    data_all = np.sort(y_pred)
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    if n1 == 0 or n2 == 0:
        raise ValueError('No data having the specified sensitive value.')

    # get cumulative distributions
    cdf1 = np.searchsorted(data1, data_all, side='right') / n1
    cdf2 = np.searchsorted(data2, data_all, side='right') / n2
    d = np.absolute(cdf1 - cdf2)

    # calculate the area of the difference between two CDFs
    area = 0.0
    for i in xrange(len(data_all) - 1):
        area += (data_all[i + 1] - data_all[i]) * d[i]
    width = data_all[-1] - data_all[0]
    if area == 0.0:
        statistic = 0.0
    else:
        statistic = area / width

    if full_output:
        return statistic, area, width
    else:
        return statistic


def chi2_statistic(
        y_pred, sen, score_domain=(0, 5, 1), sensitive_values=None,
        full_output=False):
    """
    Test the contingency over distinct sensitive groups.
    For this purpose, distributions of expected ratings are modeled by
    histogram models, and their contingency is tested by a Chi-Square test.
   
    Fairness metrics for real target variable and a categorical sensitive
    feature.

    Parameters
    ----------
    y_pred : array, shape=(n_samples,)
        predicted values
    sen : array, shape=(n_samples,)
        sensitive 
    score_domain : array, shape=(3,), optional
        Domain of scores, represented by a triple: start, end, and stride
        default=(1, 5, 1).
    sensitive_values : array-like, ndim=1, optional
        a sorted set of possible sensitive values. If None, automatically
        generated by scanning a sensitive array.
        (default=None)
    full_output : bool
        if True, optional statistics are additionally returned

    Returns
    -------
    statistic : float 
        Chi-square statistic
    p_value : float
        p-value, optional
    dof : int, optional
        degree of freedom
    """

    # check inputs
    assert_all_finite(y_pred)
    y_pred = as_float_array(y_pred)
    sen = check_sensitive(y_pred, sen, dtype=int)

    # get sensitive_values
    if sensitive_values is None:
        sensitive_values = np.unique(sen)
    else:
        sensitive_values = np.asarray(sensitive_values)

    # get bins for predicted values
    y_bins = generate_score_bins(score_domain)

    # get bins for sensitive values
    s_bins = np.hstack([-np.inf,
                        [(sensitive_values[i] + sensitive_values[i + 1]) / 2
                         for i in range(len(sensitive_values) - 1)],
                        np.inf])

    # making histogram
    hist = np.histogram2d(y_pred, sen, bins=(y_bins, s_bins))[0]

    # calc statistics
    statistic, p_value, dof, _ = chi2_contingency(hist)

    if full_output:
        return statistic, p_value, dof
    else:
        return statistic


def histogram_normalized_mutual_information(
        y_pred, sen, score_domain=(1, 5, 1), sensitive_values=None,
        full_output=False):
    """
    Mutual Information between prediction and a sensitive feature. Distribution
    of scores are modeled by a Gaussian distribution

    Fairness metrics for real target variable and a categorical sensitive
    feature.

    Parameters
    ----------
    y_pred : array, shape=(n_samples,)
        predicted values
    sen : array, shape=(n_samples,)
        sensitive
    score_domain : array, shape=(3,), optional
        Domain of scores, represented by a triple: start, end, and stride
        default=(1, 5, 1).
    sensitive_values : array-like, ndim=1, optional
        a sorted set of possible sensitive values. If None, automatically
        generated by scanning a sensitive array.
        (default=None)
    full_output : bool
        if True, optional statistics are additionally returned

    Returns
    -------
    mi : float, optional
        I(score; sensitive)
    mi_p_hy : float, optional
        I(score; sensitive) / H(score)
    mi_p_hs : float, optional
        I(score; sensitive) / H(sensitive)
    amean : float, optional
        arithmetic mean of mi_p_hy and mi_p_hs
    gmean : float
        geometric mean of mi_p_hy and mi_p_hs
    hmean : float, optional 
        harmonic mean of mi_p_hy and mi_p_hs
    """
    # check inputs
    assert_all_finite(y_pred)
    y_pred = as_float_array(y_pred)
    sen = check_sensitive(y_pred, sen, dtype=int)

    # get sensitive_values
    if sensitive_values is None:
        sensitive_values = np.unique(sen)
    else:
        sensitive_values = np.asarray(sensitive_values)

    # entropy of categorical distribution, represented by frequency
    def d_ent(x):
        return entropy((x / x.sum()).ravel())

    # get bins for predicted values
    y_bins = generate_score_bins(score_domain)

    # get bins for sensitive values
    s_bins = np.hstack([-np.inf,
                        [(sensitive_values[i] + sensitive_values[i + 1]) / 2
                         for i in range(len(sensitive_values) - 1)],
                        np.inf])

    # making histogram
    hist = np.histogram2d(y_pred, sen, bins=(y_bins, s_bins))[0]

    # entropy quantities
    ent_y = d_ent(np.sum(hist, axis=1))
    ent_s = d_ent(np.sum(hist, axis=0))
    ent_j = d_ent(hist)

    # unfairness / independence indexes
    mi = np.max((0.0, ent_y + ent_s - ent_j))
    mi_p_hy = 1.0 if ent_y <= 0.0 else mi / ent_y
    mi_p_hs = 1.0 if ent_s <= 0.0 else mi / ent_s

    if full_output:
        return (
            mi, mi_p_hy, mi_p_hs,
            np.mean([mi_p_hy, mi_p_hs]),
            gmean([mi_p_hy, mi_p_hs]),
            hmean([mi_p_hy, mi_p_hs]))
    else:
        return gmean([mi_p_hy, mi_p_hs])


def Gaussian_normalized_mutual_information(
        y_pred, sen, a=1e-8, b=1e-8,
        sensitive_values=None, full_output=False):
    """
    Mutual Information between prediction and a sensitive feature. Distribution
    of scores are modeled by a Gaussian distribution

    Fairness metrics for real target variable and a categorical sensitive
    feature.

    Parameters
    ----------
    y_pred : array, shape=(n_samples,)
        predicted values
    sen : array, shape=(n_samples,)
        sensitive
    a : int or float
        parameter of gamma prior for a Gaussian's variance.
        It should be a << n_samples. (default=1e-8)
    b : int or float
        parameter of gamma prior for a Gaussian's variance.
        It should be b << var(data) n_samples . (default=1e-8)
    sensitive_values : array-like, ndim=1, optional
        a sorted set of possible sensitive values. If None, automatically
        generated by scanning a sensitive array.
        (default=None)
    full_output : bool
        if True, optional statistics are additionally returned

    Attributes
    ----------
    mi : float
        I(score; sensitive)
    mi_p_hs : float
        I(score; sensitive) / H(sensitive)
    """

    # check inputs
    assert_all_finite(y_pred)
    y_pred = as_float_array(y_pred)
    sen = check_sensitive(y_pred, sen, dtype=int)

    # get sensitive_values
    if sensitive_values is None:
        sensitive_values = np.unique(sen)
    else:
        sensitive_values = np.asarray(sensitive_values)

    # entropy of Gaussian distribution: 0.5 log( 2 Pi E sigma^2 )
    def g_ent(x, pa, pb):
        return 0.5 * np.log(2.0 * np.pi * np.e * safe_var(x, pa, pb))

    # entropy of categorical distribution, represented by frequency
    def d_ent(x):
        return entropy((x / x.sum()).ravel())

    # calculate means, variances, and distributions of a target variable
    n = y_pred.size

    # entropy quantities
    # TODO: check coefficients of parameters variance of Gaussian mixture
    ent_y = g_ent(y_pred, a, b)
    ent_y_g_s = 0.0
    sn = np.empty_like(sensitive_values, dtype=int)
    for i, s in enumerate(sensitive_values):
        sy = y_pred[sen == s]
        sn[i] = sy.size
        ent_y_g_s += sn[i] * g_ent(sy, a, b)
    ent_y_g_s /= n
    ent_s = d_ent(sn)

    # unfairness / independence indexes
    mi = np.max((0.0, ent_y - ent_y_g_s))
    mi_p_hs = 1.0 if ent_s <= 0.0 else mi / ent_s

    # set metrics
    if full_output:
        return mi, mi_p_hs
    else:
        return mi


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
