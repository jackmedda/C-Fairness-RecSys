#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    print_function,
    division,
    absolute_import,
    unicode_literals)
from six.moves import xrange

# =============================================================================
# Imports
# =============================================================================

from numpy.testing import (
    TestCase,
    run_module_suite,
    assert_,
    assert_allclose,
    assert_array_almost_equal_nulp,
    assert_array_max_ulp,
    assert_array_equal,
    assert_array_less,
    assert_equal,
    assert_raises,
    assert_raises_regex,
    assert_warns,
    assert_string_equal)

# =============================================================================
# Variables
# =============================================================================

y_true = [4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 3, 3, 4, 4, 5, 2, 3, 3, 5]
y_pred = [
    3.57693181, 3.74035694, 4.05667626, 4.11208204, 3.90311352,
    3.45602985, 3.45850967, 3.62878891, 3.44663680, 3.90525169,
    3.83583531, 3.57535821, 3.36962744, 3.88604703, 3.64760914,
    4.28210103, 2.36221703, 3.57729091, 3.37095081, 3.80150211]
sen = [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1]

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Test Classes
# =============================================================================


def test_score_predictor_report():
    from kamiers.metrics import (
        score_predictor_and_single_categorical_report)

    stats = score_predictor_and_single_categorical_report(
        y_true, y_pred, sen, (1, 5, 4), disp=False)

    assert_allclose(stats['KS statistic'], 0.7142857142857143, rtol=1e-5)
    assert_allclose(stats['CDF difference'], 0.2193725192573639, rtol=1e-5)
    assert_allclose(stats['Histogram NMI'], 0.19676162220943524, rtol=1e-5)
    assert_allclose(stats['Gaussian NMI'], 0.20976290212377685, rtol=1e-5)


def test_score_predictor_statistics():
    from kamiers.metrics import (
        score_predictor_and_single_categorical_statistics)

    stats = score_predictor_and_single_categorical_statistics(
        y_true, y_pred, sen, score_domain=(1, 5, 4))

    sub_stats = stats['KS statistic']
    assert_allclose(
        sub_stats['p'], 0.012910336137009088, rtol=1e-5)
    assert_allclose(
        sub_stats['statistic'], 0.7142857142857143, rtol=1e-5)

    sub_stats = stats['CDF difference']
    assert_allclose(
        sub_stats['width'], 1.9198839999999997, rtol=1e-5)
    assert_allclose(
        sub_stats['area'], 0.4211697897619047, rtol=1e-5)
    assert_allclose(
        sub_stats['statistic'], 0.2193725192573639, rtol=1e-5)

    sub_stats = stats['Histogram NMI']
    assert_allclose(
        sub_stats['mi'], 0.13282862876456347, rtol=1e-5)
    assert_allclose(
        sub_stats['mi / ent_Y'], 0.17804666611393627, rtol=1e-5)
    assert_allclose(
        sub_stats['mi / ent_S'], 0.217443756850318, rtol=1e-5)
    assert_allclose(
        sub_stats['amean'], 0.19774521148212715, rtol=1e-5)
    assert_allclose(
        sub_stats['gmean'], 0.19676162220943524, rtol=1e-5)
    assert_allclose(
        sub_stats['hmean'], 0.19578292533262032, rtol=1e-5)

    sub_stats = stats['Gaussian NMI']
    assert_allclose(
        sub_stats['mi'], 0.20976290212377685, rtol=1e-5)
    assert_allclose(
        sub_stats['mi / ent_S'], 0.3433870688107866, rtol=1e-5)

    sub_stats = stats['sensitive'][0]['mean absolute error']
    assert_allclose(
        sub_stats['mean'], 0.25956266833333336, rtol=1e-5)
    assert_allclose(
        sub_stats['stdev'], 0.2225632026701066, rtol=1e-5)

    assert_equal(stats['sensitive'][0]['n_samples'], 6)

    sub_stats = stats['sensitive'][0]['predicted']
    assert_allclose(
        sub_stats['histogram density'], [0., 1.], rtol=1e-5)
    assert_allclose(
        sub_stats['mean'], 3.944464678333333, rtol=1e-5)
    assert_allclose(
        sub_stats['stdev'], 0.20295318677585414, rtol=1e-5)

    assert_allclose(
        stats['sensitive'][0]['score levels'], [1., 5.], rtol=1e-5)

    sub_stats = stats['sensitive'][0]['true']
    assert_array_equal(sub_stats['histogram'], [0, 6])
    assert_allclose(
        sub_stats['mean'], 4.166666666666667, rtol=1e-5)
    assert_allclose(
        sub_stats['stdev'], 0.37267799624996495, rtol=1e-5)

    sub_stats = stats['sensitive'][1]['mean squared error']
    assert_allclose(
        sub_stats['mean'], 0.2896729614499542, rtol=1e-5)
    assert_allclose(
        sub_stats['rmse'], 0.538212747387085, rtol=1e-5)
    assert_allclose(
        sub_stats['stdev'], 0.3406136961938204, rtol=1e-5)

    assert_equal(stats['sensitive'][1]['n_samples'], 14)

    sub_stats = stats['sensitive'][1]['predicted']
    assert_array_equal(sub_stats['histogram'], [1, 13])

    assert_allclose(
        stats['sensitive'][1]['score levels'], [1., 5.], rtol=1e-5)

    sub_stats = stats['sensitive'][1]['true']

    assert_allclose(
        sub_stats['histogram density'],
        [0.07142857142857142, 0.9285714285714286], rtol=1e-5)
    assert_allclose(
        sub_stats['mean'], 3.5714285714285716, rtol=1e-5)
    assert_allclose(
        sub_stats['stdev'], 0.7284313590846836, rtol=1e-5)


# =============================================================================
# Main Routine
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
