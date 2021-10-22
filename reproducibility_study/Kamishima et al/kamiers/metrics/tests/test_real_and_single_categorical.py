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
import numpy as np

# =============================================================================
# Variables
# =============================================================================

x = (3.22461582936, 2.99218086243, 3.9966970832, 4.37996292304, 4.31471229859,
     3.50467566847, 4.21731218391, 4.51771680394, 3.8497338358, 4.51715982848)
s = (1, 1, 1, 1, 1, 1, 0, 1, 0, 1)

# =============================================================================
# Functions
# =============================================================================


def test_KS_statistic():
    from kamiers.metrics import KS_statistic

    statistic = KS_statistic(x, s)
    assert_allclose(statistic, 0.5, rtol=1e-5)

    statistic, p_value = KS_statistic(x, s, full_output=True)
    assert_allclose(statistic, 0.5, rtol=1e-5)
    assert_allclose(p_value, 0.65087283170400867, rtol=1e-5)

    statistic, p_value = KS_statistic([1, 1, 1, 5], [0, 0, 1, 1],
        full_output=True)
    assert_allclose(statistic, 0.5, rtol=1e-5)
    assert_allclose(p_value, 0.84381982454156057, rtol=1e-5)

    statistic, p_value = KS_statistic([1, 3, 1, 5], [0, 0, 1, 1],
        full_output=True)
    assert_allclose(statistic, 0.5, rtol=1e-5)
    assert_allclose(p_value, 0.84381982454156057, rtol=1e-5)

    statistic, p_value = KS_statistic([1, 3, 1, 2, 3, 4, 5],
        [0, 0, 1, 1, 1, 1, 1], full_output=True)
    assert_allclose(statistic, 0.40000000000000002, rtol=1e-5)
    assert_allclose(p_value, 0.90927457092277386, rtol=1e-5)


def test_CDF_difference():
    from kamiers.metrics import CDF_difference

    statistic = CDF_difference(x, s)
    assert_allclose(statistic, 0.232295044302912)

    statistic, area, width = CDF_difference(x, s, full_output=True)
    assert_allclose(statistic, 0.232295044302912)
    assert_allclose(area, 0.35437443911875)
    assert_allclose(width, 1.52553594151)

    statistic = CDF_difference([1, 1, 1, 5], [0, 0, 1, 1])
    assert_allclose(statistic, 0.5)

    statistic = CDF_difference([1, 3, 1, 5], [0, 0, 1, 1])
    assert_allclose(statistic, 0.25)

    statistic = CDF_difference([1, 3, 1, 2, 3, 4, 5], [0, 0, 1, 1, 1, 1, 1])
    assert_allclose(statistic, 0.25)

    statistic = CDF_difference(
        [0.1, 0.4, 0.9, 0.2, 0.3, 0.5, 0.8], [0, 0, 0, 1, 1, 1, 1])
    assert_allclose(statistic, 0.166666666666667)

    statistic = CDF_difference([0.1, 0.4, 0.4, 0.5, 0.7], [0, 0, 0, 1, 1])
    assert_allclose(statistic, 0.5)

    statistic = CDF_difference([0.5, 0.5], [0, 1])
    assert_allclose(statistic, 0.0)

    statistic, area, width = CDF_difference([0.5, 0.5], [0, 1], full_output=True)
    assert_allclose(statistic, 0.0)
    assert_allclose(area, 0.0)
    assert_allclose(width, 0.0)

    statistic = CDF_difference(
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.7], [0, 0, 0, 0, 0, 1])
    assert_allclose(statistic, 1.0)

    with assert_raises(ValueError):
        CDF_difference([0.5, 0.5], [1, 1])


def test_chi2_statistic():
    from kamiers.metrics import chi2_statistic

    statistic = chi2_statistic(x, s, score_domain=(1, 5, 4))
    assert_allclose(statistic, 0.625, rtol=1e-5)

    statistic, p_value, dof = chi2_statistic(
        x, s, score_domain=(1, 5, 4), full_output=True)
    assert_allclose(statistic, 0.625, rtol=1e-5)
    assert_allclose(p_value, 0.42919530044034926, rtol=1e-5)
    assert_equal(dof, 1)

    statistic, p_value, dof = chi2_statistic(
        [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        score_domain=(1, 5, 2), full_output=True)
    assert_allclose(statistic, 0.0, rtol=1e-5)
    assert_allclose(p_value, 1.0, rtol=1e-5)
    assert_equal(dof, 2)

    statistic, p_value, dof = chi2_statistic(
        [1, 2, 3, 4, 5, 1, 2, 4, 4, 5, 1, 2, 2, 4, 5],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        score_domain=(2, 4, 1), full_output=True)
    assert_allclose(statistic, 2.1428571428571432, rtol=1e-5)
    assert_allclose(p_value, 0.34251885509304542, rtol=1e-5)
    assert_equal(dof, 2)


def test_histogram_normalized_mutual_information():
    from kamiers.metrics import histogram_normalized_mutual_information

    gmean = histogram_normalized_mutual_information(x, s)
    assert_allclose(gmean, 0.17183544089285782, rtol=1e-5)

    (mi, mi_p_hy, mi_p_hs, amean, gmean, hmean) = (
        histogram_normalized_mutual_information(x, s, full_output=True))
    assert_allclose(mi, 0.1184939225613002, rtol=1e-5)
    assert_allclose(mi_p_hy, 0.12469493440984918, rtol=1e-5)
    assert_allclose(mi_p_hs, 0.23679725954056538, rtol=1e-5)
    assert_allclose(amean, 0.18074609697520727, rtol=1e-5)
    assert_allclose(gmean, 0.17183544089285782, rtol=1e-5)
    assert_allclose(hmean, 0.16336407391907928, rtol=1e-5)

    (mi, mi_p_hy, mi_p_hs, amean, gmean, hmean) = (
        histogram_normalized_mutual_information(
            x, s, score_domain=(1, 5, 4), sensitive_values=(0, 1),
            full_output=True))
    assert_allclose(mi, 0.023666844386298735, rtol=1e-5)
    assert_allclose(mi_p_hy, 0.0728024729791072, rtol=1e-5)
    assert_allclose(mi_p_hs, 0.04729562302867747, rtol=1e-5)
    assert_allclose(amean, 0.06004904800389234, rtol=1e-5)
    assert_allclose(gmean, 0.05867911312873887, rtol=1e-5)
    assert_allclose(hmean, 0.057340431397882335, rtol=1e-5)

    (mi, mi_p_hy, mi_p_hs, amean, gmean, hmean) = (
        histogram_normalized_mutual_information(
            [1, 2, 3, 4, 5, 2, 3, 4], [0, 0, 0, 0, 0, 1, 1, 1],
            full_output=True))
    assert_allclose(mi, 0.1417028527380233, rtol=1e-5)
    assert_allclose(mi_p_hy, 0.090859556855540133, rtol=1e-5)
    assert_allclose(mi_p_hs, 0.21419396448413974, rtol=1e-5)
    assert_allclose(amean, 0.15252676066983994, rtol=1e-5)
    assert_allclose(gmean, 0.13950472642229811, rtol=1e-5)
    assert_allclose(hmean, 0.12759445364664126, rtol=1e-5)


def test_Gaussian_normalized_mutual_information():
    from kamiers.metrics import Gaussian_normalized_mutual_information

    mi = Gaussian_normalized_mutual_information(x, s)
    assert_allclose(mi, 0.130897515566, rtol=1e-5)

    mi, mi_p_hs = Gaussian_normalized_mutual_information(
        x, s, full_output=True)
    assert_allclose(mi, 0.130897515566, rtol=1e-5)
    assert_allclose(mi_p_hs, 0.261584495615, rtol=1e-5)

    mi, mi_p_hs = Gaussian_normalized_mutual_information(
        x, s, a=1e-1, b=1e-2, full_output=True)
    assert_allclose(mi, 0.11511810348070417, rtol=1e-5)
    assert_allclose(mi_p_hs, 0.23005105104555715, rtol=1e-5)

    mi, mi_p_hs = Gaussian_normalized_mutual_information(
        [1, 1, 1, 5], [0, 0, 1, 1], a=1e-1, b=1e-2, full_output=True)
    assert_allclose(mi, 1.3774935279784351, rtol=1e-5)
    assert_allclose(mi_p_hs, 1.9873030816711308, rtol=1e-5)

    mi, mi_p_hs = Gaussian_normalized_mutual_information(
        [1, 3, 1, 5], [0, 0, 1, 1], full_output=True)
    assert_allclose(mi, 0.159226865843, rtol=1e-5)
    assert_allclose(mi_p_hs, 0.229715809729, rtol=1e-5)

    mi, mi_p_hs = Gaussian_normalized_mutual_information(
        [1, 3, 1, 2, 3, 4, 5], [0, 0, 1, 1, 1, 1, 1], full_output=True)
    assert_allclose(mi, 0.0781846776244, rtol=1e-5)
    assert_allclose(mi_p_hs, 0.130684693182, rtol=1e-5)


# =============================================================================
# Test Classes
# =============================================================================

# =============================================================================
# Main Routine
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
