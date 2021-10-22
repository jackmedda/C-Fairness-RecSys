#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    print_function,
    division,
    absolute_import)

# =============================================================================
# Imports
# =============================================================================

from numpy.testing import (
    TestCase,
    run_module_suite,
    assert_allclose,
    assert_array_equal,
    assert_equal)
import numpy as np

import os

from kamrecsys.datasets import load_movielens_mini

# =============================================================================
# Module variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================


def load_test_data(score_domain=(1.0, 5.0, 0.5)):
    from kamrecsys.data import EventWithScoreData
    from kamrecsys.datasets import SAMPLE_PATH

    infile = os.path.join(SAMPLE_PATH, 'pci.event')
    dtype = np.dtype([('event', 'U18', 2), ('score', float)])
    x = np.genfromtxt(fname=infile, delimiter='\t', dtype=dtype)
    data = EventWithScoreData(n_otypes=2, event_otypes=np.array([0, 1]))
    data.set_event(x['event'], x['score'], score_domain=score_domain)
    return data, x


# =============================================================================
# Test Classes
# =============================================================================


class TestEventWithScoreData(TestCase):

    def test_set_event(self):
        data, x = load_test_data()

        # test info related to scores
        assert_allclose(data.score[:5], [3., 4., 3.5, 5., 3.])
        assert_allclose(data.score_domain, [1.0, 5.0, 0.5])
        self.assertEqual(data.n_score_levels, 9)

        # estimating score_domain
        data, x = load_test_data(score_domain=None)
        assert_allclose(data.score[:5], [3., 4., 3.5, 5., 3.])
        assert_allclose(data.score_domain, [1.0, 5.0, 0.5])
        self.assertEqual(data.n_score_levels, 9)

    def test_generate_score_bins(self):
        data, x = load_test_data()

        bins = data.generate_score_bins()
        assert_allclose(
            bins,
            [-np.inf, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, np.inf])

    def test_digitize_score(self):
        data, x = load_test_data()

        digitized_scores = data.digitize_score()
        assert_array_equal(digitized_scores[:5], [4, 6, 5, 8, 4])
        assert_array_equal(digitized_scores[-5:], [4, 3, 4, 5, 6])

        digitized_scores = data.digitize_score(np.linspace(1.0, 5.0, 9))
        assert_array_equal(digitized_scores, np.arange(9))

    def test_filter_event(self):
        data = load_movielens_mini()

        filtered_data = data.filter_event(data.score > 3)
        assert_allclose(
            filtered_data.score,
            [4., 4., 4., 5., 4., 5., 5., 5., 4., 5.,
             4., 5., 5., 4., 5., 4., 4., 4., 4., 4., 4.])

        assert_allclose(
            filtered_data.to_eid(0, filtered_data.event[:, 0]),
            [10, 5, 10, 1, 7, 7, 9, 7, 10, 1,
             2, 1, 7, 6, 7, 6, 1, 9, 1, 6, 10])

    def test_get_score_levels(self):
        data = load_movielens_mini()

        score_levels = data.get_score_levels()
        assert_allclose(score_levels, [1., 2., 3., 4., 5])

    def test_binarize_score(self):
        data = load_movielens_mini()

        data.binarize_score(2)
        assert_array_equal(data.score_domain, [0, 1, 1])
        assert_array_equal(
            data.score,
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        assert_equal(data.n_score_levels, 2)

        data = load_movielens_mini()
        data.binarize_score()
        assert_array_equal(
            data.score,
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1,
             0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1])


# =============================================================================
# Main Routines
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
