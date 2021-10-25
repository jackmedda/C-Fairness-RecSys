#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    print_function,
    division,
    absolute_import,
    unicode_literals)

# =============================================================================
# Imports
# =============================================================================

from numpy.testing import (
    TestCase,
    run_module_suite,
    assert_allclose,
    assert_equal)

from kamrecsys.score_predictor import BaseScorePredictor
from kamrecsys.datasets import load_movielens_mini

# =============================================================================
# Variables
# =============================================================================

true_sc = [
    3., 4., 4., 4., 5., 4., 5., 5., 5., 3.,
    3., 4., 5., 3., 4., 1., 5., 2., 3., 2.,
    5., 4., 5., 3., 4., 4., 4., 4., 4., 4.]

# =============================================================================
# Functions
# =============================================================================


class ScorePredictor(BaseScorePredictor):

    def raw_predict(self):
        pass

# =============================================================================
# Test Classes
# =============================================================================


class TestBaseScorePredictor(TestCase):

    def test_class(self):
        data = load_movielens_mini()
        rec = ScorePredictor()

        # fit()
        rec.fit(data, event_index=(0, 1))

        assert_allclose(rec.score_domain, [1., 5., 1.])
        assert_allclose(rec.score, true_sc)
        assert_equal(rec.n_score_levels, 5)

        # get_score()
        assert_allclose(rec.get_score(), true_sc)

        # remove_data
        rec.remove_data()
        self.assertIsNone(rec.score)
        assert_allclose(rec.score_domain, [1., 5., 1.])
        assert_equal(rec.n_score_levels, 5)


# =============================================================================
# Main Routine
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
