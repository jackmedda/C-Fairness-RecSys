#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    print_function,
    division,
    absolute_import)

# =============================================================================
# Imports
# =============================================================================

import os

from numpy.testing import (
    TestCase,
    run_module_suite,
    assert_allclose,
    assert_array_equal)
import numpy as np

from kamrecsys.datasets import SAMPLE_PATH

# =============================================================================
# Variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Test Classes
# =============================================================================


class TestLoadEvent(TestCase):

    def test_func(self):
        from kamrecsys.datasets import load_event

        # without event_feature ###

        # get data
        infile = os.path.join(
            os.path.dirname(__file__), 'sushi3b_test.event')
        data = load_event(
            infile, n_otypes=2, event_otypes=(0, 1))

        self.assertEqual(data.n_otypes, 2)
        assert_array_equal(data.n_objects, [20, 7])
        self.assertEqual(data.s_event, 2)
        self.assertEqual(data.n_events, 20)
        assert_array_equal(data.event_otypes, [0, 1])
        self.assertIsNone(data.event_feature)

        # with event_feature ###

        # get data
        infile = os.path.join(
            os.path.dirname(__file__), 'sushi3bs_test.event')
        data = load_event(infile, event_dtype=np.dtype([('score', int)]))

        self.assertEqual(data.n_otypes, 2)
        assert_array_equal(data.n_objects, [20, 7])
        self.assertEqual(data.s_event, 2)
        self.assertEqual(data.n_events, 20)
        assert_array_equal(data.event_otypes, [0, 1])
        event_feature = data.event_feature['score']
        assert_array_equal(event_feature[:5], [0, 0, 3, 4, 1])
        assert_array_equal(event_feature[-5:], [3, 4, 4, 4, 3])


class TestLoadEventWithScore(TestCase):

    def test_func(self):
        from kamrecsys.datasets import load_event_with_score

        # with timestamp ###

        # get data
        infile = os.path.join(SAMPLE_PATH, 'movielens_mini.event')
        event_dtype = np.dtype([('timestamp', int)])
        data = load_event_with_score(infile, event_dtype=event_dtype)

        # object information
        self.assertEqual(data.n_otypes, 2)
        assert_array_equal(data.n_objects, [8, 10])
        assert_array_equal(
            data.eid[0], [1, 2, 5, 6, 7, 8, 9, 10])
        assert_array_equal(
            data.eid[0], [1, 2, 5, 6, 7, 8, 9, 10])
        assert_array_equal(
            data.eid[1], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertDictEqual(
            data.iid[0],
            {1: 0, 2: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7})
        self.assertDictEqual(
            data.iid[1],
            {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9})
        self.assertIsNone(data.feature[0])
        self.assertIsNone(data.feature[1])

        # event information
        self.assertEqual(data.s_event, 2)
        assert_array_equal(data.event_otypes, [0, 1])
        self.assertEqual(data.n_events, 30)
        assert_array_equal(data.event[:3, :], [[2, 1], [7, 6], [2, 0]])
        assert_array_equal(data.event[-3:, :], [[0, 2], [3, 7], [7, 8]])
        ef = data.event_feature['timestamp']
        assert_array_equal(ef[:3], [875636053, 877892210, 875635748])
        assert_array_equal(ef[-3:], [878542960, 883600657, 877889005])

        # event information
        self.assertEqual(data.n_score_levels, 5)
        assert_array_equal(data.score_domain, [1, 5, 1])
        assert_allclose(data.score[:5], [3, 4, 4, 4, 5])
        assert_allclose(data.score[-5:], [4, 4, 4, 4, 4])

        # without timestamp

        # get data
        infile = os.path.join(
            os.path.dirname(__file__), 'sushi3bs_test.event')
        data = load_event_with_score(
            infile, n_otypes=2, event_otypes=(0, 1),
            score_domain=(0.0, 4.0, 1.0))

        self.assertEqual(data.n_otypes, 2)
        assert_array_equal(data.n_objects, [20, 7])
        self.assertEqual(data.s_event, 2)
        self.assertEqual(data.n_events, 20)
        assert_array_equal(data.event_otypes, [0, 1])
        self.assertIsNone(data.event_feature)
        self.assertEqual(data.n_score_levels, 5)
        assert_allclose(data.score_domain, [0., 4., 1.])


# =============================================================================
# Main Routine
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
