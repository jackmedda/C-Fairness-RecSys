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
    assert_array_equal,
    assert_equal)


# =============================================================================
# Module variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Test Classes
# =============================================================================


class TestFlixsterClass(TestCase):
    def test_load_flixster_rating(self):
        from kamrecsys.datasets import load_flixster_rating

        data = load_flixster_rating()

        assert_array_equal(
            sorted(data.__dict__.keys()),
            sorted(['event_otypes', 'n_otypes', 'n_events', 'n_score_levels',
                    'feature', 'event', 'iid', 'event_feature',
                    'score', 'eid', 'n_objects', 's_event', 'score_domain']))
        assert_array_equal(data.event_otypes, [0, 1])
        assert_equal(data.n_otypes, 2)
        assert_equal(data.n_events, 8196077)
        assert_equal(data.s_event, 2)
        assert_array_equal(data.n_objects, [147612, 48794])

        # events
        assert_array_equal(data.score_domain, [0.5, 5.0, 0.5])
        assert_array_equal(
            data.event[:5],
            [
                [124545, 57], [124545, 665], [124545, 969],
                [124545, 1650], [124545, 2230]
            ]
        )
        assert_array_equal(
            data.event[-5:],
            [
                [14217, 28183], [14217, 36255], [14217, 37636],
                [14217, 40326], [14217, 48445]
            ]
        )
        assert_array_equal(data.eid[0][:5],
                           [6, 7, 8, 9, 11])
        assert_array_equal(data.eid[0][-5:],
                           [1049477, 1049489, 1049491, 1049494, 1049508])
        assert_array_equal(data.eid[1][:5],
                           [1, 2, 3, 4, 5])
        assert_array_equal(data.eid[1][-5:],
                           [66712, 66714, 66718, 66725, 66726])
        assert_array_equal(data.score[:5], [1.5, 1.0, 2.0, 1.0, 5.0])
        assert_array_equal(data.score[-5:], [5.0, 4.0, 3.0, 4.0, 5.0])


# =============================================================================
# Main Routines
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
