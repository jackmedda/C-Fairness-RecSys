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


class TestLoadPCISample(TestCase):

    def test_load_pci_sample(self):
        from kamrecsys.datasets import load_pci_sample

        data = load_pci_sample()
        assert_array_equal(data.event_otypes, [0, 1])
        assert_equal(data.n_otypes, 2)
        assert_equal(data.n_events, 35)
        assert_array_equal(data.feature, [None, None])
        assert_array_equal(
            data.event,
            [[2, 1], [2, 2], [2, 5], [2, 3], [2, 4], [5, 1], [5, 2],
             [5, 0], [5, 3], [5, 5], [5, 4], [0, 2], [0, 0], [0, 5],
             [0, 3], [0, 4], [3, 1], [3, 2], [3, 0], [3, 3], [3, 4],
             [3, 5], [6, 2], [6, 3], [6, 5], [1, 1], [1, 2], [1, 0],
             [1, 3], [1, 5], [1, 4], [4, 1], [4, 2], [4, 3], [4, 4]])
        self.assertDictEqual(
            data.iid[0],
            {
                'Jack Matthews': 2, 'Mick LaSalle': 5, 'Claudia Puig': 0,
                'Lisa Rose': 3, 'Toby': 6, 'Gene Seymour': 1,
                'Michael Phillips': 4
            }
        )
        self.assertDictEqual(
            data.iid[1],
            {
                'Lady in the Water': 1, 'Just My Luck': 0,
                'Superman Returns': 3, 'You, Me and Dupree': 5,
                'Snakes on a Planet': 2, 'The Night Listener': 4
            }
        )
        self.assertIsNone(data.event_feature)
        assert_array_equal(
            data.score,
            [3., 4., 3.5, 5., 3., 3., 4., 2., 3., 2., 3., 3.5, 3., 2.5, 4.,
             4.5, 2.5, 3.5, 3., 3.5, 3., 2.5, 4.5, 4., 1., 3., 3.5, 1.5, 5.,
             3.5, 3., 2.5, 3., 3.5, 4.])
        assert_array_equal(
            data.eid[0],
            ['Claudia Puig', 'Gene Seymour', 'Jack Matthews', 'Lisa Rose',
             'Michael Phillips', 'Mick LaSalle', 'Toby'])
        assert_array_equal(
            data.eid[1],
            ['Just My Luck', 'Lady in the Water', 'Snakes on a Planet',
             'Superman Returns', 'The Night Listener', 'You, Me and Dupree'])
        assert_array_equal(data.n_objects, [7, 6])
        assert_equal(data.s_event, 2)
        assert_array_equal(data.score_domain, [1., 5., 0.5])


# =============================================================================
# Main Routines
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
