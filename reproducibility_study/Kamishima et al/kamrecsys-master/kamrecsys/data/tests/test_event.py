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
import numpy as np

import os

from kamrecsys.data import EventData
from kamrecsys.datasets import SAMPLE_PATH, load_movielens_mini

# =============================================================================
# Module variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================


def load_test_data():
    infile = os.path.join(SAMPLE_PATH, 'pci.event')
    dtype = np.dtype([('event', 'U18', 2), ('score', float)])
    x = np.genfromtxt(fname=infile, delimiter='\t', dtype=dtype)
    data = EventData(n_otypes=2, event_otypes=np.array([0, 1]))
    data.set_event(x['event'])
    return data, x


# =============================================================================
# Test Classes
# =============================================================================

class TestEventUtilMixin(TestCase):

    def test_to_eid_event(self):
        data, x = load_test_data()

        # test to_eid_event
        check = data.to_eid_event(data.event)
        assert_array_equal(x['event'], check)

        # test to_eid_event / per line conversion
        check = np.empty_like(data.event, dtype=x['event'].dtype)
        for i, j in enumerate(data.event):
            check[i, :] = data.to_eid_event(j)
        assert_array_equal(x['event'], check)

    def test_to_iid_event(self):
        data, x = load_test_data()

        # test EventData.to_iid_event
        assert_array_equal(data.event, data.to_iid_event(x['event']))

        # test EventData.to_iid_event / per line conversion
        check = np.empty_like(x['event'], dtype=int)
        for i, j in enumerate(x['event']):
            check[i, :] = data.to_iid_event(j)
        assert_array_equal(data.event, check)


class TestEventData(TestCase):

    def test_filter_event(self):
        from kamrecsys.data import EventWithScoreData

        # load movie_lens
        data = load_movielens_mini()

        # filter events
        filter_cond = np.arange(data.n_events) % 3 == 0
        filtered_data = super(
            EventWithScoreData, data).filter_event(filter_cond)

        assert_array_equal(
            filtered_data.event[:, 0], [1, 5, 3, 4, 0, 0, 0, 2, 2, 0])
        assert_array_equal(
            filtered_data.event[:, 1], [1, 3, 6, 5, 7, 6, 4, 0, 7, 2])

        assert_array_equal(
            filtered_data.to_eid(0, filtered_data.event[:, 0]),
            data.to_eid(0, data.event[filter_cond, 0]))
        assert_array_equal(
            filtered_data.to_eid(1, filtered_data.event[:, 1]),
            data.to_eid(1, data.event[filter_cond, 1]))

        assert_array_equal(
            filtered_data.event_feature['timestamp'],
            [875636053, 877889130, 891351328, 879362287, 878543541,
             875072484, 889751712, 883599478, 883599205, 878542960])

        assert_array_equal(filtered_data.eid[0], [1, 5, 6, 7, 8, 10])
        assert_array_equal(filtered_data.eid[1], [1, 2, 3, 4, 5, 7, 8, 9])

        assert_equal(
            filtered_data.iid[0],
            {1: 0, 5: 1, 6: 2, 7: 3, 8: 4, 10: 5})
        assert_equal(
            filtered_data.iid[1],
            {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 7: 5, 8: 6, 9: 7})

        assert_equal(
            filtered_data.feature[0]['zip'],
            [u'85711', u'15213', u'98101', u'91344', u'05201', u'90703'])
        assert_equal(
            filtered_data.feature[1]['name'],
            [u'Toy Story (1995)', u'GoldenEye (1995)', u'Four Rooms (1995)',
             u'Get Shorty (1995)', u'Copycat (1995)', u'Twelve Monkeys (1995)',
             u'Babe (1995)', u'Dead Man Walking (1995)'])

        # dummy event data
        data = EventData()
        data.set_event(np.tile(np.arange(5), (2, 2)).T)
        filtered_data = data.filter_event(
            [True, False, True, True, False, False, True, True, False, False])

        assert_equal(filtered_data.n_events, 5)
        assert_array_equal(
            filtered_data.event, [[0, 0], [2, 2], [3, 3], [1, 1], [2, 2]])


# =============================================================================
# Main Routines
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
