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
    assert_array_equal)
import numpy as np

from kamrecsys.datasets import load_movielens_mini
from kamrecsys.base import BaseRecommender, BaseEventRecommender

# =============================================================================
# Variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Classes
# =============================================================================


class Recommender(BaseRecommender):

    def predict(self, eev):
        pass


class EventRecommender(BaseEventRecommender):

    def raw_predict(self, ev):

        return np.zeros(ev.shape[0])

# =============================================================================
# Test Classes
# =============================================================================


class TestBaseRecommender(TestCase):

    def test_class(self):

        # setup
        data = load_movielens_mini()

        # __init__()
        rec = Recommender(random_state=1234)

        self.assertEqual(rec.n_otypes, 0)
        self.assertIsNone(rec.n_objects)
        self.assertIsNone(rec.eid)
        self.assertIsNone(rec.iid)
        self.assertEqual(rec.random_state, 1234)
        self.assertIsNone(rec._rng)
        self.assertDictEqual(rec.fit_results_, {})

        # _set_object_info()
        rec._set_object_info(data)

        self.assertEqual(rec.n_otypes, 2)
        assert_array_equal(rec.n_objects, [8, 10])
        assert_array_equal(rec.eid, data.eid)
        assert_array_equal(rec.iid, data.iid)

        # to_eid()
        self.assertEqual(rec.to_eid(0, 0), 1)
        self.assertEqual(rec.to_eid(1, 0), 1)

        # to_iid()
        self.assertEqual(rec.to_iid(0, 1), 0)
        self.assertEqual(rec.to_iid(1, 1), 0)


class TestBaseEventRecommender(TestCase):

    def test_class(self):

        # setup
        data = load_movielens_mini()

        rec = EventRecommender()
        rec._set_object_info(data)

        # _set_event_info
        rec._set_event_info(data)

        self.assertEqual(rec.s_event, 2)
        assert_array_equal(rec.event_otypes, [0, 1])
        self.assertEqual(rec.n_events, 30)
        assert_array_equal(
            rec.event[0, :], [rec.to_iid(0, 5), rec.to_iid(1, 2)])
        assert_array_equal(
            rec.event[-1, :], [rec.to_iid(0, 10), rec.to_iid(1, 9)])
        ts = sorted(rec.event_feature['timestamp'])
        assert_array_equal(ts[:2], [874965758, 875071561])
        assert_array_equal(ts[-2:], [891352220, 891352864])
        self.assertIsNone(rec.event_index)

        # fit
        rec.fit(data, event_index=(1, 0))
        assert_array_equal(rec.event_index, [1, 0])

        # get_event
        ev, n_objects = rec.get_event()
        assert_array_equal(ev[0, :], [rec.to_iid(1, 2), rec.to_iid(0, 5)])
        assert_array_equal(ev[-1, :], [rec.to_iid(1, 9), rec.to_iid(0, 10)])
        assert_array_equal(n_objects, [10, 8])

        # predict
        self.assertEqual(rec.predict([0, 0]).ndim, 0)
        self.assertEqual(rec.predict([[0, 0]]).ndim, 0)
        self.assertEqual(rec.predict([[0, 0], [0, 1]]).ndim, 1)
        assert_array_equal(rec.predict([[0, 0], [0, 1]]).shape, (2,))

        # remove_data
        rec.remove_data()

        self.assertEqual(rec.n_events, 30)
        self.assertIsNone(rec.event)
        self.assertIsNone(rec.event_feature)
        assert_array_equal(rec.event_index, (1, 0))


# =============================================================================
# Main Routine
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
