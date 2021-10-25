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

import os

from numpy.testing import (
    TestCase,
    run_module_suite,
    assert_array_equal)
import numpy as np

from kamrecsys.datasets import load_event_with_score

# =============================================================================
# Variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Test Classes
# =============================================================================


class TestDatasets(TestCase):

    def test_constants(self):
        from .. import (
            event_dtype_sensitive_and_timestamp, event_dtype_sensitive)

        infile = os.path.join(os.path.dirname(__file__), 'mlmini_t.event')
        data = load_event_with_score(
            infile, event_dtype=event_dtype_sensitive_and_timestamp)
        assert_array_equal(
            data.event_feature['sensitive'],
            np.array([0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0,
                      1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1]))

        infile = os.path.join(
            os.path.dirname(__file__), 'sushi3bs_test_gender.event')
        data = load_event_with_score(
            infile, event_dtype=event_dtype_sensitive)
        assert_array_equal(
            data.event_feature['sensitive'],
            np.array([0, 1, 1, 0, 1, 1, 0, 1, 0, 1,
                      1, 1, 1, 0, 1, 0, 1, 0, 0, 1]))


# =============================================================================
# Main Routines
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
