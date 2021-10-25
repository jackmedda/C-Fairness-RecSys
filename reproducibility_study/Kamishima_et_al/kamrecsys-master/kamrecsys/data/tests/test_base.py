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
    assert_equal,
    assert_raises)
import numpy as np

# =============================================================================
# Module variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Test Classes
# =============================================================================


class TestBaseData(TestCase):

    def test_class(self):
        from kamrecsys.datasets import load_pci_sample

        data = load_pci_sample()

        assert_equal(data.to_iid(0, 'Mick LaSalle'), 5)
        with assert_raises(ValueError):
            data.to_iid(0, 'Dr. X')
        assert_equal(data.to_eid(1, 4), 'The Night Listener')
        with assert_raises(ValueError):
            data.to_eid(1, 100)

    def test_gen_id_substitution_table(self):
        from kamrecsys.data import ObjectUtilMixin

        orig_obj = np.array(
            [0, 5, 11, 13, 15, 19, 20, 30, 36, 40,
             49, 57, 73, 73, 76, 77, 78, 86, 94, 95], dtype=int)
        sub_index = np.array(
            [0, 1, 4, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19], dtype=int)

        table = ObjectUtilMixin._gen_id_substitution_table(orig_obj, sub_index)
        assert_array_equal(
            table,
            [0, 1, -1, -1, 2, -1, -1, 3, 4, 5,
             -1, 6, 7, -1, 8, 9, 10, 11, -1, 12])


# =============================================================================
# Main Routines
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
