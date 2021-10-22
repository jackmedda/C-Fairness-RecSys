#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    print_function,
    division,
    absolute_import)
import six

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


class TestLoadMovielens100k(TestCase):
    def test_load_movielens100k(self):
        from kamrecsys.datasets import load_movielens100k

        data = load_movielens100k()
        assert_array_equal(
            sorted(data.__dict__.keys()),
            sorted(['event_otypes', 'n_otypes', 'n_events', 'n_score_levels',
                    'feature', 'event', 'iid', 'event_feature',
                    'score', 'eid', 'n_objects', 's_event', 'score_domain']))
        assert_array_equal(data.event_otypes, [0, 1])
        assert_equal(data.n_otypes, 2)
        assert_equal(data.n_events, 100000)
        assert_equal(data.s_event, 2)
        assert_array_equal(data.n_objects, [943, 1682])
        assert_array_equal(data.score_domain, [1., 5., 1.])
        assert_array_equal(
            data.event[:5],
            [[195, 241], [185, 301], [21, 376], [243, 50], [165, 345]])
        assert_array_equal(
            data.event[-5:],
            [[879, 475], [715, 203], [275, 1089], [12, 224], [11, 202]])
        assert_array_equal(data.eid[0][:5], [1, 2, 3, 4, 5])
        assert_array_equal(data.eid[0][-5:], [939, 940, 941, 942, 943])
        assert_array_equal(data.eid[1][:5], [1, 2, 3, 4, 5])
        assert_array_equal(data.eid[1][-5:], [1678, 1679, 1680, 1681, 1682])
        assert_array_equal(
            data.event_feature.dtype.descr, [('timestamp', '<i8')])
        assert_array_equal(
            data.event_feature[:5].astype(int),
            [881250949, 891717742, 878887116, 880606923, 886397596])
        assert_array_equal(
            data.event_feature[-5:].astype(int),
            [880175444, 879795543, 874795795, 882399156, 879959583])
        assert_array_equal(data.score[:5], [3., 3., 1., 2., 1.])
        assert_array_equal(data.score[-5:], [3., 5., 1., 2., 3.])
        assert_array_equal(
            [data.iid[0][1], data.iid[0][2], data.iid[0][3]],
            [0, 1, 2])
        assert_array_equal(
            [data.iid[1][1], data.iid[1][2], data.iid[1][3]],
            [0, 1, 2])
        assert_array_equal(
            [data.iid[0][943], data.iid[0][942], data.iid[0][900]],
            [942, 941, 899])
        assert_array_equal(
            [data.iid[1][1682], data.iid[1][1681], data.iid[1][1000]],
            [1681, 1680, 999])
        assert_array_equal(data.feature[0]['age'][:3], [24, 53, 23])
        assert_array_equal(data.feature[0]['gender'][:3], [0, 1, 0])
        assert_array_equal(data.feature[0]['occupation'][:3], [19, 1, 20])
        assert_array_equal(
            data.feature[0]['zip'][:3],
            [six.u('85711'), six.u('94043'), six.u('32067')])
        assert_array_equal(data.feature[0]['age'][-3:], [20, 48, 22])
        assert_array_equal(data.feature[0]['gender'][-3:], [0, 1, 0])
        assert_array_equal(data.feature[0]['occupation'][-3:], [18, 12, 18])
        assert_array_equal(
            data.feature[0]['zip'][-3:],
            [six.u('97229'), six.u('78209'), six.u('77841')])

        assert_equal(len(data.feature[0]), 943)
        assert_equal(data.feature[1][0][0], six.u('Toy Story (1995)'))
        assert_equal(data.feature[1][0][1], 1)
        assert_equal(data.feature[1][0][2], 1)
        assert_equal(data.feature[1][0][3], 1995)
        assert_array_equal(
            data.feature[1][0][4],
            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        assert_equal(
            data.feature[1][0][5],
            six.u('http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)'))
        assert_equal(
            data.feature[1][-1][0],
            six.u('Scream of Stone (Schrei aus Stein) (1991)'))
        assert_equal(data.feature[1][-1][1], 8)
        assert_equal(data.feature[1][-1][2], 3)
        assert_equal(data.feature[1][-1][3], 1996)
        assert_array_equal(
            data.feature[1][-1][4],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        assert_equal(
            data.feature[1][-1][5],
            six.u('http://us.imdb.com/M/title-exact'
                  '?Schrei%20aus%20Stein%20(1991)'))
        assert_equal(len(data.feature[1]), 1682)

    def test_MOVIELENS100K_INFO(self):
        from kamrecsys.datasets import MOVIELENS100K_INFO

        assert_array_equal(
            MOVIELENS100K_INFO['user_occupation'],
            ['None', 'Other', 'Administrator', 'Artist', 'Doctor', 'Educator',
             'Engineer', 'Entertainment', 'Executive', 'Healthcare',
             'Homemaker', 'Lawyer', 'Librarian', 'Marketing', 'Programmer',
             'Retired', 'Salesman', 'Scientist', 'Student', 'Technician',
             'Writer'])
        assert_array_equal(
            MOVIELENS100K_INFO['item_genre'],
            ['Action', 'Adventure', 'Animation', "Children's", 'Comedy',
             'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
             'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
             'War', 'Western'])


class TestLoadMovielens1m(TestCase):
    def test_load_movielens1m(self):
        from kamrecsys.datasets import load_movielens1m

        data = load_movielens1m()
        assert_array_equal(
            sorted(data.__dict__.keys()),
            sorted([
                'event_otypes', 'n_otypes', 'n_events',
                'n_score_levels', 'feature', 'event', 'iid',
                'event_feature', 'score', 'eid', 'n_objects',
                's_event', 'score_domain']))
        assert_array_equal(data.event_otypes, [0, 1])
        assert_equal(data.n_otypes, 2)
        assert_equal(data.n_events, 1000209)
        assert_equal(data.s_event, 2)
        assert_array_equal(data.n_objects, [6040, 3706])
        assert_array_equal(data.score_domain, [1., 5., 1.])
        assert_array_equal(
            data.to_eid_event(data.event)[:5],
            [[1, 1193], [1, 661], [1, 914], [1, 3408],
             [1, 2355]])
        assert_array_equal(
            data.to_eid_event(data.event)[-5:],
            [[6040, 1091], [6040, 1094], [6040, 562], [6040, 1096],
             [6040, 1097]])
        assert_array_equal(data.eid[0][:5], [1, 2, 3, 4, 5])
        assert_array_equal(data.eid[0][-5:], [6036, 6037, 6038, 6039, 6040])
        assert_array_equal(data.eid[1][:5], [1, 2, 3, 4, 5])
        assert_array_equal(data.eid[1][-5:], [3948, 3949, 3950, 3951, 3952])
        assert_equal(str(data.event_feature.dtype),
                         "[('timestamp', '<i8')]")
        assert_equal(
            str(data.event_feature[:5]),
            "[(978300760,) (978302109,) (978301968,)"
            " (978300275,) (978824291,)]")
        assert_equal(
            str(data.event_feature[-5:]),
            "[(956716541,) (956704887,) (956704746,)"
            " (956715648,) (956715569,)]")
        assert_array_equal(data.score[:5], [5., 3., 3., 4., 5.])
        assert_array_equal(data.score[-5:], [1., 5., 5., 4., 4.])
        assert_array_equal(
            [data.iid[0][1], data.iid[0][2], data.iid[0][3]],
            [0, 1, 2])
        assert_array_equal(
            [data.iid[1][1], data.iid[1][2], data.iid[1][3]],
            [0, 1, 2])
        assert_array_equal(
            [data.iid[0][943], data.iid[0][942], data.iid[0][900]],
            [942, 941, 899])
        assert_array_equal(
            [data.iid[1][1682], data.iid[1][1681], data.iid[1][1000]],
            [1545, 1544, 936])

        assert_array_equal(data.feature[0]['gender'][:3], [1, 0, 0])
        assert_array_equal(data.feature[0]['age'][:3], [0, 6, 2])
        assert_array_equal(data.feature[0]['occupation'][:3], [10, 16, 15])
        assert_array_equal(data.feature[0]['zip'][:3],
                           [six.u('48067'), six.u('70072'), six.u('55117')])
        assert_array_equal(data.feature[0]['gender'][-3:], [1, 1, 0])
        assert_array_equal(data.feature[0]['age'][-3:], [6, 4, 2])
        assert_array_equal(data.feature[0]['occupation'][-3:], [1, 0, 6])
        assert_array_equal(data.feature[0]['zip'][-3:],
                           [six.u('14706'), six.u('01060'), six.u('11106')])
        assert_equal(len(data.feature[0]), 6040)

        assert_equal(data.feature[1][0][0], six.u('Toy Story (1995)'))
        assert_equal(data.feature[1][0][1], 1995)
        assert_array_equal(
            data.feature[1][0][2],
            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        assert_equal(data.feature[1][-1][0], six.u('Contender, The (2000)'))
        assert_equal(data.feature[1][-1][1], 2000)
        assert_array_equal(
            data.feature[1][-1][2],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        assert_equal(len(data.feature[1]), 3706)

    def test_MOVIELENS1M_INFO(self):
        from kamrecsys.datasets import MOVIELENS1M_INFO

        assert_array_equal(
            MOVIELENS1M_INFO['user_age'],
            ['Under 18', '18-24', '25-34', '35-44', '45-49', '50-55', '56+'])
        assert_array_equal(
            MOVIELENS1M_INFO['user_occupation'],
            ['Other or Not Specified', 'Academic/Educator', 'Artist',
             'Clerical/Admin', 'College/Grad Student', 'Customer Service',
             'Doctor/Health Care', 'Executive/Managerial', 'Farmer',
             'Homemaker', 'K-12 Student', 'Lawyer', 'Programmer', 'Retired',
             'Sales/Marketing', 'Scientist', 'Self-Employed',
             'Technician/Engineer', 'Tradesman/Craftsman', 'Unemployed',
             'Writer'])
        assert_array_equal(
            MOVIELENS1M_INFO['item_genre'],
            ['Action', 'Adventure', 'Animation', "Children's", 'Comedy',
             'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
             'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
             'War', 'Western'])


# =============================================================================
# Main Routines
# =============================================================================

if __name__ == '__main__':
    run_module_suite()
