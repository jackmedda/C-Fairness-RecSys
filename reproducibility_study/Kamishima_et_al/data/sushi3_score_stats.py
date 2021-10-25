#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calculate the means of subsets of a Sushi3 score date set
"""

from __future__ import (
    print_function,
    division,
    absolute_import,
    unicode_literals)
from six.moves import xrange

# =============================================================================
# Imports
# =============================================================================

import numpy as np

from kamrecsys.datasets import load_sushi3b_score

# =============================================================================
# Module metadata variables
# =============================================================================

__author__ = "Toshihiro Kamishima ( http://www.kamishima.net/ )"
__date__ = "2014/02/16"
__version__ = "1.0.0"
__copyright__ = "Copyright (c) 2014 Toshihiro Kamishima all rights reserved."
__license__ = "MIT License: http://www.opensource.org/licenses/mit-license.php"

# =============================================================================
# Public symbols
# =============================================================================

__all__ = []

# =============================================================================
# Constants
# =============================================================================

# =============================================================================
# Module variables
# =============================================================================

# =============================================================================
# Functions 
# =============================================================================


def user_gender(d):
    """ print mean score of each gender

    Parameters
    ----------
    d : class:`kamrecsys.data.EventWithScoreData`
        target data
    """
    uid = d.event[:, 0]
    s = d.score
    f = d.feature[0]['gender']

    print("### user: gender ###")
    for v in xrange(2):
        target = (f[uid] == v)
        print("feature =", v, "\t",
              "size = ", np.sum(target), "\t",
              "mean =", np.mean(s[target]), "\t",
              "var =", np.var(s[target]))
    print("# diff =", np.mean(s[f[uid] == 0]) - np.mean(s[f[uid] == 1]))


def user_age(d, th):
    """ print mean score of each age

    Parameters
    ----------
    d : class:`kamrecsys.data.EventWithScoreData`
        target data
    th : int
        threshold of the user's age
    """
    uid = d.event[:, 0]
    s = d.score
    f = d.feature[0]['age']

    print("### user: age <", th, "###")
    target = (f[uid] < th)
    print("feature <", th, "\t",
          "size = ", np.sum(target), "\t",
          "mean =", np.mean(s[target]), "\t",
          "var =", np.var(s[target]))
    target = (f[uid] >= th)
    print("feature >=", th, "\t",
          "size = ", np.sum(target), "\t",
          "mean =", np.mean(s[target]), "\t",
          "var =", np.var(s[target]))
    print("# diff =", np.mean(s[f[uid] < th]) - np.mean(s[f[uid] >= th]))


def user_child_ew(d):
    """ print mean score of each large regions

    Parameters
    ----------
    d : class:`kamrecsys.data.EventWithScoreData`
        target data
    """
    uid = d.event[:, 0]
    s = d.score
    f = d.feature[0]['child_ew']

    print("### user: child_ew ###")
    for v in xrange(2):
        target = (f[uid] == v)
        print("feature =", v, "\t",
              "size = ", np.sum(target), "\t",
              "mean =", np.mean(s[target]), "\t",
              "var =", np.var(s[target]))
    print("# diff =", np.mean(s[f[uid] == 0]) - np.mean(s[f[uid] == 1]))


def user_current_ew(d):
    """ print mean score of each large regions

    Parameters
    ----------
    d : class:`kamrecsys.data.EventWithScoreData`
        target data
    """
    uid = d.event[:, 0]
    s = d.score
    f = d.feature[0]['current_ew']

    print("### user: current_ew ###")
    for v in xrange(2):
        target = (f[uid] == v)
        print("feature =", v, "\t",
              "size = ", np.sum(target), "\t",
              "mean =", np.mean(s[target]), "\t",
              "var =", np.var(s[target]))
    print("# diff =", np.mean(s[f[uid] == 0]) - np.mean(s[f[uid] == 1]))


def user_moved(d):
    """ print mean score of moved/non-moved users

    Parameters
    ----------
    d : class:`kamrecsys.data.EventWithScoreData`
        target data
    """
    uid = d.event[:, 0]
    s = d.score
    f = d.feature[0]['moved']

    print("### user: moved ###")
    for v in xrange(2):
        target = (f[uid] == v)
        print("feature =", v, "\t",
              "size = ", np.sum(target), "\t",
              "mean =", np.mean(s[target]), "\t",
              "var =", np.var(s[target]))
    print("# diff =", np.mean(s[f[uid] == 0]) - np.mean(s[f[uid] == 1]))


def item_maki(d):
    """ print mean score of maki/non-maki items

    Parameters
    ----------
    d : class:`kamrecsys.data.EventWithScoreData`
        target data
    """
    iid = d.event[:, 1]
    s = d.score
    f = d.feature[1]['maki']

    print("### item: maki ###")
    for v in xrange(2):
        target = (f[iid] == v)
        print("feature =", v, "\t",
              "size = ", np.sum(target), "\t",
              "mean =", np.mean(s[target]), "\t",
              "var =", np.var(s[target]))
    print("# diff =", np.mean(s[f[iid] == 0]) - np.mean(s[f[iid] == 1]))


def item_seafood(d):
    """ print mean score of seafood/non-seafood items

    Parameters
    ----------
    d : class:`kamrecsys.data.EventWithScoreData`
        target data
    """
    iid = d.event[:, 1]
    s = d.score
    f = d.feature[1]['seafood']

    print("### item: seafood ###")
    for v in xrange(2):
        target = (f[iid] == v)
        print("feature =", v, "\t",
              "size = ", np.sum(target), "\t",
              "mean =", np.mean(s[target]), "\t",
              "var =", np.var(s[target]))
    print("# diff =", np.mean(s[f[iid] == 0]) - np.mean(s[f[iid] == 1]))


def item_heaviness(d, th):
    """ print mean score: grouped by heaviness

    Parameters
    ----------
    d : class:`kamrecsys.data.EventWithScoreData`
        target data
    th : float
        threshold of the user's age
    """
    iid = d.event[:, 1]
    s = d.score
    f = d.feature[1]['heaviness']

    print("### item: heaviness <", th, "###")
    target = (f[iid] < float(th))
    print("feature <", th, "\t",
          "size = ", np.sum(target), "\t",
          "mean =", np.mean(s[target]), "\t",
          "var =", np.var(s[target]))
    target = (f[iid] >= float(th))
    print("feature >=", th, "\t",
          "size = ", np.sum(target), "\t",
          "mean =", np.mean(s[target]), "\t",
          "var =", np.var(s[target]))
    print("# diff =",
          np.mean(s[f[iid] < float(th)]) - np.mean(s[f[iid] >= float(th)]))


def item_frequency(d, th):
    """ print mean score: grouped by frequency

    Parameters
    ----------
    d : class:`kamrecsys.data.EventWithScoreData`
        target data
    th : float
        threshold of the user's age
    """
    iid = d.event[:, 1]
    s = d.score
    f = d.feature[1]['frequency']

    print("### item: frequency <", th, "###")
    target = (f[iid] < float(th))
    print("feature <", th, "\t",
          "size = ", np.sum(target), "\t",
          "mean =", np.mean(s[target]), "\t",
          "var =", np.var(s[target]))
    target = (f[iid] >= float(th))
    print("feature >=", th, "\t",
          "size = ", np.sum(target), "\t",
          "mean =", np.mean(s[target]), "\t",
          "var =", np.var(s[target]))
    print("# diff =",
          np.mean(s[f[iid] < float(th)]) - np.mean(s[f[iid] >= float(th)]))


def item_price(d, th):
    """ print mean score: grouped by price

    Parameters
    ----------
    d : class:`kamrecsys.data.EventWithScoreData`
        target data
    th : float
        threshold of the user's age
    """
    iid = d.event[:, 1]
    s = d.score
    f = d.feature[1]['price']

    print("### item: price <", th, "###")
    target = (f[iid] < float(th))
    print("feature <", th, "\t",
          "size = ", np.sum(target), "\t",
          "mean =", np.mean(s[target]), "\t",
          "var =", np.var(s[target]))
    target = (f[iid] >= float(th))
    print("feature >=", th, "\t",
          "size = ", np.sum(target), "\t",
          "mean =", np.mean(s[target]), "\t",
          "var =", np.var(s[target]))
    print("# diff =",
          np.mean(s[f[iid] < float(th)]) - np.mean(s[f[iid] >= float(th)]))


def item_supply(d, th):
    """ print mean score: grouped by supply

    Parameters
    ----------
    d : class:`kamrecsys.data.EventWithScoreData`
        target data
    th : float
        threshold of the user's age
    """
    iid = d.event[:, 1]
    s = d.score
    f = d.feature[1]['supply']

    print("### item: supply <", th, "###")
    target = (f[iid] < float(th))
    print("feature <", th, "\t",
          "size = ", np.sum(target), "\t",
          "mean =", np.mean(s[target]), "\t",
          "var =", np.var(s[target]))
    target = (f[iid] >= float(th))
    print("feature >=", th, "\t",
          "size = ", np.sum(target), "\t",
          "mean =", np.mean(s[target]), "\t",
          "var =", np.var(s[target]))
    print("# diff =",
          np.mean(s[f[iid] < float(th)]) - np.mean(s[f[iid] >= float(th)]))


# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Main routine
# =============================================================================

d = load_sushi3b_score()

# print(d.feature[0][322]) # Toshihiro Kamishima
# print(d.feature[1][8]) # Toro

# user features
user_gender(d)
for th in xrange(1, 6):
    user_age(d, th)
user_child_ew(d)
user_current_ew(d)
user_moved(d)

# item features
item_maki(d)
item_seafood(d)
for th in xrange(1, 4):
    item_heaviness(d, th)
for th in xrange(1, 3):
    item_frequency(d, th)
for th in xrange(2, 5):
    item_price(d, th)
for th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]:
    item_supply(d, th)
