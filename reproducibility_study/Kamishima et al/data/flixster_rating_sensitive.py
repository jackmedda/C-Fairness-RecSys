#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Add sensitive information to the ``flixster`` data set.

SYNOPSIS::

    SCRIPT [options]

Description
===========

Details of this script

Options
=======

-i <INPUT>, --in <INPUT>
    specify <INPUT> file name
-o <OUTPUT>, --out <OUTPUT>
    specify <OUTPUT> file name
-s, --stat
    show statics mode
-t <THRESHOLD>, --threshold <THRESHOLD>
    a sensitive: top-<THRESHOLD> most popular items or not (default=0.1)
--version
    show version
"""

from __future__ import (
    print_function,
    division,
    absolute_import)
from six.moves import xrange

# =============================================================================
# Imports
# =============================================================================

import sys
import os
import argparse
import numpy as np

from kamrecsys.datasets import load_flixster_rating

# =============================================================================
# Module metadata variables
# =============================================================================

__author__ = "Toshihiro Kamishima ( http://www.kamishima.net/ )"
__date__ = "14/02/20"
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


def calc_popularity(data):
    """ calc the number of users who rated the item and its rank

    add 'popularity' and 'popularity_rank' fields to data.feature[1]

    Parameters
    ----------
    data : kamrecsys.data.EventWithScoreData
        data loaded by :func:`kamrecsys.datasets.load_flixster_rating`
    """

    n_items = data.n_objects[1]
    popularity = np.bincount(data.event[:, 1], minlength=n_items)
    popularity_rank = np.empty(n_items, dtype=np.int)
    popularity_rank[np.argsort(popularity)] = \
        np.arange(n_items - 1, -1, -1)
    feature = np.empty(n_items,
                       dtype=[('popularity', np.int),
                              ('popularity_rank', np.int)])
    feature['popularity'] = popularity
    feature['popularity_rank'] = popularity_rank
    data.feature[1] = feature


def item_popularity(data, threshold):
    """ sensitive: item's popularity

    0: short_head, 1: long_tail

    Parameters
    ----------
    data : kamrecsys.data.EventWithScoreData
        data loaded by :func:`kamrecsys.datasets.load_flixster_rating`
    threshold : float
        threshold to determine the value of this sensitive variable

    Returns
    -------
    trg : array_like, dtype=int, shape=(n_events,)
        generated target variable
    """
    feature = data.feature[1][data.event[:, 1]]['popularity_rank']
    n_items = data.n_objects[1]
    return np.where(feature < n_items * threshold, 0, 1)


def print_stats(data, sensitive):
    """ print sizes, means, and vars for each sensitive value

    Parameters
    ----------
    data : kamrecsys.data.EventWithScoreData
        data loaded by :func:`kamrecsys.datasets.load_flixster_rating`
    sensitive : array, dtype=np.int
        sensitive values for events
    """
    s = data.score
    for v in xrange(2):
        target = (sensitive == v)
        print("V =", v, "\t",
              "size = ", np.sum(target), "\t",
              "mean =", np.mean(s[target]), "\t",
              "var =", np.var(s[target]))

# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Main routine
# =============================================================================


def stat(opt):
    """Output stats

    Parameters
    ----------
    opt : Options
        parsed command line options
    """

    # read data
    data = load_flixster_rating(infile=opt.infile)

    # generate sensitive data
    calc_popularity(data)
    sensitive = item_popularity(data, opt.threshold)
    print("### item: popularity < {0} ###".format(opt.threshold))
    print_stats(data, sensitive)

    # close file
    if opt.infile is not None:
        opt.infile.close()

    if opt.outfile is not sys.stdout:
        opt.outfile.close()

    sys.exit(0)


def main(opt):
    """ output data sets with sensitive information

    Parameters
    ----------
    opt : Options
        parsed command line options
    """

    # read data
    data = load_flixster_rating(infile=opt.infile)

    # generate sensitive data
    calc_popularity(data)
    sensitive = item_popularity(data, opt.threshold)

    # output
    for i in xrange(data.n_events):
        for j in xrange(data.s_event):
            print(data.to_eid(j, data.event[i][j]),
                  end="\t", file=opt.outfile)
        print(data.score[i],
              sensitive[i],
              sep="\t", file=opt.outfile)

    # close file
    if opt.infile is not None:
        opt.infile.close()

    if opt.outfile is not sys.stdout:
        opt.outfile.close()

    sys.exit(0)


if __name__ == '__main__':
    script_name = os.path.basename(sys.argv[0])

    ap = argparse.ArgumentParser(
        description='pydoc is useful for learning the details.')

    # common options
    ap.add_argument('--version', action='version',
                    version='%(prog)s ' + __version__)

    # basic file i/o
    ap.add_argument('-i', '--in', dest='infile',
                    default=None, type=argparse.FileType('r'))
    ap.add_argument('-o', '--out', dest='outfile',
                    default=None, type=argparse.FileType('w'))
    ap.add_argument('outfilep', nargs='?', metavar='OUTFILE',
                    default=sys.stdout, type=argparse.FileType('w'))

    # script specific options
    ap.add_argument('-s', '--stat', dest='stat', action='store_true')
    ap.set_defaults(stat=False)
    ap.add_argument('-t', '--threshold', type=float, default=0.1)

    # parsing
    opt = ap.parse_args()

    # post-processing for command-line options

    # basic file i/o
    if opt.outfile is None:
        opt.outfile = opt.outfilep
    del vars(opt)['outfilep']

    opt.script_name = script_name
    opt.script_version = __version__

    if opt.stat:
        stat(opt)
    else:
        main(opt)
