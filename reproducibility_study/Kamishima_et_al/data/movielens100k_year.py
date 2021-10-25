#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
add "year" view information to a MovieLens 100k data set or its variants

SYNOPSIS::

    SCRIPT [options] [<INPUT> [<OUTPUT>]]

Options
=======

-i <INPUT>, --in <INPUT>
    specify <INPUT> file name
-o <OUTPUT>, --out <OUTPUT>
    specify <OUTPUT> file name
-y <YEAR>, --year <YEAR>
    threshold year. (default=1990)
--version
    show version
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

import sys
import argparse
import numpy as np

from kamrecsys.datasets import load_movielens100k

# =============================================================================
# Module metadata variables
# =============================================================================

__author__ = "Toshihiro Kamishima ( http://www.kamishima.net/ )"
__date__ = "2013/02/06"
__version__ = "1.0.0"
__copyright__ = "Copyright (c) 2013 Toshihiro Kamishima all rights reserved."
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


def ml100k_year(data, year=1990):
    """
    Generate target variable for the Movielens 100k data set.

    Target variable is 1 if the release year is newer than specified year.

    Parameters
    ----------
    data : kamrecsys.data.EventWithScoreData
        movielens data by kamrecsys.datasets.load_movielens100k
    year : optional, int
        threshold year, default=1990

    Returns
    -------
    trg : array_like, dtype=int, shape=(n_events,)
        generated target variable
    """
    item_years = data.feature[1][data.event[:, 1]]['year']
    return np.where(item_years > year, 1, 0)

# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Main routine
# =============================================================================


def main(opt):
    """ Main routine that exits with status code 0

    Parameters
    ----------
    opt : Options
        parsed command line options
    """

    # read data #####
    data = load_movielens100k(infile=opt.infile)

    # generate view data
    sensitive = ml100k_year(data, year=opt.year)

    # output #####
    for i in xrange(data.n_events):
        for j in xrange(data.s_event):
            print(data.to_eid(j, data.event[i][j]),
                  end="\t", file=opt.outfile)
        print(np.int(data.score[i]),
              sensitive[i],
              data.event_feature['timestamp'][i],
              sep="\t", file=opt.outfile)

    # post process #####

    # close file
    if opt.infile is not sys.stdin:
        opt.infile.close()

    if opt.outfile is not sys.stdout:
        opt.outfile.close()

    sys.exit(0)


# Check if this is call as command script
if __name__ == '__main__':
    # set script name
    script_name = sys.argv[0].split('/')[-1]

    # command-line option parsing
    ap = argparse.ArgumentParser(
        description='pydoc is useful for learning the details.')

    # common options
    ap.add_argument('--version', action='version',
                    version='%(prog)s ' + __version__)

    # basic file i/o
    ap.add_argument('-i', '--in', dest='infile',
                    default=None, type=argparse.FileType('r'))
    ap.add_argument('infilep', nargs='?', metavar='INFILE',
                    default=sys.stdin, type=argparse.FileType('r'))
    ap.add_argument('-o', '--out', dest='outfile',
                    default=None, type=argparse.FileType('w'))
    ap.add_argument('outfilep', nargs='?', metavar='OUTFILE',
                    default=sys.stdout, type=argparse.FileType('w'))

    # script specific options
    ap.add_argument('-y', '--year', dest='year', type=int, default=1990)

    # parsing
    opt = ap.parse_args()

    # post-processing for command-line options
    # basic file i/o
    if opt.infile is None:
        opt.infile = opt.infilep
    del vars(opt)['infilep']
    if opt.outfile is None:
        opt.outfile = opt.outfilep
    del vars(opt)['outfilep']

    # set meta-data of script and machine
    opt.script_name = script_name
    opt.script_version = __version__

    # call main routine
    main(opt)
