#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
add "gender" view information to a MovieLens 1M data set or its variants

SYNOPSIS::

    SCRIPT [options] [<INPUT> [<OUTPUT>]]

Options
=======

-i <INPUT>, --in <INPUT>
    specify <INPUT> file name
-o <OUTPUT>, --out <OUTPUT>
    specify <OUTPUT> file name
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

from kamrecsys.datasets import load_movielens1m

# =============================================================================
# Module metadata variables
# =============================================================================

__author__ = "Toshihiro Kamishima ( http://www.kamishima.net/ )"
__date__ = "2013/07/25"
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


def ml1m_gender(data):
    """
    Generate target variable for the Movielens 1m data set.

    Target variable is 1 if user is female

    Parameters
    ----------
    data : kamrecsys.data.EventWithScoreData
        movielens data by kamrecsys.datasets.load_movielens1m

    Returns
    -------
    trg : array_like, dtype=int, shape=(n_events,)
        generated target variable
    """
    return data.feature[0][data.event[:, 0]]['gender']

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
    data = load_movielens1m(infile=opt.infile)

    # generate view data #####
    sensitive = ml1m_gender(data)

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
    script_name = os.path.basename(sys.argv[0])

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

    # parsing
    opt = ap.parse_args()

    # post-processing for command-line options #####
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
