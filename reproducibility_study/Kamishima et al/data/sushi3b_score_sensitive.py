#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Add sensitive information to the ``sushi3b_score`` data set.

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
-s <SENSITIVE>, --sensitive <SENSITIVE>
    type of sensitive variable, (default 0)

        0 : user's gender
        1 : user's age < <THRESHOLD>
        2 : item is seafood or not

-t <THRESHOLD>, --threshold <THRESHOLD>
    threshold to determine the value of sensitive variables
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
import os
import argparse
import numpy as np

from kamrecsys.datasets import load_sushi3b_score

# =============================================================================
# Module metadata variables
# =============================================================================

__author__ = "Toshihiro Kamishima ( http://www.kamishima.net/ )"
__date__ = "14/02/17"
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


def user_gender(data):
    """
    sensitive: user's gender

    0: male, 1: female

    Parameters
    ----------
    data : kamrecsys.data.EventWithScoreData
        data loaded by :func:`kamrecsys.datasets.load_sushi3b_score`

    Returns
    -------
    trg : array_like, dtype=int, shape=(n_events,)
        generated target variable
    """
    return data.feature[0][data.event[:, 0]]['gender']


def user_age(data, threshold):
    """
    sensitive: user's age

    0: less than <THRESHOLD>, 1: otherwise

    Parameters
    ----------
    data : kamrecsys.data.EventWithScoreData
        data loaded by :func:`kamrecsys.datasets.load_sushi3b_score`
    threshold : float
        threshold to determine the value of this sensitive variable

    Returns
    -------
    trg : array_like, dtype=int, shape=(n_events,)
        generated sensitive variable
    """
    feature = data.feature[0][data.event[:, 0]]['age']

    return np.where(feature < int(threshold), 0, 1)


def item_seafood(data):
    """
    sensitive: item is seafood or not

    0: seafood, 1: otherwise

    Parameters
    ----------
    data : kamrecsys.data.EventWithScoreData
        data loaded by :func:`kamrecsys.datasets.load_sushi3b_score`

    Returns
    -------
    trg : array_like, dtype=int, shape=(n_events,)
        generated target variable
    """
    return data.feature[1][data.event[:, 1]]['seafood']


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
    data = load_sushi3b_score(infile=opt.infile)

    # generate view data #####
    if opt.sensitive == 0:
        sensitive = user_gender(data)
    elif opt.sensitive == 1:
        sensitive = user_age(data, opt.threshold)
    elif opt.sensitive == 2:
        sensitive = item_seafood(data)
    else:
        raise TypeError('Illegal type of a sensitive')

    # output #####
    for i in xrange(data.n_events):
        for j in xrange(data.s_event):
            print(data.to_eid(j, data.event[i][j]),
                  end="\t", file=opt.outfile)
        print(np.int(data.score[i]),
              sensitive[i],
              sep="\t", file=opt.outfile)

    # post process #####

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
    ap.add_argument('-s', '--sensitive', type=int, default=0)
    ap.add_argument('-t', '--threshold', type=float, default=1)

    # parsing
    opt = ap.parse_args()

    # post-processing for command-line options

    # basic file i/o
    if opt.outfile is None:
        opt.outfile = opt.outfilep
    del vars(opt)['outfilep']

    opt.script_name = script_name
    opt.script_version = __version__

    main(opt)
