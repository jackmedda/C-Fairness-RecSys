#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert Flixster datasets to KamRecSys Sample format

Instruction
-----------

1. Download original file, ``flixster.zip``, from `Mohsen Jamali
   <http://www.cs.ubc.ca/~jamalim/datasets/>`_ 's homepage.
2. Unpack this ``flixster.zip``, and place the following files at
   this directory:
   ratings.txt
3. Run this script. As default, converted files are generated at
   ``../kamrecsys/datasets/data/`` directory. If you want change the target
   directory, you need to specify it as the first argument of this script.
4. Remove original files, if you do not need them.
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

import os
import sys
import io

# =============================================================================
# Constants
# =============================================================================

# =============================================================================
# Variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Main routine
# =============================================================================

# help message
if ('-h' in sys.argv) or ('--help' in sys.argv):
    print(__doc__, file=sys.stderr)
    sys.exit(0)

# set directories
stem = 'flixster'
pwd = os.path.dirname(__file__)
if len(sys.argv) >= 2:
    target = sys.argv[1]
else:
    target = os.path.join(pwd, '..', 'kamrecsys', 'datasets', 'data')

# convert event files of score ratings
# ---------------------------------------------------------

infile = open(os.path.join(pwd, 'ratings.txt'), 'r')
outfile = open(os.path.join(target, stem + '.event'), 'w')

print(
    "# Flixster data set\n"
    "#\n"
    "# Original files are distributed by Mohsen Jamali at the site:\n"
    "# http://www.cs.ubc.ca/~jamalim/datasets/\n"
    "# To use this data, follow the license permitted by the original "
    "distributor.\n"
    "#\n"
    "# This data set consists of:\n"
    "#\n"
    "# * 8,196,077 ratings from 147,612 users on 48,794 movies.\n"
    "#\n"
    "# Format\n"
    "# ------\n"
    "# user : int\n"
    "#     user id of the user who rated the movie\n"
    "# item : int\n"
    "#     item id of the movie rated by the user\n"
    "# score : int\n"
    "#     rating score whose range is:\n"
    "#     {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0}\n",
    end='', file=outfile)

for line in infile.readlines():
    f = line.rstrip('\r\n').split('\t')
    print(f[0], f[1], f[2], sep='\t', file=outfile)

infile.close()
outfile.close()
