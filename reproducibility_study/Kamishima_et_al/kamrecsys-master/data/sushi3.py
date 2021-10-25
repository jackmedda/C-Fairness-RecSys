#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert sushi3 data sets to KamRecSys Sample format

Instruction
-----------

1. Download original file, ``sushi3-2016.zip``, from `SUSHI Preference Data
Sets
   <http://www.kamishima.net/sushi/>`_.
2. Unpack this ``sushi3b.tgz``, and place the following files at
   this directory:
   sushi3b.5000.10.score sushi.idata sushi.udata
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
stem = 'sushi3'
pwd = os.path.dirname(__file__)
if len(sys.argv) >= 2:
    target = sys.argv[1]
else:
    target = os.path.join(pwd, '..', 'kamrecsys', 'datasets', 'data')

# convert event files of score ratings
# ---------------------------------------------------------

infile = open(os.path.join(pwd, 'sushi3b.5000.10.score'), 'r')
outfile = open(os.path.join(target, stem + 'b_score.event'), 'w')

print(
    '# Sushi3b score data set\n'
    '#\n'
    '# Original files are distributed by the Grouplens Research Project at '
    'the site:\n'
    '# http://www.kamishima.net/sushi/\n'
    '# To use this data, follow the license permitted by the original '
    'distributor.\n'
    '#\n'
    '# This data set consists of:\n'
    '#\n'
    '# * 50,000 ratings (0-4) from 5000 users on 100 sushis.\n'
    '# * Each user has rated exactly 10 sushis selected at random.\n'
    '#\n'
    '# Format\n'
    '# ------\n'
    '# user : int\n'
    '#     user id of the user who rated the sushi\n'
    '# item : int\n'
    '#     item id of the sushi rated by the user\n'
    '# score : int\n'
    '#     rating score whose range is {0, 1, 2, 3, 4}\n',
    end='', file=outfile)

uid = 0
for line in infile.readlines():
    rating = line.rstrip('\r\n').split(" ")
    for iid in xrange(len(rating)):
        if int(rating[iid]) >= 0:
            print(uid, iid, rating[iid], sep="\t", file=outfile)
    uid += 1

infile.close()
outfile.close()

# convert user file -----------------------------------------------------------

infile = open(os.path.join(pwd, 'sushi3.udata'), 'r')
outfile = open(os.path.join(target, stem + '.user'), 'w')

print(
    '# User feature file for sushi3 data sets\n'
    '#\n'
    '# The number of users is 5000.\n'
    '#\n'
    '# Format\n'
    '# ------\n'
    '# user : int\n'
    '#     user id of the users which is compatible with the event file.\n'
    '# original_uid : int\n'
    '#     uid in the original data\n'
    '# gender : int {0:male, 1:female}\n'
    '#     gender of the user\n'
    '# age : int {0:15-19, 1:20-29, 2:30-39, 3:40-49, 4:50-59, 5:60-}\n'
    '#     age of the user\n'
    '# answer_time : int\n'
    '#     the total time need to fill questionnaire form\n'
    '# child_prefecture : int {0, 1, ..., 47}\n'
    '#     prefecture ID at which you have been the most long lived\n'
    '#     until 15 years old\n'
    '# child_region : int {0, 1, ..., 11}\n'
    '#     region ID at which you have been the most long lived\n'
    '#     until 15 years old\n'
    '# child_ew : int {0: Eastern, 1: Western}\n'
    '#     east/west ID at which you have been the most long lived\n'
    '#     until 15 years old\n'
    '# current_prefecture : int {0, 1, ..., 47}\n'
    '#     prefecture ID at which you currently live\n'
    '# current_region : int {0, 1, ..., 11}\n'
    '#     regional ID at which you currently live\n'
    '# current_ew : int {0: Eastern, 1: Western}\n'
    '#     east/west ID at which you currently live\n'
    "# moved : int {0: don't move, 1: move}\n"
    '#     whether child_prefecture and current_prefecture are equal or not\n',
    end='', file=outfile)

uid = 0
for line in infile.readlines():
    user_feature = line.rstrip('\r\n').split("\t")
    print(uid, "\t".join(user_feature), sep="\t", file=outfile)
    uid += 1

infile.close()
outfile.close()

# convert item file -----------------------------------------------------------
infile = io.open(os.path.join(pwd, 'sushi3.idata'), 'r', encoding='utf-8')
outfile = io.open(os.path.join(target, stem + '.item'), 'w', encoding='utf-8')

print(
    '# Item feature file for sushi3 data sets.\n'
    '#\n'
    '# The number of movies is 100.\n'
    '#\n'
    '# Format\n'
    '# ------\n'
    '# item : int\n'
    '#     item id of the movie which is compatible with the event file.\n'
    '# name : str, encoding=utf-8\n'
    '#     title of the movie with release year\n'
    '# maki : int {0:maki, 1:otherwise}\n'
    '#     whether a style of the sushi is *maki* or not\n'
    '# seafood : int {0:seafood, 1:otherwise}\n'
    '#     whether seafood or not\n'
    '# genre : int {0, ..., 8}\n'
    '#     the genre of the sushi *neta*\n'
    '#     0:aomono (blue-skinned fish)\n'
    '#     1:akami (red meat fish)\n'
    '#     2:shiromi (white-meat fish)\n'
    '#     3:tare (something like baste; for eel or sea eel)\n'
    '#     4:clam or shell\n'
    '#     5:squid or octopus\n'
    '#     6:shrimp or crab\n'
    '#     7:roe\n'
    '#     8:other seafood\n'
    '#     9:egg\n'
    '#    10:meat other than fish\n'
    '#    11:vegetables\n'
    '# heaviness : float, range=[0-4], 0:heavy/oily\n'
    '#     mean of the heaviness/oiliness/*kotteri* in taste,\n'
    '# frequency : float, range=[0-3], 3:frequently eat\n'
    '#     how frequently the user eats the SUSHI,\n'
    '# price : float, range=[1-5], 5:expensive\n'
    '#     maki and other style sushis are normalized separately\n'
    '# supply : float, range=[0-1]\n'
    '#    the ratio of shops that supplies the sushi\n',
    end='', file=outfile)

for line in infile.readlines():
    item_feature = line.rstrip('\r\n').split("\t")
    print("\t".join(item_feature), sep="\t", file=outfile)

infile.close()
outfile.close()
