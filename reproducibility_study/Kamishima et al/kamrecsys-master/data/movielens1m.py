#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert MovieLens 1M data sets to KamRecSys Sample format

Instruction
-----------

1. Download original file, ml-1m.zip, from `MovieLens Data Sets
   <http://www.grouplens.org/node/73>`_.
2. Unpack this ml-1m.zip, and place the following files at this directory:
   ratings.dat, users.dat, movies.dat
3. Run this script. As default, converted files are generated at
   ../kamrecsys/datasets/data/ directory. If you want change the target
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
import re

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
stem = 'movielens1m'
pwd = os.path.dirname(__file__)
if len(sys.argv) >= 2:
    target = sys.argv[1]
else:
    target = os.path.join(pwd, '..', "kamrecsys", 'datasets', 'data')

# convert event files ---------------------------------------------------------

infile = open(os.path.join(pwd, 'ratings.dat'), 'r')
outfile = open(os.path.join(target, stem + '.event'), 'w')

print(
    "# Movielens 1M data set\n"
    "#\n"
    "# Original files are distributed by the Grouplens Research Project at "
    "the site:\n"
    "# http://www.grouplens.org/node/73\n"
    "# To use this data, follow the license permitted by the original "
    "distributor.\n"
    "#\n"
    "# This data set consists of:\n"
    "# * 1,000,209 ratings (1-5) from 6040 users on 3706 movies.\n"
    "# * Each user has rated at least 20 movies.\n"
    "#\n"
    "# Notes\n"
    "# -----\n"
    "# There are 3883 movies in an original file, but 3706 movies are "
    "actually\n"
    "# rated.\n"
    "#\n"
    "# Format\n"
    "# ------\n"
    "# user : int\n"
    "#     user id of the user who rated the movie\n"
    "# item : int\n"
    "#     item id of the movie rated by the user\n"
    "# score : int\n"
    "#     rating score whose range is {1, 2, 3, 4, 5}\n"
    "# timestamp : int\n"
    "#     represented in seconds since the epoch as returned by time(2)\n",
    end='', file=outfile)

for line in infile.readlines():
    f = line.rstrip('\r\n').split("::")
    print('\t'.join(f), file=outfile)

infile.close()
outfile.close()

# convert user file -----------------------------------------------------------

infile = open(os.path.join(pwd, 'users.dat'), 'r')
outfile = open(os.path.join(target, stem + '.user'), 'w')

print(
    '# User feature file for ``movielens1m.event``.\n'
    '#\n'
    '# The number of users is 6040.\n'
    '#\n'
    '# Format\n'
    '# ------\n'
    '# user : int\n'
    '#     user id of the users which is compatible with the event file.\n'
    '# gender : int\n'
    '#     gender of the user, {0:male, 1:female}\n'
    '# age : int, {0, 1,..., 6}\n'
    '#     age of the user, where\n'
    '#     1:"Under 18", 18:"18-24", 25:"25-34", 35:"35-44", 45:"45-49",\n'
    '#     50:"50-55", 56:"56+"\n'
    '# occupation : int, {0,1,...,20}\n'
    '#     the number indicates the occupation of the user as follows:\n'
    '#     0:"other" or not specified, 1:"academic/educator",\n'
    '#     2:"artist", 3:"clerical/admin", 4:"college/grad student"\n'
    '#     5:"customer service", 6:"doctor/health care", '
    '7:"executive/managerial"\n'
    '#     8:"farmer", 9:"homemaker", 10:"K-12 student", 11:"lawyer",\n'
    '#     12:"programmer", 13:"retired", 14:"sales/marketing", '
    '15:"scientist",\n'
    '#     16:"self-employed", 17:"technician/engineer", '
    '18:"tradesman/craftsman",\n'
    '#     19:"unemployed", 20:"writer"\n'
    '# zip : str, length=5\n'
    '#     zip code of 5 digits, which represents the residential area of '
    'the user\n', end='', file=outfile)

age = {'1': 0, '18': 1, '25': 2, '35': 3, '45': 4, '50': 5, '56': 6}

for line in infile.readlines():
    f = line.rstrip('\r\n').split("::")
    print(f[0], end='\t', file=outfile)
    if f[1] == 'M':
        print('0', end='\t', file=outfile)
    elif f[1] == 'F':
        print('1', end='\t', file=outfile)
    else:
        raise ValueError
    print(age[f[2]], f[3], f[4][0:5], sep='\t', file=outfile)

infile.close()
outfile.close()

# convert item files ----------------------------------------------------------

infile = io.open(os.path.join(pwd, 'movies.dat'), 'r', encoding='cp1252')
outfile = io.open(os.path.join(target, stem + '.item'), 'w', encoding='utf_8')

print(
    '# Item feature file for ``movielens1m.event``.\n'
    '#\n'
    '# The number of movies is 3883.\n'
    '#\n'
    '# Format\n'
    '# ------\n'
    '# item : int\n'
    '#     item id of the movie which is compatible with the event file.\n'
    '# name : str, length=[7, 81]\n'
    '#     title of the movie with release year\n'
    '# year : int\n'
    '#     released year\n'
    '# genre : binary_int * 18\n'
    '#     18 binary numbers represents a genre of the movie. 1 if the movie '
    'belongs\n'
    '#     to the genre; 0 other wise. All 0 implies unknown. Each column\n'
    '#     corresponds to the following genres:\n'
    "#     0:Action, 1:Adventure, 2:Animation, 3:Children's, 4:Comedy, "
    '5:Crime,\n'
    '#     6:Documentary, 7:Drama, 8:Fantasy, 9:Film-Noir, 10:Horror, '
    '11:Musical,\n'
    '#     12:Mystery, 13:Romance, 14:Sci-Fi, 15:Thriller, 16:War, '
    '17:Western\n', end='', file=outfile)

genre = {"Action": 0, "Adventure": 1, "Animation": 2, "Children's": 3,
         "Comedy": 4, "Crime": 5, "Documentary": 6, "Drama": 7,
         "Fantasy": 8, "Film-Noir": 9, "Horror": 10, "Musical": 11,
         "Mystery": 12, "Romance": 13, "Sci-Fi": 14, "Thriller": 15,
         "War": 16, "Western": 17}

year_p = re.compile(r'\((\d\d\d\d)\)$')

for line in infile.readlines():
    f = line.rstrip('\r\n').split("::")
    # for buggy character in original file
    if f[0] == '3845':
        f[1] = re.sub('&#8230;', '\u2026', f[1])
    print(f[0], f[1], sep='\t', end='\t', file=outfile)
    year = year_p.search(f[1]).group(1)
    print(year, end='\t', file=outfile)
    d = f[2].split('|')
    o = ['0'] * 18
    for i in d:
        o[genre[i]] = '1'
    print('\t'.join(o), file=outfile)

infile.close()
outfile.close()
