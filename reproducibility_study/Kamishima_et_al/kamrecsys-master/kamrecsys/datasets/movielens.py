#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Load sample Movielens data sets
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
import codecs
import logging
import numpy as np

from ..data import EventWithScoreData
from . import SAMPLE_PATH, load_event_with_score, event_dtype_timestamp

# =============================================================================
# Public symbols
# =============================================================================

__all__ = []

# =============================================================================
# Constants
# =============================================================================

# Conversion tables for mapping the numbers to names for the ``movielens100k``
# data set. available tables are ``user_occupation`` and ``item_genre``.
MOVIELENS100K_INFO = {
    'user_occupation': np.array([
        'None', 'Other', 'Administrator', 'Artist', 'Doctor', 'Educator',
        'Engineer', 'Entertainment', 'Executive', 'Healthcare', 'Homemaker',
        'Lawyer', 'Librarian', 'Marketing', 'Programmer', 'Retired',
        'Salesman', 'Scientist', 'Student', 'Technician', 'Writer']),
    'item_genre': np.array([
        "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
        "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"])
}

# Conversion tables for mapping the numbers to names for the ``movielens1m``
# data set. available tables are ``user_age``, ``user_occupation`` and
# ``item_genre``.
MOVIELENS1M_INFO = {
    'user_age': np.array([
        'Under 18', '18-24', '25-34', '35-44', '45-49', '50-55', '56+']),
    'user_occupation': np.array([
        'Other or Not Specified', 'Academic/Educator', 'Artist',
        'Clerical/Admin', 'College/Grad Student', 'Customer Service',
        'Doctor/Health Care', 'Executive/Managerial', 'Farmer', 'Homemaker',
        'K-12 Student', 'Lawyer', 'Programmer', 'Retired', 'Sales/Marketing',
        'Scientist', 'Self-Employed', 'Technician/Engineer',
        'Tradesman/Craftsman', 'Unemployed', 'Writer']),
    'item_genre': np.array([
        "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
        "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"])
}

# =============================================================================
# Module variables
# =============================================================================

# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Functions
# =============================================================================


def load_movielens100k(infile=None, event_dtype=event_dtype_timestamp):
    """ load the MovieLens 100k data set

    Original file ``ml-100k.zip`` is distributed by the Grouplens Research
    Project at the site:
    `MovieLens Data Sets <http://www.grouplens.org/node/73>`_.

    Parameters
    ----------
    infile : optional, file or str
        input file if specified; otherwise, read from default sample directory.
    event_dtype : np.dtype
        dtype of extra event features. as default, it consists of only a
        ``timestamp`` feature.

    Returns
    -------
    data : :class:`kamrecsys.data.EventWithScoreData`
        sample data

    Notes
    -----
    Format of events:

    * each event consists of a vector whose format is [user, item].
    * 100,000 events in total
    * 943 users rate 1682 items (=movies)
    * dtype=int

    Format of scores:

    * one score is given to each event
    * domain of score is [1.0, 2.0, 3.0, 4.0, 5.0]
    * dtype=float

    Default format of event_features ( `data.event_feature` ):
    
    timestamp : int
        UNIX seconds since 1/1/1970 UTC

    Format of user's feature ( `data.feature[0]` ):

    age : int
        age of the user
    gender : int
        gender of the user, {0:male, 1:female}
    occupation : int
        the number indicates the occupation of the user as follows:
        0:None, 1:Other, 2:Administrator, 3:Artist, 4:Doctor, 5:Educator,
        6:Engineer, 7:Entertainment, 8:Executive, 9:Healthcare, 10:Homemaker,
        11:Lawyer, 12:Librarian, 13:Marketing, 14:Programmer, 15:Retired,
        16:Salesman, 17:Scientist, 18:Student, 19:Technician, 20:Writer
    zip : str, length=5
        zip code of 5 digits, which represents the residential area of the user

    Format of item's feature ( `data.feature[1]` ):

    name : str, length=[7, 81], dtype=np.dtype('S81')
        title of the movie with release year
    date : int * 3
        released date represented by a tuple (year, month, day)
    genre : np.dtype(i1) * 18
        18 binary numbers represents a genre of the movie. 1 if the movie
        belongs to the genre; 0 other wise. All 0 implies unknown. Each column
        corresponds to the following genres:
        Action, Adventure, Animation, Children's, Comedy, Crime, Documentary,
        Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi,
        Thriller, War, Western
    imdb : str, length=[0, 134], dtype=np.dtype('S134')
         URL for the movie at IMDb http://www.imdb.com
    """

    # load event file
    if infile is None:
        infile = os.path.join(SAMPLE_PATH, 'movielens100k.event')
    data = load_event_with_score(
        infile, n_otypes=2, event_otypes=(0, 1),
        score_domain=(1., 5., 1.), event_dtype=event_dtype)

    # load user's feature file
    infile = os.path.join(SAMPLE_PATH, 'movielens100k.user')
    fdtype = np.dtype([('age', int), ('gender', int),
                       ('occupation', int), ('zip', 'U5')])
    dtype = np.dtype([('eid', int), ('feature', fdtype)])
    x = np.genfromtxt(fname=infile, delimiter='\t', dtype=dtype)
    data.set_feature(0, x['eid'], x['feature'])

    # load item's feature file
    infile = os.path.join(SAMPLE_PATH, 'movielens100k.item')
    fdtype = np.dtype([('name', 'U81'),
                       ('day', int),
                       ('month', int),
                       ('year', int),
                       ('genre', 'i1', 18),
                       ('imdb', 'U134')])
    dtype = np.dtype([('eid', int), ('feature', fdtype)])
    x = np.genfromtxt(fname=infile, delimiter='\t', dtype=dtype,
                      converters={1: np.char.decode})
    data.set_feature(1, x['eid'], x['feature'])

    return data


def load_movielens_mini():
    """ load the MovieLens mini data set

    This data set is the subset of the data in the `movielens100k` data set.
    Users and items whose external ids are less or equal than 10 are collected.

    You can find the format of this data set in :func:`load_movielens100k`.
    Original item database contains 3,883 movies, but 3,706 movies were rated.

    Returns
    -------
    data : :class:`kamrecsys.data.EventWithScoreData`
        sample data

    Notes
    -----

    * 30 events in total
    * 8 users rate 10 items (=movies)
    """
    infile = os.path.join(SAMPLE_PATH, 'movielens_mini.event')
    return load_movielens100k(infile=infile)


def load_movielens1m(infile=None, event_dtype=event_dtype_timestamp):
    """ load the MovieLens 1m data set

    Original file ``ml-1m.zip`` is distributed by the Grouplens Research
    Project at the site:
    `MovieLens Data Sets <http://www.grouplens.org/node/73>`_.

    Parameters
    ----------
    infile : optional, file or str
        input file if specified; otherwise, read from default sample directory.
    event_dtype : np.dtype
        dtype of extra event features. as default, it consists of only a
        ``timestamp`` feature.

    Returns
    -------
    data : :class:`kamrecsys.data.EventWithScoreData`
        sample data

    Notes
    -----
    Format of events:

    * each event consists of a vector whose format is [user, item].
    * 1,000,209 events in total
    * 6,040 users rate 3,706 items (=movies)
    * dtype=int

    Format of scores:

    * one score is given to each event
    * domain of score is [1.0, 2.0, 3.0, 4.0, 5.0]
    * dtype=float

    Default format of event_features ( `data.event_feature` ):
    
    timestamp : int
        represented in seconds since the epoch as returned by time(2)

    Format of user's feature ( `data.feature[0]` ):

    gender : int
        gender of the user, {0:male, 1:female}
    age : int, {0, 1,..., 6}
        age of the user, where
        1:"Under 18", 18:"18-24", 25:"25-34", 35:"35-44", 45:"45-49",
        50:"50-55", 56:"56+"
    occupation : int, {0,1,...,20}
        the number indicates the occupation of the user as follows:
        0:"other" or not specified, 1:"academic/educator",
        2:"artist", 3:"clerical/admin", 4:"college/grad student"
        5:"customer service", 6:"doctor/health care", 7:"executive/managerial"
        8:"farmer", 9:"homemaker", 10:"K-12 student", 11:"lawyer",
        12:"programmer", 13:"retired", 14:"sales/marketing", 15:"scientist",
        16:"self-employed", 17:"technician/engineer", 18:"tradesman/craftsman",
        19:"unemployed", 20:"writer"
    zip : str, length=5
        zip code of 5 digits, which represents the residential area of the user

    Format of item's feature ( `data.feature[1]` ):

    name : str, length=[8, 82]
        title of the movie with release year
    year : int
        released year
    genre : binary_int * 18
        18 binary numbers represents a genre of the movie. 1 if the movie
        belongs to the genre; 0 other wise. All 0 implies unknown. Each column
        corresponds to the following genres:
        0:Action, 1:Adventure, 2:Animation, 3:Children's, 4:Comedy, 5:Crime,
        6:Documentary, 7:Drama, 8:Fantasy, 9:Film-Noir, 10:Horror, 11:Musical,
        12:Mystery, 13:Romance, 14:Sci-Fi, 15:Thriller, 16:War, 17:Western
    """

    # load event file
    if infile is None:
        infile = os.path.join(SAMPLE_PATH, 'movielens1m.event')
    data = load_event_with_score(
        infile, n_otypes=2, event_otypes=(0, 1),
        score_domain=(1., 5., 1.), event_dtype=event_dtype)

    # load user's feature file
    infile = os.path.join(SAMPLE_PATH, 'movielens1m.user')
    fdtype = np.dtype([('gender', int), ('age', int),
                       ('occupation', int), ('zip', 'U5')])
    dtype = np.dtype([('eid', int), ('feature', fdtype)])
    x = np.genfromtxt(fname=infile, delimiter='\t', dtype=dtype)
    data.set_feature(0, x['eid'], x['feature'])

    # load item's feature file
    infile = os.path.join(SAMPLE_PATH, 'movielens1m.item')
    fdtype = np.dtype([('name', 'U82'),
                       ('year', int),
                       ('genre', 'i1', 18)])
    dtype = np.dtype([('eid', int), ('feature', fdtype)])
    x = np.genfromtxt(fname=infile, delimiter='\t', dtype=dtype,
                      converters={1: np.char.decode})
    data.set_feature(1, x['eid'], x['feature'])

    return data


# =============================================================================
# Module initialization
# =============================================================================

# init logging system ---------------------------------------------------------
logger = logging.getLogger('kamrecsys')
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

# =============================================================================
# Test routine
# =============================================================================


def _test():
    """ test function for this module
    """

    # perform doctest
    import doctest

    doctest.testmod()

    sys.exit(0)


# Check if this is call as command script -------------------------------------

if __name__ == '__main__':
    _test()
