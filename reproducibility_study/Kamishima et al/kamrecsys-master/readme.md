KamRecSys
=========

* Author : [Toshihiro Kamishima](http://www.kamishima.net/)
* Copyright : Copyright (c) 2012 Toshihiro Kamishima all rights reserved.
* License : [MIT License](http://www.opensource.org/licenses/mit-license.php)

Description
-----------

`kamrecsys` is a python package of algorithms for recommender systems.

This package targets experimental algorithm used for research purpose.
We concentrate on the flexibility of data structure or easiness for implementing new algorithm rather than scalability or efficiency of algorithms.

Installation
------------

First, generate sample data sets that you need. Read a `readme.md` file in a `data` directory.
You then build and install by using a `setup.py` script.

Requirements
------------

We tested this module under the following packages:

* Python 3.5+ or Python 2.7.x
* NumPy
* SciPy
* scikit-learn
* six

Algorithms
----------

### Item Finder

* Matrix Factorization

    * Logistic Probabilistic Matrix Factorization (Explicit / Implicit Ratings)

### Score Predictor

* Matrix Factorization

    * Probabilistic Matrix Factorization

* Topic Model

    * Probabilistic Latent Semantic Analysis (Multinomial)

DataSets
--------

* Programming the Collective Intelligence by Toby Segaran

    * sample_movies

* [Movielens Data Sets](http://www.grouplens.org/node/73)

    * Movielens 100k Data Set
    * Movielens 1m Data Set

* [SUSHI Data Sets](http://www.kamishima.net/sushi/)

    * Sushi3b Score Data Set

* [Flixster Data Sets](http://www.cs.ubc.ca/~jamalim/datasets/)

    * Flixster Rating Data Set
