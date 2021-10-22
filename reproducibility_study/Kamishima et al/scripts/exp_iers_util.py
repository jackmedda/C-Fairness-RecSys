#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Common part of experimental scripts
"""

from __future__ import (
    print_function,
    division,
    absolute_import)
from six.moves import xrange

# =============================================================================
# Imports
# =============================================================================

import json
import logging
import os
import sys
import datetime

import numpy as np

from kamrecsys import __version__ as kamrecsys_version
from kamrecsys.model_selection import ShuffleSplitWithinGroups
from kamrecsys.utils import get_system_info, get_version_info, json_decodable
from kamiers import __version__ as kamiers_version

# =============================================================================
# Metadata variables
# =============================================================================

__author__ = "Toshihiro Kamishima ( http://www.kamishima.net/ )"
__date__ = "2018-01-04"
__version__ = "1.0.0"
__copyright__ = "Copyright (c) 2017 Toshihiro Kamishima all rights reserved."
__license__ = "MIT License: http://www.opensource.org/licenses/mit-license.php"

# =============================================================================
# Public symbols
# =============================================================================

__all__ = ['do_task']

# =============================================================================
# Constants
# =============================================================================

# =============================================================================
# Variables
# =============================================================================

# =============================================================================
# Functions
# =============================================================================


def training(rec, data, sen):
    """
    training model

    Parameters
    ----------
    rec : EventScorePredictor
        recommender object
    data : :class:`kamrecsys.data.EventWithScoreData`
        training data
    sen : array, size=(n_events,), dtype=int
        binary sensitive features

    Returns
    -------
    res_info : dict
        Info of training results
    """

    # info of results
    res_info = {}

    # set starting time
    start_time = datetime.datetime.now()
    start_utime = os.times()[0]
    res_info['start_time'] = start_time.isoformat()
    logger.info("training_start_time = " + res_info['start_time'])

    # create and learning model
    rec.fit(data, sen)

    # set end and elapsed time
    end_time = datetime.datetime.now()
    end_utime = os.times()[0]
    res_info['end_time'] = end_time.isoformat()
    res_info['elapsed_time'] = end_time - start_time
    res_info['elapsed_utime'] = end_utime - start_utime
    logger.info("training_end_time = " + res_info['end_time'])
    logger.info("training_elapsed_time = " + str(res_info['elapsed_time']))
    logger.info("training_elapsed_utime = " + str(res_info['elapsed_utime']))

    # preserve optimizer's outputs
    res_info.update(rec.fit_results_)

    return res_info


def testing(rec, ev, sen):
    """
    test and output results

    Parameters
    ----------
    rec : EventScorePredictor
        trained recommender
    ev : array, size=(n_events, 2), dtype=int
        array of events in external ids
    sen : array, size=(n_events,), dtype=int
        binary sensitive features

    Returns
    -------
    esc : array, shape=(n_events,), dtype=float
        Estimated scores
    res_info : dict
        Info of training results
    """

    # info of results
    res_info = {}

    # set starting time
    start_time = datetime.datetime.now()
    start_utime = os.times()[0]
    res_info['start_time'] = start_time.isoformat()
    logger.info("test_start_time = " + res_info['start_time'])

    # prediction
    esc = rec.predict(ev, sen)

    # set end and elapsed time
    end_time = datetime.datetime.now()
    end_utime = os.times()[0]
    res_info['end_time'] = start_time.isoformat()
    res_info['elapsed_time'] = end_time - start_time
    res_info['elapsed_utime'] = end_utime - start_utime
    logger.info("test_end_time = " + res_info['end_time'])
    logger.info("test_elapsed_time = " + str(res_info['elapsed_time']))
    logger.info("test_elapsed_utime = " + str(res_info['elapsed_utime']))

    # preserve test info
    res_info['n_events'] = ev.shape[0]

    return esc, res_info


def holdout_test(info, load_data):
    """
    tested on specified hold-out test data

    Parameters
    ----------
    info : dict
        Information about the target task
    load_data : function
        function for loading data
    """

    # prepare training data
    train_data = load_data(info['training']['file'], info)
    train_sen = train_data.event_feature['sensitive']

    # prepare test data
    if info['test']['file'] is None:
        raise IOError('hold-out test data is required')
    test_data = load_data(info['test']['file'], info)
    test_ev = test_data.to_eid_event(test_data.event)
    test_sen = test_data.event_feature['sensitive']

    # set information about data and conditions
    info['training']['n_events'] = train_data.n_events
    info['training']['n_users'] = (
        train_data.n_objects[train_data.event_otypes[0]])
    info['training']['n_items'] = (
        train_data.n_objects[train_data.event_otypes[1]])

    info['test']['file'] = info['training']['file']
    info['test']['n_events'] = test_data.n_events
    info['test']['n_users'] = (
        test_data.n_objects[test_data.event_otypes[0]])
    info['test']['n_items'] = (
        test_data.n_objects[test_data.event_otypes[1]])

    info['condition']['n_folds'] = 1
    if train_sen.ndim == 1:
        info['condition']['n_sensitives'] = 1
        info['condition']['n_sensitive_values'] = np.unique(train_sen).size
    elif sen.ndim == 2:
        info['condition']['n_sensitives'] = sen.shape[1]
        info['condition']['n_sensitive_values'] = [
            np.unique(train_sen[:, s]).size
            for s in xrange(train_sen.shape[1])]

    # training
    rec = info['model']['recommender'](**info['model']['options'])
    training_info = training(rec, train_data, train_sen)
    info['training']['results'] = {'0': training_info}

    # test
    esc, test_info = testing(rec, test_ev, test_sen)
    info['test']['results'] = {'0': test_info}

    # set predicted result
    info['prediction']['event'] = test_data.to_eid_event(test_data.event)
    info['prediction']['true'] = test_data.score
    info['prediction']['predicted'] = esc
    info['prediction']['sensitive'] = test_sen
    if test_data.event_feature is not None:
        ef_names = list(test_data.event_feature.dtype.names)
        ef_names.remove('sensitive')
        if len(ef_names) > 0:
            info['prediction']['event_feature'] = {
                k: test_data.event_feature[k] for k in ef_names}


def cv_test(info, load_data, target_fold=None):
    """
    tested on specified hold-out test data

    Parameters
    ----------
    info : dict
        Information about the target task
    load_data : function
        function for loading data
    target_fold : int or None
        If None, all folds are processed; otherwise, specify the fold number
        to process.
    """

    # prepare training data
    data = load_data(info['training']['file'], info)
    n_folds = info['condition']['n_folds']
    ev = data.to_eid_event(data.event)
    sen = data.event_feature['sensitive']

    # set information about data and conditions
    info['training']['n_events'] = data.n_events
    info['training']['n_users'] = data.n_objects[data.event_otypes[0]]
    info['training']['n_items'] = data.n_objects[data.event_otypes[1]]
    info['test']['file'] = info['training']['file']
    info['test']['n_events'] = info['training']['n_events']
    info['test']['n_users'] = info['training']['n_users']
    info['test']['n_items'] = info['training']['n_items']
    if sen.ndim == 1:
        info['condition']['n_sensitives'] = 1
        info['condition']['n_sensitive_values'] = np.unique(sen).size
    elif sen.ndim == 2:
        info['condition']['n_sensitives'] = sen.shape[1]
        info['condition']['n_sensitive_values'] = [
            np.unique(train_sen[:, s]).size
            for s in xrange(train_sen.shape[1])]

    # cross validation
    fold = 0
    info['training']['results'] = {}
    info['test']['results'] = {}
    info['prediction']['event'] = {}
    info['prediction']['true'] = {}
    info['prediction']['predicted'] = {}
    info['prediction']['sensitive'] = {}
    info['prediction']['mask'] = {}
    if data.event_feature is not None:
        info['prediction']['event_feature'] = {
            k: {} for k in data.event_feature.dtype.names}
        if 'sensitive' in info['prediction']['event_feature']:
            del info['prediction']['event_feature']['sensitive']

    # 20% of ratings per user is used for test
    cv = ShuffleSplitWithinGroups(n_splits=n_folds, test_size=0.2)
    for train_i, test_i in cv.split(ev, groups=ev[:, 0]):
        # in an one-fold mode, non target folds are ignored
        if target_fold is not None and fold != target_fold:
            fold += 1
            continue

        # training
        logger.info("training fold = " + str(fold + 1) + " / " + str(n_folds))
        training_data = data.filter_event(train_i)
        training_sen = training_data.event_feature['sensitive']
        rec = info['model']['recommender'](**info['model']['options'])
        training_info = training(rec, training_data, training_sen)
        info['training']['results'][str(fold)] = training_info

        # test
        logger.info("test fold = " + str(fold + 1) + " / " + str(n_folds))
        esc, test_info = testing(rec, ev[test_i], sen[test_i])
        info['test']['results'][str(fold)] = test_info
        info['prediction']['event'][str(fold)] = ev[test_i, :]
        info['prediction']['true'][str(fold)] = data.score[test_i]
        info['prediction']['predicted'][str(fold)] = esc
        info['prediction']['sensitive'][str(fold)] = sen[test_i]
        info['prediction']['mask'][str(fold)] = test_i
        if data.event_feature is not None:
            ef_names = list(data.event_feature.dtype.names)
            if 'sensitive' in ef_names:
                ef_names.remove('sensitive')
            if len(ef_names) > 0:
                for k in ef_names:
                    info['prediction']['event_feature'][k][str(fold)] = (
                        data.event_feature[k][test_i])

        fold += 1


def do_task(info, load_data, target_fold=None):
    """
    Main task

    Parameters
    ----------
    info : dict
        Information about the target task
    load_data : function
        function for loading data
    target_fold : int or None
        specify the fold number to process in an one-fold mode
    """

    # suppress warnings in numerical computation
    np.seterr(all='ignore')

    # initialize random seed
    np.random.seed(info['model']['options']['random_state'])

    # check target fold number
    if (target_fold is not None and
            (target_fold < 0 or target_fold >= info['condition']['n_folds'])):
        raise TypeError(
            "Illegal specification of the target fold: {:s}".format(
                str(target_fold)))

    # update information dictionary
    rec = info['model']['recommender']
    info['model']['task_type'] = rec.task_type
    info['model']['explicit_ratings'] = rec.explicit_ratings
    info['model']['name'] = rec.__name__
    info['model']['module'] = rec.__module__

    info['environment']['script'] = {
        'name': os.path.basename(sys.argv[0]), 'version': __version__}
    info['environment']['system'] = ""#get_system_info()
    info['environment']['version'] = get_version_info()
    info['environment']['version']['kamrecsys'] = kamrecsys_version
    info['environment']['version']['kamiers'] = kamiers_version

    # select validation scheme
    if info['condition']['scheme'] == 'holdout':
        holdout_test(info, load_data)
    elif info['condition']['scheme'] == 'cv':
        cv_test(info, load_data, target_fold=target_fold)
    else:
        raise TypeError("Invalid validation scheme: {0:s}".format(opt.method))

    # output information
    outfile = info['condition']['out_file']
    info['condition']['out_file'] = str(outfile)
    json_decodable(info)
    outfile.write(json.dumps(info))
    if outfile is not sys.stdout:
        outfile.close()


# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Module initialization
# =============================================================================

# init logging system
logger = logging.getLogger('exp_iers')
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

# =============================================================================
# Test routine
# =============================================================================


def _test():
    """ test function for this module
    """

    # perform doctest
    import sys
    import doctest

    doctest.testmod()

    sys.exit(0)


# Check if this is call as command script

if __name__ == '__main__':
    _test()
