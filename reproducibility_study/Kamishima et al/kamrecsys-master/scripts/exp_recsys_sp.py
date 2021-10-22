#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experimentation script for Score Predictors

Input Format
------------

An input dataset is a tab separated file.  Each line corresponds to one 
rating behavior. Each column represents is as follows: 

* 1. A user represented by External-ID
* 2. An item rated by the user, represented by External-ID,
* 3. A rating score given by the user to the item
* 4. A timestamp of rating behavior, optional.

Output Format
-------------

Outputs of prediction are stored in a `json` formatted file.  Top-level keys 
of the outputs are as follows: 

* `condition` : experimental conditions of experimental schemes and datasets
* `environment` : hardware, system software, and experimental script
* `model` : model and its parameters used for prediction 
* `prediction` : predicted results, user-item pairs and predicted and true 
  rating scores. 
* `test` : conditions, time information in test
* `training` : conditions, time information in training

Options
=======

-i <INPUT>, --in <INPUT>
    specify training file name
-t <TEST>, --test <TEST>
    specify test file name
-o <OUTPUT>, --out <OUTPUT>
    specify output file name
-m <METHOD>, --method <METHOD>
    specify algorithm: default=pmf

    * pmf : probabilistic matrix factorization
    * plsam : pLSA (multinomial / use expectation in prediction)
    * plsamm : pLSA (multinomial / use mode in prediction)

-v <VALIDATION>, --validation <VALIDATION>
    validation scheme: default=holdout

    * holdout : tested on the specified hold-out data
    * cv : cross validation

-f <FOLD>, --fold <FOLD>
    the number of folds in cross validation, default=5
-n <FOLD_NO>, --fold-no <FOLD_NO>
    if specified, only the specified fold is tested.
--no-timestamp or --timestamp
    specify whether .event files has 'timestamp' information,
    default=timestamp
-d <DOMAIN>, --domain <DOMAIN>
    The domain of scores specified by three floats: min, max, increment
    default=auto
-C <C>, --lambda <C>
    regularization parameter, default=0.01.
-k <K>, --dim <K>
    the number of latent factors, default=1.
--alpha <ALPHA>
    smoothing parameter of multinomial pLSA
--tol <TOL>
    optimization parameter. the size of norm of gradient. default=1e-05.
--maxiter <MAXITER>
    maximum number of iterations is maxiter times the number of parameters.
-q, --quiet
    set logging level to ERROR, no messages unless errors
--rseed <RSEED>
    random number seed. if None, use /dev/urandom (default None)
-h, --help
    show this help message and exit
--version
    show program's version number and exit
"""

from __future__ import (
    print_function,
    division,
    absolute_import)

# =============================================================================
# Imports
# =============================================================================

import argparse
import logging
import sys

import numpy as np

from kamrecsys.datasets import event_dtype_timestamp, load_event_with_score

from exp_recsys_util import do_task

# =============================================================================
# Module metadata variables
# =============================================================================

__author__ = "Toshihiro Kamishima ( http://www.kamishima.net/ )"
__date__ = "2014-07-06"
__version__ = "5.0.0"
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


def load_data(fp, info):
    """
    load event with scores data

    Parameters
    ----------
    fp : string
        input file pointer
    info : dict
        Information about the target task

    Returns
    -------
    data : :class:`kamrecsys.data.EventWithScoreData`
        loaded data
    """

    has_timestamp = info['condition']['has_timestamp']
    score_domain = info['condition']['score_domain']

    # score_domain is explicitly specified?
    if np.all(np.array(score_domain) == 0):
        score_domain = None

    # load data
    data = load_event_with_score(
        fp,
        event_dtype=event_dtype_timestamp if has_timestamp else None,
        score_domain=score_domain)
    info['condition']['score_domain'] = data.score_domain

    return data


# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Main routine
# =============================================================================


def command_line_parser():
    """
    Parsing Command-Line Options
    
    Returns
    -------
    opt : argparse.Namespace
        Parsed command-line arguments
    """
    # import argparse
    # import sys

    # command-line option parsing
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)

    # common options
    ap.add_argument('--version', action='version',
                    version='%(prog)s ' + __version__)

    apg = ap.add_mutually_exclusive_group()
    apg.set_defaults(verbose=True)
    apg.add_argument('--verbose', action='store_true')
    apg.add_argument('-q', '--quiet', action='store_false', dest='verbose')

    ap.add_argument("--rseed", type=int, default=None)

    # basic file i/o
    ap.add_argument('-i', '--in', dest='infile', default=None,
                    type=argparse.FileType('rb'))
    ap.add_argument('infilep', nargs='?', metavar='INFILE', default=sys.stdin,
                    type=argparse.FileType('rb'))
    ap.add_argument('-o', '--out', dest='outfile', default=None,
                    type=argparse.FileType('w'))
    ap.add_argument('outfilep', nargs='?', metavar='OUTFILE',
                    default=sys.stdout, type=argparse.FileType('w'))
    ap.add_argument('-t', '--test', dest='testfile', default=None,
                    type=argparse.FileType('rb'))

    # script specific options
    ap.add_argument('-m', '--method', type=str, default='pmf',
                    choices=['pmf', 'plsam', 'plsamm'])
    ap.add_argument('-v', '--validation', type=str, default='holdout',
                    choices=['holdout', 'cv'])
    ap.add_argument('-f', '--fold', type=int, default=5)
    ap.add_argument('-n', '--fold-no', dest='fold_no', type=int, default=None)

    ap.add_argument('-d', '--domain', nargs=3, default=[0, 0, 0], type=float)
    apg = ap.add_mutually_exclusive_group()
    apg.set_defaults(timestamp=True)
    apg.add_argument('--no-timestamp', dest='timestamp', action='store_false')
    apg.add_argument('--timestamp', dest='timestamp', action='store_true')

    ap.add_argument('-C', '--lambda', dest='C', type=float, default=0.01)
    ap.add_argument('-k', '--dim', dest='k', type=int, default=1)
    ap.add_argument('--alpha', dest='alpha', type=float, default=1.0)
    ap.add_argument('--tol', type=float, default=1e-05)
    ap.add_argument('--maxiter', type=int, default=None)

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

    # disable logging messages by changing logging level
    if opt.verbose:
        logger.setLevel(logging.INFO)
        logging.getLogger('kamrecsys').setLevel(logging.INFO)

    # output option information
    logger.info("list of options:")
    for key_name, key_value in vars(opt).items():
        logger.info("{0}={1}".format(key_name, str(key_value)))

    return opt


def init_info(opt):
    """
    Initialize information dictionary

    Parameters
    ----------
    opt : argparse.Namespace
        Parsed command-line options

    Returns
    -------
    info : dict
        Information about the target task
    """

    info = {'condition': {}, 'environment': {}, 'training': {}, 'test': {},
            'model': {'options': {}}, 'prediction': {}}

    # files
    info['training']['file'] = opt.infile
    info['condition']['out_file'] = opt.outfile
    info['test']['file'] = opt.testfile

    # model
    if opt.method == 'pmf':
        from kamrecsys.score_predictor import PMF
        info['model']['method'] = 'probabilistic matrix factorization'
        info['model']['recommender'] = PMF
        info['model']['options'] = {'C': opt.C, 'k': opt.k, 'tol': opt.tol}
        if opt.maxiter is not None:
            info['model']['options']['maxiter'] = opt.maxiter
    elif opt.method == 'plsam':
        from kamrecsys.score_predictor import MultinomialPLSA
        info['model']['method'] = 'multionomial pLSA - expectation'
        info['model']['recommender'] = MultinomialPLSA
        info['model']['options'] = {
            'alpha': opt.alpha, 'k': opt.k, 'tol': opt.tol,
            'use_expectation': True}
        if opt.maxiter is not None:
            info['model']['options']['maxiter'] = opt.maxiter
    elif opt.method == 'plsamm':
        from kamrecsys.score_predictor import MultinomialPLSA
        info['model']['recommender'] = MultinomialPLSA
        info['model']['method'] = 'multionomial pLSA - mode'
        info['model']['options'] = {
            'alpha': opt.alpha, 'k': opt.k, 'tol': opt.tol,
            'use_expectation': False}
        if opt.maxiter is not None:
            info['model']['options']['maxiter'] = opt.maxiter
    else:
        raise TypeError(
            "Invalid method name: {0:s}".format(opt.method))
    info['model']['options']['random_state'] = opt.rseed

    # condition
    info['condition']['score_domain'] = list(opt.domain)
    info['condition']['has_timestamp'] = opt.timestamp
    info['condition']['explicit_rating'] = True
    info['condition']['scheme'] = opt.validation
    info['condition']['n_folds'] = opt.fold

    return info


def main():
    """ Main routine
    """
    # command-line arguments
    opt = command_line_parser()

    # collect assets and information
    info = init_info(opt)

    # do main task
    do_task(info, load_data, target_fold=opt.fold_no)


# top level -------------------------------------------------------------------
# init logging system
logger = logging.getLogger('exp_recsys')
logging.basicConfig(level=logging.INFO,
                    format='[%(name)s: %(levelname)s'
                           ' @ %(asctime)s] %(message)s')
logger.setLevel(logging.ERROR)
logging.getLogger('kamrecsys').setLevel(logging.ERROR)

# Call main routine if this is invoked as a top-level script environment.
if __name__ == '__main__':

    main()

    sys.exit(0)
