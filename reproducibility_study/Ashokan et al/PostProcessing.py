# this is the helper file to learn the posterior adjustments to the expected ratings for the privileged and unprivileged users

import numpy as np
from collections import defaultdict
import FairnessMetrics
from lenskit.metrics.predict import rmse as rmse
import copy


def calculate_posterior_adjustments(train, test, privileged_group, unprivileged_group, level=1.0, verbose=False):
    '''
    function returns a list of adjustments such that a) the unprivileged groups and privileged groups ratings are
    adjusted. specifically, for each item we calculate the difference between average expected and actual rating
    from the training set, and use these differences as adjustment percentages
    these can be applied posterior to the test set predictions to try to lower the fairness bias
    Args:
        train: the training rating observations
        test: the test rating observations
        privileged_group:  the information which user types are considered privileged
        unprivileged_group: the information which user types are considered unprivileged
        level: the level to which the ratings are adjusted. defaults to 1 (which corresponds to adjusting the test
        ratings with the differences learned on the training data

    Returns: the post-processed, adjusted ratings and recommendations
    '''

    # first, let's learn the necessary posterior adjustments on the training set
    posterior_adjustments_privileged = defaultdict(list)
    posterior_adjustments_unprivileged = defaultdict(list)

    unique_items = train['item'].unique()

    for item in unique_items:
        ratings_item_subset = train[train.item == item]

        if ratings_item_subset.empty :
            continue
        ratings_privileged_subset = ratings_item_subset[ratings_item_subset.user.isin(privileged_group)]
        ratings_unprivileged_subset = ratings_item_subset[ratings_item_subset.user.isin(unprivileged_group)]

        if ratings_privileged_subset.empty:
            continue
        if ratings_unprivileged_subset.empty:
            continue

        # calculate the dufferences if privileged and unprivileged groups
        E_g_r = np.mean(ratings_privileged_subset.rating)

        E_g_y = np.mean(ratings_privileged_subset.prediction)

        # calculate the adjustments and append to list
        posterior_adjustments_privileged[item].append(E_g_r - E_g_y)
        if verbose:
            print("append following adjustment for item ", item, ": ", (E_g_r - E_g_y))

        E_notg_r = np.mean(ratings_unprivileged_subset.rating)
        E_notg_y = np.mean(ratings_unprivileged_subset.prediction)

        if verbose:
            print("E_notg_r: ", E_notg_r, ", E_notg_y: ", E_notg_y, ' append to unprivileged adjustment: ', (E_notg_r - E_notg_y))

        posterior_adjustments_unprivileged[item].append(E_notg_r - E_notg_y)
        if verbose:
            print("found difference E_notg_y - E_notg_r for item ", item, ": ", (E_notg_y - E_notg_r))
            print("append following adjustment for item ", item, ": ", (E_notg_r - E_notg_y))

    # test check: given the current adjustments, does the value and/or absolute unfairness decrease for the test set?
    temp_train = copy.deepcopy(train)

    for item in unique_items:
        if verbose:
            print("adjusting the initial test set to calculate new fairness metrics")
        ratings_item_subset = temp_train[temp_train.item == item]

        if ratings_item_subset.empty:
            continue
        ratings_privileged_subset = ratings_item_subset[ratings_item_subset.user.isin(privileged_group)]
        ratings_unprivileged_subset = ratings_item_subset[ratings_item_subset.user.isin(unprivileged_group)]

        if ratings_privileged_subset.empty:
            continue
        if ratings_unprivileged_subset.empty:
            continue

        temp_train.loc[(temp_train.item == item) & (temp_train.user.isin(privileged_group)), 'prediction'] += posterior_adjustments_privileged.get(item)

        temp_train.loc[(temp_train.item == item) & (temp_train.user.isin(unprivileged_group)), 'prediction'] += posterior_adjustments_unprivileged.get(
                item)

    # calculate the various fairness metrics on the adjusted train predictions
    ii_u_val, ii_u_abs, ii_u_under, ii_u_over, ii_u_par, ii_u_disparate = FairnessMetrics.get_recommender_fairness_metrics_from_predictions(train,
                                                                           privileged_group, unprivileged_group)

    ii_u_val_post, ii_u_abs_post, ii_u_under_post, ii_u_over_post, ii_u_par_post, ii_u_disparate_post, = FairnessMetrics.get_recommender_fairness_metrics_from_predictions(temp_train,
                                                                           privileged_group, unprivileged_group)

    rmse_pre = rmse(train['prediction'], train['rating'])
    rmse_post = rmse(temp_train['prediction'], temp_train['rating'])

    if verbose:
        print('comparison of before u_val: ', ii_u_val, " and after adjustement u_val: ", ii_u_val_post)
        print('comparison of before u_abs: ', ii_u_abs, " and after adjustement u_abs: ", ii_u_abs_post)
        print('comparison of before u_par: ', ii_u_par, " and after adjustement u_par: ", ii_u_par_post)
        print('comparison of before rmse: ', rmse_pre, " and after adjustement rmse: ", rmse_post)

    ## then, apply these adjustments on the test set and return it
    test_adjusted = test.copy()

    for item in unique_items:

        ratings_item_subset = test_adjusted[test_adjusted.item == item]

        if ratings_item_subset.empty:
            continue
        ratings_privileged_subset = ratings_item_subset[ratings_item_subset.user.isin(privileged_group)]
        ratings_unprivileged_subset = ratings_item_subset[ratings_item_subset.user.isin(unprivileged_group)]

        if ratings_privileged_subset.empty:
            continue
        if ratings_unprivileged_subset.empty:
            continue

        # adjust the predictions for the privileged group
        if posterior_adjustments_privileged.get(item) is None:
            continue
        test_adjusted.loc[(test_adjusted.item == item) & (test_adjusted.user.isin(privileged_group)), 'prediction'] += \
            (level*posterior_adjustments_privileged.get(item)[0])

        # adjust the predictions for the unprivileged group
        if posterior_adjustments_unprivileged.get(item) is None:
            continue
        test_adjusted.loc[(test_adjusted.item == item) & (test_adjusted.user.isin(unprivileged_group)), 'prediction'] += \
            (level*posterior_adjustments_unprivileged.get(item)[0])

    rmse_pre = rmse(test['prediction'], test['rating'])
    rmse_post = rmse(test_adjusted['prediction'], test_adjusted['rating'])
    if verbose:
        print('test set: comparison of before rmse: ', rmse_pre, " and after adjustement rmse: ", rmse_post)

    return test_adjusted


def calculate_posterior_parity_adjustments(train, test, privileged_group, unprivileged_group, split=False):
    '''
    function returns a list of adjustments such that a) the unprivileged groups and privileged groups ratings are
    adjusted. specifically, for each item we calculate the difference between average expected and actual rating
    from the training set, and use these differences as adjustment percentages
    these can be applied posterior to the test set predictions to try to lower the fairness bias
    Args:
        train: the training rating observations
        test: the test rating observations
        privileged_group:  the information which user types are considered privileged
        unprivileged_group: the information which user types are considered unprivileged
        split: should we split the parity difference between the privileged and unprivileged group, i.e., add
        half to one and subtract half from the other (True), or should we only adjust the full parity difference
        from the privileged group (False)

    Returns: the post-processed, adjusted ratings and recommendations
    '''


    # first, let's learn the necessary posterior adjustments on the training set
    # learn the absolute difference in average scores between users

    u_par_diff = np.nanmean(train[train.user.isin(privileged_group)].prediction) - \
                            np.nanmean(train[train.user.isin(unprivileged_group)].prediction)

    # then, apply these adjustments on the test set and return it
    test_adjusted = test.copy()

    print('adjust privileged predictions by ', -1*u_par_diff)

    # adjust all predictions accordingly
    # note: we can either add the full difference to one group, or split the difference and add accordingly
    if split:
        test_adjusted.loc[test_adjusted.user.isin(privileged_group), 'prediction'] = test_adjusted.loc[
                                                                                         test_adjusted.user.isin(
                                                                                             privileged_group), 'prediction'] - 0.5* u_par_diff
        test_adjusted.loc[test_adjusted.user.isin(unprivileged_group), 'prediction'] = test_adjusted.loc[
                                                                                     test_adjusted.user.isin(
                                                                                         unprivileged_group), 'prediction'] + 0.5 * u_par_diff

    else:
        test_adjusted.loc[test_adjusted.user.isin(privileged_group),'prediction'] = \
            test_adjusted.loc[test_adjusted.user.isin(privileged_group),'prediction'] - u_par_diff

    return test_adjusted

