import numpy as np
import math

# this is the main class for defining the fairness metrics for recommender systems
# it mainly follows the paper of Yao and Huang 2017

def get_recommender_fairness_metrics_from_predictions(rating_predictions, privileged_group, unprivileged_group):
    '''
    the scores correspond to the y_ij values in the definition
    the ratings correspond to the r_ij values

    for now, privileged and unprivileged groups are given as lists of user ids belonging to the respective group

    first, for both the privileged and unprivileged group, calculate E_g (y)_j and E_not g (y)_j for each item j
    Args:
        rating_predictions: the predictions for the ratings
        privileged_group: the identifier of the privileged user group
        unprivileged_group: the identifier of the unprivileged user group

    Returns: different fairness metrics for rating predictions

    '''

    unique_items = rating_predictions['item'].unique()
    u_val = 0
    u_abs = 0
    u_under = 0
    u_over = 0
    u_par = 0
    u_disparate = 0

    count = 0
    for item in unique_items:
        print("next item: ", item)
        ratings_item_subset = rating_predictions[rating_predictions.item == item]

        if ratings_item_subset.empty:
            continue

        ratings_privileged_subset = ratings_item_subset[ratings_item_subset.user.isin(privileged_group)]
        ratings_unprivileged_subset = ratings_item_subset[ratings_item_subset.user.isin(unprivileged_group)]

        if ratings_privileged_subset.empty:
            continue
        if ratings_unprivileged_subset.empty:
            continue

        E_g_r = np.nanmean(ratings_privileged_subset.rating)
        E_notg_r = np.nanmean(ratings_unprivileged_subset.rating)

        E_g_y = np.nanmean(ratings_privileged_subset.prediction)
        E_notg_y = np.nanmean(ratings_unprivileged_subset.prediction)

        print("E_g_r: ", E_g_r)
        print("E_notg_r: ", E_notg_r)
        print("E_g_y: ", E_g_y)
        print("E_notg_y: ", E_notg_y)

        u_val += (np.abs((E_g_y - E_g_r) - (E_notg_y - E_notg_r)))
        u_abs += (np.abs(np.abs(E_g_y - E_g_r) - np.abs(E_notg_y - E_notg_r)))
        u_under += (np.abs(max(0, E_g_r - E_g_y) - max(0, E_notg_r - E_notg_y)))
        u_over += (np.abs(max(0, E_g_y - E_g_r) - max(0, E_notg_y - E_notg_r)))

        count += 1

    u_val = u_val / count
    u_abs = u_abs / count
    u_under = u_under / count
    u_over = u_over / count

    # finally, get the non-parity metric
    u_par = np.abs(np.nanmean(rating_predictions[rating_predictions.user.isin(privileged_group)].prediction) -
                   np.nanmean(rating_predictions[rating_predictions.user.isin(unprivileged_group)].prediction))

    # and the disparate impact metric
    u_disparate = np.nanmean(rating_predictions[rating_predictions.user.isin(privileged_group)].prediction) / \
                  np.nanmean(rating_predictions[rating_predictions.user.isin(unprivileged_group)].prediction)

    print("u_val: ", u_val)
    print("u_abs: ", u_abs)
    print("u_under: ", u_under)
    print("u_over: ", u_over)
    print("u_par: ", u_par)
    print("u_disparate: ", u_disparate)

    return u_val, u_abs, u_under, u_over, u_par, u_disparate


def individual_benefit_function(ratings, rating_scale_max=5):
    '''
    this defines the benefit function for each individual based on score (y_hat) and rating (r)
    Args:
        ratings: the rating scores
        rating_scale_max: the maximum rating score to normalize the values

    Returns: individual benefit

    '''

    b_i = 0
    items = ratings.item.unique()
    count = 0
    for item in items:
        rating_df = ratings[ratings.item == item]
        if rating_df.empty:
            continue
        temp = np.mean(abs(rating_df.prediction - rating_df.rating))
        if not math.isnan(temp):
            b_i += temp
            count += 1

    # print("b_i: ", b_i)
    if count == 0 | math.isnan(b_i):
        return rating_scale_max
    else:
        return rating_scale_max - (b_i / count)


def generalized_entropy_index(ratings, alpha=2, rating_scale_max=5):
    '''
    this implements the generalized entropy index based on Speicher et al. 2018

    Args:
        ratings: the ratings
        alpha: specification parameter for the generalized entropy index. see definition for common values
        rating_scale_max: the maximum rating score to normalize the values

    Returns: the value for the generalized entropy index for a given alpha

    '''

    # first, we get all of the individual benefits for the users
    num_user_count = 0
    unique_user_ids = ratings.user.unique()
    b_i_s = []
    for user_id in unique_user_ids:
        # print("calculate individual benefit b_i for user ", user_id)
        b_i = individual_benefit_function(ratings[ratings.user == user_id], rating_scale_max=rating_scale_max)
        # print("individual ratings: ", b_i)
        if b_i is not None:
            b_i_s.append(b_i)
            num_user_count += 1

    if alpha == 0:
        result = -np.mean(np.log(b_i_s / np.mean(b_i_s)) / np.mean(b_i_s))
    elif alpha == 1:
        result = np.mean(np.log((b_i_s / np.mean(b_i_s)) ** b_i_s) / np.mean(b_i_s))
    else:
        result = np.mean((b_i_s / np.mean(b_i_s))**alpha - 1) / (alpha * (alpha - 1))
    # print(result)
    return result

