from collections import defaultdict
import pandas as pd
import numpy as np


def calculate_differences_in_rating(train, test, privileged_group, unprivileged_group):

    unique_items_train = train['item'].unique()
    unique_items_test = test['item'].unique()
    all_items = list(set().union(unique_items_train, unique_items_test))

    item_based_val = defaultdict(list)

    u_val = 0
    u_abs = 0
    u_under = 0
    u_over = 0
    u_par = 0
    u_disparate = 0
    #count = 0
    for item in all_items:
        # print("next item: ", item)
        # get the average rating for the privileged group
        ratings_item_subset_train = train[train.item == item]
        ratings_item_subset_test = test[test.item == item]

        if ratings_item_subset_train.empty:
            continue
        if ratings_item_subset_test.empty:
            continue

        ratings_privileged_subset_train = ratings_item_subset_train[ratings_item_subset_train.user.isin(privileged_group)]
        ratings_privileged_subset_test = ratings_item_subset_test[ratings_item_subset_test.user.isin(privileged_group)]
        ratings_unprivileged_subset_train = ratings_item_subset_train[ratings_item_subset_train.user.isin(unprivileged_group)]
        ratings_unprivileged_subset_test = ratings_item_subset_test[ratings_item_subset_test.user.isin(unprivileged_group)]

        if ratings_privileged_subset_train.empty:
            continue
        if ratings_unprivileged_subset_test.empty:
            continue
        if ratings_privileged_subset_train.empty:
            continue
        if ratings_unprivileged_subset_test.empty:
            continue

        E_g_r_train = np.mean(ratings_privileged_subset_train.rating)
        E_notg_r_train = np.mean(ratings_unprivileged_subset_train.rating)

        E_g_r_test = np.mean(ratings_privileged_subset_test.rating)
        E_notg_r_test = np.mean(ratings_unprivileged_subset_test.rating)

        item_based_val[item].append([E_g_r_train - E_g_r_test, E_notg_r_train - E_notg_r_test])

    # finally, get the non-parity metric
    avg_rating_diff_train = np.mean(train[train.user.isin(privileged_group)].rating) - np.mean(train[train.user.isin(unprivileged_group)].rating)
    avg_rating_diff_test = np.mean(test[test.user.isin(privileged_group)].rating) - np.mean(test[test.user.isin(unprivileged_group)].rating)

    print("u_val: ", u_val)
    print("u_abs: ", u_abs)
    print("u_under: ", u_under)
    print("u_over: ", u_over)
    print("u_par: ", u_par)
    print("u_disparate: ", u_disparate)


# code from stackoverflow to create empty dataframe
def df_empty(columns, dtypes, index=None):
    assert len(columns) == len(dtypes)
    df = pd.DataFrame(index=index)
    for c, d in zip(columns, dtypes):
        df[c] = pd.Series(dtype=d)
    return df

