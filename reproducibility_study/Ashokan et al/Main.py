from lenskit import batch
from lenskit import crossfold as xf
from lenskit.algorithms import als
from lenskit.algorithms import item_knn as knn
from lenskit.metrics.predict import rmse as rmse
from lenskit.metrics.predict import mae as mae

from collections import defaultdict
import pandas as pd
import numpy as np
import FairnessMetrics
import PostProcessing
import SyntheticDataCreation

from HelperFunctions import calculate_differences_in_rating, df_empty

import os


def main():
    # first, we define the algorithms that we want to look at.
    algo_ii = knn.ItemItem(20)
    algo_als = als.BiasedMF(50)

    algo_ii_text = 'ItemItem'
    algo_als_text = 'ALS'

    algos = {algo_ii_text: algo_ii, algo_als_text: algo_als}

    # to parallalize the code, we can specify how many cores we have available
    num_cores = 4

    all_recs = []
    all_predictions = []
    test_predictions = []
    test_data = []

    data = None
    partitions = []

    num_kfold = 10
    sample_fraction = float(1.0 / num_kfold)

    # if we want to use synthetic data, specify here:
    num_users = 400
    num_items = 300

    ## specify the dataset.

    # this can either be the empirical MovieLens Dataset, or the synthetic dataset
    movielens_data = 'MovieLens1M' # this is the 1M Movielens data

    # the most unbalanced synthetic data is the last option. See SyntheticDataCreation for more details
    # synthetic_data = 'Synthetic_UnifPop_UnifObs'
    # synthetic_data = 'Synthetic_UnifPop_BiasObs'
    # synthetic_data = 'Synthetic_ImbalancedPop_UnifObs'
    synthetic_data = 'Synthetic_ImbalancedPop_BiasObs'

    dataset_name = movielens_data

    if dataset_name in ['MovieLens1M']:
        combined_data_filtered = pd.read_csv((dataset_name + ".csv"))
        # set a minimum number of ratings per user, according to Yao and Huang 2017
        min_ratings_user = 50
        min_ratings_item = 100
        # get user list and filter
        combined_data_filtered = combined_data_filtered.groupby('user').filter(lambda x: len(x) > min_ratings_user)
        combined_data_filtered = combined_data_filtered.groupby('item').filter(lambda x: len(x) > min_ratings_item)

        data = combined_data_filtered
        partitions = xf.partition_rows(data, num_kfold)
        privileged_users = data.loc[data['gender'] == 'M'].user
        unprivileged_users = data.loc[data['gender'] == 'F'].user

    else:
        if dataset_name == "Synthetic_UnifPop_UnifObs":
            pop_dist_used = SyntheticDataCreation.pop_dist_cumulative_unif
            O_used = SyntheticDataCreation.df_O_unif
        elif dataset_name == "Synthetic_UnifPop_BiasObs":
            pop_dist_used = SyntheticDataCreation.pop_dist_cumulative_unif
            O_used = SyntheticDataCreation.df_O_bias
        elif dataset_name == "Synthetic_ImbalancedPop_UnifObs":
            pop_dist_used = SyntheticDataCreation.pop_dist_cumulative_imbalanced
            O_used = SyntheticDataCreation.df_O_unif
        else:
            pop_dist_used = SyntheticDataCreation.pop_dist_cumulative_imbalanced
            O_used = SyntheticDataCreation.df_O_bias
        for i in range(0, num_kfold):
            partitions.append(SyntheticDataCreation.create_synthetic_data(pop_dist_used, O_used,
                                                                          SyntheticDataCreation.df_L, num_users,
                                                                          num_items))

    metrics_fold = []
    i = 0

    metrics = df_empty(columns=['Algorithm', 'fold', 'metric', 'value'], dtypes=[np.str, np.int64, np.str, np.float])

    ################################################################################################################
    ################################################## ADDED CODE ##################################################
    sensitive_attribute = "user_age"
    _dataset = "filtered(20)_lastfm-1K"
    # _dataset = "movielens-1m"

    train_for_reproducibility = pd.read_csv(f'{_dataset}_train_{sensitive_attribute}.csv')
    test_for_reproducibility = pd.read_csv(f'{_dataset}_test_{sensitive_attribute}.csv')
    data_for_reproducibility = pd.concat([train_for_reproducibility, test_for_reproducibility])

    partitions = [(train_for_reproducibility, test_for_reproducibility)]

    privileged_users = data_for_reproducibility[data_for_reproducibility['gender'] == 'M'].user
    unprivileged_users = data_for_reproducibility[data_for_reproducibility['gender'] == 'F'].user
    ################################################################################################################
    ################################################################################################################

    # iterate through the different partitions
    for train, test in partitions:
        # let's calculate the difference between privileged and unprivileged groups (fairness metrics) in training and
        # test set to see if there are substantial differences (which make it harder for the post-processing to work)

        # if we work with synthetic data, get the corresponding user list with privileged and unprivileged users
        # note that we also use M and F as identifiers. see SyntheticDataCreation for more details
        if dataset_name in ['Synthetic_UnifPop_UnifObs', 'Synthetic_UnifPop_BiasObs', 'Synthetic_ImbalancedPop_UnifObs',
                            'Synthetic_ImbalancedPop_BiasObs']:
            print('setting correct privileged users')
            privileged_users = train.loc[train['gender'] == 'M'].user.combine_first(
                test.loc[test['gender'] == 'M'].user)
            unprivileged_users = train.loc[train['gender'] == 'F'].user.combine_first(
                test.loc[test['gender'] == 'F'].user)

        print('num observations in train: ', len(train))
        print('num observations in test: ', len(test))
        print('unique users in train set: ', len(train.user.unique()), ' and test set: ', len(test.user.unique()))
        print('unique items in train set: ', len(train.item.unique()), ' and test set: ', len(test.item.unique()))
        calculate_differences_in_rating(train, test, privileged_users, unprivileged_users)

        test_data.append(test)
        test_predictions.append(test)

        print('next iteration, start algo eval')

        # fit and calculate the predictions
        # the rating predictions are fitted only on the training set, and then applied on the test set
        algo_pred = defaultdict()
        algo_train = defaultdict()
        for algo_text, algo in algos.items():
            algo.fit(train)
            algo_pred[algo_text] = batch.predict(algo, test, n_jobs=num_cores)
            algo_train[algo_text] = batch.predict(algo, train, n_jobs=num_cores)

            print("RMSE for train set: ", rmse(algo_train[algo_text]['prediction'],
                                  algo_train[algo_text]['rating']))
            print("RMSE for test set: ", rmse(algo_pred[algo_text]['prediction'],
                                  algo_pred[algo_text]['rating']))

        print('batch prediction done')

        # run the posterior adjustment post-processing
        # this changes both the predicted ratings and the scores

        # algo_post collects all post-processing results for all algorithms
        # top level: keys are algorithms, values are dictionaries themselves
        # value level: values are dictionaries where the key is the post-processing type and the value is the set of adjusted prediction
        algo_post = defaultdict()
        for algo_text, algo in algos.items():
            algo_post_local = defaultdict()
            # first, calculate the post-processing adjustments
            algo_train_local = algo_train.get(algo_text)
            algo_pred_local = algo_pred.get(algo_text)

            # start with no adjustments
            algo_post_local[(algo_text + '_orig')] = algo_pred.get(algo_text)
            # then, calculate the parity and value post processing adjustments
            algo_post_local[(algo_text + '_parity_adj')] = \
                PostProcessing.calculate_posterior_parity_adjustments(algo_train_local, algo_pred_local,
                                                                      privileged_users, unprivileged_users)
            algo_post_local[(algo_text + '_val_adj')] = \
                PostProcessing.calculate_posterior_adjustments(algo_train_local, algo_pred_local,
                                                               privileged_users, unprivileged_users)

            # add them to the algo_post dictionary
            algo_post[algo_text] = algo_post_local

        for algo_text, algo in algos.items():
            for algo_post_text, algo_post_pred in algo_post.get(algo_text).items():
                ########################################################################################
                ###################################### ADDED CODE ######################################
                if not os.path.exists(os.path.join('results', _dataset)):
                    os.mkdir(os.path.join('results', _dataset))

                algo_post_pred.to_csv(
                    os.path.join('results', _dataset, f'{algo_post_text}_{sensitive_attribute}.csv'),
                    index=None
                )
                continue
                ########################################################################################
                ########################################################################################

                rmse_local = rmse(algo_post_pred['prediction'],
                                  algo_post_pred['rating'])
                mae_local = mae(algo_post_pred['prediction'],
                                algo_post_pred['rating'])

                print("rmse for algo ", algo_post_text, ": ", rmse_local)
                print("mae for algo ", algo_post_text, ": ", mae_local)
                metrics_fold.append([algo_post_text, i, rmse_local, mae_local])
                algo_post_pred['Algorithm'] = algo_post_text
                all_predictions.append(algo_post_pred)

                # add the performance metrics
                metrics.loc[len(metrics)] = [algo_post_text, i, 'rmse', rmse_local]
                metrics.loc[len(metrics)] = [algo_post_text, i, 'mae', mae_local]

                # calculate the fairness metrics and add them to the metrics dataframe
                u_val, u_abs, u_under, u_over, u_par, u_disparate = FairnessMetrics.get_recommender_fairness_metrics_from_predictions(
                    algo_post_pred,
                    privileged_users, unprivileged_users)
                print("u_val for algorithm ", algo_post_text, ": ", u_val)

                rating_scale_max = 5
                if dataset_name in ['Synthetic_UnifPop_UnifObs', 'Synthetic_UnifPop_BiasObs', 'Synthetic_ImbalancedPop_UnifObs','Synthetic_ImbalancedPop_BiasObs']:
                    rating_scale_max = 2
                gei_0 = FairnessMetrics.generalized_entropy_index(algo_post_pred, alpha=0, rating_scale_max=rating_scale_max)
                gei_1 = FairnessMetrics.generalized_entropy_index(algo_post_pred, alpha=1, rating_scale_max=rating_scale_max)
                gei_2 = FairnessMetrics.generalized_entropy_index(algo_post_pred, alpha=2, rating_scale_max=rating_scale_max)

                # finally, add all metrics
                metrics.loc[len(metrics)] = [algo_post_text, i, 'u_val', u_val]
                metrics.loc[len(metrics)] = [algo_post_text, i, 'u_abs', u_abs]
                metrics.loc[len(metrics)] = [algo_post_text, i, 'u_under', u_under]
                metrics.loc[len(metrics)] = [algo_post_text, i, 'u_over', u_over]
                metrics.loc[len(metrics)] = [algo_post_text, i, 'u_par', u_par]
                metrics.loc[len(metrics)] = [algo_post_text, i, 'u_disparate', u_disparate]
                metrics.loc[len(metrics)] = [algo_post_text, i, 'u_gei_0', gei_0]
                metrics.loc[len(metrics)] = [algo_post_text, i, 'u_gei_1', gei_1]
                metrics.loc[len(metrics)] = [algo_post_text, i, 'u_gei_2', gei_2]

        i += 1
        #print(metrics)

    print('saving overall metrics')
    save_name = str('Metrics_' + dataset_name + ".csv")
    metrics.to_csv(save_name)


if __name__ == '__main__':
    print('Starting the main function')
    main()

