import os
from typing import Set, Text, DefaultDict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import data.utils as data_utils
from helpers.recsys_arg_parser import RecSysArgumentParser
from helpers.logger import RcLogger
from data.datasets.lastfm import lastfm_utils


"""
 # per_user_random train-validation-test split 70%-10%-20%, age binarised with age <= 24 
 python -m reproducibility_study.Preprocessing.preprocess_lastfm1K -dataset lastfm --dataset_size 1K 
 --subdatasets plays users --dataset_split train train --users_field user_id --items_field artist_id 
 --dataset_columns user_id artist_id plays - user_id user_gender user_age --rating_field user_rating 
 --sensitive_field user_gender --train_val_test_split_type per_user_random --train_val_test_split 70% 10% 20%
 --min_interactions 20 --attribute_to_binary user_age --binary_le_delimiter 24 -model Pointwise --n_reps 2
 --overwrite_preprocessed_dataset
"""


if __name__ == "__main__":
    args = RecSysArgumentParser().routine_arg_parser()
    RcLogger.start_logger(level="INFO")

    metadata = vars(args).copy()

    plays, users = data_utils.load_dataset(args.dataset,
                                           split=args.dataset_split,
                                           size=args.dataset_size,
                                           sub_datasets=args.subdatasets,
                                           columns=args.dataset_columns)

    plays: pd.DataFrame = tfds.as_dataframe(plays)
    users: pd.DataFrame = tfds.as_dataframe(users)

    users = users[(users["user_age"] > 0) & (users["user_gender"] != b'NA')]
    users["user_gender"] = users["user_gender"].map({b'M': True, b'F': False})

    plays = plays[plays[args.users_field].isin(users[args.users_field])]

    data = plays.set_index(args.users_field).join(users.set_index(args.users_field)).reset_index()

    if args.min_interactions > 0 or args.balance_ratio is not None or args.sample_n is not None:
        data, dataset_info = data_utils.filter_interactions(data,
                                                            args.users_field,
                                                            min_interactions=args.min_interactions,
                                                            balance_attribute=args.balance_attribute,
                                                            balance_ratio=args.balance_ratio,
                                                            sample_n=args.sample_n,
                                                            sample_attribute=args.sample_attribute)

        if dataset_info not in metadata['dataset']:
            metadata['dataset'] = dataset_info + metadata['dataset']

    if args.attribute_to_binary is not None and args.binary_le_delimiter is not None:
        data = data_utils.map_as_binary(data,
                                        args.attribute_to_binary,
                                        delimiter_value_le=args.binary_le_delimiter)

    data = tfds.as_dataframe(data)

    data = lastfm_utils.normalize_ratings(data)

    data = data.sample(frac=1)  # shuffle the dataset

    data_records = data.to_dict(orient='list')

    data = tf.data.Dataset.from_tensor_slices(data_records)

    train, val, test = data_utils.get_train_test(data,
                                                 split=args.train_val_test_split,
                                                 split_type=args.train_val_test_split_type,
                                                 seed=args.seed_shuffle,
                                                 shuffle_kwargs={
                                                     "reshuffle_each_iteration": args.reshuffle_each_iteration
                                                 },
                                                 splitter_kwargs={
                                                     "user_field": args.users_field,
                                                     "item_field": args.items_field
                                                 },
                                                 n_folds=args.n_folds)

    data_utils.save_train_test(metadata,
                               train,
                               test,
                               validation=val,
                               overwrite=args.overwrite_preprocessed_dataset)

    observed_items, unobserved_items, _ = data_utils.get_train_test_features(args.users_field,
                                                                             args.items_field,
                                                                             train_data=train,
                                                                             test_or_val_data=test)
    observed_items: DefaultDict[Text, Set]
    unobserved_items: DefaultDict[Text, Set]

    unique_items = np.unique(
        list(train.concatenate(val).concatenate(test).map(lambda x: x[args.items_field]).as_numpy_iterator())
    )

    data_utils.generate_binary_data(train,
                                    observed_items,
                                    unique_items,
                                    metadata,
                                    n_repetitions=args.n_reps,
                                    user_id_field=args.users_field,
                                    item_id_field=args.items_field,
                                    split="train")
