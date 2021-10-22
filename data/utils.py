import os
import shutil
import pickle
import json
import random
import textwrap
from typing import (
    Sequence,
    List,
    Union,
    Dict,
    Text,
    Any,
    Literal,
    Callable,
    Set,
    DefaultDict
)
from collections import namedtuple, UserDict, defaultdict

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import tqdm
import scipy.sparse

import helpers.constants as constants
import helpers.general_utils as general_utils
from helpers.logger import RcLogger, RcLoggerException
from helpers.recsys_arg_parser import RecSysArgumentParser


def load_dataset(name: str,
                 split: Sequence[str],
                 size: str = "",
                 sub_datasets: Sequence[str] = None,
                 columns: Sequence[Sequence[str]] = None,
                 **kwargs) -> Union[tf.data.Dataset, List[tf.data.Dataset]]:

    columns = [] if columns is None else columns
    size_msg = f" of size '{size}'" if size != '' else ''
    RcLogger.get().info(f"Loading '{split}' split{'s' if len(split) > 1 else ''} of "
                        f"dataset '{name}'{size_msg}")

    dataset_name = name
    pre_process_data = []

    if size != "":
        size = f"{size}-"

    if not (len(sub_datasets) == len(split) == len(columns)) and columns:
        msg = f"sub_datasets, split and columns should have same length, got lengths " \
              f"{len(sub_datasets)}, {len(split)} and {len(columns)}"
        raise RcLoggerException(AssertionError, msg)

    for ds, sp in zip(sub_datasets, split):
        pre_process_data.append(tfds.load(f"{dataset_name}/{size}{ds}", split=sp, **kwargs))

    RcLogger.get().info(f"Loaded subdatasets '{sub_datasets}' of dataset '{name}'")

    if columns:
        _pre_data = []
        for ppd, cols in zip(pre_process_data, columns):
            _pre_data.append(__map_dataset(ppd, cols))

        pre_process_data = _pre_data

        RcLogger.get().debug(f"sub_datasets mapped with mappings '{columns}'")

    if "operation" in kwargs:
        op = kwargs.pop("operation")

        if op[0] == "str_keys_to_int":
            pre_process_data = str_keys_to_int(*pre_process_data, except_items=op[1])
        else:
            raise RcLoggerException(NotImplementedError, '')

    return pre_process_data if len(pre_process_data) > 1 else pre_process_data[0]


def __map_dataset(dataset, columns):
    return dataset.map(lambda x: {col: x[col] for col in columns})


def str_keys_to_int(*args, dtype=tf.int32, except_items=None):
    except_items = [] if except_items is None else except_items
    str_to_int_datasets = []

    for dataset in args:
        entry = next(dataset.take(1).as_numpy_iterator())
        for key, value in entry.items():
            value_bytes_str = False

            # bytestring test
            if isinstance(value, bytes) or isinstance(value, bytearray):
                try:
                    value.decode('utf-8')
                    value_bytes_str = True
                except UnicodeError:
                    RcLogger.get().debug("Found value as bytes which cannot be decoded into utf-8")
                    pass

            if (isinstance(value, str) or value_bytes_str) and key not in except_items:
                dataset = dataset.map(lambda x: {**x, key: tf.strings.to_number(x[key],  dtype)})

        str_to_int_datasets.append(dataset)

    return str_to_int_datasets


def filter_interactions(data: Union[pd.DataFrame, tf.data.Dataset],
                        user_col: str,
                        min_interactions=10,
                        balance_attribute=None,
                        balance_ratio: Dict[Any, Union[int, float]] = None,
                        delimiter_value_le=25,
                        sample_n=None,
                        sample_attribute=None) -> tf.data.Dataset:
    if not isinstance(data, pd.DataFrame):
        data_df: pd.DataFrame = tfds.as_dataframe(data)
    else:
        data_df = data
    dataset_info = ''

    if min_interactions is not None and min_interactions > 0:
        RcLogger.get().info(f"Filtering users with at least {min_interactions} interactions")
        data_df = data_df.groupby(user_col).filter(lambda x: len(x) >= min_interactions)

        dataset_info += constants.PREFIX_FILTERED_DATASET.format(min_interactions=min_interactions)

    if balance_attribute is not None and balance_ratio is not None:
        RcLogger.get().info(f"Balancing users by {balance_attribute} with ratio {balance_ratio}")

        if len(data_df[balance_attribute].unique()) > 2:
            data_df[balance_attribute] = data_df[balance_attribute].map(
                lambda val: True if val <= delimiter_value_le else False
            )

        users__bal_attr = data_df.groupby(user_col).aggregate({balance_attribute: 'first'}).reset_index()
        gr_users_df = users__bal_attr.groupby(balance_attribute)

        groups = {gr: len(group) for gr, group in gr_users_df}

        ratio_value = list(balance_ratio.values())[0]

        if not isinstance(ratio_value, int) and not isinstance(ratio_value, float):
            msg = f"balance_ratio support only int or float values, gor {type(list(balance_ratio.values())[0])}"
            raise RcLoggerException(ValueError, msg)
        else:
            # convert float balance in fixed int values
            if isinstance(list(balance_ratio.values())[0], float):
                balance_ratio = general_utils.balance_groups_by_ratio(groups, balance_ratio)

            for gr, n_int_bal in balance_ratio.items():
                if n_int_bal > groups[gr]:
                    msg = f"Group {gr} contains {groups[gr]} interactions. Impossible to sample {n_int_bal} " \
                          f"interactions."
                    raise RcLoggerException(ValueError, msg)

            new_gr_users_df = []
            for name, gr in gr_users_df:
                try:
                    sampled_users = gr.sample(n=balance_ratio[name])[user_col].to_list()
                except ValueError as e:
                    msg = f" || group {name} does not contain enough samples"
                    raise RcLoggerException(ValueError, str(e) + msg)

                new_gr_users_df.extend(sampled_users)

            data_df = data_df[data_df[user_col].isin(new_gr_users_df)]

        dataset_info += constants.PREFIX_BALANCED_DATASET.format(
            balance_attribute=balance_attribute,
            balance_ratio=str(list(balance_ratio.items()))
        )

    # Sampling should be done always at the end
    if sample_n is not None:
        if sample_attribute is not None:
            RcLogger.get().info(f"Sampling {sample_n} unique values of `{sample_attribute}`")
            sample_attr_values = np.random.choice(data_df[sample_attribute].unique(), sample_n, replace=False)
            data_df = data_df[data_df[sample_attribute].isin(sample_attr_values)]
        else:
            RcLogger.get().info(f"Sampling {sample_n} dataset rows")
            data_df = data_df.sample(n=sample_n)

        dataset_info += constants.PREFIX_SAMPLED_DATASET.format(
            sample_n=sample_n,
            sample_attribute=sample_attribute
        )

    data_records = data_df.to_dict(orient='list')

    data = tf.data.Dataset.from_tensor_slices(data_records)

    return data, dataset_info


def map_as_binary(data: tf.data.Dataset,
                  attribute_field,
                  delimiter_value_le=25) -> tf.data.Dataset:
    """

    :param data: dataset to apply the map
    :param attribute_field: field of the FeatureDict that the map must be applied to
    :param delimiter_value_le: a value less or equal to this parameter will be mapped as True, otherwise False
    :return: the same dataset with the data of `attribute_field` mapped according to `delimiter_value_le`
    """

    # cast delimiter to int or float according to the type of value of attribute_field
    type_attribute_field = type(next(data.take(1).as_numpy_iterator())[attribute_field])
    delimiter_value_le = type_attribute_field(delimiter_value_le)

    if delimiter_value_le is not None:
        _data = data.map(
            lambda x: {
                **x,
                attribute_field: tf.cond(
                    tf.less_equal(x[attribute_field], delimiter_value_le),
                    lambda: True,
                    lambda: False
                )
            }
        )
    else:
        msg = f"all parameters are None. No action to perform"
        raise RcLoggerException(ValueError, msg)

    return _data


def get_train_test(data: tf.data.Dataset,
                   split: Sequence[Union[str, int, float]],
                   split_type: Union[
                       Literal["random"],
                       Literal["per_user_timestamp"],
                       Literal["per_user_random"],
                       Literal["k_fold"],
                   ] = "per_user_timestamp",
                   seed: Union[bool, int] = False,
                   shuffle_kwargs: Dict[Text, Any] = None,
                   splitter_kwargs: Dict[Text, Any] = None,
                   n_folds: int = 5):
    # TODO: add handling of a split based on a timestamp (chosen by user or "perfect" timestamp split)
    seed_msg = f"Data will be shuffled with seed '{seed if isinstance(seed, int) else 'random'}'" \
        if seed is not False else ""
    RcLogger.get().info(f"Creating train and test from data with split '{split}'. "
                        f"{seed_msg}")

    if split_type == "random":
        # dataset cardinality is used if available, otherwise it is converted to list and len is taken
        data_cardinality = _get_data_cardinality(data)

        data, split = __random_initialization_splitter(data, split, data_cardinality, seed=seed)

        train, val, test = __random_splitter(data, split)
    elif split_type == "k_fold":
        # dataset cardinality is used if available, otherwise it is converted to list and len is taken
        data_cardinality = _get_data_cardinality(data)

        split = _convert_split_for_random(split, data_cardinality)

        data = _shuffle_data(data, seed)

        train, val, test = __k_fold_splitter(data, split, data_cardinality, n_folds=n_folds)
    elif split_type == "per_user_timestamp" or split_type == "per_user_random":
        if isinstance(split[0], str):
            if False in ['%' in spl for spl in split]:
                msg = "split values as string need to contain '%'"
                raise RcLoggerException(ValueError, msg)

            split = [int(spl.replace('%', '')) / 100 for spl in split]
        elif isinstance(split[0], int):
            msg = "split values cannot be integers with `per_user_timestamp` split type"
            raise RcLoggerException(ValueError, msg)

        if len(split) == 3:
            split[1], split[2] = split[2], split[1]

        if all(x is None for x in split):
            msg = "All values in split are None"
            raise RcLoggerException(ValueError, msg)

        _splits = dict(zip(['train_split', 'test_split', 'validation_split'], split))

        if split_type == "per_user_random":
            splitter_kwargs['time_field'] = False

        train, val, test = __per_user_timestamp_random_splitter(data, **_splits, seed=seed, **splitter_kwargs)
    else:
        msg = f"split can be a str or a sequence of str or a sequence of int, got instead {split}"
        raise RcLoggerException(ValueError, msg)

    RcLogger.get().info("Train and test generated")

    return train, val, test


def _get_data_cardinality(data):
    data_cardinality = data.cardinality()

    if data_cardinality == tf.data.INFINITE_CARDINALITY or data_cardinality == tf.data.UNKNOWN_CARDINALITY:
        RcLogger.get().warn("Dataset cardinality is infinite or unknown, dataset will be converted to list and "
                            "len will be taken. Consider changing type of data "
                            "for the tf.data.Dataset to improve performance")
        data_cardinality = len(list(data))
    else:
        data_cardinality = data_cardinality.numpy()

    return data_cardinality


def _convert_split_for_random(split, data_cardinality):
    if isinstance(split[0], str):
        if False in ['%' in spl for spl in split]:
            msg = "split values as string need to contain '%'"
            raise RcLoggerException(ValueError, msg)

        split = [round(int(spl.replace('%', '')) / 100 * data_cardinality) for spl in split]
    elif isinstance(split[0], float):
        if sum(split) > 1.0:
            msg = "Sum of each split with float values cannot exceed 1.0"
            raise RcLoggerException(ValueError, msg)

        split = [round(spl * data_cardinality) for spl in split]

    return split


def _check_split_sum(split, data_cardinality):
    diff = data_cardinality - sum(split)

    if diff != 0:
        split[-1] += diff

    return split


def _shuffle_data(data: tf.data.Dataset,
                  seed: Union[int, bool]):
    if seed:  # if seed == False shuffling is not performed
        if isinstance(seed, bool):
            data: pd.DataFrame = tfds.as_dataframe(data)
            data = data.sample(frac=1)

            data = tf.data.Dataset.from_tensor_slices(data.to_dict(orient='list'))

            RcLogger.get().debug("Data has been shuffled")
        else:
            data: pd.DataFrame = tfds.as_dataframe(data)
            data = data.sample(frac=1, random_state=seed)

            data = tf.data.Dataset.from_tensor_slices(data.to_dict(orient='list'))

            RcLogger.get().debug("Data has been shuffled")

    return data


def __random_initialization_splitter(data: tf.data.Dataset,
                                     split: Sequence,
                                     data_cardinality: int,
                                     seed: Union[bool, int] = False):
    split = _convert_split_for_random(split, data_cardinality)

    split = _check_split_sum(split, data_cardinality)

    # Shuffle of TensorFlow datasets require too much memory
    # if isinstance(seed, int):
    #     tf.random.set_seed(seed)
    #     data = data.shuffle(sum(split), seed=seed, **shuffle_kwargs)
    # elif seed:
    #     data = data.shuffle(sum(split), **shuffle_kwargs)
    data = _shuffle_data(data, seed)

    return data, split


def __random_splitter(data: tf.data.Dataset,
                      split: Sequence[int]):
    if len(split) == 2:
        train = data.take(split[0])
        val = None
        test = data.skip(split[0]).take(split[1])
    elif len(split) == 3:
        train = data.take(split[0])
        val = data.skip(split[0]).take(split[1])
        test = data.skip(split[0]).skip(split[1]).take(split[2])
    else:
        msg = f"Split can contain 2 or 3 values, got {len(split)} instead"
        raise RcLoggerException(NotImplementedError, msg)

    return train, val, test


# noinspection PyUnresolvedReferences
def __k_fold_splitter(data: tf.data.Dataset,
                      split: Sequence[int],
                      data_cardinality: int,
                      n_folds: int = 5):
    """
    The folds are created maintaining the size of the test set equal for all the folds

    :param data:
    :param split:
    :param data_cardinality:
    :param n_folds:
    :return:
    """
    train, val, test = [None] * n_folds, [None] * n_folds, [None] * n_folds
    val_split = split[1] if len(split) == 3 else None

    test_size = data_cardinality // n_folds

    for k in range(n_folds):
        train[k] = data.take(k * test_size)
        test[k] = data.skip(k * test_size).take(test_size)

        # If take argument is -1 or greater than the size of this dataset,
        # the new dataset will contain all elements of this dataset
        train[k] = train[k].concatenate(data.skip(k * test_size + test_size).take(-1))

        if val_split is not None:
            val[k] = train[k].take(val_split)
            train[k] = train[k].skip(val_split).take(-1)

    return train, val if val else None, test


def __per_user_timestamp_random_splitter(interactions,
                                         train_split=0.80,
                                         test_split=None,
                                         validation_split=None,
                                         user_field='user_id',
                                         item_field='movie_id',
                                         time_field='timestamp',
                                         seed: Union[bool, int] = False):
    """

    :param interactions:
    :param train_split:
    :param test_split:
    :param validation_split:
    :param user_field:
    :param item_field:
    :param time_field: if False per user random splitting is performed
    :param seed:
    :return:
    """
    RcLogger.get().info(f"Splitting dataset per_user {'by timestamp' if time_field else 'randomly'} with split "
                        f"[{train_split},"
                        f"{f' {validation_split},' if validation_split is not None else ''}"
                        f" {test_split}]:")

    interactions: pd.DataFrame = tfds.as_dataframe(interactions)

    if not all(type(x) == type(train_split) or x is None or train_split is None for x in [train_split, test_split, validation_split]):
        msg = f"Train, test and validation split must be all of the same type"
        raise RcLoggerException(ValueError, msg)

    train_set = []
    test_set = []
    val_set = []
    groups = interactions.groupby([user_field])
    for i, (_, group) in enumerate(tqdm.tqdm(groups, desc=f"Splitting data per_user")):
        if (i % 1000) == 0:
            RcLogger.get().debug(f'Parsing user {i} of {len(groups)}')

        if time_field:
            sorted_group = group.sort_values(time_field)
        else:
            sorted_group = group

        if isinstance(train_split, float) or isinstance(test_split, float):
            n_rating_train = int(len(sorted_group.index) * train_split) if train_split is not None else 0
            n_rating_test = int(len(sorted_group.index) * test_split) if test_split is not None else 0
            n_rating_val = int(len(sorted_group.index) * validation_split) if validation_split is not None else 0

            if len(sorted_group.index) > (n_rating_train + n_rating_test + n_rating_val):
                n_rating_train += len(sorted_group.index) - (n_rating_train + n_rating_test + n_rating_val)
        else:
            msg = f"split type not accepted"
            raise RcLoggerException(ValueError, msg)

        if n_rating_train == 0:
            start_index = len(sorted_group) - n_rating_test
            start_index = start_index - n_rating_val if n_rating_val is not None else start_index
            train_set.append(sorted_group.iloc[:start_index])
        else:
            train_set.append(sorted_group.iloc[:n_rating_train])
            start_index = n_rating_train

        if n_rating_val > 0:
            val_set.append(sorted_group.iloc[start_index:(start_index + n_rating_val)])
            start_index += n_rating_val

        if n_rating_test > 0:
            test_set.append(sorted_group.iloc[start_index:(start_index + n_rating_test)])
        else:
            test_set.append(sorted_group.iloc[start_index:])

    train, test = pd.concat(train_set), pd.concat(test_set)
    validation = pd.concat(val_set) if val_set else None

    if isinstance(seed, bool):
        if seed:
            train = train.sample(frac=1)
            test = test.sample(frac=1)

            if validation is not None:
                validation = validation.sample(frac=1)
    elif isinstance(seed, int):
        train = train.sample(frac=1, random_state=seed)
        test = test.sample(frac=1, random_state=seed)

        if validation is not None:
            validation = validation.sample(frac=1, random_state=seed)

    RcLogger.get().debug(
        f"Mean number of train ratings per learner: {np.mean(train.groupby([user_field]).count()[item_field].values)}"
    )
    RcLogger.get().debug(
        f"Mean number of test ratings per learner: {np.mean(test.groupby([user_field]).count()[item_field].values)}"
    )

    train_records = train.to_dict(orient='list')
    test_records = test.to_dict(orient='list')

    train = tf.data.Dataset.from_tensor_slices(train_records)
    test = tf.data.Dataset.from_tensor_slices(test_records)

    val = None
    if validation is not None:
        val_records = validation.to_dict(orient='list')
        val = tf.data.Dataset.from_tensor_slices(val_records)

    return train, val, test


def get_train_test_features(user_field,
                            item_field,
                            train_data: tf.data.Dataset = None,
                            test_or_val_data: tf.data.Dataset = None,
                            item_popularity: bool = None,
                            sensitive_field=None,
                            rating_field=None,
                            other_returns: Sequence[str] = None):
    observed_items = defaultdict(set)
    unobserved_items = defaultdict(set)

    RcLogger.get().info(f"Extracting features from train and/or test/validation data")

    len_train_data = 0
    other_returns = [] if other_returns is None else other_returns
    train_rating_df = None
    test_rating_df = None

    if "train_rating_dataframe" in other_returns or "test_rating_dataframe" in other_returns:
        data = train_data.concatenate(test_or_val_data)
        users = data.map(lambda x: x[user_field])
        items = data.map(lambda x: x[item_field])
        users = list(np.unique(list(users.as_numpy_iterator())))
        items = list(np.unique(list(items.as_numpy_iterator())))

        # Non lexicographically ordered unique users and items
        if (isinstance(users[0], str) or isinstance(users[0], bytes)) and users[0].isdigit():
            unique_users = np.array(sorted(users, key=int))
        else:
            unique_users = np.array(sorted(users))
        if (isinstance(items[0], str) or isinstance(items[0], bytes)) and items[0].isdigit():
            unique_items = np.array(sorted(items, key=int))
        else:
            unique_items = np.array(sorted(items))

        if "train_rating_dataframe" in other_returns:
            train_rating_df = pd.DataFrame(np.nan, index=unique_users, columns=unique_items)
        if "test_rating_dataframe" in other_returns:
            test_rating_df = pd.DataFrame(np.nan, index=unique_users, columns=unique_items)

    sensitive_dict = {} if "sensitive" in other_returns else None

    if item_popularity:
        if test_or_val_data is not None:
            pop_data = train_data.concatenate(test_or_val_data)
        else:
            pop_data = train_data
        unique_items = np.unique(list(pop_data.map(lambda x: x[item_field]).as_numpy_iterator()))
        item_popularity = dict.fromkeys(unique_items, 0)
    else:
        item_popularity = None

    other_returns = {} if other_returns is None else other_returns

    if train_data is not None:
        for train_features in tqdm.tqdm(train_data, desc="Extracting features from train data"):
            if train_features is not None:
                len_train_data += 1
                user = train_features[user_field]
                user = user.numpy() if isinstance(user, tf.Tensor) else user
                item = train_features[item_field]
                item = item.numpy() if isinstance(item, tf.Tensor) else item
                if sensitive_dict is not None:
                    sens = train_features[sensitive_field]
                    sensitive_dict[user] = sens.numpy() if isinstance(sens, tf.Tensor) else sens

                observed_items[user].add(item)

                if item_popularity is not None:
                    item_popularity[item] += 1

                if train_rating_df is not None:
                    rating = train_features[rating_field]
                    rating = rating.numpy() if isinstance(rating, tf.Tensor) else rating

                    train_rating_df.loc[user, item] = rating

    if test_or_val_data is not None:
        for test_features in tqdm.tqdm(test_or_val_data, desc="Extracting features from test data"):
            if test_features is not None:
                user = test_features[user_field]
                user = user.numpy() if isinstance(user, tf.Tensor) else user
                item = test_features[item_field]
                item = item.numpy() if isinstance(item, tf.Tensor) else item

                unobserved_items[user].add(item)

                if test_rating_df is not None:
                    rating = test_features[rating_field]
                    rating = rating.numpy() if isinstance(rating, tf.Tensor) else rating

                    test_rating_df.loc[user, item] = rating

    item_popularity = None if item_popularity is None else \
        None if not any(item_popularity.values()) else item_popularity

    ret = [
        observed_items,
        unobserved_items,
        item_popularity
    ]

    _dict_returns = {}
    if "len_train_data" in other_returns:
        _dict_returns["len_train_data"] = len_train_data
    if "train_rating_dataframe" in other_returns:
        _dict_returns["train_rating_dataframe"] = train_rating_df
    if "test_rating_dataframe" in other_returns:
        _dict_returns["test_rating_dataframe"] = test_rating_df
    if "sensitive" in other_returns:
        _dict_returns["sensitive"] = sensitive_dict

    if _dict_returns:
        ret.append(_dict_returns)

    return ret


def generate_triplets_data(train_data: tf.data.Dataset,
                           observed_items,
                           unique_items,
                           dataset_metadata,
                           n_repetitions=10,
                           user_id_field='user_id',
                           item_id_field='item_id',
                           len_train_data=None,
                           save_batch=8192,
                           overwrite=False,
                           split="train"):
    first_item = next(train_data.as_numpy_iterator())

    np_item_dtype = object
    if hasattr(first_item[item_id_field], "dtype"):
        np_item_dtype = first_item[item_id_field].dtype

    l_train = len_train_data

    x = train_data.repeat(n_repetitions)
    negative_items = np.empty((l_train * n_repetitions), dtype=np_item_dtype)
    users = x.map(lambda el: el[user_id_field])

    for i, _user in tqdm.tqdm(users.enumerate(), desc="Main operation to generate triplets data"):
        user_id = _user.numpy()

        negative_items[i] = random.choice(list(set(unique_items) - observed_items[user_id]))

    tf_item_dtype = tf.string if np_item_dtype is object else tf.dtypes.as_dtype(np_item_dtype)

    negative_items = tf.data.Dataset.from_tensor_slices(
        {"negative_item": tf.constant(negative_items, dtype=tf_item_dtype)}
    )

    x = tf.data.Dataset.zip((x, negative_items))

    RcLogger.get().debug("Saving triplets")
    save_tf_features_dataset(
        x.batch(save_batch).map(lambda a, b: {**a, **b}, num_parallel_calls=tf.data.AUTOTUNE),
        dataset_metadata,
        dataset_info="triplets",
        overwrite=overwrite,
        split=split
    )


def generate_binary_data(train_data: tf.data.Dataset,
                         observed_items,
                         unique_items,
                         dataset_metadata,
                         n_repetitions=10,
                         user_id_field='user_id',
                         item_id_field='item_id',
                         len_train_data=None,
                         save_batch=8192,
                         overwrite=False,
                         split="train"):
    users_items = train_data.map(lambda el: {
        user_id_field: el[user_id_field],
        item_id_field: el[item_id_field]
    })
    l_train = len_train_data if len_train_data is not None else len(
        list(train_data.map(lambda el: el[user_id_field]).as_numpy_iterator())
    )

    items = list(users_items.map(lambda el: el[item_id_field]).as_numpy_iterator())
    max_length_items_id = len(max(items, key=lambda item_id: len(item_id)))

    x = train_data.repeat(n_repetitions)
    labels = np.empty((l_train * n_repetitions), dtype=np.int32)
    new_items = np.empty((l_train * n_repetitions), dtype=f'>S{max_length_items_id}')
    true_items_added = defaultdict(set)
    user_non_obs_items = {}

    for i, user_item in tqdm.tqdm(users_items.repeat(n_repetitions).enumerate(),
                                  desc="Main operation to generate binary data"):
        _user = user_item[user_id_field].numpy()
        _item = user_item[item_id_field].numpy()

        if _user not in user_non_obs_items:
            user_non_obs_items[_user] = list(set(unique_items) - observed_items[_user])

        if i.numpy() < l_train:
            true_items_added[_user].add(_item)

        label = np.random.choice([0, 1]) if _item in true_items_added[_user] else 0
        # if it is the last repetition and the item with label 1 had not been added yet, label = 1
        if i.numpy() > (l_train * n_repetitions - l_train) and label == 0 and _item in true_items_added[_user]:
            label = 1

        new_items[i] = random.choice(user_non_obs_items[_user]) if label == 0 else _item
        if label == 1:
            true_items_added[_user].remove(_item)

        labels[i] = label

    new_items_labels = tf.data.Dataset.from_tensor_slices({
        item_id_field: tf.constant(new_items),
        "label": tf.constant(labels)
    })

    x = tf.data.Dataset.zip((x, new_items_labels))

    RcLogger.get().debug("Saving binary data")
    save_tf_features_dataset(
        x.batch(save_batch).map(lambda a, b: {**a, **b}, num_parallel_calls=tf.data.AUTOTUNE),
        dataset_metadata,
        dataset_info="binary",
        overwrite=overwrite,
        split=split
    )


def to_input_data_from_metadata(metadata, orig_train, test, train=None, validation=None, **kwargs):
    func_names = RecSysArgumentParser._add_other_features_args.__code__.co_varnames
    func_names = list(filter(lambda param: "input_data" in param, func_names))

    true_funcs = [input_data_f for input_data_f in func_names if metadata[input_data_f]]

    if len(true_funcs) == 0:
        msg = f"No metadata is enabled to know the right function to convert the data. Enable one of the " \
              f"`create_XXXX_input_data` attributes"
        raise RcLoggerException(ValueError, msg)

    for true_f in true_funcs:
        to_func = globals()[f"to_{true_f.replace('create_', '')}"]
        to_func(metadata, orig_train, test, train=train, validation=validation, **kwargs)


def to_nlr_input_data(metadata, orig_train, test, train=None, validation=None, **kwargs):
    users_field = metadata['users_field']
    items_field = metadata['items_field']

    n_reps = metadata['n_reps']

    columns_rename = {users_field: 'uid', items_field: 'iid'}

    if train is None:
        msg = "NLR train data is generated as pointwise data with positive and negative samples. `train` cannot be None"
        raise RcLoggerException(ValueError, msg)

    out_path = os.path.join(
        constants.INPUT_DATA_REPRODUCIBILITY,
        f"{metadata['dataset']}_{metadata['dataset_size']}",
        "nlr_input_data"
    )
    out_filename = f"{metadata['dataset']}_{metadata['dataset_size']}"

    RcLogger.get().info(f"Generating NLR input data in {out_path}")

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    _train: pd.DataFrame = tfds.as_dataframe(train)
    _train = _train[[users_field, items_field, "label"]]
    _train = _train.astype(dtype={users_field: np.int32, items_field: np.int32})
    _train = _train.rename(columns=columns_rename)

    _train.to_csv(
        os.path.join(out_path, f"{out_filename}.train.csv"),
        index=None,
        sep='\t'
    )

    _orig_train: pd.DataFrame = tfds.as_dataframe(orig_train)
    _orig_train = _orig_train[[users_field, items_field]]

    _test: pd.DataFrame = tfds.as_dataframe(test)
    _test = _test[[users_field, items_field]]

    if validation is not None:
        _validation: pd.DataFrame = tfds.as_dataframe(validation)
        val_user_item = _validation[[users_field, items_field]]

        interactions = pd.concat([_orig_train, val_user_item])

        unique_items = pd.concat([interactions, _test])[items_field].unique()

        RcLogger.get().info("Generating binary data for validation set")

        if not preprocessed_dataset_exists(metadata, "binary", split="validation_binary"):
            observed_items, unobserved_items_val, _ = get_train_test_features(users_field,
                                                                              items_field,
                                                                              train_data=orig_train,
                                                                              test_or_val_data=validation)
            observed_items: DefaultDict[Text, Set]
            unobserved_items: DefaultDict[Text, Set]

            total_users = set(list(observed_items.keys())) | set(list(unobserved_items_val.keys()))

            train_val_obs_items = dict()
            for user in total_users:
                train_val_obs_items[user] = observed_items[user] | unobserved_items_val[user]

            generate_binary_data(validation,
                                 train_val_obs_items,
                                 unique_items,
                                 metadata,
                                 n_repetitions=n_reps,
                                 user_id_field=users_field,
                                 item_id_field=items_field,
                                 split="validation_binary")

        _validation = load_tf_features_dataset(metadata,
                                               dataset_info="binary",
                                               split="validation_binary")

        _validation = tfds.as_dataframe(_validation)
        _validation = _validation[[users_field, items_field, "label"]]

        _validation = _validation.astype(dtype={users_field: np.int32, items_field: np.int32})
        _validation = _validation.rename(columns=columns_rename)

        _validation.to_csv(
            os.path.join(out_path, f"{out_filename}.validation.csv"),
            index=None,
            sep='\t'
        )
    else:
        interactions = _orig_train

    # The test created in this function for NLR contains every missing pair user-item that is not present in train or
    # validation data because if only the original test set is used with NLR, the model would predict only
    # the score for the user-item pairs in test set. `unlabel_test = 1` should be set in NLR so as to this could work
    interactions["label"] = 0

    _test["label"] = 1

    _test = pd.concat([interactions, _test])
    _test = _test.astype(dtype={users_field: np.int32, items_field: np.int32})
    _test = _test.rename(columns=columns_rename)

    _test = _test.pivot(
        index=columns_rename[users_field],
        columns=columns_rename[items_field],
        values="label"
    )
    _test.fillna(value=1, inplace=True)

    _test = _test.reset_index()
    _test = _test.melt(
        id_vars=columns_rename[users_field],
        value_vars=_test.columns[_test.columns != columns_rename[users_field]],
        var_name=columns_rename[items_field],
        value_name="label"
    )

    _test = _test.astype(dtype={'label': np.int32})
    _test = _test[_test["label"] == 1]

    _test.to_csv(
        os.path.join(out_path, f"{out_filename}.test.csv"),
        index=None,
        sep='\t'
    )
    
    
def to_user_oriented_fairness_files_input_data(metadata, orig_train, test, train=None, validation=None, **kwargs):
    users_field = metadata['users_field']
    items_field = metadata['items_field']
    rating_field = metadata['rating_field']
    sensitive_field = metadata['sensitive_field']

    sensitive_map = {True: 'M', False: 'F'}

    out_path = os.path.join(
        constants.INPUT_DATA_REPRODUCIBILITY,
        f"{metadata['dataset']}_{metadata['dataset_size']}",
        "user_oriented_fairness_files_input_data"
    )

    RcLogger.get().info(f"Generating input data in {out_path} for User-oriented Fairness in Recommendation")

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if kwargs.get('train_val_as_train', False):
        orig_train = orig_train.concatenate(validation)
        
    model_results_paths = kwargs.get('model_results_paths', None)
    test_path = kwargs.get('test_path', None)
    sensitive_values = kwargs.get('sensitive_values', None)
    
    if model_results_paths is None:
        raise RcLoggerException(ValueError, '`model_results_paths` must be passed to kwargs to generate input data for user-oriented fairness')
    
    if test_path is None:
        raise RcLoggerException(ValueError, '`test_path` must be passed to kwargs to generate input data for user-oriented fairness. It is the test file generated by `create_nlr_input_data`')
        
    if sensitive_values is None:
        raise RcLoggerException(ValueError, '`sensitive_values` must be passed to kwargs to generate input data for user-oriented fairness. '
                                            'The first value will map `True` values, while the second the `False` ones')
                                            
    observed_items, unobserved_items, _, other_returns = data_utils.get_train_test_features(
        users_field,
        items_field,
        train_data=orig_train,
        test_or_val_data=test,
        item_popularity=False,
        sensitive_field=sensitive_field,
        rating_field=rating_field,
        other_returns=["sensitive"]
    )

    sensitive_group = other_returns['sensitive']
                                            
    for nlr_model_name, nlr_path in model_results_paths.items():
        nlr_rm = RelevanceMatrix.from_nlr_models_result(
            nlr_path,
            test_path
        )

        nlr_rm.to_user_oriented_fairness_files(sensitive_field,
                                               sensitive_group,
                                               f"{metadata['dataset']}_{metadata['dataset_size']}_{nlr_model_name}",
                                               unobserved_items,
                                               observed_items,
                                               test,
                                               users_field,
                                               items_field,
                                               map_values_groups=dict(zip([True, False], sensitive_values)),
                                               folderpath=out_path)


def to_co_clustering_for_fair_input_data(metadata, orig_train, test, train=None, validation=None, **kwargs):
    users_field = metadata['users_field']
    items_field = metadata['items_field']
    rating_field = metadata['rating_field']
    sensitive_field = metadata['sensitive_field']

    blocks = []
    sensitive_map = {True: 'M', False: 'F'}

    out_path = os.path.join(
        constants.INPUT_DATA_REPRODUCIBILITY,
        f"{metadata['dataset']}_{metadata['dataset_size']}",
        "co_clustering_for_fair_input_data"
    )

    RcLogger.get().info(f"Generating input data in {out_path} for Co-clustering for fair recommendation")

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if kwargs.get('train_val_as_train', False):
        orig_train = orig_train.concatenate(validation)

    _orig_train, _test, users_map, items_map, _ = __get_mapped_data_and_maps(
        orig_train,
        test,
        users_field=users_field,
        items_field=items_field
    )

    data = pd.concat([_orig_train, _test])

    unique_users = data[users_field].unique()
    unique_items = data[items_field].unique()

    data_users = data[[users_field, sensitive_field]]
    data_users = data_users.groupby(users_field).aggregate('first').reset_index()

    # The sensitive field is mapped to "gender" because the source code of Co-clustering for fair recommendation
    # only contains the experiments with gender, so any sensitive attribute will be mimicked as `gender`
    # and 'M' and 'F' values
    data_users = data_users.rename(columns={users_field: 'userid', sensitive_field: 'gender'})

    RcLogger.get().info("Mapping the values of sensitive attribute for Co-clustering for fair recommendation: "
                        "True to `M`, False to `F`")

    data_users['gender'] = data_users['gender'].map(sensitive_map)

    x_train = scipy.sparse.coo_matrix(
        (
            _orig_train[rating_field].to_numpy().copy(),
            (
                _orig_train[users_field].to_numpy().copy(),
                _orig_train[items_field].to_numpy().copy(),
            ),
        ),
        shape=(unique_users.shape[0], unique_items.shape[0]),
    )

    x_test = scipy.sparse.coo_matrix(
        (
            _test[rating_field].to_numpy().copy(),
            (
                _test[users_field].to_numpy().copy(),
                _test[items_field].to_numpy().copy(),
            ),
        ),
        shape=(unique_users.shape[0], unique_items.shape[0]),
    )

    blocks.append({"X_train": x_train, "X_test": x_test})

    # extra_data save the dictionaries used to map original user ids and item ids to ranges
    with open(os.path.join(out_path, f"{metadata['dataset']}_{metadata['dataset_size']}_extra_data_{sensitive_field}.pkl"), 'wb') as extra_file:
        pickle.dump(
            {
                'users_map': users_map,
                'items_map': items_map
            },
            extra_file,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    # Change path in source code to use this file
    with open(os.path.join(out_path, f"{metadata['dataset']}_{metadata['dataset_size']}_blocks_topk_{sensitive_field}.pkl"), "wb") as data_file:
        pickle.dump(
            {
                "data": data,
                "data_users": data_users,
                "data_movies": None,  # Source code does not use this field
                "blocks": blocks,
            },
            data_file,
            protocol=pickle.HIGHEST_PROTOCOL
        )


def to_fairgo_input_data(metadata, orig_train, test, train=None, validation=None, **kwargs):
    users_field = metadata['users_field']
    items_field = metadata['items_field']
    rating_field = metadata['rating_field']

    ratings_dict_order = [users_field, rating_field, items_field]
    map_sensitive = {True: 1, False: 0}

    out_path = os.path.join(
        constants.INPUT_DATA_REPRODUCIBILITY,
        f"{metadata['dataset']}_{metadata['dataset_size']}",
        "fairgo_input_data"
    )

    RcLogger.get().info(f"Generating input data in {out_path} for "
                        f"Learning Fair Representations for Recommendation. A Graph-based Perspective")

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    _orig_train, _test, users_map, items_map, _validation = __get_mapped_data_and_maps(
        orig_train,
        test,
        users_field=users_field,
        items_field=items_field,
        validation=validation
    )

    _orig_train = _orig_train.astype(dtype={users_field: np.int32, items_field: np.int32, rating_field: np.int32})
    _test = _test.astype(dtype={users_field: np.int32, items_field: np.int32, rating_field: np.int32})
    _validation = _validation.astype(dtype={users_field: np.int32, items_field: np.int32, rating_field: np.int32})

    np.save(
        os.path.join(out_path, f"mapping_user_item.npy"),
        [users_map, items_map],
        allow_pickle=True
    )

    data = pd.concat([_orig_train, _validation, _test]) if _validation is not None else pd.concat([_orig_train, _test])

    training_user_set, training_item_set, training_ratings_dict = {}, {}, {}
    training_u_i_set, training_i_u_set = defaultdict(set), defaultdict(set)

    for user, user_df in _orig_train.groupby(users_field):
        training_user_set[user] = user_df[items_field].to_list()

    for item, item_df in _orig_train.groupby(items_field):
        training_item_set[item] = item_df[users_field].to_list()

    np.save(
        os.path.join(out_path, f"training_set.npy"),
        [training_user_set, training_item_set, None],
        allow_pickle=True
    )

    for row_index, row in _orig_train[ratings_dict_order].iterrows():
        training_ratings_dict[row_index] = row.to_list()
        training_u_i_set[row[users_field]].add((row[rating_field], row[items_field]))
        training_i_u_set[row[items_field]].add((row[rating_field], row[users_field]))

    np.save(
        os.path.join(out_path, f"training_adj_set.npy"),
        [training_u_i_set, training_i_u_set],
        allow_pickle=True
    )

    np.save(
        os.path.join(out_path, f"training_ratings_dict.npy"),
        [training_ratings_dict, len(training_ratings_dict)],
        allow_pickle=True
    )

    if _validation is not None:
        val_user_set, val_item_set = {}, {}

        for user, user_df in _validation.groupby(users_field):
            val_user_set[user] = user_df[items_field].to_list()

        for item, item_df in _validation.groupby(items_field):
            val_item_set[item] = item_df[users_field].to_list()

        np.save(
            os.path.join(out_path, f"val_set.npy"),
            [val_user_set, val_item_set, None],
            allow_pickle=True
        )

        val_ratings_dict = {row_index: row.to_list() for row_index, row in _validation[ratings_dict_order].iterrows()}

        np.save(
            os.path.join(out_path, f"validation_ratings_dict.npy"),
            [val_ratings_dict, len(val_ratings_dict)],
            allow_pickle=True
        )

    testing_user_set, testing_item_set = {}, {}

    for user, user_df in _test.groupby(users_field):
        testing_user_set[user] = user_df[items_field].to_list()

    for item, item_df in _test.groupby(items_field):
        testing_item_set[item] = item_df[users_field].to_list()

    np.save(
        os.path.join(out_path, f"testing_set.npy"),
        [testing_user_set, testing_item_set, None],  # last value is not used
        allow_pickle=True
    )

    testing_ratings_dict = {row_index: row.to_list() for row_index, row in _test[ratings_dict_order].iterrows()}

    np.save(
        os.path.join(out_path, f"testing_ratings_dict.npy"),
        [testing_ratings_dict, len(testing_ratings_dict)],
        allow_pickle=True
    )

    user_rating_set_all = {}
    for user, user_df in data.groupby(users_field):
        user_rating_set_all[user] = set(user_df[items_field].to_list())

    np.save(
        os.path.join(out_path, f"user_rating_set_all.npy"),
        [user_rating_set_all, None, None],  # the other two attributes are not used in the code
        allow_pickle=True
    )

    # Fair Go create a numpy matrix where the rows represent the users (since they are mapped as a range(len(num_users))
    # and each column represent a sensitive attribute (ex. gender and age in MovieLens).
    # `fairgo_sensitive_fields` should be an array with the fields for each one of the selected sensitive attributes.
    # If one of the values is None the column will be randomly filled with 0s and 1s
    fair_go_sensitive_fields = kwargs.get("fairgo_sensitive_fields")
    num = len(fair_go_sensitive_fields)

    sens_list = []
    for sens_field in fair_go_sensitive_fields:
        if sens_field is not None:
            sens_field_dict = dict(zip(data[users_field].to_list(), data[sens_field].to_list()))
            sens_list.append(sens_field_dict)
        else:
            sens_list.append(None)

    users_features = []
    users_new_ids = list(range(len(users_map)))
    for i in range(num):
        if sens_list[i] is None:
            column_data = np.random.randint(0, 2, size=len(users_new_ids))
        else:
            column_data = [map_sensitive[sens_list[i][u]] if isinstance(sens_list[i][u], bool) else sens_list[i][u]
                           for u in users_new_ids ]

        users_features.append(column_data)

    np.save(os.path.join(out_path, f"users_features_{num}num.npy"), np.array(users_features).T, allow_pickle=True)


def to_all_the_cool_kids_input_data(metadata, orig_train, test, train=None, validation=None, **kwargs):
    users_field = metadata['users_field']
    items_field = metadata['items_field']
    rating_field = metadata['rating_field']
    sensitive_field = metadata['sensitive_field']

    yaml_type = 'gender' if 'gender' in sensitive_field else 'age'

    out_path = os.path.join(
        constants.INPUT_DATA_REPRODUCIBILITY,
        f"{metadata['dataset']}_{metadata['dataset_size']}",
        "all_the_cool_kids_input_data"
    )

    RcLogger.get().info(f"Generating input data in {out_path} for "
                        f"All The Cool Kids, How Do They Fit In. Popularity and Demographic Biases "
                        f"in Recommender Evaluation and Effectiveness")

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if "movielens" in metadata['dataset']:
        train_yaml = textwrap.dedent("""\
        ---
        - name: "ML-1M.train"
          type: "textfile"
          file: "movielens_1m_train.csv"
          format: "csv"
          entity_type: "rating"
          metadata: {}
        - type: "textfile"
          file: "../data/ml-1m/users.dat"
          format: "delimited"
          delimiter: "::"
          entity_type: "user"
          base_id: 0
          columns:
          - name: "id"
            type: "long"
          - name: "gender"
            type: "string"
          - name: "age"
            type: "string"
          - name: "occupation"
            type: "int"
          - name: "zip"
            type: "string"
        - type: "textfile"
          file: "../data/ml-1m/movies.dat"
          format: "delimited"
          delimiter: "::"
          entity_type: "item"
          base_id: 0
          columns:
          - name: "id"
            type: "long"
          - name: "title"
            type: "string"
          - name: "genres"
            type: "string"
        """)

        train_balanced_yaml = textwrap.dedent("""\
        ---
        - name: "ML-1M.train"
          type: "textfile"
          file: "movielens_1m_train""" + f'_{sensitive_field}_ekstrand.csv"' + """
          format: "csv"
          entity_type: "rating"
          metadata: {}
        - type: "textfile"
          file: "../data/ml-1m/users.dat"
          format: "delimited"
          delimiter: "::"
          entity_type: "user"
          base_id: 0
          columns:
          - name: "id"
            type: "long"
          - name: "gender"
            type: "string"
          - name: "age"
            type: "string"
          - name: "occupation"
            type: "int"
          - name: "zip"
            type: "string"
        - type: "textfile"
          file: "../data/ml-1m/movies.dat"
          format: "delimited"
          delimiter: "::"
          entity_type: "item"
          base_id: 0
          columns:
          - name: "id"
            type: "long"
          - name: "title"
            type: "string"
          - name: "genres"
            type: "string"
        """)
    else:
        train_yaml = textwrap.dedent("""\
        ---
        - name: "Last.FM-1K.train"
          type: "textfile"
          file: "filtered(20)_lastfm_1K_train.csv"
          format: "csv"
          entity_type: "rating"
          metadata: {}
        """)

        train_balanced_yaml = textwrap.dedent("""\
        ---
        - name: "Last.FM-1K.train"
          type: "textfile"
          file: "filtered(20)_lastfm_1K_train""" + f'_{sensitive_field}_ekstrand.csv"' + """
          format: "csv"
          entity_type: "rating"
          metadata: {}
        """)

    with open(os.path.join(out_path, f"{metadata['dataset']}_{metadata['dataset_size']}_train.yaml"), 'w') as tr_yaml_f:
        tr_yaml_f.write(train_yaml)
        
    with open(os.path.join(
            out_path,
            f"{metadata['dataset']}_{metadata['dataset_size']}_train_{yaml_type}_balanced.yaml"
    ), 'w') as tr_bal_yaml_f:
        tr_bal_yaml_f.write(train_balanced_yaml)

    if "movielens" in metadata['dataset']:
        test_yaml = textwrap.dedent("""\
        ---
        name: "ML-1M.test"
        type: "textfile"
        file: "movielens_1m_test.csv"
        format: "csv"
        entity_type: "rating"
        metadata: {}
        """)
        
        test_balanced_yaml = textwrap.dedent("""\
        ---
        name: "ML-1M.test"
        type: "textfile"
        file: "movielens_1m_test""" + f'_{sensitive_field}_ekstrand.csv"' + """
        format: "csv"
        entity_type: "rating"
        metadata: {}
        """)
    else:
        test_yaml = textwrap.dedent("""\
        ---
        name: "Last.FM-1K.test"
        type: "textfile"
        file: "filtered(20)_lastfm_1K_test.csv"
        format: "csv"
        entity_type: "rating"
        metadata: {}
        """)

        test_balanced_yaml = textwrap.dedent("""\
        ---
        name: "Last.FM-1K.test"
        type: "textfile"
        file: "filtered(20)_lastfm_1K_test""" + f'_{sensitive_field}_ekstrand.csv"' + """
        format: "csv"
        entity_type: "rating"
        metadata: {}
        """)

    with open(os.path.join(out_path, f"{metadata['dataset']}_{metadata['dataset_size']}_test.yaml"), 'w') as te_yaml_f:
        te_yaml_f.write(test_yaml)
        
    with open(os.path.join(
            out_path,
            f"{metadata['dataset']}_{metadata['dataset_size']}_test_{yaml_type}_balanced.yaml"
    ), 'w') as te_bal_yaml_f:
        te_bal_yaml_f.write(test_balanced_yaml)

    # if kwargs.get('train_val_as_train', False):
    #     orig_train = orig_train.concatenate(validation)
    #
    # _orig_train: pd.DataFrame = tfds.as_dataframe(orig_train)
    # _test: pd.DataFrame = tfds.as_dataframe(test)
    #
    # _orig_train = _orig_train.astype({users_field: np.int32, items_field: np.int32, rating_field: np.int32})
    # _test = _test.astype({users_field: np.int32, items_field: np.int32, rating_field: np.int32})
    #
    # _orig_train[[users_field, items_field, rating_field]].to_csv(
    #     os.path.join(out_path, f"{metadata['dataset']}_{metadata['dataset_size']}_train.csv"),
    #     header=None,
    #     index=None
    # )
    # _test[[users_field, items_field, rating_field]].to_csv(
    #     os.path.join(out_path, f"{metadata['dataset']}_{metadata['dataset_size']}_test.csv"),
    #     header=None,
    #     index=None
    # )


def to_rec_independence_input_data(metadata, orig_train, test, train=None, validation=None, **kwargs):
    users_field = metadata['users_field']
    items_field = metadata['items_field']
    rating_field = metadata['rating_field']
    sensitive_field = metadata['sensitive_field']

    sensitive_map = {True: 1, False: 0}

    out_path = os.path.join(
        constants.INPUT_DATA_REPRODUCIBILITY,
        f"{metadata['dataset']}_{metadata['dataset_size']}",
        "rec_independence_input_data"
    )

    RcLogger.get().info(f"Generating input data in {out_path} for Recommendation Independence")

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if kwargs.get('train_val_as_train', False):
        orig_train = orig_train.concatenate(validation)

    _orig_train: pd.DataFrame = tfds.as_dataframe(orig_train)
    _test: pd.DataFrame = tfds.as_dataframe(test)

    _orig_train = _orig_train.astype({users_field: np.int32, items_field: np.int32, rating_field: np.int32})
    _test = _test.astype({users_field: np.int32, items_field: np.int32, rating_field: np.int32})

    _orig_train[sensitive_field] = _orig_train[sensitive_field].map(sensitive_map)
    _test[sensitive_field] = _test[sensitive_field].map(sensitive_map)

    _orig_train[[users_field, items_field, rating_field, sensitive_field]].to_csv(
        os.path.join(out_path, f"{metadata['dataset']}_{metadata['dataset_size']}_train_{sensitive_field}.csv"),
        sep='\t',
        header=None,
        index=None
    )
    _test[[users_field, items_field, rating_field, sensitive_field]].to_csv(
        os.path.join(out_path, f"{metadata['dataset']}_{metadata['dataset_size']}_test_{sensitive_field}.csv"),
        sep='\t',
        header=None,
        index=None
    )


def to_antidote_data_input_data(metadata, orig_train, test, train=None, validation=None, **kwargs):
    users_field = metadata['users_field']
    items_field = metadata['items_field']
    rating_field = metadata['rating_field']
    sensitive_field = metadata['sensitive_field']

    out_path = os.path.join(
        constants.INPUT_DATA_REPRODUCIBILITY,
        f"{metadata['dataset']}_{metadata['dataset_size']}",
        "antidote_data_input_data"
    )

    RcLogger.get().info(f"Generating input data in {out_path} for "
                        f"Fighting Fire with Fire: Using Antidote Data to Improve Polarization "
                        f"and Fairness of Recommender Systems")

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if kwargs.get('train_val_as_train', False):
        orig_train = orig_train.concatenate(validation)

    _orig_train: pd.DataFrame = tfds.as_dataframe(orig_train)
    _test: pd.DataFrame = tfds.as_dataframe(test)

    _orig_train_ratings = _orig_train.pivot(index=users_field, columns=items_field, values=rating_field)
    _test_ratings = _test.pivot(index=users_field, columns=items_field, values=rating_field)

    data = pd.concat([_orig_train, _test])

    orig_train_data = data.copy()
    test_data = data.copy()

    orig_train_data[rating_field] = np.nan
    test_data[rating_field] = np.nan

    orig_train_data = orig_train_data.pivot(index=users_field, columns=items_field, values=rating_field)
    test_data = test_data.pivot(index=users_field, columns=items_field, values=rating_field)

    orig_train_data.update(_orig_train_ratings)
    test_data.update(_test_ratings)

    sens_inv = dict(data[[users_field, sensitive_field]].to_numpy())
    sens = dict.fromkeys(np.unique(list(sens_inv.values())))
    for u_id, sens_attr in sens_inv.items():
        sens[sens_attr] = sens[sens_attr] + [int(u_id)] if isinstance(sens[sens_attr], list) else [int(u_id)]

    orig_train_data.columns = [int(x) for x in orig_train_data.columns]
    orig_train_data.index = [int(x) for x in orig_train_data.index]
    test_data.columns = [int(x) for x in test_data.columns]
    test_data.index = [int(x) for x in test_data.index]

    orig_train_data.to_csv(os.path.join(out_path, f"{metadata['dataset']}_{metadata['dataset_size']}_X.csv"))
    test_data.to_csv(os.path.join(out_path, f"{metadata['dataset']}_{metadata['dataset_size']}_test.csv"))

    with open(os.path.join(out_path, f"{metadata['dataset']}_{metadata['dataset_size']}_sensitive_attribute_({sensitive_field}).pkl"), 'wb') as sens_file:
        pickle.dump(sens, sens_file, protocol=2)


def to_librec_auto_input_data(metadata, orig_train, test, train=None, validation=None, **kwargs):
    users_field = metadata['users_field']
    items_field = metadata['items_field']
    rating_field = metadata['rating_field']
    sensitive_field = metadata['sensitive_field']

    sensitive_map = {True: 'M', False: 'F'}

    out_path = os.path.join(
        constants.INPUT_DATA_REPRODUCIBILITY,
        f"{metadata['dataset']}_{metadata['dataset_size']}",
        "librec_auto_input_data"
    )

    RcLogger.get().info(f"Generating input data in {out_path} for "
                        f"Balanced Neighborhoods for Multi-sided Fairness in Recommendation")

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if kwargs.get('train_val_as_train', False):
        orig_train = orig_train.concatenate(validation)

    _orig_train: pd.DataFrame = tfds.as_dataframe(orig_train)
    _test: pd.DataFrame = tfds.as_dataframe(test)

    _orig_train = _orig_train.astype({users_field: np.int32, items_field: np.int32})
    _test = _test.astype({users_field: np.int32, items_field: np.int32})

    _orig_train[[users_field, items_field, rating_field]].to_csv(
        os.path.join(out_path, f"{metadata['dataset']}_{metadata['dataset_size']}_train.csv"),
        header=None,
        index=None
    )
    _test[[users_field, items_field, rating_field]].to_csv(
        os.path.join(out_path, f"{metadata['dataset']}_{metadata['dataset_size']}_test.csv"),
        header=None,
        index=None
    )

    RcLogger.get().info("Mapping the values of sensitive attribute for Librec Auto (Balanced Neighborhoods "
                        "for Multi-sided Fairness in Recommendation): True to `M`, False to `F`")

    data = pd.concat([_orig_train, _test])

    data_users = data[[users_field, sensitive_field]]
    data_users = data_users.groupby(users_field).aggregate('first')

    data_users[sensitive_field] = data_users[sensitive_field].map(sensitive_map)
    data_users["label"] = 1

    data_users.to_csv(
        os.path.join(out_path, f"{metadata['dataset']}_{metadata['dataset_size']}_user-features_({sensitive_field}).csv"),
        header=None
    )


def to_rating_prediction_fairness_input_data(metadata, orig_train, test, train=None, validation=None, **kwargs):
    users_field = metadata['users_field']
    items_field = metadata['items_field']
    rating_field = metadata['rating_field']
    sensitive_field = metadata['sensitive_field']

    columns_rename = {users_field: 'user', items_field: 'item', rating_field: 'rating', sensitive_field: 'gender'}
    sensitive_map = {True: 'M', False: 'F'}

    out_path = os.path.join(
        constants.INPUT_DATA_REPRODUCIBILITY,
        f"{metadata['dataset']}_{metadata['dataset_size']}",
        "rating_prediction_fairness_input_data"
    )

    RcLogger.get().info(f"Generating input data in {out_path} for "
                        f"Fairness metrics and bias mitigation strategies for rating prediction")

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if kwargs.get('train_val_as_train', False):
        orig_train = orig_train.concatenate(validation)

    _orig_train: pd.DataFrame = tfds.as_dataframe(orig_train)
    _test: pd.DataFrame = tfds.as_dataframe(test)

    _orig_train = _orig_train.astype({users_field: np.int32, items_field: np.int32})
    _test = _test.astype({users_field: np.int32, items_field: np.int32})

    _orig_train[sensitive_field] = _orig_train[sensitive_field].map(sensitive_map)
    _test[sensitive_field] = _test[sensitive_field].map(sensitive_map)

    # Possibility of using a dataset that has a field called `gender`
    if 'gender' in _orig_train.columns and sensitive_field != 'gender':
        del _orig_train['gender']
        del _test['gender']

    _orig_train = _orig_train.rename(columns=columns_rename)
    _test = _test.rename(columns=columns_rename)

    _orig_train.to_csv(
        os.path.join(out_path, f"{metadata['dataset']}_{metadata['dataset_size']}_train_{sensitive_field}.csv"),
        index=None
    )

    _test.to_csv(
        os.path.join(out_path, f"{metadata['dataset']}_{metadata['dataset_size']}_test_{sensitive_field}.csv"),
        index=None
    )


def __get_mapped_data_and_maps(train, test, users_field='user_id', items_field='item_id', validation=None):
    _orig_train: pd.DataFrame = tfds.as_dataframe(train)
    _test: pd.DataFrame = tfds.as_dataframe(test)
    _validation: pd.DataFrame = tfds.as_dataframe(validation) if validation is not None else None

    data = pd.concat([_orig_train, _validation, _test]) if _validation is not None else pd.concat([_orig_train, _test])

    unique_users = data[users_field].unique()
    unique_items = data[items_field].unique()

    users_map = dict(zip(unique_users, np.arange(unique_users.shape[0])))
    items_map = dict(zip(unique_items, np.arange(unique_items.shape[0])))

    _orig_train[users_field] = _orig_train[users_field].map(users_map)
    _orig_train[items_field] = _orig_train[items_field].map(items_map)

    _test[users_field] = _test[users_field].map(users_map)
    _test[items_field] = _test[items_field].map(items_map)

    if _validation is not None:
        _validation[users_field] = _validation[users_field].map(users_map)
        _validation[items_field] = _validation[items_field].map(items_map)

    return _orig_train, _test, users_map, items_map, _validation

# TODO: Modify structure of preprocessed_datasets
"""
 - movielens |
 .           |
 .           - 100k ..
 .           - 1m   |
 - other            |
                    - per_user_timestamp_[None, 4, 4] ..
                    - per_user_timestamp_[80%, 20%]   |
                                                      |
                                                      - orig_train
                                                      - test
                                                      - train
                                                      - filtered |
                                                                 - min_10
                                                                 - min_5
                                                                 - balanced ..
                                                                 ..
                                                      - balanced |
                                                                 - Gender_[1500, 1500] |
                                                                                       - orig_train
                                                                                       - test
                                                                                       - binary ...
                                                                 - Age_[1000, 1000]
                                                                 ..
                                                      - binary ..
                                                      - triplets |
                                                                 |
                                                                 - 10_reps
                                                                 - 5_reps
"""


def preprocessed_dataset_exists(dataset_metadata: Dict[Text, Any],
                                model_data_type: str = None,
                                split=None):
    model_data_type = _check_update_dataset_metadata(dataset_metadata, model_data_type)

    return tf_features_dataset_exists(dataset_metadata, dataset_info=model_data_type, split=split)


def load_train_val_test(dataset_metadata: Dict[Text, Any],
                        model_data_type: str = None):
    RcLogger.get().info("Loading train test data")

    model_data_type = _check_update_dataset_metadata(dataset_metadata, model_data_type)

    train = load_tf_features_dataset(dataset_metadata, dataset_info=model_data_type, split="orig_train")
    test = load_tf_features_dataset(dataset_metadata, dataset_info=model_data_type, split="test")
    if tf_features_dataset_exists(dataset_metadata, dataset_info=model_data_type, split="validation"):
        val = load_tf_features_dataset(dataset_metadata, dataset_info=model_data_type, split="validation")
    else:
        val = None

    return train, val, test


# To call only if model does not generate specific training data, like Pointwise or BPR
def save_train_test(dataset_metadata: Dict[Text, Any],
                    train: tf.raw_ops.BatchDataset,
                    test: tf.data.Dataset,
                    validation: tf.data.Dataset = None,
                    model_data_type=None,
                    overwrite=False):
    model_data_type = _check_update_dataset_metadata(dataset_metadata, model_data_type)

    if model_data_type is None:
        dataset_metadata['n_reps'] = ''

    if not preprocessed_dataset_exists(dataset_metadata, model_data_type) or overwrite:
        save_tf_features_dataset(
            train.batch(8192),
            dataset_metadata,
            dataset_info=model_data_type,
            overwrite="all",
            split="orig_train"
        )

        save_tf_features_dataset(
            test.batch(2048),
            dataset_metadata,
            dataset_info=model_data_type,
            overwrite=overwrite,
            split="test"
        )

        if validation is not None:
            save_tf_features_dataset(
                validation.batch(1024),
                dataset_metadata,
                dataset_info=model_data_type,
                overwrite=overwrite,
                split="validation"
            )


def _check_update_dataset_metadata(dataset_metadata, model_data_type):
    import models

    dataset_info = ''

    if model_data_type is None:
        model = dataset_metadata.get('model', None)

        if model is None:
            msg = f"no model_data_type has been provided and dataset_metadata does not contain `model` metadata. " \
                  f"One of them is necessary to infer the data type to load"
            raise RcLoggerException(KeyError, msg)

        if isinstance(model, str):
            model_data_type = getattr(models, model).MODEL_DATA_TYPE
        elif isinstance(model, models.Model):
            model_data_type = model.MODEL_DATA_TYPE
        else:
            msg = f"model must be a string or an instance of models.Model. Got {type(model)}"
            raise RcLoggerException(ValueError, msg)

    # some models have `model_data_type` set to None, so it could still be None and n_reps must be unset
    if model_data_type is None or model_data_type == "":
        dataset_metadata['n_reps'] = ''

    if dataset_metadata.get('min_interactions', 0) > 0:
        dataset_info += constants.PREFIX_FILTERED_DATASET.format(
            min_interactions=dataset_metadata['min_interactions']
        )

    if dataset_metadata.get('balance_ratio', None) is not None:
        dataset_info += constants.PREFIX_BALANCED_DATASET.format(
            balance_attribute=dataset_metadata['balance_attribute'],
            balance_ratio=str(list(dataset_metadata['balance_ratio'].items()))
        )

    if dataset_metadata.get('sample_n', None) is not None:
        sample_attribute = dataset_metadata['sample_attribute']

        dataset_info += constants.PREFIX_SAMPLED_DATASET.format(
            sample_n=dataset_metadata['sample_n'],
            sample_attribute="" if sample_attribute is None else sample_attribute
        )

    if dataset_info not in dataset_metadata['dataset']:
        dataset_metadata['dataset'] = dataset_info + dataset_metadata['dataset']

    return model_data_type if model_data_type is not None else ''


class ModelData(UserDict):
    ModelAttr = namedtuple('ModelAttr', ['data', 'mapping', 'batch'], defaults=(None, None))

    required_attribs = ["users", "items"]

    def __init__(self, data=None, **kwargs):
        if not kwargs.get("prepared"):
            msg = "ModelData need to be instantiated by the 'prepare_model_data' method, not directly"
            raise RcLoggerException(ValueError, msg)
        super(ModelData, self).__init__(data)

    @staticmethod
    def prepare_model_data(model_data: Dict[Text, Union[ModelAttr, tf.data.Dataset, list]],
                           batch=2048,
                           **kwargs) -> "ModelData":
        RcLogger.get().debug("Preparing model data")

        for k in ModelData.required_attribs:
            if k not in model_data:
                msg = f"{k} not present in input dictionary, it is required. " \
                      f"The required keys are {ModelData.required_attribs}"
                raise RcLoggerException(ValueError, msg)

        data = {}
        for k, v in model_data.items():
            if isinstance(v, ModelData.ModelAttr):
                if len(v) == 3:
                    """
                    if batch is not None:
                        msg = "Passing batch in sequence values and as argument is ambiguous. Did you want " \
                              "to pass only 2 values in sequence or you forgot to remove 'batch' argument?"
                        raise RcLoggerException(ValueError, msg)
                    else:
                    """
                    if v.batch is not None and v.mapping is not None:
                        data[k] = v.data.map(lambda x: x[v.mapping]).batch(v.batch)
                    else:
                        if v.batch is not None:
                            data[k] = v.data.batch(v.batch)
                        elif v.mapping is not None:
                            data[k] = v.data.map(lambda x: x[v.mapping])
                        else:
                            data[k] = v.data
                else:
                    msg = f"Only sequences of arity of 2 or 3 are supported, got {len(v)} values instead"
                    raise RcLoggerException(NotImplementedError, msg)
            elif isinstance(v, tf.data.Dataset):
                data[k] = v
            elif isinstance(v, list):
                data[k] = tf.data.Dataset.from_tensor_slices(v).batch(batch)
            else:
                msg = f"dict_users_items can only have sequences or {tf.data.Dataset} as values"
                raise RcLoggerException(NotImplementedError, msg)

        # TODO: implement various kwargs, pop each one of them in ModelData if present.
        #  They will be attributes of ModelData class

        RcLogger.get().info(f"Model data prepared")

        return ModelData(data, prepared=True, **kwargs)

    def save(self, dirpath, other_attrs: Dict[str, Any] = None):
        other_attrs = {} if other_attrs is None else other_attrs
        data = {}

        for attrib in self:
            data[attrib] = list(self[attrib].as_numpy_iterator())

        with open(os.path.join(dirpath, 'ModelData.pickle'), 'wb') as model_data_file:
            pickle.dump({**data, **other_attrs}, model_data_file, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(dirpath):
        with open(os.path.join(dirpath, 'ModelData.pickle'), 'rb') as model_data_file:
            model_data = pickle.load(model_data_file)

        for field in model_data:
            # model_data[field] = tf.data.Dataset.from_tensor_slices(general_utils.dicts_append(*model_data[field]))
            if len(model_data[field][0].shape) == 0:
                data = model_data[field]
                batch_size = None
            else:
                batch_size = model_data[field][0].shape[0]
                data = np.concatenate(model_data[field])

            model_data[field] = tf.data.Dataset.from_tensor_slices(data)

            if batch_size is not None:
                model_data[field] = model_data[field].batch(batch_size)

        return ModelData.prepare_model_data(model_data)


def save_tf_features_dataset(dataset: tf.data.Dataset,
                             metadata: Dict[Text, Any],
                             dataset_info='',
                             overwrite=False,
                             split: Union[
                                 Literal["orig_train"],
                                 Literal["train"],
                                 Literal["test"],
                                 Literal["validation"],
                                 Literal["validation_binary"]
                             ] = None):
    reps = metadata.get('n_reps', '')
    reps = f'n({reps})' if reps != '' else ''
    dataset_info = '' if dataset_info is None else dataset_info

    foldername = f"{dataset_info}{'_' if dataset_info != '' else ''}" \
                 f"{metadata['dataset']}_" \
                 f"{metadata['dataset_size']}_" \
                 f"{reps}{'_' if reps != '' else ''}" \
                 f"{metadata['train_val_test_split_type']}_" \
                 f"{str(list(metadata['train_val_test_split']))}".replace('\'', '')

    fdt_writer = FeaturesDictTensorsWriter(os.path.join(constants.SAVE_PRE_PROCESSED_DATASETS_PATH, foldername))

    RcLogger.get().info(f"Saving features dataset at path "
                        f"{foldername}{f' of split {split}' if split is not None else ''}")
    fdt_writer.save(dataset, overwrite=overwrite, split=split)

    return foldername


def load_tf_features_dataset(metadata: Dict[Text, Any],
                             dataset_info='',
                             split: Union[
                                 Literal["orig_train"],
                                 Literal["train"],
                                 Literal["test"],
                                 Literal["validation"],
                                 Literal["validation_binary"]
                             ] = None):
    reps = metadata.get('n_reps', '')
    reps = f'n({reps})' if reps != '' else ''
    dataset_info = '' if dataset_info is None else dataset_info

    foldername = f"{dataset_info}{'_' if dataset_info != '' else ''}" \
                 f"{metadata['dataset']}_" \
                 f"{metadata['dataset_size']}_" \
                 f"{reps}{'_' if reps != '' else ''}" \
                 f"{metadata['train_val_test_split_type']}_" \
                 f"{str(list(metadata['train_val_test_split']))}".replace('\'', '')

    RcLogger.get().info(f"Loading features dataset at path "
                        f"{foldername}{f' of split {split}' if split is not None else ''}")
    fdt_reader = FeaturesDictTensorsReader(os.path.join(constants.SAVE_PRE_PROCESSED_DATASETS_PATH, foldername))

    return fdt_reader.load(split=split)


def tf_features_dataset_exists(metadata: Dict[Text, Any],
                               dataset_info='',
                               split: Union[
                                   Literal["orig_train"],
                                   Literal["train"],
                                   Literal["test"],
                                   Literal["validation"],
                                   Literal["validation_binary"]
                               ] = None):
    reps = metadata.get('n_reps', '')
    reps = f'n({reps})' if reps != '' else ''
    dataset_info = '' if dataset_info is None else dataset_info

    foldername = f"{dataset_info}{'_' if dataset_info != '' else ''}" \
                 f"{metadata['dataset']}_" \
                 f"{metadata['dataset_size']}_" \
                 f"{reps}{'_' if reps != '' else ''}" \
                 f"{metadata['train_val_test_split_type']}_" \
                 f"{str(list(metadata['train_val_test_split']))}".replace('\'', '')

    fdt_reader = FeaturesDictTensorsReader(os.path.join(constants.SAVE_PRE_PROCESSED_DATASETS_PATH, foldername))

    return fdt_reader.exists(split=split)


class FeaturesDictTensorsIO(object):

    _features_dict_name = "features_dict.json"
    _tensor_path_key = "path"
    _tensor_dtype_key = "dtype"

    def __init__(self, foldername):
        self._foldername = foldername
        self._save_formatter = "tensor_{key}_{batch}.tensor"

    @staticmethod
    def _check_dataset_integrity(dataset: tf.data.Dataset):
        value_type = next(dataset.take(1).as_numpy_iterator())

        if not isinstance(value_type, dict):
            msg = f"expected tf.data.Dataset of featuresDict, got tf.data.Dataset of {value_type}"
            raise RcLoggerException(TypeError, msg)


class FeaturesDictTensorsWriter(FeaturesDictTensorsIO):

    def __init__(self, foldername):
        super(FeaturesDictTensorsWriter, self).__init__(foldername)

    def save(self, dataset: tf.data.Dataset, overwrite: Union[bool, Literal["all"]] = False, split=None):
        json_features = {}

        FeaturesDictTensorsIO._check_dataset_integrity(dataset)

        if isinstance(overwrite, bool):
            if overwrite and os.path.exists(self._foldername):
                if split is not None:
                    split_path = os.path.join(self._foldername, split)
                    if os.path.exists(split):
                        shutil.rmtree(split_path)
        elif isinstance(overwrite, str) and overwrite.lower() == "all":
            if os.path.exists(self._foldername):
                shutil.rmtree(self._foldername)

        if not os.path.exists(self._foldername):
            os.mkdir(self._foldername)
        else:
            with open(os.path.join(self._foldername, self._features_dict_name), 'r') as jsonfile:
                json_features = json.load(jsonfile)

            if split is None:
                msg = f"FeaturesDictTensors dataset already exists at path {self._foldername}"
                raise RcLoggerException(FileExistsError, msg)
            elif split in json_features:
                msg = f"The json branch {split} already exists in json features_dict"
                raise RcLoggerException(AttributeError, msg)

        try:
            features_dict = {}

            if split is not None:
                features_keys_path = os.path.join(self._foldername, split)
            else:
                features_keys_path = self._foldername

            for i, features in dataset.enumerate():
                for feature_key in features:
                    f_key_path = os.path.join(features_keys_path, feature_key)

                    if not os.path.exists(f_key_path):
                        os.makedirs(f_key_path)

                    tensors_key_path = os.path.join(f_key_path, self._save_formatter)

                    tensor_filepath = tensors_key_path.format(key=feature_key, batch=i.numpy())
                    tf.io.write_file(tensor_filepath, tf.io.serialize_tensor(features[feature_key]))

                    if feature_key not in features_dict:
                        features_dict[feature_key] = {
                            self._tensor_path_key: [tensor_filepath],
                            self._tensor_dtype_key: features[feature_key].dtype.__repr__()
                        }
                    else:
                        features_dict[feature_key][self._tensor_path_key].append(tensor_filepath)

            if split is not None:
                json_features[split] = features_dict
            else:
                json_features = features_dict

            with open(os.path.join(self._foldername, self._features_dict_name), 'w') as jsonfile:
                json.dump(json_features, jsonfile)
        except Exception as e:
            shutil.rmtree(self._foldername)
            msg = f"Error saving FeaturesDictTensors dataset at path {self._foldername} -- {e}"
            raise RcLoggerException(Exception, msg)


class FeaturesDictTensorsReader(FeaturesDictTensorsIO):

    def __init__(self, foldername):
        super(FeaturesDictTensorsReader, self).__init__(foldername)

    def load(self, split=None):
        if not os.path.exists(self._foldername):
            msg = f"{self._foldername} does not exist"
            raise RcLoggerException(FileNotFoundError, msg)
        elif not os.path.exists(os.path.join(self._foldername, self._features_dict_name)):
            msg = f"not found any FeaturesDictTensors dataset at path {self._foldername}"
            raise RcLoggerException(FileNotFoundError, msg)

        features_dict_json = os.path.join(self._foldername, self._features_dict_name)

        try:
            with open(features_dict_json, 'r') as jsonfile:
                json_features = json.load(jsonfile)

            if split is not None:
                if split not in json_features:
                    msg = f"The json features_dict does not contain the branch {split}"
                    raise RcLoggerException(AttributeError, msg)
                else:
                    features_dict = json_features[split]
            else:
                features_dict = json_features

            for feature_key in features_dict:
                tensor_dtype = tf.dtypes.as_dtype(features_dict[feature_key][self._tensor_dtype_key].replace('tf.', ''))
                serialized_tensors = []
                for serialized_tensors_path in features_dict[feature_key][self._tensor_path_key]:
                    ser_tensor = tf.io.read_file(serialized_tensors_path)
                    serialized_tensors.append(tf.io.parse_tensor(ser_tensor, out_type=tensor_dtype))

                features_dict[feature_key] = tf.concat(serialized_tensors, axis=0)
        except Exception as e:
            msg = f"Error loading FeaturesDictTensors dataset at path {self._foldername}, features description or " \
                  f"serialized tensors files could be corrupted -- {e}"
            raise RcLoggerException(Exception, msg)

        return tf.data.Dataset.from_tensor_slices(features_dict)

    def exists(self, split: str = None):
        if os.path.exists(os.path.join(self._foldername, self._features_dict_name)):
            with open(os.path.join(self._foldername, self._features_dict_name), 'r') as jsonfile:
                features_dict = json.load(jsonfile)

            return True if split is None else split in features_dict

        return False


def check_dataset_errors(data: tf.data.Dataset,
                         check_errors_func: Callable,
                         action: Union[
                             Literal["raise"],
                             Literal["print"],
                             Literal["log_info"],
                             Literal["log_debug"],
                             Callable
                         ] = "print",
                         *func_args,
                         **func_kwargs):

    def raise_msg(message):
        raise ValueError(message)

    if isinstance(action, str):
        if action == "raise":
            action = raise_msg
        elif action == "print":
            action = print
        elif action == "log_info":
            action = RcLogger.get().info
        elif action == "log_debug":
            action = RcLogger.get().debug
        else:
            msg_error = "action must be one of ('raise', 'print', 'log_info', 'log_debug') " \
                        "or a custom function"
            raise RcLoggerException(ValueError, msg_error)
    elif not isinstance(action, Callable):
        msg_error = "action must be one of ('raise', 'print', 'log_info', 'log_debug') " \
                    "or a custom function"
        raise RcLoggerException(ValueError, msg_error)

    check_errors_func(data, action, *func_args, **func_kwargs)
