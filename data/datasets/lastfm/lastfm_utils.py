from typing import Union

import pandas as pd
import sklearn.preprocessing as sk_pre

import tensorflow as tf
import tensorflow_datasets as tfds


def normalize_ratings(dataset: Union[tf.data.Dataset, pd.DataFrame]) -> Union[tf.data.Dataset, pd.DataFrame]:
    data = dataset
    if not isinstance(dataset, pd.DataFrame):
        data: pd.DataFrame = tfds.as_dataframe(dataset)

    plays = data['plays'].to_numpy()
    plays = plays.reshape((len(plays), 1))

    power_trans = sk_pre.PowerTransformer(standardize=False)
    plays_pt = power_trans.fit_transform(plays)

    min_max_scaler = sk_pre.MinMaxScaler(feature_range=(1, 5))
    plays_mms = min_max_scaler.fit_transform(plays_pt)

    data['user_rating'] = plays_mms.flatten()

    if not isinstance(dataset, pd.DataFrame):
        data_records = data.to_dict(orient='list')

        return tf.data.Dataset.from_tensor_slices(data_records)

    return data
