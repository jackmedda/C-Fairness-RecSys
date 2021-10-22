import os
import pandas as pd
import numpy as np
import datetime
import pickle
import sys
from scipy.sparse import coo_matrix

print("Start processing MovieLens 1m dataset")
dateset_file = "data/ml-1m_five_blocks_topk_gender.pkl"
print("\tDataset will be saved at " + dateset_file)

data = pd.read_csv(
    "data/ratings.dat",
    delimiter="::",
    header=None,
    names=["userid", "movieid", "rate", "timestamp"],
    engine="python",
)

data_movies = pd.read_csv(
    "data/movies.dat",
    delimiter="::",
    header=None,
    names=["movieid", "title", "gender"],
    engine="python",
)

data_users = pd.read_csv(
    "data/users.dat",
    delimiter="::",
    header=None,
    names=["userid", "gender", "age", "occupation", "zipcode"],
    engine="python",
)

# nb_tt_ratings = data.count()["rate"]
#
# mintopk = 20
# # At least 40 evaluation for each movie.
# b = (data.groupby("movieid").count() > 40).iloc[:, 0]
# movies_to_keep = b.index[b]
# data_movies = data_movies[data_movies.movieid.isin(movies_to_keep)]
# data = data[data.movieid.isin(data_movies.movieid)]
#
# # At least 40 evaluation for each user.
# a = (data.groupby("userid").count() > 40).iloc[:, 0]
# users_to_keep = a.index[a]
# data_users = data_users[data_users.userid.isin(users_to_keep)]
# data = data[data.userid.isin(users_to_keep)]
# data_movies = data_movies[data_movies.movieid.isin(data.movieid.unique())]
#
#
# data.userid = data.userid.replace(
#     data_users.userid.to_numpy(), np.arange(data_users.shape[0])
# )
# data_users.userid = data_users.userid.replace(
#     data_users.userid.to_numpy(), np.arange(data_users.shape[0])
# )
#
# data.movieid = data.movieid.replace(
#     data_movies.movieid.to_numpy(), np.arange(data_movies.shape[0])
# )
# data_movies.movieid = data_movies.movieid.replace(
#     data_movies.movieid.to_numpy(), np.arange(data_movies.shape[0])
# )
#
# blocks = []
# nb_folds = 5
# for iter in range(nb_folds):
#     print(f"""\tCreating {iter}/{nb_folds}-folds for cross validation""")
#     data_train = data[:0]
#     data_test = data[:0]
#     data = data.sample(frac=1)
#     for u in data_users.userid:
#         data_test = data_test.append(data[data.userid == u][:mintopk])
#         data_train = data_train.append(data[data.userid == u][mintopk:])
#
#     X_train = coo_matrix(
#         (
#             data_train.rate.to_numpy().copy(),
#             (
#                 data_train.userid.to_numpy().copy(),
#                 data_train.movieid.to_numpy().copy(),
#             ),
#         ),
#         shape=(data_users.shape[0], data_movies.shape[0]),
#     )
#     X_test = coo_matrix(
#         (
#             data_test.rate.to_numpy().copy(),
#             (
#                 data_test.userid.to_numpy().copy(),
#                 data_test.movieid.to_numpy().copy(),
#             ),
#         ),
#         shape=(data_users.shape[0], data_movies.shape[0]),
#     )
#     blocks.append({"X_train": X_train, "X_test": X_test})

unique_movies = np.unique(data.movieid)

users_map = dict(zip(data_users.userid, np.arange(data_users.shape[0])))
items_map = dict(zip(unique_movies, np.arange(unique_movies.shape[0])))

data.userid = data.userid.replace(users_map, value=None)
data_users.userid = data_users.userid.replace(users_map, value=None)

data.movieid = data.movieid.replace(items_map, value=None)

blocks = []

user_field = "userid"
item_field = "movieid"
time_field = "timestamp"
train_split = 0.8
test_split = 0.2
validation_split = None

train_set = []
test_set = []
val_set = []
groups = data.groupby([user_field])
for i, (_, group) in enumerate(groups):
    sorted_group = group.sort_values(time_field)

    if isinstance(train_split, float) or isinstance(test_split, float):
        n_rating_train = int(len(sorted_group.index) * train_split) if train_split is not None else 0
        n_rating_test = int(len(sorted_group.index) * test_split) if test_split is not None else 0
        n_rating_val = int(len(sorted_group.index) * validation_split) if validation_split is not None else 0

        if len(sorted_group.index) > (n_rating_train + n_rating_test + n_rating_val):
            n_rating_train += len(sorted_group.index) - (n_rating_train + n_rating_test + n_rating_val)

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

data_train, data_test = pd.concat(train_set), pd.concat(test_set)


X_train = coo_matrix(
    (
        data_train.rate.to_numpy().copy(),
        (
            data_train.userid.to_numpy().copy(),
            data_train.movieid.to_numpy().copy(),
        ),
    ),
    shape=(data_users.shape[0], unique_movies.shape[0]),
)
X_test = coo_matrix(
    (
        data_test.rate.to_numpy().copy(),
        (
            data_test.userid.to_numpy().copy(),
            data_test.movieid.to_numpy().copy(),
        ),
    ),
    shape=(data_users.shape[0], unique_movies.shape[0]),
)
blocks.append({"X_train": X_train, "X_test": X_test})


pickle.dump(
    {
        'users_map': users_map,
        'items_map': items_map
    },
    open(os.path.join('data', f"extra_data_{'age' if 'age' in dateset_file else 'gender'}.pkl"), 'wb'),
    protocol=pickle.HIGHEST_PROTOCOL
)

if "age" in dateset_file:
    data_users["gender"] = data_users["age"].map(lambda x: "F" if x <= 25 else "M")


pickle.dump(
    {
        "data": data,
        "data_users": data_users,
        "data_movies": data_movies,
        "blocks": blocks,
    },
    open(dateset_file, "wb"),
    protocol=pickle.HIGHEST_PROTOCOL
)

print("Dataset saved in " + str(dateset_file))
