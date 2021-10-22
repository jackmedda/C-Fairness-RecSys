import os
import inspect
import ast
import subprocess
import shutil
import gc
import time
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import surprise
import tensorflow_datasets as tfds

import data.utils as data_utils
from helpers.logger import RcLogger
from models.utils import RelevanceMatrix


def run_experiment(filename, algorithm, _cols):
    train_df = pd.read_csv(os.path.join(bal_surprise_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(bal_surprise_path, "test.csv"))

    unique_users = sorted([ast.literal_eval(_user) for _user in train_df[_cols[0]].unique()], key=int)
    unique_items = sorted([ast.literal_eval(_item) for _item in pd.concat([train_df, test_df])[_cols[1]].unique()],
                          key=int)

    # A reader is still needed but only the rating_scale param is required.
    reader = surprise.Reader(line_format="user item rating", sep=',', skip_lines=1)

    folds_files = [(os.path.join(bal_surprise_path, "train.csv"), os.path.join(bal_surprise_path, "test.csv"))]

    data = surprise.Dataset.load_from_folds(folds_files, reader=reader)
    pkf = surprise.model_selection.PredefinedKFold()

    if algorithm == "UserKNN":
        alg = surprise.KNNBasic(k=30, sim_options={"name": "cosine", "user_based": True})   # UserKNN
    elif algorithm == "SVD":
        alg = surprise.SVD()   # SVD
    elif algorithm == "PMF":
        alg = surprise.SVD(biased=False)   # PMF
    elif algorithm == "ItemKNN":
        alg = surprise.KNNBasic(k=20, sim_options={"name": "cosine", "user_based": False})    # ItemKNN
    else:
        raise ValueError(f"Algorithm `{algorithm}` not correct")

    for train_data, test_data in pkf.split(data):
        alg.fit(train_data)

    relevances = []
    for _user in unique_users:
        rels = []
        for _item in unique_items:
            rels.append(alg.predict(str(_user), str(_item)).est)
        relevances.append(rels)

    RelevanceMatrix(unique_users, relevances, unique_items).save(filename=filename)


def ekstrand_balance(_orig_train, _test, metadata):
    # Out type of splits guarantee that all users are in both train and test

    users_field = metadata['users_field']
    balance_attribute = metadata['balance_attribute']
    balance_ratio = metadata['balance_ratio']

    sens_map = dict(zip(_orig_train[users_field].to_list(), _orig_train[balance_attribute].to_list()))

    gr_true = [u for u in sens_map if sens_map[u]]
    gr_false = [u for u in sens_map if not sens_map[u]]

    users_true = np.random.choice(gr_true, balance_ratio[True], replace=False)
    users_false = np.random.choice(gr_false, balance_ratio[False], replace=False)

    _orig_train = _orig_train[_orig_train[users_field].isin(np.concatenate([users_true, users_false]))]
    _test = _test[_test[users_field].isin(np.concatenate([users_true, users_false]))]

    return _orig_train, _test


if __name__ == "__main__":
    RcLogger.start_logger(level="INFO")

    ########################################################################################
    ################ CHANGE PATHS AND USER_FEATURE_APPENDER IN LIBREC FILES ################
    ############ IF YOU ARE GOING TO CHANGE THE SENSITIVE ATTRIBUTE WITH LIBREC ############
    ############ OR MODIFY THE YAML FILE OR BUILD.GRADLE FILE FOR LENSKIT GROOVY ###########
    ########################################################################################

    overwrite = True
    experiment = "ekstrand"
    # experiment = "baseline"

    # dataset = "movielens_1m"
    dataset = "lastfm_1K"

    use_external_code = "lenskit_groovy"  # ["librec", "lenskit_groovy"] any other value uses surprise and `algorithm`

    current_wd = os.getcwd()
    n_experiments = 5

    base_path = os.path.dirname(inspect.getsourcefile(lambda: 0))
    baseline_path = os.path.join(base_path, "baseline")
    bal_surprise_path = os.path.join(base_path, "balanced_surprise_path")
    ekstrand_path = os.path.join(base_path, os.pardir, "Ekstrand et al")
    
    if not os.path.exists(baseline_path):
        os.mkdir(baseline_path)
        
    if not os.path.exists(bal_surprise_path):
        os.mkdir(bal_surprise_path)

    algorithm = "ItemKNN"  # used only for surprise

    if dataset == "movielens_1m":
        dataset_metadata = {
            'dataset': 'movielens',
            'dataset_size': '1m',
            'n_reps': 2,
            'train_val_test_split_type': 'per_user_timestamp',
            'train_val_test_split': ["70%", "10%", "20%"],
            'users_field': 'user_id'
        }

        # sensitive_attribute = "user_gender"
        sensitive_attribute = "bucketized_user_age"

        columns = ["user_id", "movie_id", "user_rating"]
    else:
        dataset_metadata = {
            'dataset': 'lastfm',
            'dataset_size': '1K',
            'n_reps': 2,
            'min_interactions': 20,
            'train_val_test_split_type': 'per_user_random',
            'train_val_test_split': ["70%", "10%", "20%"],
            'users_field': 'user_id'
        }

        # sensitive_attribute = "user_gender"
        sensitive_attribute = "user_age"

        columns = ["user_id", "artist_id", "user_rating"]

    print(experiment, dataset, algorithm, sensitive_attribute, f"external_code: {use_external_code}")

    orig_train, val, test = data_utils.load_train_val_test(dataset_metadata, "binary")

    if val is not None:
        orig_train = orig_train.concatenate(val)

    orig_train: pd.DataFrame = tfds.as_dataframe(orig_train)
    test: pd.DataFrame = tfds.as_dataframe(test)

    if (len(os.listdir(bal_surprise_path)) == 0 and experiment != 'baseline') or \
         (len(os.listdir(baseline_path)) == 0 and experiment == 'baseline') or overwrite:
        if experiment == "baseline":
            if use_external_code.lower() == "lenskit_groovy":
                results_path = os.path.join(
                    ekstrand_path,
                    "results",
                    f"{dataset_metadata['dataset']}_{dataset_metadata['dataset_size']}",
                    f"baseline"
                )

                if not os.path.exists(results_path):
                    os.makedirs(results_path)

                orig_train.astype(int)[columns].to_csv(
                    os.path.join(ekstrand_path, "custom_data", f"train.csv"),
                    index=False,
                    header=False
                )
                test.astype(int)[columns].to_csv(
                    os.path.join(ekstrand_path, "custom_data", f"test.csv"),
                    index=False,
                    header=False
                )

                os.chdir(ekstrand_path)

                start_time = time.time()
                if dataset == "movielens_1m":
                    print(subprocess.run(r"gradlew.bat evaluateML", capture_output=True, text=True).stdout)
                else:
                    print(subprocess.run(r"gradlew.bat evaluateLFM1KReproduced", capture_output=True, text=True).stdout)
                print(f"Time computation: {datetime.timedelta(seconds=time.time() - start_time)}")

                predictions = pd.read_csv(os.path.join(
                    ekstrand_path,
                    'build',
                    f"{dataset_metadata['dataset']}_{dataset_metadata['dataset_size']}_predictions.csv"
                ))
                recommendations = pd.read_csv(os.path.join(
                    ekstrand_path,
                    'build',
                    f"{dataset_metadata['dataset']}_{dataset_metadata['dataset_size']}_recommend.csv"
                ))

                for alg, alg_df in predictions.groupby("Algorithm"):
                    alg_df[["User", "Item", "Prediction"]].to_csv(
                        os.path.join(results_path, f"{alg}_predictions_baseline.csv")
                    )

                for alg, alg_df in recommendations.groupby("Algorithm"):
                    if 'E' not in alg:
                        alg_df[["User", "Item", "Score"]].to_csv(
                            os.path.join(results_path, f"{alg}_recommend_baseline.csv")
                        )

                os.remove(os.path.join(ekstrand_path, 'build', f"{dataset_metadata['dataset']}_{dataset_metadata['dataset_size']}_predictions.csv"))
                os.remove(os.path.join(ekstrand_path, 'build', f"{dataset_metadata['dataset']}_{dataset_metadata['dataset_size']}_recommend.csv"))

                os.chdir(current_wd)
            else:
                orig_train[columns].to_csv(os.path.join(baseline_path, "train.csv"), index=False)
                test[columns].to_csv(os.path.join(baseline_path, "test.csv"), index=False)

                run_experiment(algorithm, algorithm, columns)
        else:
            if dataset == "movielens_1m":
                dataset_metadata["balance_ratio"] = {True: 1500, False: 1500}
                dataset_metadata["balance_attribute"] = sensitive_attribute

                base_experiment_name = f"balanced[1500, 1500]-{dataset_metadata['balance_attribute']}-{algorithm}"
            else:
                dataset_metadata["balance_ratio"] = {True: 100, False: 100}
                dataset_metadata["balance_attribute"] = sensitive_attribute

                base_experiment_name = f"balanced[100, 100]-{dataset_metadata['balance_attribute']}-{algorithm}"

            for run in range(n_experiments):
                experiment_name = f"{base_experiment_name}-run#{run + 1}"

                bal_orig_train, bal_test = ekstrand_balance(orig_train, test, dataset_metadata)

                if use_external_code.lower() == "librec":
                    librec_path = r'C:\Users\Giacomo\Desktop\University\Dottorato di Ricerca\Idee paper - Progetti\Reproducibility Study\Librec-auto'

                    librec_models = ['User-User_Explicit', 'User-User', 'Item-Item', 'SVD++_Explicit', 'SVD++_Implicit']
                    librec_paths = ['User-User_Explicit', 'User-User', 'Item-Item', os.path.join('SVD++', 'Explicit'), os.path.join('SVD++', 'Implicit')]

                    if run == 0:
                        print(librec_models)

                    librec_sens = "Gender" if sensitive_attribute == "user_gender" else "Age"
                    result_files = os.listdir(os.path.join(librec_path, "data", "Results_for_Ekstrand", librec_sens, librec_paths[0]))
                    new_run = max(map(lambda x: int(x.split('-')[1].replace('.txt', '')), result_files)) + 1 if result_files else 1

                    os.chdir(librec_path)

                    if new_run == 21:
                        exit()

                    bal_orig_train.astype(int)[columns].to_csv(
                        os.path.join(
                            librec_path,
                            "data",
                            f"{dataset_metadata['dataset']}_{dataset_metadata['dataset_size']}_train_{sensitive_attribute}_ekstrand.csv"
                        ),
                        index=False,
                        header=False
                    )
                    bal_test.astype(int)[columns].to_csv(
                        os.path.join(
                            librec_path,
                            "data",
                            f"{dataset_metadata['dataset']}_{dataset_metadata['dataset_size']}_test_{sensitive_attribute}_ekstrand.csv"
                        ),
                        index=False,
                        header=False
                    )

                    for librec_m, lib_path in zip(librec_models, librec_paths):
                        print(librec_m)
                        subprocess.run(f"\"{os.path.join(librec_path, 'venv', 'Scripts', 'python')}\" "
                                       f"-m librec_auto run -t {librec_m}", stdout=subprocess.PIPE, input="y", encoding="ascii")

                        os.rename(
                            os.path.join(librec_path, librec_m, "exp00000", "result", "out-1.txt"),
                            os.path.join(librec_path, "data", "Results_for_Ekstrand", librec_sens, lib_path, f"out-{new_run}.txt")
                        )

                        shutil.rmtree(os.path.join(librec_path, librec_m, "exp00000"))

                    os.chdir(current_wd)
                elif use_external_code.lower() == "lenskit_groovy":
                    ekstr_sens = "gender" if sensitive_attribute == "user_gender" else "age"

                    results_path = os.path.join(
                        ekstrand_path,
                        "results",
                        f"{dataset_metadata['dataset']}_{dataset_metadata['dataset_size']}",
                        f"{ekstr_sens}_balanced"
                    )

                    if not os.path.exists(results_path):
                        os.makedirs(results_path)

                    result_files = os.listdir(os.path.join(results_path))
                    new_run = max(map(lambda x: int(x.split('run_')[1].replace('.csv', '')), result_files)) + 1 if result_files else 1

                    print(f"{base_experiment_name}-run#{new_run}")

                    if new_run == n_experiments + 1:
                        exit()

                    bal_orig_train.astype(int)[columns].to_csv(
                        os.path.join(
                            ekstrand_path,
                            "custom_data",
                            f"{dataset_metadata['dataset']}_{dataset_metadata['dataset_size']}_train_{sensitive_attribute}_ekstrand.csv"
                        ),
                        index=False,
                        header=False
                    )
                    bal_test.astype(int)[columns].to_csv(
                        os.path.join(
                            ekstrand_path,
                            "custom_data",
                            f"{dataset_metadata['dataset']}_{dataset_metadata['dataset_size']}_test_{sensitive_attribute}_ekstrand.csv"
                        ),
                        index=False,
                        header=False
                    )

                    os.chdir(ekstrand_path)

                    start_time = time.time()
                    if dataset == "movielens_1m":
                        print(subprocess.run(r"gradlew.bat evaluateML", capture_output=True, text=True).stdout)
                    else:
                        print(subprocess.run(r"gradlew.bat evaluateLFM1KReproduced", capture_output=True, text=True).stdout)
                    print(f"Time computation: {datetime.timedelta(seconds=time.time() - start_time)}")

                    predictions = pd.read_csv(os.path.join(
                        ekstrand_path,
                        'build',
                        f"{dataset_metadata['dataset']}_{dataset_metadata['dataset_size']}_predictions.csv"
                    ))
                    recommendations = pd.read_csv(os.path.join(
                        ekstrand_path,
                        'build',
                        f"{dataset_metadata['dataset']}_{dataset_metadata['dataset_size']}_recommend.csv"
                    ))

                    for alg, alg_df in predictions.groupby("Algorithm"):
                        alg_df[["User", "Item", "Prediction"]].to_csv(
                            os.path.join(results_path, f"{alg}_predictions_{sensitive_attribute}_run_{new_run}.csv")
                        )

                    for alg, alg_df in recommendations.groupby("Algorithm"):
                        if 'E' not in alg:
                            alg_df[["User", "Item", "Score"]].to_csv(
                                os.path.join(results_path, f"{alg}_recommend_{sensitive_attribute}_run_{new_run}.csv")
                            )

                    os.remove(os.path.join(ekstrand_path, 'build', f"{dataset_metadata['dataset']}_{dataset_metadata['dataset_size']}_predictions.csv"))
                    os.remove(os.path.join(ekstrand_path, 'build', f"{dataset_metadata['dataset']}_{dataset_metadata['dataset_size']}_recommend.csv"))

                    os.chdir(current_wd)
                else:
                    print(f"Ekstrand Experiment with {algorithm}")

                    bal_orig_train[columns].to_csv(os.path.join(bal_surprise_path, "train.csv"), index=False)
                    bal_test[columns].to_csv(os.path.join(bal_surprise_path, "test.csv"), index=False)

                    run_experiment(experiment_name, algorithm, columns)

                gc.collect()
