import argparse
import ast
from typing import Sequence, Union, Literal, Callable


class RecSysArgumentParser(object):

    def __init__(self):
        self._arg_parser = argparse.ArgumentParser(description="Routine example of usage of recommender codebase")
        self._dataset_columns_sep = '-'
        self._sized_datasets = ["movielens"]

    def recsysparser(parser):
        def wrapper(self):
            args = parser(self)

            if "dataset_columns" in args:
                if self._dataset_columns_sep in args.dataset_columns:
                    args.dataset_columns = RecSysArgumentParser._parse_dataset_columns(args.dataset_columns,
                                                                                       self._dataset_columns_sep)
                else:
                    args.dataset_columns = [args.dataset_columns]

            return args

        return wrapper

    @recsysparser
    def routine_arg_parser(self):
        self._add_logger_args()
        self._add_dataset_args()
        self._add_train_test_split_args()
        self._add_users_items_map_batch_args()
        self._add_model_args()
        self._add_index_args()
        self._add_optimizer_args()
        self._add_training_args()
        self._add_testing_args()
        self._add_saving_functionality_args()
        self._add_metrics_args()
        self._add_other_features_args()

        return self._arg_parser.parse_args()

    @recsysparser
    def train_arg_parser(self):
        # TODO: create argument parser for a possible train.py
        raise NotImplementedError()

    @recsysparser
    def test_arg_parser(self):
        self._add_logger_args()
        self._add_dataset_args()
        self._add_model_args()
        self._add_train_test_split_args()
        self._add_index_args()
        self._add_optimizer_args(optimizer=None)
        self._add_testing_args()
        self._add_saving_functionality_args()
        self._add_metrics_args()
        self._add_loading_args()
        self._add_other_features_args()

        return self._arg_parser.parse_args()

    @staticmethod
    def _parse_dataset_columns(dataset_columns: Sequence[str], sep='-'):
        return list(map(lambda x: x.split(), ' '.join(dataset_columns).split(sep)))

    @staticmethod
    def _check_bool_or_int(string):
        if string.isdigit():
            return int
        elif string == "True" or string == "False":
            return string == "True"
        else:
            raise argparse.ArgumentTypeError("argument can only be a number or a boolean (True, False)")

    @staticmethod
    def _check_none_in_nargs(val):
        return None if val == "None" else val

    @staticmethod
    def _default_or_custom_filepath(string):
        if string == "True" or string == "False":
            return string == "True"

        return r"{}".format

    def _add_logger_args(self,
                         level="INFO"):
        self._arg_parser.add_argument('--log_level',
                                      type=str,
                                      default=level,
                                      help="the level of the logger")

    def _add_dataset_args(self,
                          dataset="movielens",
                          split: Sequence[str] = None,
                          size="1m",
                          subdatasets: Sequence[str] = None,
                          dataset_columns: Sequence[Sequence[str]] = None,
                          n_reps=10,
                          min_interactions=0,
                          balance_attribute="user_gender",
                          balance_ratio=None,
                          attribute_to_binary=None,
                          binary_le_delimiter=None,
                          sample_n=None,
                          sample_attribute=None):
        split = ["train", "train"] if split is None else None
        subdatasets = ["ratings", "movies"] if subdatasets is None else None
        dataset_columns = [["user_id", "movie_id", "user_rating", "user_gender", "timestamp"],
                           ["movie_title", "movie_id", "movie_genres"]] if dataset_columns is None else None

        self._arg_parser.add_argument('-dataset',
                                      type=str,
                                      default=dataset,
                                      help="dataset to be used. ex: 'movielens'")
        self._arg_parser.add_argument('--dataset_split',
                                      default=split,
                                      help="splits of the dataset to be used. ex: 'train'",
                                      nargs='+')
        self._arg_parser.add_argument('--dataset_size',
                                      type=str,
                                      default=size if dataset in self._sized_datasets else None,
                                      help="size of the dataset. ex: '100k' for movielens")
        self._arg_parser.add_argument('--subdatasets',
                                      default=subdatasets,
                                      help="sub-datasets to be taken the data from. ex: 'ratings' for movielens",
                                      nargs='+')
        self._arg_parser.add_argument('--dataset_columns',
                                      default=dataset_columns,
                                      help="columns to be used for each dataset/subdataset. ex: 'movie_title', "
                                           "separate columns of different subdataset with a dash '-'. ex: "
                                           "'movie_title' 'user_id' '-' 'movie_id'",
                                      nargs='+')
        self._arg_parser.add_argument('--n_reps',
                                      type=int,
                                      default=n_reps,
                                      help="number of times training data must be repeated. Used for pairwise and "
                                           "pointwise models")
        self._arg_parser.add_argument('--min_interactions',
                                      type=int,
                                      default=min_interactions,
                                      help="minimum number of interactions per user to consider")
        self._arg_parser.add_argument('--balance_attribute',
                                      type=str,
                                      default=balance_attribute,
                                      help="dataset will be balanced by the values of this attribute")
        self._arg_parser.add_argument('--balance_ratio',
                                      default=balance_ratio,
                                      help="dictionary that maps `balance_attribute` values to "
                                           "proportion or fixed value that are used to sample each subgroup defined",
                                      type=ast.literal_eval)
        self._arg_parser.add_argument('--attribute_to_binary',
                                      default=attribute_to_binary,
                                      help="attribute that must be binarised to True and False. It can only contain "
                                           "numeric values",
                                      type=str)
        self._arg_parser.add_argument('--binary_le_delimiter',
                                      default=binary_le_delimiter,
                                      help="float delimiter that convert all less or equal values to `True`",
                                      type=float)
        self._arg_parser.add_argument('--sample_n',
                                      type=int,
                                      default=sample_n,
                                      help="number of data entries to sample. If sample_attribute is not None "
                                           "`sample_n` unique values of `sample_attribute` values are taken and only "
                                           "the data entries related to the sampled values are considered.")
        self._arg_parser.add_argument('--sample_attribute',
                                      type=str,
                                      default=sample_attribute,
                                      help="dataset will be filtered with the sampled values of this attribute")

    def _add_train_test_split_args(self,
                                   train_val_test_split: Union[str, Sequence[Union[str, int]]] = False,
                                   split_type="per_user_timestamp",
                                   seed=False,
                                   reshuffle_each_iteration=False,
                                   n_folds=5):
        train_val_test_split = ["80%", "20%"] if train_val_test_split is False else train_val_test_split

        self._arg_parser.add_argument('--train_val_test_split',
                                      default=train_val_test_split,
                                      type=RecSysArgumentParser._check_none_in_nargs,
                                      nargs='+',
                                      help="how to split the data in train, validation and test. If only two values "
                                           "are used they will be assigned for train and test. ex: '80%' 'None' '20%'")
        self._arg_parser.add_argument('--train_val_test_split_type',
                                      type=str,
                                      default=split_type,
                                      help="the type of splitting that must be performed")
        self._arg_parser.add_argument('--seed_shuffle',
                                      type=RecSysArgumentParser._check_bool_or_int,
                                      default=seed,
                                      help="seed to be set to shuffle the data. It can be an integer or a boolean")
        self._arg_parser.add_argument('--reshuffle_each_iteration',
                                      default=reshuffle_each_iteration,
                                      help="boolean to set if data need to be shuffled each iteration",
                                      action="store_true")
        self._arg_parser.add_argument('--n_folds',
                                      type=int,
                                      default=n_folds,
                                      help="number of folds for k-fold cross-validation")

    def _add_users_items_map_batch_args(self,
                                        users_field="user_id",
                                        users_batch=1_000_000,
                                        items_field="movie_id",
                                        items_batch=1_000,
                                        category_field=None,
                                        sensitive_field=None,
                                        rating_field="user_rating"):
        self._arg_parser.add_argument('--users_field',
                                      type=str,
                                      default=users_field,
                                      help="column to be used to create the array containing all the users")
        self._arg_parser.add_argument('--users_batch',
                                      type=int,
                                      default=users_batch,
                                      help="batch size to be used for the array containing all the users")
        self._arg_parser.add_argument('--items_field',
                                      type=str,
                                      default=items_field,
                                      help="column to be used to create the array containing all the items")
        self._arg_parser.add_argument('--items_batch',
                                      type=int,
                                      default=items_batch,
                                      help="batch size to be used for the array containing all the items")
        self._arg_parser.add_argument('--category_field',
                                      type=str,
                                      default=category_field,
                                      help="column to be used to create the array containing all the items categories "
                                           "ex. `movie_genres` in MovieLens")
        self._arg_parser.add_argument('--sensitive_field',
                                      type=str,
                                      default=sensitive_field,
                                      help="column to be used to create the dict containing sensitive data of all the "
                                           "users. ex. `user_gender` in MovieLens")
        self._arg_parser.add_argument('--rating_field',
                                      type=str,
                                      default=rating_field,
                                      help="rating field")

    def _add_model_args(self,
                        model="BPR",
                        embedding_dimension=10,
                        independence_term="mean_matching"):
        self._arg_parser.add_argument('-model',
                                      type=str,
                                      default=model,
                                      help="model to train or load")
        self._arg_parser.add_argument('--embedding_dimension',
                                      type=int,
                                      default=embedding_dimension,
                                      help="embedding dimension for user and item embedding layers")
        self._arg_parser.add_argument('--independence_term',
                                      type=str,
                                      default=independence_term,
                                      choices=["mean_matching", "bdist_matching", "mi_normal"],
                                      help="Kamishima's independence term to be used for training the model. Model "
                                           "must be a subclass of `IndependentModel`")

    def _add_index_args(self,
                        index="BruteForce",
                        k=100,
                        identifiers_map="movie_id",
                        candidates_batch=128):
        self._arg_parser.add_argument('-index',
                                      type=str,
                                      default=index,
                                      help="index for the retrieval stage")
        self._arg_parser.add_argument('--index_k',
                                      type=int,
                                      default=k,
                                      help="number of results to retrieve")
        self._arg_parser.add_argument('--identifiers_map',
                                      type=str,
                                      default=identifiers_map,
                                      help="column of items that will be returned to identify each item retrieved by "
                                           "the index")
        self._arg_parser.add_argument('--candidates_batch',
                                      type=int,
                                      default=candidates_batch,
                                      help="batch size to be used for the array containing all the candidates for the "
                                           "index")

    def _add_optimizer_args(self,
                            optimizer="Adam",
                            learning_rate=0.001):
        self._arg_parser.add_argument('--optimizer',
                                      type=str,
                                      default=optimizer,
                                      help="keras optimizer to be used")
        self._arg_parser.add_argument('--learning_rate',
                                      type=float,
                                      default=learning_rate,
                                      help="learning rate for the optimizer")

    def _add_training_args(self,
                           train_batch=8192,
                           train_be_cached=False,
                           epochs=40,
                           overwrite_preprocessed_dataset=False,
                           check_preprocessed_dataset_errors: Union[
                               Literal["raise"],
                               Literal["print"],
                               Literal["log_info"],
                               Literal["log_debug"],
                               Callable
                           ] = None):
        self._arg_parser.add_argument('--train_batch',
                                      type=int,
                                      default=train_batch,
                                      help="batch size for the data that will be used for training the model")
        self._arg_parser.add_argument('--train_be_cached',
                                      default=train_be_cached,
                                      help="boolean to choose if train data must be cached",
                                      action="store_true")
        self._arg_parser.add_argument('--epochs',
                                      type=int,
                                      default=epochs,
                                      help="number of epochs")
        self._arg_parser.add_argument('--overwrite_preprocessed_dataset',
                                      default=overwrite_preprocessed_dataset,
                                      help="boolean to choose if particular train data shoulde be recreated, "
                                           "e.g. triplets positive-negative or for binary training",
                                      action="store_true")
        self._arg_parser.add_argument('--check_preprocessed_dataset_errors',
                                      type=str,
                                      default=check_preprocessed_dataset_errors,
                                      choices=["print", "raise", "log_info", "log_debug", Callable],
                                      help="if not None check errors in data generation and its value is the action "
                                           "to perform if an error is found")

    def _add_testing_args(self,
                          test_batch=4096,
                          test_be_cached=False,
                          return_dict=True):
        self._arg_parser.add_argument('--test_batch',
                                      type=int,
                                      default=test_batch,
                                      help="batch size for the data that will be used for testing the model")
        self._arg_parser.add_argument('--test_be_cached',
                                      default=test_be_cached,
                                      help="boolean to choose if test data must be cached",
                                      action="store_true")
        self._arg_parser.add_argument('--return_evaluation_dict',
                                      default=return_dict,
                                      help="boolean to choose if model.evaluate must return a dictionary",
                                      action="store_true")

    def _add_saving_functionality_args(self,
                                       save_model: Union[str, bool] = True,
                                       save_model_format='.tf',
                                       save_relevance_matrix: Union[str, bool] = True,
                                       output_type_relevance_matrix="csv",
                                       numpy_style_relevance_matrix="arrays",
                                       save_index: Union[str, bool] = False,
                                       save_metrics=True,
                                       output_type_metrics='csv'):
        self._arg_parser.add_argument('--save_model',
                                      type=RecSysArgumentParser._default_or_custom_filepath,
                                      default=save_model,
                                      help="boolean for default filepath or string for custom filepath where the model "
                                           "will be saved")
        self._arg_parser.add_argument('--save_model_format',
                                      type=str,
                                      default=save_model_format,
                                      help="file format which will be used to save the model")
        self._arg_parser.add_argument('--save_relevance_matrix',
                                      type=RecSysArgumentParser._default_or_custom_filepath,
                                      default=save_relevance_matrix,
                                      help="boolean for default filepath or string for custom filepath where the "
                                           "relevance matrix will be saved")
        self._arg_parser.add_argument('--output_type_relevance_matrix',
                                      type=str,
                                      default=output_type_relevance_matrix,
                                      choices=["csv", "numpy"],
                                      help="the type of output to save the relevance matrix")
        self._arg_parser.add_argument('--numpy_style_relevance_matrix',
                                      type=str,
                                      default=numpy_style_relevance_matrix,
                                      choices=["arrays", "matrix"],
                                      help="the structure style to use to save the relevance matrix with NumPy")
        self._arg_parser.add_argument('--save_index',
                                      type=RecSysArgumentParser._default_or_custom_filepath,
                                      default=save_index,
                                      help="boolean for default filepath or string for custom filepath where the index "
                                           "will be saved")
        self._arg_parser.add_argument('--save_metrics',
                                      type=RecSysArgumentParser._default_or_custom_filepath,
                                      default=save_metrics,
                                      help="boolean for default filepath or string for custom filepath where the "
                                           "metrics will be saved")
        self._arg_parser.add_argument('--output_type_metrics',
                                      type=str,
                                      default=output_type_metrics,
                                      choices=["csv", "npy"],
                                      help="the type of output to save the metrics")

    def _add_metrics_args(self,
                          metrics: Union[Sequence[str], str] = "all",
                          cutoffs: Sequence[int] = None,
                          only_metrics_type=None):
        cutoffs = [1, 5, 10, 50, 100] if cutoffs is None else cutoffs

        self._arg_parser.add_argument('--metrics',
                                      default=metrics,
                                      help="list of `custom` metrics to compute",
                                      nargs='+')
        self._arg_parser.add_argument('--cutoffs',
                                      type=int,
                                      default=cutoffs,
                                      help="list of cutoffs to compute top-k metrics for each k in cutoffs",
                                      nargs='+')
        self._arg_parser.add_argument('--only_metrics_type',
                                      type=str,
                                      default=only_metrics_type,
                                      help="string to choose which metrics types should be computed only")

    def _add_loading_args(self,
                          run_id=None):
        self._arg_parser.add_argument('run_id',
                                      type=str,
                                      default=run_id,
                                      help="identifier of the session of the model")

    def _add_other_features_args(self,
                                 create_user_oriented_fairness_files=False,
                                 create_nlr_input_data=False,
                                 create_co_clustering_for_fair_input_data=False,
                                 create_fairgo_input_data=False,
                                 create_all_the_cool_kids_input_data=False,
                                 create_rec_independence_input_data=False,
                                 create_antidote_data_input_data=False,
                                 create_librec_auto_input_data=False,
                                 create_rating_prediction_fairness_input_data=False):
        self._arg_parser.add_argument('--create_user_oriented_fairness_files_input_data',
                                      default=create_user_oriented_fairness_files,
                                      help="boolean to create the files to be used in user-oriented fairness reranking "
                                           "https://github.com/rutgerswiselab/user-fairness",
                                      action="store_true")
        self._arg_parser.add_argument('--create_nlr_input_data',
                                      default=create_nlr_input_data,
                                      help="boolean to create the files to be used in NLR code "
                                           "https://github.com/rutgerswiselab/NLR",
                                      action="store_true")
        self._arg_parser.add_argument('--create_co_clustering_for_fair_input_data',
                                      default=create_co_clustering_for_fair_input_data,
                                      help="boolean to create the files to be used in Parity LBM "
                                           "https://hal.archives-ouvertes.fr/hal-03239856",
                                      action="store_true")
        self._arg_parser.add_argument('--create_fairgo_input_data',
                                      default=create_fairgo_input_data,
                                      help="boolean to create the files to be used in FairGo "
                                           "https://github.com/newlei/FairGo",
                                      action="store_true")
        self._arg_parser.add_argument('--create_all_the_cool_kids_input_data',
                                      default=create_all_the_cool_kids_input_data,
                                      help="boolean to create the files to be used for All the cool kids "
                                           "http://proceedings.mlr.press/v81/ekstrand18b/ekstrand18b.pdf",
                                      action="store_true")
        self._arg_parser.add_argument('--create_rec_independence_input_data',
                                      default=create_rec_independence_input_data,
                                      help="boolean to create the files to be used for Recommendation Independence "
                                           "https://github.com/tkamishima/kamiers",
                                      action="store_true")
        self._arg_parser.add_argument('--create_antidote_data_input_data',
                                      default=create_antidote_data_input_data,
                                      help="boolean to create the files to be used for antidote-data-framework "
                                           "https://github.com/rastegarpanah/antidote-data-framework",
                                      action="store_true")
        self._arg_parser.add_argument('--create_librec_auto_input_data',
                                      default=create_librec_auto_input_data,
                                      help="boolean to create the files to be used for librec_auto "
                                           "https://github.com/that-recsys-lab/librec-auto",
                                      action="store_true")
        self._arg_parser.add_argument('--create_rating_prediction_fairness_input_data',
                                      default=create_rating_prediction_fairness_input_data,
                                      help="boolean to create the files to be used for "
                                           "Fairness metrics and bias mitigation strategies for rating prediction "
                                           "https://doi.org/10.1016/j.ipm.2021.102646",
                                      action="store_true")
