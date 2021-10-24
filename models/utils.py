import os
import sys
import inspect
import functools
import re
import json
import pickle
from typing import Union, Optional, Literal, Dict, Text, Any, Iterable, Sequence

import tensorflow_recommenders as tfrs
import tensorflow as tf
import pandas as pd
import numpy as np
import tqdm
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from scipy.optimize.optimize import vecnorm, _line_search_wolfe12, _LineSearchError, _prepare_scalar_function, _epsilon

import helpers.filename_utils as file_utils
import helpers.general_utils as gen_utils
import helpers.constants as constants
from helpers.logger import RcLogger, RcLoggerException


class RetrievalIndex(object):

    def __init__(self, index: Union[str, tf.keras.Model], k=10, **kwargs):
        RcLogger.get().debug(f"Initializing index with k = {k}")

        self.k = k
        self._indexes_without_query_model = ["Streaming"]
        self.identifiers = None

        if "identifiers" in kwargs:
            self.identifiers = kwargs.pop("identifiers")

        if "add_wo_query_model" in kwargs:
            RcLogger.get().debug(f"Adding {kwargs.get('add_wo_query_model')} to the indexes without query model")
            self._indexes_without_query_model.extend([ind for ind in kwargs.pop("add_wo_query_model")])

        if isinstance(index, str):
            self.name = index
            top_k_classes = inspect.getmembers(sys.modules[tfrs.layers.factorized_top_k.__name__], inspect.isclass)
            top_k_classes = [cl[0] for cl in top_k_classes
                             if isinstance(cl[1], type(tfrs.layers.factorized_top_k.TopK)) and cl[0] != "TopK"]

            if index in top_k_classes:
                self._index = functools.partial(getattr(tfrs.layers.factorized_top_k, self.name), k=k, **kwargs)
            else:
                msg = f"{index} is not a Tensorflow retrieval layer"
                raise RcLoggerException(ValueError, msg)
        elif isinstance(index, tf.keras.Model):
            self._index = index
            self.name = self._index.__class__.__name__
        else:
            msg = f"{type(index)} is not a supported type for the index of RetrievalIndex"
            raise RcLoggerException(ValueError, msg)

    def __call__(self, query: tf.Tensor, k=None):
        k = self.k if k is None else k
        if not isinstance(self._index, functools.partial):
            return self._index(query, k=k)
        else:
            msg = "You must call 'complete_index' before using index method if this index is a newly generated index"
            raise RcLoggerException(ValueError, msg)

    def complete_index(self, query_model: tfrs.Model):
        RcLogger.get().debug("Completing index with query_model if index supports it")

        if self.expect_query_model() and query_model is not None:
            self._index = self._index(query_model)
        else:
            self._index = self._index()

    def expect_query_model(self):
        return self.name if self.name is None else self.name not in self._indexes_without_query_model

    def has_query_model(self):
        return bool(getattr(self._index, 'query_model', None))

    def index(self, candidates: tf.data.Dataset, identifiers: Optional[tf.data.Dataset] = None):
        if not isinstance(self._index, functools.partial):
            self._index.index(candidates, identifiers)
        else:
            msg = "You must call 'complete_index' before using index method if this index is a newly generated index"
            raise RcLoggerException(ValueError, msg)

    def save(self, filepath=file_utils.default_filepath(), model=None, save_format="tf", **kwargs):
        if self.name == "Streaming":
            msg = "Impossible to save Streaming index. Not supported by Tensorflow"
            raise RcLoggerException(NotImplementedError, msg)

        if not isinstance(self._index, functools.partial):
            filepath = f"{self.name}-{model + '-' if model is not None else ''}{filepath}"
            filepath = os.path.join(constants.SAVE_INDEXES_PATH, filepath)
            model_save(self._index.save, filepath, save_format=save_format, **kwargs)
            RcLogger.get().debug(f"Index saved at path '{filepath}'")
        else:
            msg = "'complete_index' must be called before using save method if this index is a newly generated index"
            raise RcLoggerException(ValueError, msg)

    @staticmethod
    def load(filepath=None, session_id=None):
        if filepath is not None:
            return RetrievalIndex(tf.keras.models.load_model(filepath))
        elif session_id is not None:
            indexes = [f for f in os.scandir(constants.SAVE_INDEXES_PATH) if os.path.isdir(f) and session_id in f.name]
            return RetrievalIndex(tf.keras.models.load_model(indexes[0]))
        else:
            msg = "filepath and session_id are both None, no params to load a saved index model"
            raise RcLoggerException(ValueError, msg)


class RelevanceMatrix(object):

    _as_numpy_arrays_names = ["users", "items", "relevances"]
    _map_type_func_save = {
        "csv": "_save_relevance_matrix_dataframe",
        "numpy": "_save_relevance_matrix_numpy"
    }

    def __init__(self,
                 unique_users: np.ndarray,
                 relevances: np.ndarray,
                 recommended_items: np.ndarray,
                 **kwargs):
        self._unique_users = np.asarray(unique_users)
        self._relevances = np.asarray(relevances)
        self._recommended_items = np.asarray(recommended_items)
        """
        for row in range(len(self._recommended_items)):
            items_sort_indices = np.argsort([int(i) for i in self._recommended_items[row]])
            self._recommended_items[row] = self._recommended_items[row][items_sort_indices]
            self._relevances[row] = self._relevances[row][items_sort_indices]
        """

        self._k = f"{kwargs.get('k', '')}-"
        self._model_name = f"{kwargs.get('model_name', '')}-"

    def as_dataframe(self, style="table", as_category=False):
        """
        Returns the relevance matrix as a pandas dataframe based on the data in the instance of RelevanceMatrix
        :param style: ="columns" each data array is mapped to a column in the dataframe.
                      ="table" takes for granted that each row of recommended_items contain all the items of the
                               dataset, this is necessary to use items identifiers as pandas columns
        :param as_category: user_id and item_id are converted to "category" type, original ids are mapped respectively
                            to user_id_orig and item_id_orig columns
        :return: a pandas dataframe accordingly to the chosen params
        """
        num_items = len(self._recommended_items)
        if style == "columns":
            df = pd.DataFrame({
                "user_id": self._unique_users.repeat(num_items),
                "item_id": np.tile(self._recommended_items, len(self._unique_users)),
                "relevances": self._relevances.flatten()
            })

            if as_category:
                df["user_id_orig"] = df["user_id"]
                df["user_id"] = df["user_id"].astype("category").cat.codes
                df["item_id_orig"] = df["item_id"]
                df["item_id"] = df["item_id"].astype("category").cat.codes

        elif style == "table":
            df = pd.DataFrame(
                self._relevances,
                columns=self._recommended_items,
                index=self._unique_users
            )
        else:
            msg = f"as_dataframe does not support {style} style"
            raise RcLoggerException(NotImplementedError, msg)

        return df

    def as_numpy(self, style="arrays"):
        if style == "matrix":
            return self._relevances
        elif style == "arrays":
            return self._unique_users, self._recommended_items, self._relevances
        else:
            msg = f"as_numpy does not support style = {style}"
            raise RcLoggerException(NotImplementedError, msg)

    def save(self,
             filename=None,
             output_type="csv",
             **kwargs):
        """

        :param filename:
        :param output_type: ["dataframe", "numpy"].
        :param kwargs: "numpy_style": ["matrix", "arrays"], k=k used to compute top k relevances
        :return:
        """
        RcLogger.get().debug(f"Saving relevance_matrix as '{output_type}'")

        filename = f"relevance_matrix-{file_utils.default_filepath()}" if filename is None else \
            f"{filename}-k({self._k})__{file_utils.current_run_id()}"

        numpy_style = kwargs.pop("numpy_style") if "numpy_style" in kwargs else "matrix"
        if output_type == "numpy":
            kwargs = {**kwargs, "numpy_style": numpy_style}

        filepath = os.path.join(constants.SAVE_RELEVANCE_MATRIX_PATH, f"{self._model_name}{filename}")

        getattr(self, self._map_type_func_save[output_type])(filepath, **kwargs)

    def _save_relevance_matrix_dataframe(self,
                                         filepath,
                                         file_format="csv",
                                         **kwargs):
        file_format = file_format.replace('.', '')
        out_format_func = f"to_{file_format}"

        filepath = f"{filepath}.{file_format}"

        df_func = getattr(self.as_dataframe(), out_format_func)
        if df_func is not None:
            df_func(filepath, **kwargs)
            RcLogger.get().debug(f"Saved relevance matrix as dataframe at path '{filepath}'")
        else:
            msg = f"pandas does not support {file_format} files for saving"
            raise RcLoggerException(ValueError, msg)

    def _save_relevance_matrix_numpy(self, filepath, **kwargs):
        style = kwargs.pop("numpy_style")
        if style is None:
            msg = "_save_relevance_matrix_numpy needs the style parameter not to be None"
            raise RcLoggerException(NotImplementedError, msg)

        if style == "arrays":
            os.mkdir(f"{filepath}_{style}")
            filepath = os.path.join(filepath, '')  # in order to save each array in that path by just concatenating name
            names = self._as_numpy_arrays_names
        elif style == "matrix":
            names = [""]
        else:
            msg = f"_save_relevance_matrix_numpy does not support style = {style}"
            raise RcLoggerException(NotImplementedError, msg)

        for arr, name in zip(self.as_numpy(style=style), names):
            np.save(f"{filepath}{name}", arr, **kwargs)

        RcLogger.get().debug(f"Saved relevance matrix as numpy at path '{filepath}'")

    @staticmethod
    def load(filepath, **kwargs):
        RcLogger.get().debug(f"Loading relevance_matrix at path '{filepath}'")

        if not os.path.isdir(filepath):
            _, ext = os.path.splitext(filepath)
            pandas_out_func = getattr(pd, f"read_{ext.replace('.', '')}")

            if pandas_out_func is not None:
                kwargs['index_col'] = 0
                rm_df = pandas_out_func(filepath, **kwargs)

                # Generally bytestrings are used as user ids and item ids, but they are read as strings by pandas
                re_pattern = "b\'.*\'"
                if re.match(re_pattern, rm_df.index[0]) is not None and \
                        re.match(re_pattern, rm_df.columns[0]) is not None:
                    rm_df = gen_utils.convert_dataframe_str_to_bytestr_cols_index(rm_df)

                relevances = []
                for user in rm_df.index:
                    relevances.append(rm_df.loc[user].to_numpy())
                params = [
                    rm_df.index.to_numpy(),
                    relevances,
                    rm_df.columns.to_numpy()
                ]
            else:
                msg = f"pandas does not support {ext} files for loading"
                raise RcLoggerException(ValueError, msg)
        else:
            files = [f for f in os.listdir(filepath)
                     for s in RelevanceMatrix._as_numpy_arrays_names if s in f]
            files = files if bool.__and__(*[s in f for f, s in zip(files,
                                                                   RelevanceMatrix._as_numpy_arrays_names)]) \
                else files[::-1]
            params = [np.load(f) for f in files]

            params = [params[0], params[1][:, :, 0], params[1][:, :, 1]]

        return RelevanceMatrix(*params)

    def to_user_oriented_fairness_files(self,
                                        sensitive_attribute: Union[Literal["gender"], Literal["age"], str],
                                        sensitive_dict: Dict[Text, Any],
                                        model_name: str,
                                        unobserved_items: Dict[Text, Any],
                                        observed_items: Dict[Text, Any],
                                        test_data: tf.data.Dataset,
                                        user_field: str,
                                        item_field: str,
                                        folderpath=None
                                        ):
        RcLogger.get().info("Generating files to be used on reranking `User-oriented fairness in recommendation`")
        
        if folderpath is None:
            folderpath = constants.SAVE_USER_ORIENTED_FAIRNESS_PATH

        sensitive_values = np.unique(list(sensitive_dict.values()))

        if sensitive_values.shape[0] != 2:
            msg = "Only binary attribute can be used for reranking `User-oriented fairness in recommendation`"
            raise RcLoggerException(ValueError, msg)

        RcLogger.get().info(f"Mapping `{sensitive_values[0]}` to 0 and `{sensitive_values[1]} to 1`")

        sensitive_dict = {u: 1 if sens == sensitive_values[1] else 0 for u, sens in sensitive_dict.items()}

        df = self.as_dataframe(style="columns")
        df = df.rename(columns={'user_id': 'uid', 'item_id': 'iid', 'relevances': 'score'})
        min_score = df["score"].min() - 1

        df["label"] = df.apply(lambda series: 1 if series['iid'] in unobserved_items[series['uid']] else 0, axis=1)
        df["score"] = df.apply(lambda series: min_score if series['iid'] in observed_items[series['uid']] else series["score"], axis=1)
        df = df.astype({'uid': 'int32', 'iid': 'int32'})

        df.to_csv(os.path.join(
            folderpath,
            f"{model_name}_{sensitive_attribute}_{file_utils.current_run_id()}_rank.csv"),
            index=False,
            sep='\t'
        )

        import tensorflow_datasets as tfds

        test_df = tfds.as_dataframe(test_data)
        test_df = test_df[[user_field, item_field]]
        test_df = test_df.rename(columns={user_field: 'uid', item_field: 'iid'})

        test_df["label"] = np.ones_like(test_df.index)

        test_sensitive = dict.fromkeys(sensitive_values)

        test_df[sensitive_attribute] = test_df['uid'].map(sensitive_dict)

        for gr_name, gr_df in test_df.groupby(sensitive_attribute):
            key = sensitive_values[1] if gr_name == 1 else sensitive_values[0]
            test_sensitive[key] = gr_df[["uid", "iid", "label"]]

        for sens_value, group_df in test_sensitive.items():
            group_df = group_df.astype({'uid': 'int32', 'iid': 'int32'})

            out_group_df_sens = sens_value

            group_df.to_csv(os.path.join(
                folderpath,
                f"{model_name}_{out_group_df_sens}_{file_utils.current_run_id()}_test_ratings.csv"),
                index=False,
                sep='\t'
            )

    @classmethod
    def __from_partial_pivot(cls,
                             df,
                             unique_users=None,
                             unique_items=None,
                             add_missin_items=False,
                             add_missing_users=False,
                             **kwargs):
        df: pd.DataFrame

        if add_missin_items:
            df_items = set(df.columns.to_list())
            unique_items = [int(x) for x in unique_items]  # if int stays int, if str or bytes is mapped to int

            df[list(set(unique_items) ^ df_items)] = np.nan

            # set values of non present items with the means of each row
            row_means = df.mean(axis=1, skipna=True)
            for row, value in row_means.items():
                df.loc[row].fillna(value=value, inplace=True)

        if add_missing_users:
            df_users = set(df.index.to_list())
            unique_users = [int(x) for x in unique_users]  # if int stays int, if str or bytes is mapped to int

            # set rows of non present users with random values
            df.loc[list(set(unique_users) ^ df_users)] = np.random.rand(len(df.columns))

        df.columns = [str(x).encode() for x in df.columns]
        df.index = [str(x).encode() for x in df.index]

        return cls(df.index, df.values, df.columns, **kwargs)

    @classmethod
    def from_user_oriented_fairness_files(cls, filename, **kwargs):
        df = pd.read_csv(os.path.join(constants.SAVE_USER_ORIENTED_FAIRNESS_PATH, filename))

        df["q*s"] = df['score'] * df['q']

        df = df.pivot(index='uid', columns='iid', values='q*s')

        return cls.__from_partial_pivot(df, **kwargs)

    @classmethod
    def from_rec_indep_json(cls, filename, **kwargs):
        with open(filename, 'r') as file:
            data = json.load(file)

        event = np.array(data['prediction']['event'])
        users = event[:, 0]
        items = event[:, 1]

        pred = np.array(data['prediction']['predicted'])

        df = pd.DataFrame(zip(users, items, pred), columns=['user_id', 'item_id', 'pred'])

        df = df.pivot(index='user_id', columns='item_id', values='pred')

        return cls.__from_partial_pivot(df, **kwargs)

    @classmethod
    def from_co_clustering_fair_pickle(cls, results_pickle, extras_pickle, **kwargs):
        with open(results_pickle, 'rb') as res_pk:
            results = pickle.load(res_pk)

        with open(extras_pickle, 'rb') as extras_pk:
            # Should contain a dictionary that maps:
            # 'users_map': dictionary that maps real users ids to ids used by co_clustering (simple range of users)
            # 'items_map': dictionary that maps real items ids to ids used by co_clustering (simple range of items)
            extras = pickle.load(extras_pk)

        # co-clustering saves data in list by default to process multiple files. Here only one is supported
        results = results[0] if isinstance(results, Sequence) else results
        model = results["model"]

        predictions = (model["tau_1"] @ model["pi"] @ model["tau_2"].T + model["eta_row"] + model["eta_col"]) + 1

        inv_user_map = dict(zip(extras['users_map'].values(), extras['users_map'].keys()))
        inv_item_map = dict(zip(extras['items_map'].values(), extras['items_map'].keys()))

        if isinstance(list(inv_user_map.values())[0], bytes):
            users = [inv_user_map[idx] for idx in range(predictions.shape[0])]
            items = [inv_item_map[idx] for idx in range(predictions.shape[1])]
        else:
            users = [str(inv_user_map[idx]).encode() for idx in range(predictions.shape[0])]
            items = [str(inv_item_map[idx]).encode() for idx in range(predictions.shape[1])]

        # reorder by integer ordering (if mapping are created on bytestring corrects the order)
        predictions = predictions[np.argsort([int(x) for x in users]), :][:, np.argsort([int(x) for x in items])]

        users = sorted(users, key=int)
        items = sorted(items, key=int)

        return cls(users, predictions, items, **kwargs)

    @classmethod
    def from_librec_result(cls, results_file, **kwargs):
        df = pd.read_csv(results_file, header=None)

        df = df.pivot(index=0, columns=1, values=2)

        return cls.__from_partial_pivot(df, **kwargs)

    @classmethod
    def from_antidote_data(cls, filepath, **kwargs):
        df = pd.read_csv(filepath, index_col=0)

        df.index = sorted(df.index, key=int)
        df.columns = sorted(df.columns, key=int)

        return cls.__from_partial_pivot(df, **kwargs)

    @classmethod
    def from_nlr_models_result(cls, results_file, test_file, **kwargs):
        """
        https://github.com/rutgerswiselab/NLR
        :param results_file:
        :param test_file:
        :return:
        """
        predictions = np.load(results_file)
        test_df = pd.read_csv(test_file, sep='\t')

        test_df['label'] = predictions

        test_df = test_df.pivot(index='uid', columns='iid', values='label')

        return cls.__from_partial_pivot(test_df, **kwargs)

    @classmethod
    def from_fair_go_predictions(cls, predictions, testing_ratings_dict, mapping_user_item, **kwargs):
        with open(predictions, 'rb') as pk:
            predictions = pickle.load(pk)

        test_ratings_dict, _ = np.load(testing_ratings_dict, allow_pickle=True)
        user_map, item_map = np.load(mapping_user_item, allow_pickle=True)

        inv_user_map = dict(zip(list(user_map.values()), list(user_map.keys())))  # keys and values are unique
        inv_item_map = dict(zip(list(item_map.values()), list(item_map.keys())))  # keys and values are unique

        df = pd.DataFrame(list(test_ratings_dict.values()))

        df[0] = df[0].map(inv_user_map)
        df[2] = df[2].map(inv_item_map)
        df[1] = predictions

        df = df.pivot(index=0, columns=2, values=1)

        df = df[sorted(df.columns.to_list(), key=int)]
        df = df.loc[sorted(df.index.to_list(), key=int)]

        return cls(df.index, df.values, df.columns, **kwargs)

    @classmethod
    def from_cool_kids_result(cls, result_file, **kwargs):
        df = pd.read_csv(result_file, index_col=0)

        values_column = "Prediction" if "Prediction" in df else "Score"
        if values_column == "Score":
            df = df[df["Score"] > 0]

        df = df.pivot(index='User', columns='Item', values=values_column)

        return cls.__from_partial_pivot(df, **kwargs)

    @classmethod
    def from_rating_prediction_fairness_result(cls, filepath, **kwargs):
        df = pd.read_csv(filepath)

        df = df[['user', 'item', 'prediction']]

        df = df.pivot(index='user', columns='item', values='prediction')

        return cls.__from_partial_pivot(df, **kwargs)


class EarlyStoppingWithLimit(tf.keras.callbacks.Callback):

    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 limit=None,
                 restore_best_weights=False):
        super(EarlyStoppingWithLimit, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.limit = limit
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.best = None

        if mode not in ['auto', 'min', 'max']:
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if self.restore_best_weights and self.best_weights is None:
            # Restore the weights after first epoch if no progress is ever made.
            self.best_weights = self.model.get_weights()

        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            # Only restart wait if we beat both the baseline and our previous best.
            if self.baseline is None or self._is_improvement(current, self.baseline):
                self.wait = 0

        if self.wait >= self.patience or (self.limit is not None and self._is_improvement(current, self.limit)):
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose > 0:
                    print('Restoring model weights from the end of the best epoch.')
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)

        return monitor_value

    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value - self.min_delta, reference_value)


class GC(tf.keras.optimizers.Optimizer):

    def __init__(self,
                 loss_f,
                 grad_f,
                 x0,
                 args,
                 name="GC",
                 **kwargs):
        super(GC, self).__init__(name=name, **kwargs)

        sf = _prepare_scalar_function(loss_f, x0, jac=grad_f, args=args, epsilon=_epsilon, finite_diff_rel_step=None)

        self._loss_f = sf.fun
        self._grad_f = sf.grad

        self.set_hyper("norm", np.Inf)
        self.set_hyper("gtol", 1e-5)

    def _create_slots(self, var_list):
        pass

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(GC, self)._prepare_local(var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)]["norm"] = array_ops.identity(
            self._get_hyper("norm", var_dtype))
        apply_state[(var_device, var_dtype)]["gtol"] = array_ops.identity(
            self._get_hyper("gtol", var_dtype))

    def _resource_apply_dense(self, grad, handle, apply_state):
        var_device, var_dtype = handle.device, handle.dtype.base_dtype

        norm = apply_state[(var_device, var_dtype)]['norm']
        gtol = apply_state[(var_device, var_dtype)]['gtol']

        f = self._loss_f
        myfprime = self._grad_f

        old_fval = f(handle)
        gfk = grad

        if not np.isscalar(old_fval):
            try:
                old_fval = old_fval.item()
            except (ValueError, AttributeError) as e:
                raise ValueError("The user-provided "
                                 "objective function must "
                                 "return a scalar value.") from e

        xk = handle
        # Sets the initial step guess to dx ~ 1
        old_old_fval = old_fval + np.linalg.norm(gfk) / 2

        pk = -gfk
        gnorm = vecnorm(gfk, ord=norm)

        sigma_3 = 0.01

        if gnorm < gtol:
            raise Exception()

        deltak = np.dot(gfk, gfk)

        cached_step = [None]

        def polak_ribiere_powell_step(alpha, gfkp1=None):
            xkp1 = xk + alpha * pk
            if gfkp1 is None:
                gfkp1 = myfprime(xkp1)
            yk = gfkp1 - gfk
            beta_k = max(0, np.dot(yk, gfkp1) / deltak)
            pkp1 = -gfkp1 + beta_k * pk
            gnorm = vecnorm(gfkp1, ord=norm)
            return alpha, xkp1, pkp1, gfkp1, gnorm

        def descent_condition(alpha, xkp1, fp1, gfkp1):
            # Polak-Ribiere+ needs an explicit check of a sufficient
            # descent condition, which is not guaranteed by strong Wolfe.
            #
            # See Gilbert & Nocedal, "Global convergence properties of
            # conjugate gradient methods for optimization",
            # SIAM J. Optimization 2, 21 (1992).
            cached_step[:] = polak_ribiere_powell_step(alpha, gfkp1)
            alpha, xk, pk, gfk, gnorm = cached_step

            # Accept step if it leads to convergence.
            if gnorm <= gtol:
                return True

            # Accept step if sufficient descent condition applies.
            return np.dot(pk, gfk) <= -sigma_3 * np.dot(gfk, gfk)

        try:
            alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                _line_search_wolfe12(f, myfprime, xk, pk, gfk, old_fval,
                                     old_old_fval, c2=0.4, amin=1e-100, amax=1e100,
                                     extra_condition=descent_condition)
        except _LineSearchError:
            # Line search failed to find a better solution.
            warnflag = 2

        # Reuse already computed results if possible
        if alpha_k == cached_step[0]:
            alpha_k, xk, pk, gfk, gnorm = cached_step
        else:
            alpha_k, xk, pk, gfk, gnorm = polak_ribiere_powell_step(alpha_k, gfkp1)

        var_update = state_ops.assign(gfk)

        return control_flow_ops.group(*[var_update])

    def _resource_apply_sparse(self, grad, handle, indices, apply_state):
        raise NotImplementedError("CG should not work on sparse tenors")

    def get_config(self):
        config = super(GC, self).get_config()
        config.update({
            "norm": self._serialize_hyperparameter("norm"),
            "gtol": self._serialize_hyperparameter("gtol"),
        })
        return config


def model_save(save_function, filepath, save_format="tf", **kwargs):
    save_format = save_format.replace('.', '')

    try:
        if save_format == "tf":
            os.mkdir(filepath)
            filepath = os.path.join(filepath, '')
        else:
            filepath += '.h5' if '.h5' not in filepath else ''
        save_function(filepath, save_format=save_format, **kwargs)
    except Exception as e:
        if save_format == "tf":
            os.rmdir(filepath)
        msg = f"Impossible to save model at path '{filepath}', got Exception '{e}'"
        raise RcLoggerException(Exception, msg)


def get_correlation_loss(y_true, y_pred):
    keras_sum, keras_square = tf.keras.backend.sum, tf.keras.backend.square

    x = y_true
    y = y_pred
    mx = tf.keras.backend.mean(x)
    my = tf.keras.backend.mean(y)
    xm, ym = x-mx, y-my

    r_num = tf.keras.backend.sum(tf.multiply(xm, ym))
    r_den = tf.keras.backend.sqrt(tf.multiply(keras_sum(keras_square(xm)), keras_sum(keras_square(ym))))

    r = r_num / tf.where(tf.equal(r_den, 0), 1e-3, r_den)
    r = tf.keras.backend.abs(tf.keras.backend.maximum(tf.keras.backend.minimum(r, 1.0), -1.0))

    return tf.keras.backend.square(r)


def get_dot_difference(parameter_matrix_list):
    user_embedding_matrix, item_positive_embedding_matrix, item_negative_embedding_matrix = parameter_matrix_list
    positive_batch_dot = tf.keras.backend.batch_dot(user_embedding_matrix, item_positive_embedding_matrix, axes=1)
    if item_negative_embedding_matrix is not None:
        negative_batch_dot = tf.keras.backend.batch_dot(user_embedding_matrix, item_negative_embedding_matrix, axes=1)

        return positive_batch_dot - negative_batch_dot
    else:
        return positive_batch_dot


def get_dot_difference_shape(shape_vector_list):
    user_embedding_shape_vector, _, _ = shape_vector_list
    return user_embedding_shape_vector[0], 1
