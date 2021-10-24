import os
import inspect
import time
from typing import (
    Dict,
    Sequence,
    Union,
    Set,
    Any,
    Iterable,
    Tuple,
    Literal
)
from collections import defaultdict
import itertools

import scipy.stats
import pandas as pd
import numpy as np
import sklearn.metrics as sk_metrics
import matplotlib.pyplot as plt
import tqdm

import helpers.constants as constants
import helpers.filename_utils as file_utils
import metrics.utils as metric_utils
import helpers.general_utils as general_utils
from helpers.logger import RcLogger, RcLoggerException


class Metrics(object):
    _compute_prefix = '_compute_'
    _methods_get_prefix = '_get_'
    _metric_k_format = "{metric}/top_{k}"

    def __init__(self, model_metric=None, metrics=None, model_name=None, **kwargs):
        log_model_metric = f" with model_metric '{model_metric.name}'" if model_metric is not None else ""
        RcLogger.get().info(f"Initializing Metrics class{log_model_metric}")

        self._model_metric = model_metric
        self._model_metrics = metrics if metrics is not None else {}
        self._custom_metrics = defaultdict(dict)
        self._custom_dependant_metrics = defaultdict(dict)
        self._custom_fairness_metrics = defaultdict(dict)
        self.individual_custom_metrics = defaultdict(lambda: defaultdict(dict))
        self._cutoffs = [1, 5, 10, 50, 100] if kwargs.get("cutoffs") is None else kwargs.pop("cutoffs")
        self._model_name = model_name
        self.plot = PlotAccessor(
            self._model_metrics,
            self._custom_metrics,
            self.individual_custom_metrics,
            self._custom_fairness_metrics,
            self._custom_dependant_metrics
        )

    def __call__(self, *args, **kwargs):
        return self.model_metric(*args, **kwargs)

    def save(self, filename=None, output_type="csv"):
        RcLogger.get().info(f"Saving metrics with format '{output_type}'")

        metrics = self.metrics

        model_metric_name = self.model_metric.name + '-' if self.model_metric is not None else ''
        model_name = f"-{self._model_name}" if self._model_name is not None else ''

        if filename is None:
            filename = f"metrics{model_name}-{model_metric_name}{file_utils.default_filepath()}"
        else:
            filename = f"{filename}__{file_utils.current_run_id()}"

        filename = os.path.join(constants.SAVE_METRICS_PATH, filename)

        if output_type == "csv":
            pd.DataFrame(metrics, index=[0]).to_csv(f"{filename}.{output_type}")
        elif output_type == "npy":
            metrics = np.array(list(metrics.items()))
            np.save(filename, metrics)
        else:
            msg = "Custom metrics saving only supports csv (pandas) and npy (numpy) formats"
            raise RcLoggerException(NotImplementedError, msg)

        RcLogger.get().info(f"metrics saved at path '{filename}'")

    @property
    def metrics(self):
        custom_metrics = self._custom_metrics

        metrics = {
            **self._model_metrics,
            **{Metrics._metric_k_format.format(metric=_metric, k=cut): v for _metric in custom_metrics
               for cut, v in custom_metrics[_metric].items()}
        }

        return metrics

    def set_model_metrics(self, metrics: Dict[str, float]):
        self._model_metrics = metrics
        self.plot._model_metrics = metrics

    def compute_metrics(self,
                        metrics: Union[Sequence[str], str, Literal["all"]] = "all",
                        cutoffs: Sequence[int] = None,
                        only: Sequence[str] = None,
                        **kwargs) -> "Metrics":
        cutoffs = self._cutoffs if cutoffs is None else cutoffs
        RcLogger.get().info(f"Retrieving the metrics: \"{metrics}\" with the topK: {cutoffs}")

        start = time.perf_counter()

        only = ["custom", "model"] if only is None else only

        overwrite = kwargs.pop("overwrite") if "overwrite" in kwargs else False

        for metric_type in only:
            attr = f'_{metric_type}_metrics'
            if not hasattr(self, attr):
                msg = f"'{metric_type}' is not a metric type supported by this class, " \
                      f"expected one of ['custom', 'model']"
                raise RcLoggerException(ValueError, msg)

            curr_metrics = getattr(self, attr)
            if metric_type == "model":
                if curr_metrics is None:
                    msg = "evaluate method of the model should be called to get model metrics"
                    raise RcLoggerException(ReferenceError, msg)

                # no code is necessary here because curr_metrics contain model metrics, but this structure is necessary
                # to avoid to call the code in 'else' branch
                pass
            elif metric_type == "custom":
                self._compute_custom_metrics(metrics=metrics, cutoffs=cutoffs, overwrite=overwrite, **kwargs)
            else:
                # This code should never be reachable if all the metrics types are handled by the class
                msg = f"'{metric_type}' is a type handled with an attribute but it is not handled in 'get_metrics'"
                raise RcLoggerException(NotImplementedError, msg)

        RcLogger.get().info(f"Custom metrics retrieving/computation duration: "
                            f"{time.strftime('%H:%M:%S', time.gmtime(time.perf_counter() - start))} seconds")

        return self

    def _compute_custom_metrics(self,
                                metrics: Union[Sequence[str], str],
                                cutoffs: Sequence[int],
                                overwrite=False,
                                **kwargs):
        RcLogger.get().info("Computing the metrics")

        metrics = metrics if metrics != "all" else self.supported_custom_metrics()
        out_metrics = self._custom_metrics

        metrics_to_compute = set(metrics) - set(self.metrics_dependant)
        metrics_to_compute = metrics_to_compute - set(out_metrics) if not overwrite else metrics_to_compute

        metrics_dependencies = self._dependant_metrics_dependecies
        for metric in metrics:
            if metric in metrics_dependencies:
                for dep in metrics_dependencies[metric]:
                    metrics_to_compute.add(dep)

        predictions = kwargs.get("predictions")
        observed_items = kwargs.get("observed_items")
        if metrics_to_compute:
            for user_id in tqdm.tqdm(predictions.index, desc="Computing custom metrics"):
                user_unobserved_top_items = predictions.loc[user_id][
                    ~predictions.loc[user_id].index.isin(observed_items[user_id])
                ]
                user_unobserved_top_items.sort_values(ascending=False, inplace=True)

                for cut in cutoffs:
                    for _metric in metrics_to_compute:
                        if _metric not in out_metrics and cut not in out_metrics.get(_metric, {}) or overwrite:
                            metric_compute_func = f'{Metrics._compute_prefix}{_metric}'
                            if not hasattr(self, metric_compute_func):
                                msg = f"cannot compute '{_metric}' because it is not a supported metric"
                                raise RcLoggerException(NotImplementedError, msg)

                            _compute_kwargs = {**kwargs, "user_unobserved_top_items": user_unobserved_top_items}

                            user_metric_value = getattr(self, metric_compute_func)(user_id, cut, **_compute_kwargs)
                            self.individual_custom_metrics[_metric][user_id][cut] = user_metric_value

        dependant_metrics = (set(self.metrics_dependant) & set(metrics)) - set(self._custom_metrics)

        for cut in cutoffs:
            for _metric in tqdm.tqdm(itertools.chain(metrics_to_compute, dependant_metrics),
                                     desc="Computing custom dependant/fairness metrics"):
                # metrics dependant on other metrics are computed before the get method
                if _metric in self.metrics_dependant:
                    if cut not in out_metrics.get(_metric, {}) or overwrite:
                        dep_metric_compute_func = f'{Metrics._compute_prefix}{_metric}'
                        if not hasattr(self, dep_metric_compute_func):
                            msg = f"cannot compute '{_metric}' because it is not a supported metric"
                            raise RcLoggerException(NotImplementedError, msg)

                        metric_value = getattr(self, dep_metric_compute_func)(cut, **kwargs)
                        if _metric not in self.fairness_metrics:
                            self._custom_dependant_metrics[_metric][cut] = metric_value
                        else:
                            self._custom_fairness_metrics[_metric][cut] = metric_value

                metric_get_func = f'{Metrics._methods_get_prefix}{_metric}'
                if not hasattr(self, metric_get_func):
                    msg = f"cannot get '{_metric}' because it is not a supported metric"
                    raise RcLoggerException(NotImplementedError, msg)

                if cut not in out_metrics.get(_metric, {}) or overwrite:
                    out_metrics[_metric][cut] = getattr(self, metric_get_func)(cut, **kwargs)

    def get(self, metric, k=10, user_id=None, roundness=4, raw=False):
        metric_get_func = f'{Metrics._methods_get_prefix}{metric}'
        if not hasattr(self, metric_get_func):
            msg = f"cannot get '{metric}' because it is not a supported metric"
            raise RcLoggerException(NotImplementedError, msg)

        return getattr(self, metric_get_func)(k, user_id=user_id, roundness=roundness, raw=raw)

    def __get_with_user_ids(self, metric, k, user_id=None, roundness=4, raw=False, **kwargs):
        user_id = [user_id] if not isinstance(user_id, Iterable) and user_id is not None else user_id

        result = [
            m[k] for u, m in self.individual_custom_metrics[metric].items()
            if np.isfinite(m[k]) and (u in user_id if user_id is not None else True)
        ]

        if not raw:
            result = round(np.mean(result), roundness)

        return result

    def _compute_tp(self, user_id, k, **kwargs):
        # type hints
        unobserved_items: Dict[str, Set[str]]
        user_unobserved_top_items: pd.Series

        unobserved_items = kwargs.pop("unobserved_items")

        user_unobserved_items = unobserved_items[user_id]
        user_unobserved_top_items = kwargs.pop("user_unobserved_top_items")
        user_unobserved_top_k = user_unobserved_top_items[:k]

        if len(user_unobserved_top_k) < k:
            Metrics._metric_cutoff_lower_than_topk_warning(
                inspect.currentframe().f_code.co_name.replace('_compute_', ''),
                len(user_unobserved_top_k),
                k
            )

        tp = len(user_unobserved_top_k.index & user_unobserved_items)
        self.individual_custom_metrics['tp'][user_id][k] = tp

        return tp

    def _get_tp(self, k, user_id=None, **kwargs):
        return self.__get_with_user_ids("tp", k, user_id=user_id, **kwargs)

    def _compute_tn(self, user_id, k, **kwargs):
        # type hints
        predictions: pd.DataFrame
        observed_items: Dict[str, Set[str]]
        unobserved_items: Dict[str, Set[str]]

        predictions = kwargs.pop("predictions")
        observed_items = kwargs.pop("observed_items")
        unobserved_items = kwargs.get("unobserved_items")

        relevant_items = self._compute_tp(user_id, k, **kwargs)

        items = predictions.columns
        user_non_interactions_items = set(items) - observed_items[user_id] - unobserved_items[user_id]

        tn = len(user_non_interactions_items) - (k - relevant_items)

        self.individual_custom_metrics['tn'][user_id][k] = tn

        return tn

    def _get_tn(self, k, user_id=None, **kwargs):
        return self.__get_with_user_ids("tn", k, user_id=user_id, **kwargs)

    def _compute_fp(self, user_id, k, **kwargs):
        relevant_items = self._compute_tp(user_id, k, **kwargs)

        fp = k - relevant_items

        self.individual_custom_metrics['fp'][user_id][k] = fp

        return fp

    def _get_fp(self, k, user_id=None, **kwargs):
        return self.__get_with_user_ids("fp", k, user_id=user_id, **kwargs)

    def _compute_fn(self, user_id, k, **kwargs):
        unobserved_items: Dict[str, Set[str]]

        unobserved_items = kwargs.get("unobserved_items")

        relevant_items = self._compute_tp(user_id, k, **kwargs)

        fn = len(unobserved_items[user_id]) - relevant_items

        self.individual_custom_metrics['fn'][user_id][k] = fn

        return fn

    def _get_fn(self, k, user_id=None, **kwargs):
        return self.__get_with_user_ids("fn", k, user_id=user_id, **kwargs)

    def _compute_precision(self, user_id, k, **kwargs):
        RcLogger.get().debug(f"Computing precision of user '{user_id}' with k = {k}")
        return self._compute_tp(user_id, k, **kwargs) / k

    def _get_precision(self, k, user_id=None, **kwargs):
        return self.__get_with_user_ids("precision", k, user_id=user_id, **kwargs)

    def _compute_recall(self, user_id, k, **kwargs):
        RcLogger.get().debug(f"Computing recall of user '{user_id}' with k = {k}")
        # type hints
        unobserved_items: Dict[str, Set[str]]

        unobserved_items = kwargs.get("unobserved_items")

        user_unobserved_items = unobserved_items[user_id]
        return self._compute_tp(user_id, k, **kwargs) / len(user_unobserved_items) \
               if len(user_unobserved_items) > 0 else 1

    def _get_recall(self, k, user_id=None, **kwargs):
        return self.__get_with_user_ids("recall", k, user_id=user_id, **kwargs)

    def _compute_f1_score(self, user_id, k, **kwargs):
        RcLogger.get().debug(f"Computing F1 Score of user '{user_id}' with k = {k}")
        precision = self._compute_precision(user_id, k, **kwargs)
        recall = self._compute_recall(user_id, k, **kwargs)

        return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    def _get_f1_score(self, k, user_id=None, **kwargs):
        return self.__get_with_user_ids("f1_score", k, user_id=user_id, **kwargs)

    def _compute_tpr(self, user_id, k, **kwargs):
        RcLogger.get().debug(f"Computing TPR of user '{user_id}' with k = {k}")
        tp = self._compute_tp(user_id, k, **kwargs)
        fn = self._compute_fn(user_id, k, **kwargs)

        # return self._get_recall(k, **kwargs)
        return round(tp / (tp + fn), 4) if tp + fn > 0 else 0

    def _get_tpr(self, k, user_id=None, **kwargs):
        return self.__get_with_user_ids("tpr", k, user_id=user_id, **kwargs)

    def _compute_tnr(self, user_id, k, **kwargs):
        RcLogger.get().debug(f"Computing TNR of user '{user_id}' with k = {k}")
        tn = self._compute_tn(user_id, k, **kwargs)
        fp = self._compute_fp(user_id, k, **kwargs)

        return round(tn / (tn + fp), 4) if tn + fp > 0 else 0

    def _get_tnr(self, k, user_id=None, **kwargs):
        return self.__get_with_user_ids("tnr", k, user_id=user_id, **kwargs)

    def _compute_fpr(self, user_id, k, **kwargs):
        RcLogger.get().debug(f"Computing FPR of user '{user_id}' with k = {k}")
        return 1 - self._compute_tnr(user_id, k, **kwargs)

    def _get_fpr(self, k, user_id=None, **kwargs):
        return self.__get_with_user_ids("fpr", k, user_id=user_id, **kwargs)

    def _compute_fnr(self, user_id, k, **kwargs):
        RcLogger.get().debug(f"Computing FNR of user '{user_id}' with k = {k}")
        return 1 - self._compute_tpr(user_id, k, **kwargs)

    def _get_fnr(self, k, user_id=None, **kwargs):
        return self.__get_with_user_ids("fnr", k, user_id=user_id, **kwargs)

    def _compute_predicted_positive(self, k, user_id: Union[int, str, Iterable], **kwargs):
        pp = self._compute_tp(user_id, k, **kwargs) + self._compute_fp(user_id, k, **kwargs)

        self.individual_custom_metrics['predicted_positive'][user_id][k] = pp

        return pp

    def _get_predicted_positive(self, k, user_id=None, **kwargs):
        return self.__get_with_user_ids("predicted_positive", k, user_id=user_id, **kwargs)

    @staticmethod
    def _compute_mrr(user_id, k, **kwargs):
        RcLogger.get().debug(f"Computing MRR of user '{user_id}' with k = {k}")
        # type hints
        # type hints
        unobserved_items: Dict[str, Set[str]]
        user_unobserved_top_items: pd.Series

        unobserved_items = kwargs.pop("unobserved_items")

        user_unobserved_items = unobserved_items[user_id]
        user_unobserved_top_items = kwargs.pop("user_unobserved_top_items")
        user_unobserved_top_k = user_unobserved_top_items[:k]

        if len(user_unobserved_top_k) < k:
            Metrics._metric_cutoff_lower_than_topk_warning(
                inspect.currentframe().f_code.co_name.replace('_compute_', ''),
                len(user_unobserved_top_k),
                k
            )

        mrr = 0
        for rank, (top_item, _) in enumerate(user_unobserved_top_k.items()):
            if top_item in user_unobserved_items:
                mrr += 1 / (rank + 1)

        return mrr / k

    def _get_mrr(self, k, user_id=None, **kwargs):
        return self.__get_with_user_ids("mrr", k, user_id=user_id, **kwargs)

    @staticmethod
    def _compute_ndcg(user_id, k, **kwargs):
        RcLogger.get().debug(f"Computing nDCG of user '{user_id}' with k = {k}")
        # type hints
        unobserved_items: Dict[str, Set[str]]
        observed_items: Dict[str, Set[str]]
        predictions: pd.DataFrame

        unobserved_items = kwargs.pop("unobserved_items")
        observed_items = kwargs.pop("observed_items")
        predictions = kwargs.pop("predictions")

        user_obs_items = observed_items[user_id] & set(predictions.columns)
        user_unobs_items = unobserved_items[user_id] & set(predictions.columns)

        y_pred = predictions.loc[user_id].to_numpy()
        y_pred[[predictions.columns.get_loc(item) for item in user_obs_items]] = np.nanmin(y_pred) - 1
        y_true = np.zeros(len(predictions.columns))
        y_true[[predictions.columns.get_loc(item) for item in user_unobs_items]] = 1.0

        pred_mask = ~np.isnan(y_pred)

        y_pred = y_pred[pred_mask]
        y_true = y_true[pred_mask]

        return sk_metrics.ndcg_score([y_true], [y_pred], k=k)

    def _get_ndcg(self, k, user_id=None, **kwargs):
        return self.__get_with_user_ids("ndcg", k, user_id=user_id, **kwargs)

    @staticmethod
    def _compute_ndcg_user_oriented_fairness(user_id, k, **kwargs):
        def dcg_at_k(r, k, method=0):
            """Score is discounted cumulative gain (dcg)
            Relevance is positive real values.  Can use binary
            as the previous methods.
            Example from
            http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
            >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
            >>> dcg_at_k(r, 1)
            3.0
            >>> dcg_at_k(r, 1, method=1)
            3.0
            >>> dcg_at_k(r, 2)
            5.0
            >>> dcg_at_k(r, 2, method=1)
            4.2618595071429155
            >>> dcg_at_k(r, 10)
            9.6051177391888114
            >>> dcg_at_k(r, 11)
            9.6051177391888114
            Args:
                r: Relevance scores (list or numpy) in rank order
                    (first element is the first item)
                k: Number of results to consider
                method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                        If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
            Returns:
                Discounted cumulative gain
            """
            r = np.asfarray(r)[:k]
            if r.size:
                if method == 0:
                    return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
                elif method == 1:
                    return np.sum(r / np.log2(np.arange(2, r.size + 2)))
                else:
                    raise ValueError('method must be 0 or 1.')
            return 0.

        RcLogger.get().debug(f"Computing nDCG-UOF of user '{user_id}' with k = {k}")
        # type hints
        unobserved_items: Dict[str, Set[str]]
        observed_items: Dict[str, Set[str]]
        predictions: pd.DataFrame

        unobserved_items = kwargs.pop("unobserved_items")
        observed_items = kwargs.pop("observed_items")
        predictions = kwargs.pop("predictions")
        method = kwargs.get("method", 1)

        user_obs_items = observed_items[user_id] & set(predictions.columns)
        user_unobs_items = unobserved_items[user_id] & set(predictions.columns)

        y_pred = predictions.loc[user_id].to_numpy()
        y_pred[[predictions.columns.get_loc(item) for item in user_obs_items]] = np.nanmin(y_pred) - 1
        y_true = np.zeros(len(predictions.columns))
        y_true[[predictions.columns.get_loc(item) for item in user_unobs_items]] = 1.0

        pred_mask = ~np.isnan(y_pred)

        y_pred = y_pred[pred_mask]
        y_true = y_true[pred_mask]

        sorted_y_true = y_true[np.argsort(y_pred)[::-1]][:k]

        dcg_max = dcg_at_k(sorted(sorted_y_true, reverse=True), k, method)
        if not dcg_max:
            return 0.
        return dcg_at_k(sorted_y_true, k, method) / dcg_max

    def _get_ndcg_user_oriented_fairness(self, k, user_id=None, **kwargs):
        return self.__get_with_user_ids("ndcg_user_oriented_fairness", k, user_id=user_id, **kwargs)

    @staticmethod
    def _compute_rmse(user_id, k, **kwargs):
        RcLogger.get().debug(f"Computing RMSE of user '{user_id}' with k = {k}")

        predictions: pd.DataFrame
        test_rating_df: pd.DataFrame

        predictions = kwargs.pop("predictions")
        test_rating_df = kwargs.pop("test_rating_dataframe")

        user_pred = predictions.loc[user_id].values
        user_true = test_rating_df.loc[user_id].values

        test_ratings_idxs = ~pd.isnull(user_true)

        y_pred = user_pred[test_ratings_idxs]
        y_true = user_true[test_ratings_idxs]

        if len(y_true) > 0 and len(y_pred) > 0:
            return sk_metrics.mean_squared_error(y_true, y_pred, squared=False)
        else:
            return 0

    def _get_rmse(self, k, user_id=None, **kwargs):
        return self.__get_with_user_ids("rmse", k, user_id=user_id, **kwargs)

    @staticmethod
    def _compute_mae(user_id, k, **kwargs):
        RcLogger.get().debug(f"Computing MAE of user '{user_id}' with k = {k}")

        predictions: pd.DataFrame
        test_rating_df: pd.DataFrame

        predictions = kwargs.pop("predictions")
        test_rating_df = kwargs.pop("test_rating_dataframe")

        user_pred = predictions.loc[user_id].values
        user_true = test_rating_df.loc[user_id].values

        test_ratings_idxs = ~pd.isnull(user_true)

        y_pred = user_pred[test_ratings_idxs]
        y_true = user_true[test_ratings_idxs]

        if len(y_true) > 0 and len(y_pred) > 0:
            return sk_metrics.mean_absolute_error(y_true, y_pred)
        else:
            return 0

    def _get_mae(self, k, user_id=None, **kwargs):
        return self.__get_with_user_ids("mae", k, user_id=user_id, **kwargs)

    @staticmethod
    def _compute_category_calibration(user_id, k, **kwargs):
        #RcLogger.get().warning("Category Calibration has not been implemented yet, "
        #               "cannot be computed, no category_per_item in datasets")
        pass

    def _get_category_calibration(self, k, **kwargs):
        #RcLogger.get().warning("Category Calibration has not been implemented yet, cannot get it, "
        #               "no category_per_item in datasets")
        pass

    @staticmethod
    def _compute_mean_popularity(user_id, k, **kwargs):
        RcLogger.get().debug(f"Computing mean popularity of user '{user_id}' with k = {k}")
        # type hints
        item_popularity: Dict[str, int]
        len_observed_data: int
        unobserved_top_k: pd.DataFrame

        item_popularity = kwargs.pop("item_popularity")
        len_observed_data = kwargs.pop("len_observed_data")

        user_unobserved_top_items = kwargs.pop("user_unobserved_top_items")
        user_unobserved_top_k = user_unobserved_top_items[:k]

        top_k_items_popularity = user_unobserved_top_k.index & set(item_popularity)

        top_k_popularity = np.array([item_popularity[item] for item in top_k_items_popularity])

        return np.mean(top_k_popularity * len_observed_data)

    def _get_mean_popularity(self, k, user_id=None, **kwargs):
        return self.__get_with_user_ids("mean_popularity", k, user_id=user_id, **kwargs)

    @staticmethod
    def _compute_diversity(user_id, k, **kwargs):
        #RcLogger.get().warning("Diversity has not been implemented yet, cannot be computed, no category_per_item in datasets")
        pass

    def _get_diversity(self, k, **kwargs):
        #RcLogger.get().warning("Diversity has not been implemented yet, cannot get it, no category_per_item in datasets")
        pass

    @staticmethod
    def _compute_novelty(user_id, k, **kwargs):
        RcLogger.get().debug(f"Computing novelty of user '{user_id}' with k = {k}")
        # type hints
        item_popularity: Dict[str, int]
        users: np.ndarray
        unobserved_top_k: pd.DataFrame
        unobserved_items: Dict[str, Set[str]]

        item_popularity = kwargs.pop("item_popularity")
        len_observed_data = kwargs.pop("len_observed_data")
        predictions = kwargs.pop("predictions")
        n_users = len(predictions.index)

        user_unobserved_top_items = kwargs.pop("user_unobserved_top_items")
        user_unobserved_top_k = user_unobserved_top_items[:k]

        novelty = 0
        for item in user_unobserved_top_k.index:
            novelty += 1 - (item_popularity[item] * len_observed_data) / n_users

        return novelty / k

    def _get_novelty(self, k, user_id=None, **kwargs):
        return self.__get_with_user_ids("novelty", k, user_id=user_id, **kwargs)

    @staticmethod
    def _compute_item_coverage(user_id, k, **kwargs):
        RcLogger.get().debug(f"Computing coverage of user '{user_id}' with k = {k}")
        # type hints
        unobserved_top_k: pd.DataFrame

        user_unobserved_top_items = kwargs.pop("user_unobserved_top_items")
        user_unobserved_top_k = user_unobserved_top_items[:k]

        coverage = defaultdict(int)
        for item in user_unobserved_top_k.index:
            coverage[item] += 1

        return coverage

    def _get_item_coverage(self, k, **kwargs):
        n_items = len(kwargs.pop("predictions").columns)
        items_coverage_k = general_utils.dicts_sum(
            *[m[k] for m in self.individual_custom_metrics["item_coverage"].values()]
        )
        return round(len(list(filter(lambda item_cov: item_cov > 0, items_coverage_k.values()))) / n_items, 2)

    def _compute_user_coverage(self, k, **kwargs):
        return round(len([m[k] for m in self.individual_custom_metrics["ndcg"].values() if m[k] != 0]), 4)

    def _get_user_coverage(self, k, **kwargs):
        return self._custom_dependant_metrics["user_coverage"][k]

    def _items_coverage_k(self, k, items):
        items_coverage_k = general_utils.dicts_sum(
            *[m[k] for m in self.individual_custom_metrics["item_coverage"].values()]
        )
        return np.array([items_coverage_k[item] if item in items_coverage_k else 0 for item in items])

    @staticmethod
    def _compute_item_tpr(user_id, k, **kwargs):
        RcLogger.get().debug(f"Computing item TPR of user '{user_id}' with k = {k}")
        # type hints
        user_unobserved_top_items: pd.DataFrame
        unobserved_items: Dict[str, Set[str]]

        user_unobserved_top_items = kwargs.pop("user_unobserved_top_items")
        user_unobserved_top_k = user_unobserved_top_items[:k]

        unobserved_items = kwargs.pop("unobserved_items")

        item_tpr = {}
        for item in user_unobserved_top_k.index:
            item_tpr[item] = 0
            if item in unobserved_items[user_id]:
                item_tpr[item] += 1

        return item_tpr

    def _get_item_tpr(self, k, item_id=None, **kwargs):
        item_tpr_k = general_utils.dicts_sum(*[m[k] for m in self.individual_custom_metrics["item_tpr"].values()])

        # TODO: find way not to return dictionaries
        return -1 #item_tpr_k if item_id is None else item_tpr_k.get(item_id, 0)

    def _compute_psp(self, k, **kwargs):
        RcLogger.get().debug(f"Computing PSP with k = {k}")
        # type hints
        predictions: pd.DataFrame

        predictions = kwargs.pop("predictions")
        items = predictions.columns
        n_users = len(predictions)

        items_coverage_k = self._items_coverage_k(k, items)
        # TODO: gini gives IndexError (to check)
        return 1  # - metric_utils.gini(items_coverage_k / (k * n_users))

    def _get_psp(self, k, **kwargs):
        return self._custom_dependant_metrics["psp"][k]

    def _compute_peo(self, k, **kwargs):
        RcLogger.get().debug(f"Computing PEO with k = {k}")
        # type hints
        predictions: pd.DataFrame
        observed_items: Dict[str, Set[str]]

        predictions = kwargs.pop("predictions")
        unobserved_items = kwargs.pop("unobserved_items")

        items = predictions.columns

        tpr_item_per_user_ratio = []
        for item in items:
            times_item_per_user = sum([1 if item in user_items else 0 for user_items in unobserved_items.values()])
            if times_item_per_user > 0:
                tpr_item_per_user_ratio.append(self._get_item_tpr(k, item_id=item) / times_item_per_user)
        # TODO: gini gives IndexError (to check)
        return 1  # - metric_utils.gini(np.array(tpr_item_per_user_ratio))

    def _get_peo(self, k, **kwargs):
        return self._custom_dependant_metrics["peo"][k]

    def get_confusion_matrix(self, k, user_id=None, **kwargs) -> pd.DataFrame:
        user_label = f"for user: {user_id}" if user_id is not None else 'all'
        labels = [f"predicted item in top_{k} {user_label}", f"predicted item not in top_{k} {user_label}"]

        if user_id is None:
            data = [
                [
                    round(np.mean([m[k] for m in self.individual_custom_metrics["tp"].values()]), 4),
                    round(np.mean([m[k] for m in self.individual_custom_metrics["fp"].values()]), 4)
                ],
                [
                    round(np.mean([m[k] for m in self.individual_custom_metrics["fn"].values()]), 4),
                    round(np.mean([m[k] for m in self.individual_custom_metrics["tn"].values()]), 4)
                ]
            ]
        else:
            data = [
                [self.individual_custom_metrics['tp'][user_id][k], self.individual_custom_metrics['fp'][user_id][k]],
                [self.individual_custom_metrics['fn'][user_id][k], self.individual_custom_metrics['tn'][user_id][k]]
            ]

        return pd.DataFrame(
            data,
            columns=labels,
            index=labels
        )

    def _compute_demographic_parity(self, k, **kwargs):
        RcLogger.get().debug(f"Computing Demographic Parity")
        sensitive_groups: Dict[str, bool]

        # True values represent Males, False represent Females
        sensitive_groups = kwargs.pop("sensitive")

        sens_gr1 = [g for g in sensitive_groups if sensitive_groups[g]]
        sens_gr2 = [g for g in sensitive_groups if not sensitive_groups[g]]

        predicted_positive_g1 = self._get_predicted_positive(k, user_id=sens_gr1, **kwargs) / len(sens_gr1)
        predicted_positive_g2 = self._get_predicted_positive(k, user_id=sens_gr2, **kwargs) / len(sens_gr2)

        return abs(predicted_positive_g1 - predicted_positive_g2)

    def _get_demographic_parity(self, k, **kwargs):
        return round(self._custom_fairness_metrics["demographic_parity"][k], 4)

    def _compute_equal_opportunity(self, k, **kwargs):
        RcLogger.get().debug(f"Computing Equal Opportunity")
        sensitive_groups: Dict[str, bool]

        # True values represent Males, False represent Females
        sensitive_groups = kwargs.pop("sensitive")

        sens_gr1 = [g for g in sensitive_groups if sensitive_groups[g]]
        sens_gr2 = [g for g in sensitive_groups if not sensitive_groups[g]]

        tpr_g1 = self._get_tpr(k, user_id=sens_gr1)
        tpr_g2 = self._get_tpr(k, user_id=sens_gr2)

        return abs(tpr_g1 - tpr_g2)

    def _get_equal_opportunity(self, k, **kwargs):
        return round(self._custom_fairness_metrics["equal_opportunity"][k], 4)

    @staticmethod
    def _compute_rating_demographic_parity(k, **kwargs):
        RcLogger.get().debug(f"Computing Rating Demographic Parity")
        sensitive_groups: Dict[str, bool]
        predictions: pd.DataFrame

        # True values represent Males, False represent Females
        sensitive_groups = kwargs.pop("sensitive")
        predictions = kwargs.pop("predictions")

        sens_gr1 = [g for g in sensitive_groups if sensitive_groups[g]]
        sens_gr2 = [g for g in sensitive_groups if not sensitive_groups[g]]

        rating_demo_parity = []
        for item in predictions.columns:
            value_gr1 = np.nanmean(predictions.loc[[u for u in sens_gr1 if u in predictions.index], item])
            value_gr2 = np.nanmean(predictions.loc[[u for u in sens_gr2 if u in predictions.index], item])

            value_gr1 = value_gr1 if value_gr1 != np.nan else 0
            value_gr2 = value_gr2 if value_gr2 != np.nan else 0

            rating_demo_parity.append(abs(value_gr1 - value_gr2))

        return np.mean(rating_demo_parity)

    def _get_rating_demographic_parity(self, k, **kwargs):
        return round(self._custom_fairness_metrics["rating_demographic_parity"][k], 4)

    @staticmethod
    def _compute_rating_equal_opportunity(k, **kwargs):
        RcLogger.get().debug(f"Computing Rating Equal Opportunity")
        sensitive_groups: Dict[str, bool]
        predictions: pd.DataFrame
        test_rating_df: pd.DataFrame

        # True values represent Males, False represent Females
        sensitive_groups = kwargs.pop("sensitive")
        predictions = kwargs.pop("predictions")
        test_rating_df = kwargs.pop("test_rating_dataframe")

        sens_gr1 = [g for g in sensitive_groups if sensitive_groups[g]]
        sens_gr2 = [g for g in sensitive_groups if not sensitive_groups[g]]

        rating_equal_opp = []
        for item in predictions.columns:
            value_gr1 = np.nanmean(
                (
                        predictions.loc[[u for u in sens_gr1 if u in predictions.index], item] -
                        test_rating_df.loc[[u for u in sens_gr1 if u in predictions.index], item]
                ).abs()
            )
            value_gr2 = np.nanmean(
                (
                        predictions.loc[[u for u in sens_gr2 if u in predictions.index], item] -
                        test_rating_df.loc[[u for u in sens_gr2 if u in predictions.index], item]
                ).abs())

            value_gr1 = value_gr1 if not np.isnan(value_gr1) and value_gr1 is not None else 0
            value_gr2 = value_gr2 if not np.isnan(value_gr2) and value_gr1 is not None else 0

            rating_equal_opp.append(abs(value_gr1 - value_gr2))

        return np.mean(rating_equal_opp)

    def _get_rating_equal_opportunity(self, k, **kwargs):
        return round(self._custom_fairness_metrics["rating_equal_opportunity"][k], 4)

    @staticmethod
    def __compute_fn_on_group_predictions(k, fn, **kwargs):
        sensitive_groups: Dict[str, bool]
        predictions: pd.DataFrame

        # True values represent Males, False represent Females
        sensitive_groups = kwargs.pop("sensitive")
        predictions = kwargs.pop("predictions")

        sens_gr1 = [g for g in sensitive_groups if sensitive_groups[g]]
        sens_gr2 = [g for g in sensitive_groups if not sensitive_groups[g]]

        pred_gr1 = predictions.loc[[u for u in sens_gr1 if u in predictions.index]]
        pred_gr2 = predictions.loc[[u for u in sens_gr2 if u in predictions.index]]

        return fn(pred_gr1, pred_gr2)

    @staticmethod
    def _compute_mad(k, **kwargs):
        """ Mean Absolute Difference

        :param k: unused
        :param kwargs:
        :return:
        """
        RcLogger.get().debug(f"Computing Mean Absolute Difference")

        return Metrics.__compute_fn_on_group_predictions(
            k,
            lambda p_gr1, p_gr2: abs(np.nanmean(p_gr1.values) - np.nanmean(p_gr2.values)),
            **kwargs
        )

    def _get_mad(self, k, **kwargs):
        return round(self._custom_fairness_metrics["mad"][k], 4)

    @staticmethod
    def _compute_non_parity(k, **kwargs):
        """ Non Parity
            https://arxiv.org/pdf/1705.08804.pdf Yao & Huang
        :param k: unused
        :param kwargs:
        :return:
        """
        RcLogger.get().debug(f"Computing Non Parity")

        return Metrics._compute_mad(k, **kwargs)

    def _get_non_parity(self, k, **kwargs):
        return round(self._custom_fairness_metrics["non_parity"][k], 4)

    @staticmethod
    def _compute_ks(k, **kwargs):
        """ Kolmogorov-Smirnov statistic test (KS)

        :param k:
        :param kwargs:
        :return:
        """
        RcLogger.get().debug(f"Computing Kolmogorov-Smirnov")

        return Metrics.__compute_fn_on_group_predictions(
            k,
            lambda p_gr1, p_gr2: scipy.stats.kstest(
                p_gr1.values.flatten()[~np.isnan(p_gr1.values.flatten())],
                p_gr2.values.flatten()[~np.isnan(p_gr2.values.flatten())],
                alternative="two-sided"
            ),
            **kwargs
        )

    def _get_ks(self, k, **kwargs):
        return self._custom_fairness_metrics["ks"][k]

    @staticmethod
    def _compute_mannwhitneyu(k, **kwargs):
        """ Mann-Whitney U statistic test

        :param k:
        :param kwargs:
        :return:
        """
        RcLogger.get().debug(f"Computing Mann-Whitney U")

        return Metrics.__compute_fn_on_group_predictions(
            k,
            lambda p_gr1, p_gr2: scipy.stats.mannwhitneyu(
                p_gr1.values.flatten()[~np.isnan(p_gr1.values.flatten())],
                p_gr2.values.flatten()[~np.isnan(p_gr2.values.flatten())],
            ),
            **kwargs
        )

    def _get_mannwhitneyu(self, k, **kwargs):
        return self._custom_dependant_metrics["mannwhitneyu"][k]

    @staticmethod
    def _compute_epsilon_fairness(k, **kwargs):
        """
        Gabriel Frisch, Jean-Benoist Leger, Yves Grandvalet. Co-clustering for fair recommendation. 2021. hal-03239856

        Since it is based on combinations between items, not all data is considered, but just a sample
        It uses relevance, so only for ranking target should be used
        :param kwargs:
        :return:
        """
        RcLogger.get().debug(f"Computing Epsilon-Fairness")
        sensitive_groups: Dict[str, bool]
        predictions: pd.DataFrame

        # True values represent Males, False represent Females
        sensitive_groups = kwargs.pop("sensitive")
        predictions = kwargs.pop("predictions")

        sens_gr1 = [g for g in sensitive_groups if sensitive_groups[g]]
        sens_gr2 = [g for g in sensitive_groups if not sensitive_groups[g]]

        sens_gr1 = [u for u in sens_gr1 if u in predictions.index]
        sens_gr2 = [u for u in sens_gr2 if u in predictions.index]

        items = predictions.columns.copy()

        result = []
        sample_size = 5000
        for i, (item1, item2) in enumerate(itertools.combinations(items, 2)):
            if i == sample_size:
                break

            df = predictions[[item1, item2]]
            df = df.dropna()

            gr1 = [u for u in sens_gr1 if u in df.index]
            gr2 = [u for u in sens_gr2 if u in df.index]

            if len(gr1) == 0 or len(gr2) == 0:  # no one of one group or both has relevance for one of the 2 items
                sample_size += 1
                continue

            mask: pd.Series = df[df.columns[0]] > df[df.columns[1]]

            preference_gr1 = len(mask.loc[gr1].index[mask.loc[gr1]])
            preference_gr2 = len(mask.loc[gr2].index[mask.loc[gr2]])

            result.append(abs(preference_gr1 / len(gr1) - preference_gr2 / len(gr2)))

        return np.mean(result)

    def _get_epsilon_fairness(self, k, **kwargs):
        return round(self._custom_fairness_metrics["epsilon_fairness"][k], 4)

    @staticmethod
    def __compute_yao_huang_fairness_metric(func, **kwargs):
        """

        :param func: function that takes in order pred_gr1_mean, pred_gr2_mean, true_gr1_mean, true_gr2_mean and apply
                     one of the yao and huang fairness metrics
                     https://arxiv.org/pdf/1705.08804.pdf
        :param kwargs:
        :return:
        """
        sensitive_groups: Dict[str, bool]
        predictions: pd.DataFrame
        test_rating_df: pd.DataFrame

        # True values represent Males, False represent Females
        sensitive_groups = kwargs.pop("sensitive")
        predictions = kwargs.pop("predictions")
        test_rating_df = kwargs.pop("test_rating_dataframe")

        sens_gr1 = [g for g in sensitive_groups if sensitive_groups[g]]
        sens_gr2 = [g for g in sensitive_groups if not sensitive_groups[g]]

        unfairness_scores = []

        for item in predictions.columns:
            pred_gr1_mean = np.nanmean(predictions.loc[[u for u in sens_gr1 if u in predictions.index], item])
            pred_gr2_mean = np.nanmean(predictions.loc[[u for u in sens_gr2 if u in predictions.index], item])

            true_gr1_mean = np.nanmean(test_rating_df.loc[[u for u in sens_gr1 if u in test_rating_df.index], item])
            true_gr2_mean = np.nanmean(test_rating_df.loc[[u for u in sens_gr2 if u in test_rating_df.index], item])

            unfairness_scores.append(abs(func(pred_gr1_mean, pred_gr2_mean, true_gr1_mean, true_gr2_mean)))

        return np.nanmean(unfairness_scores)

    @staticmethod
    def _compute_value_unfairness(k, **kwargs):
        RcLogger.get().debug(f"Computing Value Unfairness")

        def value_unfairness(pred_gr1_mean, pred_gr2_mean, true_gr1_mean, true_gr2_mean):
            return (pred_gr1_mean - true_gr1_mean) - (pred_gr2_mean - true_gr2_mean)

        return Metrics.__compute_yao_huang_fairness_metric(value_unfairness, **kwargs)

    def _get_value_unfairness(self, k, **kwargs):
        return round(self._custom_fairness_metrics["value_unfairness"][k], 4)

    @staticmethod
    def _compute_absolute_unfairness(k, **kwargs):
        RcLogger.get().debug(f"Computing Absolute Unfairness")

        def absolute_unfairness(pred_gr1_mean, pred_gr2_mean, true_gr1_mean, true_gr2_mean):
            return abs(pred_gr1_mean - true_gr1_mean) - abs(pred_gr2_mean - true_gr2_mean)

        return Metrics.__compute_yao_huang_fairness_metric(absolute_unfairness, **kwargs)

    def _get_absolute_unfairness(self, k, **kwargs):
        return round(self._custom_fairness_metrics["absolute_unfairness"][k], 4)

    @staticmethod
    def _compute_underestimation_unfairness(k, **kwargs):
        RcLogger.get().debug(f"Computing Underestimation Unfairness")

        def underestimation_unfairness(pred_gr1_mean, pred_gr2_mean, true_gr1_mean, true_gr2_mean):
            return max(0, true_gr1_mean - pred_gr1_mean) - max(0, true_gr2_mean - pred_gr2_mean)

        return Metrics.__compute_yao_huang_fairness_metric(underestimation_unfairness, **kwargs)

    def _get_underestimation_unfairness(self, k, **kwargs):
        return round(self._custom_fairness_metrics["underestimation_unfairness"][k], 4)

    @staticmethod
    def _compute_overestimation_unfairness(k, **kwargs):
        RcLogger.get().debug(f"Computing Overestimation Unfairness")

        def overestimation_unfairness(pred_gr1_mean, pred_gr2_mean, true_gr1_mean, true_gr2_mean):
            return max(0, pred_gr1_mean - true_gr1_mean) - max(0, pred_gr2_mean - true_gr2_mean)

        return Metrics.__compute_yao_huang_fairness_metric(overestimation_unfairness, **kwargs)

    def _get_overestimation_unfairness(self, k, **kwargs):
        return round(self._custom_fairness_metrics["overestimation_unfairness"][k], 4)

    @staticmethod
    def _compute_equity_score(k, **kwargs):
        """
        Consumer Fairness Equity Score
        http://proceedings.mlr.press/v81/burke18a/burke18a.pdf
        :param k:
        :param kwargs:
        :return:
        """
        RcLogger.get().debug(f"Computing Equity Score")
        sensitive_groups: Dict[str, bool]
        predictions: pd.DataFrame
        observed_items: Dict[str, Set[str]]
        categories: Dict[str, Iterable]

        # True values represent Males, False represent Females
        sensitive_groups = kwargs.pop("sensitive")
        predictions = kwargs.pop("predictions")
        observed_items = kwargs.pop("observed_items")
        categories = kwargs.pop("categories")

        sens_gr1 = [g for g in sensitive_groups if sensitive_groups[g]]
        sens_gr2 = [g for g in sensitive_groups if not sensitive_groups[g]]

        unique_categories = np.unique(np.concatenate(list(categories.values())))

        equity_score = dict.fromkeys(unique_categories)

        for cat in unique_categories:
            # the current category is considered to be protected

            equity_gr1 = []
            for u_gr1 in sens_gr1:
                pred_u = predictions.loc[
                    u_gr1,
                    [[item for item in predictions.columns if item not in observed_items[u_gr1]]]
                ]
                top_k_items = pred_u.sort_values(ascending=False)[:k].index
                for item in top_k_items:
                    if cat in categories[item]:
                        equity_gr1.append(1)

            equity_gr2 = []
            for u_gr2 in sens_gr2:
                pred_u = predictions.loc[
                    u_gr2,
                    [[item for item in predictions.columns if item not in observed_items[u_gr2]]]
                ]
                top_k_items = pred_u.sort_values(ascending=False)[:k].index
                for item in top_k_items:
                    if cat in categories[item]:
                        equity_gr2.append(1)

            equity_score[cat] = (equity_gr1 / len(sens_gr1)) / (equity_gr2 / len(sens_gr2))

        return equity_score

    def _get_equity_score(self, k, **kwargs):
        return self._custom_fairness_metrics["equity_score"][k]

    @staticmethod
    def _compute_gei(k, **kwargs):
        """
        Generalized Entropy Index
        https://www.sciencedirect.com/science/article/pii/S0306457321001369

        Adapted for group fairness according to https://dl.acm.org/doi/pdf/10.1145/3219819.3220046
        :param k:
        :param kwargs:
        :return:
        """
        RcLogger.get().debug(f"Computing Generalized Entropy Index")
        sensitive_groups: Dict[str, bool]
        predictions: pd.DataFrame
        test_rating_df: pd.DataFrame

        # True values represent Males, False represent Females
        sensitive_groups = kwargs.pop("sensitive")
        predictions = kwargs.pop("predictions")
        test_rating_df = kwargs.pop("test_rating_dataframe")

        sens_gr1 = [g for g in sensitive_groups if sensitive_groups[g]]
        sens_gr2 = [g for g in sensitive_groups if not sensitive_groups[g]]
        groups = [sens_gr1, sens_gr2]

        alpha = kwargs.get("alpha", 2)
        rating_scale_max = kwargs.get("rating_scale_max", 5)

        groups_gei = []
        group_benefits = []
        for gr in groups:
            users_individual_benefit = [
                Metrics.__individual_benefit_function(
                    predictions.loc[user],
                    test_rating_df.loc[user],
                    rating_scale_max=rating_scale_max) for user in gr if user in predictions.index
            ]

            users_individual_benefit = np.array(users_individual_benefit)
            group_benefits.append(users_individual_benefit)
            u_ind_ben_mean = np.mean(users_individual_benefit)

            if alpha == 0:
                gr_gei = -np.mean(
                    np.log(users_individual_benefit / u_ind_ben_mean)
                )
            elif alpha == 1:
                gr_gei = np.mean(
                    (users_individual_benefit / u_ind_ben_mean) * np.log((users_individual_benefit / u_ind_ben_mean))
                )
            else:
                gr_gei = np.mean((users_individual_benefit / u_ind_ben_mean) ** alpha - 1) / (alpha * (alpha - 1))

            groups_gei.append(gr_gei)

        n_tot_users = len(sens_gr1) + len(sens_gr2)
        mean_tot_ben = np.concatenate(group_benefits).mean()
        gr_first_values = []
        gr_second_values = []
        for _gr, _gr_ben, _gr_gei in zip(groups, group_benefits, groups_gei):
            gr_first_values.append((len(_gr) / n_tot_users) * ((np.mean(_gr_ben) / mean_tot_ben) ** alpha) * _gr_gei)

            _gr_ben_mean_div_mean_tot = np.mean(_gr_ben) / mean_tot_ben
            if alpha == 0:
                gr_second_values.append(
                    -((len(_gr) / n_tot_users) * np.log(_gr_ben_mean_div_mean_tot))
                )
            elif alpha == 1:
                gr_second_values.append(
                    (len(_gr) / n_tot_users) * _gr_ben_mean_div_mean_tot * np.log(_gr_ben_mean_div_mean_tot)
                )
            else:
                gr_second_values.append(
                    (len(_gr) / (n_tot_users * alpha * (alpha - 1))) * (_gr_ben_mean_div_mean_tot ** alpha - 1)
                )

        return sum([sum(gr_first_values), sum(gr_second_values)])

    def _get_gei(self, k, **kwargs):
        return self._custom_fairness_metrics["gei"][k]

    @staticmethod
    def _compute_theil(k, **kwargs):
        """
        Theil Index
        https://www.sciencedirect.com/science/article/pii/S0306457321001369
        :param k:
        :param kwargs:
        :return:
        """
        RcLogger.get().debug(f"Computing Theil Index")

        return Metrics._compute_gei(k, **{**kwargs, 'alpha': 1})

    def _get_theil(self, k, **kwargs):
        return self._custom_fairness_metrics["theil"][k]

    @staticmethod
    def __individual_benefit_function(prediction, true_ratings, rating_scale_max=5):
        b_i = []
        unique_items = prediction.columns
        for item in unique_items:
            item_benefit = np.nanmean(abs(prediction.loc[item] - true_ratings.loc[item]))
            if not np.isnan(item_benefit):
                b_i.append(item_benefit)

        return rating_scale_max if len(b_i) == 0 else rating_scale_max - np.mean(b_i)

    @staticmethod
    def supported_custom_metrics():
        get_prefix = Metrics._methods_get_prefix
        metrics_methods = inspect.getmembers(Metrics, predicate=inspect.isroutine)
        metrics_methods = [m[0] for m in metrics_methods
                           if m[0].startswith(get_prefix)]
        metrics_methods = [m.replace(get_prefix, '') for m in metrics_methods]
        return metrics_methods

    @property
    def metrics_dependant(self):
        # TODO: extend the code by usign a "priority" fashion. Metrics with same priority will be computed in the same
        #  cycle and the cycles will follow ascending order. In this way metrics (priority 3) depend on
        #  metrics (priority 2) that depend
        return list(self._dependant_metrics_dependecies.keys())

    @property
    def _dependant_metrics_dependecies(self):
        dep = {
            'user_coverage': ["ndcg"],
            'psp': ["item_coverage"],
            'peo': ["item_tpr"],
            'demographic_parity': ["tp", "fp"],
            'equal_opportunity': ["tpr"],
            'mad': [],
            'ks': [],
            'value_unfairness': [],
            'absolute_unfairness': [],
            'underestimation_unfairness': [],
            'overestimation_unfairness': [],
            'non_parity': [],
            'epsilon_fairness': [],
            'mannwhitneyu': [],
            'equity_score': [],
            'gei': [],
            'theil': [],
            'rating_demographic_parity': [],
            'rating_equal_opportunity': []
        }
        return dep

    @property
    def fairness_metrics(self):
        return [
            'demographic_parity',
            'equal_opportunity',
            'mad',
            'ks',
            'value_unfairness',
            'absolute_unfairness',
            'underestimation_unfairness',
            'overestimation_unfairness',
            'non_parity',
            'epsilon_fairness',
            'equity_score',
            'gei',
            'theil',
            'rating_demographic_parity',
            'rating_equal_opportunity'
        ]

    @property
    def model_metric(self):
        return self._model_metric

    @staticmethod
    def _metric_cutoff_lower_than_topk_warning(metric_name, topk_len, cutoff):
        RcLogger.get().warn(f"<{metric_name}> not accurate metric: top_k items length "
                            f"({topk_len})is lower than cutoff ({cutoff})")

    @staticmethod
    def best_metrics_value(path: Union[str, os.PathLike],
                           metric: str,
                           model: str,
                           k: int = None) -> Union[Tuple[str, Dict], Dict]:
        """
        Function to find the best metrics values for a particular model and metric among several csv files of metrics
        :param path:
        :param metric:
        :param model:
        :param k:
        :return:
        """
        # TODO: implement to choose the K and return the best among files with that K
        # if path is a file than the best among k is chosen, if path is a dir than the best among the files in the dir

        def best_metric_among_k(_df: pd.DataFrame, _metric: str, file):
            _df_metric = _df[_df.columns[_df.columns.str.startswith(metric)]]
            if _df_metric.empty:
                _msg = f"No metric named {_metric} inside " \
                       f"{file.path if isinstance(file, os.DirEntry) else os.path.abspath(file)}"
                raise RcLoggerException(ValueError, _msg)

            return _df_metric.sort_values(by=[0], axis=1, ascending=False).to_dict(orient='split')

        if os.path.isdir(path):
            files = os.scandir(path)
            files = list(filter(lambda file: model in file.path, files))
            if not files:
                msg = f"No metrics file of model {model}"
                raise RcLoggerException(FileNotFoundError, msg)

            best_run_id = os.path.splitext(files[0].name)[0].split('_')[-1]
            first_df = pd.read_csv(files[0].path)
            best_metric = first_df[first_df.columns[first_df.columns.str.startswith(metric)]]
            if best_metric.empty:
                msg = f"No metric named {metric} inside {files[0].path}"
                raise RcLoggerException(ValueError, msg)

            best_metric = best_metric.sort_values(by=[0], axis=1, ascending=False).to_dict(orient='split')
            best_metric_str = best_metric['columns'][0]
            best_metric_value = best_metric['data'][0][0]

            for f in files[1:]:
                df_metric = best_metric_among_k(pd.read_csv(f.path), metric, f)
                if df_metric['data'][0][0] > best_metric_value:
                    best_run_id = os.path.splitext(f.name)[0].split('_')[-1]
                    best_metric_str = df_metric['columns'][0]
                    best_metric_value = df_metric['data'][0][0]
        else:
            best_metric = best_metric_among_k(pd.read_csv(path), metric, path)
            best_run_id = None
            best_metric_str = best_metric['columns'][0]
            best_metric_value = best_metric['data'][0][0]

        best_returned = {'k': int(best_metric_str.split('_')[-1]), metric: best_metric_value}

        return (best_run_id, best_returned) if best_run_id is not None else best_returned


# TODO: use seaborn instead of pandas plotting
class PlotAccessor(object):

    def __init__(self, model_metrics, custom_metrics, individual_custom_metrics, fairness_metrics, dependant_metrics):
        self._model_metrics = model_metrics
        self._custom_metrics = custom_metrics
        # TODO: use all the metrics together (remember they are references of Metrics attributes)
        self._individual_custom_metrics = individual_custom_metrics
        self._fairness_metrics = fairness_metrics
        self._dependant_metrics = dependant_metrics

    @metric_utils.show
    def bar(self, y: Union[str, Sequence[str]], x: Union[str, Sequence[Any]] = "k", **kwargs):
        return self._bar_plot(x, y, orientation="v", **kwargs)

    @metric_utils.show
    def barh(self, y: Union[str, Sequence[str]], x: Union[str, Sequence[Any]] = "k", **kwargs):
        return self._bar_plot(x, y, orientation="h", **kwargs)

    def _bar_plot(self, x, y, orientation="v", **kwargs):
        _x, _y = None, {}

        if isinstance(y, str):
            y = [y]

        bar_plot_kind = "bar" if orientation.lower() == "v" else "barh" if orientation.lower() == "h" else None
        if bar_plot_kind is None:
            raise RcLoggerException(ValueError, f"Cannot plot bar plot with orientation = '{orientation.lower()}'")

        # k if k is present else a random one inside individual_custom_metrics is taken (not used if x == "k")
        if "k" in kwargs:
            k = kwargs.pop("k")
        else:
            k = list(list(list(self._individual_custom_metrics.values())[0].values())[0].keys())[0]

        if x == "k":
            for metr in y:
                if metr in self._custom_metrics:
                    if _x is None:
                        _x = list(self._custom_metrics[metr].keys())
                    _y[metr] = list(self._custom_metrics[metr].values())
                else:
                    msg = f"metric '{metr}' has not been computed yet, call 'get_metrics' first"
                    raise RcLoggerException(AttributeError, msg)

            title = f"K Bar{'h' if orientation == 'h' else ''} plot over {list(_y.keys())}"
            return getattr(pd.DataFrame({'k': _x, **_y}).plot, bar_plot_kind)(x='k', y=y, title=title, **kwargs)

        elif x == "per_user":
            if not isinstance(k, int):
                msg = f"k parameter must be an integer, got '{k}' instead"
                raise RcLoggerException(ValueError, msg)

            for metr in y:
                if metr in self._individual_custom_metrics:
                    if _x is None:
                        _x = list(self._individual_custom_metrics[metr].keys())
                    _y[metr] = [self._individual_custom_metrics[metr][user][k] for user in _x]
                else:
                    msg = f"metric '{metr}' with k = '{k}' has not been computed yet, call 'get_metrics' first"
                    raise RcLoggerException(AttributeError, msg)

        elif isinstance(x, Sequence):
            # x as a Sequence is treated as a sequence of users, which will be used as categories for the bar plot
            _x = x
            for metr in y:
                if metr in self._individual_custom_metrics:
                    _y[metr] = [self._individual_custom_metrics[metr][user][k] for user in _x]
                else:
                    msg = f"metric '{metr}' with k = '{k}' has not been computed yet, call 'get_metrics' first"
                    raise RcLoggerException(AttributeError, msg)
            else:
                _y = y
        else:
            msg = f"Cannot create plot with x = '{x}'. Value not supported"
            raise RcLoggerException(ValueError, msg)

        title = f"Bar{'h' if orientation == 'h' else ''} plot over {list(_y.keys())} with x-axis {x} and k: {k}"
        return getattr(pd.DataFrame(_y, index=_x).plot, bar_plot_kind)(title=title, **kwargs)

    @metric_utils.show
    def box(self, metrics: Union[str, Sequence[str]], by="k", **kwargs):
        by_values = ["k", "per_user"]

        if isinstance(metrics, str):
            metrics = [metrics]

        k = kwargs.pop("k") if "k" in kwargs else None

        if by == "k":
            data = {m: list(self._custom_metrics[m].values()) for m in metrics}
        elif by == "per_user":
            data = {}

            if k is not None:
                for m in metrics:
                    data[m] = [u[k] for u in self._individual_custom_metrics[m].values()]
            else:
                for m in metrics:
                    data = {}
                    ks = list(list(self._individual_custom_metrics[m].values())[0].keys())

                    title = f"Box plot over {m} by {by}"
                    x, y = max(2, len(ks) // 2), 2 + (len(ks) % 2)
                    fig = plt.figure()
                    fig.suptitle(title)

                    for i, _k in enumerate(ks):
                        data[m] = [u[_k] for u in self._individual_custom_metrics[m].values()]
                        ax = fig.add_subplot(x, y, i+1)

                        pd.DataFrame(data).plot.box(title=f"k: {_k}", ax=ax)
                    return
        else:
            msg = f"by can be one of {by_values}, got by = '{by}' instead"
            raise RcLoggerException(ValueError, msg)

        title = f"Box plot over {metrics} by {by}{f' with k: {k}' if k is not None else ''}"

        return pd.DataFrame(data).plot.box(title=title)

    @metric_utils.show
    def density(self, metrics: Union[str, Sequence[str]], by="k", **kwargs):
        by_values = ["k", "per_user"]

        if isinstance(metrics, str):
            metrics = [metrics]

        k = kwargs.pop("k") if "k" in kwargs else None

        if by == "k":
            data = {m: list(self._custom_metrics[m].values()) for m in metrics}
        elif by == "per_user":
            data = {}

            if k is not None:
                for m in metrics:
                    data[m] = [u[k] for u in self._individual_custom_metrics[m].values()]
            else:
                for m in metrics:
                    data = {}
                    ks = list(list(self._individual_custom_metrics[m].values())[0].keys())

                    title = f"Density plot over {m} by {by}"
                    x, y = max(2, len(ks) // 2), 2 + (len(ks) % 2)
                    fig = plt.figure()
                    fig.suptitle(title)

                    for i, _k in enumerate(ks):
                        data[m] = [u[_k] for u in self._individual_custom_metrics[m].values()]
                        ax = fig.add_subplot(x, y, i + 1)

                        pd.DataFrame(data).plot.kde(title=f"k: {_k}", ax=ax)

                    return
        else:
            msg = f"by can be one of {by_values}, got by = '{by}' instead"
            raise RcLoggerException(ValueError, msg)

        title = f"Density plot over {metrics} by {by}{f' with k: {k}' if k is not None else ''}"

        return pd.DataFrame(data).plot.kde(title=title, **kwargs)

    @metric_utils.show
    def hexbin(self, x, y, **kwargs):
        raise NotImplementedError()

    @metric_utils.show
    def hist(self, metrics: Union[str, Sequence[str]], by="k", **kwargs):
        by_values = ["k", "per_user"]

        if isinstance(metrics, str):
            metrics = [metrics]

        k = kwargs.pop("k") if "k" in kwargs else None

        if by == "k":
            data = {m: list(self._custom_metrics[m].values()) for m in metrics}
        elif by == "per_user":
            data = {}

            if k is not None:
                for m in metrics:
                    data[m] = [u[k] for u in self._individual_custom_metrics[m].values()]
            else:
                for m in metrics:
                    data = {}
                    ks = list(list(self._individual_custom_metrics[m].values())[0].keys())

                    title = f"Histogram plot over {m} by {by}"
                    rows, cols = len(ks) // 3 + 1, min(len(ks), 3)
                    fig = plt.figure()
                    fig.suptitle(title)

                    for i, _k in enumerate(ks):
                        data[m] = [u[_k] for u in self._individual_custom_metrics[m].values()]
                        ax = fig.add_subplot(rows, cols, i + 1)

                        pd.DataFrame(data).plot.hist(title=f"k: {_k}", ax=ax)

                    return
        else:
            msg = f"by can be one of {by_values}, got by = '{by}' instead"
            raise RcLoggerException(ValueError, msg)

        title = f"Histogram plot over {metrics} by {by}{f' with k: {k}' if k is not None else ''}"

        return pd.DataFrame(data).plot.hist(title=title)

    @metric_utils.show
    def kde(self, metrics: Union[str, Sequence[str]], by="k", **kwargs):
        return self.density(metrics, by=by, **kwargs)

    @metric_utils.show
    def line(self, y: Union[str, Sequence[str]], x: Union[str, Sequence[Any]] = "k", **kwargs):
        _x, _y = None, {}

        if isinstance(y, str):
            y = [y]

        # k if k is present else a random one inside individual_custom_metrics is taken (not used if x == "k")
        if "k" in kwargs:
            k = kwargs.pop("k")
        else:
            k = list(list(list(self._individual_custom_metrics.values())[0].values())[0].keys())[0]

        if x == "k":
            for metr in y:
                if metr in self._custom_metrics:
                    if _x is None:
                        _x = list(self._custom_metrics[metr].keys())
                    _y[metr] = list(self._custom_metrics[metr].values())
                else:
                    msg = f"metric '{metr}' has not been computed yet, call 'get_metrics' first"
                    raise RcLoggerException(AttributeError, msg)

            title = f"K Line plot over {list(_y.keys())}"
            return pd.DataFrame({'k': _x, **_y}).plot.line(x='k', y=y, title=title, **kwargs)

        elif x == "per_user":
            if not isinstance(k, int):
                msg = f"k parameter must be an integer, got '{k}' instead"
                raise RcLoggerException(ValueError, msg)

            for metr in y:
                if metr in self._individual_custom_metrics:
                    if _x is None:
                        _x = list(self._individual_custom_metrics[metr].keys())
                    _y[metr] = [self._individual_custom_metrics[metr][user][k] for user in _x]
                else:
                    msg = f"metric '{metr}' with k = '{k}' has not been computed yet, call 'get_metrics' first"
                    raise RcLoggerException(AttributeError, msg)

        elif isinstance(x, Sequence):
            # x as a Sequence is treated as a sequence of users, which will be used as categories for the line plot
            _x = x
            for metr in y:
                if metr in self._individual_custom_metrics:
                    _y[metr] = [self._individual_custom_metrics[metr][user][k] for user in _x]
                else:
                    msg = f"metric '{metr}' with k = '{k}' has not been computed yet, call 'get_metrics' first"
                    raise RcLoggerException(AttributeError, msg)

        else:
            msg = f"Cannot create plot with x = '{x}'. Value not supported"
            raise RcLoggerException(ValueError, msg)

        x_str_title = x if not isinstance(x, Sequence) else f"of {len(x)} users"

        title = f"Line plot over {list(_y.keys())} with x-axis {x_str_title} and k: {k}"
        return pd.DataFrame(_y, index=_x).plot.line(title=title, **kwargs)

    @metric_utils.show
    def pie(self, y: Sequence[Any], kind="gender", **kwargs):
        kind_values = ["gender"]

        if kind not in kind_values:
            msg = f"{kind} is not a supported kind for pie"
            raise RcLoggerException(ValueError, msg)

        return pd.DataFrame({"gender": y}, index=["Male", "Female"]).plot.pie(**kwargs)

    @metric_utils.show
    def scatter(self,
                x: Union[str, Sequence[str]],
                y: Union[str, Sequence[str]],
                maps: Union[str, Sequence[Any]] = "k",
                **kwargs):
        _x, _y = {}, {}

        if len(x) != len(y):
            msg = f"x and y must be the same length, got lenghts x: {len(x)}, y: {len(y)}"
            raise RcLoggerException(ValueError, msg)

        """
        if isinstance(x, str):
            x = [x]
        if isinstance(y, str):
            y = [y]
        """

        # k if k is present else a random one inside individual_custom_metrics is taken (not used if x == "k")
        if "k" in kwargs:
            k = kwargs.pop("k")
        else:
            k = list(list(list(self._individual_custom_metrics.values())[0].values())[0].keys())[0]

        # TODO: supports x and y as sequences
        # Currently only lenghts of 1 for x and y are supported
        if maps == "k":
            ks = sorted(self._custom_metrics[x[0]].keys())
            for x_metr, y_metr in zip(x, y):
                if x_metr in self._custom_metrics and y_metr in self._custom_metrics:
                    _x[x_metr] = [self._custom_metrics[x_metr][_k] for _k in ks]
                    _y[y_metr] = [self._custom_metrics[y_metr][_k] for _k in ks]
                else:
                    msg = f"metrics '{x_metr}' or '{y_metr}' have not been computed yet, call 'get_metrics' first"
                    raise RcLoggerException(AttributeError, msg)

            title = f"K Scatter plot over {list(_y.keys())}"
            return pd.DataFrame({**_x, **_y}).plot.scatter(x=x[0], y=y[0], title=title, **kwargs)

        elif maps == "per_user":
            if not isinstance(k, int):
                msg = f"k parameter must be an integer, got '{k}' instead"
                raise RcLoggerException(ValueError, msg)

            for x_metr, y_metr in zip(x, y):
                if x_metr in self._individual_custom_metrics and \
                        y_metr in self._individual_custom_metrics:
                    _x[x_metr] = [self._individual_custom_metrics[x_metr][user][k]
                                  for user in self._individual_custom_metrics[x_metr]]
                    _y[y_metr] = [self._individual_custom_metrics[y_metr][user][k]
                                  for user in self._individual_custom_metrics[y_metr]]
                else:
                    msg = f"metrics '{x_metr}' or '{y_metr}' with k = '{k}' have not been computed yet, " \
                          f"call 'get_metrics' first"
                    raise RcLoggerException(AttributeError, msg)

        elif isinstance(maps, Sequence):
            for x_metr, y_metr in zip(x, y):
                if x_metr in self._individual_custom_metrics and \
                        y_metr in self._individual_custom_metrics:
                    _x[x_metr] = [self._individual_custom_metrics[x_metr][user][k] for user in maps]
                    _y[y_metr] = [self._individual_custom_metrics[y_metr][user][k] for user in maps]
                else:
                    msg = f"metrics '{x_metr}' or '{y_metr}' with k = '{k}' have not been computed yet, " \
                          f"call 'get_metrics' first"
                    raise RcLoggerException(AttributeError, msg)
            else:
                _y = y
        else:
            msg = f"Cannot create plot with maps = '{maps}'. Value not supported"
            raise RcLoggerException(ValueError, msg)

        title = f"Scatter plot over {list(_y.keys())} with maps {maps} and k: {k}"
        return pd.DataFrame(_y, index=_x).plot.scatter(title=title, **kwargs)

    @metric_utils.show
    def scatter_matrix(self):
        raise NotImplementedError
