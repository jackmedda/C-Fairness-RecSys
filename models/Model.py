import os
import shutil
import time
import inspect
import random
from collections import defaultdict
from typing import (
    Dict,
    Text,
    Union,
    Sequence,
    Any,
    Literal,
    Callable
)

import tensorflow_recommenders as tfrs
import tensorflow as tf
import numpy as np
import tqdm

import rc_types as rc_types
import data.utils as data_utils
import models as models_module
import models.utils as model_utils
import helpers.constants as constants
import helpers.filename_utils as file_utils
from metrics import Metrics
from .utils import RetrievalIndex
from helpers.logger import RcLogger, RcLoggerException

"""
class RankingModel(tf.keras.Model):

    def __init__(self, unique_user_ids, unique_movie_titles, embedding_dimension=32):
        super().__init__()

        # Compute embeddings for users.
        self.user_embeddings = tf.keras.Sequential([
          tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=unique_user_ids, mask_token=None),
          tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
        ])

        # Compute embeddings for movies.
        self.movie_embeddings = tf.keras.Sequential([
          tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=unique_movie_titles, mask_token=None),
          tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
        ])

        # Compute predictions.
        self.ratings = tf.keras.Sequential([
          # Learn multiple dense layers.
          tf.keras.layers.Dense(256, activation="relu"),
          tf.keras.layers.Dense(64, activation="relu"),
          # Make rating predictions in the final layer.
          tf.keras.layers.Dense(1)
        ])

    def call(self, inputs, **kwargs):

        user_id, movie_title = inputs

        user_embedding = self.user_embeddings(user_id)
        movie_embedding = self.movie_embeddings(movie_title)

        return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))


class MovielensModel(tfrs.models.Model):

    def __init__(self, user_model, movie_model, task):
        super().__init__()
        self.ranking_model: tf.keras.Model = RankingModel()
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        rating_predictions = self.ranking_model(
            (features["user_id"], features["movie_title"]))

        return self.task(labels=features["user_rating"], predictions=rating_predictions)
"""


class Model(tfrs.Model):

    MODEL_DATA_TYPE = None

    def __init__(self,
                 model_data: rc_types.ModelData,
                 dataset_metadata: Dict[Text, Any],
                 batch_candidates=128,
                 index: RetrievalIndex = None,
                 embedding_dimension=32,
                 **kwargs):
        super(Model, self).__init__(name=kwargs.get("name", "Model"))

        RcLogger.get().info(f"Model '{self.name}' initialized")
        # tf.debugging.enable_check_numerics()  # checks for Nan or Inf values

        self._model_data = model_data
        self._dataset_metadata = dataset_metadata
        self._batch_candidates = batch_candidates
        self._embedding_dimension = embedding_dimension
        self._index_param = index

        self._train_data: tf.data.Dataset = kwargs.pop('train_data') if "train_data" in kwargs else None
        self._test_data: tf.data.Dataset = kwargs.pop('test_data') if "test_data" in kwargs else None
        self._val_data: tf.data.Dataset = kwargs.pop('val_data') if "validation_data" in kwargs else None
        self._n_train_interactions = None

        self._users_as_array = np.concatenate(list(self.users.as_numpy_iterator()))
        self._items_as_array = np.concatenate(list(self.items.as_numpy_iterator()))
        # self._categories_as_array = np.concatenate(list(self.categories.as_numpy_iterator())) if self.categories is not None else None

        # TODO: extract the fields names directly from `model_data`
        self._user_id_field = kwargs.get('user_id_field', 'user_id')
        self._item_id_field = kwargs.get('item_id_field', 'movie_id')
        self._category_field = kwargs.get('category_field', 'genres')
        self._rating_field = kwargs.get('rating_field', 'user_rating')
        self.sensitive_field = kwargs.get('sensitive_field', 'user_gender')

        # Non lexicographically ordered unique users and items
        if self._users_as_array[0].isdigit():
            self.unique_users = np.array(sorted(list(np.unique(self._users_as_array)), key=int))
        else:
            self.unique_users = np.array(sorted(list(np.unique(self._users_as_array))))
        if self._items_as_array[0].isdigit():
            self.unique_items = np.array(sorted(list(np.unique(self._items_as_array)), key=int))
        else:
            self.unique_items = np.array(sorted(list(np.unique(self._items_as_array))))
        # self.unique_categories = np.unique(self._categories_as_array)

        self.observed_items = defaultdict(set)
        self.unobserved_items = defaultdict(set)
        self.item_popularity = dict.fromkeys(self.unique_items, 0)
        self.rating_dataframe = None
        self.user_sensitive = {}

        self._relevance_matrix = None
        self._full_relevance_matrix = None

        self._optimizer = None
        self._optimizer_as_str = kwargs.pop("optimizer") if "optimizer" in kwargs else "Adagrad"
        self._learning_rate = kwargs.pop("learning_rate") if "learning_rate" in kwargs else 0.01

        self._patience = kwargs.pop("patience") if "patience" in kwargs else 4
        self._limit_early_stopping = kwargs.pop("limit_early_stopping") if "limit_early_stopping" in kwargs else 5e-4
        if kwargs.get("callbacks", False) is not False:
            self._callbacks = kwargs.pop("callbacks")
        else:
            checkpoints_path = os.path.join(constants.SAVE_MODELS_CHECKPOINTS_PATH,
                                            f"{self.name}-{file_utils.default_filepath()}")
            # os.mkdir(checkpoints_path)
            self._callbacks = [

                tf.keras.callbacks.ProgbarLogger(
                    count_mode='steps', stateful_metrics=['loss']
                )
            ]
            # TODO:
            """
            self._callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(checkpoints_path, f'{self.name}' + '.{epoch:}-{loss:.2f}.tf'),
                    monitor="loss",
                    save_best_only=True,
                    save_weights_only=False
                ),
                tf.keras.callbacks.LearningRateScheduler(
                    lambda epoch, lr: lr if epoch < 5 else lr * tf.math.exp(-0.1)
                )
            ]
            """

        self.user_model = kwargs.get("user_model")
        self.item_model = kwargs.get("item_model")
        self.inner_models = []

        self._metric_handler = None
        self._index = None
        self.task = kwargs.pop("task") if "task" in kwargs else "Retrieval"

        self._prepare_embedding_models(
            self._embedding_dimension,
            user_emb_kwargs=kwargs.get("user_embedding_kwargs", {}),
            item_emb_kwargs=kwargs.get("item_embedding_kwargs", {})
        )

        self._candidates = self.items.unbatch().batch(self._batch_candidates).map(self.item_model)

    @property
    def users(self):
        return self._model_data["users"]

    @property
    def items(self):
        return self._model_data["items"]

    @property
    def categories(self):
        return self._model_data["categories"] if "categories" in self._model_data else None

    @property
    def sensitive(self):
        return self._model_data["sensitive"] if "sensitive" in self._model_data else None

    def _prepare_embedding_models(self,
                                  embedding_dimension: Union[Sequence[int], int],
                                  user_emb_kwargs=None,
                                  item_emb_kwargs=None):
        user_emb_kwargs = {} if user_emb_kwargs is None else user_emb_kwargs
        item_emb_kwargs = {} if item_emb_kwargs is None else item_emb_kwargs

        if isinstance(embedding_dimension, Sequence):
            user_embedding_dimension, item_embedding_dimension = embedding_dimension
        else:
            user_embedding_dimension = embedding_dimension
            item_embedding_dimension = embedding_dimension

        if self.user_model is None:
            RcLogger.get().info(f"Using default user_model with embedding dimension = {user_embedding_dimension}")
            self.user_model: tf.keras.Model = tf.keras.Sequential([
                tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=self.unique_users, mask_token=None,
                                                                        num_oov_indices=0, name='user_embedding_lookup'),
                tf.keras.layers.Embedding(len(self.unique_users), user_embedding_dimension, name='user_embedding', **user_emb_kwargs)
            ], name="user_model")

        if self.item_model is None:
            RcLogger.get().info(f"Using default item_model with embedding dimension = {item_embedding_dimension}")
            self.item_model: tf.keras.Model = tf.keras.Sequential([
                tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=self.unique_items, mask_token=None,
                                                                        num_oov_indices=0, name='item_embedding_lookup'),
                tf.keras.layers.Embedding(len(self.unique_items), item_embedding_dimension, name='item_embedding', **item_emb_kwargs)
            ], name="item_model")

        self.inner_models = [(self.user_model, "user_model"), (self.item_model, "item_model")]

    def _prepare_task(self, loss=None, model_metric=None):
        model_metric = tfrs.metrics.FactorizedTopK(self._candidates) if model_metric is None else model_metric

        if self._metric_handler is None:
            if self.task is not None:
                if isinstance(self.task, str):
                    self._metric_handler = Metrics(model_metric=model_metric, model_name=self.name)

                    RcLogger.get().info(f"Using metrics '{self._metric_handler.model_metric.name}'")
                    RcLogger.get().info(f"Using task '{self.task}'")

                    if hasattr(tfrs.tasks, self.task):
                        if loss is None:
                            self.task: tf.keras.layers.Layer = getattr(tfrs.tasks, self.task)(
                                metrics=model_metric
                            )
                        else:
                            self.task: tf.keras.layers.Layer = getattr(tfrs.tasks, self.task)(
                                loss=loss,
                                metrics=model_metric
                            )
                    else:
                        msg = f"Tensorflow recommenders does not have a task called '{self.task}'"
                        raise RcLoggerException(ValueError, msg)

                elif isinstance(self.task, tfrs.tasks.Task):
                    self._metric_handler = Metrics(model_name=self.name)
                else:
                    msg = f"Task must be a string or a tfrs.tasks.Task, got {self.task} with type {type(self.task)}"
                    raise RcLoggerException(ValueError, msg)
            else:
                self._metric_handler = Metrics(model_name=self.name)

    def _prepare_index(self):
        RcLogger.get().info(f"Using index '{'BruteForce' if self._index_param is None else self._index_param.name}'")

        self._index = self._index_param if self._index_param is not None else RetrievalIndex("BruteForce")
        items_dataset = tf.data.Dataset.from_tensors(self.items)
        candidates = items_dataset.map(self.item_model)

        if self._index.identifiers is None:
            identifiers = items_dataset
            """
            items_identifiers = self.items
            if self._index.has_query_model():
                identifiers = items_identifiers.unbatch().batch(self._batch_candidates)
            else:
                identifiers = items_identifiers.unbatch()
            """
            self._index.identifiers = identifiers
        else:
            identifiers = self._index.identifiers

        self._index.complete_index(self.user_model)
        self._index.index(candidates, identifiers)

        RcLogger.get().info(f"Index correctly configured to make predictions")

    def _get_optimizer_from_str(self) -> "Class tf.keras.optimizers.Optimizer":
        optimizer = getattr(tf.keras.optimizers, self._optimizer_as_str, None)

        if optimizer is None:
            msg = f"'{self._optimizer_as_str}' is not a supported optimizer."
            raise RcLoggerException(ValueError, msg)

        return optimizer

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        user_embeddings = self.user_model(features[self._user_id_field])
        positive_item_embeddings = self.item_model(features[self._item_id_field])

        return self.task(user_embeddings, positive_item_embeddings)

    def test(self,
             x=None,
             y=None,
             batch_size=None,
             verbose=1,
             sample_weight=None,
             steps=None,
             callbacks=None,
             max_queue_size=10,
             workers=1,
             use_multiprocessing=False,
             return_dict=False,
             overwrite_saved_test=False):

        if x is not None:
            if self._test_data is None:
                self._test_data = x.unbatch()
        else:
            x = self._test_data
            x = x.batch(batch_size)

        if not data_utils.preprocessed_dataset_exists(self._dataset_metadata,
                                                      model_data_type=self.MODEL_DATA_TYPE,
                                                      split="test") or overwrite_saved_test:
            data_utils.save_tf_features_dataset(self._test_data.batch(2048),
                                                self._dataset_metadata,
                                                dataset_info=self.MODEL_DATA_TYPE,
                                                split="test",
                                                overwrite=overwrite_saved_test)

        RcLogger.get().info("Evaluation (test) of the model")
        start = time.perf_counter()
        metrics = self.evaluate(x=x,
                                y=y,
                                batch_size=batch_size,
                                verbose=verbose,
                                sample_weight=sample_weight,
                                steps=steps,
                                callbacks=callbacks,
                                max_queue_size=max_queue_size,
                                workers=workers,
                                use_multiprocessing=use_multiprocessing,
                                return_dict=return_dict)

        RcLogger.get().info(f"Test (evaluate) duration: "
                            f"{time.strftime('%H:%M:%S', time.gmtime(time.perf_counter() - start))} seconds")
        RcLogger.get().debug(f"Test (evaluate) metrics: {metrics}")

        self._metric_handler.set_model_metrics(metrics)

        if not return_dict:
            metrics = list(metrics.values())

        return metrics

    def train(self,
              x: tf.raw_ops.BatchDataset = None,
              y=None,
              batch_size=None,
              epochs=1,
              verbose=1,
              callbacks=None,
              validation_split=0.,
              validation_data=None,
              shuffle=True,
              class_weight=None,
              sample_weight=None,
              initial_epoch=0,
              steps_per_epoch=None,
              validation_steps=None,
              validation_batch_size=None,
              validation_freq=1,
              max_queue_size=10,
              workers=1,
              use_multiprocessing=False,
              **kwargs):

        if self._train_data is None:
            self._train_data = x.unbatch()

        self._prepare_task()

        if self._optimizer is None:
            self._optimizer = self._get_optimizer_from_str()(learning_rate=self._learning_rate)
        self.compile(optimizer=self._optimizer)
        RcLogger.get().info(f"Using optimizer '{self._optimizer_as_str}' with learning rate '{self._learning_rate}'")

        callbacks = self._callbacks if callbacks is None else callbacks

        RcLogger.get().info("Train (fit) of the model")
        start = time.perf_counter()
        history = self.fit(x=x,
                           y=y,
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=verbose,
                           callbacks=callbacks,
                           validation_split=validation_split,
                           validation_data=validation_data,
                           shuffle=shuffle,
                           class_weight=class_weight,
                           sample_weight=sample_weight,
                           initial_epoch=initial_epoch,
                           steps_per_epoch=steps_per_epoch,
                           validation_steps=validation_steps,
                           validation_batch_size=validation_batch_size,
                           validation_freq=validation_freq,
                           max_queue_size=max_queue_size,
                           workers=workers,
                           use_multiprocessing=use_multiprocessing)

        RcLogger.get().info(f"Train (fit) duration: "
                            f"{time.strftime('%H:%M:%S', time.gmtime(time.perf_counter() - start))} seconds")

        RcLogger.get().debug(f"Train (fit) history: {history.history} after {epochs}")

        self.summary(print_fn=RcLogger.get().debug)

        # TODO: uncomment once the index bugs have been resolved
        # self._prepare_index()

        return history

    def index(self, user_id: Union[tf.data.Dataset, tf.Tensor], **kwargs):
        RcLogger.get().info(f"Started predictions of the index with k = {kwargs.get('k', self._index.k)}")

        k = kwargs.pop("k") if "k" in kwargs else self._index.k

        if isinstance(user_id, tf.Tensor):
            RcLogger.get().info(f"Making predictions with data as tf.Tensor")
            if not self._index.has_query_model():
                user_id = self.user_model(user_id)

            return self._index(user_id, k=k)
        elif isinstance(user_id, tf.data.Dataset):
            RcLogger.get().info(f"Making predictions in a generator manner with data as tf.data.Dataset")
            return self._index_generator(user_id, k=k)
        else:
            msg = f"Expected tf.data.Dataset, tf.Tensor or List, got {type(user_id)}"
            raise RcLoggerException(NotImplementedError, msg)

    def _index_generator(self, user_id: tf.data.Dataset, k=None):
        k = self._index.k if k is None else k
        for data in user_id:
            if not self._index.has_query_model():
                user_embedding = self.user_model(data)
                yield self._index(user_embedding, k=k)
            else:
                yield self._index(data, k=k)

    def save_index(self, filepath=file_utils.default_filepath(), **kwargs):
        RcLogger.get().info(f"Saving index")
        self._index.save(filepath, model=self.name, **kwargs)

    def save(self, foldername=None, save_format="tf", **kwargs):
        # TODO: analyze and consider the parameter 'custom_objects' of keras "save" function
        RcLogger.get().info(f"Saving model in format = '{save_format}'")

        foldername = file_utils.default_filepath() if foldername is None else foldername
        if file_utils.current_run_id() in foldername:
            folder_path = os.path.join(constants.SAVE_MODELS_PATH, f"{self.name}-{self._optimizer_as_str}-{foldername}")
        else:
            folder_path = os.path.join(constants.SAVE_MODELS_PATH, f"{foldername}__{file_utils.current_run_id()}")
        os.mkdir(folder_path)

        try:
            # model_utils.model_save(super(Model, self).save, os.path.join(folder_path, "model"), save_format="tf")
            np.save(os.path.join(folder_path, "model_weights.npy"), np.asarray(self.get_weights(), dtype=object))
            """
            for model, name in self.inner_models:
                out_path = os.path.join(folder_path, f"{name}-{foldername}")
                model_utils.model_save(model.save, out_path, save_format=save_format, **kwargs)
            """
            self._model_data.save(folder_path)
            """
            self._model_data.save(folder_path, other_attrs={
                "batch_candidates": self._batch_candidates,
                "embedding_dimension": self._embedding_dimension
            })
            """
            RcLogger.get().info(f"Saved model at path '{folder_path}'")
        except Exception as e:
            shutil.rmtree(folder_path)
            msg = f"Error saving model with message '{e}'"
            raise RcLoggerException(Exception, msg)

        return file_utils.current_run_id()

    @classmethod
    def load(cls,
             session_id,
             dataset_preprocessing_metadata,
             index: Union[os.PathLike, RetrievalIndex, bool] = None,
             model_name=None,
             train_for_weights=None,
             **kwargs):
        """

        :param session_id:
        :param dataset_preprocessing_metadata:
        :param index:
        :param model_name:
        :param train_for_weights: the dataset that can be used to create the weights. It is necessary to set the loaded
                                  weights
        :param kwargs:
        :return:
        """

        RcLogger.get().info(f"Loading model with session_id '{session_id}'")

        tf_folders = [f for f in os.scandir(constants.SAVE_MODELS_PATH) if os.path.isdir(f) and session_id in f.name][0]
        """
        inner_models_folders = [f for f in os.scandir(tf_folders)]

        user_model = tf.keras.models.load_model([f for f in inner_models_folders if "user_model" in f.name][0].path)
        item_model = tf.keras.models.load_model([f for f in inner_models_folders if "item_model" in f.name][0].path)

        if isinstance(index, str):
            index = RetrievalIndex.load(filepath=index)
        elif isinstance(index, bool):
            index = RetrievalIndex.load(session_id=session_id)
        """

        model_data = data_utils.ModelData.load(tf_folders)

        if kwargs.get("optimizer") is None:
            tf_optimizers = list(
                filter(lambda opt: opt != "Optimizer",
                       [t[0] for t in inspect.getmembers(tf.keras.optimizers, predicate=inspect.isclass)])
            )

            optimizer_name = tf_folders.name.upper()
            saved_optimizer = [opt for opt in tf_optimizers if opt.upper() in optimizer_name]

            if saved_optimizer:
                optimizer = saved_optimizer[0].title()
            else:
                msg = f"There is no optimizer in the saved model name. Error in filename or specify it as argument"
                raise RcLoggerException(ValueError, msg)
        else:
            optimizer = kwargs.get("optimizer")

        kwargs["optimizer"] = optimizer

        # Model can be loaded from calling class or from Model if specific class is written in saved model filepath
        if model_name is None:
            rc_models = inspect.getmembers(models_module, predicate=inspect.isclass)
            rc_models = [_rc_model for _rc_model in rc_models if _rc_model[0] in tf_folders.name]

            if rc_models:
                name = rc_models[0][0]
                model = rc_models[0][1]
                if cls.__name__ != name and cls is not Model:
                    msg = f"Trying to load model of class '{name}' calling load from class '{cls.__name__}'"
                    raise RcLoggerException(ValueError, msg)
            else:
                RcLogger.get().warning(f"filepath with session_id '{session_id}' does not "
                                       f"contain information about the model to load. The model will be loaded "
                                       f"as the calling class '{cls.__name__}'")
                model = cls
                name = cls.__name__
        else:
            name = model_name
            model = getattr(models_module, name)

        loaded = model(model_data,
                       dataset_preprocessing_metadata,
                       name=name,
                       **kwargs)

        if train_for_weights is not None:
            train = train_for_weights
            sample = train_for_weights.take(1).batch(1)
        elif loaded.MODEL_DATA_TYPE is not None:
            train = data_utils.load_tf_features_dataset(
                dataset_preprocessing_metadata,
                dataset_info=loaded.MODEL_DATA_TYPE,
                split="train"
            )
            sample = train.take(1).batch(1)
        else:
            original_train = data_utils.load_tf_features_dataset(
                dataset_preprocessing_metadata,
                dataset_info=None,
                split="orig_train"
            )
            sample = original_train.take(1).batch(1)

            train = original_train

        # weights must be created before being loaded.
        # train is used instead of `call` because self.task is not used inside `call`, but in `compute_loss`
        loaded.train(sample, _loading=True)

        weights = np.load(os.path.join(tf_folders, 'model_weights.npy'), allow_pickle=True)
        loaded.set_weights(weights)

        return loaded, optimizer

    def get_relevance_matrix(self, coverage="standard", k=None, overwrite=False) -> model_utils.RelevanceMatrix:
        RcLogger.get().info(f"Retrieving relevance matrix")

        if coverage == "full":
            if k is not None:
                RcLogger.get().warning("parameter 'k' is not used with full coverage")

            relevance_matrix = self._full_relevance_matrix
            k = len(self.unique_items)
        else:
            relevance_matrix = self._relevance_matrix
            k = self._index.k if k is None else k

        # if relevance_matrix is None or k != self._index.k or overwrite:
        if relevance_matrix is None or overwrite:
            RcLogger.get().info(f"Generating relevance matrix with coverage '{coverage}' and k = {k}")
            # TODO: TRY TO CREATE SAME FUNCTIONALITY OF INDEX, SO USING A SPECIFIC K

            predictions = self.get_predictions()
            #relevances, recommended_items = self.index(tf.constant(self.unique_users), k=k)
            # TODO: add real ratings in this moment (dataframe update) to make possible to save properly the matrix
            relevance_matrix = model_utils.RelevanceMatrix(self.unique_users,
                                                           predictions,
                                                           self.unique_items,
                                                           model_name=self.name,
                                                           k=k)

            if coverage == "full":
                self._full_relevance_matrix = relevance_matrix
            elif k == self._index.k:
                self._relevance_matrix = relevance_matrix

        return relevance_matrix

    def get_predictions(self):
        user_emb = self.user_model(tf.constant(self.unique_users))
        item_emb = self.item_model(tf.constant(self.unique_items))
        emb_scores = tf.linalg.matmul(user_emb, item_emb, transpose_b=True)

        return emb_scores.numpy()

    def get_metrics(self,
                    metrics: Union[Sequence[str], str] = "all",
                    cutoffs: Sequence[int] = None,
                    only: Sequence[str] = None,
                    overwrite=False,
                    validation_set: tf.data.Dataset = None):
        RcLogger.get().info("Retrieving data from train and test splits to compute the metrics")

        if self._test_data is None and validation_set is None:
            msg = "need to call 'test' before computing the metrics"
            raise RcLoggerException(AttributeError, msg)

        not_compute_sensitive = True if self.user_sensitive else False
        not_compute_rating = True if self.rating_dataframe is not None else False

        if not (self.observed_items and self._n_train_interactions is not None and
                (any(self.item_popularity.values()) or self.item_popularity is None) and
                not_compute_sensitive and not_compute_rating):
            train_data = self._train_data
        else:
            train_data = None

        if validation_set is None:
            test_data = self._test_data if self._test_data is not None and not self.unobserved_items else None

            self.observed_items, self.unobserved_items, item_popularity, other_returns = data_utils.get_train_test_features(
                self._user_id_field,
                self._item_id_field,
                train_data=train_data,
                test_or_val_data=test_data,
                rating_field=self._rating_field,
                item_popularity=True,
                sensitive_field=self.sensitive_field,
                other_returns=["len_train_data", "train_rating_dataframe", "test_rating_dataframe", "sensitive"]
            )

            unobserved_items = self.unobserved_items
        else:
            self.observed_items, validation_items, item_popularity, other_returns = data_utils.get_train_test_features(
                self._user_id_field,
                self._item_id_field,
                train_data=train_data,
                test_or_val_data=validation_set,
                rating_field=self._rating_field,
                item_popularity=True,
                sensitive_field=self.sensitive_field,
                other_returns=["len_train_data", "train_rating_dataframe", "test_rating_dataframe", "sensitive"]
            )

            unobserved_items = validation_items

        # 1st any gives False if all item popularity = 0 (self.item_popularity's not been updated)
        if not any(self.item_popularity.values()) and any(item_popularity.values()):
            sum_item_pop = sum(list(item_popularity.values()))
            for item in item_popularity:
                item_popularity[item] /= sum_item_pop
            self.item_popularity = item_popularity

        if "sensitive" in other_returns and other_returns["sensitive"]:
            self.user_sensitive = other_returns["sensitive"]

        predictions = self.get_relevance_matrix(coverage="full", overwrite=overwrite).as_dataframe()

        if "train_rating_dataframe" in other_returns:
            if not other_returns["train_rating_dataframe"].empty:
                self.rating_dataframe = other_returns["train_rating_dataframe"]
                predictions.update(other_returns["train_rating_dataframe"])
                self._full_relevance_matrix._relevances = predictions.values

        if "len_train_data" in other_returns:
            self._n_train_interactions = other_returns["len_train_data"]

        kwargs = {
            "item_popularity": self.item_popularity,
            "len_observed_data": self._n_train_interactions,
            "observed_items": self.observed_items,
            "unobserved_items": unobserved_items,
            "predictions": predictions,
            "sensitive": self.user_sensitive
        }

        if "test_rating_dataframe" in other_returns:
            kwargs.update({'test_rating_dataframe': other_returns["test_rating_dataframe"]})

        self._metric_handler.compute_metrics(metrics, cutoffs=cutoffs, only=only, overwrite=overwrite, **kwargs)
        RcLogger.get().debug(f"Metrics: {self._metric_handler.metrics}")

        return self._metric_handler

    def save_metrics(self, **kwargs):
        self._metric_handler.save(**kwargs)

    def plot(self, *args, kind="scatter", **kwargs):
        kwargs = {"maps": "per_user"} if not kwargs else kwargs

        x = args[0]
        y = args[1] if len(args) > 1 else None

        if hasattr(self._metric_handler.plot, kind):
            getattr(self._metric_handler.plot, kind)(x, y, **kwargs)
        else:
            msg = f"Metrics class does not support {kind} as a plotting kind"
            raise RcLoggerException(ValueError, msg)

    def generate_data(self,
                      train_data: tf.raw_ops.BatchDataset,
                      n_reps=10,
                      shuffle=True,
                      overwrite=False,
                      check_errors: Union[
                          Literal["raise"],
                          Literal["print"],
                          Literal["log_info"],
                          Literal["log_debug"],
                          Callable
                      ] = None):

        gen_funcs = {'triplets': self._generate_triplets, 'binary': self._generate_binary_data}
        check_errors_funcs = {'triplets': self._check_triplets_errors, 'binary': self._check_binary_data_errors}

        self._dataset_metadata = {**self._dataset_metadata, "n_reps": n_reps}

        _batch_size = next(train_data.take(1).as_numpy_iterator())[self._user_id_field].shape[0]

        self._train_data = train_data.unbatch()

        if not data_utils.preprocessed_dataset_exists(self._dataset_metadata,
                                                      model_data_type=self.MODEL_DATA_TYPE,
                                                      split="orig_train"):
            data_utils.save_tf_features_dataset(
                self._train_data.batch(8192),
                self._dataset_metadata,
                dataset_info=self.MODEL_DATA_TYPE,
                overwrite=overwrite,
                split="orig_train"
            )

        # noinspection PyArgumentList
        x = gen_funcs[self.MODEL_DATA_TYPE](_batch_size,
                                            n_repetitions=n_reps,
                                            overwrite=overwrite,
                                            shuffle=shuffle)

        if check_errors is not None:
            # noinspection PyArgumentList
            check_errors_funcs[self.MODEL_DATA_TYPE](x.unbatch(), action=check_errors)

        return x

    def _generate_triplets(self,
                           batch_size,
                           n_repetitions=10,
                           shuffle=True,
                           overwrite=False):
        save_batch = 8192
        shuffle_buffer_size = 100_000
        start_time = time.time()

        if not data_utils.preprocessed_dataset_exists(self._dataset_metadata,
                                                      model_data_type="triplets",
                                                      split="train") or overwrite:
            RcLogger.get().info("Generating triplets")
            self.observed_items, _, _, other_returns = data_utils.get_train_test_features(
                self._user_id_field,
                self._item_id_field,
                train_data=self._train_data,
                other_returns=["len_train_data"]
            )

            data_utils.generate_triplets_data(self._train_data,
                                              self.observed_items,
                                              self.unique_items,
                                              self._dataset_metadata,
                                              n_repetitions=n_repetitions,
                                              user_id_field=self._user_id_field,
                                              item_id_field=self._item_id_field,
                                              len_train_data=other_returns["len_train_data"],
                                              save_batch=save_batch,
                                              overwrite=overwrite,
                                              split="train")
        else:
            RcLogger.get().info("Triplets with this configuration exist. Loading triplets...")

        # dataset is always reloaded for performance improvements
        x = data_utils.load_tf_features_dataset(self._dataset_metadata, dataset_info="triplets", split="train")
        RcLogger.get().info(f"Triplets retrieval time: {time.time() - start_time:.2f}s")

        if shuffle:
            RcLogger.get().debug("Shuffling triplets")
            x = x.shuffle(shuffle_buffer_size)

        return x.batch(batch_size)

    def _check_triplets_errors(self,
                               data: tf.data.Dataset,
                               action: Union[
                                   Literal["raise"],
                                   Literal["print"],
                                   Literal["log_info"],
                                   Literal["log_debug"],
                                   Callable
                               ] = None):

        def check_errors(_data, _action: Callable, observed_items=None, user_field="user_id", item_field="item_id"):
            RcLogger.get().debug("check errors in triplets")

            for el in _data:
                user = el[user_field].numpy()
                pos_item = el[item_field].numpy()
                neg_item = el["negative_item"].numpy()

                if pos_item not in observed_items[user]:
                    msg = f"pos item error: user {user}, item {pos_item}"
                    _action(msg)

                if neg_item in observed_items[user]:
                    msg = f"neg item error: user {user}, item {neg_item}"
                    _action(msg)

        self.observed_items = self.observed_items or data_utils.get_train_test_features(self._user_id_field,
                                                                                        self._item_id_field,
                                                                                        train_data=self._train_data)[0]
        data_utils.check_dataset_errors(data,
                                        check_errors_func=check_errors,
                                        action=action,
                                        observed_items=self.observed_items,
                                        user_field=self._user_id_field,
                                        item_field=self._item_id_field)

    def _generate_binary_data(self,
                              batch_size,
                              n_repetitions=10,
                              shuffle=True,
                              overwrite=False,
                              save_batch=8192):
        shuffle_buffer_size = 100_000
        start_time = time.time()

        if not data_utils.preprocessed_dataset_exists(self._dataset_metadata,
                                                      model_data_type="binary",
                                                      split="train") or overwrite:
            RcLogger.get().info("Generating binary data")
            self.observed_items, _, _, other_returns = data_utils.get_train_test_features(
                self._user_id_field,
                self._item_id_field,
                train_data=self._train_data,
                other_returns=["len_train_data"]
            )

            data_utils.generate_binary_data(self._train_data,
                                            self.observed_items,
                                            self.unique_items,
                                            self._dataset_metadata,
                                            n_repetitions=n_repetitions,
                                            user_id_field=self._user_id_field,
                                            item_id_field=self._item_id_field,
                                            len_train_data=other_returns["len_train_data"],
                                            save_batch=save_batch,
                                            overwrite=overwrite,
                                            split="train")

        else:
            RcLogger.get().info("Binary data with this configuration exist. Loading data...")

        # dataset is always reloaded for performance improvements
        x = data_utils.load_tf_features_dataset(self._dataset_metadata, dataset_info="binary", split="train")
        RcLogger.get().info(f"Binary data retrieval time: {time.time() - start_time:.2f}s")

        if shuffle:
            RcLogger.get().debug("Shuffling binary data")
            x = x.shuffle(shuffle_buffer_size)

        return x.batch(batch_size)

    def _check_binary_data_errors(self,
                                  data: tf.data.Dataset,
                                  action: Union[
                                      Literal["raise"],
                                      Literal["print"],
                                      Literal["log_info"],
                                      Literal["log_debug"],
                                      Callable
                                  ] = None):

        def check_errors(_data, _action: Callable, observed_items=None, user_field="user_id", item_field="item_id"):
            print("check errors in binary data")

            check_observed_items = defaultdict(set)

            for el in _data:
                user = el[user_field].numpy()
                item = el[item_field].numpy()
                label = el["label"].numpy()

                msg = f"item error: user {user}, item {item}, label {label}"

                if (item not in observed_items[user] and label != 0) or (item in observed_items[user] and label != 1):
                    _action(msg)

                if item in check_observed_items[user]:
                    _action(msg + " multiple times label = 1")

                if label == 1:
                    check_observed_items[user].add(item)

        self.observed_items = self.observed_items or data_utils.get_train_test_features(self._user_id_field,
                                                                                        self._item_id_field,
                                                                                        train_data=self._train_data)[0]
        data_utils.check_dataset_errors(data,
                                        check_errors_func=check_errors,
                                        action=action,
                                        observed_items=self.observed_items,
                                        user_field=self._user_id_field,
                                        item_field=self._item_id_field)

    def to_user_oriented_fairness_files(self, sensitive_attribute: Union[Literal["gender"], Literal["age"], str]):
        self._full_relevance_matrix.to_user_oriented_fairness_files(sensitive_attribute,
                                                                    self.user_sensitive,
                                                                    self.name,
                                                                    self.unobserved_items,
                                                                    self.observed_items,
                                                                    self._test_data,
                                                                    self._user_id_field,
                                                                    self._item_id_field)
