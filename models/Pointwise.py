from typing import Dict, Text, Any, Sequence

import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np

import rc_types
import data.utils as data_utils
from .Model import Model


class Pointwise(Model):

    MODEL_DATA_TYPE = "binary"

    def __init__(self,
                 model_data: rc_types.ModelData,
                 dataset_metadata: Dict[Text, Any],
                 name="Pointwise",
                 optimizer="Adam",
                 learning_rate=0.001,
                 layers=None,
                 **kwargs):

        user_embedding_kwargs = kwargs.pop("user_embedding_kwargs") if "user_embedding_kwargs" in kwargs else {}
        item_embedding_kwargs = kwargs.pop("item_embedding_kwargs") if "item_embedding_kwargs" in kwargs else {}

        super(Pointwise, self).__init__(model_data,
                                        dataset_metadata,
                                        name=name,
                                        optimizer=optimizer,
                                        learning_rate=learning_rate,
                                        user_embedding_kwargs=user_embedding_kwargs.update({'input_length': 1}),
                                        item_embedding_kwargs=item_embedding_kwargs.update({'input_length': 1}),
                                        **kwargs)

        layers = [64, 32, 16, 8] if layers is None else layers
        self._layers_sizes = layers

        self._pointwise_mlp_emb_user = tf.keras.Sequential([
            self.user_model.get_layer(index=0),
            tf.keras.layers.Embedding(len(self.unique_users), int(layers[0]//2), name="mlp_embedding_user", input_length=1)
        ], name="mlp_user_model")
        self._pointwise_mlp_emb_item = tf.keras.Sequential([
            self.item_model.get_layer(index=0),
            tf.keras.layers.Embedding(len(self.unique_items), int(layers[0]//2), name="mlp_embedding_item", input_length=1)
        ], name="mlp_item_model")

        self._pointwise_mlp_layers_model = tf.keras.Sequential(name="mlp_layers_model")
        for i, _l in enumerate(layers[1:]):
            self._pointwise_mlp_layers_model.add(tf.keras.layers.Dense(_l, activation='relu', name=f'layer{i}'))

        self._pointwise_prediction = tf.keras.layers.Dense(1, activation='sigmoid', name="prediction")

        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.BinaryCrossentropy() #,
            # metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy")]
        )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        if training:
            _features, labels = features, features["label"]
        else:
            _features, labels = features[0], features[1]

        out = self(_features, training=training)

        return self.task(labels, out)

    def call(self, inputs, training=None, mask=None):
        mf_user_latent = tf.keras.layers.Flatten()(self.user_model(inputs[self._user_id_field]))
        mf_item_latent = tf.keras.layers.Flatten()(self.item_model(inputs[self._item_id_field]))
        mf_vector = tf.keras.layers.Multiply()([mf_user_latent, mf_item_latent])

        mlp_user_latent = tf.keras.layers.Flatten()(self._pointwise_mlp_emb_user(inputs[self._user_id_field]))
        mlp_item_latent = tf.keras.layers.Flatten()(self._pointwise_mlp_emb_item(inputs[self._item_id_field]))
        mlp_vector = tf.keras.layers.Concatenate()([mlp_user_latent, mlp_item_latent])
        mlp_vector = self._pointwise_mlp_layers_model(mlp_vector)

        predict_vector = tf.keras.layers.Concatenate()([mf_vector, mlp_vector])
        prediction = self._pointwise_prediction(predict_vector)

        return prediction

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

        x = x.prefetch(1024)

        return super(Pointwise, self).train(x=x,
                                            y=y,
                                            batch_size=batch_size,
                                            epochs=epochs,
                                            verbose=verbose,
                                            callbacks=self._callbacks,
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

        def _test_add_labels(data):
            labels = data.map(lambda el: tf.constant([1], dtype=tf.int32))
            return tf.data.Dataset.zip((data, labels))

        if x is None:
            _batch_size = batch_size
            x = self._test_data
            x = _test_add_labels(x)
        else:
            _batch_size = next(x.take(1).as_numpy_iterator())[self._user_id_field].shape[0]
            self._test_data = x.unbatch()
            x = _test_add_labels(self._test_data)

        x = x.batch(_batch_size)

        return super(Pointwise, self).test(x=x,
                                           y=y,
                                           batch_size=None,
                                           verbose=verbose,
                                           sample_weight=sample_weight,
                                           steps=steps,
                                           callbacks=callbacks,
                                           max_queue_size=max_queue_size,
                                           workers=workers,
                                           use_multiprocessing=use_multiprocessing,
                                           return_dict=True)

    def get_predictions(self):
        predictions = []
        for user_id in self.unique_users:
            user_data = np.array([user_id] * len(self.unique_items))
            item_data = np.array(self.unique_items)
            predictions.append(np.squeeze(self({
                self._user_id_field: user_data,
                self._item_id_field: item_data
            }, training=False).numpy()))

        return predictions
