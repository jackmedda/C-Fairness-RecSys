# recommender_codebase

## Code Folders Structure

### data
This folder contains one script `utils.py`, which handles all the data manipulation (dataset loading,
attributes mapping, data filtering, train-test splitting, data serialization and saving), and several subfolders
to store different types of data computed during the framework routines. These subfolders are described in
[Data Folders Structure](#data-folders-structure).  

The type of data *recommender_codebase* works with: the entire framework is based on datasets with a structure similar to
[Features Dictionaries](https://www.tensorflow.org/datasets/api_docs/python/tfds/features/FeaturesDict).
*recommender_codebase* expects dictionaries of tensors, whose structure in a TensorFlow Dataset is:
> <Dataset shapes: {a: (), b: ()}, types: {a: tf.int32, b: tf.int32}>  

This is a powerful structure because maps each data tensor to a key, i.e. an attribute of our dataset, and operations
can be applied without losing dictionary form, in fact obtaining interesting results, such as `batch` op:
> <BatchDataset shapes: {a: (None,), b: (None,)}, types: {a: tf.int32, b: tf.int32}>

### helpers  
This folder is used to store scripts which manage parts of secondary importance of the framework. Currently it contains
5 scripts, amongst which are `constants.py` which contains all constants (paths, parameters) used in the framework,
`filename_utils.py` which handles run id creation and default filenames, `logger.py` which manages logging
functionalities of executions and `recsys_arg_parser.py` which lists all command line parameters the framework is
able to parse.

### metrics
This folder is used to store scripts related to metrics computation, handling and plotting funcionalities. The main
script is `metrics.py`, which contains the class `Metrics` that computes, stores and saves all metrics, and the class
PlotAccessor to easily create plots on the basis of the values of the metrics as functions of other parameters.

### models
This folder is used to store all scripts concerning the models, hence it contains the father class `Model.py` and it is
intended to store all child recommender systems to be integrated in the framework. `Model.py` handles all main
attributes, actions and parameters to train, test and compute the metrics of a recommender systems, with saving and
loading functionalities. `Model.py` can be used as the simplest recommender system in the framework, since it follows
the TFRS [basic retrieval tutorial](https://www.tensorflow.org/recommenders/examples/basic_retrieval).

### rc_types
Simple `__init__` to import and re-export types created in the codebase, such as classes, enums etc. It can be useful to
add type to parameters or return types.

## `data` Folders Structure

> ***datasets***  
> Its purpose is to store TensorFlow custom datasets

> ***logs***  
> Its purpose is to store logs of the application with different log levels to record every step of the code executions

> ***preprocessed_datasets***  
> Its purpose is to store the datasets preprocessed to train particular recommender systems. `data.utils` preprocess
> datasets in time-consuming ways, which can be saved with the `FeaturesDictTensorsWriter` and loaded with the
> `FeaturesDictTensorsReader`. The functions `save_tf_features_dataset`, `load_tf_features_dataset` of `data/utils.py`
> can be used to easily interact with the writer and the reader.

> ***relevance_matrices***  
> Its purpose is to store the relevance matrices. Currently two types of storing are supported:
> * **csv**: by means of a pandas DataFrame with items as header, users as index and predictions as values.
> Currently, output file extension is only '.csv'.
> * **numpy**: by means of NumPy and two styles are currently supported,
>     * *matrix*: only the predictions are saved without information about users and items.
>     * *arrays*: users, items and predictions are saved as distinct arrays.

Each file saved during a run contains a random id in its filename to associate saved models, metrics and other data
together to the same run.
