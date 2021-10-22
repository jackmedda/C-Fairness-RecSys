import os
import inspect

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(inspect.getsourcefile(lambda: 0)), os.pardir, 'data'))
TIME_FORMATTER = "%Y_%m_%d_%H-%M-%S"
LOG_FILE_FORMATTER = "[{asctime}] {levelname}: {message}"
LOG_STREAM_FORMATTER = "{levelname}:{name} > {message}"
LOG_ERROR_FORMATTER = "|{exc}| {msg}"
PREFIX_FILTERED_DATASET = "filtered({min_interactions})_"
PREFIX_BALANCED_DATASET = "{balance_attribute}_balanced({balance_ratio})_"
PREFIX_SAMPLED_DATASET = "sampled({sample_n} {sample_attribute})_"

SAVE_MODELS_PATH = os.path.join(BASE_PATH, 'models')
SAVE_MODELS_CHECKPOINTS_PATH = os.path.join(BASE_PATH, 'models', 'checkpoints')
SAVE_RELEVANCE_MATRIX_PATH = os.path.join(BASE_PATH, 'relevance_matrices')
SAVE_INDEXES_PATH = os.path.join(BASE_PATH, 'indexes')
SAVE_METRICS_PATH = os.path.join(BASE_PATH, 'metrics')
SAVE_PRE_PROCESSED_DATASETS_PATH = os.path.join(BASE_PATH, 'preprocessed_datasets')
SAVE_USER_ORIENTED_FAIRNESS_PATH = os.path.join(BASE_PATH, 'user-oriented fairness files')
LOGGER_NAME = 'recommender_codebase'
LOGS_PATH = os.path.join(BASE_PATH, 'logs')

REPRODUCIBILITY_STUDY_PATH = os.path.join(os.path.dirname(BASE_PATH), 'reproducibility_study')
INPUT_DATA_REPRODUCIBILITY = os.path.join(REPRODUCIBILITY_STUDY_PATH, 'Preprocessing', 'input_data')
