# Preprocessing

This folder contains the scripts to preprocess the raw datasets and, once they have been preprocessed, to generate the
input data for each reproduced algorithm.

## Raw Datasets Preprocessing

The two scripts `preprocess_ml1m` and `preprocess_lastfm1K` to preprocess the raw datasets are similar and the proposed
commands are the ones to perfectly re-create the datasets that we used in our experiments. In particular, the dataset
will be splitted in train, validation, test splits with ratio 7:1:2. If a reproduced algorithm does not use a validation
set, then the train set becomes the concatenation of train and validation, but this operation will be performed by the
other script in this folder.

The preprocessing of *MovieLens 1M* is performed using the file `preprocess_ml1m`, and the following command should be executed
from the root of the repository:
```shell script
python -m Preprocessing.preprocess_ml1m -dataset movielens --dataset_size 1m 
 --subdatasets ratings --dataset_split train --users_field user_id --items_field artist_id 
 --dataset_columns user_id movie_id user_rating user_gender bucketized_user_age --rating_field user_rating 
 --sensitive_field user_gender --train_val_test_split_type per_user_timestamp --train_val_test_split 70% 10% 20%
 --attribute_to_binary bucketized_user_age --binary_le_delimiter 34 -model Pointwise --n_reps 2
 --overwrite_preprocessed_dataset
```

The preprocessing of *Last.FM 1K* is performed using the file `preprocess_lastfm1K`, and the following command should be
executed from the root of the repository:
```shell script
python -m Preprocessing.preprocess_lastfm1K -dataset lastfm --dataset_size 1K 
 --subdatasets plays users --dataset_split train train --users_field user_id --items_field artist_id 
 --dataset_columns user_id artist_id plays - user_id user_gender user_age --rating_field user_rating 
 --sensitive_field user_gender --train_val_test_split_type per_user_random --train_val_test_split 70% 10% 20%
 --min_interactions 20 --attribute_to_binary user_age --binary_le_delimiter 24 -model Pointwise --n_reps 2
 --overwrite_preprocessed_dataset
```

*Last.FM 1K* is not part of [TensorFlow Datasets](https://www.tensorflow.org/datasets) catalog, so if TensorFlow Datasets
library does not do it automatically, it could be necessary to execute the following command inside the directory
**data/datasets/lastfm**:
```shell script
tfds build
```
which will download the dataset from the original website and prepare it for preprocessing.

The commands are not written in one line to better visualize them, but the "newline" at the end of each line should be
deleted and the command should be executed copying it "one-line" form.

## Input Data Generation

Once the datasets have been preprocessed, you can create the input data necessary for each reproduced algorithm. The input
data will be saved in a folder with the same name of the folder inside **reproducibility\_study** that contains the related
algorithm of the reproduced paper. Before executing this script some variables could be necessary to be changed because
of different filenames. This is the case of the variable `kwargs` at the line 62 for MovieLens 1M and line 76 for Last.FM 1K:
- `model_paths`: this could be necessary to be modified depending on the filename of the predictions on the test set for
                 each model. So it could be possible that you need to modify, for instance, the path of the predictions
                 on the test set for *PMF* at the key `'PMF'` of this internal dictionary using the value of your own
                 predictions for the same model for MovieLens 1M (`kwargs` at line 62) or Last.FM 1K (`kwargs` ar line 76)
- `test_path`: the path of the test set used to evaluate the models of **NLR** codebase of *Li et al* (this should not
               change if you do not rename the file name of the generated test set)
               
If all the paths are correct you can generate the input data by executing the following commands from the root of the
repository:

> MovieLens 1M Gender
```shell script
python -m Preprocessing.generate_input_data -dataset movielens 
 --dataset_size 1m -model Pointwise --n_reps 2 --train_val_test_split_type per_user_timestamp
 --train_val_test_split 70% 10% 20% --users_field user_id --items_field movie_id --rating_field user_rating
 --sensitive_field user_gender
```

> MovieLens 1M Age
```shell script
python -m Preprocessing.generate_input_data -dataset movielens 
 --dataset_size 1m -model Pointwise --n_reps 2 --train_val_test_split_type per_user_timestamp
 --train_val_test_split 70% 10% 20% --users_field user_id --items_field movie_id --rating_field user_rating
 --sensitive_field bucketized_user_age
```

> Last.FM 1K Gender
```shell script
python -m Preprocessing.generate_input_data -dataset lastfm --dataset_size 1K
 -model Pointwise --n_reps 2 --min_interactions 20 --train_val_test_split_type per_user_random 
 --train_val_test_split 70% 10% 20% --users_field user_id --items_field artist_id --rating_field user_rating
 --sensitive_field user_gender
```

> Last.FM 1K Age
```shell script
python -m Preprocessing.generate_input_data -dataset lastfm --dataset_size 1K
 -model Pointwise --n_reps 2 --min_interactions 20 --train_val_test_split_type per_user_random 
 --train_val_test_split 70% 10% 20% --users_field user_id --items_field artist_id --rating_field user_rating
 --sensitive_field user_age
```

Here we explain some of the parameters passed to this script to better understand their behaviour:
- `-dataset`: it is the name of the dataset. It can be one of [`movielens`, `lastfm`].
- `--dataset_size`: it is the size of the dataset. For MovieLens it could be `100k` or `1m`. For Last.FM it could be
                    `1K` or `360K`
- `--sensitive_field`: it is the sensitive field of the dataset that will be used to create files that contain sensitive
                       information and that are necessary for some of the reproduced algorithms.
- `-model`: it is `Pointwise` because `NLR` codebase of *Li et al* use pointwise data to train their models. The parameter
            `--n_reps` specify the number of times the train data should be repeated with negative labels. Setting it to
            2 will generate 1 negative sample for each interaction in the train set (it will be performed also for
            the validation set for the `NLR` codebase of *Li et al*)

However, these commands are not yet complete because we still need to specify the data of which reproduced paper we want
to generate. This can be done adding one or more of the following parameters that can be also found inside the function
`_add_other_features_args` of the Argument Parser in [helpers/recsys_arg_parser.py](../helpers/recsys_arg_parser.py):
- `--create_user_oriented_fairness_files_input_data`: it creates the input data for the fairness-aware post-processing
  method of *Li et al*, which will be saved in the folder **reproducibility_study/Preprocessing/input_data/DATASET/Li et al**
- `--create_nlr_input_data`: it creates the input data for the `NLR` codebase of *Li et al*, which will be saved in
  the folder **reproducibility_study/Preprocessing/input_data/DATASET/Li et al/NLR**
- `--create_co_clustering_for_fair_input_data`: it creates the input data for the reproduced algorithm of *Frisch et al*,
  which will be saved in the folder **reproducibility_study/Preprocessing/input_data/DATASET/Frisch et al**
- `--create_fairgo_input_data`: it creates the input data for the reproduced algorithm of *Wu et al*,
  which will be saved in the folder **reproducibility_study/Preprocessing/input_data/DATASET/Wu et al**
- `--create_all_the_cool_kids_input_data`: it creates the input data for the reproduced algorithm of *Ekstrand et al*,
  which will be saved in the folder **reproducibility_study/Preprocessing/input_data/DATASET/Ekstrand et al**
- `--create_rec_independence_input_data`: it creates the input data for the reproduced algorithm of *Kamishima et al*,
  which will be saved in the folder **reproducibility_study/Preprocessing/input_data/DATASET/Kamishima et al**
- `--create_antidote_data_input_data`: it creates the input data for the reproduced algorithm of *Rastegarpanah et al*,
  which will be saved in the folder **reproducibility_study/Preprocessing/input_data/DATASET/Rastegarpanah et al**
- `--create_librec_auto_input_data`: it creates the input data for the reproduced algorithm of *Burke et al*,
  which will be saved in the folder **reproducibility_study/Preprocessing/input_data/DATASET/Burke et al**
- `--create_rating_prediction_fairness_input_data`: it creates the input data for the reproduced algorithm of *Ashokan and Haas*,
  which will be saved in the folder **reproducibility_study/Preprocessing/input_data/DATASET/Ashokan and Haas**
  
The term **DATASET** inside the paths will be one of [`movielens_1m`, `filtered(20)_lastfm_1K`].

As a demonstration, to create the input data for *Ekstrand et al* and *Burke et al* for Last.FM 1K for gender groups the
following command must be executed from the root of the repository:
```shell script
python -m Preprocessing.generate_input_data -dataset lastfm --dataset_size 1K
 -model Pointwise --n_reps 2 --min_interactions 20 --train_val_test_split_type per_user_random 
 --train_val_test_split 70% 10% 20% --users_field user_id --items_field artist_id --rating_field user_rating
 --sensitive_field user_gender --create_all_the_cool_kids_input_data --create_librec_auto_input_data
```
