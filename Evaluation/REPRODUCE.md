# `metrics_reproduced`

Here this script will be described in terms of arguments to be passed via commmand line, the instructions to modify it
to use your own predictions and what are its functions and outcomes.

## Command Line Arguments

This script has hardcoded metadata to load the right preprocessed dataset, which must be specified as well as other
important parameters. It is easy thanks to pass them via command line thanks to the argument parser inside this script:
- `-dataset`: it is the dataset the metrics should be computed of. You can choose between two values:
    - *movielens_1m*: it starts the evaluation of predictions on MovieLens 1M. The script will use the variables
    `experiments_models_gender_ml1m` (line 65), `experiments_models_age_ml1m` (line 175), which contains the metadata of
    the predictions. The section [Predictions Metadata](#predictions-metadata) thoroughly describe the way these values
    should be modified to use your predictions.
    - *filtered(20)\_lastfm\_1K*: it starts the evaluation of predictions on Last.FM 1K. The script will use the variables
    `experiments_models_gender_lfm1k` (line 285), `experiments_models_age_lfm1k` (line 395), which contains the metadata
    of the predictions. The section [Predictions Metadata](#predictions-metadata) thoroughly describe the way these values
    should be modified to use your predictions.
- `-sensitive_attribute`: this argument can be one of [`Gender`, `Age`], which tells the script to only consider the
   predictions where the mitigations have been applied on gender or age groups, respectively. Setting this parameter to 
   `Gender` will tell the script to use `experiments_models_gender_ml1m` for MovieLens 1M or `experiments_models_gender_lfm1k`
   for Last.FM 1K, while `Age` use `experiments_models_age_ml1m` for MovieLens 1M or `experiments_models_age_lfm1k` for
   Last.FM 1K
- `--only_plot`: setting this will make the script recreate the plots and tables using the files created in previous
   executions and created inside the same folder of the script, without loading or preparing the data for classic execution.
   
Here some examples of usage of this script. The following command must be executed from the root of the repository:

> Evaluation of predictions of MovieLens 1M on gender groups
```shell script
python -m Evaluation.metrics_reproduced -dataset movielens_1m -sensitive_attribute Gender
``` 

> Evaluation of predictions of Last.FM 1K on gender groups
```shell script
python -m Evaluation.metrics_reproduced -dataset filtered(20)_lastfm_1K -sensitive_attribute Gender
``` 

> Evaluation of predictions of Last.FM 1K on age groups
```shell script
python -m Evaluation.metrics_reproduced -dataset filtered(20)_lastfm_1K -sensitive_attribute Age
``` 

> Evaluation of predictions of MovieLens 1M on age groups to only re-create the plots and tables (some tables need the
> file of *results* described in [Functionality](#functionality) of both datasets of a sensitive attribute to be properly
> created)
```shell script
python -m Evaluation.metrics_reproduced -dataset movielens_1m -sensitive_attribute Age --only_plot
``` 

## Predictions Metadata

All of the code of the repository is prepared to create the results according to specific names that the script
`metrics_reproduced` can find automatically. There could still be some little problems related to wrong file names or
paths that need to be adjusted to make everything work. Here it is described the structure of the metadata inside this
script and how it is used to compute the metrics.
After the argument parser, at the top of the script there are 4 variables: `experiments_models_gender_ml1m`,
`experiments_models_age_ml1m`, `experiments_models_gender_lfm1k`, `experiments_models_age_lfm1k`. They have been already
described in above parameters and are related to the predictions of the two datasets and two sensitive_attributes.
Each one of these variables is a list of tuples, where each tuple contains the following values in order:
- `target`: `Ranking` (top-n recommendation) or `Rating` (rating prediction), it is the prediction target,
- `paper`: name or nickname that identify the related paper,
- `model name`: the model to which the fairness approach is applied or the new name of a fairness-aware baseline,
                e.g. ParityLBM, BN-SLIM-U.
- `path`: it can be
    - a full path
    - a list of paths (e.g. Ekstrand et al. experiments contain multiple runs, the paths of the predictions need to be in
      a list)
    - callable that returns paths as a list, even with one only path (not used, but it is supported)
- `baseline`: it can be
    - same as `path` but for the baseline
    - a tuple where the first value is the same for `path` and the second value is the specific name of the baseline,
    (e.g. GCN, SLIM-U), otherwise the string *baseline* will be added at the end of `model name`,
- `function to retrieve data` (OPTIONAL): function to read the file containing the predictions (it should be one of the 
   functions of the class RelevanceMatrix that start with `from_...` inside [models\utils.py](../models/utils.py))
- `function to retrieve baseline data` (OPTIONAL): the same for `function to retrieve data (OPTIONAL)` but for the
   baseline predictions
   
## Functionality

The script will load all the specified predictions of the metadata, according to the dataset and sensitive attribute
selected and for each entry will compute the metrics inside the dictionary `metrics` (line 538), grouped by the `target`
of the model. You can comment or uncomment the names of the metrics that you want the script to compute. The names of the
metrics supported by the codebase can be obtained by importing **metrics.metrics** and calling the static method
`supported_custom_metrics` without any parameter. Some of the metrics are not fully supported, except all of the metrics
proposed by the reproduced papers, which work perfectly. NOTICE: even if `equity_score` (Burke et al) has been implemented,
it is not properly usable, since Last.FM does not have a clear concept of "category" in the data, so it cannot be computed
in this script.
The code will create *results* and *stats* pickle files on the **Evaluation** folder and they are used to store the already
computed metrics, as well as to generate the plots and the latex tables inside the directories present in **Evaluation**,
divided by the considered dataset (each plot or table file will have the analyzed sensitive attribute in the file name).

The script does not re-compute metrics for the papers and models already present in the related *results* file. To
re-compute the metrics you need to delete the *results* and *stats* files of the selected dataset and sensitive attribute.

