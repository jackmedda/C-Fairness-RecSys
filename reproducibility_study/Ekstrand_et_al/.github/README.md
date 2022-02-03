# Ekstrand et al. [All the cool kids, how do they fit in?: Popularity and demographicbiases in recommender evaluation and effectiveness.](http://proceedings.mlr.press/v81/ekstrand18b.html)
The original code is property of *Michael D. Ekstrand* and *Mucun Tian* and *Ion Madrazo Azpiazu* and *Jennifer D. Ekstrand* and *Oghenemaro Anuyah* and *David McNeill*
and *Maria Soledad Pera*, as well as the work proposed in their paper. We thank them for providing this code.

## 1. Requirements
The source code is structured as a black box that must be executed with a script. The softwares and libraries required for execution are listed and
described in the `README.md` file in this directory. We will list them here for consistency:
- Java (we installed the [OpenJDK 8](https://www.openlogic.com/openjdk-downloads))
- R (we installed it from the [R website](https://cran.r-project.org/bin/))
- Jupyter
- Tidyverse
- MASS
- plyr

Jupyter can be installed using `pip` with the following command in Windows on a Python 3 system:
```shell script
pip install jupyter
```
or the following one in Linux:
```shell script
pip3 install jupyter
```

The other libraries have been installed using `R`. The [README.md](README.md) file of the authors recommend using Anaconda,
but we cannot give hints on this procedure, since our study did not depend on conda.

## 2. Original code and Modifications
The original code is linked in the paper and available in a [public repository](https://scholarworks.boisestate.edu/cs_scripts/4/).

We modified the main file, which is [build.gradle](build.gradle), by commenting the section that use a 5-fold cross-validation of `evaluateML` task and adding the lines to use
custom train and test files, and modified the `predict` and `recommend` task to also save the predictions and not only the computed metrics. We created a new task
similar to `evaluateML`, named it `evaluateLFM1KReproduced` and modified it to prepare it for input data of Last.FM 1K.

## 3. Input Data Preparation and Hyper-parameters Setting
The input data can be generated with the script `generate_input_data.py` inside **/reproducibility_study/Preprocessing** by using the generation commands
inside the same file or the [REPRODUCE.md](../../Preprocessing/REPRODUCE.md) file inside the same folder. Once you specified the selected metadata for the dataset and sensitive attribute,
you need to add the argument `--create_all_the_cool_kids_input_data` to generate the input files for this paper. You must copy the files from the
related directory and copy them inside the folder `custom_data` that you can find in this directory.
To select which input data to use you need to uncomment/comment the lines of the `evaluateML` task (if you want to execute an experiment for MovieLens 1M)
or the `evaluateLFM1KReproduced` task (if you want to execute an experiment for Last.FM 1K) that have have as a first comment a string that start as `reproduced custom XXXX data`,
where `XXXX` can be:
- `baseline`: to execute the experiment for the baselines, so without any balancing of the input data
- `gender-balanced`: to execute the experiment with input data where users are balanced by gender attribute
- `age-balanced`: to execute the experiment with input data where users are balanced by age attribute

For instance, to execute `gender-balanced` experiment you need to **uncomment** the 4 lines after `reproduced custom gender-balanced data`, and the 4 lines under 
`reproduced custom baseline data` and `reproduced custom age-balanced data` should be **commented** respectively because the experiments are mutually exclusive.

For MovieLens 1M from each gender or age group 1500 users are randomly sampled (as stated in the paper) and for Last.FM 1K 100 users for each gender or age group.

The hyper-parameters for each algorithm can be modified in [algorithms](algorithms/ml) folder, but only the ones inside `ml` folder, since also the `evaluateLFM1KReproduced` task
uses the algorithms prepared for MovieLens 1M.

The hyper-parameters used for each model are the followings:
- **Top-N Recommendation**
    - *ItemKNN (Implicit)*
        - maximum number of neighbors: 40
    - *UserKNN (Implicit)*
        - maximum number of neighbors: 40
    - *SVD (Implicit)*
        - features: 60
        - iterations: 150
- **Rating Prediction**
    - *ItemKNN*
        - maximum number of neighbors: (MovieLens 1M: 20, Last.FM 1K: 40)
    - *UserKNN*
        - maximum number of neighbors: (MovieLens 1M: 40, Last.FM 1K: 20)
    - *SVD*
        - features: 60
        - iterations: (MovieLens 1M: 200, Last.FM 1K: 150)

## 4. Code Execution
Once the input data files have been copied inside the folder `custom_data` that you can find in this directory, you can execute the code inside the file 
[train.py](train.py) which is not part of the original dataset of *Ekstrand et al*. This code automatically creates the
input csv files pointed by the `yaml` files that you copied inside the folder `custom_data`, saves them in `custom_data`, execute the source code of this paper
using the input csv files and saves the predictions grouped by model inside the folder `results`. The folder `results` is present in this directory and
structued as **DATASET/XXXX**, with **DATASET** being `movielens_1m` or `filtered(20)_lastfm_1K` and `XXXX` one of the list described in
[Section 3](#3-input-data-preparation-and-hyper-parameters-setting) of this document.

The script to execute contains different values that must be modified to change the dataset or the experiment you want to perform:
- `overwrite` (line 92): must be always `True` or the baseline will not be re-computed (used for other purposes, not actually for the reproducibility)
- `experiment` (line 93\-94): if is `baseline` will compute the task `baseline`. If it is `ekstrand` will balance the train and test based on the selected sensitive attribute
- `dataset` (line 96\-97): the dataset to be used. Choose line 96 for MovieLens 1M or line 97 for Last.FM 1K
- `use_external_code` (line 99): should always be set to `lenskit_groovy`
- `n_experiments` (line 102): the number of experiments to repeat the random sampling for the balancing approach (only used if experiment = `ekstrand`)
- `sensitive_attribute` (line 127\-128/142\-143): sensitive field used to balance the users representation.

Once you have selected the values for the above variables you can execute the code from the root directory of this repository with the following command in Windows:
```shell script
python -m reproducibility_study.Ekstrand_et_al.train
```
or the following one in Linux:
```shell script
python3 -m reproducibility_study.Ekstrand_et_al.train
```

## 5. Predictions Extraction
The prediction files will be saved inside the folder `results` in this directory, and, in particular, will be saved following the respective dataset and task used,
as previously explained in [Section 4](#4-code-execution). For instance, if you execute an experiment for MovieLens 1M and `gender-balanced` task, the outputs will be
saved in **results/movielens_1m/gender_balanced** directory.

The metrics can be computed by adding the filepath of the results to `metrics_reproduced.py` inside **/reproducibility_study/Evaluation** and following
the instruction inside the [REPRODUCE.md](../../Evaluation/REPRODUCE.md) file present in the same folder. In particular, the results contain different predictions and recommendation based on the
trained model or the run of the experiments in case of a balanced setup. For the baselines only the correct filepath must be added, while the several predictions and
recommendations files of each specific run must be added together in a list, by specifying the filepath for each run manually or loading them easily with `os.scandir()`.

## 6. Further Notes
Nothing relevant to be mentioned.

# Citation
Ekstrand, M.D., Tian, M., Azpiazu, I.M., Ekstrand, J.D., Anuyah, O., McNeill,D., Pera, M.S.: All the cool kids, how do they fit in?: Popularity and demographicbiases in
recommender evaluation and effectiveness. In: Friedler, S.A., Wilson, C.(eds.) Conference on Fairness, Accountability and Transparency, FAT 2018, 23-24 February 2018,
New York, NY, USA. Proceedings of Machine Learning Re-search, vol. 81, pp. 172–186. PMLR (2018), http://proceedings.mlr.press/v81/ekstrand18b.html