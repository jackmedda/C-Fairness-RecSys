# Frisch et al. [Co-clustering for fair recommendation](https://hal.archives-ouvertes.fr/hal-03239856)
The original code is property of *Gabriel Frisch* and *Jean-Benoist Leger* and *Yves Grandvalet*, as well as the work proposed in their paper.
We thank them for providing this code.

## 1. Requirements
The source code contains a `requirements.txt` file that can be used to install the dependencies. We modified only the versione of `scikit-learn` because
the present version could not be installed.
The dependencies can be installed with the following command in Windows on Python 3:
```shell script
pip install -r requirements.txt
```
or the following one in Linux:
```shell script
pip3 install -r requirements.txt
```
## 2. Original code and Modifications
The original code was not available in a public repository, so we contacted the authors, who sent us the archive `parityLBM_SOURCE_CODE2.zip`,
which is present in this folder. The code was not perfectly set to also compute the baseline in an easy way, so the authors sent us also the file `lbm_ordinal2.py`,
which can be used without the mitigation procedure if no covariates are passed to the model, but we used `lbm_ordinal.py` anyway for the mitigation procedure and
`lbm_ordinal2.py` only for the baseline. This is done for consistency to maintain the setup near as equal to the original source code.

We only modified the file `start_experiments.py` by commenting different parts of the code, as the metrics computation (NDCG, Chi-square) and the lines that loads
and process the test data in order to reduce the memory requirement by avoiding the loading of the test set. We also set the `callback` parameter of the model
to `None`, since the callback computes the Chi-square, which increases even more the memory impact. As other source codes, we modified the input loading and the output
saving sections to better load and save the files based on the dataset name and sensitive attribute selected.

## 3. Input Data Preparation and Hyper-parameters Setting
The input data can be generated with the script `generate_input_data.py` inside **/reproducibility_study/Preprocessing** by using the generation commands
inside the same file or the [REPRODUCE.md](../../Preprocessing/REPRODUCE.md) file inside the same folder. Once you specified the selected metadata for the dataset and sensitive attribute,
you need to add the argument `--create_co_clustering_for_fair_input_data` to generate the input files for this paper. You must copy the files from the
related directory and copy them inside this directory `data` that you can found in this folder.
To select which input data to use you need to modify the following variables:
- `sensitive_attribute` (line 44): it is the attribute used to generate the input data and it is also present at the end of the input filename.
                                    NOTICE: this source code works with the field `gender`. To reduce the modifications the selected sensitive attribute 
									will be always mapped to the field `gender`, even if it contains the labels related to age groups. Users labelled
									with `M` are the users of the majority group (male, young) that in the pre-processed datasets are identified with `True`,
									while `F` are the users of the minority group (female, old) that in the pre-processed datasets are identified with `False`.
- `dataset` (line 45): dataset used to generate the input data. It is also present at the beginning of the input filename.
- `covariates` (line 90): 

We set the number of iterations to 25 to repeat the training process of 300 epochs and use the following hyper-parameters:
- **MovieLens 1M**
    - nq (number of row groups): 25
    - nl (number of column groups): 25
    - learning rate: 0.02
- **Last.FM 1K**
    - nq (number of row groups): 15
    - nl (number of column groups): 25
    - learning rate: 0.05

## 4. Code Execution
Once the variables to choose the input data and the sensitive attribute to consider have been set to the right value, the code can be executed with
the following command inside this folder in Windows:
```shell script
python start_experiments.py
```
or the following one in Linux:
```shell script
python3 start_experiments.py
```

## 5. Predictions Extraction
The prediction files will be saved inside the directory `results` and they will be named according to the selected dataset and sensitive attribute.

The metrics can be computed by adding the filepath of the results to `metrics_reproduced.py` inside **/reproducibility_study/Evaluation** and following
the instruction inside the [REPRODUCE.md](../../Evaluation/REPRODUCE.md) file present in the same folder.

## 6. Further Notes
Nothing relevant to be mentioned.

# Citation
Frisch, G., Leger, J.B., Grandvalet, Y.: Co-clustering  for  fair  recommendation(2021), https://hal.archives-ouvertes.fr/hal-03239856