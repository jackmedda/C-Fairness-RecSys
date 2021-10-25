# Ashokan and Haas. [Fairness metrics and bias mitigation strategies for rating predictions](https://doi.org/10.1016/j.ipm.2021.102646)
The original code is property of *Ashwathy Ashokan* and *Christian Haas*, as well as the work proposed in their paper.
We thank them for providing this code.

## 1. Requirements
There wasn't a `requirements.txt` file in the source code, but Python 3 is used and the necessary libraries have been mentioned inside the
[README.txt](../README.txt) file of the original code:
- Pandas
- Numpy
- Lenskit

We added the file [requirements.txt](requirements.txt) for a direct installation of required libraries. The dependencies can be installed with the following command in Windows:
```shell script
pip install -r requirements.txt
```
or the following one in Linux:
```shell script
pip3 install -r requirements.txt
```

## 2. Original code and Modifications
The original code was not available in a public repository, so we contacted the authors, who sent us the archive `RatingPredictionFairness.zip`,
which is present in this folder.

Only the file `Main.py` has been modified in order to load custom input files and to save the predictions before and after the mitigation procedure.
In particular the modifications are present in two ranges of lines:
- 98\-113: code that loads input data and modifies the values of `partitions`, `privileged_users`, `unprivileged_users` used in the code.
- 185\-196: save the original, value adjusted and parity adjusted prediction, mapped with the key `algo_post_text` inside the dictionary `algo_post`.
            The keyword `continue` in line 194 (added to improve execution time) can be removed to let the code compute the metrics.

## 3. Input Data Preparation and Hyper-parameters Setting
The input data can be generated with the script `generate_input_data.py` inside **/reproducibility_study/Preprocessing** by using the generation commands
inside the same file or the [REPRODUCE.md](../../Preprocessing/REPRODUCE.md) file inside the same folder. Once you specified the selected metadata
for the dataset and sensitive attribute, you need to add the argument `--create_rating_prediction_fairness_input_data` to generate the input files
for this paper. You must copy the files from the related directory and copy them inside this directory.
To select which input data to use you need to modify the following variables:
- `sensitive_attribute` (line 100): it is the attribute used to generate the input data and it is also present at the end of the input filename.
                                    NOTICE: this source code works with the field `gender`. To reduce the modifications the selected sensitive attribute 
									will be always mapped to the field `gender`, even if it contains the labels related to age groups. Users labelled
									with `M` are the users of the majority group (male, young) that in the pre-processed datasets are identified with `True`,
									while `F` are the users of the minority group (female, old) that in the pre-processed datasets are identified with `False`.
- `_dataset` (line 101): dataset used to generate the input data. It is also present at the beginnning of the input filename.

In `Main.py` we set the maximum number of neighbors of ItemKNN to 20 and the number of features of BiasedMF to 50.

## 4. Code Execution
Once the variables to choose the input data and the sensitive attribute to consider have been set to the right value, the code can be executed with
the following command inside this folder in Windows:
```shell script
python Main.py
```
or the following one in Linux:
```shell script
python3 Main.py
```

## 5. Predictions Extraction
The prediction files will be saved inside the directory `results` and they will be named according to the selected dataset, sensitive attribute and the 
denomination of the paper:
- `_orig`: files containing this string are the baseline of the model preceding this string.
- `_val_adj`: files containing this string identify the predictions adjusted with *value-based fairness*
- `_parity_adj`: files containing this string identify the predictions adjusted with *parity-based fairness*

The metrics can be computed by adding the filepath of the results to `metrics_reproduced.py` inside **/reproducibility_study/Evaluation** and following
the instruction inside the [REPRODUCE.md](../../Evaluation/REPRODUCE.md) file present in the same folder.

## 6. Further Notes
Nothing relevant to be mentioned.

# Citation
Ashokan, A., Haas, C.: Fairness metrics and bias mitigation strategies for rating predictions. Inf. Process. Manag.58(5), 102646 (2021).
https://doi.org/10.1016/j.ipm.2021.102646, https://doi.org/10.1016/j.ipm.2021.1026464.
