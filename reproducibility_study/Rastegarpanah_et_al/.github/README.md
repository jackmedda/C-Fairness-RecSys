# Rastegarpanah et al. [Fighting Fire with Fire: Using Antidote Data to Improve Polarization and Fairness of Recommender Systems](https://doi.org/10.1145/3289600.3291002)
The original code is property of *Bashir Rastegarpanah* and *Krishna P. Gummadi* and *Mark Crovella*, as well as the work proposed in their paper.
We thank them for providing this code.

## 1. Requirements
There wasn't a `requirements.txt` file in the source code, so we added a `requirements.txt` for this source code that works under Python 2.
The dependencies can be installed with the following command in Windows and Linux:
```shell script
pip install -r requirements.txt
```

## 2. Original code and Modifications
The source code was not linked in the paper, but it is available in a [public repository](https://github.com/rastegarpanah/antidote-data-framework).

Modifications have been necessary to load custom input data and to save the predictions before and after the mitigation procedure. In particular,
our goal is the mitigation of group unfairness, but the codebase only contains an example of improvement of the polarization. We used the file
[minimze_polarization.py](minimze_polarization.py) as a guide to create the file [minimize_group_unfairness.py](minimize_group_unfairness.py),
which is not provided in the original repository. So, we made use of the utility `group_loss_variance` and adjusted the code
to improve the fairness based on this utility.

## 3. Input Data Preparation and Hyper-parameters Setting
The input data can be generated with the script `generate_input_data.py` inside **/reproducibility_study/Preprocessing** by using the generation commands
inside the same file or the [REPRODUCE.md](../../Preprocessing/REPRODUCE.md) file inside the same folder. Once you specified the selected metadata for the dataset and sensitive attribute,
you need to add the argument `--create_antidote_data_input_data` to generate the input files for this paper. You must copy the files from the
related directory and copy them inside the **Data** folder inside this directory.
To select which input data to use you need to modify the following variables:
- `dataset` (line 16\-17): it is the dataset that you select for the experiment and it represents also the beginning of the input filename
- `sensitive_attribute` (line 20\-21/23\-24): sensitive attribute to consider in the mitigation procedure and it represented the end of the input filename

For the ALS baseline we used the following hyper-parameters:
- rank: 4
- &#955;<sub>1</sub> (lambda): (MovieLens 1M: 0.1, Last.FM 1K: 10)

For "group loss variance" utility the parameters values that we used are:
- runs: 10
- budget percentage: (MovieLens 1M: 1.0, Last.FM 1K: 10.0)

## 4. Code Execution
Once the variables to select the input data and the sensitive attribute to consider have been set to the right value, the code can be executed with
the following command inside this folder in Windows and Linux:
```shell script
python minimize_group_unfairness.py
```

## 5. Predictions Extraction
The prediction files will be saved in this directory. The predicionts of the baselines will have the filename ending with `baseline`, while the predictions
after the mitigation procedure will have the filename ending with the selected sensitive attribute.

The metrics can be computed by adding the filepath of the results to `metrics_reproduced.py` inside **/reproducibility_study/Evaluation** and following
the instruction inside the [REPRODUCE.md](../../Evaluation/REPRODUCE.md) file present in the same folder.

## 6. Further Notes
Nothing relevant to be mentioned.

# Citation
Rastegarpanah, B., Gummadi, K.P.  Crovella, M.: Fighting fire with fire: Using antidote data to improve polarization and fairness of recommender systems.
In: Proceedings of the Twelfth ACM International Conference on WebSearch and Data Mining, WSDM 2019, Melbourne, VIC, Australia, February 11\-15, 2019.
pp. 231–239. ACM (2019). https://doi.org/10.1145/3289600.3291002, https://doi.org/10.1145/3289600.3291002
