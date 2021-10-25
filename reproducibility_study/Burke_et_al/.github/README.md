# Burke et al. [Balanced Neighborhoods for Multi-sided Fairness in Recommendation](https://proceedings.mlr.press/v81/burke18a.html)
The original code is property of *Robin Burke* and *Nasim Sonboli* and *Aldo Ordo&ntilde;ez-Gauger*, as well as the work proposed in their paper.
We thank them for providing this code.

## 1. Requirements
There isn't a source code linked in the paper, but the authors only mentioned the library used, which is [Librec 2.0](https://github.com/that-recsys-lab/librec/tree/2.0.0), that can be used with
the python wrapper [librec-auto](https://github.com/that-recsys-lab/librec-auto) with the version 0.1.2 on Python 3.
We added the file `requirements.txt` for a direct installation of required libraries. The dependencies can be installed with the following command in Windows:
```shell script
pip install -r requirements.txt
```
or the following one in Linux:
```shell script
pip3 install -r requirements.txt
```

## 2. Original code and Modifications
The original code was not available in a public repository, but they presented a tutorial at the conference FAT '20, showing experiments of fairness-aware
recommendations with *librec-auto*. The [repository of the tutorial](https://github.com/that-recsys-lab/librec-auto-tutorial) contains an example (example04) that makes use
of the fairness-aware approach for provider fairness. We used this example as a starting point, modifying the model used, the hyper-parameters, the xml entries
for the input data and the type of train-test splitting.
In particular, we make use of predefined train and test files, so we need `<train-file format="text">` xml entry instead of `<data-file format="text">`,
the following splitter section:
```xml
<splitter>
	<model>testset</model>
</splitter>
```
We also modified the appender class by using `net.librec.data.convertor.appender.UserFeatureAppender`, the algorithm class with the following xml code:
```xml
<alg>
	<class>net.librec.recommender.cf.ranking.UBLNSLIMRecommender</class>
	<early-stop>true</early-stop>
	<similarity type="user">cos</similarity>
	<neighborhood-size>50</neighborhood-size>
	<shrinkage>10</shrinkage>
	<bold-driver>false</bold-driver>
	<iterator-max>10</iterator-max>
	<l1-reg>0.1</l1-reg>
	<l2-reg>0.001</l2-reg>
	<l3-reg>25</l3-reg>
	<min-sim>0.0</min-sim>
</alg>
```
The `<list-size>` of the metrics section has been modified setting it to the maximum number of items in the selected dataset, and the `<protected-feature>`
has been set to `F`, which always represents the minority group (females, old).

Actually, *librec-auto* 0.1.2 does not fully support the usage of a predefined test file, so we had to dig into the source code of the library and modify
the file `librec_cmd.py`. Inside this directory you will find the [modified version](../librec_cmd.py) with modifications at the lines 115\-129.
The added lines use the right test set according to the selected dataset and it will work if the same input data used in our reproducibility study
will be used. If you want to use different test sets or different datasets you need to modify these lines accordingly.

## 3. Input Data Preparation and Hyper-parameters Setting
The input data can be generated with the script `generate_input_data.py` inside **/reproducibility_study/Preprocessing** by using the generation commands
inside the same file or the [REPRODUCE.md](../../../Preprocessing/REPRODUCE.md) file inside the same folder. Once you specified the selected metadata for the dataset and sensitive attribute,
you need to add the argument `--create_librec_auto_input_data` to generate the input files for this paper. You must copy the files from the
related directory and copy them inside the specific directory for the selected experiment inside this path. For instance, if you create the input data
of MovieLens 1M and selecting `user_gender` as sensitive attribute for gender experiments, you should copy the input data inside the directory **movielens_1m_gender_experiment/data**.
NOTICE: to reduce the number of labels we only use the labels `F` and `M`. The users labelled with `M` are the users of the majority group (male, young)
that in the pre-processed datasets are identified with `True`, while `F` are the users of the minority group (female, old) that in the pre-processed datasets
are identified with `False`, as done by the authors, who treat females as the protected class.

The hyper-parameters for the *BN-SLIM-U* model that we used are:
- early-stop: true
- similarity (type="user"): cos
- neighborhood-size: 50
- shrinkage: 10
- bold-driver: false
- iterator-max: 10
- &#955;<sub>1</sub> (`<l1-reg>`): 0.1
- &#955;<sub>2</sub> (`<l2-reg>`): 0.001
- &#955;<sub>3</sub> (`<l3-reg>`): 25
- min-sim: 0.0

The baseline *SLIM-U* can be obtained by setting &#955;<sub>3</sub> to 0.0.

## 4. Code Execution
The specific experiment, represented by the name of the folder, can be executed with the following command inside this folder in Windows:
```shell script
python -m librec_auto run EXPERIMENT_FOLDER_NAME
```
or the following one in Linux:
```shell script
python3 -m librec_auto run EXPERIMENT_FOLDER_NAME
```
For instance, in Windows you can execute the experiment with Last.FM 1K for gender experiment with the following command:
```shell script
python -m librec_auto run filtered(20)_lastfm_1K_gender_experiment
```

## 5. Predictions Extraction
The prediction file will be saved inside the directory **exp00000/result** with the name **out-1.txt** inside the directory of the selected experiment.

The metrics can be computed by adding the filepath of the results to `metrics_reproduced.py` inside **/reproducibility_study/Evaluation** and following
the instruction inside the [REPRODUCE.md](../../../Evaluation/REPRODUCE.md) file present in the same folder.

## 6. Further Notes
Nothing relevant to be mentioned.

# Citation
Burke, R., Sonboli, N., Ordonez-Gauger, A.: Balanced neighborhoods for multi-sided fairness in recommendation. In: Friedler, S.A., Wilson, C. 
(eds.) Conferenceon Fairness, Accountability and Transparency, FAT 2018, 23-24 February 2018, New York, NY, USA. Proceedings of Machine 
Learning Research, vol. 81, pp. 202–214. PMLR (2018),http://proceedings.mlr.press/v81/burke18a.html7.
