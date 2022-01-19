# Kamishima et al. [Recommendation Independence](http://proceedings.mlr.press/v81/kamishima18a.html)
The original code is property of *Toshihiro Kamishima* and *Shotaro Akaho* and *Hideki Asoh* and *Jun Sakuma*, as well as the work proposed in their paper.
We thank them for providing this code.

## 1. Requirements
The dependencies can be easily installed using the `setup.py` provided by the authors with the following command in Windows or in Linux (Python 2):
```shell script
python -m setup.py install
```
or in Linux (Python 3):
```shell script
python3 -m setup.py install
```

## 2. Original code and Modifications
The original code is available in a [public repository](https://github.com/tkamishima/kamiers), and makes use of a codebase owned by the same authors and
shared as well in a [public repository](https://github.com/tkamishima/kamrecsys).

In particular, the files [kamiers/sp_pmf/bdist_match.py](kamiers/sp_pmf/bdist_match.py) and [kamiers/sp_pmf/mi_normal.py](kamiers/sp_pmf/mi_normal.py)
has been modified in the constructor the properly set the variables `a` and `b`.

## 3. Input Data Preparation and Hyper-parameters Setting
The input data can be generated with the script `generate_input_data.py` inside **/reproducibility_study/Preprocessing** by using the generation commands
inside the same file or the [REPRODUCE.md](../../Preprocessing/REPRODUCE.md) file inside the same folder. Once you specified the selected metadata for the dataset and sensitive attribute,
you need to add the argument `--create_rec_independence_input_data` to generate the input files for this paper. You must copy the files from the
related directory and copy them inside this directory `data` which is also present in the original code.

The hyper-parameters have been set modifying the files [kamiers/sp_pmf/bdist_match.py](kamiers/sp_pmf/bdist_match.py) and
[kamiers/sp_pmf/mi_normal.py](kamiers/sp_pmf/mi_normal.py) by setting the variables `a` and `b` in the constructor.
The hyper-parameters values that we selected for the baseline are:
- **MovieLens 1M**
    - k (embedding size): 7
    - C (regularization parameter): 10
- **Last.FM 1K**
    - k (embedding size): 30
    - C (regularization parameter): 20
    
For each type of independence term applied to the PMF baseline we selected the following parameters:
- **Mean Matching**
    - *MovieLens 1M*
        - &#951; (eta): 1e8
    - *Last.FM 1K*
        - &#951; (eta): (Gender: 1e6, Age: 1e7)
- **BDist Matching**
    - *MovieLens 1M*
        - a = 1e-8
        - b = (Gender: 1e-24, Age: 1e-12)
        - &#951; (eta): 1e8
    - *Last.FM 1K*
        - a = (Gender: 1e-8, Age: 1e-4)
        - b = 1e-24
        - &#951; (eta): (Gender: 1e6, Age: 1e8)
- **Mi Normal**
    - *MovieLens 1M*
        - a = (Gender: 1e-4, Age: 1e-8)
        - b = (Gender: 1e-12, Age: 1e-24)
        - &#951; (eta): Age: 1e8
    - *Last.FM 1K*
        - a = (Gender: 1e-8, Age: 1e-4)
        - b = (Gender: 1e-12, Age: 1e-24)
        - &#951; (eta): (Gender: 1e6, Age: 1e8)


## 4. Code Execution
Differently from others source codes, this codebase perfectly accomodate different data in input with an easy-to-use argument parser to pass the information
as arguments via command line. The following commands can be used to execute all the experiments in Windows for Python 3 (in Linux for Python 3 `python` must
be replaced by `python3`) and must be executed from this directory:
MovieLens 1M PMF Mean Matching Gender
	
	MovieLens 1M PMF Mean Matching Gender
	python -m scripts.exp_iers_sp -i data\movielens_1m_train_gender.csv -o movielens_1m_out_mean_matching_gender.json -t data\movielens_1m_test_gender.csv -m pmf_mean_match --no-timestamp -e 1 -C 1 -k 7 -d 1 5 1
	MovieLens 1M PMF BDist Matching Gender
	python -m scripts.exp_iers_sp -i data\movielens_1m_train_gender.csv -o movielens_1m_out_bdist_matching_gender.json -t data\movielens_1m_test_gender.csv -m pmf_bdist_match --no-timestamp -e 1 -C 1 -k 7 -d 1 5 1
	MovieLens 1M PMF Mi Normal Gender
	python -m scripts.exp_iers_sp -i data\movielens_1m_train_gender.csv -o movielens_1m_out_mi_normal_gender.json -t data\movielens_1m_test_gender.csv -m pmf_mi_normal --no-timestamp -e 1 -C 1 -k 7 -d 1 5 1
	MovieLens 1M PMF Mean Matching Age
	python -m scripts.exp_iers_sp -i data\movielens_1m_train_age.csv -o movielens_1m_out_mean_matching_age.json -t data\movielens_1m_test_age.csv -mpmf_mean_match --no-timestamp -e 1 -C 1 -k 7 -d 1 5 1
	MovieLens 1M PMF BDist Matching Age
	python -m scripts.exp_iers_sp -i data\movielens_1m_train_age.csv -o movielens_1m_out_bdist_matching_age.json -t data\movielens_1m_test_age.csv -m pmf_bdist_match --no-timestamp -e 1 -C 1 -k 7 -d 1 5 1
	MovieLens 1M PMF Mi Normal Age
	python -m scripts.exp_iers_sp -i data\movielens_1m_train_age.csv -o movielens_1m_out_mi_normal_age.json -t data\movielens_1m_test_age.csv -m pmf_mi_normal --no-timestamp -e 1 -C 1 -k 7 -d 1 5 1
	
	Last.FM 1K filtered(20) Mean Matching Gender
	python -m scripts.exp_iers_sp -i "data\filtered(20)_lastfm-1K_train_user_gender.csv" -o "filtered(20)_lastfm-1K_out_mean_matching_gender.json" -t "data\filtered(20)_lastfm_1K_test_user_gender.csv" -m pmf_mean_match --no-timestamp -e 1 -C 1 -k 7 -d 1 5 1
	Last.FM 1K filtered(20) BDist Matching Gender
	python -m scripts.exp_iers_sp -i "data\filtered(20)_lastfm-1K_train_user_gender.csv" -o "filtered(20)_lastfm-1K_out_bdist_matching_gender.json" -t "data\filtered(20)_lastfm_1K_test_user_gender.csv" -m pmf_bdist_match --no-timestamp -e 1 -C 1 -k 7 -d 1 5 1
	Last.FM 1K filtered(20) Mi Normal Gender
	python -m scripts.exp_iers_sp -i "data\filtered(20)_lastfm-1K_train_user_gender.csv" -o "filtered(20)_lastfm-1K_out_mi_normal_gender.json" -t "data\filtered(20)_lastfm_1K_test_user_gender.csv" -m pmf_mi_normal --no-timestamp -e 1 -C 1 -k 7 -d 1 5 1
	Last.FM 1K filtered(20) Mean Matching Age
	python -m scripts.exp_iers_sp -i "data\filtered(20)_lastfm-1K_train_user_age.csv" -o "filtered(20)_lastfm-1K_out_mean_matching_age.json" -t "data\filtered(20)_lastfm_1K_test_user_age.csv" -m pmf_mean_match --no-timestamp -e 1 -C 1 -k 7 -d 1 5 1
	Last.FM 1K filtered(20) BDist Matching Age
	python -m scripts.exp_iers_sp -i "data\filtered(20)_lastfm-1K_train_user_age.csv" -o "filtered(20)_lastfm-1K_out_bdist_matching_age.json" -t "data\filtered(20)_lastfm_1K_test_user_age.csv" -m pmf_bdist_match --no-timestamp -e 1 -C 1 -k 7 -d 1 5 1
	Last.FM 1K filtered(20) Mi Normal Age
	python -m scripts.exp_iers_sp -i "data\filtered(20)_lastfm-1K_train_user_age.csv" -o "filtered(20)_lastfm-1K_out_mi_normal_age.json" -t "data\filtered(20)_lastfm_1K_test_user_age.csv" -m pmf_mi_normal --no-timestamp -e 1 -C 1 -k 7 -d 1 5 1

## 5. Predictions Extraction
The prediction files will be saved in this directory in the form of json files, named according to the dataset used, the model with the specific independence term
and the sensitive attribute (except for the baseline), as specified by the argument of the `-o` placeholder in all of the commands

The metrics can be computed by adding the filepath of the results to `metrics_reproduced.py` inside **/reproducibility_study/Evaluation** and following
the instruction inside the [REPRODUCE.md](../../Evaluation/REPRODUCE.md) file present in the same folder. In particular, since this paper use 3 models with a different in-processing mitigation
and one only baseline, the predictions files of all the 3 models related to the same dataset and sensitive attribute will be passed as a list, and the names
of the 3 models will be passed as a list, following the order of the prediction files.

## 6. Further Notes
Nothing relevant to be mentioned.

# Citation
Kamishima, T., Akaho, S., Asoh, H., Sakuma, J.: Recommendation independence. In: Friedler, S.A., Wilson, C. (eds.) Conference on Fairness, Accountability and
Transparency, FAT 2018, 23-24 February 2018, New York, NY, USA. Proceedings of Machine Learning Research, vol. 81, pp. 187–201. PMLR (2018), http://proceedings.mlr.press/v81/kamishima18a.html