# Li et al. [User-oriented Fairness in Recommendation](https://doi.org/10.1145/3442381.3449866)
The original code is property of *Yunqi Li* and *Hanxiong Chen* and *Zuohui Fu* and *Yingqiang Ge* and *Yongfeng Zhang*, as well as the work proposed in their paper.
We thank them for providing this code.

## 1. Requirements
There wasn't a `requirements.txt` file in the source code, but Python 3 is used and the necessary libraries have been mentioned inside the `README.txt` file of the original code:
- Pandas
- Numpy
- Lenskit
We added the file `requirements.txt` for a direct installation of required libraries. The dependencies can be installed with the following command in Windows:
```shell script
pip install -r requirements.txt
```
or the following one in Linux:
```shell script
pip3 install -r requirements.txt
```

The only one that worked in our Windows system was the version [torch-1.0.1-cp37-cp37m-win_amd64](https://download.pytorch.org/whl/cpu/torch-1.0.1-cp37-cp37m-win_amd64.whl) for Python 3.7.
Other versions for other systems and Python versions can be found in [PyTorch Website](https://download.pytorch.org/whl/torch_stable.html).
It can be installed with the following commmand in Windows:
```shell script
pip install torch-1.0.1-cp37-cp37m-win_amd64.whl
```
or the following one in Linux (possible command and wheel file):
```shell script
pip3 install torch-1.0.0-cp37-cp37m-linux_x86_64.whl
```

The fairness-aware post-processing method requires differnet libraries that can be installed in the same way using the `requirements.txt` file
inside **src**. In particular, this paper makes use of [Gurobi](https://www.gurobi.com/), a commercial optimization solver. To run this code,
it is necessary to install Gurobi and purchase a license. Without a license or fail to install properly, this code will not be able to run.
A license can be purchased or obtained for [academic use only](https://www.gurobi.com/academia/academic-program-and-licenses/).

## 2. Original code and Modifications
The original code was linked in the paper and available in a [public repository](https://github.com/rutgerswiselab/user-fairness), which, however, contains only the
fairness-aware post-processing method. We contacted the authors that sent us the link of another [public repository](https://github.com/rutgerswiselab/NLR) that contains the training and testing
pipeline of all the models used in their study. However, this codebase was missing two models that have been taken into account in their paper: PMF and NeuMF (NCF). We contacted
the authors again and they promptly sent us the scripts related to these two models.

These two models have been slightly modified to adjust some parts, since they seemed to be part of a new version of their codebase. Other modifications are related
to some bugs that we encountered during the usage of the codebase or to reduce the execution time.

## 3. Input Data Preparation and Hyper-parameters Setting
The input data can be generated with the script `generate_input_data.py` inside **/reproducibility_study/Preprocessing** by using the generation commands
inside the same file or the [REPRODUCE.md](../../../Preprocessing/REPRODUCE.md) file inside the same folder. Once you specified the selected metadata for the dataset and sensitive attribute,
you need to add the argument `--create_nlr_input_data` to generate the input files to train and test the baselines. You must copy the files from the
related directory and copy them inside **NLR/dataset/DATASET_NAME**, where **DATASET_NAME** is one of `movielens_1m` (for MovieLens 1M input data) and
`filtered(20)_lastfm_1K` (for Last.FM 1K input data). 

Once the models have been trained and tested, you need to take the predictions on the test set and use them to compute the input data for the fairness-aware
post-processing method. Use the same procedure that you used to create the first input data, but this time you need to add the argument
`--create_user_oriented_fairness_files_input_data`. You also need to specify the paths of the predictions obtained from the training and testing of the baselines
inside the `kwargs` dictionary of `generate_input_data.py`, and the `test_path` attribute, which is the test file of the relative dataset used to train and test
the baselines. These attribute and values are necessary to create the input data for the fairness-aware post-processing method.
The input data just generated need to be copied inside the folder **data/DATASET_NAME** inside this directory, where **DATASET_NAME** is the name
of the dataset that you specified for the variable `dataset_name`, which is described in the list below.
To select which input data to use in the fairness-aware post-processing method you need to modify the following variables of the file **src/model.py**:
- `dataset_name` (line 236-237): the dataset name (ex. `movielens_1m`) 
- `model_name` (line 239): the selected model (ex. `NCF`)
- `group_1` and `group_2` (line 244\-245/249\-250/253\-254): only one `group_1` and one `group_2` must be set for each experiment, and they should obviously
                                                             related (you cannot mix gender and age groups).
- `run_id` (line 256): it is a unique id that is added by `generate_input_data.py` in the filename for the creation of the input files for the fairness-aware
post-processing method. It is used to identify the right files with the string `rank` and the values of `group_1` and `group_2`.

The hyper-parameters that we used are:
- learning rate: 0.001
- l2-regularization: 0.00001
- epochs: 100 (best model is chosen on validation set)
- embedding size: 64

Some models other hyper-parameters could be selected and we set them to the following values:
- NeuMF (NCF):
    - MLP size: [32,16,8]
    - Final Output Layer size: [64]
- STAMP:
    - Maximum User History Length: 30

For the training phase we added 1 negative sample for each interaction in the training set, and the same for the validation set.
For the fairness-aware post-processing method we set `epsilon` to 0.0 to evaluate the consequences of aiming at a "perfect" fairness.

## 4. Code Execution
The code to train and test the models can be executed directly via command line. In particular, the commands to execute each model with the right parameters
must be launched in the folder **NLR/src**. The commands in Windows (for Linux `python` must be replaced with `python3`) are:

	MovieLens 1M
	PMF: python main.py --rank 1 --model_name PMF --optimizer Adam --lr 0.001 --l2 0.00001 --dataset movielens_1m --metric ndcg@10 --random_seed 2018 --gpu 0 --verbose -1 --unlabel_test 1
	STAMP: python main.py --rank 1 --model_name STAMP --optimizer Adam --lr 0.001 --l2 0.00001 --dataset movielens_1m --metric ndcg@10 --random_seed 2018 --gpu 0 --verbose -1 --unlabel_test 1 --max_his 30
	BiasedMF: python main.py --rank 1 --model_name BiasedMF --optimizer Adam --lr 0.001 --l2 0.00001 --dataset movielens_1m --metric ndcg@10 --random_seed 2018 --gpu 0 --verbose -1 --unlabel_test 1
	NeuMF (NCF.py): python main.py --rank 1 --model_name NCF --optimizer Adam --lr 0.001 --l2 0.00001 --dataset movielens_1m --metric ndcg@10 --random_seed 2018 --gpu 0 --verbose -1 --unlabel_test 1 --layers [32,16,8] --p_layers [64]
	
	Last.FM 1K
	PMF: python main.py --rank 1 --model_name PMF --optimizer Adam --lr 0.001 --l2 0.00001 --dataset filtered(20)_lastfm_1K --metric ndcg@10 --random_seed 2018 --gpu 0 --verbose -1 --unlabel_test 1
	STAMP: python main.py --rank 1 --model_name STAMP --optimizer Adam --lr 0.001 --l2 0.00001 --dataset filtered(20)_lastfm_1K --metric ndcg@10 --random_seed 2018 --gpu 0 --verbose -1 --unlabel_test 1 --max_his 30
	BiasedMF: python main.py --rank 1 --model_name BiasedMF --optimizer Adam --lr 0.001 --l2 0.00001 --dataset filtered(20)_lastfm_1K --metric ndcg@10 --random_seed 2018 --gpu 0 --verbose -1 --unlabel_test 1
	NeuMF (NCF.py): python main.py --rank 1 --model_name NCF --optimizer Adam --lr 0.001 --l2 0.00001 --dataset filtered(20)_lastfm_1K --metric ndcg@10 --random_seed 2018 --gpu 0 --verbose -1 --unlabel_test 1 --layers [32,16,8] --p_layers [64]
	


## 5. Predictions Extraction
The predictions files of the baselines will be saved in the folder **NLR/result/MODEL**, where **MODEL** is the name related to the parameter `model` of
the above commands. The filename of the predictions files contains the name of the dataset used to train and the string `test`.
The re-ranked recommendation list obtained by the fairness-aware post-processing method will be saved inside the folder **out_results** that is present in this directory.

The metrics can be computed by adding the filepath of the re-ranked recommendation list to `metrics_reproduced.py` inside **/reproducibility_study/Evaluation** and following
the instruction inside the [REPRODUCE.md](../../../Evaluation/REPRODUCE.md) file present in the same folder.

## 6. Further Notes
The code inside NLR is used to train and test the baselines. When the training procedure is executed other files will be created next to the input data and
it is perfectly normal, since the authors use a BPR loss for their models and negative and positive instances must be computed.

# Citation
Li, Y., Chen, H., Fu, Z., Ge, Y., Zhang, Y.: User-oriented fairness in recommendation. In: Leskovec, J., Grobelnik, M., Najork, M., Tang, J., Zia, L.
(eds.) WWW ’21: The Web Conference 2021, Virtual Event / Ljubljana, Slovenia, April 19\-23, 2021.pp. 624–632. ACM / IW3C2 (2021).
https://doi.org/10.1145/3442381.3449866, https://doi.org/10.1145/3442381.3449866
