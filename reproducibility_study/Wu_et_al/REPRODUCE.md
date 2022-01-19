# Wu et al. [Learning Fair Representations for Recommendation. A Graph-based Perspective](https://doi.org/10.1145/3442381.3450015)
The original code is property of *Le Wu* and *Lei Chen* and *Pengyang Shao* and *Richang Hong* and *Xiting Wang* and *Meng Wang*, as well as the work proposed in their paper.
We thank them for providing this code.

## 1. Requirements
There wasn't a `requirements.txt` file in the source code, but the repository writes the following dependendcies and needed software:
- Python 3.5
- PyTorch
We added the file `requirements.txt` for a direct installation of required libraries. The dependencies can be installed with the following command in Windows:
```shell script
pip install -r requirements.txt
```
or the following one in Linux:
```shell script
pip3 install -r requirements.txt
```

## 2. Original code and Modifications
The original code is linked in the paper and it is available in a [public repository](https://github.com/newlei/FairGo), but it only contains the mitigation algorithm, without the part
necessary to train and save the embeddings of the GCN. We contacted the authors that sent us the link of another [public repository](https://github.com/newlei/LR-GCCF) that they used as a template
to create GCN for rating prediction (replacing the loss from a ranking one to a rating one). We performed the same task by adding the implementation of the GCN, the training procedure,
the input data loading and the embeddings saving inside the `filter_layer.py` file of each code folder. The code for MovieLens 1m is contained in **code/code_ml**, while
the one for Last.FM 1K is inside the directory **code/code_lastfm_1K**.

The modifications have been done on the files `FairGo_gcn_remove_age.py`, `FairGo_gcn_remove_gender.py`, `filter_layer.py` to remove the usage of CUDA, to load our input data, to better construct
the embeddings (to use the full number of users and items) and to train the GCN network, whose code is all contained in `filter_layer.py`. The paths are ready to be used and there is no need to modify
the code in any way.

## 3. Input Data Preparation and Hyper-parameters Setting
The input data can be generated with the script `generate_input_data.py` inside **/reproducibility_study/Preprocessing** by using the generation commands
inside the same file or the [REPRODUCE.md](../../Preprocessing/REPRODUCE.md) file inside the same folder. Once you specified the selected metadata for the dataset and sensitive attribute,
you need to add the argument `--create_fairgo_input_data` to generate the input files for this paper. You must copy the files from the
related directory and copy them inside the directory **filtered(20)_lastfm_1K_reproduce_data** for Last.FM 1K and **movielens_1m_reproduce_data** for MovieLens 1M.

The embedding of the GCN baseline have been created using the following hyper-parameters:
- GCN embedding size: (MovieLens 1M: 64, Last.FM 1K: 128)
- GCN learning rate: (MovieLens 1M: 0.01, Last.FM 1K: 0.005)

The parameters selected for the filtering mitigation approach are: 
- **MovieLens 1M**:
    - d_optimizer learning rate (Learning rate of the discriminators optimizer): 0.01
    - f_optimizer learning rate (Learning rate of the filters optimizer): 0.005
    - &#955; (lambda): 0.2
- **Last.FM 1K**:
    - d_optimizer learning rate (Learning rate of the discriminators optimizer): (Gender: 0.005, Age: 0.01)
    - f_optimizer learning rate (Learning rate of the filters optimizer): (Gender: 0.01, Age: 0.005)
    - &#955; (lambda): 0.3

## 4. Code Execution
The code to train the GCN and save the embeddings can be executed with the following command in Windows inside the folder **code/code_ml** for MovieLens 1M or inside the folder
**code/code_lastfm_1K** for Last.FM 1K:
```shell script
python filter_layer.py
```
or the following one in Linux:
```shell script
python3 filter_layer.py
```
The mitigation procedure can be executed inside the same folder of `filter_layer.py`, and calling `FairGo_gcn_remove_age.py` for age experiment or `FairGo_gcn_remove_gender.py`
for gender experiment. For instance, the gender experiment can be computed with the following command in Windows:
```shell script
python FairGo_gcn_remove_gender.py
```
or the following one in Linux:
```shell script
python3 FairGo_gcn_remove_gender.py
```

The best embeddings of the GCN are chosen on validation data performance. The same approach is used for the unfairness mitigation procedure, such that
only the predictions on test data of the epoch with the best performance on validation set are saved.

## 5. Predictions Extraction
The most performant GCN embeddings will be saved inside **code/code_ml/new_gcn_embs** for MovieLens 1M and inside **code/code_lastfm_1K/new_gcn_embs** for Last.FM 1K.
These embeddings will be loaded by the unfairness mitigation scripts (gender or age) as well as the input data. Once the training procedure is complete the predictions will be
saved in the same folder that contains the input data, so **filtered(20)_lastfm_1K_reproduce_data** for Last.FM 1K and **movielens_1m_reproduce_data** for MovieLens 1M.
The end of the filename of the predictions will be `age` for experiment on age groups or `gender` for experiment on gender groups.

The metrics can be computed by adding the filepath of the results to `metrics_reproduced.py` inside **/reproducibility_study/Evaluation** and following
the instruction inside the [REPRODUCE.md](../../Evaluation/REPRODUCE.md) file present in the same folder.

## 6. Further Notes
The script [generate_input_data.py](../../Preprocessing/generate_input_data.py) must be executed only once for this paper,
since the sensitive attributes (gender and age) are saved together in a common data structure.

# Citation
Wu, L., Chen, L., Shao, P., Hong, R., Wang, X., Wang, M.: Learning fair representations for recommendation: A graph-based perspective.
In: WWW ’21: TheWeb Conference 2021, Virtual Event / Ljubljana, Slovenia, April 19\-23, 2021.pp. 2198–2208. ACM / IW3C2 (2021). https://doi.org/10.1145/3442381.3450015, https://doi.org/10.1145/3442381.3450015
