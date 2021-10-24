# Consumer Fairness in Recommender Systems: Contextualizing Definitions and Mitigations
![reproducibility task](research_method.jpg)

This is the repository for the paper *Consumer Fairness in Recommender Systems: Contextualizing Definitions and Mitigation*,
developed by Giacomo Medda, PhD student at University of Cagliari, with the support of [Mirko Marras](https://www.mirkomarras.com/),
Postdoctoral Research at University of Cagliari, and [Ludovico Boratto](https://www.mirkomarras.com/), Researcher at 
University of Cagliari.

The goal of the paper was to find a common understanding and practical benchmarks on how and when each procedure of consumer
fairness in recommender systems can be used in comparison to the others.

## Repository Organization

### Subfolders

The root of the repository contains several subfolders. The folder `reproducibility_study` contains the reproduced algorithms, the scripts to run
each of them and the description of how each reproduced paper should be used. It is the core of this study, while the other
folders contain code or ready-to-use subdirectories to support the scripts to:

  1. *Preprocess* the raw Datasets
  2. *Generate* the input data for each reproduced paper
  3. *Execute* the source code of the reproduced papers
  4. *Load* the predictions in a common format
  5. *Compute* the metrics
  
These directories are their contents are described by [README_codebase](README_codebase.MD), since the structure and code inside these
folders is only used to support the reproducibility study and make it independent from the specific implementation of each paper.

### `reproducibility_study`

This is the directory that contains the source code of each reproduced paper identified by the surname of the first author
of the respective paper. It also contains the folders to execute the 5 points described in [Subfolders](subfolders).

**Reproduced Papers**.
  - **Ashokan et al**: *Fairness metrics and bias mitigation strategies for rating predictions*
  - **Burke et al**: *Balanced Neighborhoods for Multi-sided Fairness in Recommendation*
  - **Ekstrand et al**: *All The Cool Kids, How Do They Fit In. Popularity and Demographic Biases in Recommender Evaluation and Effectiveness*
  - **Frisch et al**: *Co-clustering for fair recommendation*
  - **Kamishima et al**: *Recommendation Independence*
  - **Li et al**: *User-oriented Fairness in Recommendation*
  - **Rastegarpanah et al**: *Fighting Fire with Fire. Using Antidote Data to Improve Polarization and Fairness of Recommender Systems*
  - **Wu et al**: *Learning Fair Representations for Recommendation. A Graph-based Perspective*
  
**Reproducibility Pipeline**.
* **Preprocessing**.
    * `preprocess_ml1m`: script to preprocess MovieLens 1M (best described in the relative [REPRODUCE.md](reproducibility_study/Preprocessing/REPRODUCE.md))
    * `preprocess_lastfm1K`: script to preprocess Last.FM 1K (best described in the relative [REPRODUCE.md](reproducibility_study/Preprocessing/REPRODUCE.md))
    * `generate_input_data`: script to generate the input data of each reproduced paper (best described in the relative [REPRODUCE.md](Preprocessing/REPRODUCE.md))
* **Evaluation**.
    * `metrics_reproduced`: script the loads all the predictions of relevance scores and computes the metrics in form of
                            plots and latex tables (best described in the relative [REPRODUCE.md](reproducibility_study/Evaluation/REPRODUCE.md))

**external_cool_kids_exp**.
Script that automatize the process of execution of the source code of **Ekstrand et al**.

## Installation

Considering the codebase and the different versions of libraries used by each paper, the multiple Python versions are 
mandatory to execute properly this cose. 

The codebase (that is the code not inside `reproducibility_study`) needs a Python 3.8 installation and all the necessary libraries 
can be installed with the `requirements.txt` file in the root of the repository with the following command in Windows:
```shell script
pip install -r requirements.txt
```
or in Linux:
```shell script
pip3 install -r requirements.txt
```

The installation of each reproducible paper is thoroughly described in the `RECOMMEND.md` that you can find in each paper
folder, but every folder contains a `requirements.txt` file that you can use to install the dependencies in the same way.
We recommend to use virtual environments at least for each reproduced paper, since some require specific versions
of Python (2, 3, 3.7) and a virtual environment for each paper will maintain a good order in the code organization.
Virtual environments can be created in different ways depending on the Python version and on the system. The
[Python Documentation](https://docs.python.org/3/library/venv.html) describes the creation of virtual environments for
Python >= 3.5, while the [virtualenv Website](https://virtualenv.pypa.io/en/latest/index.html) can be used for Python 2. 

## Input Data

This section describes the procedure to 1) preprocess the raw dataset and 2) generate the input data for the reproduced papers
from the preprocessed dataset. Each section is also described inside the related folder.

### Preprocessing

The preprocessing of the raw dataset is performed by the script `preprocess_ml1m` for MovieLens 1M or `preprocess_lastfm1K`
for Last.FM 1K. MovieLens 1M is provided by [TensorFlow Datasets](https://www.tensorflow.org/datasets), while Last.FM 1K
has been manually added to the catalog of TensorFlow Datasets.
The commands to preprocess each dataset is present at the top of the related dataset script, but the procedure is also better
described inside the [REPRODUCE.md](reproducibility_study/Preprocessing/REPRODUCE.md)
The preprocessed datasets will be saved in **data/preprocessed_datasets** and they will be automatically loaded by the
codebase depending on the metadata of the selected dataset.

### Generation

Once the MovieLens 1M and the Last.FM 1K dataset have been processed, we can pass to the generation of the input data
for each reproduced paper. This is performed using the script `generate_input_data` that contains at the top the commands
to generate the data for each dataset and for each selected sensitive attribute. This is performed by the `data.utils`
module that contain a specific function to transform the preprocessed dataset in input data for a specific paper.
The commands and a deeper description of them is available inside the
[REPRODUCE.md](reproducibility_study/Preprocessing/REPRODUCE.md) file.

## Training/Mitigation Procedures

Each paper (folder) listed in [`reproducibility_study`](reproducibility-study) contains a **REPRODUCE.md** file that 
describe everything to setup, prepare and run each reproduced paper. In particular, instructions to install the dependencies
are provided, as well as the specific subfolders to fill with the input data generated in the previous step, in order to
properly run the experiments of the selected paper. The procedure for each source code is better described in the already
mentioned **REPRODUCE.md** file.

## Evaluation

The **REPRODUCE.md** file contained in each "paper" folder describes also where the predictions can be found at the end 
of the mitigation procedure and guide the developer on following the instructions of the 
[REPRODUCE.md](reproducibility_study/Evaluation/metrics_reproduced.py) of the **Evaluation**. This is the script that
must be configured the most, since the paths of the specific predictions of each paper and model must be copy and pasted
inside the script. When the script is executed the predictions are loaded in a common form, the metrics are computed and
finally converted in plot and latex table forms. The **REPRODUCE.MD** already mentioned better described these steps and
specifying which are the commands to execute to get the desired results.

# Results

## Top-N Recommendation Gender
| Metric   | Paper           | Model      | Result Type   | Value                    |
|:---------|:----------------|:-----------|:--------------|:-------------------------|
| NDCG     | Burke et al.    | SLIM-U     | Base          | 0.197                    |
| NDCG     | Burke et al.    | SLIM-U     | Mit           | 0.458                    |
| NDCG     | Ekstrand et al. | FunkSVD    | Base          | 0.011                    |
| NDCG     | Ekstrand et al. | FunkSVD    | Mit           | 0.011                    |
| NDCG     | Ekstrand et al. | ItemKNN    | Base          | 0.219                    |
| NDCG     | Ekstrand et al. | ItemKNN    | Mit           | 0.215                    |
| NDCG     | Ekstrand et al. | TopPopular | Base          | 0.312                    |
| NDCG     | Ekstrand et al. | TopPopular | Mit           | 0.321                    |
| NDCG     | Ekstrand et al. | UserKNN    | Base          | 0.407                    |
| NDCG     | Ekstrand et al. | UserKNN    | Mit           | 0.407                    |
| NDCG     | Frisch et al.   | LBM        | Base          | 0.181                    |
| NDCG     | Frisch et al.   | LBM        | Mit           | 0.216                    |
| NDCG     | Li et al.       | BiasedMF   | Base          | 0.246                    |
| NDCG     | Li et al.       | BiasedMF   | Mit           | 0.245                    |
| NDCG     | Li et al.       | NCF        | Base          | 0.237                    |
| NDCG     | Li et al.       | NCF        | Mit           | 0.235                    |
| NDCG     | Li et al.       | PMF        | Base          | 0.247                    |
| NDCG     | Li et al.       | PMF        | Mit           | 0.244                    |
| NDCG     | Li et al.       | STAMP      | Base          | 0.112                    |
| NDCG     | Li et al.       | STAMP      | Mit           | 0.110                    |
| DS       | Burke et al.    | SLIM-U     | Base          | ^-0.055 |
| DS       | Burke et al.    | SLIM-U     | Mit           | ^-0.106 |
| DS       | Ekstrand et al. | FunkSVD    | Base          | -0.008                   |
| DS       | Ekstrand et al. | FunkSVD    | Mit           | -0.003                   |
| DS       | Ekstrand et al. | ItemKNN    | Base          | *-0.079    |
| DS       | Ekstrand et al. | ItemKNN    | Mit           | *-0.101    |
| DS       | Ekstrand et al. | TopPopular | Base          | *-0.085    |
| DS       | Ekstrand et al. | TopPopular | Mit           | *-0.102    |
| DS       | Ekstrand et al. | UserKNN    | Base          | ^-0.100 |
| DS       | Ekstrand et al. | UserKNN    | Mit           | *-0.100    |
| DS       | Frisch et al.   | LBM        | Base          | *-0.049    |
| DS       | Frisch et al.   | LBM        | Mit           | *-0.053    |
| DS       | Li et al.       | BiasedMF   | Base          | ^-0.076 |
| DS       | Li et al.       | BiasedMF   | Mit           | *-0.049    |
| DS       | Li et al.       | NCF        | Base          | *-0.069    |
| DS       | Li et al.       | NCF        | Mit           | -0.051                   |
| DS       | Li et al.       | PMF        | Base          | *-0.065    |
| DS       | Li et al.       | PMF        | Mit           | -0.041                   |
| DS       | Li et al.       | STAMP      | Base          | -0.022                   |
| DS       | Li et al.       | STAMP      | Mit           | -0.016                   |
| KS       | Burke et al.    | SLIM-U     | Base          | ^0.012  |
| KS       | Burke et al.    | SLIM-U     | Mit           | ^0.008  |
| KS       | Ekstrand et al. | FunkSVD    | Base          | ^0.128  |
| KS       | Ekstrand et al. | FunkSVD    | Mit           | ^0.140  |
| KS       | Ekstrand et al. | ItemKNN    | Base          | ^0.026  |
| KS       | Ekstrand et al. | ItemKNN    | Mit           | ^0.036  |
| KS       | Ekstrand et al. | TopPopular | Base          | ^0.001  |
| KS       | Ekstrand et al. | TopPopular | Mit           | ^0.002  |
| KS       | Ekstrand et al. | UserKNN    | Base          | ^0.067  |
| KS       | Ekstrand et al. | UserKNN    | Mit           | ^0.074  |
| KS       | Frisch et al.   | LBM        | Base          | ^0.122  |
| KS       | Frisch et al.   | LBM        | Mit           | ^0.128  |
| KS       | Li et al.       | BiasedMF   | Base          | ^0.026  |
| KS       | Li et al.       | BiasedMF   | Mit           | ^0.001  |
| KS       | Li et al.       | NCF        | Base          | ^0.040  |
| KS       | Li et al.       | NCF        | Mit           | ^0.001  |
| KS       | Li et al.       | PMF        | Base          | ^0.028  |
| KS       | Li et al.       | PMF        | Mit           | ^0.001  |
| KS       | Li et al.       | STAMP      | Base          | ^0.001  |
| KS       | Li et al.       | STAMP      | Mit           | ^0.001  |

## Top-N Recommendation Age
| Metric   | Paper           | Model      | Result Type   | Value                    |
|:---------|:----------------|:-----------|:--------------|:-------------------------|
| NDCG     | Burke et al.    | SLIM-U     | Base          | 0.197                    |
| NDCG     | Burke et al.    | SLIM-U     | Mit           | 0.448                    |
| NDCG     | Ekstrand et al. | FunkSVD    | Base          | 0.011                    |
| NDCG     | Ekstrand et al. | FunkSVD    | Mit           | 0.011                    |
| NDCG     | Ekstrand et al. | ItemKNN    | Base          | 0.219                    |
| NDCG     | Ekstrand et al. | ItemKNN    | Mit           | 0.204                    |
| NDCG     | Ekstrand et al. | TopPopular | Base          | 0.312                    |
| NDCG     | Ekstrand et al. | TopPopular | Mit           | 0.315                    |
| NDCG     | Ekstrand et al. | UserKNN    | Base          | 0.407                    |
| NDCG     | Ekstrand et al. | UserKNN    | Mit           | 0.397                    |
| NDCG     | Frisch et al.   | LBM        | Base          | 0.181                    |
| NDCG     | Frisch et al.   | LBM        | Mit           | 0.186                    |
| NDCG     | Li et al.       | BiasedMF   | Base          | 0.246                    |
| NDCG     | Li et al.       | BiasedMF   | Mit           | 0.246                    |
| NDCG     | Li et al.       | NCF        | Base          | 0.237                    |
| NDCG     | Li et al.       | NCF        | Mit           | 0.238                    |
| NDCG     | Li et al.       | PMF        | Base          | 0.247                    |
| NDCG     | Li et al.       | PMF        | Mit           | 0.246                    |
| NDCG     | Li et al.       | STAMP      | Base          | 0.112                    |
| NDCG     | Li et al.       | STAMP      | Mit           | 0.111                    |
| DS       | Burke et al.    | SLIM-U     | Base          | ^-0.066 |
| DS       | Burke et al.    | SLIM-U     | Mit           | -0.062                   |
| DS       | Ekstrand et al. | FunkSVD    | Base          | -0.002                   |
| DS       | Ekstrand et al. | FunkSVD    | Mit           | -0.002                   |
| DS       | Ekstrand et al. | ItemKNN    | Base          | 0.018                    |
| DS       | Ekstrand et al. | ItemKNN    | Mit           | 0.026                    |
| DS       | Ekstrand et al. | TopPopular | Base          | -0.044                   |
| DS       | Ekstrand et al. | TopPopular | Mit           | -0.050                   |
| DS       | Ekstrand et al. | UserKNN    | Base          | -0.023                   |
| DS       | Ekstrand et al. | UserKNN    | Mit           | -0.026                   |
| DS       | Frisch et al.   | LBM        | Base          | -0.024                   |
| DS       | Frisch et al.   | LBM        | Mit           | -0.023                   |
| DS       | Li et al.       | BiasedMF   | Base          | -0.044                   |
| DS       | Li et al.       | BiasedMF   | Mit           | *-0.060    |
| DS       | Li et al.       | NCF        | Base          | -0.040                   |
| DS       | Li et al.       | NCF        | Mit           | *-0.052    |
| DS       | Li et al.       | PMF        | Base          | -0.046                   |
| DS       | Li et al.       | PMF        | Mit           | *-0.062    |
| DS       | Li et al.       | STAMP      | Base          | ^-0.037 |
| DS       | Li et al.       | STAMP      | Mit           | ^-0.039 |
| KS       | Burke et al.    | SLIM-U     | Base          | ^0.015  |
| KS       | Burke et al.    | SLIM-U     | Mit           | ^0.033  |
| KS       | Ekstrand et al. | FunkSVD    | Base          | ^0.029  |
| KS       | Ekstrand et al. | FunkSVD    | Mit           | ^0.037  |
| KS       | Ekstrand et al. | ItemKNN    | Base          | ^0.131  |
| KS       | Ekstrand et al. | ItemKNN    | Mit           | ^0.123  |
| KS       | Ekstrand et al. | TopPopular | Base          | ^0.006  |
| KS       | Ekstrand et al. | TopPopular | Mit           | ^0.007  |
| KS       | Ekstrand et al. | UserKNN    | Base          | ^0.036  |
| KS       | Ekstrand et al. | UserKNN    | Mit           | ^0.030  |
| KS       | Frisch et al.   | LBM        | Base          | ^0.129  |
| KS       | Frisch et al.   | LBM        | Mit           | ^0.152  |
| KS       | Li et al.       | BiasedMF   | Base          | ^0.016  |
| KS       | Li et al.       | BiasedMF   | Mit           | ^0.005  |
| KS       | Li et al.       | NCF        | Base          | ^0.010  |
| KS       | Li et al.       | NCF        | Mit           | ^0.005  |
| KS       | Li et al.       | PMF        | Base          | ^0.016  |
| KS       | Li et al.       | PMF        | Mit           | ^0.005  |
| KS       | Li et al.       | STAMP      | Base          | ^0.005  |
| KS       | Li et al.       | STAMP      | Mit           | ^0.005  |

## Rating Prediction Gender
| Metric   | Paper                | Model            | Result Type   | Value                   |
|:---------|:---------------------|:-----------------|:--------------|:------------------------|
| RMSE     | Ashokan et al.       | ALS BiasedMF Par | Base          | 1.147                   |
| RMSE     | Ashokan et al.       | ALS BiasedMF Par | Mit           | 1.148                   |
| RMSE     | Ashokan et al.       | ALS BiasedMF Val | Base          | 1.147                   |
| RMSE     | Ashokan et al.       | ALS BiasedMF Val | Mit           | 1.152                   |
| RMSE     | Ashokan et al.       | ItemKNN Par      | Base          | 1.182                   |
| RMSE     | Ashokan et al.       | ItemKNN Par      | Mit           | 1.191                   |
| RMSE     | Ashokan et al.       | ItemKNN Val      | Base          | 1.182                   |
| RMSE     | Ashokan et al.       | ItemKNN Val      | Mit           | 1.179                   |
| RMSE     | Ekstrand et al.      | AvgRating        | Base          | 1.239                   |
| RMSE     | Ekstrand et al.      | AvgRating        | Mit           | 1.246                   |
| RMSE     | Ekstrand et al.      | FunkSVD          | Base          | 1.254                   |
| RMSE     | Ekstrand et al.      | FunkSVD          | Mit           | 1.265                   |
| RMSE     | Ekstrand et al.      | ItemKNN          | Base          | 1.227                   |
| RMSE     | Ekstrand et al.      | ItemKNN          | Mit           | 1.237                   |
| RMSE     | Ekstrand et al.      | UserKNN          | Base          | 1.226                   |
| RMSE     | Ekstrand et al.      | UserKNN          | Mit           | 1.235                   |
| RMSE     | Kamishima et al.     | PMF BDist        | Base          | 1.329                   |
| RMSE     | Kamishima et al.     | PMF BDist        | Mit           | 1.365                   |
| RMSE     | Kamishima et al.     | PMF Mean         | Base          | 1.329                   |
| RMSE     | Kamishima et al.     | PMF Mean         | Mit           | 1.361                   |
| RMSE     | Kamishima et al.     | PMF Mi           | Base          | 1.329                   |
| RMSE     | Kamishima et al.     | PMF Mi           | Mit           | 1.374                   |
| RMSE     | Rastegarpanah et al. | ALS              | Base          | 2.280                   |
| RMSE     | Rastegarpanah et al. | ALS              | Mit           | 2.271                   |
| RMSE     | Rastegarpanah et al. | LMaFit           | Base          | 2.568                   |
| RMSE     | Rastegarpanah et al. | LMaFit           | Mit           | 2.549                   |
| RMSE     | Wu et al.            | FairGo GCN       | Base          | 1.663                   |
| RMSE     | Wu et al.            | FairGo GCN       | Mit           | 1.261                   |
| DS       | Ashokan et al.       | ALS BiasedMF Par | Base          | 0.015                   |
| DS       | Ashokan et al.       | ALS BiasedMF Par | Mit           | 0.017                   |
| DS       | Ashokan et al.       | ALS BiasedMF Val | Base          | 0.015                   |
| DS       | Ashokan et al.       | ALS BiasedMF Val | Mit           | 0.018                   |
| DS       | Ashokan et al.       | ItemKNN Par      | Base          | *0.034    |
| DS       | Ashokan et al.       | ItemKNN Par      | Mit           | ^0.049 |
| DS       | Ashokan et al.       | ItemKNN Val      | Base          | *0.034    |
| DS       | Ashokan et al.       | ItemKNN Val      | Mit           | *0.036    |
| DS       | Ekstrand et al.      | AvgRating        | Base          | 0.025                   |
| DS       | Ekstrand et al.      | AvgRating        | Mit           | 0.024                   |
| DS       | Ekstrand et al.      | FunkSVD          | Base          | *0.040    |
| DS       | Ekstrand et al.      | FunkSVD          | Mit           | 0.041                   |
| DS       | Ekstrand et al.      | ItemKNN          | Base          | *0.041    |
| DS       | Ekstrand et al.      | ItemKNN          | Mit           | 0.034                   |
| DS       | Ekstrand et al.      | UserKNN          | Base          | ^0.048 |
| DS       | Ekstrand et al.      | UserKNN          | Mit           | *0.042    |
| DS       | Kamishima et al.     | PMF BDist        | Base          | 0.037                   |
| DS       | Kamishima et al.     | PMF BDist        | Mit           | 0.023                   |
| DS       | Kamishima et al.     | PMF Mean         | Base          | 0.037                   |
| DS       | Kamishima et al.     | PMF Mean         | Mit           | 0.026                   |
| DS       | Kamishima et al.     | PMF Mi           | Base          | 0.037                   |
| DS       | Kamishima et al.     | PMF Mi           | Mit           | 0.009                   |
| DS       | Rastegarpanah et al. | ALS              | Base          | 0.052                   |
| DS       | Rastegarpanah et al. | ALS              | Mit           | 0.045                   |
| DS       | Rastegarpanah et al. | LMaFit           | Base          | 0.048                   |
| DS       | Rastegarpanah et al. | LMaFit           | Mit           | 0.024                   |
| DS       | Wu et al.            | FairGo GCN       | Base          | ^0.146 |
| DS       | Wu et al.            | FairGo GCN       | Mit           | 0.028                   |
| KS       | Ashokan et al.       | ALS BiasedMF Par | Base          | ^0.048 |
| KS       | Ashokan et al.       | ALS BiasedMF Par | Mit           | 0.014                   |
| KS       | Ashokan et al.       | ALS BiasedMF Val | Base          | ^0.048 |
| KS       | Ashokan et al.       | ALS BiasedMF Val | Mit           | ^0.051 |
| KS       | Ashokan et al.       | ItemKNN Par      | Base          | ^0.054 |
| KS       | Ashokan et al.       | ItemKNN Par      | Mit           | ^0.057 |
| KS       | Ashokan et al.       | ItemKNN Val      | Base          | ^0.054 |
| KS       | Ashokan et al.       | ItemKNN Val      | Mit           | ^0.045 |
| KS       | Ekstrand et al.      | AvgRating        | Base          | ^0.060 |
| KS       | Ekstrand et al.      | AvgRating        | Mit           | ^0.070 |
| KS       | Ekstrand et al.      | FunkSVD          | Base          | ^0.044 |
| KS       | Ekstrand et al.      | FunkSVD          | Mit           | ^0.047 |
| KS       | Ekstrand et al.      | ItemKNN          | Base          | ^0.054 |
| KS       | Ekstrand et al.      | ItemKNN          | Mit           | ^0.060 |
| KS       | Ekstrand et al.      | UserKNN          | Base          | ^0.038 |
| KS       | Ekstrand et al.      | UserKNN          | Mit           | ^0.045 |
| KS       | Kamishima et al.     | PMF BDist        | Base          | ^0.041 |
| KS       | Kamishima et al.     | PMF BDist        | Mit           | ^0.065 |
| KS       | Kamishima et al.     | PMF Mean         | Base          | ^0.041 |
| KS       | Kamishima et al.     | PMF Mean         | Mit           | ^0.080 |
| KS       | Kamishima et al.     | PMF Mi           | Base          | ^0.041 |
| KS       | Kamishima et al.     | PMF Mi           | Mit           | ^0.066 |
| KS       | Rastegarpanah et al. | ALS              | Base          | ^0.008 |
| KS       | Rastegarpanah et al. | ALS              | Mit           | ^0.010 |
| KS       | Rastegarpanah et al. | LMaFit           | Base          | ^0.007 |
| KS       | Rastegarpanah et al. | LMaFit           | Mit           | ^0.022 |
| KS       | Wu et al.            | FairGo GCN       | Base          | ^0.087 |
| KS       | Wu et al.            | FairGo GCN       | Mit           | ^0.052 |

## Rating Prediction Age

| Metric   | Paper                | Model            | Result Type   | Value                   |
|:---------|:---------------------|:-----------------|:--------------|:------------------------|
| RMSE     | Ashokan et al.       | ALS BiasedMF Par | Base          | 1.146                   |
| RMSE     | Ashokan et al.       | ALS BiasedMF Par | Mit           | 1.148                   |
| RMSE     | Ashokan et al.       | ALS BiasedMF Val | Base          | 1.146                   |
| RMSE     | Ashokan et al.       | ALS BiasedMF Val | Mit           | 1.151                   |
| RMSE     | Ashokan et al.       | ItemKNN Par      | Base          | 1.182                   |
| RMSE     | Ashokan et al.       | ItemKNN Par      | Mit           | 1.184                   |
| RMSE     | Ashokan et al.       | ItemKNN Val      | Base          | 1.182                   |
| RMSE     | Ashokan et al.       | ItemKNN Val      | Mit           | 1.178                   |
| RMSE     | Ekstrand et al.      | AvgRating        | Base          | 1.239                   |
| RMSE     | Ekstrand et al.      | AvgRating        | Mit           | 1.248                   |
| RMSE     | Ekstrand et al.      | FunkSVD          | Base          | 1.254                   |
| RMSE     | Ekstrand et al.      | FunkSVD          | Mit           | 1.271                   |
| RMSE     | Ekstrand et al.      | ItemKNN          | Base          | 1.227                   |
| RMSE     | Ekstrand et al.      | ItemKNN          | Mit           | 1.238                   |
| RMSE     | Ekstrand et al.      | UserKNN          | Base          | 1.226                   |
| RMSE     | Ekstrand et al.      | UserKNN          | Mit           | 1.237                   |
| RMSE     | Kamishima et al.     | PMF BDist        | Base          | 1.329                   |
| RMSE     | Kamishima et al.     | PMF BDist        | Mit           | 1.368                   |
| RMSE     | Kamishima et al.     | PMF Mean         | Base          | 1.329                   |
| RMSE     | Kamishima et al.     | PMF Mean         | Mit           | 1.369                   |
| RMSE     | Kamishima et al.     | PMF Mi           | Base          | 1.329                   |
| RMSE     | Kamishima et al.     | PMF Mi           | Mit           | 1.362                   |
| RMSE     | Rastegarpanah et al. | ALS              | Base          | 2.280                   |
| RMSE     | Rastegarpanah et al. | ALS              | Mit           | 2.253                   |
| RMSE     | Rastegarpanah et al. | LMaFit           | Base          | 2.568                   |
| RMSE     | Rastegarpanah et al. | LMaFit           | Mit           | 2.538                   |
| RMSE     | Wu et al.            | FairGo GCN       | Base          | 1.663                   |
| RMSE     | Wu et al.            | FairGo GCN       | Mit           | 1.273                   |
| DS       | Ashokan et al.       | ALS BiasedMF Par | Base          | 0.044                   |
| DS       | Ashokan et al.       | ALS BiasedMF Par | Mit           | *0.047    |
| DS       | Ashokan et al.       | ALS BiasedMF Val | Base          | 0.044                   |
| DS       | Ashokan et al.       | ALS BiasedMF Val | Mit           | *0.046    |
| DS       | Ashokan et al.       | ItemKNN Par      | Base          | 0.027                   |
| DS       | Ashokan et al.       | ItemKNN Par      | Mit           | 0.030                   |
| DS       | Ashokan et al.       | ItemKNN Val      | Base          | 0.027                   |
| DS       | Ashokan et al.       | ItemKNN Val      | Mit           | 0.030                   |
| DS       | Ekstrand et al.      | AvgRating        | Base          | 0.040                   |
| DS       | Ekstrand et al.      | AvgRating        | Mit           | 0.048                   |
| DS       | Ekstrand et al.      | FunkSVD          | Base          | 0.032                   |
| DS       | Ekstrand et al.      | FunkSVD          | Mit           | 0.037                   |
| DS       | Ekstrand et al.      | ItemKNN          | Base          | 0.016                   |
| DS       | Ekstrand et al.      | ItemKNN          | Mit           | 0.025                   |
| DS       | Ekstrand et al.      | UserKNN          | Base          | 0.031                   |
| DS       | Ekstrand et al.      | UserKNN          | Mit           | 0.027                   |
| DS       | Kamishima et al.     | PMF BDist        | Base          | 0.022                   |
| DS       | Kamishima et al.     | PMF BDist        | Mit           | 0.053                   |
| DS       | Kamishima et al.     | PMF Mean         | Base          | 0.022                   |
| DS       | Kamishima et al.     | PMF Mean         | Mit           | 0.048                   |
| DS       | Kamishima et al.     | PMF Mi           | Base          | 0.022                   |
| DS       | Kamishima et al.     | PMF Mi           | Mit           | 0.051                   |
| DS       | Rastegarpanah et al. | ALS              | Base          | *0.115    |
| DS       | Rastegarpanah et al. | ALS              | Mit           | *0.127    |
| DS       | Rastegarpanah et al. | LMaFit           | Base          | *0.128    |
| DS       | Rastegarpanah et al. | LMaFit           | Mit           | ^0.141 |
| DS       | Wu et al.            | FairGo GCN       | Base          | *0.091    |
| DS       | Wu et al.            | FairGo GCN       | Mit           | *0.058    |
| KS       | Ashokan et al.       | ALS BiasedMF Par | Base          | ^0.080 |
| KS       | Ashokan et al.       | ALS BiasedMF Par | Mit           | *0.017    |
| KS       | Ashokan et al.       | ALS BiasedMF Val | Base          | ^0.080 |
| KS       | Ashokan et al.       | ALS BiasedMF Val | Mit           | ^0.075 |
| KS       | Ashokan et al.       | ItemKNN Par      | Base          | ^0.083 |
| KS       | Ashokan et al.       | ItemKNN Par      | Mit           | ^0.022 |
| KS       | Ashokan et al.       | ItemKNN Val      | Base          | ^0.083 |
| KS       | Ashokan et al.       | ItemKNN Val      | Mit           | ^0.080 |
| KS       | Ekstrand et al.      | AvgRating        | Base          | ^0.080 |
| KS       | Ekstrand et al.      | AvgRating        | Mit           | ^0.092 |
| KS       | Ekstrand et al.      | FunkSVD          | Base          | ^0.086 |
| KS       | Ekstrand et al.      | FunkSVD          | Mit           | ^0.095 |
| KS       | Ekstrand et al.      | ItemKNN          | Base          | ^0.084 |
| KS       | Ekstrand et al.      | ItemKNN          | Mit           | ^0.094 |
| KS       | Ekstrand et al.      | UserKNN          | Base          | ^0.088 |
| KS       | Ekstrand et al.      | UserKNN          | Mit           | ^0.099 |
| KS       | Kamishima et al.     | PMF BDist        | Base          | ^0.070 |
| KS       | Kamishima et al.     | PMF BDist        | Mit           | ^0.093 |
| KS       | Kamishima et al.     | PMF Mean         | Base          | ^0.070 |
| KS       | Kamishima et al.     | PMF Mean         | Mit           | ^0.089 |
| KS       | Kamishima et al.     | PMF Mi           | Base          | ^0.070 |
| KS       | Kamishima et al.     | PMF Mi           | Mit           | ^0.101 |
| KS       | Rastegarpanah et al. | ALS              | Base          | ^0.011 |
| KS       | Rastegarpanah et al. | ALS              | Mit           | ^0.006 |
| KS       | Rastegarpanah et al. | LMaFit           | Base          | ^0.005 |
| KS       | Rastegarpanah et al. | LMaFit           | Mit           | ^0.011 |
| KS       | Wu et al.            | FairGo GCN       | Base          | ^0.039 |
| KS       | Wu et al.            | FairGo GCN       | Mit           | ^0.133 |