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

TABLES OF RESULTS
