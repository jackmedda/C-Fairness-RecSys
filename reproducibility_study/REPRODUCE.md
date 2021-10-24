# `reproducibility_study` Folder Organization

This document briefly describes the folders contained in this directory and how they are divided:

- **Evaluation**: contains the scripts and folders to evaluate the outcomes of the reproduced algorithms
- **Preprocessing**: contains the scripts to preprocess the raw datasets and a script to create the input data from the
                     preprocessed dataset for each reproduced algorithm
- **external_cool_kids_exp**: contains the script to reproduce the *Esktrand et al* approach, where the same algorithms
                              are re-trained for several runs with balanced data by an equal number of users for each
                              demographic group, randomly sampled from the original dataset. The results are averaged
                              among all the predictions of each run.
- *Paper* **et al**: *Paper* is the name of the first author of each reproduced paper and each folder contains the code
                     to reproduce the respective paper. 

For more detailed information you can consult the REPRODUCE.md files inside each one of the directories.
