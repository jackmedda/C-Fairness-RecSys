## All The Cool Kids, How Do They Fit In?: Popularity and Demographic Biases in Recommender Evaluation and Effectiveness

This archive contains the scripts to reproduce the paper “All The Cool Kids, How Do They Fit In?: Popularity and Demographic Biases in Recommender Evaluation and Effectiveness” by Michael D. Ekstrand, Mucun Tian, Ion Madrazo Azpiazu, Jennifer D. Ekstrand, Oghenemaro Anuyah, David McNeill, & Maria Soledad Pera in the Fairness, Accountability and Transparency (FAT) 2018 Machine Learning Proceedings.

### Requirements

* Java
* The [LastFM 360K](http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-360K.html) and [LastFM 1K](http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-1K.html) data sets, extracted into data
* The [MovieLens 1M](https://grouplens.org/datasets/movielens/) data set, extracted into data (should have directory data/ml-1m)
* An R, Jupyter, Tidyverse, MASS, and plyr installation; we recommend using Anaconda.

The included environment file (environment-linux-x64.yml) can be used to create an Anaconda environment on 64-bit Linux that contains all required analysis packages; just add data and Java.

### Instructions

Steps to run:

1. Install software and download data listed in Requirements. When you are done, you should have the following directories and files:
    - ML-1M in data/ml-1m (e.g. data/ml-1m/ratings.dat)
    - Last.FM 360K in data/lastfm-dataset-360k (e.g. data/lastfm-dataset-360k/usersha1-artmbid-artname-plays.tsv)
    - Last.FM 1K in data/lastfm-dataset-1k (e.g. data/lastfm-dataset-1k/userid-timestamp-artid-artname-traid-traname.tsv)
2. Run LensKit experiments: 
    - To run all evaluations, run `./gradlew evaluateAll`
    - As is typical with Gradle projects, all output files go in the `build` directory, where they can be removed with `./gradlew clean`
4. Run the pre-load scripts:
    ```
    Rscript pre-lastfm-360k.R
    Rscript preload-train-ratings.R
    ```
3. Run Jupyter Notebooks in order:
	1. InputAnalysis.ipynb (Contains information about Raw Data Distribution)
	2. BasicResults.ipynb (Contains the Raw Data Results Analysis)
	3. ControlForProfilesSize.ipynb (Control for Profile size)
	4. BasicResults-balanced.ipynb (Balance sampling)
	5. U1R.ipynb (Popularity Analysis)
	6. InputAnalysis-PerSessionCount.ipynb (Retention Analysis)
