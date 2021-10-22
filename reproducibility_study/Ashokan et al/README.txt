README for Rating Prediction Code

This is the code used for the simulations in Ashokan and Haas, Fairness Metrics and Bias Mitigation Strategies for Rating Predictions, Information Processing and Management, 2021.


## Structure ##
The code has 5 main files:

- Main.py: the core logic of the simulation. Here, you can specify if you want to run the simulation on a specific, provided dataset (e.g., MovieLens), or the synthetic dataset. 
Note: the dataset needs to have a variable that specifies privileged and unprivileged groups. See code for an example.

- FairnessMetrics.py: defines the functions to calculate various fairness metrics for rating predictions

- PostProcessing.py: defines the postprocessing adjustments according to the bias mitigation strategy suggested by Ashokan and Haas, 2021.

- HelperFunctions.py: various helper functions used in the other modules

- SyntheticDataCreation.py: defines and creates syntethic data following the example in Yao and Huang 2017.

## Packages needed ##
The following packages need to be installed for the code to run:
Pandas
Numpy
Lenskit 

## Running the code ##

In Main.py, specify which dataset to use. Currently, choices are between empirical (provided) datasets (here: MovieLens1M), or a syntethic dataset. 
Then, the code splits the overall data into k folds, trains the predictions (using a variety of algorithms, provided by LensKit) on the training set, and predicts that ratings for the test partitions.
The post-processing functions adapt the ratings based on the learned differences in the training sets to counteract specific unfairness/bias.
Lastly, for the predicted ratings as well as post-processed (bias mitigated) ratings on the test sets, various performance and fairness metrics are calculated and saved in a csv file for further analysis.

