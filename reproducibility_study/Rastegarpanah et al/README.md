# antidote-data-framework
This is the Python implementation of the antidote data generation framework introduced in "Fighting Fire with Fire: Using Antidote Data to Improve Polarization and Fairness of Recommender Systems", WSDM 2019

The current implementation generates antidote data for matrix factorization based recommender systems. The framework consists of the following components:

- ("MF.py") provides an abstraction of a recommender system. An instance of this class will be sent to the optimization algorithm that generates the antidote data. Two algorithms are implemented for solving the matrix factorization: regularized alternating least squares and a Python implementation of LMaFit (http://lmafit.blogs.rice.edu/).

- ("utilities.py") contains the social objectives for which we want to generate antidote data. The utilities that are implemented in the current version are: polarization, individual fairness, and group fairness. Detailed description
of each social objective is provided in the paper. A new objective functions can be easily added to this file as a new class. Each class has two main methods: evaluate() defines the social property as a function of partially observed ratings  and/or estimated ratings; gradient() provides the gradient of the objective function with respect to the estimated ratings.

- ("optimization.py") contains different optimization algorithms for generating antidote data. gradient_descent_LS() is the implementation of the projected gradient descent algorithm presented as Algorithm 1 in the paper. The optimal step size is selected via a grid search.

- ("minimze_polarization.py") provides an example of how to use the framework. A subset of 300 users and 500 movies from the MovieLens 1M dataset is selected and antidote data is generated to minimize the average user polarization per movie. The resulting plots show the effect of antidote data on the rating estimations. More details about how to set the parameters of the optimization framework are provided in the code.

We have used the MovieLens 1M Dataset: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History
and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872

For reproducing the results you can download the dataset from (https://grouplens.org/datasets/movielens/) and place it in the 'Data/' folder.

