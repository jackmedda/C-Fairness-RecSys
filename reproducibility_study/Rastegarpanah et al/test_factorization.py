import numpy as np
import scipy as sp
import pandas as pd
import timeit
import MF
reload(MF)

#Read Movielens Dataset
n_users=  1000
n_movies= 1000
top_users= True
X, genres, user_info = MF.read_movielens_1M(n_movies, n_users, top_users)


known = X.count().sum() / (1.0*X.size)
print known

n_known_item = (~X.isnull()).sum(axis=0).sort_values()
n_known_user = (~X.isnull()).sum(axis=1).sort_values()

#Factorization parameters
rank = 20
lambda_ = 20 #Ridge regularizer parameter

#Initiate a recommender system of type ALS
RS = MF.als_MF(rank,lambda_)

#Initiate a recommender system of type lmafit
#RS = MF.lmafit_MF(rank)

start = timeit.default_timer()
pred,error = RS.fit_model(X)
time = timeit.default_timer() - start
print 'factorization time:',time
print 'RMSE:',np.sqrt(error/X.count().sum())

V = RS.get_V()
U = RS.get_U()
