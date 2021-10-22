import os
import pickle

import pandas as pd

import optimization as OPT
import MF
import utilities as UT

reload(MF)
reload(OPT)
reload(UT)

Data_path = 'Data/'

# dataset = "movielens_1m"
dataset = "filtered(20)_lastfm_1K"
algorithm = 'gradient_descent'
if dataset == "ml1m":
    # sensitive_attribute = "user_gender"
    sensitive_attribute = "bucketized_user_age"
else:
    sensitive_attribute = "user_gender"
    # sensitive_attribute = "user_age"

model = 'als_MF'
# model = 'lmafit_MF'

save_baseline = True

# Read Movielens Dataset
n_users = 300
n_movies = 500
top_users = False

# X, test, sens = MF.reproduce_read_movielens_1M(sensitive_attribute, data_dir=Data_path+'MovieLens-1M')

"""
X, genres, user_info = MF.read_movielens_1M(n_movies, n_users, top_users, data_dir=Data_path+'MovieLens-1M')

sens_inv = dict(user_info.reset_index()[['index', sensitive_attribute]].to_numpy())
sens = dict.fromkeys(np.unique(list(sens_inv.values())))
for u_id, sens_attr in sens_inv.items():
    sens[sens_attr] = sens[sens_attr] + [u_id] if isinstance(sens[sens_attr], list) else [u_id]
"""

X = pd.read_csv(os.path.join(Data_path, "{}_X.csv".format(dataset)), index_col=0)
test = pd.read_csv(os.path.join(Data_path, "{}_test.csv".format(dataset)), index_col=0)
with open(os.path.join(Data_path, "{}_sensitive_attribute_({}).pkl".format(dataset, sensitive_attribute)), 'rb') as f:
    sens = pickle.load(f)

n_users = len(X.index)
omega = ~X.isnull()

rank = 8
lambda_ = 1

RS = getattr(MF, model)(rank, lambda_)
utility = UT.group_loss_variance(X, omega, sens, axis=1)

pred, error = RS.fit_model(X)
X_est = RS.pred.copy()
# print X_est
if save_baseline:
    X_est.to_csv('{}_antidote_group_out_{}_baseline.csv'.format(dataset, model))
print "before:", utility.evaluate(X_est)

n, d = X.shape
budget_perc = 0.5
budget = int(n_users * budget_perc / 100.0)
print 'budget:', budget

#------------------------find optimal antidote--------------------------
runs = 10
stepsize = 1.0  # this only determines the direction if the gradient step. Positive values will minimize the objective and negative values will maximize it
projection = OPT.projection((0, 5))
max_iter = 20
threshold = 0.1
window = 1
steps = OPT.LineSearch_steps(10 ** 3, 6)  # if None, the steps will be selected automatically based on the magnitude of the values in the gradient matrix
initial_data = 'random'

alg = OPT.gradient_descent_LS(max_iter, stepsize, threshold, steps, window)
results = alg.run(RS, X, budget, projection, utility, initial_data, runs)
X_antidote = results['X_antidote']
obj_history = results['obj_hist']
#-----------------------------------------------------------------------

#----------------------apply antidote data------------------------------
U_final, V_final, X_final = MF.antidote_effect(RS, X, X_antidote)
obj_after = utility.evaluate(X_final)
RMSE_after = MF.compute_RMSE(X, X_final, omega)
print RMSE_after
RMSE_test = MF.compute_RMSE(X_final, test, omega, not_omega=True)
print RMSE_test
print X_final

X_final.to_csv('{}_antidote_group_out_{}_{}.csv'.format(dataset, model, sensitive_attribute))

print obj_history
