import sys
import cPickle
import numpy as np
import scipy as sp
import pandas as pd
import os
import itertools
import optimization as OPT
import MF
import utilities as UT
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import seaborn as sns

reload(MF)
reload(OPT)
reload(UT)

Data_path = 'Data/'

algorithm = 'gradient_descent'

#Read Movielens Dataset
n_users=  300
n_movies= 500
top_users = False

X, genres, user_info = MF.read_movielens_1M(n_movies, n_users, top_users, data_dir = Data_path+'MovieLens-1M')
omega = ~X.isnull()

rank = 4
lambda_ = 0.1

RS = MF.als_MF(rank,lambda_)
utility = UT.group_loss_variance()

pred,error = RS.fit_model(X)
X_est = RS.pred.copy()
print "before:", utility.evaluate(X_est)

n,d = X.shape
budget_perc = 1.0
budget = int(n_users*budget_perc/100.0)
print 'budget:',budget

#------------------------find optimal antidote--------------------------
runs = 3
stepsize = 1.0 #this only determines the direction if the gradient step. Positive values will minimize the objective and negative values will maximize it
projection = OPT.projection((0,5))
max_iter = 20
threshold = 0.1
window  = 1
steps = OPT.LineSearch_steps(10**3,6) #if None, the steps will be selected automatically based on the magnitude of the values in the gradient matrix
initial_data = 'random'

alg = OPT.gradient_descent_LS(max_iter, stepsize, threshold, steps, window)
results = alg.run(RS,X,budget,projection,utility,initial_data,runs)
X_antidote = results['X_antidote']
obj_history = results['obj_hist']
#-----------------------------------------------------------------------

#----------------------apply antidote data------------------------------
U_final,V_final,X_final = MF.antidote_effect(RS,X,X_antidote)
obj_after = utility.evaluate(X_final)
RMSE_after = MF.compute_RMSE(X,X_final,omega)
exit()
#-----------------------------------------------------------------------

#-----------------------------plots-------------------------------------
rc('text', usetex=True)
rc('font', size=13)
rc('font', family='serif')
    
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=0, hspace=0)
path = 'Plots/'

#plot the resulting antidote data
figsize =(8,6)    
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(1,1,1)
plt.imshow(X_antidote,cmap=plt.get_cmap('bwr'), interpolation="nearest", aspect='auto',vmin=0,vmax=5)
plt.colorbar(orientation='horizontal')
plt.ylabel("Antidote users",fontsize=16)
plt.xlabel("Movies",fontsize=16)
majorLocator = MultipleLocator(1)
majorFormatter = FormatStrFormatter('%d')
ax.yaxis.set_major_locator(majorLocator)
ax.yaxis.set_major_formatter(majorFormatter)
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top') 
plt.tight_layout()
plt.savefig(path + 'optimal_antidote.pdf',format='pdf')
plt.close()

#plot the distrtibution of polarization per items before and after antidote data injection
figsize =(8,6)    
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(1,1,1)
sns.kdeplot(X_est.var(ddof=0), shade=True, color="y",ax=ax)
ax.axvline(X_est.var(ddof=0).mean(), color='y', linewidth=0.9, label='mean(before)=%.2f'%X_est.var(ddof=0).mean())
sns.kdeplot(X_final.var(ddof=0), shade=True, color="r",ax=ax)
ax.axvline(X_final.var(ddof=0).mean(), color='r', linewidth=0.9, label='mean(after)=%.2f'%X_final.var(ddof=0).mean())
plt.ylabel("Density",fontsize =15)
plt.xlabel("Variance of per-item ratings",fontsize =15)
plt.legend(numpoints=1, fontsize =11.5)
ymin, ymax = ax.get_ylim() 
ax.set_ylim((0,13.5))
ax.set_xlim((0,2))
plt.tight_layout()
plt.savefig(path + 'Effect_on_item_polarization.pdf',format='pdf')
plt.close()

#plot the effect of antidote data on polarization of highly polarized items
items = X_est.var(ddof=0).sort_values(ascending=False).index
for movie in items[0:3]:
    #print movie, omega[movie].sum()
    figsize =(8,6)    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1,1,1)
    sns.kdeplot(X[movie], bw='scott', shade=True, color="blue",ax=ax, clip=(1,5),
                lw=1.5, label='Known ratings')
    
    sns.kdeplot(X_est[movie], shade=True, color="r",ax=ax, 
                lw=1.5, label='Estimated ratings (before)')
    
    sns.kdeplot(X_final[movie], shade=True, color="lime",ax=ax,
                lw=1.5, ls='dashed', label='Estimated ratings (after)')
    plt.ylabel("Density",fontsize =15)
    plt.xlabel("Rating",fontsize =15)
    plt.legend(numpoints=1, fontsize =10, loc=2)
    ymin, ymax = ax.get_ylim() 
    ax.set_ylim((ymin,ymax*2))
    ax.set_ylim((0,1.06))
    ax.set_xlim((-6,11))
    majorLocator = MultipleLocator(5)
    majorFormatter = FormatStrFormatter('%d')
    minorLocator = MultipleLocator(1)
    ax.xaxis.set_major_locator(majorLocator)
    ax.xaxis.set_major_formatter(majorFormatter)
    ax.xaxis.set_minor_locator(minorLocator)
    plt.tight_layout()
    plt.savefig(path + '%s.pdf'%movie,format='pdf')
    plt.close()
