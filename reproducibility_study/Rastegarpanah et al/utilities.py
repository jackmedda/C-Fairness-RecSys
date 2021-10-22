import numpy as np
from scipy.spatial.distance import pdist
from scipy.special import comb
import pandas as pd

class polarization():
    
    def evaluate(self, X_est):
        return X_est.var(axis=0,ddof=0).mean()

    def gradient(self, X_est):
        """
        Returns the gradient of the divergence utility defined on the
        estimated ratings of the original users.
        The output is an n by d matrix which is flatten.
        """
        D = X_est - X_est.mean()
        G = D.values
        return  G


class individual_loss_variance():
    
    def __init__(self, X, omega, axis):
        self.axis = axis
        self.omega = omega
        self.X = X.mask(~omega)
        self.omega_user = omega.sum(axis=axis)
        
    def get_losses(self, X_est):
        X = self.X
        X_est = X_est.mask(~self.omega)
        E = (X_est - X).pow(2)
        losses = E.mean(axis=self.axis)
        return losses
        
    def evaluate(self, X_est):
        losses = self.get_losses(X_est)
        var =  losses.values.var()
        return var

    def gradient(self, X_est):
        """
        Returns the gradient of the utility.
        The output is an n by d matrix which is flatten.
        """
        X = self.X
        X_est = X_est.mask(~self.omega)
        diff = X_est - X
        if self.axis == 0:
            diff = diff.T
            
        losses = self.get_losses(X_est)
        B = losses - losses.mean()
        C = B.divide(self.omega_user)
        D = diff.multiply(C,axis=0)
        G = D.fillna(0).values
        if self.axis == 0:
            G = G.T
        return  G


class group_loss_variance():
    
    def __init__(self, X, omega, G, axis):
        self.X = X 
        self.omega = omega
        self.G = G
        self.axis = axis
        
        if self.axis == 0:
            self.X = self.X.T
            self.omega = self.omega.T
            
        self.group_id ={}
        for group in self.G:
            for user in G[group]:
                self.group_id[user] = group
        
        self.omega_group = {}
        for group in self.G:
            self.omega_group[group] = (~self.X.mask(~self.omega).loc[self.G[group]].isnull()).sum().sum()
        
        omega_user = {}
        for user in self.X.index:
            omega_user[user] = self.omega_group[self.group_id[user]]
        self.omega_user = pd.Series(omega_user)
        
    def get_losses(self, X_est):
        if self.axis == 0:
            X_est = X_est.T
            
        X = self.X.mask(~self.omega)
        X_est = X_est.mask(~self.omega)
        E = (X_est - X).pow(2)
        if not E.shape == X.shape:
            print 'dimension error'
            return
        losses = {}
        for group in self.G:
            losses[group] = np.nanmean(E.loc[self.G[group]].values)
        losses = pd.Series(losses)
        return losses
        
    def evaluate(self, X_est):
        losses = self.get_losses(X_est)
        var =  losses.values.var()
        return var

    def gradient(self, X_est):
        """
        Returns the gradient of the utility.
        The output is an n by d matrix which is flatten.
        """
        group_losses = self.get_losses(X_est)
        #n_group = len(self.G)
        
        X = self.X.mask(~self.omega)
        if self.axis == 0:
            X_est = X_est.T
        
        X_est = X_est.mask(~self.omega)
        diff = X_est - X
        if not diff.shape == X.shape:
            print 'dimension error'
            return
        
        user_group_losses ={}
        for user in X.index:
            user_group_losses[user] = group_losses[self.group_id[user]]
        losses = pd.Series(user_group_losses)
        
        B = losses - group_losses.mean()
        C = B.divide(self.omega_user)
        #C = (4.0/n_group) * C
        D = diff.multiply(C,axis=0)
        G = D.fillna(0).values
        if self.axis == 0:
            G = G.T
        return  G
