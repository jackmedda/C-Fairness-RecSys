import numpy as np
import scipy as sp
import pandas as pd
import numpy.ma as ma
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from abc import ABCMeta, abstractmethod
import random 
import timeit
import itertools
from functools import partial

def outer_sum(M,indices):
        S = np.zeros((M.shape[1],M.shape[1]))
        for i in indices:
            S += np.outer(M[i],M[i])
        return S

def d_theta_wrt_antidote_general(U, V, U_tilde, W, W_antidote, lambda_):
    """
    Returns the gradient of each element of the optimal 
    factor V with respect to each element of the antidote data.
    The output is a n'*d by l*d matrix that contains aproximated 
    partial derivates computed based on the method in [li2016data].
    """
    n,l = U.shape
    d = V.shape[0]
    n_prime = U_tilde.shape[0]
    
    def outer_sum(M,indices):
        S = np.zeros((M.shape[1],M.shape[1]))
        for i in indices:
            S += np.outer(M[i],M[i])
        return S
    
    known_ratings_per_item = [np.where(W.T[i])[0] for i in range(W.shape[1])]
    known_ratings_per_item_antidote = [np.where(W_antidote.T[i])[0] for i in range(W_antidote.shape[1])]
    
    sigma_V = [outer_sum(U,known_ratings_per_item[j]) +
               outer_sum(U_tilde,known_ratings_per_item_antidote[j]) for j in range(d)]
    #sigma_V = [U.T.dot(np.diag(1.0*W.T[j])).dot(U) +
    #           U_tilde.T.dot(np.diag(1.0*W_antidote.T[j])).dot(U_tilde) for j in range(d)]
    sigma_V_inv = [np.linalg.inv(sigma_Vj + lambda_*np.eye(l)) for sigma_Vj in sigma_V]

    L = []
    for i in range(n_prime):
        diag = [sigma_V_inv[j].dot(U_tilde[i]) for j in range(d)]
        L.append(sp.linalg.block_diag(*diag))
    D = np.vstack(tuple(L))
    return D
    
def d_theta_wrt_antidote(U, V, U_tilde, W, W_antidote, lambda_):
    """
    Returns the gradient of each element of the optimal 
    factor V with respect to each element of the antidote data.
    The output is a n'*d by l*d matrix that contains aproximated 
    partial derivates computed based on the method in [li2016data].
    """
    n,l = U.shape
    d = V.shape[0]
    n_prime = U_tilde.shape[0]
               
    sigma_V = [U.T.dot(np.diag(1.0*W.T[j])).dot(U) + 
               U_tilde.T.dot(U_tilde) + 
               lambda_*np.eye(l)   for j in range(d)]
    sigma_V_inv = [np.linalg.inv(sigma_Vj) for sigma_Vj in sigma_V]
    
    L = []
    for i in range(n_prime):
        diag = [sigma_V_inv[j].dot(U_tilde[i]) for j in range(d)]
        L.append(sp.linalg.block_diag(*diag))
    D = np.vstack(tuple(L))
    return D
        
def d_theta_wrt_antidote_fast(U, V, U_tilde, W, W_antidote, lambda_):
    """
    Returns the gradient of each element of the optimal 
    factor V with respect to each element of the antidote data.
    The output is a n'*d by l*d matrix that contains aproximated 
    partial derivates computed based on the method in [li2016data].
    """
    n,l = U.shape
    d = V.shape[0]
    n_prime = U_tilde.shape[0]
    
    known_ratings_per_item = [np.where(W.T[i])[0] for i in range(W.shape[1])]
               
    sigma_V = [outer_sum(U,known_ratings_per_item[j]) + 
               U_tilde.T.dot(U_tilde) + 
               lambda_*np.eye(l)   for j in range(d)]
    sigma_V_inv = [np.linalg.inv(sigma_Vj) for sigma_Vj in sigma_V]
    
    L = []
    for i in range(n_prime):
        diag = [sigma_V_inv[j].dot(U_tilde[i]) for j in range(d)]
        L.append(sp.linalg.block_diag(*diag))
    D = np.vstack(tuple(L))
    return D

def d_est_wrt_theta(U,V):
    """
    Returns the gradient of the estimated rating matrix of original users
    with respect to the optimal factor V.
    The output is a l*d by n*d matrix.
    """
    n,l = U.shape
    d = V.shape[0]
    D = np.hstack(tuple([sp.linalg.block_diag(*(d*[U[i].reshape(1,len(U[i])).T])) for i in range(n)]))
    return D
    
def compute_gradient_slow(MF,utility, X, X_antidote, U, V, U_tilde):
    X_est = U.dot(V)
    U = U.values
    U_tilde = U_tilde.values
    V = V.T.values
    W = ~np.isnan(X).values
    W_antidote = ~np.isnan(X_antidote).values
    
    G1 = d_theta_wrt_antidote(U,V,U_tilde,W,W_antidote,MF.lambda_)
    G2 = d_est_wrt_theta(U,V)
    G3 = utility.gradient(X_est).flatten() #d_utility_wrt_est

    n,d = X.shape
    if n*d<10000:
        gradient = G1.dot(G2).dot(G3)
    else:        
        G1 = csr_matrix(G1)
        G2 = csr_matrix(G2)
        gradient = G1.dot(G2.dot(G3))
    return gradient.reshape((X_antidote.shape[0],X_antidote.shape[1]))

def compute_gradient(MF,utility, X, X_antidote, U, V, U_tilde):
    X_est = U.dot(V)
    U = U.values
    U_tilde = U_tilde.values
    V = V.T.values
    W = ~np.isnan(X).values
    W_antidote = ~np.isnan(X_antidote).values
    
    #~ if X.shape[0]>900:
        #~ G1 = d_theta_wrt_antidote_fast(U,V,U_tilde,W,W_antidote,MF.lambda_)
    #~ else:
        #~ G1 = d_theta_wrt_antidote(U,V,U_tilde,W,W_antidote,MF.lambda_)
    
    G1 = d_theta_wrt_antidote_fast(U,V,U_tilde,W,W_antidote,MF.lambda_)
    G2 = U.T
    G3 = utility.gradient(X_est)#d_utility_wrt_est
    
    gradient = G1.dot((G2.dot(G3)).reshape(G1.shape[1],1,order='F'))
    return gradient.reshape((X_antidote.shape[0],X_antidote.shape[1]))
    
def compute_gradient_fast(MF,utility, X, X_antidote, U, V, U_tilde):
    X_est = U.dot(V)
    U = U.values
    U_tilde = U_tilde.values
    V = V.T.values
    W = ~np.isnan(X).values

    n,l = U.shape
    d = V.shape[0]

    I = MF.lambda_*np.eye(l)
    UTU_tilde = U_tilde.T.dot(U_tilde)
    
    known_ratings_per_item = [np.where(W.T[i])[0] for i in range(W.shape[1])]
               
    sigma_V_inv = [np.linalg.inv(outer_sum(U,known_ratings_per_item[j]) + UTU_tilde + I)
                   for j in range(d)]
                   
    G3 = utility.gradient(X_est)
    D = G3.T.dot(U)
    C = [vec.dot(sigma_V_inv[j]) for j,vec in enumerate(D)]
    C = np.array(C)
    G = C.dot(U_tilde.T)
    gradient = G.T
    return gradient
    
def compute_gradient_fast2(MF,utility, X, X_antidote, U, V, U_tilde):
    X_est = U.dot(V)
    U = U.values
    U_tilde = U_tilde.values
    V = V.T.values
    W = ~np.isnan(X).values

    n,l = U.shape
    d = V.shape[0]

    I = MF.lambda_*np.eye(l)
    UTU_tilde = U_tilde.T.dot(U_tilde)
    
    known_ratings_per_item = [np.where(W.T[i])[0] for i in range(W.shape[1])]
               
    sigma_V_inv = [np.linalg.inv(outer_sum(U,known_ratings_per_item[j]) + UTU_tilde + I)
                   for j in range(d)]
                   
    G3 = utility.gradient(X_est)
    D = U.T.dot(G3)
    C = [sigma_V_inv[j].dot(vec) for j,vec in enumerate(D.T)]
    G = [ [u_tilde.dot(c) for c in C] for u_tilde in U_tilde]
        
    gradient = np.array(G)
    return gradient
    
def theta(MF, X, X_antidote, init=None):
    ratings = pd.concat([X,X_antidote])
    pred,error = MF.fit_model(ratings)
    V = MF.get_V()
    U = MF.get_U().loc[X.index]
    U_tilde = MF.get_U().loc[X_antidote.index]
    return U, U_tilde, V, error

def obj_fun(X_input, X, MF, utility, budget, direction, init=None):
    X_antidote = X_input.reshape((budget,X.shape[1]))
    new_users = ['u%d'%(i+1) for i in range(budget)]
    X_antidote = pd.DataFrame(X_antidote, index=new_users, columns=X.columns)        
    U, U_tilde, V, error = theta(MF,X,X_antidote,init)
    obj = np.sign(direction) * utility.evaluate(U.dot(V))
    return obj

def obj_fun_gradient(X_input, X, MF, utility, budget, direction, init=None):
    X_antidote = X_input.reshape((budget,X.shape[1]))
    new_users = ['u%d'%(i+1) for i in range(budget)]
    X_antidote = pd.DataFrame(X_antidote, index=new_users, columns=X.columns)        
    U, U_tilde, V, error = theta(MF,X,X_antidote,init)
    G = compute_gradient(MF,utility,X,X_antidote,U,V,U_tilde)
    return np.sign(direction)*G.flatten()
        
def optimal_stepsize(MF,X,X_antidote,utility,G,direction,projection,steps):
    results = []
    for alpha in steps:
        X_antidote_new = X_antidote - (direction*alpha*G)
        X_antidote_new = projection.project(X_antidote_new)
        U, U_tilde, V, error = theta(MF,X,X_antidote_new)
        obj = utility.evaluate(U.dot(V))
        results.append((alpha,obj))
    best = min(results,key=lambda a:a[1]) if direction>0 else max(results,key=lambda a:a[1])
    print "optimal step:", best[0]
    return best[0]
    
def LineSearch_steps(max_step,num_steps):
    return [(1.0*max_step)/10**k for k in range(num_steps)]
    
def single_antidote_hueristic1(MF, X, projection, utility, direction,threshold):
    n,d = X.shape
    initial_data = 2.5 * np.ones((1,d))
    X_antidote = pd.DataFrame(initial_data, index=['u1'], columns=X.columns)
    U, U_tilde, V, error = theta(MF,X,X_antidote)
    G = compute_gradient_fast(MF,utility,X,X_antidote,U,V,U_tilde)
    a = (-1.0*np.sign(direction))*G
    a = np.where(np.abs(a)<threshold, 0, a)
    return (np.sign(a)+1.0)*(projection.max_edge/2.0)

def generate_antidote_single_step(MF, X, budget, projection, utility, direction,threshold):
    n,d = X.shape
    initial_data = 2.5 * np.ones((budget,d))
    new_users = ['u%d'%(i+1) for i in range(budget)]
    X_antidote = pd.DataFrame(initial_data, index=new_users, columns=X.columns)
    U, U_tilde, V, error = theta(MF,X,X_antidote)
    G = compute_gradient_fast(MF,utility,X,X_antidote,U,V,U_tilde)
    a = (-1.0*np.sign(direction))*G
    a = np.where(np.abs(a)<threshold, 0, a)
    X_tilde = (np.sign(a)+1.0)*(projection.max_edge/2.0)
    X_antidote = pd.DataFrame(X_tilde, index=new_users, columns=X.columns)
    return X_antidote

def single_antidote_hueristic2(X_est, U, projection, utility, direction,threshold):
    G = utility.gradient(X_est)
    b = U.sum(axis=1)
    gradient_appr = b.dot(G)
    a = (-1.0*np.sign(direction))*gradient_appr
    a = np.where(np.abs(a)<threshold, 0, a)
    return (np.sign(a)+1.0)*(projection.max_edge/2.0)
    
def generate_random_antidote(X, budget, max_edge):
    n,d = X.shape
    initial_data = max_edge*np.random.rand(budget,d)
    new_users = ['u%d'%(i+1) for i in range(budget)]
    X_antidote = pd.DataFrame(initial_data.copy(), index=new_users, columns=X.columns) 
    return X_antidote


class projection():
    
    def __init__(self,bounds):
        self.min_edge = bounds[0]
        self.max_edge = bounds[1]
        
    def project(self, X):
        return np.clip(X,self.min_edge,self.max_edge)


class opt_alg():
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def optimize():
        pass
        
    def run(self, MF, X, budget, projection, utility, init, n_runs):
        n,d = X.shape
        
        results = []
        for i in range(n_runs):
            print "Run %d"%(i+1)
            if isinstance(init, basestring):
                if init == 'random':
                    initial_data = projection.max_edge*np.random.rand(budget,d)
                else:
                    print 'initialization unknown!'
            else:
                initial_data = init
            R = self.optimize(MF, X, budget, projection, utility, initial_data)
            results.append((R,R['optimal_obj']))
        best = min(results,key=lambda a:a[1]) if self.stepsize>0 else max(results,key=lambda a:a[1])
        return best[0]

    
class gradient_descent(opt_alg):
    
    def __init__(self, max_iter, stepsize, threshold):
        self.max_iter = max_iter
        self.stepsize = stepsize
        self.threshold = threshold
                 
    def optimize(self, MF, X, budget, projection, utility, initial_data):
        new_users = ['u%d'%(i+1) for i in range(budget)]
        X_antidote = pd.DataFrame(initial_data.copy(), index=new_users, columns=X.columns)        
        
        X_antidote_hist = []
        obj_value_hist = []
        obj_value_before_proj_hist = []
        error_hist = []
        
        obj = np.sign(self.stepsize)*np.inf
        for i in range(self.max_iter):
            #print i
            U, U_tilde, V, error = theta(MF,X,X_antidote)
            obj_new = utility.evaluate(U.dot(V))
            #print obj_new
            if np.sign(self.stepsize)*(obj - obj_new) < self.threshold*obj:
                break
            obj = obj_new
            X_antidote_hist.append(X_antidote.copy())
            error_hist.append(error)
            obj_value_hist.append(obj)
            
            G = compute_gradient(MF,utility,X,X_antidote,U,V,U_tilde)
            X_antidote_new = X_antidote - (self.stepsize*G)
            X_antidote = projection.project(X_antidote_new)
            
        return {'X_antidote_history':X_antidote_hist,'obj_hist':obj_value_hist,
                'error':error_hist,
                'optimal_obj':obj_value_hist[-1],'X_antidote':X_antidote_hist[-1]}


class gradient_descent_LS(opt_alg):
    
    def __init__(self, max_iter, stepsize, threshold, steps, window):
        self.max_iter = max_iter
        self.stepsize = stepsize
        self.threshold = threshold
        self.steps = steps
        self.window = window
    
    def optimize(self, MF, X, budget, projection, utility, initial_data):
        new_users = ['u%d'%(i+1) for i in range(budget)]
        X_antidote = pd.DataFrame(initial_data.copy(), index=new_users, columns=X.columns)
        
        X_antidote_hist = []
        obj_value_hist = []
        error_hist = []
        
        obj = np.sign(self.stepsize)*np.inf
        t=1
        first_iteration = True
        
        for i in range(self.max_iter):
            U, U_tilde, V, error = theta(MF,X,X_antidote)
            obj_new = utility.evaluate(U.dot(V))
            print "obj_new: ",obj_new
            if np.abs(obj - obj_new) < 1e-6:
                break
            if np.sign(self.stepsize)*(obj - obj_new) < self.threshold*obj:
                if t > self.window:
                    break
                else:
                    t+= 1
            else:
                obj = obj_new
                t = 1
            X_antidote_hist.append(X_antidote.copy())
            error_hist.append(error)
            obj_value_hist.append(obj_new)
            
            G = compute_gradient_fast(MF,utility,X,X_antidote,U,V,U_tilde)
            
            #choose step size
            if first_iteration:
                first_iteration = False
                if self.steps is None:
                    power = np.rint(np.log10(np.abs(G).max()))
                    
                    #~ largest_step = -power+6 #polarization
                    #~ steps = LineSearch_steps(10**largest_step,8)
                    
                    largest_step = -power+5 
                    steps = LineSearch_steps(10**largest_step,8)
                    
                    #~ largest_step = -power+3 #genres
                    #~ steps = LineSearch_steps(10**largest_step,6)
                else:
                    steps = self.steps
                print "steps",steps
            
            #~ print np.sort(G.flatten())
            #~ print np.argmax(G)
            #print "max G",np.abs(G).max()

            alpha = optimal_stepsize(MF,X,X_antidote,utility,G,np.sign(self.stepsize),projection,steps)
            
            X_antidote_new = X_antidote - (np.sign(self.stepsize)*alpha*G)
            X_antidote = projection.project(X_antidote_new)
            
            optimal_obj = np.min(obj_value_hist) if self.stepsize>0 else np.max(obj_value_hist)
            optimal_X_antidote = X_antidote_hist[np.argmin(obj_value_hist)] if self.stepsize>0 else X_antidote_hist[np.argmax(obj_value_hist)]
        return {'X_antidote_history':X_antidote_hist,'obj_hist':obj_value_hist,
                'error':error_hist,
                'optimal_obj':optimal_obj,'X_antidote':optimal_X_antidote}


class coordinate_descent(opt_alg):
    
    def __init__(self, max_iter, stepsize, threshold, steps, window):
        self.max_iter = max_iter
        self.stepsize = stepsize
        self.threshold = threshold
        self.steps = steps
        self.window = window
    
    def optimal_stepsize(self,MF,X,X_antidote,utility,G,projection,row):
        results = []
        for alpha in self.steps:
            X_antidote_new = X_antidote.copy()
            X_antidote_new.iloc[row] = X_antidote_new.iloc[row] - (np.sign(self.stepsize)*alpha*G)
            X_antidote_new = projection.project(X_antidote_new)
            U, U_tilde, V, error = theta(MF,X,X_antidote_new)
            obj = utility.evaluate(U.dot(V))
            results.append((alpha,obj))
        best = min(results,key=lambda a:a[1]) if self.stepsize>0 else max(results,key=lambda a:a[1])
        return best[0]

    def d_theta_wrt_antidote_row(self,U, V, U_tilde, W, W_antidote, row, lambda_):
        """
        Returns the gradient of each element of the optimal 
        factor V with respect to each element of given row of antidote data.
        The output is a 1*d by l*d matrix that contains aproximated 
        partial derivates computed based on the method in [li2016data].
        """
        n,l = U.shape
        d = V.shape[0]
        n_prime = U_tilde.shape[0]
                   
        sigma_V = [U.T.dot(np.diag(1.0*W.T[j])).dot(U) +
                   U_tilde.T.dot(np.diag(1.0*W_antidote.T[j])).dot(U_tilde) for j in range(d)]
        
        diag = [np.linalg.inv(sigma_V[j] + lambda_*np.eye(l)).dot(U_tilde[row]) for j in range(d)]
        D = sp.linalg.block_diag(*diag)
        return D
    
    def compute_coordinate_gradient(self,MF,utility,X,X_antidote,U,V,U_tilde,row):#gradient w.r.t each row
        X_est = U.dot(V)
        U = U.values
        U_tilde = U_tilde.values
        V = V.T.values
        W = ~np.isnan(X).values
        W_antidote = ~np.isnan(X_antidote).values
        
        G1 = self.d_theta_wrt_antidote_row(U,V,U_tilde,W,W_antidote,row,MF.lambda_)
        G2 = d_est_wrt_theta(U,V)
        G3 = utility.gradient(X_est).flatten() #d_utility_wrt_est

        n,d = X.shape
        if n*d<10000:
            gradient = G1.dot(G2).dot(G3)
        else:        
            G1 = csr_matrix(G1)
            G2 = csr_matrix(G2)
            gradient = G1.dot(G2.dot(G3))

        return gradient

    def optimize(self, MF, X, budget, projection, utility, initial_data):
        new_users = ['u%d'%(i+1) for i in range(budget)]
        X_antidote = pd.DataFrame(initial_data.copy(), index=new_users, columns=X.columns)
        
        X_antidote_hist = []
        obj_value_hist = []
        error_hist = []
        
        obj = np.sign(self.stepsize)*np.inf
        t=0
        rows = range(budget)
        random.shuffle(rows)
        coordinates = itertools.cycle(rows)
        for i in range(self.max_iter):
            #print i
            U, U_tilde, V, error = theta(MF,X,X_antidote)
            obj_new = utility.evaluate(U.dot(V))
            #print obj_new
            if np.sign(self.stepsize)*(obj - obj_new) < self.threshold*obj:
                if t > self.window:
                    break
                else:
                    t+= 1
            else:
                obj = obj_new
                t = 0
                
            X_antidote_hist.append(X_antidote.copy())
            error_hist.append(error)
            obj_value_hist.append(obj_new)
            
            #row = random.choice(range(budget))
            row = coordinates.next()
            #print "row:%d"%row
            G = self.compute_coordinate_gradient(MF,utility,X,X_antidote,U,V,U_tilde,row)
            alpha = self.optimal_stepsize(MF,X,X_antidote,utility,G,projection,row)
            X_antidote.iloc[row] = X_antidote.iloc[row] - (np.sign(self.stepsize)*alpha*G)
            X_antidote = projection.project(X_antidote)
            
            optimal_obj = np.min(obj_value_hist) if self.stepsize>0 else np.max(obj_value_hist)
            optimal_X_antidote = X_antidote_hist[np.argmin(obj_value_hist)] if self.stepsize>0 else X_antidote_hist[np.argmax(obj_value_hist)]
        return {'X_antidote_history':X_antidote_hist,'obj_hist':obj_value_hist,
                'error':error_hist,
                'optimal_obj':optimal_obj,'X_antidote':optimal_X_antidote}

    
class random_descent_LS(opt_alg):


    def __init__(self, max_iter, stepsize, threshold, steps, window):
        self.max_iter = max_iter
        self.stepsize = stepsize
        self.threshold = threshold
        self.steps = steps
        self.window = window
    
    def optimize(self, MF, X, budget, projection, utility, initial_data):
        new_users = ['u%d'%(i+1) for i in range(budget)]
        X_antidote = pd.DataFrame(initial_data.copy(), index=new_users, columns=X.columns)        
        
        X_antidote_hist = []
        obj_value_hist = []
        error_hist = []
        
        obj = np.sign(self.stepsize)*np.inf
        t=0
        for i in range(self.max_iter):
            #print i
            U, U_tilde, V, error = theta(MF,X,X_antidote)
            obj_new = utility.evaluate(U.dot(V))
            #print obj_new
            if np.sign(self.stepsize)*(obj - obj_new) < self.threshold*obj:
                if t > self.window:
                    break
                else:
                    t+= 1
            else:
                obj = obj_new
                t = 0
                
            X_antidote_hist.append(X_antidote.copy())
            error_hist.append(error)
            obj_value_hist.append(obj_new)
            
            G = np.random.rand(budget,X.shape[1]) - 0.5
            alpha = optimal_stepsize(MF,X,X_antidote,utility,G,-1.0,projection,self.steps)
            X_antidote_new = X_antidote + alpha*G
            X_antidote = projection.project(X_antidote_new)
            
            optimal_obj = np.min(obj_value_hist) if self.stepsize>0 else np.max(obj_value_hist)
            optimal_X_antidote = X_antidote_hist[np.argmin(obj_value_hist)] if self.stepsize>0 else X_antidote_hist[np.argmax(obj_value_hist)]
        return {'X_antidote_history':X_antidote_hist,'obj_hist':obj_value_hist,
                'error':error_hist,
                'optimal_obj':optimal_obj,'X_antidote':optimal_X_antidote}


class scipy_opt():


    def __init__(self, max_iter, direction, method, input_gradient=False,display=False):
        self.max_iter = max_iter
        self.direction = direction
        self.method = method
        self.input_gradient = input_gradient
        self.display = display
    
    def run(self, MF, X, budget, projection, utility, init, n_runs):
        n,d = X.shape
        
        results = []
        for i in range(n_runs):
            #print 'run:',i
            if isinstance(init, basestring):
                if init == 'random':
                    initial_data = projection.max_edge*np.random.rand(budget,d)
                else:
                    print 'initialization unknown!'
            else:
                initial_data = init
            
            new_users = ['u%d'%(i+1) for i in range(budget)]
            X_antidote_initial = pd.DataFrame(initial_data.copy(), index=new_users, columns=X.columns)
            
            U, U_tilde, V, error = theta(MF,X,X_antidote_initial)
            initial_obj = utility.evaluate(U.dot(V))
            
            if self.input_gradient:
                jac = obj_fun_gradient
            else:
                jac = None
            
            X0 = X_antidote_initial.values.flatten()
            bounds = [(projection.min_edge,projection.max_edge)]*len(X0)
            
            OptimizeResult = minimize(obj_fun, X0, args=(X,MF,utility,budget,self.direction), method=self.method, 
                                                     jac=jac, bounds=bounds, 
                                                     options={'maxiter':self.max_iter,'disp':self.display})

            X_antidote = pd.DataFrame(OptimizeResult.x.reshape((budget,d)), index=new_users, columns=X.columns)
            X_antidote = projection.project(X_antidote)
            
            U, U_tilde, V, error = theta(MF,X,X_antidote)
            optimal_obj = utility.evaluate(U.dot(V))
            optimal_obj_before_proj = OptimizeResult.fun if self.direction>0 else -1.0*(OptimizeResult.fun)
            
            R = {'X_antidote_initial':X_antidote_initial,
                 'X_antidote':X_antidote,
                 'X_antidote_history':[X_antidote_initial,X_antidote],
                 'optimal_obj':optimal_obj,
                 'obj_hist':[initial_obj,optimal_obj],
                 'optimal_obj_before_proj':optimal_obj_before_proj,
                 'OptimizeResult':OptimizeResult}
            results.append((R,R['optimal_obj']))
        best = min(results,key=lambda a:a[1]) if self.direction>0 else max(results,key=lambda a:a[1])
        return best[0]
