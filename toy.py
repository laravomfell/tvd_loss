#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:15:17 2020

@author: jeremiasknoblauch

Description:
    Class hierarchy
"""

import autograd.numpy as np
import autograd.numpy.random as npr
#from autograd.scipy.misc import logsumexp
from autograd.scipy.stats import poisson
from autograd import grad

import matplotlib.pyplot as plt

import statsmodels.api as sm


# 

""" 

Class instantiator

"""

class TVDMaster():
    """
    Role: This class contains all objects relevant to inference, including
          model description, data and the inference algorithm
          
    Attributes:
        
    
    """
    
    def __init__(self, X, Y, lik):
        self.X = X
        self.Y = Y
        self.lik = lik        
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.params = None
        
        
    def get_histogram(self):    
        
        # there must be some in-built function for this functionality, 
        # but I could not find it.
        #
        # what it does: it takes an np.array X which contains each entry of
        #               a second np.array X_unique at least once and returns an 
        #               np.array ind of indices such that ind[i] is the index 
        #               of X_unique which maps onto the i-1 th entry of X.
        #               I.e., X[ind[i]] = X_unique[i]
        def find_unique_indices(X_, X_unique):                        
            myL = []
            for x in X_:
                ind = 0
                for xu in X_unique:
                    if len(X_.shape) > 1:
                        if (x == xu).all():
                            myL += [ind]
                    else:
                        if x == xu:
                            myL += [ind]
                    ind += 1
            return np.array(myL)
        
        # get unique vals for X, the indices mapping X to the unique vals
        # as well as the empirical pmf. Do the same for Y and YX
        self.X_unique, self.X_unique_indices, X_counts = np.unique(
            self.X, return_counts = True, return_inverse = True, axis=0)
        self.X_pmf = X_counts / self.n
        
        self.Y_unique, self.Y_unique_indices, Y_counts = np.unique(
            self.Y, return_counts = True, return_inverse = True, axis=0)
        self.Y_pmf = Y_counts / self.n
        
        self.YX_unique, self.YX_unique_indices, YX_counts = np.unique(
            np.hstack((np.atleast_2d(self.Y).T, self.X)), 
                return_counts = True, return_inverse = True, axis=0)
        self.YX_pmf = YX_counts / self.n
        
        # from the above, compute the conditional pmf of Y|X for all
        # y \in Y_unique and x \in X_unique
        
        # STEP 1: Find the mapping from the X in the joint distribution (Y,X) 
        #         to the same X in the marginal of X
        map_to_Y = find_unique_indices(self.YX_unique[:,0], self.Y_unique)
        map_to_X = find_unique_indices(self.YX_unique[:,1:], self.X_unique)
        
        # STEP 2: Create the relevant matrix and fill it in
        self.Y_cond_X_pmf = np.zeros((self.Y_unique.shape[0], 
                                      self.X_unique.shape[0]))
        
        
        # fill the matrix with the JOINT probabilities
        self.Y_cond_X_pmf[map_to_Y, map_to_X] = self.YX_pmf
        # divide by the MARGINAL probabilities
        self.Y_cond_X_pmf = self.Y_cond_X_pmf / self.X_pmf
    
    
    
    def run(self):
        
        # Step 1: construct the empirical pmfs
        self.get_histogram()
        
        # Step 2: Initialize the parameter vector
        # at the MLE values
        self.params = sm.GLM(self.Y, self.X, 
                             family = sm.families.Poisson()).fit().params
        
        # Step 3: Define/obtain gradient w.r.t. the parameters theta
        def TVD_loss(params):
            # define objective w.r.t. params
            
            # Get the number of unique X and Y observations
            n_X_unique = self.X_unique.shape[0]
            n_Y_unique = self.Y_unique.shape[0]
            
            
            #---------------------------------------------
            # PART DEPENDING ON LIKELIHOOD (SEPARATE OUT)
            #---------------------------------------------
            
            #Assign params their internal value
            coefs = params
            
            # Get the model pmf for all X and Y in the sample (exactly once),ie
            # evaluate p_{\theta}(y_j|x_i) for all unique y_j, x_i
            #
            # IM ALSO SURE THIS IS CORRECT

            lambdas = np.exp(np.matmul(self.X_unique, coefs))
            Y_cond_X_model = poisson.pmf(
                np.repeat(self.Y_unique,n_X_unique).reshape(n_Y_unique, n_X_unique),
                np.tile(lambdas, n_Y_unique).reshape(n_Y_unique, n_X_unique) 
                )
            

            
            # Get the model pmf for all Y \notin sample, i.e.
            # evaluate p_{\theta}(y \notin {y_1, ... y_n}|x_i) for all 
            # not-observed values of y. 
            # Easier to get as 1-\sum_{y \in sample}p_{\theta}(y|x_i) for all
            # unique values of x_i in the sample.
            #
            # LARA I'M PRETTY SURE THIS IS CORRECT
            remainder_lik = 1.0 - np.sum( 
                    Y_cond_X_model, axis=0)

            
            # Compute the TVD as follows: 1., observe that
            #
            #   E_{X}[ TVD( p_{\theta}(y|x) || \hat{p}(y|x) ) ]
            # = \sum_{x \in X} \hat{p}(x) TVD(p_{\theta}(y|x) || \hat{p}(y|x))
            # 
            # and 2. that for S_y being the unique values of y in the sample,
            #
            #   TVD(p_{\theta}(y|x) || \hat{p}(y|x))
            # = \sum_{y \in Y}|p_{\theta}(y|x) - \hat{p}(y|x)|
            # = \sum_{y \in S_y}|p_{\theta}(y|x) - \hat{p}(y|x)| + 
            #   \sum_{y \notin S_y}p_{\theta}(y|x)
            #
            # where the last equality follows because \hat{p}(y|x) = 0 
            # whenever y \notin S_y.
            # Note also that \hat{p}(x) = 0 whenever x \notin S_x, so that the
            # sum approximating the outer expectation is sparse!
            #
            # shape = scalar
            
            expected_TVD = 0.5 * np.sum(
                self.X_pmf * 
                ( np.sum(np.abs(Y_cond_X_model - self.Y_cond_X_pmf), axis=0) +
                  remainder_lik)
                )
                
            #print(expected_TVD)
                     
            return expected_TVD
    
        #STEP 4: Define the variational objective
        def variational_objective(kappa_params, num_theta_samples):
            
            # First, draw num_theta_samples from the current posterior
            thetas = np.random.normal()
        
                    
        gradient = grad(TVD_loss)
        
        
        second_order = False
        ADAM = True
        
        if second_order:
            from scipy.optimize import minimize
            
            x0 = np.ones(self.p) 
#            + npr.normal(0, 0.001, self.p)
            res = minimize(TVD_loss, x0, 
                           method= 'Nelder-Mead', 
                           jac=gradient,
                    options={'disp': True})
            
            self.params = res        
            self.fittedvalues = np.exp(np.matmul(self.X, self.params.x))
        
        
        if not second_order:
            
            if not ADAM:
            
                # Step 4: Perform iterative optimization
                for i in range(0,1000):
                    
                    step_size = 0.0001
                    
                    # Step 4.1: Compute gradient
                    gradient_value = gradient(self.params)
                    
                    # Step 4.2: Gradient step
                    self.params = self.params + step_size * gradient_value



        
            if ADAM:
                
                m1 = 0
                m2 = 0
                beta1 = 0.9
                beta2 = 0.999
                epsilon = 1e-8
                t = 0
                learning_rate = 0.001
                
                # Step 4: Perform iterative optimization
                for i in range(0,100):
                    t+=1
                    gradient_value = gradient(self.params)
                    
                    m1 = beta1 * m1 + (1 - beta1) * gradient_value
                    m2 = beta2 * m2 + (1 - beta2) * gradient_value**2
                    m1_hat = m1 / (1 - beta1**t)
                    m2_hat = m2 / (1 - beta2**t)
                    self.params -= learning_rate * m1_hat / (np.sqrt(m2_hat) + epsilon)
                    
                    print(TVD_loss(self.params))
                    
            self.fittedvalues = np.exp(np.matmul(self.X, self.params))
        
                
        plot = False    
        if plot:
            param_full = np.linspace(-10, 10, 1000)
            
            fct_val = np.zeros(1000)
        
            for (i,p) in zip(range(0,1000), param_full):
                fct_val[i] = TVD_loss(np.array([p]))
        
            self.param_full = param_full
            self.fct_val = fct_val
            
