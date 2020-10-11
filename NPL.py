#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 15:35:05 2020

@author: jeremiasknoblauch


Description: This implements a simplified version of nonparametric Bayesian
Learning as outlined by Fong et al. (2019) 
"""

import numpy as np
from scipy.stats import dirichlet
from scipy.optimize import minimize
from scipy.stats import poisson
from likelihood_functions import SoftMaxNN


class NPL():
    """This class contains everything that is needed for inference with a 
    given model. To this end, the user supplies
        likelihood  A likelihood function
        optimizer   The (scipy) optimizer to be used for optimization
    """
    
    def __init__(self, likelihood, optimizer="BFGS"):
        self.lklh = likelihood
        self.optimizer = optimizer
        
    
    def draw_samples(self, Y, X, B, seed = 0, display_opt = True):
        """Draws B samples from the nonparametric posterior specified via
        the likelihood and the data (Y, X)"""
            
        # compute the empirical measures and associated quantities
        self.get_empirical_measures(Y,X)
        
        # create/re-set to zero objects to log all the optimization
        self.create_recorders(B)
        
        sample = []
        mle = []
        
        
        for j in range(0, B):
            
            # draw Dirichlet weights
            weights = dirichlet.rvs(np.ones(self.n), size = 1, 
                                    random_state = seed).flatten()
            
            # if we want, we can also initialize anew for each weight sample
            initializer = self.lklh.initialize(Y,X, weights)
            mle = mle + [initializer]
            
            # compute the minimum
            theta_j = self.minimize_TVD(initializer, weights, display_opt, j)
            
            # add to sample
            sample = sample + [theta_j]
            
            # add +1 to the seed for the next sample
            seed = seed + 1
            
            
        
        # if we store a NN, then we get layer-wise np arrays (for each sample).
        # thus, we cannot numpy-force these
        if isinstance(self.lklh, SoftMaxNN):
            self.sample = sample
            self.mle = mle
        else:
            self.sample = np.array(sample)
            self.mle = np.array(mle)
    
    def predict(self, Y,X):
        return self.lklh.predict(Y,X,self.sample)
    
    def predict_log_loss(self, Y, X):
        return self.lklh.predict(Y,X,self.mle)
    
    def minimize_TVD(self, initializer, weights, display_opt, iteration):
        """Depending on the likelihood function, we can either use standard
        optimizers (scipy-built BFGS) or we have to use pytorch's methods
        (if the likelihood function corresponds to a NN)"""
        if isinstance(self.lklh, SoftMaxNN):
            return self.minimize_TVD_NN(
                                initializer, weights, display_opt, iteration)
        else:
            return self.minimize_TVD_scipy(
                                initializer, weights, display_opt, iteration)
    
    def minimize_TVD_NN(self, initializer, weights, display_opt, iteration):
        """This function uses self.lklh and its inputs to compute the TVD
        between the (re-weighted) empirical measures and the model using 
        first-order optimizers from pytorch (SGD)"""
        
        # call directly back to likelihood function (assuming that each 
        # observation is *unique*, which is true for out experiments 
        # since we will have  at least *one* continuous covariate)
        
        # It's slightly inelegant to shift this minimization into the loss
        # function (from a software point of view), but what happens there is
        # essentially the same to what happens inside the _scipy function.
        
        return self.lklh.minimize_TVD(self.Y_unique, self.X_unique, weights)
        
    
    def minimize_TVD_scipy(self, initializer, weights, display_opt, iteration):
        """This function uses self.lklh and its inputs to compute the TVD
        between the (re-weighted) empirical measures and the model using 
        second-order optimizers from scipy (BFGS)"""
        
        # call to a separate function that performs the re-weighting step
        rw_X_pmf, rw_Y_given_X_pmf = self.reweight_empirical_measures(weights)
        
        # define function to estimated TVD between (re-weighted)
        # empirical measure and the model likelihood for a given parameter 
        # value params
        def TVD(params):
            
            # Get the number of unique X and Y observations
            # n_X_unique = self.X_unique.shape[0]
            # n_Y_unique = self.Y_unique.shape[0]
            
            # Get the model pmf for all X and Y in the sample (exactly once),ie
            # evaluate p_{\theta}(y_j|x_i) for all unique y_j, x_i
            Y_cond_X_model = self.lklh.evaluate(params, self.Y_unique, 
                    self.X_unique)


            # DEBUG: This should be separated out into a likelihood function
            # so that the interface is like self.lklh(params, args...) here.
            # (see above)
            # lambdas = np.exp(np.matmul(self.X_unique, coefs))
            # Y_cond_X_model = poisson.pmf(
            #     np.repeat(self.Y_unique,n_X_unique).reshape(n_Y_unique, n_X_unique),
            #     np.tile(lambdas, n_Y_unique).reshape(n_Y_unique, n_X_unique) 
            #     )
            

            
            # Get the model pmf for all Y \notin sample, i.e.
            # evaluate p_{\theta}(y \notin {y_1, ... y_n}|x_i) for all 
            # not-observed values of y. 
            # Easier to get as 1-\sum_{y \in sample}p_{\theta}(y|x_i) for all
            # unique values of x_i in the sample.
            #
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
            estimated_TVD = 0.5 * np.sum(
                rw_X_pmf * 
                ( np.sum(np.abs(Y_cond_X_model - rw_Y_given_X_pmf), axis=0) +
                  remainder_lik)
                )
            
            return estimated_TVD
        
        
        # call the relevant scipy optimizer routine, initialized at the 
        # appropriate value (i.e., initializer)
        optimization_result = minimize(TVD, initializer, 
                           method= self.optimizer,
                           options={'disp': display_opt, 'maxiter': 100})
        
        # store relevant meta-information to check if all optimizations 
        # were performed successfully
        self.record_optimization_outcomes(optimization_result, iteration)
        
        # return the value at optimum
        return optimization_result.x
        
        
    def reweight_empirical_measures(self, weights):
        """use the outputs of 'get_empirical_measures' to efficiently modify
        the empirical measure with the weights vector from a dirichlet.
        
        Specifically, use the X_unique_indices arrays to match each entry in  
        the weights vector to the correct entry in the X_unique values + do 
        the same for the YX_unique_incides/YX_unique values.   
        NOTE: Doing so computes the reweighted marginal+joint pmfs!"""
            
        rw_X_pmf = np.zeros(self.X_pmf.shape)
        rw_YX_pmf = np.zeros(self.YX_pmf.shape)
        
        # loop from 1:n and match the weights. This computes the marginal
        # and joint (reweighted) pmfs. (Note that if all weights were 1/n, the
        # below would just fill the rw_X_pmf / rw_YX_pmf arrays with relative
        # frequencies.)
        for i in range(0, self.n):
            rw_X_pmf[self.X_unique_indices[i]] += weights[i]
            rw_YX_pmf[self.YX_unique_indices[i]] += weights[i]
    
        # repeat the steps from 'get_empirical_measures' that produced 
        # the conditionals from the joint and marginal            
        rw_Y_given_X_pmf = np.zeros((self.Y_unique.shape[0], 
                                      self.X_unique.shape[0]))
        rw_Y_given_X_pmf[self.map_to_Y, self.map_to_X] = rw_YX_pmf
        rw_Y_given_X_pmf = rw_Y_given_X_pmf / rw_X_pmf
        
        return (rw_X_pmf, rw_Y_given_X_pmf)
    
    
    def create_recorders(self, B):
        """create the objects that record the optimization outcomes"""
        
        self.successful_terminations = np.ones(B,dtype=bool)
        self.termination_status = np.ones(B,dtype=int)
        self.number_iterations = np.zeros(B, dtype=int)
        
    def record_optimization_outcomes(self, optimization_result, iteration):
        """log the most important things about the optimization routine. 
        This functions like a diagnostic check in regular sampling methods:
        If individual optimizations did not complete, we can exclude the 
        relevant samples. Alternatively, we can change the optimizer if the 
        current one does not find the minima"""
        
        self.successful_terminations[iteration] = optimization_result.success
        self.termination_status[iteration] = optimization_result.status
        self.number_iterations[iteration] = optimization_result.nit
    
    
    def get_empirical_measures(self, Y,X):
        """compute and return 
            n (the number of observations) 
            the empirical pmf for X 
            the empirical pmf for Y|X
            the mapping from the observations to the unique values.
        The last output is needed for the  re-weighting operations later"""
        
        # get n
        n = Y.shape[0]
        self.n = n
        
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
        X_unique, X_unique_indices, X_counts = np.unique(
            X, return_counts = True, return_inverse = True, axis=0)
        X_pmf = X_counts / n
        
        Y_unique, Y_unique_indices, Y_counts = np.unique(
            Y, return_counts = True, return_inverse = True, axis=0)
        Y_pmf = Y_counts / n
        
        
        YX_unique, YX_unique_indices, YX_counts = np.unique(
            np.hstack((np.atleast_2d(Y).T, X)), 
                return_counts = True, return_inverse = True, axis=0)
        YX_pmf = YX_counts / n
        
        # from the above, compute the conditional pmf of Y|X for all
        # y \in Y_unique and x \in X_unique
        
        # STEP 1: Find the mapping from the X in the joint distribution (Y,X) 
        #         to the same X in the marginal of X
        map_to_Y = find_unique_indices(YX_unique[:,0], Y_unique)
        map_to_X = find_unique_indices(YX_unique[:,1:], X_unique)
        
        # STEP 2: Create the relevant matrix and fill it in
        Y_given_X_pmf = np.zeros((Y_unique.shape[0], 
                                      X_unique.shape[0]))
        
        
        # fill the matrix with the JOINT probabilities
        Y_given_X_pmf[map_to_Y, map_to_X] = YX_pmf
        # divide by the MARGINAL probabilities
        Y_given_X_pmf = Y_given_X_pmf / X_pmf
        
        # store the objects that will be needed later to compute re-weighted
        # empirical measures
        self.map_to_Y = map_to_Y
        self.map_to_X = map_to_X
        self.X_unique_indices = X_unique_indices
        self.YX_unique_indices = YX_unique_indices
        self.X_unique = X_unique
        self.Y_unique = Y_unique
        
        # store the original pmfs (without weighting)
        self.X_pmf = X_pmf
        self.YX_pmf = YX_pmf
        self.Y_given_X_pmf = Y_given_X_pmf
        
        
        
        
        