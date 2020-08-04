#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:17:11 2019

@author: jeremiasknoblauch

Description: Class that does BB-GVI inference on BLR
    
"""

#from __future__ import absolute_import
#from __future__ import print_function
import jax.numpy as jnp
import numpy as np
import jax.random as random
from jax.scipy.special import logsumexp
from jax import grad

from DataPreProcessor import DataPreProcessor

import sys

import math


"""AUXILIARY OBJECT. Purpose is to provide wrapper for all q_params"""
class ParamParser(object):
    """A helper class to index into a parameter vector."""
    def __init__(self):
        self.idxs_and_shapes = {}
        self.num_params = 0

    def add_shape(self, name, shape):
        start = self.num_params
        self.num_params += np.prod(shape)
        self.idxs_and_shapes[ name ] = (slice(start, self.num_params), shape)

    def get(self, vect, name):
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[ idxs ], shape)

    def get_indexes(self, vect, name):
        idxs, shape = self.idxs_and_shapes[name]
        return idxs


class BBGVI():
    """This builds an object that performs Black Box GVI inference. 
    Internal states of the object:
        D       A divergence object (will work with a fixed prior)
        L       A loss object (determines the parameter wrapping)
        Y       A data vector
        X       A data matrix (optional)
        K       The number of samples drawn from q (default = 100)
        M       The number of samples drawn from (X,Y) (default = max(1000, len(Y)))
        n       = len(Y)
        q_params  The parameters of the MFN that we fit (= q)
        q_parser  The locations/names of the parameters 
    """
    
    def __init__(self, D, L, jax_key_seed = 0):
        """create the object"""
        
        #DEBUG: Still need to account for case where X = None!
        self.D, self.L = D, L
        self.jax_key = random.PRNGKey(jax_key_seed)
        
        self.q_parser, self.q_params, self.converter = self.make_q_params() 
    
    def set_jax_key(self, jax_key_seed):
        self.jax_key = random.PRNGKey(jax_key_seed)
          
    def draw_samples(self,q_params, jax_key_integer):
        if jax_key_integer is None:
            # in this case, the loss function takes care of changing the 
            # random seed. Every time we call the loss function without 
            # providing a key, its internal key is used. Afterwards, the  seed 
            # of this internal key is advanced by +1
            return self.L.draw_samples(self.q_parser, q_params, self.K,
                                   None)
        else:
            return self.L.draw_samples(self.q_parser, q_params, self.K,
                                       self.jax_key+jax_key_integer)
          
    def create_GVI_objective(self):
                
        """create the objective function that we want to differentiate"""
        def GVI_objective(q_params, q_parser, converter, Y_, X_, indices, 
                          jax_key_integer):           
            """Y_, X_ will be sub-samples of Y & X"""
            
            """Make sure to re-weight the M data samples by n/M"""
            q_sample = self.draw_samples(q_params, jax_key_integer)
            
            if False: #Great for debugging
                print("Loss:", jnp.mean(self.n * 
                    self.L.avg_loss(q_sample, Y_, X_, indices))
                    )
                print("Div:", self.D.prior_regularizer(q_params, q_parser, 
                                                       converter))
            
            return (jnp.mean(self.n * 
                    self.L.avg_loss(q_sample, Y_, X_, indices))
                    + self.D.prior_regularizer(q_params, q_parser, converter))
        
        return GVI_objective
        
        
        
    def make_q_params(self):
        """depending on the parameters (i.e., the loss), I need to 
        create a container for the q_params, too!"""
        parser = ParamParser()

        """Give the parser the right entries & names (for BLR)"""
        parser, params, converter = self.L.make_parser(parser)
        return (parser, params, converter)
    
    
    def fit_q(self, Y, X=None, K=100, M = 1000, 
              epochs = 500, learning_rate = 0.0001):
        """This function puts everything together and performs BB-GVI"""
        
        """STEP 0: Make sure our M is AT MOST as large as data"""
        self.K = K
        self.n = Y.shape[0]
        self.M = min(M, self.n)
        
        """STEP 1: Get objective & take gradient"""
        GVI_obj = self.create_GVI_objective()
        GVI_obj_grad = grad(GVI_obj)
        q_params = self.q_params
        
        
        """STEP 2: Sample from X, Y and perform ADAM steps"""

        """STEP 2.1: These are just the ADAM optimizer default settings"""  
        m1 = 0
        m2 = 0
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        t = 0
    
        """STEP 2.2: Loop over #epochs and take step for each subsample"""
        for epoch in range(epochs):
            
            """STEP 2.2.1: For each epoch, shuffle the data"""
            permutation = np.random.choice(range(Y.shape[ 0 ]), Y.shape[ 0 ], 
                                           replace = False)

            """HERE: Should add a print statement here to monitor algorithm!"""
            if epoch % 100 == 0:
                print("epoch #", epoch, "/", epochs)
                #print("sigma2", np.exp(-q_params[3]))
            
            """STEP 2.2.2: Process M data points together and take one step"""
            for i in range(0, int(self.n/self.M)):
                
                """Get the next M observations (or less if we would run out
                of observations otherwise)"""
                end = min(self.n, (i+1)*self.M)
                indices = permutation[(i*self.M):end]
                
                
                """ADAM step for this batch"""
                t+=1
                if X is not None:
                    grad_q_params = GVI_obj_grad(q_params,self.q_parser, self.converter,
                                             Y[ indices ], X[ indices,: ], indices,
                                             jax_key_integer=(t+1)*(epoch+1))
                else:
                    grad_q_params = GVI_obj_grad(q_params,self.q_parser, self.converter,
                                             Y[ indices ], X_=None, indices=indices,
                                             jax_key_integer=(t+1)*(epoch+1))
                
                
                m1 = beta1 * m1 + (1 - beta1) * grad_q_params
                m2 = beta2 * m2 + (1 - beta2) * grad_q_params**2
                m1_hat = m1 / (1 - beta1**t)
                m2_hat = m2 / (1 - beta2**t)
                q_params -= learning_rate * m1_hat / (np.sqrt(m2_hat) + epsilon)
            
        self.q_params = q_params
                

    def report_parameters(self):
        # get two lists of param names & values
        names, values = self.L.report_parameters(self.q_parser, self.q_params)        
        for i in range(0, len(names)):
            print(names[i], values[i])
        
        
        


class BBGVI_TVD(BBGVI):
    """This builds an object that performs Black Box GVI inference with the 
        TVD-loss. Note that the gradient steps always use all the data.
        
    Internal states of the object:
        Same as for BBGVI + 
        
        X_unique        Unique X-values in sample
        Y_unique        Unique Y-values in sample
        Y_cond_X_pmf    empirical pmf of Y|X
        
    """        
    
    def create_GVI_objective(self):
        # NOTE: Difference to vanilla objective: We need the pmfs to go 
        #       in there + the unique X and Y values 
        #       (all of which are stored in the histogram_object)
                
        """create the objective function that we want to differentiate"""
        def GVI_objective_JAX(q_params, q_parser, converter, histogram_object,
                          jax_key_integer):           
            """Y_, X_ will be sub-samples of Y & X"""
            
            """Make sure to re-weight the M data samples by n/M"""
            q_sample = self.draw_samples(q_params, jax_key_integer)
            
            if jax_key_integer % 100 == 0: #Great for debugging
                print("Loss:", jnp.mean(self.n * self.L.avg_loss(
                        q_sample, histogram_object)))
                print("Div:", self.D.prior_regularizer(q_params, 
                                                      q_parser, converter)
                    )
            
            return (jnp.mean(self.n * self.L.avg_loss(q_sample, 
                                histogram_object))
                    + self.D.prior_regularizer(q_params, q_parser, converter))
        
        return GVI_objective_JAX
    
    
    def create_GVI_objective_SCIPY(self, q_parser, converter, 
                                   histogram_object):
        # NOTE: Difference to vanilla objective: We need the pmfs to go 
        #       in there + the unique X and Y values 
        #       (all of which are stored in the histogram_object)
                
        """create the objective function that we want to differentiate"""
        def GVI_objective_SCIPY(q_params):           
            """Y_, X_ will be sub-samples of Y & X"""
            
            """Make sure to re-weight the M data samples by n/M"""
            q_sample = self.draw_samples(q_params, None)
            
            
            return (jnp.mean(self.n * self.L.avg_loss(q_sample, 
                                histogram_object))
                    + self.D.prior_regularizer(q_params, q_parser, converter))
        
        return GVI_objective_SCIPY
    
        
    # DEBUG: Need to deal with the case where X=None!
    def fit_q(self, Y, X=None, K=100, 
              epochs = 500, learning_rate = 0.0001, optimizer = "ADAM"):
        """This function puts everything together and performs BB-GVI"""
        
        """STEP 0: Make sure our M is AT MOST as large as data"""
        self.K = K
        self.n = Y.shape[0]
        self.M = self.n
        
        """initialize means of the variational distros at the MLE and
        get the empirical distributions that are needed later in the 
        optimization"""
        q_params = self.L.initializer(X, Y, self.q_params, self.q_parser)
        self.q_params_initializer = q_params
        
        # DEBUG: This does not work if I have X=None! So if X=None, my default
        #        assumption should be that I only have intercepts (meaning that
        #        in this case, we should set X = np.ones((self.n,1))
        histogram_object = DataPreProcessor().get_histogram(X,Y,self.n)
        
        
        """STEP 2: Perform ADAM steps"""
        if optimizer == "ADAM":
            
            """STEP 2.0: a) Get objective, b) define gradient"""
            GVI_obj = self.create_GVI_objective()
            GVI_obj_grad = grad(GVI_obj)

            """STEP 2.1: These are just the ADAM optimizer default settings"""  
            m1 = 0
            m2 = 0
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-8
            t = 0
        
            """STEP 2.2: Loop over #epochs and take step for each subsample"""
            for epoch in range(epochs):
                
                if epoch % 100 == 0:
                    print("epoch #", epoch, "/", epochs)
                    #print("sigma2", np.exp(-q_params[3]))
                
                """gradient step"""
                t+=1
                grad_q_params = GVI_obj_grad(q_params,self.q_parser, 
                        self.converter, histogram_object, jax_key_integer=(t+1)*(epoch+1))
                    
                
                DEBUG = False
                if DEBUG:
                    print("gradient value", grad_q_params)
                    obj = GVI_obj(q_params,self.q_parser, 
                        self.converter, histogram_object, jax_key_integer=(t+1)*(epoch+1))
                    print("objective value", obj)
            
                                  
                if np.any(np.isnan(grad_q_params)):
                    self.q_params = q_params 
                    print("Nans encountered." + 
                          " This is what the loss and divergence look like:")
                    
                    jax_key_integer=(t+1)*(epoch+1)
                    q_sample = self.draw_samples(q_params, jax_key_integer)
                    print("Loss:", jnp.mean(self.n * self.L.avg_loss(
                            q_sample, histogram_object)))
                    print("Div:", self.D.prior_regularizer(q_params, 
                             self.q_parser, self.converter)
                        )
                    
                    def func(q_params):
                        q_sample = self.draw_samples(q_params, jax_key_integer)
                        return jnp.mean(self.n * self.L.avg_loss(
                            q_sample, histogram_object))
                    
                    grad1 = grad(func)
                    grad2 = grad(self.D.prior_regularizer)
                    
                    print("Loss grad:", grad1(q_params))
                    print("Div grad:", grad2(q_params))
                        
                    
                    sys.exit("NAN IN GRADIENT! Computation aborted")
                    
                    
                # make sure we store the old gradient so that if it becomes nan
                # at the next iteration, we can just use the old gradient
                if epoch == 0:
                    average_grad = grad_q_params
                average_grad = (average_grad * (epoch) + grad_q_params)/(epoch + 1)
                             
                m1 = beta1 * m1 + (1 - beta1) * grad_q_params
                m2 = beta2 * m2 + (1 - beta2) * grad_q_params**2
                m1_hat = m1 / (1 - beta1**t)
                m2_hat = m2 / (1 - beta2**t)
                q_params -= learning_rate * m1_hat / (np.sqrt(m2_hat) + epsilon)
                
                
            self.q_params = q_params     
        
        # NOTE: We need to ensure jax uses 64-bits, otherwise 
        #       this won't work
        elif optimizer != "ADAM":
            
            # create an objective amenable to scipy optimizers
            GVI_obj = self.create_GVI_objective_SCIPY(self.q_parser, 
                            self.converter,  histogram_object)
            GVI_obj_grad = grad(GVI_obj)
        
            # make sure we have efficient second order optimizers
            from scipy.optimize import minimize
            
            # GVI objective amenable to scipy
            # NOTE: We need to ensure jax uses 64-bits, otherwise 
            #       this won't work
        
            x0 = self.q_params_initializer
#            + npr.normal(0, 0.001, self.p)
            res = minimize(GVI_obj, x0, 
                           method= optimizer,
                           jac=GVI_obj_grad,
                           options={'disp': True, 'maxiter': epochs})
            
            self.res = res
            self.q_params = res.x        

        
    def report_parameters(self):
        super().report_parameters()
        # additionally, also report the initialization/MLE.
        print("MLE: ", self.q_params_initializer)