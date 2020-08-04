#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 18:49:30 2019

@author: jeremiasknoblauch

Description: Loss objects (generic superclass and a few special cases)

"""

#from __future__ import absolute_import
#from __future__ import print_function
import jax.numpy as jnp
import numpy as np
import jax.random as random
from jax.scipy.special import logsumexp
from jax import grad

from jax.scipy.stats import poisson
import statsmodels.api as sm

import math
import sys


class Loss():
    """Compute loss between data & parameters, create q_parameter objects for
    the BBGVI class etc. Internal states of the object are none because
    it is EMPTY/ABSTRACT.            
    """

    def __init__(self):
        self.losstype = 0
    
    def make_parser(self, parser):
        return 0
    
    def get_num_global_params(self):
        return 0 #return number of variables we need posterior for
    
    def draw_samples(self, q_parser, q_params, K):
        return 0
        
    def avg_loss(self, params, parser, Y_, X_):
        return 0
    
    def report_parameters(self, params, parser):
        return 0
    

class TvdLossPoissonGLM(Loss):
    """Compute the TVD loss between a Poisson GLM model and empirical pmf.
        Poisson parameter is modelled/given by
        
        \lambda|x_1, ... x_d = \exp(\sum_{i=1}^n \theta_i * x_i)
        
        
        NOTE: Unlike other loss classes, we need to pass in the UNIQUE values
              for X and Y. 
    """
    
    def __init__(self, d, weight=1.0):
        self.d = d
        # this object is only used if 'None' is passed into the 'draw_samples'
        # function. This ensures we can use the loss functions with scipy
        # optimizers, as it gives the loss a way to change its own seed
        self.jax_prng_key = random.PRNGKey(0)
        self.weight = weight
        
        
    def initializer(self, X, Y, params, parser):
        """Returns the Maximum Likelihood Estimate, which we use to initialize
        the variational parameters"""
        MLE = sm.GLM(Y, X, family = sm.families.Poisson()).fit().params
        params[ parser.get_indexes(params, 'mean') ] = MLE
        return params
    
    
    def make_parser(self, parser):
        """Create the parser object"""       
        parser.add_shape('mean', (self.d, 1))
        parser.add_shape('log_variance', (self.d,1))
        
        """Initialize the parser object"""
        params = 0.1 * np.random.randn(parser.num_params)        
        params[ parser.get_indexes(params, 'log_variance') ] = (
            params[ parser.get_indexes(params, 'log_variance') ] - np.log(0.001))
        
        log_conversion = None #np.zeros((self.d, 1), dtype=bool)        

        return (parser, params, log_conversion)
    
    def get_num_global_params(self):
        return self.d
    
    
    def draw_samples(self, q_parser, q_params, K, jax_prng_key):
        
        """Use parser & params to produce K samples of the parameters for which
        q produces a posterior distribution. Do this row-wise."""

        v_c = jnp.exp(-q_parser.get(q_params, 'log_variance'))[ :, 0 ]
        m_c = q_parser.get(q_params, 'mean')[ :, 0 ] 
        
        # make sure that we definitely don't always get the same jax key
        if jax_prng_key is None:
            jax_prng_key = self.jax_prng_key
            self.jax_prng_key = self.jax_prng_key+1

        coef_sample = random.normal(jax_prng_key, (K, self.d)) * jnp.sqrt(v_c) + m_c
        
        return coef_sample
    
    
    
    def get_lambdas(self, X_unique, coef_sample):
        """This loss function uses the GLM reparameterization of lambda|X, 
           I.e. we use lambda(x) = exp(theta * x) 
        """
        return jnp.transpose(
                jnp.exp(jnp.matmul(X_unique, jnp.transpose(coef_sample)))
            )
    
    
    def avg_loss(self, q_sample, histogram_object):
        """Retrieve the parameters (using the parser) and compute the 
        average tvd across the q_sample as well as the X/Y sub-sample
        
        NOTE: Unlike with the over losses, we need the pre-processed UNIQUE
                values of X and Y being passed to the loss function (together)
                with the empirical pmfs
        """ 
        
        
        """Note 1: The q_sample is a sample from the coefficients b in 
                
                \lambda|X = \exp{\theta * X}
                Y|\lambda ~ Poisson (\lambda)
        
        And so we compute for each b-sample the term X * b = Y_hat"""
        X_unique, Y_unique, X_pmf, Y_cond_X_pmf = histogram_object
        
        coef_sample = q_sample
        K = coef_sample.shape[0]
        n_Y_unique = len(Y_unique)
        n_X_unique = X_unique.shape[0]
        
        
        # X_Unique is (n_X_unique, d), coef_sample is (K, d), so lambdas 
        # will be (n_X_unique, K)
        
        
        # shape = (K, n_X_unique)
        lambdas = self.get_lambdas(X_unique, coef_sample)
        
        
        # Compute the conditional model probabilities, i.e. p(Y=y|X=x) for 
        # all values of y and x that were observed in-sample"""
        #
        # I.e., we have K blocks (stacked) so that the k-th block has shape
        # (n_Y_unique, n_X_unique) and is the model pmf for the k-th sample
        
        # Y_K_blocks, lambda_blocks shapes = (n_Y_unique*K, n_X_unique)
        Y_block = jnp.repeat(Y_unique, n_X_unique).reshape(n_Y_unique, n_X_unique)
        # create K identical blocks of the unique Ys (rows) and Xs (cols)
        Y_K_blocks = jnp.tile(Y_block, (K,1)) 
        # repeat each  of the K rows of lambdas n_Y_unique times
        lambdas_blocks = jnp.tile(lambdas, (1,n_Y_unique)).reshape(K*n_Y_unique, n_X_unique)
        
        # Get the conditional distribution of Y|X for each param sample.
        # For each of the K samples, we have a block shaped 
        # (n_Y_unique, n_X_unique) so that the shape in total is
        # shapes = (n_Y_unique*K, n_X_unique)
        Y_cond_X_model = poisson.pmf(
                Y_K_blocks, lambdas_blocks 
            )
        
        
        # Get the sum of all probabilities for values that were NOT observed 
        # in-sample. Specifically:
        # Get the model pmf for all Y \notin sample, i.e.
        # evaluate 
        #
        #       p_{\theta}(y \notin {y_1, ... y_n}|x_i) for all 
        #       not-observed values of y. 
        #
        # Computationally, it's easier to get it as 
        # 1-\sum_{y \in sample}p_{\theta}(y|x_i) for all unique x_i in-sample.
        #
        # shape = n_X_unique (we sum out over Y and samples K )
        
        
        # Get a vector s.t. we have an entry for each unique x
        remainder_lik = K - jnp.sum(Y_cond_X_model, axis=0)
        
        
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

        
        # as before with the Y_blocks, stack the Y|X pmf K times
        pmf_K_blocks = jnp.tile(Y_cond_X_pmf, (K,1)) 
        
        expected_TVD = 0.5 * jnp.sum(
                X_pmf * 
                ( jnp.sum(jnp.abs(Y_cond_X_model - pmf_K_blocks), axis=0) +
                  remainder_lik
                )
            )
        
        #Division by K, not n! We divide by n in main code (as this function
        #computes the AVERAGE loss.)
        return (self.weight * expected_TVD / K) 
    
    
    
        
    
    def report_parameters(self, q_parser, q_params):
        
        v_c = np.exp(-q_parser.get(q_params, 'log_variance'))[ :, 0 ]
        m_c = q_parser.get(q_params, 'mean')[ :, 0 ] 

        
        
        param_names = ['coefficient means (GLM)', 'coefficient variances'
                       ]
        param_values = [m_c, v_c]
        
        return (param_names, param_values)
    
    
    
class TvdLossPoissonSoftplus(TvdLossPoissonGLM):
    """Same as the TVD loss Poisson GLM, but the link function for lambda is
    different. In particular, we have lambda(x) = softplus(x), which hopefully
    makes the gradients nicer"""
    
    def get_lambdas(self, X_unique, coef_sample):
        """This loss function uses the GLM reparameterization of lambda|X, 
           I.e. we use lambda(x) = exp(theta * x) 
        """
        return jnp.transpose(
                jnp.logaddexp(jnp.matmul(X_unique, jnp.transpose(coef_sample)), 0)
            )
    
    def initializer(self, X, Y, params, parser):
        """Returns the Maximum Likelihood Estimate, which we use to initialize
        the variational parameters"""
        MLE = sm.GLM(Y, X, family = sm.families.Poisson()).fit().params
        X_mean = np.mean(X, 0)
        params[ parser.get_indexes(params, 'mean') ] = (MLE * X_mean)
        return params
    
    
    def report_parameters(self, q_parser, q_params):
        
        v_c = np.exp(-q_parser.get(q_params, 'log_variance'))[ :, 0 ]
        m_c = q_parser.get(q_params, 'mean')[ :, 0 ] 

        
        param_names = ['coefficient means (SOFTPLUS)', 'coefficient variances'
                       ]
        param_values = [m_c, v_c]
        
        return (param_names, param_values)
    
    
class TvdLossPoissonSquare(TvdLossPoissonGLM):
    """Same as the TVD loss Poisson GLM, but the link function for lambda is
    different. In particular, we have lambda(x) = square(x), which hopefully
    makes the gradients nicer"""
    
    def get_lambdas(self, X_unique, coef_sample):
        """This loss function uses the GLM reparameterization of lambda|X, 
           I.e. we use lambda(x) = exp(theta * x) 
        """
        return jnp.transpose(         
                jnp.matmul(X_unique, jnp.transpose(coef_sample)) ** 2
            )
    
    def initializer(self, X, Y, params, parser):
        """Returns the Maximum Likelihood Estimate, which we use to initialize
        the variational parameters"""
        MLE = sm.GLM(Y, X, family = sm.families.Poisson()).fit().params
        X_mean = np.mean(X, 0)
        params[ parser.get_indexes(params, 'mean') ] = np.sqrt(np.exp(MLE * X_mean))/X_mean
        return params
    
    
    def report_parameters(self, q_parser, q_params):
        
        v_c = np.exp(-q_parser.get(q_params, 'log_variance'))[ :, 0 ]
        m_c = q_parser.get(q_params, 'mean')[ :, 0 ] 

        
        param_names = ['coefficient means (SQUARE)', 'coefficient variances'
                       ]
        param_values = [m_c, v_c]
        
        return (param_names, param_values)


class TvdLossPoissonSqrt(TvdLossPoissonGLM):
    """Same as the TVD loss Poisson GLM, but the link function for lambda is
    different. In particular, we have lambda(x) = |abs(x)|^1/2, which hopefully
    makes the gradients nicer"""
    
    def __init__(self, d, weight=1.0, root = 2.0):
        self.d = d
        # this object is only used if 'None' is passed into the 'draw_samples'
        # function. This ensures we can use the loss functions with scipy
        # optimizers, as it gives the loss a way to change its own seed
        self.jax_prng_key = random.PRNGKey(0)
        self.weight = weight
        self.root = root
    
    def get_lambdas(self, X_unique, coef_sample):
        """This loss function uses the GLM reparameterization of lambda|X, 
           I.e. we use lambda(x) = exp(theta * x) 
        """
        return jnp.transpose(         
                jnp.power(jnp.abs(jnp.matmul(X_unique, jnp.transpose(coef_sample))), 1.0/self.root)
            )
    
    def initializer(self, X, Y, params, parser):
        """Returns the Maximum Likelihood Estimate, which we use to initialize
        the variational parameters"""
        MLE = sm.GLM(Y, X, family = sm.families.Poisson()).fit().params
        X_mean = np.mean(X, 0)
        params[ parser.get_indexes(params, 'mean') ] = np.power(np.exp(MLE * X_mean), self.root) /np.abs(X_mean)
        return params
    
    
    def report_parameters(self, q_parser, q_params):
        
        v_c = np.exp(-q_parser.get(q_params, 'log_variance'))[ :, 0 ]
        m_c = q_parser.get(q_params, 'mean')[ :, 0 ] 

        
        param_names = ['coefficient means (' + str(self.root) + '-th root)', 'coefficient variances'
                       ]
        param_values = [m_c, v_c]
        
        return (param_names, param_values)
    
    
class LogNormalLossBLR(Loss):
    """Compute log likelihood loss between data & parameters (NOT q_params,
    but model params), create q_parameter objects for the BBGVI class etc. 
    Internal states of the object are:
        d           dimension of X (regressor matrix)
    """
    
    def __init__(self, d):
        self.d = d
    
    def make_parser(self, parser):
        
        """Create the parser object"""  
        # d coefficients and 1 variance (d+1) parameters in total
        parser.add_shape('mean', (self.d+1, 1))
        parser.add_shape('log_variance', (self.d+1,1))
        
        """Initialize the parser object"""
        params = 0.1 * np.random.randn(parser.num_params)        
        params[ parser.get_indexes(params, 'log_variance') ] = (
            params[ parser.get_indexes(params, 'log_variance') ] + 10.0)
        
        log_conversion = np.zeros((self.d+1, 1), dtype=bool)
        log_conversion[-1:,0] = True
        
        #Make sure to set the initial noise low!
        params[ parser.get_indexes(params, 'mean') ][-1] = ( -np.log(0.1) )

        return (parser, params, log_conversion)
    
    
    def draw_samples(self, q_parser, q_params, K, jax_prng_key):
        
        """Use parser & params to produce K samples of the parameters for which
        q produces a posterior distribution. Do this row-wise."""

        v_c = jnp.exp(-q_parser.get(q_params, 'log_variance'))[ :-1, 0 ]
        m_c = q_parser.get(q_params, 'mean')[ :-1, 0 ] 

        coef_sample = random.normal(jax_prng_key, (K, self.d)) * jnp.sqrt(v_c) + m_c
        
        v_sigma = jnp.exp(-q_parser.get(q_params, 'log_variance'))[ -1, 0 ]
        m_sigma = jnp.exp((-q_parser.get(q_params, 'mean')))[ -1, 0 ] 
        
        sigma2_sample = random.normal(jax_prng_key,(K, 1)) * jnp.sqrt(v_sigma) + m_sigma
        
        
        return (coef_sample, sigma2_sample)
        
        
    def avg_loss(self, q_sample, Y_, X_, indices, jax_key):
        """Retrieve the parameters (using the parser) and compute the 
        average negative log likelihood across the q_sample as well as 
        the X/Y sub-sample"""
        
        """Note 1: The q_sample is a sample from the coefficients b in 
                
                Y = X * b + e
        
        And so we compute for each b-sample the term X * b = Y_hat"""
        coef_sample = q_sample[0]
        K = coef_sample.shape[0]
        M = len(Y_)
        sigma2_sample = q_sample[1].reshape(K)
        
        """X_ is M x d, coef_sample is K x d, so Y_hat will be M x K"""
        Y_hat = jnp.matmul(X_, jnp.transpose(coef_sample))
        
        """Next, get average neg log lklhood"""
        neg_log_lkl = -( 
            - 0.5 * jnp.log(2 * jnp.pi * (sigma2_sample)) 
            - 0.5 * (jnp.tile(Y_.reshape(M,1), (1, K)) - Y_hat)**2 /sigma2_sample
            )
        
        return jnp.mean(neg_log_lkl, 0) #Division by K, not M!
    

    def report_parameters(self, q_parser, q_params):
        
        v_c = np.exp(-q_parser.get(q_params, 'log_variance'))[ :-1, 0 ]
        m_c = q_parser.get(q_params, 'mean')[ :-1, 0 ] 
        
        v_sigma = np.exp(-q_parser.get(q_params, 'log_variance'))[ -1, 0 ]
        m_sigma = np.exp((-q_parser.get(q_params, 'mean')))[ -1, 0 ] 
        
        
        param_names = ['coefficient means', 'coefficient variances', 
                       'sigma2 mean', 'sigma2 variance']
        param_values = [m_c, v_c, m_sigma, v_sigma]
        
        return (param_names, param_values)


class LogNormalLossBLRFixedSig(Loss):
    """Compute log likelihood loss between data & parameters (NOT q_params,
    but model params), create q_parameter objects for the BBGVI class etc. 
    Internal states of the object are:
        d           dimension of X (regressor matrix)
    """
    
    def __init__(self, d):
        self.d = d
    
    def make_parser(self, parser):
        
        """Create the parser object"""
        
        parser.add_shape('mean', (self.d, 1))
        parser.add_shape('log_variance', (self.d,1))
        parser.add_shape('log_sigma2', (1,1))
        
        log_conversion = None
        
        """Initialize the parser object"""
        params = 0.1 * np.random.randn(parser.num_params)
        
        params[ parser.get_indexes(params, 'log_variance') ] = (
            params[ parser.get_indexes(params, 'log_variance') ] + 10.0)
        params[ parser.get_indexes(params, 'log_sigma2') ] = (
            params[ parser.get_indexes(params, 'log_sigma2') ]  -np.log(0.1) )
        
        return (parser, params, log_conversion)
    
    
    def draw_samples(self, q_parser, q_params, K, jax_prng_key):
        
        """Use parser & params to produce K samples of the parameters for which
        q produces a posterior distribution. Do this row-wise."""
        
        v_c = jnp.exp(-q_parser.get(q_params, 'log_variance'))[ :, 0 ]
        m_c = q_parser.get(q_params, 'mean')[ :, 0 ] 
        coef_sample = random.normal(jax_prng_key ,(K, self.d)) * np.sqrt(v_c) + m_c
        
        v_sigma = jnp.exp(-q_parser.get(q_params, 'log_sigma2')) 
        
        return (coef_sample, v_sigma)
        
        
    def avg_loss(self, q_sample, Y_, X_, indices):
        """Retrieve the parameters (using the parser) and compute the 
        average negative log likelihood across the q_sample as well as 
        the X/Y sub-sample"""
        
        """Note 1: The q_sample is a sample from the coefficients b in 
                
                Y = X * b + e
        
        And so we compute for each b-sample the term X * b = Y_hat"""
        coef_sample = q_sample[0]
        v_sigma = q_sample[1]
        K = coef_sample.shape[0]
        M = len(Y_)
        
        """X_ is M x d, coef_sample is K x d, so Y_hat will be M x K"""
        Y_hat = jnp.matmul(X_, jnp.transpose(coef_sample))
        
        """Next, get average neg log lklhood"""
        neg_log_lkl = -( 
            - 0.5 * jnp.log(2 * jnp.pi * (v_sigma)) 
            - 0.5 * (jnp.tile(Y_.reshape(M,1), (1, K)) - Y_hat)**2 /v_sigma
            )
      
        return jnp.mean(neg_log_lkl, 0) #Division by K, not M!
    
    def report_parameters(self, q_parser, q_params):
        
        v_c = np.exp(-q_parser.get(q_params, 'log_variance'))[ :, 0 ]
        m_c = q_parser.get(q_params, 'mean')[ :, 0 ] 
        
        v_sigma = np.exp(-q_parser.get(q_params, 'log_sigma2'))
        
        
        param_names = ['coefficient means', 'coefficient variances', 
                       'sigma2 point estimate']
        param_values = [m_c, v_c, v_sigma]
        
        return (param_names, param_values)



class LogLaplaceLossBLR(LogNormalLossBLR):
    """Compute log likelihood loss between data & parameters (NOT q_params,
    but model params), create q_parameter objects for the BBGVI class etc. 
    Internal states of the object are:
        d           dimension of X (regressor matrix)
    """
               
    def avg_loss(self, q_sample, Y_, X_, indices):
        """Retrieve the parameters (using the parser) and compute the 
        average negative log likelihood across the q_sample as well as 
        the X/Y sub-sample"""
        
        """Note 1: The q_sample is a sample from the coefficients b in 
                
                Y = X * b + e
        
        And so we compute for each b-sample the term X * b = Y_hat"""
        coef_sample = q_sample[0]
        K = coef_sample.shape[0]
        M = len(Y_)
        sigma2_sample = q_sample[1].reshape(K)
        
        """X_ is M x d, coef_sample is K x d, so Y_hat will be M x K"""
        Y_hat = jnp.matmul(X_, jnp.transpose(coef_sample))
        
        """Next, get average neg log lklhood"""
        neg_log_lkl = -( 
            - 0.5 * jnp.log(2 * jnp.pi * (sigma2_sample)) 
            - 0.5 * jnp.abs(jnp.tile(Y_.reshape(M,1), (1, K)) - Y_hat) /sigma2_sample
            )
        
        return jnp.mean(neg_log_lkl, 0) #Division by K, not M!


class AbsoluteLoss(Loss):
    """Compute log likelihood loss between data & parameters (NOT q_params,
    but model params), create q_parameter objects for the BBGVI class etc. 
    Internal states of the object are:
        d           dimension of X (regressor matrix)
    """
               
    def __init__(self, d):
        self.d = d
    
    def make_parser(self, parser):
        
        """Create the parser object"""       
        parser.add_shape('mean', (self.d, 1))
        parser.add_shape('log_variance', (self.d,1))
        
        """Initialize the parser object"""
        params = 0.1 * np.random.randn(parser.num_params)        
        params[ parser.get_indexes(params, 'log_variance') ] = (
            params[ parser.get_indexes(params, 'log_variance') ] - np.log(1.0))

        return (parser, params, None)
    
    
    def draw_samples(self, q_parser, q_params, K, jax_prng_key):
        
        """Use parser & params to produce K samples of the parameters for which
        q produces a posterior distribution. Do this row-wise."""

        v_c = jnp.exp(-q_parser.get(q_params, 'log_variance'))[ :, 0 ]
        m_c = q_parser.get(q_params, 'mean')[ :, 0 ] 

        coef_sample = random.normal(jax_prng_key, (K, self.d)) * jnp.sqrt(v_c) + m_c
                
        return coef_sample
        
        
    def avg_loss(self, q_sample, Y_, X_, indices):
        """Retrieve the parameters (using the parser) and compute the 
        average negative log likelihood across the q_sample as well as 
        the X/Y sub-sample"""
        
        """Note 1: The q_sample is a sample from the coefficients b in 
                
                Y = X * b + e
        
        And so we compute for each b-sample the term X * b = Y_hat"""
        coef_sample = q_sample
        K = coef_sample.shape[0]
        M = len(Y_)
        
        """X_ is M x d, coef_sample is K x d, so Y_hat will be M x K"""
        Y_hat = jnp.matmul(X_, jnp.transpose(coef_sample))
        
        """Next, get average neg log lklhood"""
        loss = jnp.abs(np.tile(Y_.reshape(M,1), (1, K)) - Y_hat)
                    
        return jnp.mean(loss, 0)
    
    
    def report_parameters(self, q_parser, q_params):
        
        v_c = np.exp(-q_parser.get(q_params, 'log_variance'))[ :, 0 ]
        m_c = q_parser.get(q_params, 'mean')[ :, 0 ] 
                
        param_names = ['absolute value means', 'absolute value variances']
        param_values = [m_c, v_c]
        
        return (param_names, param_values)



class BMMLogLossFixedSigma(Loss):
    """Compute log likelihood loss between data & parameters (NOT q_params,
    but model params), create q_parameter objects for the BBGVI class etc. 
    Internal states of the object are:
            d           dimension of X (regressor matrix)
            K           number of clusters
            
    """
    
    def __init__(self,d, K, n, gamma = None):
        self.d = d
        self.K = K
        self.n = n
        self.gamma = gamma
        
    def make_parser(self, parser):
        """This parser needs: 
             - mu K-vector
             - sigma2 K-vector (in log form)
             - clusters n-vector (these are the [latent] cluster memberships)
         NOTE: clusters are NOT penalized with a div, obviously! They comple-
               tely occur inside the loss function and are treated accordingly    
        """
         
        """Create the parser object"""
        
        #mean and log variance for mu + sigma2
        parser.add_shape('mean', (self.d, self.K)) #for mu, sigma2
        parser.add_shape('log_variance', (self.d, self.K)) #for mu, sigma2

        #individual-specific latent terms. Categorial RV
        parser.add_shape('cluster_prob', (self.n, self.K))
        
        """Initialize the parser object"""
#        
#        """This means that the entries from self.K onwards in 'mean' are 
#        stored in log form & need to be transformed back"""
        """I.e., none of the mean parameters are stored as logs because
        we do not perform variational inference for the variances"""
        log_conversion = None 
        
        """Just produce some very small random numbers for the mean + var 
        of the clusters"""
        global_params = 0.1 * np.random.randn(self.K*2*self.d) #for global vars
        
        """For the discrete latent variable, just assign probability 1/K to
        each category for each observation & maybe slightly perturb it"""
        cluster_membership = np.ones((self.n,self.K))*(1.0/self.K) 
            
        """Set the log variances to be small (conversion: exp(-log_var))"""
        global_params[ parser.get_indexes(global_params, 'log_variance') ] = (
            global_params[ parser.get_indexes(global_params, 'log_variance') ] + 10.0)

        cluster_membership[parser.get_indexes(cluster_membership, 'cluster_prob') ] = (
            cluster_membership[ parser.get_indexes(cluster_membership, 'cluster_prob') ])
         
        """package all global variational parameters together -- only global
        parameters are passed directly to the divergence object & so we don't
        have to worry about passing along any of the other params. 
        BUT: We do need to pass along the mean log conversion object"""
        
        #PUT cluster_membership and global_params together
        all_params = np.concatenate((global_params.flatten(), cluster_membership.flatten()))
        
        return (parser, all_params, log_conversion)


    def draw_samples(self, q_parser, q_params, K, jax_prng_key):

        num_samples = K
        
        """global variational params"""
        mu_cluster_m = q_parser.get(q_params, 'mean')[:, :] 
        mu_cluster_v = jnp.exp(-q_parser.get(q_params, 'log_variance'))[ :, :]

        """local variational params for each x_i. Reparameterization used here
            is the softmax/kategorical RV"""
        c = q_parser.get(q_params, 'cluster_prob')
        cluster_probs_xi = jnp.exp(c) / jnp.sum(jnp.exp(c), axis=1).reshape(self.n,1)
        
        """Draw from cluster locations & variances"""
        #dim: d x K x S
        cluster_locations = (random.normal(jax_prng_key, 
            (self.d, self.K, num_samples)) * 
            jnp.sqrt(mu_cluster_v)[:,:,jnp.newaxis] + 
            mu_cluster_m[:,:,jnp.newaxis])
        
        """Don't draw from cluster assignments for the x_i -- the cluster_probs
        themselves are ALREADY defining a distribution!"""
        
        return (cluster_locations, cluster_probs_xi)


    
    def avg_loss(self, q_sample, Y_, X_=None, indices = None):
        """The average loss is a sum over the cluster probabilities (for each 
        x_i) and the samples from cluster centers & variances. objective is
        the following:
        
            E_{q(mu,sigma2)}[ 
                \sum_{i=1}^n\sum_{j=1}^K log(p(c_j) * p(x_i|c_{j,i}, mu_j, sigma_j)) 
            ] 
            + D(q||pi)
        
        OUTLINE OF COMPUTATIONS:
           
            We have the following hierarchy:
                
                \mu_{1:Kd} \sim prior_{\mu}
                \sigma_{1:Kd}^2 \sim prior_{\sigma^2}
                
                c_i \sim Categorical(1/K)
                x_i|c_i=c, \mu_{1:Kd}, \sigma_{1:Kd}^2 \sim 
                        N(x_i|\mu(c), \sigma^2(c))
                        
                where we have that for c_i = c
                
                \mu(c) = \mu_{(c-1)*d:c*d}
                \sigma(c) = \sigma^2_{(c-1)*d:c*d}
            
            Noticing that the prior terms will be dealt with inside the 
            divergence, we can focus on the likelihood computation. 
            Notice that an individual likelihood term is given by
            
                p(x_i| \mu_{1:K}, \sigma^2_{1:K}) = 
                    \sum_{j=1}^K prob(c_i = j) * N(x_i|\mu(j), \sigma^2(j))
                
            and the log likelihood for all data points is
            
              p(x_{1:n}| \mu_{1:K}, \sigma^2_{1:K})
                \sum_{j=1}^n \log\left(
                    \sum_{i=1}^K prob(c_i = j) * N(x_i|\mu(j), \sigma^2(j))  
                \right)
              
        
        NOTE: We do all of this ONLY for the indices of x that are in the 
              current batch. These are in the np.array indices
        """

        
        cluster_locations, cluster_probs_xi = q_sample
        
        #DEBUG: Does this make sense?
        cluster_probs_xi = cluster_probs_xi[indices]
        
        num_samples = cluster_locations.shape[-1] 
        n = Y_.shape[0]
        d = Y_.shape[1]
        LOG_FORMAT = True
    
        
        """STEP 1.1: Extract the Normal likelihood parts into n x S"""
            
        #dim: n x d x K x S  
        """NOTE: contains all likelihood terms on each dimension, for each 
                 cluter and for each sample"""
        negative_log_likelihood_raw = (
            (jnp.tile(Y_[:,:,jnp.newaxis,jnp.newaxis], (1, 1, self.K, num_samples)) - 
                     cluster_locations[jnp.newaxis,:,:,:])**2 /
                         1.0
                        #cluster_variances[np.newaxis:,:,:]
                        )
             
        #dim: d x K x S     
        """NOTE: contains all likelihood terms on each dimension"""
        log_dets = jnp.log(2 * jnp.pi * 
                          1.0
                          #(cluster_variances[:,:,:])
                          ) * jnp.ones((negative_log_likelihood_raw.shape[1], self.K,num_samples))
        
        #dim: n x d x K x S
        negative_log_likelihood_raw = negative_log_likelihood_raw + log_dets[jnp.newaxis, :,:,:]
        
        #dim: n x K x S
        """NOTE: likelihood terms aggregated across dimensions. This 
                 corresponds to independence across d."""
        negative_log_likelihood_raw = jnp.sum(negative_log_likelihood_raw, axis=1)
        
        
        """STEP 1.2: Multiply with the relevant individual-specific 
                     cluster-probabilities"""
        
        if LOG_FORMAT:
            #dim: n x K x S
            log_likelihoods_clusters = (-negative_log_likelihood_raw[:,:,:] + 
                                 jnp.log(cluster_probs_xi[:,:])[:,:,jnp.newaxis])
        elif not LOG_FORMAT:
            #dim: n x K x S
            likelihoods_clusters = (jnp.exp(-negative_log_likelihood_raw[:,:,:]) *  
                             (cluster_probs_xi[:,:])[:,:,jnp.newaxis])

            
        
        """STEP 2: Take the raw likelihoods we have and finally get the 
                   average log likelihood.
                   
                   We need two steps: 
                       
                       1. get to 
                   
                       \sum_{j=1}^K prob(c_i = j) * N(x_i|\mu(j), \sigma^2(j))
                
                   For each sample s.
                   
                       2. get the actual sample average over S and N.
                   
                   """
                   
        """STEP 2.1: row-logsumexp to get to n x S"""
        
        
        
        if LOG_FORMAT: 
            #dim: n x S
            logsumexp_observation_log_likelihoods = logsumexp(
                    log_likelihoods_clusters, axis=1)
            log_likelihoods = np.mean(logsumexp_observation_log_likelihoods)
            
            #Use robust losses where we approximate the integral with the observations
            ROBUST = (self.gamma is not None)
            if ROBUST:
                gamma = self.gamma
                log_integral = logsumexp(
                    (gamma) * logsumexp_observation_log_likelihoods - jnp.log(n),
                    axis=0)
                log_gamma_score = (
                    jnp.log(gamma / (gamma - 1.0)) + 
                    logsumexp_observation_log_likelihoods * (gamma - 1.0) + 
                    ((gamma - 1.0)/gamma)*log_integral)
                log_likelihoods = jnp.mean(jnp.exp(log_gamma_score))
                
            
        elif not LOG_FORMAT:
            #dim: n x S
            observation_log_likelihoods = jnp.log(
                    jnp.sum(likelihoods_clusters, axis=1))
            #dim: scalar
            log_likelihoods = jnp.mean(observation_log_likelihoods)
      
        
        """STEP 2.2: return the average over the sample- & observation axis"""
        
        return -log_likelihoods
    
    
    def report_parameters(self, q_parser, q_params):
        
        """global variational params"""
        mu_cluster_m = q_parser.get(q_params, 'mean')[:, :] 
        mu_cluster_v = np.exp(-q_parser.get(q_params, 'log_variance'))[ :, :]

        """local variational params for each x_i"""
        c = q_parser.get(q_params, 'cluster_prob')
        cluster_probs_xi = np.exp(c) / np.sum(np.exp(c), axis=1).reshape(self.n,1)
        
        #print("cluster_probs_xi", cluster_probs_xi)
                
        param_names = ['cluster membership probabilities', 
                       'cluster position mean', 'cluster position variance',
                       ]
        param_values = [cluster_probs_xi, 
                        mu_cluster_m, mu_cluster_v
                        ]
        
        return (param_names, param_values)





#DEBUG: DOES NOT WORK YET
class BMMLogLoss(Loss):
    """Compute log likelihood loss between data & parameters (NOT q_params,
    but model params), create q_parameter objects for the BBGVI class etc. 
    Internal states of the object are:
            d           dimension of X (regressor matrix)
            K           number of clusters
            
    """
    
    def __init__(self,d, K, n):
        self.d = d
        self.K = K
        self.n = n
        
    def make_parser(self, parser):
        """This parser needs: 
             - mu K-vector
             - sigma2 K-vector (in log form)
             - clusters n-vector (these are the [latent] cluster memberships)
         NOTE: clusters are NOT penalized with a div, obviously! They comple-
               tely occur inside the loss function and are treated accordingly    
        """
         
        """Create the parser object"""
        #add information to parser: Indices which model on the log scale
        #parser.add_shape('mean_log_conversion', (self.d,self.K*2))
        
        #mean and log variance for mu + sigma2
        parser.add_shape('mean', (self.d, self.K*2)) #for mu, sigma2
        parser.add_shape('log_variance', (self.d, self.K*2)) #for mu, sigma2

        #individual-specific latent terms. Categorial, i.e. we have K-1  
        #free parameters per observation & optimize over those
        parser.add_shape('cluster_prob', (self.n, self.K-1))
        
        """Initialize the parser object"""
        
        """This means that the entries from self.K onwards in 'mean' are 
        stored in log form & need to be transformed back"""
        log_conversion = np.zeros(( self.d, 2*self.K), dtype=bool)
        log_conversion[:, self.K:] = True
        
        """Just produce some very small random numbers for the mean + var 
        of the clusters"""
        global_params = 0.1 * np.random.randn(self.K*2*self.d*2) #for global vars
        
        """For the discrete latent variable, just assign probability 1/K to
        each category for each observation & maybe slightly perturb it"""
        cluster_membership = np.ones((self.n,self.K-1))*(1.0/self.K) #(1.0/self.K)

        global_params[ parser.get_indexes(global_params, 'log_variance') ] = (
            global_params[ parser.get_indexes(global_params, 'log_variance') ] + 3.0)
        cluster_membership[parser.get_indexes(cluster_membership, 'cluster_prob') ] = (
            cluster_membership[ parser.get_indexes(cluster_membership, 'cluster_prob') ])
         
        """package all global variational parameters together -- only global
        parameters are passed directly to the divergence object & so we don't
        have to worry about passing along any of the other params. 
        BUT: We do need to pass along the mean log conversion object"""
        
        #PUT cluster_membership and global_params 
        all_params = np.concatenate((global_params.flatten(), cluster_membership.flatten()))
        
        return (parser, all_params, log_conversion)


    def draw_samples(self, q_parser, q_params, K, jax_prng_key):

        num_samples = K
        
        """global variational params"""
        mu_cluster_m = q_parser.get(q_params, 'mean')[:, :self.K ] 
        mu_cluster_v = jnp.exp(-q_parser.get(q_params, 'log_variance'))[ :, :self.K ]
        
        sigma2_cluster_m = jnp.exp(-q_parser.get(q_params, 'mean'))[ :, self.K: ]
        sigma2_cluster_v = jnp.exp(-q_parser.get(q_params, 'log_variance'))[ :, self.K: ]

        """local variational params for each x_i"""
        cluster_probs_xi = q_parser.get(q_params, 'cluster_prob')
        
        """Draw from cluster locations & variances"""
        cluster_locations = (random.normal(jax_prng_key, (self.d, self.K, num_samples)) * 
            jnp.sqrt(mu_cluster_v)[:,:,jnp.newaxis] + 
            mu_cluster_m[:,:,jnp.newaxis])
        cluster_variances = (random.normal(jax_prng_key, (self.d, self.K, num_samples)) * 
            jnp.sqrt(sigma2_cluster_v)[:,:,jnp.newaxis] + 
            sigma2_cluster_m[:,:,jnp.newaxis])
        
        """Don't draw from cluster assignments for the x_i -- the cluster_probs
        themselves are ALREADY defining a distribution!"""
        
        return (cluster_locations, cluster_variances, cluster_probs_xi)


    
    def avg_loss(self, q_sample, Y_, X_=None, indices = None):
        """The average loss is a sum over the cluster probabilities (for each 
        x_i) and the samples from cluster centers & variances. objective is
        the following:
        
            E_{q(mu,sigma2)}[ 
                \sum_{i=1}^n\sum_{j=1}^K log(p(c_j) * p(x_i|c_{j,i}, mu_j, sigma_j)) 
            ] 
            + D(q||pi)
        
        OUTLINE OF COMPUTATIONS:
           
            We have the following hierarchy:
                
                \mu_{1:Kd} \sim prior_{\mu}
                \sigma_{1:Kd}^2 \sim prior_{\sigma^2}
                
                c_i \sim Categorical(1/K)
                x_i|c_i=c, \mu_{1:Kd}, \sigma_{1:Kd}^2 \sim 
                        N(x_i|\mu(c), \sigma^2(c))
                        
                where we have that for c_i = c
                
                \mu(c) = \mu_{(c-1)*d:c*d}
                \sigma(c) = \sigma^2_{(c-1)*d:c*d}
            
            Noticing that the prior terms will be dealt with inside the 
            divergence, we can focus on the likelihood computation. 
            Notice that an individual likelihood term is given by
            
                p(x_i| \mu_{1:K}, \sigma^2_{1:K}) = 
                    \sum_{j=1}^K prob(c_i = j) * N(x_i|\mu(j), \sigma^2(j))
                
            and the log likelihood for all data points is
            
              p(x_{1:n}| \mu_{1:K}, \sigma^2_{1:K})
                \sum_{j=1}^n \log\left(
                    \sum_{i=1}^K prob(c_i = j) * N(x_i|\mu(j), \sigma^2(j))  
                \right)
              
        """

        
        cluster_locations, cluster_variances, cluster_probs_xi = q_sample
        num_samples = cluster_locations.shape[-1] 
        n = Y_.shape[0]
    
        
        """STEP 1.1: Extract the Normal likelihood parts into n x S"""
            
        #dim: n x d x K x S  
        """NOTE: contains all likelihood terms on each dimension, for each 
                 cluter and for each sample"""
        negative_log_likelihood_raw = (
            (jnp.tile(Y_[:,:,jnp.newaxis,jnp.newaxis], (1, 1, self.K, num_samples)) - 
                     cluster_locations[jnp.newaxis,:,:,:])**2 /
                         cluster_variances[jnp.newaxis:,:,:]
                        )
             
        #dim: d x K x S     
        """NOTE: contains all likelihood terms on each dimension"""
        log_dets = jnp.log(2 * jnp.pi * 
                          (cluster_variances[:,:,:])
                          ) * jnp.ones((negative_log_likelihood_raw.shape[1], self.K,num_samples))
        
        #dim: n x d x K x S
        negative_log_likelihood_raw = negative_log_likelihood_raw + log_dets[np.newaxis, :,:,:]
        
        #dim: n x K x S
        """NOTE: likelihood terms aggregated across dimensions. This 
                 corresponds to independence across d."""
        negative_log_likelihood_raw = jnp.sum(negative_log_likelihood_raw, axis=1)

        
        """STEP 1.2: Multiply with the relevant individual-specific 
                     cluster-probabilities"""
                     
        #dim: n x (K-1) x S
        log_likelihoods_Km1_clusters = (-negative_log_likelihood_raw[:,:-1,:] + 
                         jnp.log(cluster_probs_xi[:,:])[:,:,jnp.newaxis])
        #dim: n x 1 x S
        log_likelihoods_K_cluster = (
                -negative_log_likelihood_raw[:,-1,:] + jnp.log(1.0 - 
                        jnp.sum(cluster_probs_xi[:,:],axis=1))[:,jnp.newaxis]
            ).reshape(n, 1, num_samples)
        
        
        
        """STEP 2: Take the raw likelihoods we have and finally get the 
                   average log likelihood.
                   
                   We need two steps: 
                       
                       1. get to 
                   
                       \sum_{j=1}^K prob(c_i = j) * N(x_i|\mu(j), \sigma^2(j))
                
                   For each sample s.
                   
                       2. get the actual sample average over S and N.
                   
                   """
                   
        """STEP 2.1: row-logsumexp to get to n x S"""
        
        #dim: n x 1 x S
        logsumexp_Km1 = logsumexp(log_likelihoods_Km1_clusters, axis=1)
        
        #dim: n x 1 x S
        max_vals = jnp.maximum(logsumexp_Km1, log_likelihoods_K_cluster)
        log_likelihoods = jnp.log(np.exp(logsumexp_Km1-max_vals) + 
                        jnp.exp(log_likelihoods_K_cluster-max_vals))
        log_likelihoods += max_vals
        
        """STEP 2.2: return the average over the sample- & observation axis"""
        
        return -jnp.mean(log_likelihoods)
    
    
    def report_parameters(self, q_parser, q_params):
        
        """global variational params"""
        mu_cluster_m = q_parser.get(q_params, 'mean')[:, :self.K ] 
        mu_cluster_v = np.exp(-q_parser.get(q_params, 'log_variance'))[ :, :self.K ]
        
        sigma2_cluster_m = np.exp(-q_parser.get(q_params, 'mean'))[ :, self.K: ]
        sigma2_cluster_v = np.exp(-q_parser.get(q_params, 'log_variance'))[ :, self.K: ]

        """local variational params for each x_i"""
        cluster_probs_xi = q_parser.get(q_params, 'cluster_prob')
        #print("cluster_probs_xi", cluster_probs_xi)
                
        param_names = ['cluster membership probabilities', 
                       'cluster position mean', 'cluster position variance',
                       'cluster variance mean', 'cluster variance variance',
                       ]
        param_values = [cluster_probs_xi, 
                        mu_cluster_m, mu_cluster_v,
                        sigma2_cluster_m, sigma2_cluster_v
                        ]
        
        return (param_names, param_values)



#Convince yourself that autograd can transcend classes...
if False:
    #from __future__ import absolute_import
    #from __future__ import print_function
    import autograd.numpy as np
    import autograd.numpy.random as npr
    from autograd.scipy.misc import logsumexp
    from autograd import grad
    
    def fun(x):
        return x**2
    
    class wrapper():
        
        def __init__(self,f):
            self.f = f
            
        def eval_f(self,x):
            return self.f(x)
    
    
    f_grad = grad(fun)
    
    fobj = wrapper(fun)
    
    def fun_via_obj(x):
        return fobj.eval_f(x)
    
    f_grad2 = grad(fun_via_obj)
    
    print(f_grad(2.45), f_grad(6.0))
    print(f_grad2(2.45), f_grad2(6.0))






        
        
        
    