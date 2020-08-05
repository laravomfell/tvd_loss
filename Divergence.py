#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:05:57 2019

@author: jeremiasknoblauch

Description: Divergence part (of the objective), i.e. this term 
             penalizes how far prior diverges from posterior.
"""

#from __future__ import absolute_import
#from __future__ import print_function
import jax.numpy as np
import jax.random as npr
from jax.scipy.special import logsumexp
from jax import grad

import math


class Divergence():
     """Compute divergence between prior & q. Internal states of the object 
     are none because it is EMPTY/ABSTRACT.            
     """
     
     def __init__(self):
         return 0
     
     def unpack(self, params, parser, converter):
         """Unpack the params"""
         
         if converter is not None:
             q_variances_sig = np.exp(-parser.get(params, 'log_variance'))[converter].flatten()
             q_means_sig = np.exp((-parser.get(params, 'mean'))[converter]).flatten()
             
             q_variances = np.exp(-parser.get(params, 'log_variance'))[converter == False].flatten()
             q_means = (parser.get(params, 'mean')[converter == False] ).flatten()
         else: 
             q_variances_sig = None
             q_means_sig = None
             
             q_variances = np.exp(-parser.get(params, 'log_variance')).flatten()
             q_means = (parser.get(params, 'mean')).flatten()
         
         
         return (q_variances, q_means, q_variances_sig, q_means_sig)
     
     def prior_regularizer(self, params, parser, converter):
         return 0
     
        
class MFN_MFN_KLD(Divergence):
    """KLD(q||pi) where pi and q are both mean field normals.
    Internal states of the object:
        means       Means of the prior (one mean for each variable)
        variances   Variances of the prior (one for each variable)       
    """
    
    def __init__(self, means, variances, weight = 1.0):
        """MFN for prior specified by vector of means & variances"""
        self.pi_means = means
        self.pi_variances = variances
        self.weight = weight
        
    
    def KL_term(self, q_variances, pi_variances, q_means, pi_means):
        KL = -(np.sum(-0.5 * np.log(2 * math.pi * pi_variances) 
                -0.5 * ((q_means-pi_means)**2 + q_variances) / pi_variances) 
                -np.sum(-0.5 * np.log(2 * math.pi * q_variances * np.exp(1)))
                ) 
        return KL
    
    
    def prior_regularizer(self, q_params, q_parser, converter):
        """Simply unpack means & variances + compute discrepancy"""


        q_variances, q_means, q_variances_sig, q_means_sig = self.unpack(
                q_params, q_parser, converter)
       
        #NOTE: If q_variances_sig and q_means_sig is NONE, then just doo what
        #       you would do for fixedSigma version...
        
        #POTENTIAL PROBLEM: If we flatten converter, then we make an assumption
        #                   about which entries in pi (i.e. the prior) corresponnd
        #                   to which parameter. Need not be a problem so long as
        #                   all priors are the same anyways! BUT means we have to make
        #                   sure wee know which poosition in the prioor corresponds to
        #                   which parameter!
        
    
        if q_variances_sig is None and q_means_sig is None:
            KL = self.KL_term(q_variances, self.pi_variances, q_means, self.pi_means)
        else:            
 
            KL = self.KL_term(q_variances, self.pi_variances[converter.flatten()], 
                         q_means, self.pi_means[converter.flatten()])
            
            KL = KL + self.KL_term(q_variances_sig, self.pi_variances[converter.flatten() == False],
                                   q_means_sig, self.pi_means[converter.flatten() == False])
        
        return self.weight * KL


class MFN_MFN_reverse_KLD(MFN_MFN_KLD):
    """KLD(q||pi) where pi and q are both mean field normals.
    Internal states of the object:
        means       Means of the prior (one mean for each variable)
        variances   Variances of the prior (one for each variable)       
    """
    
    def prior_regularizer(self, q_params, q_parser, converter):
        """Simply unpack means & variances + compute discrepancy"""


        q_variances, q_means, q_variances_sig, q_means_sig = self.unpack(
                q_params, q_parser, converter)
        
    
        if q_variances_sig is None and q_means_sig is None:
            revKL = self.KL_term(self.pi_variances,q_variances, self.pi_means, q_means)
        else:            
 
            revKL = self.KL_term(self.pi_variances[converter.flatten()], q_variances, 
                         self.pi_means[converter.flatten()], q_means)
            
            revKL = revKL + self.KL_term(self.pi_variances[converter.flatten() == False],q_variances_sig, 
                                   self.pi_means[converter.flatten() == False], q_means_sig)
        
        return self.weight * revKL


class MFN_MFN_JeffreysD(MFN_MFN_KLD):
    """Jeffrey's Divergence where q and pi are both mean field normals. 
    This is just KLD(q||pi) + KLD(pi||q)
     Internal states of the object:
        means           Means of the prior (one mean for each variable)
        variances       Variances of the prior (one for each variable)     
        relative_weight relative_weight * KLD(q||pi) + (1-relw) * KLD(pi||q)    
    """

    def __init__(self, means, variances, relative_weight = 0.5, weight = 1.0):
        """MFN for prior specified by vector of means & variances"""
        self.pi_means = means
        self.pi_variances = variances
        self.relative_weight = relative_weight
        self.weight = weight
        
        self.rev_KL = MFN_MFN_reverse_KLD(means, variances)
        self.KL = MFN_MFN_KLD(means, variances)

    def prior_retularizer(self, q_params, q_parser, converter):
        """Compute relative_weight * KLD(q||pi) + (1-relw) * KLD(pi||q)"""
        revKL =  self.rev_KL(q_params, q_parser, converter)
        KL =  self.KL(q_params, q_parser, converter)
        return self.weight * ((1.0 - self.relative_weight) * revKL + self.relative_weight * KL)

       
class MFN_MFN_RAD(Divergence):
    """Renyi-alpha(q||pi) where pi and q are both mean field normals.
    Internal states of the object:
        means       Means of the prior (one mean for each variable)
        variances   Variances of the prior (one for each variable)       
    """
    
    def __init__(self, means, variances, alpha, weight = 1.0):
        """MFN for prior specified by vector of means & variances"""
        self.pi_means = means
        self.pi_variances = variances
        self.alpha = alpha
        self.weight = weight
        
    def RAD_term(self, q_variances, pi_variances, q_means, pi_means):
        logZpi =  0.5 * np.log(pi_variances) + 0.5 * pi_means ** 2 / pi_variances
        logZq = (0.5 * np.log(q_variances) + 0.5 * q_means**2 / q_variances)
        
        new_var = 1.0 / (self.alpha / q_variances + (1.0-self.alpha) /pi_variances)
        new_mean_ = (self.alpha * q_means/q_variances + (1-self.alpha) * pi_means/pi_variances) 
        logZnew = (0.5 * np.log(new_var) + 0.5 * (new_mean_**2) * new_var)
        
        log_reg1 = np.sum(logZnew - self.alpha*logZq - (1-self.alpha)*logZpi)
        
        """return it"""
        return (log_reg1)
        
    
    def prior_regularizer(self, q_params, q_parser, converter):
        """Simply unpack means & variances + compute discrepancy"""
        
        q_variances, q_means, q_variances_sig, q_means_sig = self.unpack(
                q_params, q_parser, converter)
       
        if q_variances_sig is None and q_means_sig is None:
            
            RAD = self.RAD_term(q_variances, self.pi_variances, 
                                q_means, self.pi_means)
            
        else:
            
            RAD = self.RAD_term(q_variances, 
                         self.pi_variances[converter.flatten()], 
                         q_means, self.pi_means[converter.flatten()])
            RAD = RAD + self.RAD_term(q_variances_sig, 
                         self.pi_variances[converter.flatten() == False],
                         q_means_sig, self.pi_means[converter.flatten() == False])
        
        return self.weight * (1.0 / (self.alpha-1.0)) * RAD

    

class MFN_MFN_AD(MFN_MFN_RAD):
    """AlphaD(q||pi) where pi and q are both mean field normals.
    Internal states of the object:
        means       Means of the prior (one mean for each variable)
        variances   Variances of the prior (one for each variable)       
    """
    
    def prior_regularizer(self, q_params, q_parser, converter):
        """Simply unpack means & variances + compute discrepancy"""
        
        q_variances, q_means, q_variances_sig, q_means_sig = self.unpack(
                q_params, q_parser, converter)
        
        if q_variances_sig is None and q_means_sig is None:
            
            RAD = self.RAD_term(q_variances, self.pi_variances, 
                                q_means, self.pi_means)
            
        else:
            
            RAD = self.RAD_term(q_variances, 
                         self.pi_variances[converter.flatten()], 
                         q_means, self.pi_means[converter.flatten()])
            RAD = RAD + self.RAD_term(q_variances_sig, 
                         self.pi_variances[converter.flatten() == False],
                         q_means_sig, self.pi_means[converter.flatten() == False])
        
        return self.weight * (1.0 / ((1.0 - self.alpha)*self.alpha)) * (1.0 - np.exp(RAD))

      
class MFN_MFN_FD(Divergence):
    """Fisher-Div(q||pi) where pi and q are both mean field normals.
    Internal states of the object:
        means       Means of the prior (one mean for each variable)
        variances   Variances of the prior (one for each variable)     
                
    """
    
    def __init__(self, means, variances, weight = 1.0):
        """MFN for prior specified by vector of means & variances"""
        self.pi_means = means
        self.pi_variances = variances
        self.weight = weight
        
    def FD_term(self, q_variances, pi_variances, q_means, pi_means):
        C1_coef = q_means/q_variances - pi_means/pi_variances
        C2_coef = 1.0/q_variances - 1.0/pi_variances
        
        """compute & return it"""
        coef_part = np.sum(
                (C1_coef ** 2) #C1 squared
                +
                2.0 * (C1_coef * C2_coef) * q_means
                +
                (C2_coef ** 2) * (q_means ** 2 + q_variances)
                )
        
        return coef_part
    
    def prior_regularizer(self, q_params, q_parser, converter):
        """Simply unpack means & variances + compute discrepancy"""
        
        q_variances, q_means, q_variances_sig, q_means_sig = self.unpack(
                q_params, q_parser, converter)
        
        if q_variances_sig is None and q_means_sig is None:
            """Compute fisher divergence between the coef terms"""
            FD = self.FD_term(q_variances, self.pi_variances, 
                              q_means, self.pi_means)

        else:
            """Compute fisher divergence between the coef terms"""
            FD = self.FD_term(q_variances, 
                         self.pi_variances[converter.flatten()], 
                         q_means, self.pi_means[converter.flatten()])
            FD = FD + self.FD_term(q_variances_sig, 
                         self.pi_variances[converter.flatten() == False],
                         q_means_sig, self.pi_means[converter.flatten() == False])

        return self.weight * FD
        

class MFN_MFN_ED(Divergence):
    """Exponential Divergence where q and pi are both mean field normals. 
     Internal states of the object:
        means       Means of the prior (one mean for each variable)
        variances   Variances of the prior (one for each variable)     
    """
 
    
    def __init__(self, means, variances, weight = 1.0):
        """MFN for prior specified by vector of means & variances"""
        self.pi_means = means
        self.pi_variances = variances
        self.weight = weight
        
        
    def ED_term(self, q_variances, pi_variances, q_means, pi_means):
        """Compute the inner coefficients before taking the squares"""
        #Ck_coef = polynomial coefficient for (k-1)-th order
        C1_coef = 0.5*(np.log(pi_variances / q_variances) + 
                  (pi_means ** 2 / pi_variances) - 
                  (q_means ** 2) / q_variances
            )
        C2_coef = q_means/q_variances - pi_means/pi_variances
        C3_coef = (1.0 / 2.0 * pi_variances - 1.0 / 2.0 * q_variances)
        
        """Compute coefficients needed after taking squares"""
        #Dk_coef = polynomial coefficient for (k-1)-th order
        D1_coef = C1_coef ** 2
        D2_coef = 2 * C1_coef * C2_coef
        D3_coef = 2 * C1_coef*C3_coef + C2_coef ** 2
        D4_coef = 2 * C2_coef * C3_coef
        D5_coef = C3_coef ** 2
        
        """Compute the k-th moments of the normal"""
        #Mk = (k-1)-th moment
        M1_coef = 1.0
        M2_coef = q_means
        M3_coef = q_means ** 2 + q_variances
        M4_coef = q_means ** 3 + 3 * q_means * q_variances
        M5_coef = q_means ** 4 + 6 * q_means ** 2 * q_variances + 3 * q_variances ** 2
        
        """compute & return it"""
        coef_part = np.sum(
                D1_coef * M1_coef + 
                D2_coef * M2_coef + 
                D3_coef * M3_coef + 
                D4_coef * M4_coef + 
                D5_coef * M5_coef
                )
        return coef_part


    def prior_regularizer(self, q_params, q_parser, converter):
        """Simply unpack means & variances + compute discrepancy"""
        
        q_variances, q_means, q_variances_sig, q_means_sig = self.unpack(
                q_params, q_parser, converter)
        
        if q_variances_sig is None and q_means_sig is None:
            
            ED = self.ED_term(q_variances, self.pi_variances, 
                              q_means, self.pi_means)

        else:
            
            ED = self.ED_term(q_variances, 
                         self.pi_variances[converter.flatten()], 
                         q_means, self.pi_means[converter.flatten()])
            ED = ED + self.ED_term(q_variances_sig, 
                         self.pi_variances[converter.flatten() == False],
                         q_means_sig, self.pi_means[converter.flatten() == False])

        return self.weight*ED