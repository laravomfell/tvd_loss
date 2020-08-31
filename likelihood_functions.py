#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 09:35:37 2020

@author: jeremiasknoblauch

Description: Likelihood function wrappers for use within NPL class
"""

import numpy as np
from scipy.stats import poisson, norm
from scipy.optimize import minimize
import statsmodels.api as sm

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

# set the seeds and make sure computations are reproducible
torch.manual_seed(0)


class Likelihood():
    """An empty/abstract class that dictates what functions a sub-class of the 
    likelihood type needs to have"""
    
    def __init__(self, name = None):
        self.name = name
    
    def initialize(self,Y,X):
        """Returns an initialization for the likelihood parameters, typically
        based on the maximum likelihood estimate."""
        return 0

    def evaluate(self, params, Y, X):
        """Letting Y be of shape (n,) and X of shape (n,d), compute the 
        likelihoods of each pair (Y[i], X[i,:]) at parameter value param"""
        return 0
    

class NN_logsoftmax(nn.Module):
    """Build a new class for the network you want to run, returning log 
    softmax"""
    
    def set_parameters(self, initializers):
        """Set the parameter values obtained from vanilla NN as initializers"""
        with torch.no_grad():
            self.fc1.weight.data = torch.from_numpy(initializers[0].copy())
            self.fc1.bias.data = torch.from_numpy(initializers[1].copy())
            self.fc2.weight.data = torch.from_numpy(initializers[2].copy())
            self.fc2.bias.data = torch.from_numpy(initializers[3].copy())
          
    """Single layer network with layer_size nodes"""
    def __init__(self, d, layer_size, num_classes):
        super(NN_logsoftmax, self).__init__()
        self.fc1 = nn.Linear(d, layer_size)
        self.fc2 = nn.Linear(layer_size, num_classes)
    
    """Return the log softmax values for each of the classes"""
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim = 1)


class NN_softmax(NN_logsoftmax):
    """Build a new class for the network you want to run, returning non-log 
    softmax"""

    """Return the softmax values for each of the classes"""
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim = 1)
    

class SoftMaxNN(Likelihood):
    """Get a single-layer softmax-final-layer neural network to classify data"""
    
    def __init__(self, d, layer_size, num_classes, reset_initializer,
                 batch_size = 64, epochs_vanilla = 10, epochs_TVD = 100, 
                 learning_rate = 0.01):
        
        self.NN_vanilla = NN_logsoftmax(d,layer_size, num_classes)
        self.NN_TVD = NN_softmax(d,layer_size, num_classes)
        self.d = d
        self.layer_size = layer_size
        self.num_classes = num_classes
        self.reset_initializer = reset_initializer
        self.batch_size = batch_size 
        # for the SGD-optimizer of initializer AND of TVD
        self.epochs_vanilla = epochs_vanilla
        self.epochs_TVD = epochs_TVD
        self.learning_rate = learning_rate
        
    
    def store_transformed_Y(self, Y):
        """From Y = [0,2,1] -> Y = [[1,0,0], [0,0,1], [0,1,0]]"""
        n = len(Y)
        self.Y_new = np.zeros((n,self.num_classes))
        #DEBU:G THIS ONLY WORKS IF Y IS BINARY!
        self.Y_new[range(0,n), Y.copy()] = 1.0
        self.Y_new = torch.from_numpy(self.Y_new.copy()).float()
    
    
    def initialize(self, Y, X, weights = None):
            
        # helper function to make batches
        def make_batches(N_data, batch_size):
            return [ slice(i, min(i + batch_size, N_data)) 
                    for i in range(0, N_data, batch_size) ]

        
        # for the future minimization of the TVD-networks, store the Y's
        # in a different format
        self.store_transformed_Y(Y)
        
        # make the batches
        batches = make_batches(X.shape[0], self.batch_size)
    
        # create a stochastic gradient descent optimizer
        optimizer = torch.optim.SGD(self.NN_vanilla.parameters(), 
                                    lr=self.learning_rate, 
                                    momentum=0.9)
        # create a loss function
        criterion = nn.NLLLoss()
    
        # run the main training loop
        for epoch in range(self.epochs_vanilla):
            
            # Shuffle the data differently, depending on weights
            if weights is None:
                permutation = np.random.choice(range(X.shape[ 0 ]), 
                                         X.shape[ 0 ], replace = False)
            else:
                permutation = np.random.choice(range(X.shape[ 0 ]), 
                         X.shape[ 0 ], replace = True, 
                         p = weights.flatten())
            
   
            for batch_slice in batches:
                
                # convert the slice into a range object (i.e., indices)
                batch_indices = range(*batch_slice.indices(X.shape[0]))
                
                # Sub-set the data with these indices
                X_batch = X[permutation[batch_indices],:]
                Y_batch = Y[permutation[batch_indices]]
                
                # convert to tensor
                X_batch = torch.from_numpy(X_batch).float()
                Y_batch = torch.from_numpy(Y_batch).long()
                
                X_batch, Y_batch = Variable(X_batch), Variable(Y_batch)
                
                # pytorch accumulates gradients, so make sure to reset to 0
                optimizer.zero_grad()
                
                # feed your X batch to the network, defining a function 
                # f: X -> Probabilities({1,2,... num_classes})
                net_out = self.NN_vanilla(X_batch)
                
                # determine the optimality criterion for the parameters of 
                # this function by specifying a loss 
                loss = criterion(net_out, Y_batch)
                
                # this computes the gradient (backward propagation)
                loss.backward()
                
                # take a gradient step
                optimizer.step()
                
        print('Train Epoch: {} Loss: {:.6f}'.format(
                    epoch, loss))
        
        
        
        # store the resulting parameters
        self.initializer = self.get_network_parameters(self.NN_vanilla)
        
        # set parameters in TVD network to those optimal for vanilla NN
        self.NN_TVD.set_parameters(self.initializer)
        
        # ensure that we can store the initializers in the NPL object
        return self.initializer
        
        # if weights are NOT none, I need to perform SGD with 
        # weighted re-sampling
        
        # NOTE: IT MAKES SENSE TO FIRST TRY IF WE CAN JUST INITIALIZE TO THE
        #       SAME PARAMETERS (REGARDLESS OF WEIGHTS)...
    
    def get_network_parameters(self, network):
        initializers = []
        
        # for name, param in self.NN_vanilla.named_parameters():
        #     if False:
        #         print('name: ', name)
        #         print(type(param))
        #         print('param.shape: ', param.shape)
        #         print('param', param.data)
        #         print('param.requires_grad: ', param.requires_grad)
        #         print('=====')
        #     initializers += [param.data] # add param values of this layer
        #     names += [name] # add param name of this layer
        
        for name, param in network.named_parameters():
            initializers += [param.data.detach().numpy().copy()]
        return initializers
            
    
    def evaluate(self, Y_unique, X_unique):
        return None
    
    
    def minimize_TVD(self, Y, X, weights):
        
        # set parameters in TVD network to those optimal for vanilla NN  
              
        if self.reset_initializer:
            self.NN_TVD.set_parameters(self.initializer)
    
        # create the TVD loss function
        def TVD_loss(network_outputs, weights, transformed_Y):
            
            # Note that if all observation pairs (X,Y) are unique, we can 
            # simplify the loss to 
            # \sum_{x_i \in \sample} w(x) * 
            #           \sum_{y=1,...,num_classes}|p_\theta(y|x) - 1_{y_i = y}| 
            
            Y_cond_X_model = network_outputs
            
            estimated_TVD = 0.5 * torch.sum(
                torch.from_numpy(weights).float() * 
                torch.sum(torch.abs(Y_cond_X_model - transformed_Y), axis=1) 
                )
             
            return estimated_TVD
        
        
        # convert X and Y to Variables
        X = torch.from_numpy(X).float()
        
        # define a closure (which clears gradients and computes the loss)
        def closure():
            net_out = self.NN_TVD(X)
            loss = TVD_loss(net_out, weights, self.Y_new)
            self.optimizer.zero_grad()
            loss.backward()
            return loss
        
        # create a stochastic gradient descent optimizer
        self.optimizer = torch.optim.LBFGS(self.NN_TVD.parameters(), lr=self.learning_rate)
        self.optimizer.step(closure)
        
        # extract the parameter value at the optimum and return it
        return self.get_network_parameters(self.NN_TVD)
    
    
    # NOTE: We don't use this
    def minimize_TVD_SGD(self, Y, X, weights):
        
        # set parameters in TVD network to those optimal for vanilla NN  
              
        if self.reset_initializer:
            self.NN_TVD.set_parameters(self.initializer)
        
        # for name, param in self.NN_TVD.named_parameters():
        #     if False:
        #         # print('name: ', name)
        #         # print(type(param))
        #         # print('param.shape: ', param.shape)
        #         print('param', param.data)
        #         # print('param.requires_grad: ', param.requires_grad)
        #         print('=====')
        
    
        # create a stochastic gradient descent optimizer
        optimizer = torch.optim.SGD(self.NN_TVD.parameters(), 
                                    lr=self.learning_rate, 
                                    momentum=0.9)
        
        # create the TVD loss function
        def TVD_loss(network_outputs, weights, transformed_Y):
            
            # Note that if all observation pairs (X,Y) are unique, we can 
            # simplify the loss to 
            # \sum_{x_i \in \sample} w(x) * 
            #           \sum_{y=1,...,num_classes}|p_\theta(y|x) - 1_{y_i = y}| 
            
            Y_cond_X_model = network_outputs
            
            estimated_TVD = 0.5 * torch.sum(
                torch.from_numpy(weights).float() * 
                torch.sum(torch.abs(Y_cond_X_model - transformed_Y), axis=1) 
                )
             
            return estimated_TVD
        
        
        # convert X and Y to Variables
        X = torch.from_numpy(X).float()
        # Y = torch.from_numpy(Y).long()
    
        # run the main training loop
        for epoch in range(self.epochs_TVD):
                
            # pytorch accumulates gradients, so make sure to reset to 0
            optimizer.zero_grad()
                
            # feed your X batch to the network, defining a function 
            # f: X -> Probabilities({1,2,... num_classes})
            net_out = self.NN_TVD(X)
                
            # determine the optimality criterion for the parameters of 
            # this function by specifying a loss 
            loss = TVD_loss(net_out, weights, self.Y_new)
            
            # this computes the gradient (backward propagation)
            loss.backward()
            
            # take a gradient step
            optimizer.step()
                
        print('Train Epoch: {} Loss: {:.6f}'.format(
                    epoch, loss))
        
        # extract the parameter value at the optimum and return it
        return self.get_network_parameters(self.NN_TVD)
    

    def predict(self, Y,X, parameter_sample):
        """Given a sample of (X,Y) as well as a sample of network parameters, 
        compute p_{\theta}(Y|X) and compare against the actual values of Y"""
        
        n,d = X.shape
        
        Y_new = np.zeros((n,self.num_classes))
        Y_new[range(0,n), Y] = 1.0
        
        # create an empty list to store full predictions, accuracy (0-1) & CE
        predictions = []
        accuracy = []
        cross_entropy = []
        
        # loop over parameter samples; each theta will be a list
        for theta in parameter_sample:
            
            # set the parameter values of the NN to theta
            # NOTE: We use the vanilla NN because it returns the log-softmax
            #       (rather than the softmax). Since we endow it with the 
            #       parameters extracted from the TVD network, this means that
            #       we will get the log softmax with the TVD-optimal weights 
            self.NN_vanilla.set_parameters(theta)
    
            # loop over observations
            for i in range(0,n):
                
                # compute the model probability vectors p_{\theta}(Y|X) w. fixed X
                # print("X[i,:]", X[i,:].shape)
                log_probabilities = self.NN_vanilla(torch.from_numpy(
                    X[i,:].reshape(1,d)).float())
                # print(model_probabilities.type)
                # print(model_probabilities.shape)
                
                # extract the probabilities and store them
                log_model_probabilities = log_probabilities.data.detach().numpy().copy()
                predictions += [log_model_probabilities]
            
                # compute accuracy (whether or not we made mistake in prediction)
                biggest_probability_index = np.argmax(log_model_probabilities)
                if biggest_probability_index == Y[i]:
                    accuracy += [1]
                else:
                    accuracy += [0]
                
                # compute cross-entropy 
                cross_entropy += [-np.sum(log_model_probabilities * Y_new[i,:])]
        
        # convert into numpy arrays
        accuracy = np.array(accuracy)
        predictions = np.exp(np.array(predictions))
        cross_entropy = np.array(cross_entropy)     
        
        return (predictions, accuracy, cross_entropy)

    def predict_initializer(self, Y, X):
        # safety-check: Does the TVD actually improve things? To check it does,
        # we also compute predictions, accuracy and cross-entropy wrt initializer
        
        n,d = X.shape
        
        Y_new = np.zeros((n,self.num_classes))
        Y_new[range(0,n), Y] = 1.0
        
        # create an empty list to store full predictions, accuracy (0-1) & CE
        predictions = []
        accuracy = []
        cross_entropy = []
        
        self.NN_vanilla.set_parameters(self.initializer)
    
        # loop over observations
        for i in range(0,n):
            
            # compute the model probability vectors p_{\theta}(Y|X) w. fixed X
            # print("X[i,:]", X[i,:].shape)
            log_probabilities = self.NN_vanilla(torch.from_numpy(
                X[i,:].reshape(1,d)).float())
            # print(model_probabilities.type)
            # print(model_probabilities.shape)
            
            # extract the probabilities and store them
            log_model_probabilities = log_probabilities.data.detach().numpy().copy()
            predictions += [log_model_probabilities]
        
            # compute accuracy (whether or not we made mistake in prediction)
            biggest_probability_index = np.argmax(log_model_probabilities)
            if biggest_probability_index == Y[i]:
                accuracy += [1]
            else:
                accuracy += [0]
            
            # compute cross-entropy 
            cross_entropy += [-np.sum(log_model_probabilities * Y_new[i,:])]
        
        # convert into numpy arrays
        accuracy = np.array(accuracy)
        predictions = np.exp(np.array(predictions))
        cross_entropy = np.array(cross_entropy)     
        
        return (predictions, accuracy, cross_entropy)

class ProbitLikelihood(Likelihood):

    def __init__(self, name="Probit"):
        self.name = name
        
    def initialize(self, Y, X, weights = None):
        # return MLE init
        # model = sm.GLM(Y, X, family = sm.families.Binomial(), weight = weights)
        # MLE = model.fit()
        
        MLE = sm.GLM(Y, X, family = sm.families.Binomial(
            sm.families.Binomial.links[1]),
                      weight = weights).fit().params
        # MLE = sm.Probit(Y,X).fit().params

        return MLE
    
    def evaluate(self, params, Y_unique, X_unique):
        
        # first comupute coefficients * X
        inner_products = np.matmul(X_unique, params)
        
        # Second, use the normal cdf to transform into [0,1]
        probs = norm.cdf(inner_products)

        # Y|X is given by probs if Y = 1 and 1 - probs if Y = 0
        n_X_unique = X_unique.shape[0]
        n_Y_unique = 2 # WE ASSUME BOTH LABELS ARE OBSERVED AT LEAST ONCE
        
        Y_given_X_model = np.zeros((n_Y_unique, n_X_unique))

        Y_given_X_model[0,:] = 1.0 - probs
        Y_given_X_model[1,:] = probs
                
        return Y_given_X_model
    
    def predict(self, Y,X, parameter_sample):
        """Given a sample of (X,Y) as well as a sample of network parameters, 
        compute p_{\theta}(Y|X) and compare against the actual values of Y"""
        
        # first comupute coefficients * X; shape (n,B)
        n = X.shape[0]
        B, _ = parameter_sample.shape
        inner_products = np.matmul(X, np.transpose(parameter_sample))
        
        # Second, use the normal cdf to transform into [0,1]
        log_probs = norm.logcdf(inner_products)
        
        # compute predictions
        #print("[np.where(probs > 0.5)]", list(zip(np.where(mat > 0.5)[0], np.where(mat > 0.5)[1])).shape)
        predictions = np.zeros((n,B))
        tuples = np.where(np.exp(log_probs) > 0.5)
        indices = np.array(list(zip(tuples[0], tuples[1])))
        predictions[indices] = 1
        
        # use predictions for accuracy computation
        accuracy = np.abs(predictions - Y[:,np.newaxis])
        
        # compute cross-entropy
        cross_entropy = -(log_probs * Y[:, np.newaxis] + 
                         np.logaddexp(0,-log_probs) * (1.0-Y[:, np.newaxis]))
       
        return (log_probs, accuracy, cross_entropy) 
    

class PoissonLikelihood(Likelihood):
    """Use the traditional link function lambda(x) = exp(xb)"""

    def __init__(self, name = "Poisson with log-link"):
         self.name = name
        
    def initialize(self, Y, X, weights = None):
        # return MLE init
        MLE = sm.GLM(Y, X, family = sm.families.Poisson(sm.families.Binomial.links[1]),
                     weight = weights).fit().params      
        return MLE
    
    def evaluate(self, params, Y_unique, X_unique):
        # first comupute lambda
        lambdas = np.exp(np.matmul(X_unique, params))
        
        # Second, use the standard poisson pmf to evaluate the likelihood
        n_X_unique = X_unique.shape[0]
        n_Y_unique = Y_unique.shape[0]
        
        Y_given_X_model = poisson.pmf(
                np.repeat(Y_unique,n_X_unique).reshape(n_Y_unique, n_X_unique),
                np.tile(lambdas, n_Y_unique).reshape(n_Y_unique, n_X_unique) 
                )
                
        return Y_given_X_model
    
    def predict(self, Y,X, parameter_sample):
        """Given a sample of (X,Y) as well as a sample of network parameters, 
        compute p_{\theta}(Y|X) and compare against the actual values of Y"""
        
        B = parameter_sample.shape[0]
        
        
        # Get the lambda(X, \theta_i) for all parameters \theta_i in sample.
        # The i-th column corresponds to lambda(X, \theta_i).
        lambdas = np.exp(np.matmul(X, np.transpose(parameter_sample)))
        
        # Get the predictive/test likelihoods
        predictive_likelihoods = poisson.pmf(
            np.repeat(Y, B).reshape(len(Y), B),lambdas)   
        
        # Get the MSE and MAE
        SE = (np.repeat(Y, B).reshape(len(Y), B) - lambdas)**2
        AE = np.abs(np.repeat(Y, B).reshape(len(Y), B) - lambdas)
        
        return (predictive_likelihoods, SE, AE)
        


class PoissonLikelihoodSqrt(Likelihood):
    """Use the link function lambda(x) = |abs(x)|^1/2 to make the gradients
    nicer. We still use the (transformed) MLE for initialization"""
    
    def __init__(self, d):
        self.d = d
        self.X_mean = None
        
    def set_X_mean(self, X_mean):
        self.X_mean = X_mean
    
    def initialize(self, Y, X, weights = None):
        
        # check if the mean has been computed before
        if self.X_mean is None:
            self.set_X_mean(np.mean(X,0))
                
        MLE = sm.GLM(Y, X, family = sm.families.Poisson(), 
                     freq_weights = weights).fit().params
        
        # Taking a = parameter for the link function lambda(x) = exp(a*x), 
        # we use the standard MLE procedure to get the best a. Then,
        # we want to solve for param b in link function lambda(x) = |bx|^1/2.
        # PROBLEM: We won't get a one-to-one mapping because x_i varies with i
        # SOLUTION: Solve b for exp(a * E[x]) = |b * E[x]|^{1/2}
        # RATIONALE: E[x] should be representative for x_i
        params = np.power(np.exp(MLE * self.X_mean), 2.0) /np.abs(self.X_mean)
        return params

    
    def evaluate(self, params, Y_unique, X_unique):
        # First, compute lambda(x) = |bx|^1/2
        lambdas = np.power(np.abs(np.matmul(X_unique, 
                                            np.transpose(params))),1.0/2.0)
        # Second, use the standard poisson pmf to evaluate the likelihood
        n_X_unique = X_unique.shape[0]
        n_Y_unique = Y_unique.shape[0]
        
        Y_given_X_model = poisson.pmf(
                np.repeat(Y_unique,n_X_unique).reshape(n_Y_unique, n_X_unique),
                np.tile(lambdas, n_Y_unique).reshape(n_Y_unique, n_X_unique) 
                )
                
        return Y_given_X_model