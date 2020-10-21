// Basic Poisson model
data {
  // number of observations
  int<lower=0> N;
  // outcome
  int<lower=0> Y[N];
  
  // prediction stuff
  //dimension of prediction data
  int<lower=0> N_test;
  // y_pred to evaluate loglik
  int<lower=0> Y_test[N_test];
}

parameters {
  // name lambda 'beta' because that works better with rest of simulations
  real<lower=0> beta;
}

model {
  // prior on lambda
  beta ~ gamma(4, 1);
  
  Y ~ poisson(beta);
  
}

generated quantities {
  real Y_pred[N_test];
  real loglik[N_test];
  
  vector[N_test] betas = rep_vector(beta, N_test);
  
  Y_pred = poisson_rng(betas);
  
  for (n in 1:N_test){
    loglik[n] = poisson_lpmf(Y_test[n] | beta);
  }
}
