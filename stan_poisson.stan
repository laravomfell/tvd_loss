// Basic Poisson model
data {
    // number of observations
    int<lower=0> N;
    // dimension of X
    int<lower=0> P;
    // provide X
    matrix[N,P] X;
    // outcome
    int<lower=0> Y[N];
    
    // prediction stuff
    //dimension of prediction data
    int<lower=0> N_test;
    // X_pred
    matrix[N_test,P] X_test;
    // y_pred to evaluate loglik
    int<lower=0> Y_test[N_test];
}

parameters {
    vector[P] beta;
}

model {
    // prior on beta
    beta ~ normal(0, 1);
    // keep everything on the log scale
    Y ~ poisson_log(X * beta);
    
}

generated quantities {
    real Y_pred[N_test];
    real loglik[N_test];
    vector[N_test] XB;
    
    XB = X_test * beta;
    
    Y_pred = poisson_log_rng(XB);
    
    for (n in 1:N_test){
        loglik[n] = poisson_log_lpmf(Y_test[n] | XB[n]);
    }
}
