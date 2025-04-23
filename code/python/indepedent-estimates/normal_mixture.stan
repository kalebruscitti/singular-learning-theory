functions {
  real mixture_lpdf(real x, vector rho, vector mu) {
    int k = num_elements(rho);
    vector[k] log_weights;
    for(j in 1:k) {
      log_weights[j] = log(rho[j]) + normal_lpdf(x | mu[j], 1);
    }
    return log_sum_exp(log_weights);
  }
}
data {
  int<lower=0> n;     // number of samples 
  vector[n] x;        // observations
  real<lower=0> beta; // inverse temperature
  int<lower=1> n_components;     // number of components to mix
}
parameters {
  simplex[n_components] rho;
  vector[n_components] mu;
}
model {
  // priors
  rho ~ dirichlet(rep_vector(1, n_components));
  for(j in 1:n_components) {
    mu[j] ~ normal(0, 2);
  }
  
  for (i in 1:n) {
    target += beta*mixture_lpdf(x[i]| rho, mu);
  }
}
generated quantities {
  real log_prob_data = 0; //compute the log probability of the data using posterior samples
  for (i in 1:n) {
    log_prob_data += mixture_lpdf(x[i] | rho, mu);
  }
}