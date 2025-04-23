from scipy.stats import norm
import numpy as np
import pandas as pd
import stan
from tqdm import trange


'''
Parameters
'''

n_trials = 10
data_per_trial = 1000
posterior_samples_per_trial = 1000
chains_per_trial = 4
# Folder must already exist, and have subfolders /data and /posterior
data_path = './experiment_temp' 

with open('normal_mixture.stan','r') as f:
    stan_model = f.read()
    f.close()
with open('normal_mixture_product.stan', 'r') as f:
    stan_joint = f.read()
    f.close()

''' Functions '''
def generate_data_sample(n_data):
    """ Generate a random sample from two independent Gaussian r.v. """
    mu1, mu2 = norm.rvs(size=2) 
    x1 = norm.rvs(size=n_data) + mu1
    x2 = norm.rvs(size=n_data) + mu2 
    df = pd.DataFrame(
        {'x1':x1, 'x2':x2}
    )
    return df

def generate_marginal_posterior_sample(data, num_samples, num_chains):
    """ Use stan to generate a sample from the posterior given 1d data """
    n_components = 2
    n = data.size
    data = {'n':n, 'x':data, 'beta':1/np.log(n), 'n_components':n_components}
    model = stan.build(stan_model, data=data) 
    fit = model.sample(
        num_chains=num_chains,
        num_samples=num_samples/num_chains,
    )
    df = fit.to_frame() 
    return df

def generate_joint_posterior_sample(data1, data2, num_samples, num_chains):
    """ Use stan to generate a sample from the posterior given 2d data """
    n_components = 2
    n = data1.size
    data = {'n':n, 'x1':data1, 'x2':data2, 'beta':1/np.log(n), 'n_components':n_components}
    model = stan.build(stan_joint, data=data) 
    fit = model.sample(
        num_chains=num_chains,
        num_samples=num_samples/num_chains,
    )
    df = fit.to_frame() 
    return df

def generate_one_trial(num_data, num_samples, num_chains):
    data_df = generate_data_sample(num_data)
    posterior_df_joint = generate_joint_posterior_sample(
        data_df['x1'].to_numpy(),
        data_df['x2'].to_numpy(),
        num_samples,
        num_chains,
    )
    posterior_df_x1 = generate_marginal_posterior_sample(
        data_df['x1'].to_numpy(),
        num_samples,
        num_chains,
    )
    posterior_df_x2 = generate_marginal_posterior_sample(
        data_df['x2'].to_numpy(),
        num_samples,
        num_chains,
    )
    return (data_df, posterior_df_x1, posterior_df_x2, posterior_df_joint)

''' Data Generation '''
if __name__ == '__main__':
    print(f"Beginning generation of {n_trials} trials.")
    error = False
    for t in trange(n_trials):
        data_df, posterior_df_x1, posterior_df_x2, posterior_df_joint = generate_one_trial(
            data_per_trial,
            posterior_samples_per_trial,
            chains_per_trial,
        )
        try:
            data_df.to_csv(data_path+f'/data/trial_{t}.csv')
            posterior_df_joint.to_csv(data_path+f'/posterior/trial_{t}_joint.csv')
            posterior_df_x1.to_csv(data_path+f'/posterior/trial_{t}_x1.csv')
            posterior_df_x2.to_csv(data_path+f'/posterior/trial_{t}_x2.csv')
        except OSError as e:
            print(e)
            print("Make sure you set up your data & posterior subfolders in advance!")
            error = True
            break
    if not error:
        print("Complete!")
    else:
        print("Exited with error ^^^ ")



