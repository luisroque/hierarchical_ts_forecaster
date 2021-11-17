import os
import torch
import tqdm
import math
import gpytorch
from torch.nn import Linear
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution, \
    LMCVariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from matplotlib import pyplot as plt
import pandas as pd
from libs.metrics import calculate_metrics, metrics_to_table, metrics_to_latex
from libs.pre_processing import generate_groups_data_flat, generate_groups_data_matrix,data_transform
from libs.visual_analysis import visualize_fit, visualize_predict, traceplot, visualize_prior, plot_elbo, plot_gps_components, model_graph
from libs.model import HGPforecaster
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error


prison = pd.read_csv('./data/prisonLF.csv', sep=",")
prison = prison.drop('Unnamed: 0', axis =1)
prison['t'] = prison['t'].astype('datetime64[ns]')
prison_pivot = prison.pivot(index='t',columns=['state', 'gender', 'legal'], values='count')
groups_input = {
    'state': [0],
    'gender': [1],
    'legal': [2]
}

groups = generate_groups_data_flat(y = prison_pivot, 
                               groups_input = groups_input, 
                               seasonality=4, 
                               h=8)

groups = generate_groups_data_matrix(groups)
dataset = 'prison' 
dt = data_transform(groups)
groups = dt.std_transf_train()
train_x = torch.arange(40)
train_x = train_x.type(torch.DoubleTensor)
train_x = train_x.unsqueeze(-1)
train_y = torch.from_numpy(groups['train']['data'])

training_iterations=200

idxs = []
for k, val in groups['train']['groups_idx'].items():
    idxs.append(val)

# build the matrix
#     Group1     |   Group2
# GP1, GP2, GP3  | GP1, GP2
# 0  , 1  , 1    | 0  , 1  
# 1  , 0  , 0    | 1  , 0  
# 0  , 1, , 1    | 0  , 1 
# 1  , 0  , 1    | 1  , 0  

idxs_t = np.array(idxs).T
n_groups = np.sum(np.fromiter(groups['train']['groups_n'].values(), dtype='int32'))
known_mixtures = np.zeros((groups['train']['s'], n_groups))
k=0

for j in range(groups['train']['g_number']):
    for i in range(np.max(idxs_t[:,j])+1):
        idx_to_1 = np.where(idxs_t[:,j]==i)
        known_mixtures[:,k][idx_to_1] = 1
        k+=1

plt.plot(train_y[:,0]);
covs = []
for i in range(1, n_groups+1):
    cov = gpytorch.kernels.RBFKernel() + gpytorch.kernels.PeriodicKernel()
    covs.append(cov)

# apply mixtures to covariances
selected_covs = []
mixed_covs = []
for i in range(groups['train']['s']):
    mixture_weights = known_mixtures[i]
    for w_ix in range(n_groups):
        w = mixture_weights[w_ix]
        if w == 1.0:
            selected_covs.append(covs[w_ix])
    mixed_cov = selected_covs[0]
    for cov in range(1,len(selected_covs)):
        mixed_cov += selected_covs[cov] # because GP(cov1 + cov2) = GP(cov1) + GP(cov2)
    mixed_covs.append(mixed_cov) 
    selected_covs = [] # clear out cov list

class LinearMean(gpytorch.means.Mean):
    def __init__(self, input_size, batch_shape=torch.Size(), bias=True):
        super().__init__()
        self.register_parameter(name="weights", parameter=torch.nn.Parameter(torch.randn(*batch_shape, input_size, 1)))
        if bias:
            self.register_parameter(name="bias", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))
        else:
            self.bias = None

    def forward(self, x):
        x = x.float()
        res = x.matmul(self.weights).squeeze(-1)
        if self.bias is not None:
            res = res + self.bias
        return res

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, cov):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = LinearMean(train_x.size(-1))
        #print(self.mean_module(torch.arange(40).float()))
        self.covar_module = cov

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

model_list = []
likelihood_list = []
for i in range(groups['train']['s']):
    likelihood_list.append(gpytorch.likelihoods.GaussianLikelihood())
    model_list.append(ExactGPModel(train_x, train_y[:,i], likelihood_list[i], mixed_covs[i]))

model = gpytorch.models.IndependentModelList(*model_list)
likelihood = gpytorch.likelihoods.LikelihoodList(*likelihood_list)
from gpytorch.mlls import SumMarginalLogLikelihood

mll = SumMarginalLogLikelihood(likelihood, model)
# this is for running the notebook in our testing framework

# Find optimal model hyperparameters

model.train()
likelihood.train()

# Use the Adam optimizer

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

for i in range(200):
    optimizer.zero_grad()
    output = model(*model.train_inputs)
    loss = -mll(output, model.train_targets)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
    optimizer.step()

# Set into eval mode
model.eval()
likelihood.eval()

# Make predictions (use the same test points)
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = torch.arange(48).type(torch.DoubleTensor)
    # This contains predictions for both outcomes as a list
    #predictions = likelihood(*model(test_x, test_x))
    predictions = likelihood(*model(*[test_x for i in range(10)]))

mean = np.zeros((1, groups['predict']['n'], groups['predict']['s']))
lower = np.zeros((1, groups['predict']['n'], groups['predict']['s']))
upper = np.zeros((1, groups['predict']['n'], groups['predict']['s']))

i=0
for pred in predictions:
    mean[:,:,i] = pred.mean
    lower[:,:,i], upper[:,:,i] = pred.confidence_region()
    i+=1

mean = ((mean*dt.std_data) + dt.mu_data)
lower = ((lower*dt.std_data) + dt.mu_data)
upper = ((upper*dt.std_data) + dt.mu_data)
groups = dt.inv_transf_train()

pred = groups['predict']['data'].reshape((groups['predict']['s'], -1)).T

def mase(n,seas,h,y,f):
    return np.mean(((n-seas)/h
            * (np.sum(np.abs(y[n:n+h,:] - f), axis=0)
               / np.sum(np.abs(y[seas:n, :] - y[:n-seas, :]), axis=0))))


def calculate_metrics(pred_samples,
                      groups):

    pred_s0 = pred_samples.shape[0]
    pred_s1 = pred_samples.shape[1]
    pred_s2 = pred_samples.shape[2]

    pred_samples = pred_samples.reshape(pred_s0, pred_s1*pred_s2, order='F')

    seasonality = groups['seasonality']
    h = groups['h']
    
    n = groups['predict']['n']
    s = groups['predict']['s']
    y_f = groups['predict']['data'].reshape(s, n).T

    y_all_g = {}
    f_all_g = {}
    
    mase_ = {}
    rmse_ = {}
    
    # Bottom
    y_all_g['bottom'] = y_f
    f_all_g['bottom'] = np.mean(pred_samples, axis=0).reshape(s, n).T[n-h:n,:]
    
    mase_['bottom'] = np.round(mase(n=n-h, 
                                     seas=seasonality, 
                                     h=h, 
                                     y=y_f, 
                                     f=np.mean(pred_samples, axis=0).reshape(s, n).T[n-h:n,:]),3)
    rmse_['bottom'] = np.round(mean_squared_error(y_f[n-h:n,:], np.mean(pred_samples, axis=0).reshape(s, n).T[n-h:n,:], squared=False), 3)

    # Total
    y_all_g['total'] = np.sum(y_f, axis=1).reshape(-1,1)
    f_all_g['total'] = np.sum(np.mean(pred_samples, axis=0).reshape(s, n).T[n-h:n,:], axis=1).reshape(-1,1)
    
    mase_['total'] = np.round(mase(n=n-h, 
                                     seas=seasonality, 
                                     h=h, 
                                     y=np.sum(y_f, axis=1).reshape(-1,1), 
                                     f=np.sum(np.mean(pred_samples, axis=0).reshape(s, n).T[n-h:n,:], axis=1).reshape(-1,1))
                            ,3)
    rmse_['total'] = np.round(mean_squared_error(np.sum(y_f, axis=1).reshape(-1,1)[n-h:n,:], 
                                             np.sum(np.mean(pred_samples, axis=0).reshape(s, n).T[n-h:n,:],axis=1).reshape(-1,1), 
                                             squared=False), 3)

    # Groups
    idx_dict_new = {}
    for group in list(groups['predict']['groups_names'].keys()):
        y_g = np.zeros((groups['predict']['n'], groups['predict']['groups_names'][group].shape[0]))
        f_g = np.zeros((h, groups['predict']['groups_names'][group].shape[0]))

        for idx, name in enumerate(groups['predict']['groups_names'][group]):               

            g_n = groups['predict']['groups_n'][group]

            idx_dict_new[name] = np.where(groups['predict']['groups_idx'][group]==idx,1,0)

            y_g[:,idx] = np.sum(idx_dict_new[name]*y_f, axis=1)
            f_g[:,idx] = np.sum(idx_dict_new[name]*np.mean(pred_samples, axis=0).reshape(s, n).T, axis=1)[n-h:n]

        y_all_g[group] = np.sum(y_g, axis=1).reshape(-1,1)
        f_all_g[group] = np.sum(f_g, axis=1).reshape(-1,1)

        mase_[group] = np.round(mase(n=n-h, 
                                     seas=seasonality, 
                                     h=h, 
                                     y=y_g, 
                                     f=f_g)
                                ,3)

        rmse_[group] = np.round(mean_squared_error(y_g[n-h:n,:], f_g, squared=False), 3)

    # All
    y_all = np.concatenate([y_all_g[x] for x in y_all_g], 1)
    f_all = np.concatenate([f_all_g[x] for x in f_all_g], 1)

    mase_['all'] = np.round(mase(n=n-h, 
                         seas=seasonality, 
                         h=h, 
                         y=y_all, 
                         f=f_all),3)
    rmse_['all'] = np.round(mean_squared_error(y_all[n-h:n,:], f_all, squared=False), 3)
    
    results = {}
    results['mase'] = mase_
    results['rmse'] = rmse_
    return results

res = calculate_metrics(mean, groups)

with open(f'results_gp_cov_{dataset}.pickle', 'wb') as handle:
    pickle.dump(res, handle, pickle.HIGHEST_PROTOCOL)
