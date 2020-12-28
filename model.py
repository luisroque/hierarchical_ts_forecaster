import pymc3 as pm
import theano.tensor as tt
import numpy as np

'''
# Features

1. Allows choosing what levels of the hierarchy are used in the estimation 
    (i.e. considered to generate new gaussian processes) - 
    fewer levels don't use all the potential information 
    nested in the hierarchies but can boost performance
2. Allows the user to choose what seasonality to consider (i.e. the periodic 
    kernel defined for the cov function of the GP)
'''

class HGPforecaster:
    """HGP forecaster

    Parameters
    ----------
    groups: dict
                train
                predict
                    'groups_idx'
                    'groups_n'
                    'groups_names'
                    'n'
                    's'
                    'n_series_idx'
                    'n_series'
                    'g_number'
                    'data'
    season: seasonality 
    levels: list
                levels to be used in the estimation (default uses all levels)

    """
    def __init__(
        self,
        groups_data,
        season,
        levels
    ):
        self.model = pm.Model()
        self.priors = {}
        self.params = {}
        self.trace = {}
        self.start = {}
        self.priors_names = {}
        self.chains = None
        self.season = season
        self.g = groups_data
        self.y_pred = None
        self.mp = None
        self.gp_dict={}
        self.pred_samples_fit=None
        self.pred_samples_predict=None
        if levels:
            self.levels = levels
        else:
            self.levels = list(self.g['train']['groups_names'].keys())
        
        self.X = np.arange(self.g['train']['n']).reshape(-1,1)
        
        self.group_gen = (x for x in groups['train']['groups_names'] if x in levels)
            
    def generate_priors(self):
        """Set up the priors for the model."""
        with self.model:
            # prior for the periodic kernel (seasonality)
            self.priors["period"] = pm.Laplace(
                    "period", self.season, 0.1)
            self.priors["hy_a0"] = pm.Normal(
                    "hy_a0", mu=0.0, sd=10)
            self.priors["a0"] = pm.Normal(
                    "a0", mu=0.0, sd=5, shape = self.g['train']['s'])
            for group in self.levels:
                # priors for the kernels of each group
                self.priors["l_t_%s" %group] = pm.Gamma(
                    'l_t_%s' %group, 
                    alpha=5, 
                    beta=1, 
                    shape = self.g['train']['groups_n'][group])
                self.priors["l_p_%s" %group] = pm.Gamma(
                    'l_p_%s' %group, 
                    alpha=2, 
                    beta=1, 
                    shape = self.g['train']['groups_n'][group])
                self.priors["eta_t_%s" %group] = pm.HalfNormal(
                    'eta_t_%s' %group, 
                    0.15,
                    shape = self.g['train']['groups_n'][group])
                self.priors["eta_p_%s" %group] = pm.HalfNormal(
                    'eta_p_%s' %group, 
                    0.5,
                    shape = self.g['train']['groups_n'][group])
                self.priors["sigma_%s" %group] = pm.HalfNormal(
                    'sigma_%s' %group, 
                    0.01,
                    shape = self.g['train']['groups_n'][group])

                # Priors for hyperparamters
                self.priors["hy_a_%s" %group] = pm.Normal(
                    "hy_a_%s" %group, 
                    mu=0.0, 
                    sd=5.)
                self.priors["hy_b_%s" %group] = pm.Normal(
                    "hy_b_%s" %group, 
                    mu=0.0, 
                    sd=0.5)

                # priors for the group effects
                self.priors["a_%s" %group] = pm.Normal(
                    'a_%s' %group, 
                    self.priors["hy_a_%s" %group],
                    1,
                    shape = self.g['train']['groups_n'][group])
                self.priors["b_%s" %group] = pm.Normal(
                    'b_%s' %group, 
                    self.priors["hy_b_%s" %group],
                    0.1,
                    shape = self.g['train']['groups_n'][group])

        self.priors_names = {k: v.name for k, v in self.priors.items()}

    def generate_GPs(self):
        self.generate_priors()
        
        gp_dict = {}
        f_dict = {}
        f_flat = {}
        idx_dict = {}
        
        with self.model:
            for group in self.levels:
                for idx, name in enumerate(self.g['train']['groups_names'][group]):

                    # index varible that indicates where a specific GP is active
                    # for instance, GP_fem is only active in fem time series
                    idx_dict[name] = np.where(self.g['train']['groups_idx'][group]==idx,1,0)

                    # mean function for the GP with specific parameters per group
                    mu_func = pm.gp.mean.Linear(intercept = self.priors["a_%s" %group][idx],
                                            coeffs = self.priors["b_%s" %group][idx])

                    # cov function for the GP with specific parameters per group
                    cov = (self.priors["eta_t_%s" %group][idx]**2 * pm.gp.cov.Matern32(input_dim=1, ls=self.priors["l_t_%s" %group][idx])
                            + self.priors["eta_p_%s" %group][idx]**2 * pm.gp.cov.Periodic(1, period=self.priors["period"], ls=self.priors["l_p_%s" %group][idx]) 
                            + pm.gp.cov.WhiteNoise(self.priors["sigma_%s" %group][idx]))

                    self.gp_dict[name] = pm.gp.Latent(mean_func=mu_func, cov_func=cov)
                    f_dict[name] = self.gp_dict[name].prior('f_%s' % name, X=self.X, reparameterize=True)
                    f_flat[name] = tt.tile(f_dict[name], (self.g['train']['s'],)) * idx_dict[name]

            self.f = sum(f_flat.values())
        
    def likelihood(self):
        self.generate_GPs()

        with self.model:
            self.y_pred = pm.Poisson('y_pred', mu=tt.exp(self.f + self.priors["a0"][self.g['train']['n_series_idx']]), observed=self.g['train']['data'])

    def fit(self):
        self.likelihood()
        with self.model:
            print('Fitting model...')
            self.mp = pm.find_MAP(maxeval=5000)
            self.trace = {k: np.array([v]) for k, v in self.mp.items()}
            print('Sampling...')
            self.pred_samples_fit = pm.sample_posterior_predictive([self.mp], 
                                                  vars=[self.y_pred], 
                                                  samples=500)

    def predict(self):
        f_new = {}
        f_flat_new = {}
        idx_dict_new = {}

        n_new = self.g['predict']['n']
        X_new = np.arange(n_new).reshape(-1,1)

        with self.model:
            for group in self.levels:
                for idx, name in enumerate(self.g['predict']['groups_names'][group]):
                    idx_dict_new[name] = np.where(self.g['predict']['groups_idx'][group]==idx,1,0)
                    f_new[name] = self.gp_dict[name].conditional('f_new%s'%name, Xnew = X_new)
                    f_flat_new[name] = tt.tile(f_new[name], (self.g['predict']['s'],)) * idx_dict_new[name]

            f_ = sum(f_flat_new.values())
    
            y_pred_new = pm.Poisson("y_pred_new", 
                            mu=pm.math.exp(f_ + self.priors["a0"][self.g['predict']['n_series_idx']]), 
                            shape=n_new * self.g['predict']['s'])
            print('Sampling...')
            self.pred_samples_predict = pm.sample_posterior_predictive([self.mp], 
                                          vars=[y_pred_new], 
                                          samples=500)