import pymc3 as pm
import theano.tensor as tt
import numpy as np
from libs.pre_processing import generate_groups_data_matrix_minibatch, generate_groups_data_matrix

'''
Features:
1. The levels of the hierarchy that are used in the estimation 
    (i.e. considered to generate new gaussian processes) - 
    fewer levels don't use all the potential information 
    nested in the hierarchies but can boost performance
2. The seasonality to consider (i.e. the periodic 
    kernel defined for the cov function of the GP)
3. Option to define a piecewise function for the GP mean and respective
    selection of the number of changepoints
4. Option to define a linear function (defined as log linear considering 
    the log-link function used with the Poisson distribution)
5. Option to use MAP or VI to estimate the parameter values (VI is advised)
6. Possibility to use Minibatch that ensures scalability of the model
'''

class LogLinear(pm.gp.mean.Mean):
    def __init__(self, b, a=0):
        self.a = a
        self.b = b

    def __call__(self, X):
        # Log linear mean function -> adding 1 to avoid inf when X = 0
        return tt.squeeze(tt.dot(tt.log(X+1), self.b) + self.a)


class PiecewiseLinearChangepoints(pm.gp.mean.Mean):
    def __init__(self, 
                 k, 
                 m,
                 b,
                 intercept, 
                 changepoints):
        self.k = k
        self.m = m
        self.b = b
        self.a = intercept
        self.changepoints = changepoints

    def create_changepoints(X, changepoints):
        return (0.5 * (1.0 + tt.sgn(tt.tile(X.reshape((-1,1)), (1,len(changepoints_t))) - changepoints_t)))

    def __call__(self, X):
        size_r = X.shape[0]

        X = theano.shared(X)
            
        A = self.create_changepoints(X[:n,0], self.changepoints)
        
        piecewise = (self.k + tt.dot(A, self.b.reshape((-1,1))))*X + (self.m + tt.dot(A, (-self.changepoints * self.b).reshape((-1,1))))
        
        return (piecewise.reshape((-1,))
                       + tt.tile(self.a, (size_r,)))

class HGPforecaster:
    """HGP forecaster
    Parameters
    ----------
    groups_data: dict
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
                seasonality
                horizon
    levels: list
                levels to be used in the estimation (default uses all levels)
    changepoints: int
                define a piecewise function as the mean of the GPs based on the number of 
                changepoints defined by the user (uniformly distributed across time)
    n_iterations: int
                number of iterations to run on the optimization (MAP or VI)
    minibatch: list[n_points, n_series]
                list with number of points and number of series to consider
    log_lin_mean: bool
                define a linear (log-linear considering the log-link function used) 
                function as the mean of the GP
    """
    def __init__(
        self,
        groups_data,
        levels=None,
        changepoints=None,
        n_iterations=10000,
        minibatch=None,
        log_lin_mean = None
    ):
        self.model = pm.Model()
        self.priors = {}
        self.g = groups_data
        self.y_pred = None
        self.mp = None
        self.gp_dict={}
        self.pred_samples_fit=None
        self.pred_samples_predict=None
        self.season = self.g['seasonality']
        self.changepoints = changepoints
        self.n_iterations = n_iterations
        self.trace_vi = None
        self.pred_samples_fit = None
        if levels:
            self.levels = levels
        else:
            self.levels = list(self.g['train']['groups_names'].keys())

        self.minibatch = minibatch
        self.log_lin_mean = log_lin_mean
        if self.minibatch:
            self.g, self.X_mi = generate_groups_data_matrix_minibatch(self.g, self.minibatch[0], self.minibatch[1])
        else:
            self.g = generate_groups_data_matrix(self.g)

        self.X = np.arange(self.g['train']['n']).reshape(-1,1)

    def generate_priors(self):
        """Set up the priors for the model."""
        with self.model:

            if self.minibatch:
                self.series = self.g['train']['n_series_idx'].eval()
            else:
                self.series = self.g['train']['n_series_idx']
            
            self.series_full = self.g['train']['n_series_idx_full']

            self.priors["a0"] = pm.Normal(
                "a0", 
                mu=tt.log(np.mean(self.g['train']['full_data'][:,self.series_full], axis=0)), 
                sd=0.1, 
                shape = self.g['train']['s']) 

            # prior for the periodic kernel (seasonality)
            self.priors["period"] = pm.Laplace(
                    "period", self.season, 0.1)

            for group in self.levels:
                # priors for the kernels of each group

                # The inverse gamma is very useful to inform our prior dist of the length scale
                # because it supresses both zero and infinity.
                # The data don't inform length scales larger than the maximum covariate distance 
                # and shorter than the minimum covariate distance (distance between time points which 
                # is always 1 in our case).
                self.priors["l_t_%s" %group] = pm.InverseGamma(
                    'l_t_%s' %group, 
                    4, 
                    self.g['train']['n']/4, 
                    shape = self.g['train']['groups_n'][group])
                self.priors["l_p_%s" %group] = pm.InverseGamma(
                    'l_p_%s' %group, 
                    4, 
                    self.g['train']['n'], 
                    shape = self.g['train']['groups_n'][group])
                self.priors["eta_t_%s" %group] = pm.HalfNormal(
                    'eta_t_%s' %group, 
                    0.1,
                    shape = self.g['train']['groups_n'][group])
                self.priors["eta_p_%s" %group] = pm.HalfNormal(
                    'eta_p_%s' %group, 
                    0.2,
                    shape = self.g['train']['groups_n'][group])
                self.priors["sigma_%s" %group] = pm.HalfNormal(
                    'sigma_%s' %group, 
                    0.02,
                    shape = self.g['train']['groups_n'][group])

                if self.log_lin_mean:
                    self.priors["hy_b_%s" %group] = pm.Normal(
                        "hy_b_%s" %group, 
                        mu=0.0, 
                        sd=1.)
                    self.priors["b_%s" %group] = pm.Normal(
                        'b_%s' %group, 
                        self.priors["hy_b_%s" %group],
                        0.01,
                        shape = self.g['train']['groups_n'][group])
                elif self.changepoints:
                    # Priors for hyperparamters
                    self.priors["hy_b_%s" %group] = pm.Normal(
                        "hy_b_%s" %group, 
                        mu=0.0, 
                        sd=0.5)
                    self.priors["hy_a_%s" %group] = pm.Normal(
                        "hy_a_%s" %group, 
                        mu=0.0, 
                        sd=5.)

                    # priors for the group effects
                    self.priors["b_%s" %group] = pm.Normal(
                        'b_%s' %group, 
                        self.priors["hy_b_%s" %group],
                        0.1,
                        shape = self.g['train']['groups_n'][group])
                    self.priors["a_%s" %group] = pm.Normal(
                        'a_%s' %group, 
                        self.priors["hy_a_%s" %group],
                        1,
                        shape = self.g['train']['groups_n'][group])
                    self.priors["k_%s" %group] = pm.Normal(
                        'k_%s' %group, 
                        0.0,
                        0.1,
                        shape = self.g['train']['groups_n'][group])
                    self.priors["m_%s" %group] = pm.Normal(
                        'm_%s' %group, 
                        0.0,
                        0.1,
                        shape = self.g['train']['groups_n'][group])
                # Using linear kernel to model the mean of the GP (exponential)
                else:
                    # Linear mean
                    self.priors["c_%s" %group] = pm.Normal(
                        'c_%s' %group, 
                        0, 
                        0.05, 
                        shape = self.g['train']['groups_n'][group])
                    self.priors["eta_l_%s" %group] = pm.HalfNormal(
                        'eta_l_%s' %group, 
                        0.01,
                        shape = self.g['train']['groups_n'][group])


    def generate_GPs(self):
        self.generate_priors()
        
        gp_dict = {}
        f_dict = {}
        f_flat = {}
        idx_dict = {}
        
        with self.model:
            for group in self.levels:
                for idx, name in enumerate(self.g['train']['groups_names'][group]):

                    # mean function for the GP with specific parameters per group

                    if self.log_lin_mean:
                        mu_func = LogLinear(b = self.priors["b_%s" %group][idx])
                        
                        cov = (self.priors["eta_t_%s" %group][idx]**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=self.priors["l_t_%s" %group][idx])                                
                                + self.priors["eta_p_%s" %group][idx]**2 * pm.gp.cov.Periodic(1, period=self.priors["period"], ls=self.priors["l_p_%s" %group][idx]) 
                                + pm.gp.cov.WhiteNoise(self.priors["sigma_%s" %group][idx]))

                    elif self.changepoints:
                        mu_func = PiecewiseLinearChangepoints(intercept = self.priors["a_%s" %group][idx],
                                                              b = self.priors["b_%s" %group][idx],
                                                              changepoints = self.changepoints,
                                                              k = self.priors["k_%s" %group][idx],
                                                              m = self.priors["m_%s" %group][idx])

                        cov = (self.priors["eta_t_%s" %group][idx]**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=self.priors["l_t_%s" %group][idx])                                
                                + self.priors["eta_p_%s" %group][idx]**2 * pm.gp.cov.Periodic(1, period=self.priors["period"], ls=self.priors["l_p_%s" %group][idx]) 
                                + pm.gp.cov.WhiteNoise(self.priors["sigma_%s" %group][idx]))

                    else:
                        mu_func = pm.gp.mean.Zero()
                        # cov function for the GP with specific parameters per group
                        cov = (self.priors["eta_t_%s" %group][idx]**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=self.priors["l_t_%s" %group][idx])
                                + self.priors["eta_l_%s" %group][idx]**2 * pm.gp.cov.Linear(input_dim=1, c=self.priors["c_%s" %group][idx])
                                + self.priors["eta_p_%s" %group][idx]**2 * pm.gp.cov.Periodic(1, period=self.priors["period"], ls=self.priors["l_p_%s" %group][idx]) 
                                + pm.gp.cov.WhiteNoise(self.priors["sigma_%s" %group][idx]))

                    if self.minibatch:
                        # index varible that indicates where a specific GP is active
                        # for instance, GP_fem is only active in fem time series
                        idx_dict[name] = np.where(self.g['train']['groups_idx'][group].eval()==idx,1,0)

                        self.gp_dict[name] = pm.gp.Latent(mean_func=mu_func, cov_func=cov)
                        f_dict[name] = self.gp_dict[name].prior('f_%s' % name, X=self.X_mi, reparameterize=True, shape = self.minibatch[0])
                        f_flat[name] = f_dict[name].reshape((-1,1)) * idx_dict[name].reshape((1,-1))
                    else:
                        # index varible that indicates where a specific GP is active
                        # for instance, GP_fem is only active in fem time series
                        idx_dict[name] = np.where(self.g['train']['groups_idx'][group]==idx,1,0)

                        self.gp_dict[name] = pm.gp.Latent(mean_func=mu_func, cov_func=cov)
                        f_dict[name] = self.gp_dict[name].prior('f_%s' % name, X=self.X, reparameterize=True)
                        f_flat[name] = f_dict[name].reshape((-1,1)) * idx_dict[name].reshape((1,-1))

            self.f = sum(f_flat.values())
        
    def likelihood(self):
        self.generate_GPs()

        if self.minibatch:
            with self.model:
                self.y_pred = pm.Poisson('y_pred', 
                                mu=tt.exp(self.f + self.priors['a0'][self.series].reshape((1,-1))), 
                                observed=self.g['train']['data'], 
                                total_size=(self.g['train']['n'],self.g['train']['s']))
        else:
            with self.model:
                self.y_pred = pm.Poisson('y_pred', 
                                        mu=tt.exp(self.f + self.priors['a0'].reshape((1,-1))), 
                                        observed=self.g['train']['data'])


    def fit_map(self):
        self.likelihood()

        if self.minibatch:
            raise ValueError('Cannot use MAP with minibatch. Please call the `fit_vi` method.')

        with self.model:
            print('Fitting model...')
            self.mp = pm.find_MAP(maxeval=self.n_iterations)
            print('Sampling...')
            self.pred_samples_fit = pm.sample_posterior_predictive([self.mp], 
                                                    vars=[self.y_pred], 
                                                    samples=500)
            
    def fit_vi(self):
        self.likelihood()
        with self.model:
            print('Fitting model...')
            self.trace_vi = pm.fit(self.n_iterations)
            print('Sampling...')
            self.trace_vi_samples = self.trace_vi.sample()
            self.pred_samples_fit = pm.sample_posterior_predictive(self.trace_vi_samples,
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
                    f_flat_new[name] = f_new[name].reshape((-1,1)) * idx_dict_new[name].reshape((1,-1))

            f_ = sum(f_flat_new.values())
    
            y_pred_new = pm.Poisson("y_pred_new", 
                            mu=tt.exp(f_ + self.priors['a0'].reshape((1,-1))), 
                            shape=(n_new, self.g['predict']['s']))
            print('Sampling...')
            if self.trace_vi_samples:
                # Sampling using trace from VI
                self.pred_samples_predict = pm.sample_posterior_predictive(self.trace_vi_samples, 
                                              vars=[y_pred_new], 
                                              samples=500)
            else:
                # Sampling using points from MAP
                self.pred_samples_predict = pm.sample_posterior_predictive([self.mp], 
                              vars=[y_pred_new], 
                              samples=500)