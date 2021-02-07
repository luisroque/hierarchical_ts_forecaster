import pymc3 as pm
import theano.tensor as tt
import theano
import numpy as np
from libs.pre_processing import generate_groups_data_matrix_minibatch, generate_groups_data_matrix, data_transform

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
5. Option to use MAP or VI to estimate the parameter values (VI is advised 
    using normal likelihood)
6. Possibility to use Minibatch that ensures scalability of the model
7. Option to choose Poisson or Normal likelihood (for the normal case,
    it is assumed that the data is standardized)
'''

class LogLinear(pm.gp.mean.Mean):
    # Log linear mean function to get diminishing returns on the GP mean functions
    # -> adding 1 to avoid inf when X = 0
    def __init__(self, b, a=0):
        self.a = a
        self.b = b

    def __call__(self, X):
        return tt.squeeze(tt.dot(tt.log(X+1), self.b) + self.a)


class Linear(pm.gp.mean.Mean):
    # This linear function with a log-link function results in an 
    # exponential mean 
    def __init__(self, b, a=0):
        self.a = a
        self.b = b

    def __call__(self, X):
        return tt.squeeze(tt.dot(X, self.b) + self.a)


class PiecewiseLinearChangepoints(pm.gp.mean.Mean):
    # Piecewise linear function to be used as the mean function of the GPs
    def __init__(self, 
                 k, 
                 m,
                 b,
                 intercept, 
                 changepoints,
                 groups):
        self.k = k
        self.m = m
        self.b = b
        self.a = intercept
        self.g = groups
        self.changepoints = changepoints

    def create_changepoints(self, X, changepoints):
        return (0.5 * (1.0 + tt.sgn(tt.tile(X.reshape((-1,1)), (1,len(changepoints))) - changepoints)))

    def __call__(self, X):
        size_r = X.shape[0]

        X = theano.shared(X)
            
        A = self.create_changepoints(X, self.changepoints)

        piecewise = (self.k + tt.dot(A, self.b.reshape((-1,1))))*tt.squeeze(X) + (self.m + tt.dot(A, (-self.changepoints * self.b).reshape((-1,1))))
        
        return (piecewise + self.a).reshape((-1,))


class OutPiecewiseLinearChangepoints():
    # Piecewise linear function applied outside of the GPs and added in the likelihood parameter
    def __init__(self, 
                 k, 
                 m,
                 b,
                 changepoints,
                 groups):
        self.k = k
        self.m = m
        self.b = b
        self.g = groups
        self.changepoints = changepoints

    def create_changepoints(self, X, changepoints):
        return (0.5 * (1.0 + tt.sgn(tt.tile(X.reshape((-1,1)), (1,len(changepoints))) - changepoints)))

    def build(self, X):
        size_r = X.shape[0]

        X = theano.shared(X)
            
        A = self.create_changepoints(X, self.changepoints)

        piecewise = (self.k.reshape((1, -1)) + tt.dot(A, self.b))*tt.tile(X, (1, self.g['train']['s'])) + (self.m.reshape((1,-1)) + tt.dot(A, (-self.changepoints.reshape((-1,1)) * self.b)))
        
        return piecewise


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
    likelihood: list
            ['normal', 'poisson'] -> possibility to choose to use a normal likelihood (it is assumed that 
            the data is standardized) or a poisson likelihood
    piecewise_out: bool
            possibility to define a piecewise linear function outside of the GPs definition. It will be summed 
            to the GPs output in the likelihood
    """
    def __init__(
        self,
        groups_data,
        levels=None,
        changepoints=None,
        n_iterations=10000,
        minibatch=None,
        log_lin_mean=None,
        likelihood='normal',
        piecewise_out = False,
        #kernel_lin_mean=None,
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
        self.n_iterations = n_iterations
        self.trace_vi = None
        self.pred_samples_fit = None
        self.trace_vi_samples = None
        self.likelihood = likelihood
        self.piecewise_out = piecewise_out

        if changepoints:
            self.changepoints = np.linspace(0, self.g['train']['n'], changepoints+2)[1:-1]
        else:
            self.changepoints = np.array(())

        if levels:
            self.levels = levels
        else:
            self.levels = list(self.g['train']['groups_names'].keys())

        self.minibatch = minibatch
        self.log_lin_mean = log_lin_mean

        # transform the data to matrix form
        self.g = generate_groups_data_matrix(self.g)
        self.dt = data_transform(self.g)

        if self.likelihood == 'normal':
            # if likelihood is normal standardize data
            self.g = self.dt.std_transf_train()

        # to use minibatch transform the arrays to minibatch tensors
        if self.minibatch:
            self.g, self.X_mi = generate_groups_data_matrix_minibatch(self.g, self.minibatch[0], self.minibatch[1])

        self.X = np.arange(self.g['train']['n']).reshape(-1,1)

    def generate_priors(self):
        """Set up the priors for the model."""
        with self.model:

            if self.minibatch:
                self.series = self.g['train']['n_series_idx'].eval()
            else:
                self.series = self.g['train']['n_series_idx']
            
            self.series_full = self.g['train']['n_series_idx_full']

            if self.likelihood=='poisson':
                # In the case of the normal likelihood it is assumed that the data is standardized
                self.priors["a0"] = pm.Normal(
                    "a0", 
                    mu=tt.log(np.mean(self.g['train']['full_data'][:,self.series_full], axis=0)), 
                    sd=0.2, 
                    shape = self.g['train']['s'])
            else:
                # In the case of a normal likelihood we need to define sigma
                self.priors['sigma'] = pm.HalfNormal(
                    'sigma',
                    2.,
                    shape = self.g['train']['s'])

            if self.piecewise_out:
                if self.likelihood == 'poisson':
                    self.priors["k"] = pm.Normal(
                        'k',
                        0.0,
                        0.01,
                        shape = self.g['train']['s'])
                    self.priors["m"] = pm.Normal(
                        'm', 
                        0.0,
                        0.01,
                        shape = self.g['train']['s'])

                    self.priors["b"] = pm.Normal(
                        'b', 
                        0.,
                        0.01,
                        shape = (self.changepoints.shape[0], self.g['train']['s']))
                if self.likelihood == 'poisson':
                    self.priors["k"] = pm.Normal(
                        'k',
                        0.0,
                        1,
                        shape = self.g['train']['s'])
                    self.priors["m"] = pm.Normal(
                        'm', 
                        0.0,
                        1,
                        shape = self.g['train']['s'])

                    self.priors["b"] = pm.Normal(
                        'b', 
                        0.,
                        1,
                        shape = (self.changepoints.shape[0], self.g['train']['s']))

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
                    self.g['train']['n'], 
                    shape = self.g['train']['groups_n'][group])
                self.priors["l_p_%s" %group] = pm.InverseGamma(
                    'l_p_%s' %group, 
                    4, 
                    self.g['train']['n'], 
                    shape = self.g['train']['groups_n'][group])
                self.priors["eta_t_%s" %group] = pm.HalfNormal(
                    'eta_t_%s' %group, 
                    0.1, # 0.02
                    shape = self.g['train']['groups_n'][group])
                self.priors["eta_p_%s" %group] = pm.HalfNormal(
                    'eta_p_%s' %group, 
                    0.5, #0.05
                    shape = self.g['train']['groups_n'][group])
                self.priors["sigma_%s" %group] = pm.HalfNormal(
                    'sigma_%s' %group, 
                    0.005, #0.001
                    shape = self.g['train']['groups_n'][group])

                if self.log_lin_mean:
                    self.priors["hy_b_%s" %group] = pm.Normal(
                        "hy_b_%s" %group, 
                        mu=0.0, 
                        sd=0.01)
                    self.priors["b_%s" %group] = pm.Normal(
                        'b_%s' %group, 
                        self.priors["hy_b_%s" %group],
                        0.005,
                        shape = self.g['train']['groups_n'][group])
                elif np.any(self.changepoints):
                    if self.likelihood == 'poisson':
                        # Priors for hyperparamters
                        self.priors["hy_b_%s" %group] = pm.Normal(
                            "hy_b_%s" %group, 
                            mu=0.0, 
                            sd=0.02,
                            shape=self.g['train']['groups_n'][group])
                        # priors for the group effects
                        self.priors["b_%s" %group] = pm.Normal(
                            'b_%s' %group, 
                            self.priors["hy_b_%s" %group],
                            0.005,
                            shape = (self.changepoints.shape[0], self.g['train']['groups_n'][group]))
                        self.priors["k_%s" %group] = pm.Normal(
                            'k_%s' %group, 
                            0.0,
                            0.005,
                            shape = self.g['train']['groups_n'][group])
                        self.priors["m_%s" %group] = pm.Normal(
                            'm_%s' %group, 
                            0.0,
                            0.005,
                            shape = self.g['train']['groups_n'][group])
                    else:
                        # Parameters for the piecewise linear function defined as GPs mean functions
                        # with a normal likelihood -> wider intervals as we don't have the log-link
                        # function
                        self.priors["hy_b_%s" %group] = pm.Normal(
                            "hy_b_%s" %group, 
                            mu=0.0, 
                            sd=1.,
                            shape=self.g['train']['groups_n'][group])
                        # priors for the group effects
                        # we want a partial pooling effect, so reduce the sd of the several parameters,
                        # while defining a wider hyperparameter
                        self.priors["b_%s" %group] = pm.Normal(
                            'b_%s' %group, 
                            self.priors["hy_b_%s" %group],
                            0.1,
                            shape = (self.changepoints.shape[0], self.g['train']['groups_n'][group]))
                        self.priors["k_%s" %group] = pm.Normal(
                            'k_%s' %group, 
                            0.0,
                            1.,
                            shape = self.g['train']['groups_n'][group])
                        self.priors["m_%s" %group] = pm.Normal(
                            'm_%s' %group, 
                            0.0,
                            1.,
                            shape = self.g['train']['groups_n'][group])
                else:
                    pass
                    # Using linear kernel to model the mean of the GP (exponential)
                    # if self.likelihood=='poisson':
                    #     self.priors["c_%s" %group] = pm.Normal(
                    #         'c_%s' %group, 
                    #         0, 
                    #         0.05, 
                    #         shape = self.g['train']['groups_n'][group])
                    #     self.priors["eta_l_%s" %group] = pm.HalfNormal(
                    #         'eta_l_%s' %group, 
                    #         0.01,
                    #         shape = self.g['train']['groups_n'][group])
                    # else:
                    #     self.priors["c_%s" %group] = pm.Normal(
                    #         'c_%s' %group, 
                    #         0, 
                    #         1, 
                    #         shape = self.g['train']['groups_n'][group])
                    #     self.priors["eta_l_%s" %group] = pm.HalfNormal(
                    #         'eta_l_%s' %group, 
                    #         1,
                    #         shape = self.g['train']['groups_n'][group])


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

                    elif np.any(self.changepoints):
                        mu_func = PiecewiseLinearChangepoints(intercept = np.zeros(1),
                                                              b = self.priors["b_%s" %group][:, idx],
                                                              changepoints = self.changepoints,
                                                              k = self.priors["k_%s" %group][idx],
                                                              m = self.priors["m_%s" %group][idx],
                                                              groups = self.g)

                        cov = (self.priors["eta_t_%s" %group][idx]**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=self.priors["l_t_%s" %group][idx])                                
                                + self.priors["eta_p_%s" %group][idx]**2 * pm.gp.cov.Periodic(1, period=self.priors["period"], ls=self.priors["l_p_%s" %group][idx]) 
                                + pm.gp.cov.WhiteNoise(self.priors["sigma_%s" %group][idx]))

                    else:
                        mu_func = pm.gp.mean.Zero()
                        # cov function for the GP with specific parameters per group
                        cov = (self.priors["eta_t_%s" %group][idx]**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=self.priors["l_t_%s" %group][idx])
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
                        idx_dict[name] = np.where(self.g['train']['groups_idx'][group]==idx,1,0)

                        self.gp_dict[name] = pm.gp.Latent(mean_func=mu_func, cov_func=cov)
                        f_dict[name] = self.gp_dict[name].prior('f_%s' % name, X=self.X, reparameterize=True)
                        f_flat[name] = f_dict[name].reshape((-1,1)) * idx_dict[name].reshape((1,-1))

            self.f = sum(f_flat.values())
        
    def likelihood_fn(self):
        # Avoid generating priors if already created
        if not self.priors:
            self.generate_GPs()

        if self.minibatch:
            # Using minibatch
            with self.model:
                if self.likelihood=='poisson':
                    # defining a poisson likelihood
                    self.y_pred = pm.Poisson('y_pred', 
                                mu=tt.exp(self.f + self.priors['a0'][self.series].reshape((1,-1))), 
                                observed=self.g['train']['data'], 
                                total_size=(self.g['train']['n'],self.g['train']['s']))
                else:
                    # defining a normal likelihood with minibatch
                    self.y_pred = pm.Normal('y_pred', 
                                mu=self.f,
                                sd=self.priors['sigma'],
                                observed=self.g['train']['data'], 
                                total_size=(self.g['train']['n'],self.g['train']['s']))

        else:
            # Not using minibatch
            with self.model:
                if self.likelihood=='poisson':
                    # defining a poisson likelihood
                    if self.piecewise_out:
                        # poisson likelihood with a piecewise function summed
                        piece = OutPiecewiseLinearChangepoints(k = self.priors["k"],
                                                            m = self.priors["m"],
                                                            b = self.priors['b'],
                                                            changepoints = self.changepoints,
                                                            groups = self.g).build(self.X)
                        self.y_pred = pm.Poisson('y_pred', 
                            mu=tt.exp(self.f + self.priors['a0'].reshape((1,-1)) + piece), 
                            observed=self.g['train']['data'])
                    else:
                        # not using a piecewise linear function outside of the GPs
                        self.y_pred = pm.Poisson('y_pred', 
                                mu=tt.exp(self.f + self.priors['a0'][self.series].reshape((1,-1))), 
                                observed=self.g['train']['data'])
                else:
                    if self.piecewise_out:
                        # normal likelihood with a piecewise function summed
                        piece = OutPiecewiseLinearChangepoints(k = self.priors["k"],
                                                            m = self.priors["m"],
                                                            b = self.priors['b'],
                                                            changepoints = self.changepoints,
                                                            groups = self.g).build(self.X)
                        self.y_pred = pm.Normal('y_pred', 
                            mu=self.f + piece, 
                            sd=self.priors['sigma'],
                            observed=self.g['train']['data'])
                    else:
                        # normal likelihood without minibatch
                        self.y_pred = pm.Normal('y_pred', 
                                mu=self.f,
                                sd=self.priors['sigma'],
                                observed=self.g['train']['data'])


    def prior_predictive_checks(self):
        # Avoid generating priors if already created
        if not self.priors:
            self.generate_GPs()

        with self.model:
            if self.likelihood=='poisson':
                # defining a poisson likelihood
                if self.piecewise_out:
                    # poisson likelihood with a piecewise function summed
                    piece = OutPiecewiseLinearChangepoints(k = self.priors["k"],
                                                        m = self.priors["m"],
                                                        b = self.priors['b'],
                                                        changepoints = self.changepoints,
                                                        groups = self.g).build(self.X)
                    # this value is the value used as the lambda for the poisson likelihood
                    self.prior_like = pm.Poisson('prior_like', 
                                        mu=tt.exp(self.f + self.priors['a0'].reshape((1,-1)) + piece),
                                        shape=(self.g['train']['data'].shape[0], self.g['train']['data'].shape[1]))
                else:
                    # not using a piecewise linear function outside of the GPs
                    self.prior_like = pm.Poisson('prior_like', 
                                        mu=tt.exp(self.f + self.priors['a0'].reshape((1,-1)) + piece),
                                        shape=(self.g['train']['data'].shape[0], self.g['train']['data'].shape[1]))
            else:
                if self.piecewise_out:
                    # normal likelihood with a piecewise function summed
                    piece = OutPiecewiseLinearChangepoints(k = self.priors["k"],
                                                        m = self.priors["m"],
                                                        b = self.priors['b'],
                                                        changepoints = self.changepoints,
                                                        groups = self.g).build(self.X)
                    self.prior_like = pm.Normal('prior_like', 
                        mu=self.f + piece, 
                        sd=self.priors['sigma'],
                        shape=(self.g['train']['data'].shape[0], self.g['train']['data'].shape[1]))
                else:
                    # normal likelihood
                    self.prior_like = pm.Normal('prior_like', 
                        mu=self.f, 
                        sd=self.priors['sigma'],
                        shape=(self.g['train']['data'].shape[0], self.g['train']['data'].shape[1]))

            self.prior_checks = pm.sample_prior_predictive(200)

    def fit_map(self):
        self.likelihood_fn()

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
        self.likelihood_fn()
        with self.model:
            print('Fitting model...')
            self.trace_vi = pm.fit(self.n_iterations)
            print('Sampling...')
            self.trace_vi_samples = self.trace_vi.sample()
            self.pred_samples_fit = pm.sample_posterior_predictive(self.trace_vi_samples,
                                                                   vars=[self.y_pred]
                                                                   samples=500)

        # backtransform the sampling of the fit for the original scale
        if self.likelihood == 'normal':
            self.pred_samples_fit = self.dt.inv_transf_general(self.pred_samples_fit['y_pred'])


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

            if self.likelihood=='poisson':
                if self.piecewise_out:
                    piece = OutPiecewiseLinearChangepoints(k = self.priors['k'],
                                                m = self.priors['m'],
                                                b = self.priors['b'],
                                                changepoints = self.changepoints,
                                                groups = self.g).build(X_new)

                    y_pred_new = pm.Poisson("y_pred_new", 
                                    mu=tt.exp(f_ + self.priors['a0'].reshape((1,-1)) + piece), 
                                    shape=(n_new, self.g['predict']['s']))
                else:
                    y_pred_new = pm.Poisson("y_pred_new", 
                                    mu=tt.exp(f_ + self.priors['a0'].reshape((1,-1))), 
                                    shape=(n_new, self.g['predict']['s']))
            else:
                if self.piecewise_out:
                    piece = OutPiecewiseLinearChangepoints(k = self.priors['k'],
                                                m = self.priors['m'],
                                                b = self.priors['b'],
                                                changepoints = self.changepoints,
                                                groups = self.g).build(X_new)

                    y_pred_new = pm.Normal("y_pred_new", 
                                    mu=f_ + piece,
                                    sd=self.priors['sigma'],
                                    shape=(n_new, self.g['predict']['s']))
                else:
                    y_pred_new = pm.Normal("y_pred_new", 
                                    mu=f_, 
                                    sd=self.priors['sigma'],
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

        # backtransform the data and predictions for the original scale
        if self.likelihood == 'normal':
            self.g = self.dt.inv_transf_train()
            self.pred_samples_predict = self.dt.inv_transf_general(self.pred_samples_predict['y_pred_new'])