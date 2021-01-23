import matplotlib.pyplot as plt
import numpy as np

def visualize_fit(groups, pred_samples_fit, n_series_to_show):
    assert n_series_to_show%2 == 0, "'n_series_to_show' must be an integer even number"
    fig, ax = plt.subplots(int(n_series_to_show/2), 2, figsize=(12,n_series_to_show*2))
    n = groups['train']['n']
    ax = np.ravel(ax)
    for i in range(n_series_to_show):
        ax[i].plot(np.arange(n), pred_samples_fit.T[i*n:i*n+n,:], alpha=0.003, color='orange', label='model fit')
        ax[i].plot(np.arange(n), groups['train']['data'][i*n:i*n+n], label='data')


def visualize_predict(groups, pred_samples_predict, n_series_to_show):
    assert n_series_to_show%2 == 0, "'n_series_to_show' must be an integer even number"
    fig, ax = plt.subplots(int(n_series_to_show/2), 2, figsize=(12,n_series_to_show*2))
    ax = np.ravel(ax)
    n = groups['train']['n']
    n_new = groups['predict']['n']

    for i in range(n_series_to_show):
        ax[i].fill_between(np.arange(n_new), 
                           np.percentile(pred_samples_predict.T[i*n_new:i*n_new+n_new,:], axis=1, q=2.5),
                          np.percentile(pred_samples_predict.T[i*n_new:i*n_new+n_new,:], axis=1, q=97.5),
                          label='95% CI', alpha=0.1)
        ax[i].plot(np.arange(n_new), np.median(pred_samples_predict['y_pred_new'].T[i*n_new:i*n_new+n_new,:], axis=1), color='tab:blue', alpha=0.7, label='median')
        ax[i].plot(np.arange(n_new), np.mean(pred_samples_predict['y_pred_new'].T[i*n_new:i*n_new+n_new,:], axis=1), color='b', label='mean')
        ax[i].set_ylim(0,max(groups['predict']['data'][i*n_new:i*n_new+n_new])*1.5)
        ax[i].plot(groups['train']['data'][i*n:i*n+n], color='darkorange', label='training data')
        ax[i].plot(np.arange(n, n_new), groups['predict']['data'][i*n_new+n:i*n_new+n_new], color='r', label='forecasting data')
        ax[i].legend()