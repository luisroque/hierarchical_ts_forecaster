import matplotlib.pyplot as plt
import numpy as np

def visualize_fit(groups, pred_samples_fit, n_series_to_show):
    assert n_series_to_show%2 == 0, "'n_series_to_show' must be an integer even number"
    fig, ax = plt.subplots(int(n_series_to_show/2), 2, figsize=(12,n_series_to_show*2))
    n = groups['train']['n']
    ax = np.ravel(ax)
    for i in range(n_series_to_show):
        ax[i].plot(np.arange(n), pred_samples_fit.T[i*n:i*n+n,:], alpha=0.003, color='orange', label='model fit')
        ax[i].plot(np.arange(n), groups['train']['full_data'][i*n:i*n+n], label='data')


def visualize_predict(groups, pred_samples_predict, n_series_to_show, levels=[0,1,2]):
    """
    Parameters
    ----------
    levels: list
                0 -> total, 1 -> groups, 2 -> bottom
                Default to [0,1,2]
    """

    assert n_series_to_show%2 == 0, "'n_series_to_show' must be an integer even number"
    fig, ax = plt.subplots(int(n_series_to_show/2), 2, figsize=(12,n_series_to_show*2))
    ax = np.ravel(ax)
    n = groups['train']['n']
    n_new = groups['predict']['n']

    # Total
    if 0 in levels:
        y_f = groups['predict']['data'].reshape(groups['predict']['s'], groups['predict']['n']).T
        y_all_g = np.sum(y_f, axis=1).reshape(-1,1)

        ax[0].fill_between(np.arange(n_new), 
                        np.percentile(np.sum(pred_samples_predict, axis=2).T, axis=1, q=2.5),
                        np.percentile(np.sum(pred_samples_predict, axis=2).T, axis=1, q=97.5),
                        label='95% CI', alpha=0.1)
        ax[0].plot(np.arange(n_new), 
                np.median(np.sum(pred_samples_predict, axis=2).T, axis=1),
                color='tab:blue', alpha=0.7, label='median')
        ax[0].plot(np.arange(n_new), 
                np.mean(np.sum(pred_samples_predict, axis=2).T, axis=1),
                color='b', label='mean')
        ax[0].set_ylim(0,max(np.sum(groups['predict']['data'].reshape(groups['predict']['s'], groups['predict']['n']).T, axis=1)*1.5))
        ax[0].plot(np.sum(groups['train']['full_data'], axis=1), 
                color='darkorange', label='training data')
        ax[0].plot(np.arange(n, n_new),
                np.sum(groups['predict']['data'].reshape(groups['predict']['s'], groups['predict']['n']).T, axis=1)[n:],
                color='r', label='forecasting data')
        ax[0].legend()
        ax[0].set_title('Total: sum of all series')
    elif 1 in levels:
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
    elif 2 in levels:
        for i in range(n_series_to_show-1):
            ax[i+1].fill_between(np.arange(n_new), 
                            np.percentile(pred_samples_predict[:,:,i], axis=0, q=2.5),
                            np.percentile(pred_samples_predict[:,:,i], axis=0, q=97.5),
                            label='95% CI', alpha=0.1)
            ax[i+1].plot(np.arange(n_new), np.median(pred_samples_predict[:,:,i], axis=0), color='tab:blue', alpha=0.7, label='median')
            ax[i+1].plot(np.arange(n_new), np.mean(pred_samples_predict[:,:,i], axis=0), color='b', label='mean')
            ax[i+1].set_ylim(0,max(groups['predict']['data'][i*n_new:i*n_new+n_new])*1.5)
            ax[i+1].plot(groups['train']['full_data'][:,i], color='darkorange', label='training data')
            ax[i+1].plot(np.arange(n, n_new), groups['predict']['data'][i*n_new+n:i*n_new+n_new], color='r', label='forecasting data')
            ax[i+1].legend()
            ax[i+1].set_title(f'Series {i}',)