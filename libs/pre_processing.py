import pandas as pd
import numpy as np
import pymc3 as pm

def generate_groups_data_flat(y,
                         groups_input,
                         seasonality,
                         h):
    '''
    It works for two kinds of structures:
        1) The name of the columns have specific length for specific groups
        2) There is a multiIndex column structure for each group
    '''
    groups = {}
    for i in ['train', 'predict']:
        if i == 'train':  
            y_ = y.iloc[:-h,:]
        else:
            y_ = y
        groups[i] = {}
        groups[i]['groups_idx'] = {}
        groups[i]['groups_n'] = {}
        groups[i]['groups_names'] = {}

        groups[i]['n'] = y_.shape[0]
        groups[i]['s'] = y_.shape[1]
        n_series = y.columns.unique().shape[0]

        # Test if we are receiving format 1) or 2)
        if len(next(iter(groups_input.values()))) == 1:
            for g in groups_input:
                group_idx = pd.get_dummies(
                        [i[groups_input[g][0]] for i in y_]
                    ).values.argmax(1)
                groups[i]['groups_idx'][g] = np.tile(group_idx, (groups[i]['n'],1)).flatten('F')
                groups[i]['groups_n'][g] = np.unique(group_idx).shape[0]
                group_names = [i[groups_input[g][0]] for i in y_]
                groups[i]['groups_names'][g] = np.unique(group_names)
        else:
            for g in groups_input:
                group_idx = pd.get_dummies(
                        [i[groups_input[g][0]:groups_input[g][1]] for i in y_]
                    ).values.argmax(1)
                groups[i]['groups_idx'][g] = np.tile(group_idx, (groups[i]['n'],1)).flatten('F')
                groups[i]['groups_n'][g] = np.unique(group_idx).shape[0]
                group_names = [i[groups_input[g][0]:groups_input[g][1]] for i in y_]
                groups[i]['groups_names'][g] = np.unique(group_names)

        groups[i]['n_series_idx'] = np.tile(np.arange(groups[i]['s']), (groups[i]['n'],1)).flatten('F')
        groups[i]['n_series'] = np.arange(groups[i]['s'])

        groups[i]['g_number'] = len(groups_input)

        groups[i]['data'] = y_.values.T.ravel()

    groups['seasonality'] = seasonality
    groups['h'] = h

    print("Number of groups: " + str(len(groups['train']['groups_names'])))
    for name,group in groups['train']['groups_names'].items():
        print('\t' + str(name) + ': ' + str(len(group)))
    print('Total number of series: ' + str(groups['train']['s']))
    print('Number of points per series for train: ' + str(groups['train']['n']))
    print('Total number of points: ' + str(groups['predict']['n']))
    print('Seasonality: ' + str(seasonality))
    print('Forecast horizon: ' + str(h))

    return groups


def generate_groups_data_matrix(groups):

    for group in groups['train']['groups_idx'].keys():
        groups['train']['groups_idx'][group] = groups['train']['groups_idx'][group].reshape(groups['train']['s'], groups['train']['n']).T[0,:]
        groups['predict']['groups_idx'][group] = groups['predict']['groups_idx'][group].reshape(groups['predict']['s'], groups['predict']['n']).T[0,:]

    groups['train']['full_data'] = groups['train']['data'].reshape(groups['train']['s'], groups['train']['n']).T
    groups['train']['data'] = groups['train']['data'].reshape(groups['train']['s'], groups['train']['n']).T

    groups['train']['n_series_idx_full'] = groups['train']['n_series_idx'].reshape(groups['train']['s'], groups['train']['n']).T[0,:]
    groups['train']['n_series_idx'] = groups['train']['n_series_idx'].reshape(groups['train']['s'], groups['train']['n']).T[0,:]

    groups['predict']['n_series_idx'] = groups['predict']['n_series_idx'].reshape(groups['predict']['s'], groups['predict']['n']).T[0,:]

    return groups


def generate_groups_data_matrix_minibatch(groups,
                                          n_mi,
                                          s_mi):

    groups = generate_groups_data_matrix(groups)

    groups['train']['n_series_idx'] = pm.Minibatch(groups['train']['n_series_idx'], s_mi)
    groups['train']['groups_idx']['state'] = pm.Minibatch(groups['train']['groups_idx']['state'], s_mi)
    groups['train']['groups_idx']['gender'] =pm.Minibatch(groups['train']['groups_idx']['gender'], s_mi)
    groups['train']['groups_idx']['legal'] = pm.Minibatch(groups['train']['groups_idx']['legal'], s_mi)
    groups['train']['data'] = pm.Minibatch(groups['train']['data'], ((n_mi,s_mi)))
    
    X = np.arange(groups['train']['n']).reshape(-1,1)
    X_mi = pm.Minibatch(X.ravel(), n_mi).reshape((-1,1))

    return groups, X_mi