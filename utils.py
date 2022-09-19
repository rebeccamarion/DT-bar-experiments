#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom functions.

@author: Rebecca Marion, University of Namur
"""

# Necessary imports
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import re
import itertools
import kuncheva
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from scipy import stats
import seaborn as sns
from sklearn.utils.validation import check_X_y



#### Functions for importing and exporting data ####

def load_data (file) : 
    
    with open(file, "br") as f:
         data = pkl.load(f)
    return data

def save_data (file, element) : 
    
    with open(file, "bw") as f:
         pkl.dump(element, f)

def get_file_name_contents (prefix, file_name, suffix = ".pkl") :
    
    search_string = prefix + "(.*)" + suffix
    contents = re.search(search_string, file_name).group(1)
    return contents

def get_method_type_from_file_name (prefix, file_name):
    
    if file_name.find("_") < 0 :
        search_string = prefix + "(.*).pkl"
    else : 
        search_string = prefix + "(.*)_"
        
    method_type = re.search(search_string, file_name).group(1)
    
    return method_type


def create_results_file_name (folder_path, method, hp_val = None):
    
    if hp_val:
        method_file_name = folder_path + method + "_" + "{}".format(hp_val) + ".pkl"
    else :
        method_file_name = folder_path + method + ".pkl"
        
    return method_file_name

#### General utils ####

def plot_results (plot_df, file, plot_order, hue_order, y_var = "stab", 
                  plot_type = "stability", plot_markers = False, y_label = "stability", font_scale = 2, legend_shift = 0.66,
                  n_shades = 3):
    
    n_methods = len(plot_df.loc[:, "method"].unique())
    sns.set(font_scale = font_scale, rc = {'legend.handlelength': 4.0})    
    sns.set_style(style = "whitegrid")  
    palette_init = sns.color_palette("Greys", n_colors = n_shades)[::-1]
    n_repeated_shades = round((n_methods - n_shades)/n_shades) + 1
    palette = sorted(sum(list(itertools.repeat(palette_init, n_repeated_shades)), []))[:n_methods]
    
    if plot_type == "stability":
        share_y = True
    else :
        share_y = False
    
    dashes_init = [(1, 0), (1, 1), (4, 2)]
    dashes = sum(list(itertools.repeat(dashes_init, n_repeated_shades)), [])[:n_methods]
    g = sns.relplot(data = plot_df, x = 'nb_features', 
                                    y = y_var, col = 'dataset',  hue = 'method', palette = palette,
                                    kind = 'line', linewidth = 4, col_wrap=4, 
                                    style = 'method', col_order = plot_order, hue_order = hue_order, style_order = hue_order, 
                                    facet_kws={'sharey': share_y, 'sharex': False, 'legend_out': True
                                               },
                                    dashes = dashes)
    g.set_titles(col_template = '{col_name}')
    
    sns.move_legend(g, "upper left", bbox_to_anchor=(legend_shift, .22))
    plt.subplots_adjust(hspace=0.5)
    
    if plot_type == "stability":
        # iterate through the axes
        grouped = plot_df.groupby(["dataset"])
        
        if plot_markers :
            axes = g.axes.flat
            for ax in axes:
                
                
                # extract the region from the title for use in selecting the index of fpt
                dataset = ax.get_title()
                # get x values for lines
                marker_vals_df = grouped.get_group(dataset)
                #ax.set_xlim(marker_vals_df.nb_features.min(), marker_vals_df.nb_features.max())
                marker_vals_df = marker_vals_df.loc[marker_vals_df.nb_features == marker_vals_df.nb_features_min_error, :]
                
                sns.scatterplot(data = marker_vals_df, x = 'nb_features', y = 'stab', ax = ax, marker = "s", hue = 'method', hue_order = hue_order, s = 300, legend = False, palette = palette)
    
    g._legend.set_title("")
    for line in g._legend.get_lines():
        line.set_linewidth(6.0)
        
    g.set_ylabels(y_label, clear_inner=True)
    g.set_xlabels("# of features", clear_inner=True)
    plt.subplots_adjust(hspace=0.25)
    
        
    g.savefig(file)
    

def find_keys_containing_str(dictionary, searchString):
    return [key for key,val in dictionary.items() if any(searchString in s for s in val)]

def enforce_2D(X):
    """Make sure X is a 2D matrix if it is not yet the case.
    
    Parameters
    ----------
    X: X (array)
    
    Returns
    X: X (array)
    
    """
    
    if len(X.shape) == 1:
        return X.reshape((-1, 1))
    else:
        return X    
    
def normalize(train, test):
    
    scaler = StandardScaler()
    scaler.fit(train)
    train_out = scaler.transform(train)
    test_out = scaler.transform(test)
    
    return train_out, test_out
    

#### Functions for feature selection/ranking ####

def DT_bar_criterion_multiple_K (X_train, y_train, feature_index, max_neighbors):
    
    X_feat = enforce_2D(X_train[:, feature_index])
    y = y_train.copy()
    sq_diff_vals = sq_diffs(X = X_feat, y = y, n_neighbors = max_neighbors)
    cum_sum = np.cumsum(sq_diff_vals, axis = 1)
    n_neighbors = np.arange(1, max_neighbors)
    cum_mean = cum_sum / n_neighbors
    crits = np.mean(cum_mean, axis = 0)
    
    return crits

def fit_KNN(X, n_neighbors = 2):
    knn_search = NearestNeighbors(n_neighbors=n_neighbors)
    knn_search.fit(X)
    
    return knn_search

def get_NNs(knn_search, X, n_neighbors = 2):
    
    dists, ids = knn_search.kneighbors(X, return_distance = True)
    ids = ids[:, 1:n_neighbors]
    dists = dists[:, 1:n_neighbors]
    
    return dists, ids


def delta_test(X, y):
    
    knn_search = fit_KNN(X, n_neighbors = 2)
    dists, ids = get_NNs(knn_search, X, n_neighbors = 2)
    y_pred = y[ids]
    y_tile = np.tile(y, (y_pred.shape[1], 1)).T
    sq_diff = (y_tile-y_pred)**2
    criterion = 0.5 * np.mean(sq_diff)
    
    return criterion

def sq_diffs(X, y, n_neighbors = 2):
    
    knn_search = fit_KNN(X, n_neighbors = n_neighbors)
    
    dists, ids = get_NNs(knn_search, X, n_neighbors = n_neighbors)
    y_pred = y[ids]
    y_tile = np.tile(y, (y_pred.shape[1], 1)).T
    sq_diff = (y_tile-y_pred)**2
    
    return sq_diff

def product_estimator(X, y):
    
    knn_search = fit_KNN(X, n_neighbors = 3)
    dists, ids = get_NNs(knn_search, X, n_neighbors = 3)
    y_pred = y[ids]
    y_tile = np.tile(y, (y_pred.shape[1], 1)).T
    diff = (y_tile-y_pred)
    criterion = np.mean(np.prod(diff, axis = 1))
    
    return criterion


def delta_test_bar(X, y, n_neighbors = 2):
    
    knn_search = fit_KNN(X, n_neighbors = n_neighbors)
    
    # gamma_k and delta_k 
    dists, ids = get_NNs(knn_search, X, n_neighbors = n_neighbors)
    y_pred = y[ids]
    y_tile = np.tile(y, (y_pred.shape[1], 1)).T
    sq_diff = (y_tile-y_pred)**2
    criterion = 0.5 * np.mean(np.mean(sq_diff, axis = 1))
    
    return criterion


def gamma_test(X, y, n_neighbors = 2):
    
    knn_search = fit_KNN(X, n_neighbors = n_neighbors)
    
    # gamma_k and delta_k 
    dists, ids = get_NNs(knn_search, X, n_neighbors = n_neighbors)
    y_pred = y[ids]
    y_tile = np.tile(y, (y_pred.shape[1], 1)).T
    gamma_k = 0.5 * np.mean(((y_tile-y_pred)**2), axis = 0)
    delta_k = np.mean(dists, axis = 0)
    slope, intercept, r, p, se = stats.linregress(x = delta_k, y = gamma_k)
    
    return intercept

def delta_test_boot(X, y, n_iterations = 1000):
    
    criterion = 0
    for i in range(n_iterations):
        X_bs, y_bs = resample(X, y, replace=True)
        criterion += delta_test(X_bs, y_bs)/n_iterations
    
    return criterion

def FS_scorer(X, y, **kwargs):
    
    X = enforce_2D(X)
    
    if kwargs["type"] == "delta-test":
       
        criterion = delta_test(X, y)
        
    if kwargs["type"] == "delta-boot":
       
        criterion = delta_test_boot(X, y)
    
    if kwargs["type"] == "gamma-test":
        
        criterion = gamma_test(X, y, n_neighbors = kwargs["hp"] + 1)
        
    if kwargs["type"] == "delta-test-bar":
        
        criterion = delta_test_bar(X, y, n_neighbors = kwargs["hp"] + 1)
        
    if kwargs["type"] == "product-estimator":
        
        criterion = product_estimator(X, y)
        
    return criterion

#### Feature ranking function ####

def feature_ranking (X_train, y_train, selection_criterion = FS_scorer, 
                     selection_strategy = "min", d_max = None, scorer_hp = {}):
    
    X, y = check_X_y(X_train, y_train)
    X = enforce_2D(X)
    d = X.shape[1]
    
    # Calculate FS scores for each feature
    X_list = X.T.tolist()
    scores_arr = np.array([selection_criterion(X = np.array(x), y = y, **scorer_hp) for x in X_list])
    
    if selection_strategy == "min":
        sign_val = 1
        
    else : 
        sign_val = -1
        
    solution_path = np.argsort(sign_val*scores_arr)
    scores = scores_arr[solution_path]
    
    if d_max > d:
        d_keep = d
    else:
        d_keep = d_max
    
     
    return solution_path[:d_keep], scores[:d_keep]


#### Functions for evaluating performance ####

def stability_scores(selection_paths_arr, d):
    
    d_keep = selection_paths_arr.shape[1]
    stab = np.zeros((d_keep, 1))
    
    for nb_features in range(stab.shape[0]):
        sub_path = selection_paths_arr[:, 0:(nb_features+1)]
        feature_list = sub_path.tolist()
        if nb_features < d - 1 :
            stab[nb_features, 0] = kuncheva.get_kuncheva_index(subsets = feature_list, n = d)
        else :
            stab[nb_features, 0] = float("nan")
            
    return stab
            
    


def prediction_score(X_train, t_train, X_test, t_test, task, selection_path, predictor, predictor_hp={}):
    """Compute the prediction score along a feature selection path.
    
    Parameters
    ----------
    X_train: features for train (np.array, shape (n, d))
    t_train: target for train (np.array, shape (n, ))
    X_test: features for test (np.array, shape (n, d))
    t_test: target for test (np.array, shape (n, ))
    task: 'classification' or 'regression' (str)
    selection_path: selected feature along the search (np.array, shape (d_max, ))
    predictor: model used for prediction (class)
    predictor_hp: model hyperparameters (dict, opt)
    
    Returns
    -------
    scores: scores for each feature subset size (np.array, size (d+1, ))
    
    Notes
    -----
    scores[0] is the variance of the target in regression and the prior probability
    (i.e., the proportion thereof) of the majority class in classification.
    
    This function was written by Benoît Frénay and Rebecca Marion
    
    """
    
    if task == 'classification':
        scores = [np.max(np.unique(t_test, return_counts=True)[1])/t_test.size]
    else:
        scores = [np.mean((t_test-np.mean(t_test))**2)]
    
    d_max = selection_path.size
    
    for nb_features in range(1, d_max+1):
        X_train_red = enforce_2D(X_train[:, selection_path[:nb_features]])
        X_test_red = enforce_2D(X_test[:, selection_path[:nb_features]])
        
        model = predictor(**predictor_hp)
        model.fit(X_train_red, t_train)
        y_test = model.predict(X_test_red)
        
        if task == 'classification':
            scores.append(np.mean(y_test==t_test))
        else:
            scores.append(np.mean((t_test-y_test)**2))
    
    return scores

def calc_rel_error (baseline_error_arr, error_arr):
    """Compute relative error.
    
    Parameters
    ----------
    baseline_error_arr: 1d np.array of n repetitions 
    error_arr: 2d np.array (n x d) of n reptitions for d features
    
    Returns
    -------
    rel_err: error_arr divided by baseline_error_arr
    
    """
    rel_error = (error_arr / baseline_error_arr.reshape(-1, 1))
    
    return rel_error



