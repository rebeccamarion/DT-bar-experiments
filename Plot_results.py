#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code plots the mean relative MSEP and stability scores with respect to
the number of features selected for each criterion. For methods with 
hyperparameters, these are chosen using the method that optimizes the criterion
hp_selection for all datasets combined.

@author: Rebecca Marion, University of Namur
"""

import utils

import numpy as np
import pandas as pd
import glob


# Hp selection method
hp_selection = ["min_mean_error"]

# Figure file path
figure_path = "Figures/"

# Name mapping for plots
method_rename = {"delta-boot": "delta bootstrap",
                 "delta-test-bar": "DT-bar",
                 "delta-test": "delta test",
                 "gamma-test": "gamma test",
                 "product-estimator": "product estimator"}

# Results
results_path = "Results/"
results_path_ast = "Results/" + "*"
result_file_names = sorted(glob.glob(results_path_ast))
method_names = [utils.get_file_name_contents(results_path, file_name) for file_name in result_file_names]
method_types = [utils.get_method_type_from_file_name(results_path, file_name) for file_name in result_file_names]

# Order for plotting
sort_by = "dim_ratio"

# Data info
data_stats = utils.load_data(file = "Data_stats/data_stats.pkl")
data_names = data_stats.sort_values(sort_by).dataset

# Put data together in one DataFrame
results_df = pd.DataFrame(columns = ["dataset", "method", "hp", "stab", "nb_features", "mean_error", "sd_error", "nb_runs", "nb_features_min_error"])
for method_index in range(len(method_names)):
    method_name = method_names[method_index]
    method_type = method_types[method_index]
    method_renamed = method_rename[method_type]
    results = utils.load_data(file = "Results/" + method_name + ".pkl")
    
    for data_name in data_names:
        if data_name in results.keys():
            
            pred_error = results[data_name]["rel_pred_scores"]
            stab = results[data_name]["stab"][:, 0]
            hp = results["hp"]
            
            mean_pred_error = np.mean(pred_error, axis = 0)
            sd_pred_error = np.std(pred_error, axis = 0)
            nb_features_min_error = np.argmin(mean_pred_error) + 1
            d_max = pred_error.shape[1] 
            nb_runs = pred_error.shape[0]
            
            temp = pd.DataFrame(index = range(d_max), columns = results_df.columns)
            temp = temp.assign(dataset = data_name,
                               method = method_renamed,
                               hp = hp,
                               stab = stab,
                               nb_features = range(1, len(temp) + 1),
                               mean_error = mean_pred_error,
                               sd_error = sd_pred_error,
                               nb_runs = nb_runs,
                               nb_features_min_error = nb_features_min_error)

            results_df = pd.concat([results_df, temp], ignore_index=True)
        
# Prepare a DataFrame for plotting
# Keep only the hyperparameters selected with the hp_selection criterion 
grouped = results_df.groupby("method")
groups = list(grouped.groups.keys())

group_index = 2
plot_list = list()


chosen_hp = dict()
for group_index in range(len(groups)):
    
    # Grouped by hp values
    grouped_sub = grouped.get_group(groups[group_index]).groupby("hp")
    
    # If the method has hyperparameters
    if len(grouped_sub.groups.keys()) > 1:
        
        stat_list = list()
        for group_name, group_df in grouped_sub :
            
            # Get mean over feature set sizes for each dataset, then get mean
            mean_stab = group_df.groupby("dataset")["stab"].mean().mean()
            mean_err = group_df.groupby("dataset")["mean_error"].mean().mean()
            
            method = group_df.method.unique()
            hp = group_df.hp.unique()
            
            stats = pd.DataFrame(columns = ["stab_mean", "err_mean", "method", "hp"])
            stats = stats.assign(stab_mean = [mean_stab],
                                 err_mean = [mean_err],
                                 method = method,
                                 hp = hp)            
           
            stat_list.append(stats)
        
        stats_df = pd.concat(stat_list).reset_index(drop = True)
            
        if "max_mean_stab" in set(hp_selection):
            
            best_hp = stats_df.loc[stats_df.stab_mean.idxmax(), "hp"]
            
        if "min_mean_error" in set(hp_selection):
            
            best_hp = stats_df.loc[stats_df.err_mean.idxmin(), "hp"]
            
            
        chosen_hp[stats_df.method.unique()[0]] = best_hp
        # keep the corresponding rows in hp_grouped
        tmp = grouped_sub.get_group(best_hp)
        plot_list.append(tmp)
    
    # If the method doesn't have hyperparameters       
    else :
        
        tmp = grouped.get_group(groups[group_index])
        plot_list.append(tmp)
                

plot_df = pd.concat(plot_list).reset_index()  

stab_file = figure_path + "stab_" + hp_selection[0] +  "_overall_best.png"
err_file = figure_path + "err_" + hp_selection[0] +  "_overall_best.png"

plot_order = data_stats.sort_values(sort_by).dataset 
methods = plot_df.method.unique()
hue_order = methods[[0, 2, 1, 3, 4]]

utils.plot_results(plot_df = plot_df, file = stab_file, plot_order = plot_order, hue_order = hue_order, y_var = "stab", y_label = "stability", plot_type = "stability", plot_markers = True, font_scale = 1.7, legend_shift = 0.64)
utils.plot_results(plot_df = plot_df, file = err_file, plot_order = plot_order, hue_order = hue_order, y_var = "mean_error", y_label = "relative error", plot_type = "error", plot_markers = False, font_scale = 1.7, legend_shift = 0.64)

print("Chosen hp for selection strategy " + hp_selection[0] + ":")
print(chosen_hp)