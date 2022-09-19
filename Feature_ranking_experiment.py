#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This code runs the experiment for feature ranking based on feature selection
criteria related to the delta test (delta test, delta test bar, bootstrapped
delta test, gamma test, product estimator). Results are saved in the path
output_file_path and include prediction error (MSEP), relative MSEP, stability
(Kuncheva index) and criterion values for nb_runs and 1 to min(d, d_max)
selected features.

@author: Rebecca Marion, University of Namur
"""
# Custom functions
import utils

# Other useful functions
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


output_file_path = "Results/"

# Experiment parameters 
nb_runs = 100
d_max = 20 # maximum number of features to select
gamma_test_hp = np.arange(2, 42, 2)
DT_bar_hp = gamma_test_hp
prop_train = 0.7

# Prediction parameters
task = 'regression'
predictor = KNeighborsRegressor
predictor_hp = {'n_neighbors':5}

# Methods and hyperparameters
methods = ["delta-test", "gamma-test", "delta-test-bar", "product-estimator",  "delta-boot"]
hp_vals = dict(zip(methods, [[None]]*len(methods)))
hp_vals.update({methods[1] : gamma_test_hp})
hp_vals.update({methods[2] : DT_bar_hp})
print(hp_vals)

# Dataset file paths, file names and names
data_path = "Data/"
data_path_ast = "Data/" + "*"
file_names = glob.glob(data_path_ast)
data_names = [utils.get_file_name_contents(data_path, file_name) for file_name in file_names]

# Result files
file_names_results = glob.glob(output_file_path + "*")

for method_index in range(len(methods)):
    
    method_name = methods[method_index]
    nb_hps = len(hp_vals[method_name])
   
    for hp_index in range(nb_hps):
        
        hp_val = hp_vals[method_name][hp_index]
        output_file_name = utils.create_results_file_name(folder_path = output_file_path, method = method_name, hp_val = hp_val)
        
        # If results file already exists, load it
        if output_file_name in file_names_results:
            method_results = utils.load_data(output_file_name)
        # If not, create a dict
        else :
            method_results = {"method": method_name,
                              "hp": hp_val}

        # Dictionary of arguments for utils.feature_ranking() function
        scorer_hp = {"type": method_name,
                     "hp": hp_val}
        
        print('\n method = %s, hp_id = %d' % (method_name, hp_index))
        print('\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        
        for data_index in range(len(data_names)):
            
            db_name = data_names[data_index]
            
            # If results for db_name are not already in method_results
            if not db_name in method_results.keys():
                
                file_name = data_path + db_name + ".pkl"
                
                data = utils.load_data(file_name)
                X = data['X']
                y = data['y'].astype(np.double)
                d = X.shape[1]
                n = X.shape[0]
                
                # Set the number of neighbors based on hp_val
                if hp_val:
                    n_neighbors = hp_val
                else:
                    n_neighbors = 1
                    
                # If the number of neighbors is smaller than the training size
                if n_neighbors < (n*prop_train - 1):
                    
                    print('\n processing %s (%d x %d)' % (db_name, n, d))
                    print ('[', end='')
                    
                    prediction_scores_list = []
                    crit_vals_list = []
                    selection_paths_list = []
                    for run_id in range(nb_runs):
                        
                        print('.', end='')
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=prop_train, random_state = run_id)
                        
                        # Normalize X and y
                        X_train, X_test = utils.normalize(X_train, X_test)
                        y_train, y_test = utils.normalize(y_train, y_test)
                      
                        # Add small noise to avoid KNN issues
                        np.random.seed(run_id)
                        X_train += np.random.normal(0, 1e-12, X_train.shape)
                        X_test += np.random.normal(0, 1e-12, X_test.shape)
                        
                        # Rank features
                        selection_path, criteria = utils.feature_ranking(X_train = X_train, 
                                                                         y_train = y_train[:, 0], 
                                                                         selection_criterion = utils.FS_scorer, 
                                                                         scorer_hp = scorer_hp, 
                                                                         d_max = d_max)
                        # Calculate prediction performance
                        prediction_score = utils.prediction_score(X_train, y_train, X_test, 
                                                                  y_test.astype(np.double), 
                                                                  task, selection_path, 
                                                                  predictor, predictor_hp)
                        
                        # Append results
                        selection_paths_list.append(selection_path)
                        crit_vals_list.append(criteria)
                        prediction_scores_list.append(prediction_score)
                        
                    print (']', end='')
                        
                    
                    # Prediction error
                    prediction_scores_arr = np.array(prediction_scores_list)
                    # Relative prediction error
                    rel_prediction_scores_arr = utils.calc_rel_error(baseline_error_arr = prediction_scores_arr[:, 0], 
                                                                     error_arr = prediction_scores_arr[:, 1:])
                    # Stability
                    selection_paths_arr = np.array(selection_paths_list)
                    stab_scores_arr = utils.stability_scores(selection_paths_arr = selection_paths_arr, d = d)
                    # Criterion values
                    crit_vals_arr = np.array(crit_vals_list)
                    
                    # Compile results
                    results = {"pred_scores": prediction_scores_arr,
                               "rel_pred_scores": rel_prediction_scores_arr,
                               "stab": stab_scores_arr,
                               "crit": crit_vals_arr}
                     
                    # Add results to dictionary
                    method_results[db_name] = results
                     
                    # Save results
                    utils.save_data(file = output_file_name, element = method_results)