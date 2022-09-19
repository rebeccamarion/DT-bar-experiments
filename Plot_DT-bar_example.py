#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code plots an example of the mean and std of the DT-bar criterion for 
different values of the hyperparameter K. It also plots empirical densities 
of the DT-bar criterion for the different features in the example dataset.

@author: Rebecca Marion, University of Namur
"""
# Custom functions
import utils

# Other useful functions
import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# Experiment parameters 
nb_runs = 100
d_max = 20
max_neighbors = 40
prop_train = 0.7

# Prediction parameters
task = 'regression'
predictor = KNeighborsRegressor
predictor_hp = {'n_neighbors':5}

# Figure filepath
figure_path = "Figures/"

# Datasets
data_path = "Data/"
data_path_ast = "Data/" + "*"
file_names = glob.glob(data_path_ast)
data_names = [utils.get_file_name_contents(data_path, file_name) for file_name in file_names]

# Load data
db_name = "concrete"
file_name = data_path + db_name + ".pkl"
data = utils.load_data(file_name)
X = data['X']
y = data['y'].astype(np.double)
d = X.shape[1]
n = X.shape[0]

feature_crits = [] # each list element contains all criterion values for a feature
emp_mean_list = [] # each list element contains mean values for a feature
emp_sd_list = [] # each list element contains sd values for a feature
for feature_index in range(d):
    
    crits_list = []
    for run_id in range(nb_runs):
        
        # split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=prop_train, random_state = run_id)
        
    
        # Normalize X and y
        X_train, X_test = utils.normalize(X_train, X_test)
        y_train, y_test = utils.normalize(y_train, y_test)
      
        # add a small noise to avoid kNN issues
        np.random.seed(10)
        X_train += np.random.normal(0, 1e-12, X_train.shape)
        
        # Calculate DT-bar criterio for all K between 1 and max_neighbors
        crits = utils.DT_bar_criterion_multiple_K(X_train = X_train, 
                                                  y_train = y_train[:, 0], 
                                                  feature_index = feature_index, 
                                                  max_neighbors = max_neighbors)
        crits_list.append(crits)
    
    
    crits_all = np.array(crits_list)
    feature_crits.append(crits_all) 
    
    # Mean and sd for feature number feature_index
    emp_mean = np.mean(crits_all, axis = 0)
    emp_sd = np.std(crits_all, axis = 0)
    
    # Add to list
    emp_mean_list.append(emp_mean)
    emp_sd_list.append(emp_sd)
    
## PLOTS ##

n_shades = 3
n_lines = len(feature_crits)

palette_init = sns.color_palette("Greys", n_colors = n_shades)[::-1]
n_repeated_shades = round((n_lines - n_shades)/n_shades) + 1
palette = sorted(sum(list(itertools.repeat(palette_init, n_repeated_shades)), []))[:n_lines]

dashes_init = ["-","--","-."]
dashes = sum(list(itertools.repeat(dashes_init, n_repeated_shades)), [])[:n_lines]

lwd = 3
plt.rcParams['font.size'] = 20

# Mean of DT-bar
emp_mean_all = pd.DataFrame(np.array(emp_mean_list).T)
g = emp_mean_all.plot(legend = False, linewidth = lwd, title = "Mean of DT-bar across repetitions", color = palette, style = dashes)
g.set_xlabel("# of neighbors")
g.figure.set_size_inches(8, 7)
g.figure.savefig(figure_path + "mean_by_feature.png")

# Sd of DT-bar
emp_sd_all = pd.DataFrame(np.array(emp_sd_list).T)
g = emp_sd_all.plot(legend = False, linewidth = lwd, title = "Std Dev of DT-bar across repetitions", color = palette, style = dashes)
g.set_xlabel("# of neighbors")
g.figure.set_size_inches(8, 7)
g.figure.savefig(figure_path + "sd_by_feature.png")

# Example densities for K = 1 and K = 30

# Get criterio values for K = 1 and K = 30
crits1 = np.array([x[:, 0] for x in feature_crits]).T
crits30 = np.array([x[:, 29] for x in feature_crits]).T

# Determine x limits for plotting
x_max = np.maximum(crits1.max(), crits30.max())
x_min = np.minimum(crits1.min(), crits30.min())
x_range = x_max - x_min
eps = 0.1*x_range
x_lims = (x_min - eps, x_max + eps)

# Plot for K = 1
crits_K = pd.DataFrame(crits1)
g = crits_K.plot.density(legend = False, linewidth = lwd, title = "1 neighbor", color = palette, style = dashes)
g.set_xlabel("DT-bar value")
g.figure.set_size_inches(8, 7)
g.set_xlim(x_lims)
g.figure.savefig(figure_path + "1_neighbor.png")

# Plot for K = 30

crits_K = pd.DataFrame(crits30)
g = crits_K.plot.density(legend = False, linewidth = lwd, title = "30 neighbors", color = palette, style = dashes)
g.set_xlabel("DT-bar value")
g.set_xlim(x_lims)
g.figure.set_size_inches(8, 7)
g.figure.savefig(figure_path + "30_neighbor.png")


  