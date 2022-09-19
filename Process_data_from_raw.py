#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code processes the datasets in the path raw_data_path and saves them in 
the path final_data_path
Various data stats are recorded and saved in the path data_stats_path and 
printed in format LaTex

@author: Rebecca Marion, University of Namur
"""
# Custom functions
import utils

# Other useful functions/libraries
import glob
import numpy as np
import pandas as pd
from scipy import io
from tabulate import tabulate
from texttable import Texttable

# Dataset file paths
raw_data_path = "Data_raw/"
final_data_path = "Data/"
data_stats_path = "Data_stats/"
data_path_ast = raw_data_path + "*"
file_names = glob.glob(data_path_ast)
db_names = sorted([utils.get_file_name_contents(raw_data_path, file_name, ".mat") for file_name in file_names], key=str.casefold)

# Info about data sources
data_sources = {"StatLib": ["pollution", "tecator", "NO2"],
                "MLG": ["wine", "nitrogen"],
                "LIAAD": ["stock"],
                "UCI": ["yacht", "concrete", "abalone", "energy_cooling", "machine_cpu", "triazines", "pyrim"],
                "LIBSVM": ["bodyfat", "mg", "space_ga"],
                "FRB": ["mortgage"]}

data_stats_list = list()
for db_name in db_names:
    
    # Load dataset
    mat_file = io.loadmat(raw_data_path + '%s.mat' % db_name, appendmat=False, struct_as_record=True)
    data_source = utils.find_keys_containing_str(data_sources, db_name)
    
    # Extract data for X and y
    if db_name == 'fat':
        mat_file['X'] = mat_file['X_fat']
        mat_file['y'] = mat_file['Y_fat']
        
    elif db_name == 'mortgage':
        mat_file['X'] = mat_file['X_mortgage']
        mat_file['y'] = mat_file['Y_mortgage']
        
    elif db_name == 'nitrogen':
        mat_file['X'] = mat_file['X_nitro']
        mat_file['y'] = mat_file['Y_nitro']
       
    elif db_name == 'tecator':
        mat_file['X'] = mat_file['X_tecator']
        mat_file['y'] = mat_file['Y_tecator']
        
    elif db_name in ('machine_cpu', 'stock', 'wine'):
        mat_file['X'] = mat_file['X']
        mat_file['y'] = mat_file['T']
        
    else:
        if mat_file['y'].__class__ != np.ndarray:
            tmp = mat_file['y'].toarray()
            mat_file['y'] = mat_file['X']
            mat_file['X'] = tmp
    
    # Save processed data
    file_name = final_data_path + db_name + ".pkl"
    utils.save_data(file_name, mat_file)
            

    # Collect stats about data
    X = mat_file['X']
    d = X.shape[1]
    n = X.shape[0]
    dim_ratio = d/n
    data_stats = {"dataset": [db_name],
                  "n": [n],
                 "d": [d],
                 "dim_ratio": [dim_ratio],
                 "data_source": data_source}
    data_stats_list.append(pd.DataFrame.from_dict(data_stats))
            
    
# Compile all data stats
df = pd.concat(data_stats_list).sort_values(by = "dim_ratio").reset_index(drop = True) 
# Calculate log of dim_ratio 
df['log_dim_ratio'] = np.log10(df["dim_ratio"])

# Save data stats
file_name = data_stats_path + "data_stats.pkl"
utils.save_data(file = file_name, element = df)

# Create LaTex table of data stats
df_latex = df[["dataset", "n", "d", "log_dim_ratio", "data_source"]]
df_latex = df.assign(log_dim_ratio = df_latex['log_dim_ratio'].map('{:,.2f}'.format))
col_names = np.array(df_latex.columns).reshape((1, -1))
rows = np.array(df_latex)
rows = np.concatenate((col_names, rows))

table = Texttable()
table.set_deco(Texttable.HEADER | Texttable.VLINES)
table.add_rows(rows)

print('\nTabulate Latex:')
print(tabulate(rows, headers='firstrow', tablefmt='latex', colalign = ["center"] * 5))

# tabulate doc: https://pypi.org/project/tabulate/