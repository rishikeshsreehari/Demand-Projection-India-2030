# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 18:46:57 2023

@author: Vasudha Foundation
"""
#trying again huubhjbu
#Peak demand estimation through scipy model

import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize
import time
import concurrent.futures



sub_sheets = ['agri_bau','ser_bau','res_bau','ind_bau']

state_wide_consumption_agri = pd.read_excel("annex.xlsx", sheet_name=sub_sheets[0],index_col=0)

time_Series_Demand_all = pd.read_excel("statewise_demand.xlsx",index_col=0)
time_Series_Demand_all.index = time_Series_Demand_all.index.round("H")


time_Series_all_states_2017 = time_Series_Demand_all[time_Series_Demand_all.index.year==2017]
time_Series_all_states_2030 = pd.DataFrame(index=time_Series_all_states_2017.index,columns=time_Series_all_states_2017.columns)


#%%
# peak_shaped_value = time_Series_all_states_2017['Punjab'].max()

#select only 2000 timestamps for now
punjab_sample_demand = list(time_Series_all_states_2017['Punjab'].round()[:3000])
punjab_total_forecast_demand = sum(punjab_sample_demand)*1.5


# Define the target sum
target_sum = punjab_total_forecast_demand

# Define the index of the number you want to remain the same
fixed_index = 3

start_time = time.time()

# Define the optimization function
def optimize_sum(x):
    # print("begin optimization")
    return abs(sum(x) - target_sum)

# Define the constraint function
def constraint(x):
    return x[fixed_index] - punjab_sample_demand[fixed_index]

# Use the minimize function to find the optimal values for the time series
result = minimize(optimize_sum, punjab_sample_demand, constraints={'type': 'eq', 'fun': constraint})

# The optimized time series will be stored in the x attribute of the result object
optimized_data = result.x

# print(result.x)

end_time = time.time()
elapsed_time = end_time - start_time

print("Elapsed time in seconds: ", elapsed_time)

#%%


def optimize_time_series(punjab_sample_demand):
    target_sum = sum(punjab_sample_demand)*1.5

    # Define the index of the number you want to remain the same
    fixed_index = 3

    # Define the optimization function
    def optimize_sum(x):
        return abs(sum(x) - target_sum)

    # Define the constraint function
    def constraint(x):
        return x[fixed_index] - punjab_sample_demand[fixed_index]

    # Use the minimize function to find the optimal values for the time series
    result = minimize(optimize_sum, punjab_sample_demand, constraints={'type': 'eq', 'fun': constraint})

    # The optimized time series will be stored in the x attribute of the result object
    return result.x

start_time = time.time()

with concurrent.futures.ThreadPoolExecutor() as executor:
    punjab_sample_demands = [list(time_Series_all_states_2017['Punjab'].round()[i:i+1000]) for i in range(0, len(time_Series_all_states_2017['Punjab']), 1000)]
    optimized_data = list(executor.map(optimize_time_series, punjab_sample_demands))

end_time = time.time()
elapsed_time = end_time - start_time

print("Elapsed time in seconds: ", elapsed_time)

#%%
optimized_data_df = [pd.DataFrame(x) for x in optimized_data]
merged_df = pd.concat(optimized_data_df, axis=0, ignore_index=True)
merged_df.rename(columns={0:'optimized_data'},inplace=True)

#%%
for i in range(len(optimized_data)):
    
    print(optimized_data[i].sum())







