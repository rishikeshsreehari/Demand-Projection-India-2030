# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 10:01:22 2022

@author: Rishikesh Sreehari
"""

#import spyder_notebook
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import numpy as np
from scipy.optimize import root_scalar




df_hourly_state_demand = pd.read_excel(r"G:\My Drive\Work\Vasudha\Demand_Projection\statewise_demand.xlsx",index_col=0)
df_losses = pd.read_excel(r"G:\My Drive\Work\Vasudha\Demand_Projection\losses.xlsx",index_col=0)


res_bau=pd.read_excel(r"G:\My Drive\Work\Vasudha\Demand_Projection\annex.xlsx",'res_bau',index_col=0)
ser_bau=pd.read_excel(r"G:\My Drive\Work\Vasudha\Demand_Projection\annex.xlsx",'ser_bau',index_col=0)                       
agri_bau=pd.read_excel(r"G:\My Drive\Work\Vasudha\Demand_Projection\annex.xlsx",'agri_bau',index_col=0)   
ind_bau=pd.read_excel(r"G:\My Drive\Work\Vasudha\Demand_Projection\annex.xlsx",'ind_bau',index_col=0) 



index_values=['2016-2017', '2017-2018', '2018-2019', '2019-2020', '2020-2021','2021-2022','2029-2030']


projected_demand_bau=res_bau+ser_bau+agri_bau+ind_bau

years_1=index_values

states = projected_demand_bau.columns

projected_demand_bau_matrix=pd.DataFrame(index=years_1,columns=states)

for state in states:
    for year in years_1:
        projected_demand_bau_matrix.loc[year][state]=projected_demand_bau.loc[year][state]



#Create consumption dataframe with index and columns as same as demand
df_hourly_state_consumption = pd.DataFrame(data=None, columns=df_hourly_state_demand.columns, index=df_hourly_state_demand.index)


loss_dict = df_losses.T.to_dict()

# Iterate over the states
for state in states:
    print(state)
    # Extract the year for each index
    years = df_hourly_state_consumption.index.year
    # Extract the loss values for each state-year pair
    losses = [loss_dict[state][year] for year in years]
    losses = pd.Series(losses, index=df_hourly_state_consumption.index)
    df_hourly_state_consumption[state] = df_hourly_state_demand[state] * (1 - losses)







# for state in states:
#     print(state)
#     for i in range(len(df_hourly_state_consumption)):
#         print(df_hourly_state_consumption.index[i].year)
#         df_hourly_state_consumption.iloc[i][state]= df_hourly_state_demand.iloc[i][state]*(1-df_losses.loc[state][df_hourly_state_consumption.index[i].year])




df_hourly_state_consumption.index = df_hourly_state_consumption.index.round("H")

#df_hourly_state_consumption.reset_index(level=0, inplace=True)


max_consumption=df_hourly_state_consumption.max()    




df_max=pd.DataFrame()
df_max['Max']=max_consumption


#df_hourly_state_consumption_1=df_hourly_state_consumption.drop('Date', axis=1)

max_consumption_index = df_hourly_state_consumption.astype(float).idxmax()
#df_hourly_state_consumption=df_hourly_state_consumption.astype(float)

df_max['Index']=max_consumption_index
df_max['Date'] = pd.to_datetime(df_max['Index']).dt.date


# for state in states[:3]:
#     print(state)
#     #df_hourly_state_consumption.loc['2021-07-07':'2021-07-07',state].plot(title=state)
    
#     # df_hourly_state_consumption.loc[df_max.loc[state]['Date']:df_max.loc[state]['Date'],state].plot(title=state)
#     df_hourly_state_consumption[state][(df_hourly_state_consumption.index.day==df_max.loc[state].Date.day)&(df_hourly_state_consumption.index.year==df_max.loc[state].Date.year)&(df_hourly_state_consumption.index.month==df_max.loc[state].Date.month)].plot(title=state)
#     plt.show()
    


df_hourly_state_consumption['Date'] = df_hourly_state_consumption.index
df_hourly_state_consumption['Financial Year'] = df_hourly_state_consumption['Date'].dt.to_period('Q-MAR').dt.qyear.apply(lambda x: str(x-1) + "-" + str(x))

years=df_hourly_state_consumption['Financial Year'].unique()



peak_demand_matrix=pd.DataFrame(index=years,columns=states)
avg_demand_matrix=pd.DataFrame(index=years,columns=states)

for state in states:
    for year in years:
        peak_demand_matrix.loc[year,state]=df_hourly_state_consumption[state][df_hourly_state_consumption['Financial Year']==year].max()
        avg_demand_matrix.loc[year,state]=df_hourly_state_consumption[state][df_hourly_state_consumption['Financial Year']==year].mean()
        
        
 
# for state in states:
#     for year in years:
#         peak_demand_matrix.loc[year,state]=df_hourly_state_consumption[state][df_hourly_state_consumption.index.year==year].max()
#         avg_demand_matrix.loc[year,state]=df_hourly_state_consumption[state][df_hourly_state_consumption.index.year==year].mean()       
        
        

load_factor_matrix = (avg_demand_matrix/peak_demand_matrix)*100




s = pd.Series()
load_factor_matrix = load_factor_matrix.append(s,ignore_index=True)
load_factor_matrix.index = index_values



# plot_lf = load_factor_matrix.plot(kind='line', title='Load Factor Trend', xlabel='Years',ylabel='Load Factor(%)',xticks=years)

#load_factor_matrix = load_factor_matrix.append(pd.Series(index=load_factor_matrix.columns, name='2029'))



# covid_cagr_correction = 0
# cagr_limit = 0.07



# projection_year = '2029-2030'
# n = 8
# period=4

# for state in states:
#     last_value=load_factor_matrix.loc[(years[-1])][state]  
#     first_value=load_factor_matrix.loc[(years[1])][state] 
#     cagr = (((last_value/first_value)**(1/period))-1 ) + covid_cagr_correction
#     if(cagr>cagr_limit):
#         cagr=cagr_limit
#     projected_lf = ((cagr + 1)**n)*last_value
#     load_factor_matrix.iloc[-1][state] = projected_lf
    







peak_demand_matrix = peak_demand_matrix.append(s,ignore_index=True)
peak_demand_matrix.index = index_values


projection_year = '2029-2030'
n = 8
period = 4


for state in states:
    last_value=peak_demand_matrix.loc[(years[-1])][state]  
    first_value=peak_demand_matrix.loc[(years[1])][state] 
    cagr = (((last_value/first_value)**(1/period)-1 ))
    projected_lf = ((cagr + 1)**n)*last_value
    projected_peak_demand = ((cagr + 1)**n)*last_value
    peak_demand_matrix.iloc[-1][state] = projected_peak_demand    
    




# load_factor_matrix_cal = (projected_demand_bau_matrix/peak_demand_matrix)*100


# plot_lf = load_factor_matrix[1:].plot(kind='line', title='CAGR Factor Projection', xlabel='Years',ylabel='Load Factor(%)')
# plt.xticks(rotation = 45)
# plt.show()


# plot_lf_1 = load_factor_matrix_cal[1:].plot(kind='line', title='BAU based Load Factor Projection', xlabel='Years',ylabel='Load Factor(%)')
# plt.xticks(rotation = 45)
# plt.show()


start_date = '2029-04-01 00:00:00'
end_date = '2030-03-31 23:00:00'
index = pd.date_range(start_date, end_date, freq='H')


demand_2029_peak = pd.DataFrame(index=index,columns=states)

df_hourly_state_consumption_2019_20 = df_hourly_state_consumption[df_hourly_state_consumption['Financial Year']=='2019-2020']


date_to_drop = '2020-02-29'

mask = df_hourly_state_consumption_2019_20.index.date != pd.to_datetime(date_to_drop).date()

# Use the mask to select the rows you want to keep
df_hourly_state_consumption_2019_20 = df_hourly_state_consumption_2019_20[mask]


for state in states:
    demand_2029_peak.loc[:,state] = ( df_hourly_state_consumption_2019_20[state].values * peak_demand_matrix.loc['2029-2030',state] )/df_hourly_state_consumption_2019_20[state].max()





#punjb_ele_prj_2030 = projected_demand_bau_matrix.loc['2029-2030','Punjab'] * 8760

punjb_ele_prj_2030 = projected_demand_bau_matrix.loc['2029-2030','Punjab']*1000

demand_2029_reshaped = pd.DataFrame(index=index,columns=states)

# goal_seek_value = 1

# demand_2029_reshaped = ((demand_2029_peak - peak_demand_matrix.loc['2029-2030','Punjab'] )* goal_seek_value) + peak_demand_matrix.loc['2029-2030','Punjab'] 



def goal_seek(demand_2029_peak, peak_demand_matrix, target_val):
    def func(goal_seek_value):
        demand_2029_reshaped = ((demand_2029_peak - peak_demand_matrix.loc['2029-2030','Uttar Pradesh'] )* goal_seek_value) + peak_demand_matrix.loc['2029-2030','Uttar Pradesh']
        return demand_2029_reshaped.sum().iat[0] - target_val

    result = root_scalar(func, x0=0.1, x1=2 )
    if result.converged:
        return result.root
    else:
        raise ValueError("Goal seeking failed to converge.")

for state in states: 
    target_val = projected_demand_bau_matrix.loc['2029-2030',state]*1000
    result = goal_seek(demand_2029_peak, peak_demand_matrix, target_val)
    demand_2029_reshaped = (((demand_2029_peak - peak_demand_matrix.loc['2029-2030',state] )* result) + peak_demand_matrix.loc['2029-2030',state])


target_val = projected_demand_bau_matrix.loc['2029-2030','Uttar Pradesh']*1000
result = goal_seek(demand_2029_peak, peak_demand_matrix, target_val)
demand_2029_reshaped = (((demand_2029_peak - peak_demand_matrix.loc['2029-2030','Uttar Pradesh'] )* result) + peak_demand_matrix.loc['2029-2030','Uttar Pradesh'])





#%% Looping for states

def goal_seek(demand_2029_peak, peak_demand_matrix, target_val,state):
    def func(goal_seek_value):
        demand_2029_reshaped = ((demand_2029_peak - peak_demand_matrix.loc['2029-2030',state] )* goal_seek_value) + peak_demand_matrix.loc['2029-2030',state]
        return demand_2029_reshaped.sum().iat[0] - target_val

    result = root_scalar(func, x0=0.1, x1=2 )
    if result.converged:
        return result.root
    else:
        raise ValueError("Goal seeking failed to converge.")

for state in states: 
    target_val = projected_demand_bau_matrix.loc['2029-2030',state]*1000
    result = goal_seek(demand_2029_peak, peak_demand_matrix, target_val,state)
    demand_2029_reshaped.loc[:,state] = (((demand_2029_peak.loc[:,state] - peak_demand_matrix.loc['2029-2030',state] )* result) + peak_demand_matrix.loc['2029-2030',state])



import plotly.graph_objects as go

from plotly.offline import plot


for state in states:

    fig = go.Figure()
    
    


    fig.add_trace(go.Scatter(x=demand_2029_peak.index, y=demand_2029_peak[state],
                    mode='lines',
                    name='Peak'))

    fig.add_trace(go.Scatter(x=demand_2029_peak.index, y=demand_2029_reshaped[state],
                    mode='lines',
                    name='adjusted'))

    fig.add_trace(go.Scatter(x=demand_2029_peak.index, y=df_hourly_state_consumption_2019_20[state],
                    mode='lines',
                    name='base'))
    
    fig.update_layout(title=state)

    plot(fig)








#%%

result = goal_seek(demand_2029_peak, peak_demand_matrix, target_val)

demand_2029_reshaped = (((demand_2029_peak - peak_demand_matrix.loc['2029-2030','Punjab'] )* result) + peak_demand_matrix.loc['2029-2030','Punjab'])




# demand_2029_reshaped.to_csv("demand_2029_reshaped.csv")
# demand_2029_peak.to_csv("demand_2029_peak.csv")



# fig, ax = plt.subplots()
# ax.plot(demand_2029_peak.index, demand_2029_peak['Punjab'], label='Peak')
# ax.plot(demand_2029_reshaped.index, demand_2029_reshaped['Punjab'], label='Adjusted')
# ax.plot(demand_2029_reshaped.index, df_hourly_state_consumption_2019_20['Punjab'], label='Base')
# ax.legend()
# plt.show()
# # Show the plot

import plotly.graph_objects as go

from plotly.offline import plot

fig = go.Figure()


fig.add_trace(go.Scatter(x=demand_2029_peak.index, y=demand_2029_peak['Uttar Pradesh'],
                    mode='lines',
                    name='Peak'))

fig.add_trace(go.Scatter(x=demand_2029_peak.index, y=demand_2029_reshaped['Uttar Pradesh'],
                    mode='lines',
                    name='adjusted'))

fig.add_trace(go.Scatter(x=demand_2029_peak.index, y=df_hourly_state_consumption_2019_20['Uttar Pradesh'],
                    mode='lines',
                    name='base'))

plot(fig)

fig.show()














