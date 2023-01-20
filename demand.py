# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 10:01:22 2022

@author: Rishikesh Sreehari
"""

#import spyder_notebook
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np




df_hourly_state_demand = pd.read_excel("G:\My Drive\Work\Vasudha\Demand_Projection\statewise_demand.xlsx",index_col=0)
df_losses = pd.read_excel("G:\My Drive\Work\Vasudha\Demand_Projection\losses.xlsx",index_col=0)


states = df_hourly_state_demand.columns
states=states[:3]   #Slicer for debugging

#Create consumption dataframe with index and columns as same as demand
df_hourly_state_consumption = pd.DataFrame(data=None, columns=df_hourly_state_demand.columns, index=df_hourly_state_demand.index)


for state in states:
    print(state)
    for i in range(len(df_hourly_state_consumption)):
        print(df_hourly_state_consumption.index[i].year)
        df_hourly_state_consumption.iloc[i][state]= df_hourly_state_demand.iloc[i][state]*(1-df_losses.loc[state][df_hourly_state_consumption.index[i].year])

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


for state in states[:3]:
    print(state)
    #df_hourly_state_consumption.loc['2021-07-07':'2021-07-07',state].plot(title=state)
    
    # df_hourly_state_consumption.loc[df_max.loc[state]['Date']:df_max.loc[state]['Date'],state].plot(title=state)
    df_hourly_state_consumption[state][(df_hourly_state_consumption.index.day==df_max.loc[state].Date.day)&(df_hourly_state_consumption.index.year==df_max.loc[state].Date.year)&(df_hourly_state_consumption.index.month==df_max.loc[state].Date.month)].plot(title=state)
    plt.show()
    


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


index_values=['2016-2017', '2017-2018', '2018-2019', '2019-2020', '2020-2021','2021-2022','2029-2030']

s = pd.Series()
load_factor_matrix = load_factor_matrix.append(s,ignore_index=True)
load_factor_matrix.index = index_values



# plot_lf = load_factor_matrix.plot(kind='line', title='Load Factor Trend', xlabel='Years',ylabel='Load Factor(%)',xticks=years)

#load_factor_matrix = load_factor_matrix.append(pd.Series(index=load_factor_matrix.columns, name='2029'))



covid_cagr_correction = 0
cagr_limit = 0.08



projection_year = '2029-2030'
n = 8
period=4

for state in states:
    last_value=load_factor_matrix.loc[(years[-1])][state]  
    first_value=load_factor_matrix.loc[(years[1])][state] 
    cagr = (((last_value/first_value)**(1/period))-1 ) + covid_cagr_correction
    if(cagr>cagr_limit):
        cagr=cagr_limit
    projected_lf = ((cagr + 1)**n)*last_value
    load_factor_matrix.iloc[-1][state] = projected_lf
    



res_bau=pd.read_excel(r"G:\My Drive\Work\Vasudha\Demand_Projection\annex.xlsx",'res_bau',index_col=0)
ser_bau=pd.read_excel(r"G:\My Drive\Work\Vasudha\Demand_Projection\annex.xlsx",'ser_bau',index_col=0)                       
agri_bau=pd.read_excel(r"G:\My Drive\Work\Vasudha\Demand_Projection\annex.xlsx",'agri_bau',index_col=0)   
ind_bau=pd.read_excel(r"G:\My Drive\Work\Vasudha\Demand_Projection\annex.xlsx",'ind_bau',index_col=0) 


projected_demand_bau=res_bau+ser_bau+agri_bau+ind_bau

years_1=index_values



projected_demand_bau_matrix=pd.DataFrame(index=years_1,columns=states)

for state in states:
    for year in years_1:
        projected_demand_bau_matrix.loc[year][state]=projected_demand_bau.loc[year][state]


projected_demand_bau_matrix=(projected_demand_bau_matrix/8760)

projected_demand_bau_matrix = projected_demand_bau_matrix * 1000



peak_demand_matrix = peak_demand_matrix.append(s,ignore_index=True)
peak_demand_matrix.index = index_values


projection_year = '2029-2030'
n = 8
period = 4


for state in states:
    last_value=peak_demand_matrix.loc[(years[-1])][state]  
    first_value=peak_demand_matrix.loc[(years[1])][state] 
    cagr = (((last_value/first_value)**(1/len(years)))-1 ) + covid_cagr_correction
    if(cagr>cagr_limit):
        cagr=cagr_limit
    projected_lf = ((cagr + 1)**n)*last_value
    projected_peak_demand = ((cagr + 1)**n)*last_value
    peak_demand_matrix.iloc[-1][state] = projected_peak_demand    
    




# for state in states:
#     last_value=projected_demand_bau_matrix.loc[('2021-2022')][state]  
#     first_value=projected_demand_bau_matrix.loc['2017-2018'][state] 
#     cagr = (((last_value/first_value)**(1/period))-1 ) #+  covid_cagr_correction
#     projected_peak_demand = ((cagr + 1)**n)*peak_demand_matrix.loc['2021-2022'][state]
#     peak_demand_matrix.iloc[-1][state] = projected_peak_demand


# peak_demand_matrix = peak_demand_matrix.append(s,ignore_index=True)
# peak_demand_matrix.index = index_values


# for state in states:
#     last_value=peak_demand_matrix.loc[(years[-1])][state]  
#     first_value=peak_demand_matrix.loc[(years[0])][state] 
#     cagr = (((last_value/first_value)**(1/len(years)))-1 ) + covid_cagr_correction
#     projected_peak_demand = ((cagr + 1)**n)*first_value
#     peak_demand_matrix.iloc[-1][state] = projected_peak_demand    
    





load_factor_matrix_cal = (projected_demand_bau_matrix/peak_demand_matrix)*100


plot_lf = load_factor_matrix[1:].plot(kind='line', title='CAGR Factor Projection', xlabel='Years',ylabel='Load Factor(%)')
plt.xticks(rotation = 45)
plt.show()


plot_lf_1 = load_factor_matrix_cal[1:].plot(kind='line', title='BAU based Load Factor Projection', xlabel='Years',ylabel='Load Factor(%)')
plt.xticks(rotation = 45)
plt.show()


start_date = '2029-04-01 00:00:00'
end_date = '2030-03-31 23:00:00'
index = pd.date_range(start_date, end_date, freq='H')


demand_2029_peak = pd.DataFrame(index=index,columns=states)

df_hourly_state_consumption_2019_20 = df_hourly_state_consumption[df_hourly_state_consumption['Financial Year']=='2019-2020']

# df_hourly_state_consumption_2019_20 = df_hourly_state_consumption_2019_20[~df_hourly_state_consumption_2019_20.index.date.contains('2030-02-29')]

df_hourly_state_consumption_2019_20 = df_hourly_state_consumption_2019_20[:8760]


demand_2029_peak.loc[:,'Punjab'] = (peak_demand_matrix.loc['2029-2030','Punjab']/(peak_demand_matrix.loc['2019-2020','Punjab'])*df_hourly_state_consumption_2019_20['Punjab'].values)

punjb_ele_prj_2030 = projected_demand_bau_matrix.loc['2029-2030','Punjab']


demand_2029_reshaped = pd.DataFrame(index=index,columns=states)



goal_seek = 0.7

demand_2029_reshaped = ((demand_2029_peak['Punjab']- peak_demand_matrix.loc['2019-2020','Punjab'])*goal_seek ) + demand_2029_peak['Punjab']


demand_2029_reshaped = demand_2029_reshaped.to_frame()

peak_value = demand_2029_peak['Punjab'].max()

target_sum = 8302
current_sum = demand_2029_peak['Punjab'].sum()
diff = target_sum - current_sum

goal_seek = diff / (demand_2029_peak['Punjab'].count() - 1)

demand_2029_reshaped = ((demand_2029_peak['Punjab']- peak_demand_matrix.loc['2019-2020','Punjab'])*goal_seek ) + demand_2029_peak['Punjab'] 


import numpy as np

def goal_seek(f, x0, target, args=(), xtol=1e-5, maxiter=50):
    """
    Goal seek function to find the root of a function near a given starting point.
    
    Parameters:
        f (callable): The function to find the root of.
        x0 (float or array-like): The starting point for the search.
        target (float): The target value to find the root of f(x) - target.
        args (tuple, optional): Additional arguments to pass to f.
        xtol (float, optional): The tolerance for the root. The search will stop
            when the absolute difference between f(x) and target is less than xtol.
            Default is 1e-5.
        maxiter (int, optional): The maximum number of iterations to perform.
            Default is 50.
    
    Returns:
        float: The root of the function that is closest to the target value.
    """
    x = x0
    for i in range(maxiter):
        fx = f(x, *args) - target
        if abs(fx.item()) < xtol:
            return x
        dx = fx / f(x, *args)
        x -= dx
    raise ValueError("Failed to converge after {} iterations".format(maxiter))

# function to calculate the sum of demand_2029_reshaped
def sum_demand(goal_seek, peak_value, demand_2029_peak, peak_demand_matrix):
    demand_2029_reshaped = ((demand_2029_peak - peak_demand_matrix) * goal_seek) + demand_2029_peak - (goal_seek * (demand_2029_peak - peak_value))
    return demand_2029_reshaped.sum()

peak_value = demand_2029_peak['Punjab'].max()

# use the goal_seek function to find the goal_seek value that results in a sum of 150000
goal_seek = goal_seek(sum_demand, 1, 150000, args=(peak_value, demand_2029_peak, peak_demand_matrix))

demand_2029_reshaped = ((demand_2029_peak - peak_demand_matrix) * goal_seek) + demand_2029_peak - (goal_seek * (demand_2029_peak - peak_value))


    


