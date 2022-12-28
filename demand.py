# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 10:01:22 2022

@author: Rishikesh Sreehari
"""

#import spyder_notebook
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import datetime




df_hourly_state_demand = pd.read_excel("statewise_demand.xlsx",index_col=0)
df_losses = pd.read_excel("losses.xlsx",index_col=0)


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
    
years=df_hourly_state_consumption.index.year.unique()


peak_demand_matrix=pd.DataFrame(index=years,columns=states)
avg_demand_matrix=pd.DataFrame(index=years,columns=states)

for state in states:
    for year in years:
        peak_demand_matrix.loc[year,state]=df_hourly_state_consumption[state][df_hourly_state_consumption.index.year==year].max()
        avg_demand_matrix.loc[year,state]=df_hourly_state_consumption[state][df_hourly_state_consumption.index.year==year].mean()

load_factor_matrix = (avg_demand_matrix/peak_demand_matrix)*100



plot_lf = load_factor_matrix.plot(kind='line', title='Load Factor Trend', xlabel='Years',ylabel='Load Factor(%)',xticks=years)

load_factor_matrix = load_factor_matrix.append(pd.Series(index=load_factor_matrix.columns, name='2031'))



projection_year = 2031
n = projection_year - years[-1]

for state in states:
    last_value=load_factor_matrix.loc[(years[-1])][state]  
    first_value=load_factor_matrix.loc[(years[0])][state] 
    cagr = ((last_value/first_value)**(1/len(years)))-1
    projected_lf = ((cagr + 1)**n)*first_value
    load_factor_matrix.iloc[-1][state] = projected_lf
    
plot_lf = load_factor_matrix.plot(kind='line', title='Load Factor Projection', xlabel='Years',ylabel='Load Factor(%)')


