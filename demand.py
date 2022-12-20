# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 10:01:22 2022

@author: Rishikesh Sreehari
"""


import pandas as pd
import datetime
import matplotlib.pyplot as plt




df_hourly_state_demand = pd.read_excel("statewise_demand.xlsx",index_col=0)
df_losses = pd.read_excel("losses.xlsx",index_col=0)


states = df_hourly_state_demand.columns


#Create consumption dataframe with index and columns as same as demand
df_hourly_state_consumption = pd.DataFrame(data=None, columns=df_hourly_state_demand.columns, index=df_hourly_state_demand.index)


for state in states[:3]:
    print(state)
    for i in range(len(df_hourly_state_consumption)):
        print(df_hourly_state_consumption.index[i].year)
        df_hourly_state_consumption.iloc[i][state]= df_hourly_state_demand.iloc[i][state]*(1-df_losses.loc[state][df_hourly_state_consumption.index[i].year])


#df_hourly_state_consumption.reset_index(level=0, inplace=True)


max_consumption=df_hourly_state_consumption.max()    

max_consumption=max_consumption[1:]


df_max=pd.DataFrame()
df_max['Max']=max_consumption


#df_hourly_state_consumption_1=df_hourly_state_consumption.drop('Date', axis=1)

max_consumption_index = df_hourly_state_consumption.astype(float).idxmax()

df_max['Index']=max_consumption_index

df_max['Date'] = pd.to_datetime(df_max['Index']).dt.date


ndata = df_hourly_state_consumption[ (df_hourly_state_consumption['Date'] > '2014-01') & (df_hourly_state_consumption['Date'] < '2020-12')] 


plt.plot(df_hourly_state_consumption.index, df_hourly_state_consumption["Punjab"])




df_hourly_state_consumption.loc['2021-07-07':'2021-07-07','Haryana'].plot(title ='Demand')









# df_hourly_state_consumption.reset_index(level=0, inplace=True)

# df_max.reset_index(level=0, inplace=True)

# df_max['Date']=''


# for i in range(len(df_max)):
#     df_max.at[i,'Date']= df_hourly_state_consumption.loc[(df_max.loc[i]['Index'])][1]
#     print(i)
    
    
    
#     df_hourly_state_consumption.loc[576]['Date']
    
    
    
    
    



