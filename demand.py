
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
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import plot
from scipy.optimize import root_scalar
import time
import datetime
import matplotlib.dates as mdates







df_hourly_state_demand = pd.read_excel(r"G:\My Drive\Work\Vasudha\Demand_Projection\statewise_demand.xlsx",index_col=0)
df_losses = pd.read_excel(r"G:\My Drive\Work\Vasudha\Demand_Projection\losses.xlsx",index_col=0)


res_bau=pd.read_excel(r"G:\My Drive\Work\Vasudha\Demand_Projection\annex.xlsx",'res_bau',index_col=0)
ser_bau=pd.read_excel(r"G:\My Drive\Work\Vasudha\Demand_Projection\annex.xlsx",'ser_bau',index_col=0)                       
agri_bau=pd.read_excel(r"G:\My Drive\Work\Vasudha\Demand_Projection\annex.xlsx",'agri_bau',index_col=0)   
ind_bau=pd.read_excel(r"G:\My Drive\Work\Vasudha\Demand_Projection\annexure_results.xlsx",'ind_bau_ut',index_col=0) 





index_values=['2016-2017', '2017-2018', '2018-2019', '2019-2020', '2020-2021','2021-2022','2022-2023','2029-2030']


projected_demand_bau=res_bau+ser_bau+agri_bau+ind_bau

projected_demand_bau=projected_demand_bau.drop(columns=['Puducherry'])

years_1=index_values

states = projected_demand_bau.columns

projected_demand_bau_matrix=pd.DataFrame(index=years_1,columns=states)

for state in states:
    for year in years_1:
        projected_demand_bau_matrix.loc[year][state]=projected_demand_bau.loc[year][state]

verification = pd.DataFrame(index = states, columns=['CAGR','Peak Demand','Peak Demand LF','Reshaped LF','Base Year LF','Projected Demand'])



#Create consumption dataframe with index and columns as same as demand
df_hourly_state_consumption = pd.DataFrame(data=None, columns=df_hourly_state_demand.columns, index=df_hourly_state_demand.index)
india_hourly_demand = pd.DataFrame(data=None,index=df_hourly_state_demand.index)
india_hourly_demand = df_hourly_state_demand.sum(axis=1)


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




df_hourly_state_consumption.index = df_hourly_state_consumption.index.round("H")
india_hourly_demand.index = india_hourly_demand.index.round("H")




max_consumption=df_hourly_state_consumption.max()    




df_max=pd.DataFrame()
df_max['Max']=max_consumption


#df_hourly_state_consumption_1=df_hourly_state_consumption.drop('Date', axis=1)

max_consumption_index = df_hourly_state_consumption.astype(float).idxmax()
#df_hourly_state_consumption=df_hourly_state_consumption.astype(float)

df_max['Index']=max_consumption_index
df_max['Date'] = pd.to_datetime(df_max['Index']).dt.date



df_hourly_state_consumption['Date'] = df_hourly_state_consumption.index
df_hourly_state_consumption['Financial Year'] = df_hourly_state_consumption['Date'].dt.to_period('Q-MAR').dt.qyear.apply(lambda x: str(x-1) + "-" + str(x))

india_hourly_demand = df_hourly_state_demand.sum(axis=1)

india_hourly_demand = india_hourly_demand.to_frame()
india_hourly_demand['Calendar Year'] = pd.to_datetime(india_hourly_demand.index, format='%Y-%m-%d %H:%M:%S').year

calendar_years = [2017, 2018, 2019, 2020, 2021,2022]


years=df_hourly_state_consumption['Financial Year'].unique()



peak_demand_matrix=pd.DataFrame(index=years,columns=states)
avg_demand_matrix=pd.DataFrame(index=years,columns=states)

for state in states:
    for year in years:
        peak_demand_matrix.loc[year,state]=df_hourly_state_consumption[state][df_hourly_state_consumption['Financial Year']==year].max()
        avg_demand_matrix.loc[year,state]=df_hourly_state_consumption[state][df_hourly_state_consumption['Financial Year']==year].mean()

#%% India Peak Demand Growth

india_hourly_demand = india_hourly_demand.rename(columns={0: 'Data'})
india_peak_demand_matrix=pd.DataFrame(index=calendar_years,columns=['Peak Demand'])
      
  
for year in calendar_years:
    india_peak_demand_matrix.loc[year,'Peak Demand']= india_hourly_demand.loc[india_hourly_demand['Calendar Year'] == year, 'Data'].max()       
  
import matplotlib.pyplot as plt

plt.plot(india_peak_demand_matrix.index, india_peak_demand_matrix['Peak Demand']/1000)
plt.title('Growth of Peak Demand')
plt.xlabel('Year')
plt.ylabel('Peak Demand (GW)')
plt.xticks(india_peak_demand_matrix.index)
plt.show()

#%%Finding Peak days for India

yearly_demand = india_hourly_demand.groupby(by=india_hourly_demand.index.year)


max_demand_indexes = yearly_demand['Data'].idxmax()



peak_demand_days_in = pd.Series(max_demand_indexes.dt.date, index=max_demand_indexes.dt.year)


peak_demand_days_in = peak_demand_days_in.rename_axis('Year')

peak_demand_days_in = peak_demand_days_in.to_frame()

peak_demand_days_in = peak_demand_days_in.rename(columns={'Data': 'Peak Demand Date'})

peak_days = peak_demand_days_in['Peak Demand Date'].tolist()

peak_days_str = [d.strftime('%Y-%m-%d %H:%M:%S') for d in peak_days]


for day in peak_days_str:
    ((india_hourly_demand.loc[day[:10],'Data'])/1000).plot(title=day[:10])
    plt.show()



    
#%%


load_factor_matrix = (avg_demand_matrix/peak_demand_matrix)*100




s = pd.Series()
# load_factor_matrix = load_factor_matrix.append(s,ignore_index=True)
# load_factor_matrix.index = index_values



peak_demand_matrix = peak_demand_matrix.append(s,ignore_index=True)
peak_demand_matrix.index = index_values

cagr_upper_limit = 0.07
cagr_lower_limit = 0.04

projection_year = '2029-2030'
n = 8
period = 5


df_cagr = pd.DataFrame(index=states,columns=['CAGR','Peak_Demand_2030'])


for state in states:
    last_value=peak_demand_matrix.loc['2021-2022'][state]  
    first_value=peak_demand_matrix.loc['2016-2017'][state] 
    cagr = (((last_value/first_value)**(1/period)-1 ))
    if cagr > cagr_upper_limit:
        cagr = cagr_upper_limit
    if cagr < cagr_lower_limit:
        cagr = cagr_lower_limit    
    df_cagr.loc[state,'CAGR'] = cagr*100
    verification.loc[state,'CAGR'] = cagr*100
    projected_lf = ((cagr + 1)**n)*last_value
    projected_peak_demand = ((cagr + 1)**n)*last_value
    peak_demand_matrix.loc['2029-2030'][state] = projected_peak_demand
    df_cagr.loc[state,'Peak_Demand_2030'] = projected_peak_demand 
    verification.loc[state,'Peak Demand'] = projected_peak_demand


state='Tamil Nadu'

last_value=peak_demand_matrix.loc['2021-2022'][state]  

cagr = 0.06  #CAGR manually calcualted
df_cagr.loc[state,'CAGR'] = cagr*100
verification.loc[state,'CAGR'] = cagr*100
projected_lf = ((cagr + 1)**n)*last_value
projected_peak_demand = ((cagr + 1)**n)*last_value
peak_demand_matrix.loc['2029-2030'][state] = projected_peak_demand
df_cagr.loc[state,'Peak_Demand_2030'] = projected_peak_demand 
verification.loc[state,'Peak Demand'] = projected_peak_demand









start_date = '2029-04-01 00:00:00'
end_date = '2030-03-31 23:00:00'
index = pd.date_range(start_date, end_date, freq='H')


demand_2029_peak = pd.DataFrame(index=index,columns=states)

df_hourly_state_consumption_2018_19 = df_hourly_state_consumption[df_hourly_state_consumption['Financial Year']=='2018-2019']
df_hourly_state_consumption_2019_20 = df_hourly_state_consumption[df_hourly_state_consumption['Financial Year']=='2019-2020']
df_hourly_state_consumption_2021_22 = df_hourly_state_consumption[df_hourly_state_consumption['Financial Year']=='2021-2022']



# date_to_drop = '2020-02-29'

# mask = df_hourly_state_consumption_2019_20.index.date != pd.to_datetime(date_to_drop).date()

# # Use the mask to select the rows you want to keep
# df_hourly_state_consumption_2019_20 = df_hourly_state_consumption_2019_20[mask]


for state in states:
    demand_2029_peak.loc[:,state] = ( df_hourly_state_consumption_2021_22[state].values * peak_demand_matrix.loc['2029-2030',state] )/df_hourly_state_consumption_2021_22[state].max()




#%% Correct Code

demand_2029_reshaped = pd.DataFrame(index=index,columns=states)




start_time = time.time()



for state in states:
    goal_step = 0.0001
    goal_value = 0.2 
    threshold = 1
    print(state)
    target_val = projected_demand_bau_matrix.loc['2029-2030',state]*1000
    demand_2029_reshaped.loc[:,state] = (((demand_2029_peak.loc[:,state] - peak_demand_matrix.loc['2029-2030',state] )* goal_value) + peak_demand_matrix.loc['2029-2030',state])
    while(demand_2029_reshaped[state].sum() - target_val > threshold):
        demand_2029_reshaped.loc[:,state] = (((demand_2029_peak.loc[:,state] - peak_demand_matrix.loc['2029-2030',state] )* goal_value) + peak_demand_matrix.loc['2029-2030',state])
        #print(goal_value)
        goal_value = goal_value + goal_step
    
    print(goal_value)
print("--- %s seconds ---" % (time.time() - start_time))







for state in states:

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=demand_2029_reshaped.index, y=demand_2029_reshaped[state],
                    mode='lines',
                    name='Reshaped'))
    
    fig.add_trace(go.Scatter(x=demand_2029_reshaped.index, y=demand_2029_peak[state],
                    mode='lines',
                    name='Peak'))

    

    fig.add_trace(go.Scatter(x=demand_2029_reshaped.index, y=df_hourly_state_consumption_2019_20[state],
                    mode='lines',
                    name='Base'))
    
    fig.update_layout(title=state)

    plot(fig)


#%%Peak CAGR vs Energy CAGR

veri = pd.DataFrame(index = states, columns=['Peak CAGR','Energy CAGR','Diff'])

projection_year = '2029-2030'
n = 8
period = 6




for state in states:
    last_value=peak_demand_matrix.loc['2021-2022'][state]  
    first_value=peak_demand_matrix.loc['2016-2017'][state] 
    cagr = (((last_value/first_value)**(1/period)-1 )) 
    veri.loc[state,'Peak CAGR'] = cagr*100
    
    first_value=projected_demand_bau_matrix.loc['2021-2022'][state]  
    last_value=projected_demand_bau_matrix.loc['2029-2030'][state] 
    cagr = (((last_value/first_value)**(1/n)-1 )) 
    veri.loc[state,'Energy CAGR'] = cagr*100
    
veri['diff']=(veri['Energy CAGR']-veri['Peak CAGR'])
    
    
    
#%% Top 10 States



# extract data for year 2029-2030
df_2029_2030_peak = peak_demand_matrix.loc['2029-2030']

# sort data in descending order based on peak demand
df_2029_2030_sorted = df_2029_2030_peak.sort_values(ascending=False)

# extract top 10 states
top_10_states = df_2029_2030_sorted.index[:10].tolist()

peak_days_states = demand_2029_reshaped.idxmax(axis=0)


peak_days_states = pd.to_datetime(peak_days_states)

# extract the day component and store it in a new Series
peak_days_states = peak_days_states.dt.date



#Peak Day Load Curves for Top 10 States

for state in top_10_states:
    # extract the peak day for the current state as a date object
    peak_day = peak_days_states[state]
    
    # convert the date object to a datetime object with a default time of 00:00:00
    peak_day_dt = datetime.datetime.combine(peak_day, datetime.time())
    peak_day_str = peak_day_dt.strftime("%Y-%m-%d")
    
    # extract the demand data for the current state and peak day
    data = demand_2029_reshaped.loc[peak_day_str, state] / 1000
    
    # create the plot and set the title using the state and peak day
    fig, ax = plt.subplots()
    ax.plot(data, label=state)
    ax.set_ylabel("Demand (GW)")
    ax.set_xlabel("Hour of Day")
    ax.set_title('2029 Peak Day Load Curve of '+ state + ' for ' + peak_day_str)
    ax.legend()
   
   # set x-axis ticks to be hourly and display only the hours
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)
   
    plt.show()
    
#Yearly Load Curves for Top 10 states

for state in top_10_states:
    data = demand_2029_reshaped[state] / 1000
    plt.plot(data)
    plt.ylabel("Demand(GW)")
    plt.xlabel("Day")
    plt.title('Yearly Load Curve of '+ state)
    plt.legend()
    plt.show()
    


#%% Load Duration Curve for top 10 states




import matplotlib.pyplot as plt
import numpy as np

sorted_data = pd.DataFrame(columns=states)

sorted_data = pd.DataFrame(data=[sorted(col, reverse=True) for _, col in demand_2029_reshaped.items()]).T

# set column names for sorted_data
sorted_data.columns = demand_2029_reshaped.columns


# Calculate the cumulative frequency of the load demand for each state
cumulative_freq = sorted_data.cumsum() / sorted_data.sum()

# Get the maximum demand value for each state
max_demand = demand_2029_reshaped.max()

load_factor = sorted_data.apply(lambda x: x / max_demand[x.name])


import numpy as np

for state in demand_2029_reshaped.columns:
    # Get the minimum and maximum load factor values for the state
    min_load_factor = load_factor[state].min() * 100
    max_load_factor = load_factor[state].max() * 100
    
    # Plot the load duration curve
    plt.plot(cumulative_freq[state]*100, load_factor[state]*100)
    plt.xlabel('Percentage of Time')
    plt.ylabel('Load Factor')
    plt.title('Load Duration Curve of ' + state)
    plt.legend()
    
    # Add grid lines only along the x-axis
    plt.grid(axis='y')
    
    # # Set x and y ticks with 5% intervals
    # x_step_size = 5
    # y_step_size = 5
    # x_ticks = np.arange(0, 101, x_step_size)
    # y_ticks = np.arange(min_load_factor, 100, y_step_size)
    # plt.xticks(x_ticks)
    # plt.yticks(y_ticks)
    
    plt.show()






#%% Rectification for TN, GJ, MH

demand_2029_peak.loc[:,state] = ( df_hourly_state_consumption_2019_20[state].values * peak_demand_matrix.loc['2029-2030',state] )/df_hourly_state_consumption_2021_22[state].max()


demand_2029_reshaped = pd.DataFrame(index=index,columns=states)




start_time = time.time()


state = 'Tamil Nadu'

demand_2029_peak.loc[:,state] = ( df_hourly_state_consumption_2021_22[state].values * peak_demand_matrix.loc['2029-2030',state] )/df_hourly_state_consumption_2021_22[state].max()




goal_step = 0.0001
goal_value = 0.2 
threshold = 1
print(state)
target_val = projected_demand_bau_matrix.loc['2029-2030',state]*1000
demand_2029_reshaped.loc[:,state] = (((demand_2029_peak.loc[:,state] - peak_demand_matrix.loc['2029-2030',state] )* goal_value) + peak_demand_matrix.loc['2029-2030',state])
while(demand_2029_reshaped[state].sum() - target_val > threshold):
     demand_2029_reshaped.loc[:,state] = (((demand_2029_peak.loc[:,state] - peak_demand_matrix.loc['2029-2030',state] )* goal_value) + peak_demand_matrix.loc['2029-2030',state])
     print(goal_value)
     goal_value = goal_value + goal_step





fig = go.Figure()
    
fig.add_trace(go.Scatter(x=demand_2029_reshaped.index, y=demand_2029_reshaped[state],
                    mode='lines',
                    name='Reshaped'))
fig.add_trace(go.Scatter(x=demand_2029_reshaped.index, y=demand_2029_peak[state],
                    mode='lines',
                    name='Peak'))

    
fig.add_trace(go.Scatter(x=demand_2029_reshaped.index, y=df_hourly_state_consumption_2019_20[state],
                    mode='lines',name='Base'))

fig.update_layout(title=state)

plot(fig, filename='G:/My Drive/Work/Vasudha/Demand_Projection/temp-plot.html')


plot(fig)














import matplotlib.pyplot as plt
import numpy as np

# Sort the demand data in descending order for each state
sorted_data = demand_2029_reshaped.apply(lambda x: -np.sort(-x))

# Calculate the cumulative frequency of the load demand for each state
cumulative_freq = sorted_data.cumsum() / sorted_data.sum()

# Plot the Load Duration Curve for each state
for state in demand_2029_reshaped.columns:
    plt.plot(cumulative_freq[state], label=state)

# Set the plot parameters
plt.xlabel('Percentage of Time')
plt.ylabel('Load Demand (MW)')
plt.title('Load Duration Curve')
plt.legend()
plt.grid()

plt.show()
#%% Verification


for state in states:
    verification.loc[state,'Peak Demand LF'] = (demand_2029_peak[state].mean()/demand_2029_peak[state].max())*100
    verification.loc[state,'Reshaped LF'] = (demand_2029_reshaped[state].mean()/demand_2029_reshaped[state].max())*100
    verification.loc[state,'Base Year LF'] = (df_hourly_state_consumption_2019_20[state].mean()/df_hourly_state_consumption_2019_20[state].max())*100
    verification.loc[state,'Projected Demand'] = projected_demand_bau_matrix.loc['2029-2030'][state]




verification.to_csv("check.csv")





demand_2029_reshaped = pd.DataFrame(index=index,columns=states)

state = 'Punjab'

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


def goal_seek(demand_2029_peak, peak_demand_matrix, target_val,state):
    def func(goal_seek_value):
        demand_2029_reshaped.loc[:,state] = ((demand_2029_peak.loc[:,state] - peak_demand_matrix.loc['2029-2030',state] )* goal_seek_value) + peak_demand_matrix.loc['2029-2030',state]
        return demand_2029_reshaped[state].sum()- target_val

    result = root_scalar(func, x0=0.1, x1=2 )
    if result.converged:
        return result.root
    else:
        raise ValueError("Goal seeking failed to converge.")














#%% India calculation


demand_2029_peak_exbus = pd.DataFrame(index=demand_2029_peak.index, columns=demand_2029_peak.columns)
demand_2029_reshaped_exbus= pd.DataFrame(index=demand_2029_reshaped.index, columns=demand_2029_reshaped.columns)
df_hourly_state_consumption_2021_22_exbus = pd.DataFrame(index=df_hourly_state_consumption_2021_22.index, columns=df_hourly_state_consumption_2021_22.columns)


for state in states:
    print(state)
    demand_2029_peak_exbus[state] = demand_2029_peak[state]/(1 - df_losses.loc[state][2029])
    demand_2029_reshaped_exbus[state] = demand_2029_reshaped[state]/(1 - df_losses.loc[state][2029])                                      
    df_hourly_state_consumption_2021_22_exbus[state] = df_hourly_state_consumption_2021_22[state]/(1 - df_losses.loc[state][2021])


india_2029_demand_peak = demand_2029_peak_exbus.sum(axis=1)
india_2029_demand_reshaped = demand_2029_reshaped_exbus.sum(axis=1)
india_base = df_hourly_state_consumption_2021_22_exbus.sum(axis=1)




fig = go.Figure()
  
fig.add_trace(go.Scatter(x=india_2029_demand_reshaped.index, y=india_2029_demand_reshaped,
                 mode='lines',
                 name='Reshaped'))
 
fig.add_trace(go.Scatter(x=india_2029_demand_reshaped.index, y=india_2029_demand_peak,
                 mode='lines',
                 name='Peak'))

 

fig.add_trace(go.Scatter(x=india_2029_demand_reshaped.index, y=india_base,
                 mode='lines',
                 name='Base'))
 
fig.update_layout(title=state)

plot(fig)




#%% India Peak Day 2029 vs India Peak Day 2030 Graph


india_2029_demand_reshaped = pd.DataFrame(data=india_2029_demand_reshaped,index=india_2029_demand_reshaped.index)
india_base = pd.DataFrame(data=india_base,index=india_base.index)







# Convert column to datetime format
india_2029_demand_reshaped.index = pd.to_datetime(india_2029_demand_reshaped.index)
india_base.index = pd.to_datetime(india_base.index)



# Extract year from datetime column
# india_2029_demand_reshaped['Calendar Year'] = india_2029_demand_reshaped['Datetime'].dt.year

india_2029_demand_reshaped = india_2029_demand_reshaped.rename(columns={0: 'Demand'})
india_base = india_base.rename(columns={0: 'Demand'})



max_demand_indexes_1 = india_2029_demand_reshaped['Demand'].idxmax()
max_demand_indexes_2 = india_base['Demand'].idxmax()


date_string_1 = max_demand_indexes_1.strftime('%Y-%m-%d')
date_string_2 = max_demand_indexes_2.strftime('%Y-%m-%d')


    
x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

y1 = (india_2029_demand_reshaped.loc[date_string_1,'Demand'])/1000
y2 = (india_base.loc[date_string_2,'Demand'])/1000

y1_max_idx = y1.argmax()
y2_max_idx = y2.argmax()


plt.plot(x, y1, label ='Peak Day of 2029')
plt.plot(x, y2, label ='Peak Day of 2019')


plt.annotate(f"Peak: {y1[y1_max_idx]:.2f} GW", xy=(y1_max_idx, y1[y1_max_idx]), xytext=(y1_max_idx + 1, y1[y1_max_idx] + 2), color='blue', fontsize=10)
plt.annotate(f"Peak: {y2[y2_max_idx]:.2f} GW", xy=(y2_max_idx, y2[y2_max_idx]), xytext=(y2_max_idx + 1, y2[y2_max_idx] + 2), color='orange', fontsize=10)



plt.ylabel("Demand(GW)")
plt.xlabel("Hour of Day")
plt.legend()
plt.title('Peak Day 2019 vs peak Day 2030')
plt.show()


#%% Adding Seasons for 2029

# Create a dictionary to map months to seasons
seasons = {
    1: 'Winter',
    2: 'Winter',
    3: 'Summer',
    4: 'Summer',
    5: 'Summer',
    6: 'Summer',
    7: 'Summer',
    8: 'Monsoon',
    9: 'Monsoon',
    10: 'Monsoon',
    11: 'Winter',
    12: 'Winter'
}

# Use apply method along with lambda function to create new 'Season' column
india_2029_demand_reshaped['Season'] = india_2029_demand_reshaped.index.month.map(lambda x: seasons[x])


# Group the data by 'Season' column and find the peak days within each group
summer_peak_day = india_2029_demand_reshaped[india_2029_demand_reshaped['Season'] == 'Summer']['Demand'].idxmax().strftime('%Y-%m-%d')
monsoon_peak_day = india_2029_demand_reshaped[india_2029_demand_reshaped['Season'] == 'Monsoon']['Demand'].idxmax().strftime('%Y-%m-%d')
winter_peak_day = india_2029_demand_reshaped[india_2029_demand_reshaped['Season'] == 'Winter']['Demand'].idxmax().strftime('%Y-%m-%d')


x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
y1 = (india_2029_demand_reshaped.loc[summer_peak_day,'Demand'])/1000
y2 = (india_2029_demand_reshaped.loc[monsoon_peak_day,'Demand'])/1000
y3 = (india_2029_demand_reshaped.loc[winter_peak_day,'Demand'])/1000


y1_max_idx = y1.argmax()
y2_max_idx = y2.argmax()
y3_max_idx = y3.argmax()



plt.plot(x, y1, label ='Peak Day of Summer')
plt.plot(x, y2, label ='Peak Day of Monsoon')
plt.plot(x, y3, label ='Peak Day of Winter')


plt.annotate(f"Peak: {y1[y1_max_idx]:.2f} GW", xy=(y1_max_idx, y1[y1_max_idx]), xytext=(y1_max_idx + 1, y1[y1_max_idx] + 2), color='blue', fontsize=10)
plt.annotate(f"Peak: {y2[y2_max_idx]:.2f} GW", xy=(y2_max_idx, y2[y2_max_idx]), xytext=(y2_max_idx + 1, y2[y2_max_idx] + 2), color='orange', fontsize=10)
plt.annotate(f"Peak: {y3[y3_max_idx]:.2f} GW", xy=(y3_max_idx, y3[y3_max_idx]), xytext=(y3_max_idx + 1, y3[y3_max_idx] + 2), color='green', fontsize=10)


plt.ylabel("Demand(GW)")
plt.xlabel("Hour of Day")
plt.legend()
plt.title('Seasonal Variations in Peak Day for 2029')
plt.show()



#%% India Base year with Seasonal Variation. Note: Repeated code, with same variables.



# Create a dictionary to map months to seasons
seasons = {
    1: 'Winter',
    2: 'Winter',
    3: 'Summer',
    4: 'Summer',
    5: 'Summer',
    6: 'Summer',
    7: 'Summer',
    8: 'Monsoon',
    9: 'Monsoon',
    10: 'Monsoon',
    11: 'Winter',
    12: 'Winter'
}

# Use apply method along with lambda function to create new 'Season' column
india_base['Season'] = india_base.index.month.map(lambda x: seasons[x])


# Group the data by 'Season' column and find the peak days within each group
summer_peak_day = india_base[india_base['Season'] == 'Summer']['Demand'].idxmax().strftime('%Y-%m-%d')
monsoon_peak_day = india_base[india_base['Season'] == 'Monsoon']['Demand'].idxmax().strftime('%Y-%m-%d')
winter_peak_day = india_base[india_base['Season'] == 'Winter']['Demand'].idxmax().strftime('%Y-%m-%d')


x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
y1 = (india_base.loc[summer_peak_day,'Demand'])/1000
y2 = (india_base.loc[monsoon_peak_day,'Demand'])/1000
y3 = (india_base.loc[winter_peak_day,'Demand'])/1000


y1_max_idx = y1.argmax()
y2_max_idx = y2.argmax()
y3_max_idx = y3.argmax()



plt.plot(x, y1, label ='Peak Day of Summer')
plt.plot(x, y2, label ='Peak Day of Monsoon')
plt.plot(x, y3, label ='Peak Day of Winter')


plt.annotate(f"Peak: {y1[y1_max_idx]:.2f} GW", xy=(y1_max_idx, y1[y1_max_idx]), xytext=(y1_max_idx + 1, y1[y1_max_idx] + 2), color='blue', fontsize=10)
plt.annotate(f"Peak: {y2[y2_max_idx]:.2f} GW", xy=(y2_max_idx, y2[y2_max_idx]), xytext=(y2_max_idx + 1, y2[y2_max_idx] + 2), color='orange', fontsize=10)
plt.annotate(f"Peak: {y3[y3_max_idx]:.2f} GW", xy=(y3_max_idx, y3[y3_max_idx]), xytext=(y3_max_idx + 1, y3[y3_max_idx] + 2), color='green', fontsize=10)


plt.ylabel("Demand(GW)")
plt.xlabel("Hour of Day")
plt.legend()
plt.title('Seasonal Variations in Peak Day for Base Year(2021)')
plt.show()


#%% Regional Calculation



state_to_region = {'Telangana':'SR', 'Andhra Pradesh':'SR','Assam':'NER', 'Bihar':'ER', 'Chhattisgarh':'WR',
            'Goa':'WR', 'Gujarat':'WR', 'Haryana':'NR', 'Himachal Pradesh':'NR',
           'Jammu and Kashmir':'NR', 'Jharkhand':'ER', 'Karnataka':'SR', 'Kerala':'SR',
           'Madhya Pradesh':'WR', 'Maharashtra':'WR', 'Manipur':'NER',
           'Punjab':'NR', 'Rajasthan':'NR',
           'Tamil Nadu':'SR', 'Tripura':'NER', 'Uttar Pradesh':'NR', 'Uttarakhand':'NR',
           'West Bengal':'ER', 'Odisha':'ER', 'Meghalaya':'NER',
           'Mizoram':'NER', 'Nagaland':'NER', 'Delhi':'NR'}

# Group the columns in demand_2029_reshaped_exbus by their regions
region_to_columns = {}
for state, region in state_to_region.items():
    if region not in region_to_columns:
        region_to_columns[region] = []
    region_to_columns[region].append(state)

# Compute the sum of each region's columns and create a new DataFrame
region_to_sum_2029 = {}
region_to_sum_base = {}

for region, columns in region_to_columns.items():
    region_to_sum_2029[region] = demand_2029_reshaped_exbus[columns].sum(axis=1)
    region_to_sum_base[region] = df_hourly_state_consumption_2021_22_exbus[columns].sum(axis=1)
    
regional_demand_2029 = pd.DataFrame(region_to_sum_2029)
#regional_demand_2029 = regional_demand_2029.iloc[:-1]

regional_demand_base = pd.DataFrame(region_to_sum_base)
#regional_demand_base = regional_demand_base.iloc[:-1]



plt.figure(figsize=(10, 6))

# Loop over the columns in the dataframe and plot them
for column in regional_demand_2029.columns:
    plt.plot(regional_demand_2029.index, (regional_demand_2029[column])/1000, label=column)

# Set the x-axis and y-axis labels and legend
plt.xlabel('Date')
plt.ylabel('Demand in GW')
plt.title('Regional Demand Profile for 2029')
plt.legend()

# Display the plot
plt.show()

##Base year graph

plt.figure(figsize=(10, 6))

# Loop over the columns in the dataframe and plot them
for column in regional_demand_base.columns:
    plt.plot(regional_demand_base.index, (regional_demand_base[column])/1000, label=column)

# Set the x-axis and y-axis labels and legend
plt.xlabel('Date')
plt.ylabel('Demand in GW')
plt.title('Regional Demand Profile for 2021')
plt.legend()

# Display the plot
plt.show()



#%% Peak Day plotting regionally

region_to_sum_all = {}

for region, columns in region_to_columns.items():
    region_to_sum_all[region] = df_hourly_state_demand[columns].sum(axis=1)
    
    
    
df_hourly_state_demand_regional = pd.DataFrame(region_to_sum_all)

#df_hourly_state_demand_regional['Calendar Year'] = pd.to_datetime(df_hourly_state_demand_regional.index, format='%Y-%m-%d %H:%M:%S').year

df_hourly_state_demand_regional.index = df_hourly_state_demand_regional.index.round("H")



for day in peak_days_str:
    data = df_hourly_state_demand_regional.loc[day[:10]] / 1000
    plt.plot(data, label=data.columns)
    plt.ylabel("Demand(GW)")
    plt.xlabel("Hour of Day")
    plt.title('Peak Day Regional Load Curve for  '+ day)
    plt.legend()
    plt.show()


  
#%% Veri

regional_projection = {}

for year in years_1:
    for region, columns in region_to_columns.items():
        regional_projection[region] = projected_demand_bau_matrix[columns].sum(axis=1)
        

regional_projection = pd.DataFrame(regional_projection)

projected_demand_bau_matrix.to_csv("G:\My Drive\Work\Vasudha\Demand_Projection\shub.csv")

regional_projection.to_csv("G:\My Drive\Work\Vasudha\Demand_Projection\ceri.csv")


india_2029_demand_reshaped




max_consumption_india = india_2029_demand_reshaped.max()    



df_max_india=pd.DataFrame()
df_max_india['Max']=max_consumption_india


#df_hourly_state_consumption_1=df_hourly_state_consumption.drop('Date', axis=1)

max_consumption_index_india = india_2029_demand_reshaped.astype(float).idxmax()
#df_hourly_state_consumption=df_hourly_state_consumption.astype(float)

df_max_india['Index']=max_consumption_index_india
df_max_india['Date'] = pd.to_datetime(df_max_india['Index']).dt.date



# df_hourly_state_consumption.loc['2029-06-14':'2029-06-14',state].plot(title=state) df_hourly_state_consumption.loc[df_max.loc[state]['Date']:df_max.loc[state]['Date'],state].plot(title=state)
# df_hourly_state_consumption[state][(df_hourly_state_consumption.index.day==df_max.loc[state].Date.day)&(df_hourly_state_consumption.index.year==df_max.loc[state].Date.year)&(df_hourly_state_consumption.index.month==df_max.loc[state].Date.month)].plot(title=state)
# plt.show()






















