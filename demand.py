
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


df_hourly_state_consumption['Calendar Year'] = df_hourly_state_consumption.index.year





india_hourly_demand = df_hourly_state_demand.sum(axis=1)

india_hourly_demand = india_hourly_demand.to_frame()
india_hourly_demand['Calendar Year'] = pd.to_datetime(india_hourly_demand.index, format='%Y-%m-%d %H:%M:%S').year

calendar_years = [2017, 2018, 2019,2020,2021,2022,2029]


years=df_hourly_state_consumption['Financial Year'].unique()



peak_demand_matrix=pd.DataFrame(index=calendar_years,columns=states)
avg_demand_matrix=pd.DataFrame(index=calendar_years,columns=states)

for state in states:
    for year in calendar_years:
        peak_demand_matrix.loc[year,state]=df_hourly_state_consumption[state][df_hourly_state_consumption['Calendar Year']==year].max()
        avg_demand_matrix.loc[year,state]=df_hourly_state_consumption[state][df_hourly_state_consumption['Calendar Year']==year].mean()

#%% India Peak Demand Growth

india_hourly_demand = india_hourly_demand.rename(columns={0: 'Data'})
india_peak_demand_matrix=pd.DataFrame(index=calendar_years,columns=['Peak Demand'])
      
  
for year in calendar_years[:-1]:
    india_peak_demand_matrix.loc[year,'Peak Demand']= india_hourly_demand.loc[india_hourly_demand['Calendar Year'] == year, 'Data'].max()       
  
import matplotlib.pyplot as plt

plt.plot(india_peak_demand_matrix.index, india_peak_demand_matrix['Peak Demand']/1000)
plt.title('Growth of Peak Demand')
plt.xlabel('Year')
plt.ylabel('Peak Demand (GW)')
plt.xticks(calendar_years[:-1])
plt.show()


import plotly.graph_objs as go

fig = go.Figure()

fig.add_trace(go.Scatter(x=calendar_years[:-1], y=india_peak_demand_matrix['Peak Demand']/1000,
                 mode='lines+markers',
                 name='Peak Demand',
                 marker=dict(size=7.5)))

annotations = []
for x, y in zip(calendar_years[:-1], india_peak_demand_matrix['Peak Demand']/1000):
    annotations.append(dict(x=x, y=y, text=f"{y:.0f} GW", showarrow=False, yshift=20))

fig.update_layout(title='Growth of Peak Demand', annotations=annotations)
fig.update_xaxes(title_text="Year")
fig.update_yaxes(title_text="Peak Demand (GW)")

fig.write_html('G:\My Drive\Work\Vasudha\Demand_Projection\graphs\india_peak_demand_growth.html')


#%%Finding Peak days for India

yearly_demand = india_hourly_demand.groupby(by=india_hourly_demand.index.year)


max_demand_indexes = yearly_demand.idxmax()



peak_demand_days_in = pd.Series(max_demand_indexes.dt.date, index=max_demand_indexes.dt.year)


peak_demand_days_in = peak_demand_days_in.rename_axis('Year')

peak_demand_days_in = peak_demand_days_in.to_frame()

peak_demand_days_in = peak_demand_days_in.rename(columns={'Data': 'Peak Demand Date'})

peak_days = peak_demand_days_in['Peak Demand Date'].tolist()

peak_days_str = [d.strftime('%Y-%m-%d %H:%M:%S') for d in peak_days]







import plotly.graph_objs as go
import pandas as pd

# Create a new dataframe with hourly demand data for all peak days
df = pd.concat([india_hourly_demand.loc[day[:10]].reset_index(drop=True) for day in peak_days_str], axis=1)
df.columns = [day[:10] for day in peak_days_str]


    
traces = []
annotations = []
for day in peak_days_str:
    trace = go.Scatter(
        x=df.index,
        y=df[day[:10]] / 1000, # Convert demand from MW to GW
        name=day[:10]
    )
    traces.append(trace)

    # Find the peak value and index for each day
    peak_value = max(trace.y)
    peak_index = trace.y.argmax()
    peak_hour = peak_index % 24
    peak_demand = round(peak_value, 2)
    peak_day = trace.name

    # Add annotation for each peak
    annotation = dict(
        x=peak_hour,
        y=peak_demand,
        xref='x',
        yref='y',
        text=f'Peak: {peak_demand} GW',
        showarrow=True,
        arrowhead=7,
        ax=0,
        ay=-40
    )
    annotations.append(annotation)

# Create the plot
layout = go.Layout(
    title='Peak Day Load Curves 2017-2022',
    xaxis=dict(
        title='Hour of Day',
        tickmode='linear',
        tick0=0,
        dtick=1,
        range=[0, 23]
    ),
    yaxis=dict(
        title='Demand (GW)'
    ),
    annotations=annotations
)
fig = go.Figure(data=traces, layout=layout)
fig.write_html('G:\My Drive\Work\Vasudha\Demand_Projection\graphs\peak_days_india_demand.html')



#%%


# load_factor_matrix = (avg_demand_matrix/peak_demand_matrix)*100




# s = pd.Series()
# # load_factor_matrix = load_factor_matrix.append(s,ignore_index=True)
# # load_factor_matrix.index = index_values



# peak_demand_matrix = peak_demand_matrix.append(s,ignore_index=True)
# peak_demand_matrix.index = index_values

cagr_upper_limit = 0.07
cagr_lower_limit = 0.04

projection_year = '2029'
n = 7
period = 5


df_cagr = pd.DataFrame(index=states,columns=['CAGR','Peak_Demand_2030'])


for state in states:
    last_value=peak_demand_matrix.loc[2022][state]  
    first_value=peak_demand_matrix.loc[2017][state] 
    cagr = (((last_value/first_value)**(1/period)-1 ))
    if cagr > cagr_upper_limit:
        cagr = cagr_upper_limit
    if cagr < cagr_lower_limit:
        cagr = cagr_lower_limit    
    df_cagr.loc[state,'CAGR'] = cagr*100
    verification.loc[state,'CAGR'] = cagr*100
    projected_lf = ((cagr + 1)**n)*last_value
    projected_peak_demand = ((cagr + 1)**n)*last_value
    peak_demand_matrix.loc[2029][state] = projected_peak_demand
    df_cagr.loc[state,'Peak_Demand_2030'] = projected_peak_demand 
    verification.loc[state,'Peak Demand'] = projected_peak_demand


state='Tamil Nadu'

last_value=peak_demand_matrix.loc[2022][state]  

cagr = 0.06  #CAGR manually calcualted
df_cagr.loc[state,'CAGR'] = cagr*100
verification.loc[state,'CAGR'] = cagr*100
projected_lf = ((cagr + 1)**n)*last_value
projected_peak_demand = ((cagr + 1)**n)*last_value
peak_demand_matrix.loc[2029][state] = projected_peak_demand
df_cagr.loc[state,'Peak_Demand_2030'] = projected_peak_demand 
verification.loc[state,'Peak Demand'] = projected_peak_demand









start_date = '2029-01-01 00:00:00'
end_date = '2029-12-31 23:00:00'
index = pd.date_range(start_date, end_date, freq='H')


demand_2029_peak = pd.DataFrame(index=index,columns=states)

df_hourly_state_consumption_2018 = df_hourly_state_consumption[df_hourly_state_consumption['Calendar Year']==2018]
df_hourly_state_consumption_2019 = df_hourly_state_consumption[df_hourly_state_consumption['Calendar Year']==2019]
df_hourly_state_consumption_2022 = df_hourly_state_consumption[df_hourly_state_consumption['Calendar Year']==2022]



# date_to_drop = '2020-02-29'

# mask = df_hourly_state_consumption_2019_20.index.date != pd.to_datetime(date_to_drop).date()

# # Use the mask to select the rows you want to keep
# df_hourly_state_consumption_2019_20 = df_hourly_state_consumption_2019_20[mask]


for state in states:
    demand_2029_peak.loc[:,state] = ( df_hourly_state_consumption_2022[state].values * peak_demand_matrix.loc[2029,state] )/df_hourly_state_consumption_2022[state].max()




#%% Correct Code

demand_2029_reshaped = pd.DataFrame(index=index,columns=states)




start_time = time.time()



for state in states:
    goal_step = 0.0001
    goal_value = 0.2 
    threshold = 1
    print(state)
    target_val = projected_demand_bau_matrix.loc['2029-2030',state]*1000
    demand_2029_reshaped.loc[:,state] = (((demand_2029_peak.loc[:,state] - peak_demand_matrix.loc[2029,state] )* goal_value) + peak_demand_matrix.loc[2029,state])
    while(demand_2029_reshaped[state].sum() - target_val > threshold):
        demand_2029_reshaped.loc[:,state] = (((demand_2029_peak.loc[:,state] - peak_demand_matrix.loc[2029,state] )* goal_value) + peak_demand_matrix.loc[2029,state])
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














#%% All India Demand Projection calculation


demand_2029_peak_exbus = pd.DataFrame(index=demand_2029_peak.index, columns=demand_2029_peak.columns)
demand_2029_reshaped_exbus= pd.DataFrame(index=demand_2029_reshaped.index, columns=demand_2029_reshaped.columns)
df_hourly_state_consumption_2022_exbus = pd.DataFrame(index=df_hourly_state_consumption_2022.index, columns=df_hourly_state_consumption_2022.columns)


for state in states:
    print(state)
    demand_2029_peak_exbus[state] = demand_2029_peak[state]/(1 - df_losses.loc[state][2029])
    demand_2029_reshaped_exbus[state] = demand_2029_reshaped[state]/(1 - df_losses.loc[state][2029])                                      
    df_hourly_state_consumption_2022_exbus[state] = df_hourly_state_consumption_2022[state]/(1 - df_losses.loc[state][2021])


india_2029_demand_peak = demand_2029_peak_exbus.sum(axis=1)
india_2029_demand_reshaped = demand_2029_reshaped_exbus.sum(axis=1)
#india_base = df_hourly_state_consumption_2022_exbus.sum(axis=1)
india_base = india_hourly_demand.loc['2022-01-01':'2022-12-31']
india_base.index = india_base.index.round("H")




fig = go.Figure()
  
fig.add_trace(go.Scatter(x=india_2029_demand_reshaped.index, y=((india_2029_demand_reshaped['Demand'])/1000),
                 mode='lines',
                 name='2029'))
 
# fig.add_trace(go.Scatter(x=india_2029_demand_reshaped.index, y=india_2029_demand_peak,
#                  mode='lines',
#                  name='Peak'))

 

fig.add_trace(go.Scatter(x=india_2029_demand_reshaped.index, y=(india_base['Demand'])/1000,
                 mode='lines',
                 name='2022'))

fig.update_xaxes(title_text="Days")

# Update the y-axis title
fig.update_yaxes(title_text="Demand(GW)")

 
fig.update_layout(title='India Demand Curve - 2029 vs 2022')


fig.write_html('G:/My Drive/Work/Vasudha/Demand_Projection/graphs/allindia_demand.html')




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


import plotly.graph_objects as go

date_string_1 = max_demand_indexes_1.strftime('%Y-%m-%d')
date_string_2 = max_demand_indexes_2.strftime('%Y-%m-%d')

x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

y1 = (india_2029_demand_reshaped.loc[date_string_1,'Demand'])/1000
y2 = (india_base.loc[date_string_2,'Demand'])/1000

y1_max_idx = y1.argmax()
y2_max_idx = y2.argmax()

fig = go.Figure()

fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='Peak Day Load Curve of 2029'))
fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='Peak Day Load Curve of 2022'))

fig.add_annotation(x=y1_max_idx, y=y1[y1_max_idx], text=f"Peak: {y1[y1_max_idx]:.2f} GW", showarrow=True, arrowhead=1, arrowcolor='blue')
fig.add_annotation(x=y2_max_idx, y=y2[y2_max_idx], text=f"Peak: {y2[y2_max_idx]:.2f} GW", showarrow=True, arrowhead=1, arrowcolor='orange')

fig.update_xaxes(title_text='Hour of Day')
fig.update_yaxes(title_text='Demand (GW)')

fig.update_layout(title='Peak Day 2022 vs Peak Day 2030', legend=dict(x=0.01, y=0.99))

fig.write_html('G:/My Drive/Work/Vasudha/Demand_Projection/graphs/all_india_peak_day_base_2029.html')




#%% Adding Seasons for 2029, 2029 seasonal calculation

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



fig = go.Figure()

fig.add_trace(go.Scatter(x=x, y=y1,
                    mode='lines',
                    name='Peak Day of Summer',
                    line=dict(color='orange')))
fig.add_trace(go.Scatter(x=x, y=y2,
                    mode='lines',
                    name='Peak Day of Monsoon',
                    line=dict(color='blue')))
fig.add_trace(go.Scatter(x=x, y=y3,
                    mode='lines',
                    name='Peak Day of Winter',
                    line=dict(color='green')))

fig.add_annotation(x=y1_max_idx, y=y1[y1_max_idx],
            text=f"Peak: {y1[y1_max_idx]:.2f} GW",
            showarrow=True,
            arrowhead=1,
            font=dict(color='orange', size=10),
            xshift=10)

fig.add_annotation(x=y2_max_idx, y=y2[y2_max_idx],
            text=f"Peak: {y2[y2_max_idx]:.2f} GW",
            showarrow=True,
            arrowhead=1,
            font=dict(color='blue', size=10),
            xshift=10)

fig.add_annotation(x=y3_max_idx, y=y3[y3_max_idx],
            text=f"Peak: {y3[y3_max_idx]:.2f} GW",
            showarrow=True,
            arrowhead=1,
            font=dict(color='green', size=10),
            xshift=10)

fig.update_layout(
    title='Seasonal Variations in Peak Day for 2029',
    xaxis_title='Hour of Day',
    yaxis_title='Demand (GW)',
    legend=dict(x=0.01, y=0.95),
)

fig.write_html('G:/My Drive/Work/Vasudha/Demand_Projection/graphs/seasonal_india_peak_day_2029.html')

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



fig = go.Figure()

fig.add_trace(go.Scatter(x=x, y=y1,
                    mode='lines',
                    name='Peak Day of Summer',
                    line=dict(color='orange')))
fig.add_trace(go.Scatter(x=x, y=y2,
                    mode='lines',
                    name='Peak Day of Monsoon',
                    line=dict(color='blue')))
fig.add_trace(go.Scatter(x=x, y=y3,
                    mode='lines',
                    name='Peak Day of Winter',
                    line=dict(color='green')))

fig.add_annotation(x=y1_max_idx, y=y1[y1_max_idx],
            text=f"Peak: {y1[y1_max_idx]:.2f} GW",
            showarrow=True,
            arrowhead=1,
            font=dict(color='orange', size=10),
            xshift=10)

fig.add_annotation(x=y2_max_idx, y=y2[y2_max_idx],
            text=f"Peak: {y2[y2_max_idx]:.2f} GW",
            showarrow=True,
            arrowhead=1,
            font=dict(color='blue', size=10),
            xshift=10)

fig.add_annotation(x=y3_max_idx, y=y3[y3_max_idx],
            text=f"Peak: {y3[y3_max_idx]:.2f} GW",
            showarrow=True,
            arrowhead=1,
            font=dict(color='green', size=10),
            xshift=10)

fig.update_layout(
    title='Seasonal Variations in Peak Day for Base Year(2022)',
    xaxis_title='Hour of Day',
    yaxis_title='Demand (GW)',
    legend=dict(x=0.01, y=0.95),
)

fig.write_html('G:/My Drive/Work/Vasudha/Demand_Projection/graphs/seasonal_india_peak_day_base.html')


#%% Regional Calculation and Graphs



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
    region_to_sum_base[region] = df_hourly_state_consumption_2022_exbus[columns].sum(axis=1)
    
regional_demand_2029 = pd.DataFrame(region_to_sum_2029)
#regional_demand_2029 = regional_demand_2029.iloc[:-1]

regional_demand_base = pd.DataFrame(region_to_sum_base)
#regional_demand_base = regional_demand_base.iloc[:-1]





##Plotting for base year old method, backup

# fig = go.Figure()

# for column in regional_demand_base.columns:
#     #plt.plot(regional_demand_base.index, (regional_demand_base[column])/1000, label=column)
#     fig.add_trace(go.Scatter(x=regional_demand_base.index, y=(regional_demand_base[column])/1000,
#                  mode='lines',
#                  name=column))
 
# fig.update_xaxes(title_text="Days")

# # Update the y-axis title
# fig.update_yaxes(title_text="Demand(GW)")

 
# fig.update_layout(title='India Regional Demand Distribution-2022')


# fig.write_html('G:/My Drive/Work/Vasudha/Demand_Projection/graphs/regional_demand_base.html')



import plotly.graph_objs as go

fig = go.Figure()

# Plot the bottommost line
fig.add_trace(go.Scatter(x=regional_demand_base.index, y=(regional_demand_base['NER'])/1000,
                         mode='lines', name='NER'))

# Plot the next line on top of the previous one
fig.add_trace(go.Scatter(x=regional_demand_base.index, y=(regional_demand_base['ER'])/1000,
                         fill='tonexty', mode='lines', name='ER'))

# Plot the next line on top of the previous one
fig.add_trace(go.Scatter(x=regional_demand_base.index, y=(regional_demand_base['SR'])/1000,
                         fill='tonexty', mode='lines', name='SR'))

fig.add_trace(go.Scatter(x=regional_demand_base.index, y=(regional_demand_base['WR'])/1000,
                         fill='tonexty', mode='lines', name='WR'))

fig.add_trace(go.Scatter(x=regional_demand_base.index, y=(regional_demand_base['NR'])/1000,
                         fill='tonexty', mode='lines', name='NR'))


fig.update_xaxes(title_text="Days")

# Update the y-axis title
fig.update_yaxes(title_text="Demand (GW)")

fig.update_layout(title='India Regional Demand Distribution - 2022')


fig.write_html('G:/My Drive/Work/Vasudha/Demand_Projection/graphs/regional_demand_base.html')



##Old plot for backup for 2029

# fig = go.Figure()

# for column in regional_demand_base.columns:
#     #plt.plot(regional_demand_base.index, (regional_demand_base[column])/1000, label=column)
#     fig.add_trace(go.Scatter(x=regional_demand_2029.index, y=(regional_demand_2029[column])/1000,
#                  mode='lines',
#                  name=column))
 
# fig.update_xaxes(title_text="Days")

# # Update the y-axis title
# fig.update_yaxes(title_text="Demand(GW)")

 
# fig.update_layout(title='India Regional Demand Distribution-2029')


# fig.write_html('G:/My Drive/Work/Vasudha/Demand_Projection/graphs/regional_demand_2029.html')

import plotly.graph_objs as go

fig = go.Figure()

# Plot the bottommost line
fig.add_trace(go.Scatter(x=regional_demand_2029.index, y=(regional_demand_2029['NER'])/1000,
                         mode='lines', name='NER'))

# Plot the next line on top of the previous one
fig.add_trace(go.Scatter(x=regional_demand_2029.index, y=(regional_demand_2029['ER'])/1000,
                         fill='tonexty', mode='lines', name='ER'))

# Plot the next line on top of the previous one
fig.add_trace(go.Scatter(x=regional_demand_2029.index, y=(regional_demand_2029['SR'])/1000,
                         fill='tonexty', mode='lines', name='SR'))

fig.add_trace(go.Scatter(x=regional_demand_2029.index, y=(regional_demand_2029['WR'])/1000,
                         fill='tonexty', mode='lines', name='WR'))

fig.add_trace(go.Scatter(x=regional_demand_2029.index, y=(regional_demand_2029['NR'])/1000,
                         fill='tonexty', mode='lines', name='NR'))


fig.update_xaxes(title_text="Days")

# Update the y-axis title
fig.update_yaxes(title_text="Demand (GW)")

fig.update_layout(title='India Regional Demand Distribution - 2029')


fig.write_html('G:/My Drive/Work/Vasudha/Demand_Projection/graphs/regional_demand_2029.html')




#%% Peak Day plotting regionally
import plotly.graph_objs as go

region_to_sum_all = {}

for region, columns in region_to_columns.items():
    region_to_sum_all[region] = df_hourly_state_demand[columns].sum(axis=1)

df_hourly_state_demand_regional = pd.DataFrame(region_to_sum_all)
df_hourly_state_demand_regional.index = df_hourly_state_demand_regional.index.round("H")

for day in peak_days_str:
    data = df_hourly_state_demand_regional.loc[day[:10]] / 1000
    traces = []
    for column in data.columns:
        trace = go.Scatter(
            x=data.index,
            y=data[column],
            name=column
        )
        traces.append(trace)
    layout = go.Layout(
        title=f"Peak Day Regional Load Curve for {day[:10]}",
        xaxis=dict(
            title="Hour of Day"
        ),
        yaxis=dict(
            title="Demand (GW)"
        )
    )
    fig = go.Figure(data=traces, layout=layout)
    filename = f"G:/My Drive/Work/Vasudha/Demand_Projection/graphs/regional_peak_days_{day[:4]}.html"
    fig.write_html(filename)
    
    
##Side by side graph experiment

import plotly.subplots as sp

region_to_sum_all = {}

for region, columns in region_to_columns.items():
    region_to_sum_all[region] = df_hourly_state_demand[columns].sum(axis=1)

df_hourly_state_demand_regional = pd.DataFrame(region_to_sum_all)
df_hourly_state_demand_regional.index = df_hourly_state_demand_regional.index.round("H")

subplot_titles = [day[:4] for day in peak_days_str]  # Extract date part of title

fig = sp.make_subplots(rows=3, cols=2, subplot_titles=subplot_titles)
legend_added = False  # Flag to track whether legend has been added

for i, day in enumerate(peak_days_str):
    row = (i // 2) + 1
    col = (i % 2) + 1
    data = df_hourly_state_demand_regional.loc[day[:10]] / 1000
    for column in data.columns:
        trace = go.Scatter(
            x=data.index,
            y=data[column],
            name=column,
            showlegend=not legend_added  # Only show legend if not already added
        )
        fig.add_trace(trace, row=row, col=col)
    

    fig.update_xaxes(title_text="Hour of Day", row=row, col=col)
    fig.update_yaxes(title_text="Demand (GW)", row=row, col=col)
    if not legend_added:  # Add legend if not already added
        fig.update_layout(title={"text": "Peak Day Regional Load Curves",
                                  "y": 0.95,
                                  "x": 0.5,
                                  "xanchor": "center",
                                  "yanchor": "top"},
                           showlegend=True)
        legend_added = True  # Set flag to True after adding legend
        
    
        

filename = "G:/My Drive/Work/Vasudha/Demand_Projection/graphs/regional_peak_days.html"
fig.write_html(filename)

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



#%% Min/Max/Avg Graphs Yearly


# Compute the min, max, and mean demand values for each calendar year
india_demand_stats = india_hourly_demand.groupby('Calendar year')['Data'].agg(['min', 'max', 'mean']).round(0)

# Set the calendar year as the index of the resulting DataFrame
india_demand_stats.index = pd.to_datetime(india_demand_stats.index, format='%Y').strftime('%Y')

# Rename the columns to be more descriptive
india_demand_stats.columns = ['Min demand', 'Min demand', 'Min demand']



#Line Graph for showing stats
# Create a new figure
fig = go.Figure()

# Add the min, max, and avg demand curves to the figure
fig.add_trace(go.Scatter(x=india_demand_stats.index, y=india_demand_stats['Min demand']/1000, name='Min demand'))
fig.add_trace(go.Scatter(x=india_demand_stats.index, y=india_demand_stats['Max demand']/1000, name='Max demand'))
fig.add_trace(go.Scatter(x=india_demand_stats.index, y=india_demand_stats['Avg demand']/1000, name='Avg demand'))

# Update the x-axis and y-axis labels
fig.update_xaxes(title_text='Year')
fig.update_yaxes(title_text='Demand (GW)')

# Add a title to the figure
fig.update_layout(title_text='Demand Statistics by Year')

filename = "G:/My Drive/Work/Vasudha/Demand_Projection/graphs/demand_stats_line.html"
fig.write_html(filename)


#Grouped Bar chart for showing stats


years = [2017, 2018, 2019, 2020, 2021, 2022]

# Create a bar chart with the data
fig = go.Figure()
fig.add_trace(go.Bar(x=years, y=india_demand_stats['Min demand']/1000, name='Min demand', text=india_demand_stats['Min demand']/1000, textposition='outside'))
fig.add_trace(go.Bar(x=years, y=india_demand_stats['Max demand']/1000, name='Max demand', text=india_demand_stats['Max demand']/1000, textposition='outside'))
fig.add_trace(go.Bar(x=years, y=india_demand_stats['Avg demand']/1000, name='Avg demand', text=india_demand_stats['Avg demand']/1000, textposition='outside'))

# Round the text values to the nearest integer
for trace in fig.data:
    trace.text = [round(t) for t in trace.text]

# Create a line chart for each data series
fig.add_trace(go.Scatter(x=years, y=india_demand_stats['Min demand']/1000, name='Min demand',mode='lines', line=dict(dash='dash')))
fig.add_trace(go.Scatter(x=years, y=india_demand_stats['Max demand']/1000, name='Max demand',mode='lines', line=dict(dash='dash')))
fig.add_trace(go.Scatter(x=years, y=india_demand_stats['Avg demand']/1000, name='Avg demand', mode='lines', line=dict(dash='dash')))


bar_width = 0.4

# Shift the Min demand line graph to the left by one bar width
fig.update_traces(x=[year - bar_width/2 for year in years], selector=dict(name='Min demand'))

# Shift the Avg demand line graph to the right by one bar width
fig.update_traces(x=[year + bar_width/2 for year in years], selector=dict(name='Avg demand'))


# Update the layout of the chart
fig.update_layout(barmode='group', title='Demand Statistics by Year in India (2017-2022)')

fig.update_xaxes(title_text='Year')
fig.update_yaxes(title_text='Demand (GW)')

# Show the chart
fig.show()

india_hourly_demand.to_csv("G:/My Drive/Work/Vasudha/Demand_Projection/india_hourly.csv")



#%% Min/Max/Avg graphs monthly





import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objs as go

# Group the dataframe by week and calculate the min, avg, and max demand
india_weekly_demand = india_hourly_demand.groupby(pd.Grouper(freq='W'))['Data'].agg(['min', 'mean', 'max'])

# Create a line chart with Plotly
fig = px.line(india_weekly_demand/1000, x=india_weekly_demand.index, y=['min', 'mean', 'max'], 
              title='Weekly Min, Avg, and Max Demand in India')

# Add linear trend lines to the chart
for i, y_col in enumerate(['min', 'mean', 'max']):
    x = india_weekly_demand.index.astype(np.int64) // 10**9
    y = india_weekly_demand[y_col] / 1000
    m, b = np.polyfit(x, y, 1)
    trendline = m * x + b
    line_color = fig.data[i].line.color
    fig.add_trace(
        go.Scatter(
            x=india_weekly_demand.index,
            y=trendline,
            name=f'{y_col} Trendline',
            line=dict(dash='dash', color=line_color)
        )
    )

# Update the chart layout with axes titles and no legend title
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Demand (GW)',
    legend_title='',
    template='plotly_white'
)

filename = "G:/My Drive/Work/Vasudha/Demand_Projection/graphs/india_min_max_avg.html"
fig.write_html(filename)








#%%#%% Top 10 States



# extract data for year 2029-2030
df_2029_2030_peak = peak_demand_matrix.loc[2029]

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




import plotly.express as px

sorted_data = pd.DataFrame(data=[sorted(col, reverse=True) for _, col in demand_2029_reshaped_exbus.items()]).T

# set column names for sorted_data
sorted_data.columns = demand_2029_reshaped_exbus.columns

# Calculate the cumulative frequency of the load demand for each state
cumulative_freq = sorted_data.cumsum() / sorted_data.sum()

for state in top_10_states:
    # Plot the load duration curve
    fig = px.line(x=cumulative_freq[state]*100, y=sorted_data[state]/1000, title=f"Load Duration Curve of {state}")
    fig.update_layout(xaxis_title="Percentage of Time", yaxis_title="Load(GW)")
    filename = f"G:/My Drive/Work/Vasudha/Demand_Projection/graphs/{state}_load_duration.html"
    fig.write_html(filename)


    
#%% Annexure data for top 10 states


import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Create a list to store the subplots for each state
subplots = []

for state in top_10_states:
    # Peak day load curve
    peak_day = peak_days_states[state]
    peak_day_dt = datetime.datetime.combine(peak_day, datetime.time())
    peak_day_str = peak_day_dt.strftime("%Y-%m-%d")
    data = demand_2029_reshaped.loc[peak_day_str, state] / 1000
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data.index, y=data, mode='lines', name='Peak Day Load Curve'))
    fig1.update_layout(title='2029 Peak Day Load Curve of ' + state + ' for ' + peak_day_str, xaxis_title='Hour of Day', yaxis_title='Demand (GW)', legend_title='')

    # Load duration curve
    sorted_data = pd.DataFrame(data=[sorted(col, reverse=True) for _, col in demand_2029_reshaped_exbus.items()]).T
    sorted_data.columns = demand_2029_reshaped_exbus.columns
    cumulative_freq = sorted_data.cumsum() / sorted_data.sum()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=cumulative_freq[state]*100, y=sorted_data[state]/1000, mode='lines', name='Load Duration Curve'))
    fig2.update_layout(title='Load Duration Curve of ' + state, xaxis_title='Percentage of Time', yaxis_title='Load (GW)', legend_title='')

    # Yearly load curve
    data = demand_2029_reshaped[state] / 1000
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=data.index, y=data, mode='lines', name='Yearly Load Curve'))
    fig3.update_layout(title='Yearly Load Curve of ' + state, xaxis_title='Day', yaxis_title='Demand (GW)', legend_title='')

   # Combine the subplots for the current state
    fig_combined = sp.make_subplots(rows=2, cols=2, subplot_titles=[f"Peak Day Load Curve of {state} for {peak_day_str}", f"Load Duration Curve of {state}", f"Yearly Load Curve of {state}"])

# Add the traces to the subplot
    fig_combined.add_trace(fig1.data[0], row=1, col=1)
    fig_combined.add_trace(fig2.data[0], row=1, col=2)
    fig_combined.add_trace(fig3.data[0], row=2, col=1, colspan=2)

# Update the layout of the subplot to make fig3 wide
    fig_combined.update_layout(height=800, width=1000, showlegend=False, 
                           template="plotly_white", margin=dict(l=20, r=20, t=60, b=20), 
                           subplot_titles=dict(x=0.5, y=0.93, font=dict(size=18)))
    fig_combined.update_xaxes(title_text="Hour of Day", row=1, col=1)
    fig_combined.update_yaxes(title_text="Demand (GW)", row=1, col=1)
    fig_combined.update_xaxes(title_text="Percentage of Time", row=1, col=2)
    fig_combined.update_yaxes(title_text="Load (GW)", row=1, col=2)
    fig_combined.update_xaxes(title_text="Day", row=2, col=1)
    fig_combined.update_yaxes(title_text="Demand (GW)", row=2, col=1)

# Write the subplots for each state to separate HTML files
for i in range(len(top_10_states)):
    filename = f"G:/My Drive/Work/Vasudha/Demand_Projection/graphs/top10/{top_10_states[i]}_combined.html"

    subplots[i].write_html(filename)


import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Create a list to store the subplots for each state
subplots = []

for state in top_10_states:
    # Peak day load curve
    peak_day = peak_days_states[state]
    peak_day_dt = datetime.datetime.combine(peak_day, datetime.time())
    peak_day_str = peak_day_dt.strftime("%Y-%m-%d")
    data = demand_2029_reshaped.loc[peak_day_str, state] / 1000
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data.index, y=data, mode='lines', name='Peak Day Load Curve'))
    fig1.update_layout(title='2029 Peak Day Load Curve for ' + peak_day_str, xaxis_title='Hour of Day', yaxis_title='Demand (GW)', legend_title='')
    fig1.update
    # Load duration curve
    sorted_data = pd.DataFrame(data=[sorted(col, reverse=True) for _, col in demand_2029_reshaped_exbus.items()]).T
    sorted_data.columns = demand_2029_reshaped_exbus.columns
    cumulative_freq = sorted_data.cumsum() / sorted_data.sum()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=cumulative_freq[state]*100, y=sorted_data[state]/1000, mode='lines', name='Load Duration Curve'))
    fig2.update_layout(title='Load Duration Curve', xaxis_title='Percentage of Time', yaxis_title='Load (GW)', legend_title='')

    # Yearly load curve
    data = demand_2029_reshaped[state] / 1000
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=data.index, y=data, mode='lines', name='Yearly Load Curve'))
    fig3.update_layout(title='2029 Yearly Load Curve', xaxis_title='Day', yaxis_title='Demand (GW)', legend_title='')

    # Combine the subplots for the current state
    fig_combined = make_subplots(rows=2, cols=2, specs=[[{}, {}], [{"colspan": 2}, None]], subplot_titles=[f"Peak Day Load Curve of {state} for {peak_day_str}", f"Load Duration Curve of {state}", f"Yearly Load Curve of {state}"])
    fig_combined.add_trace(fig1.data[0], row=1, col=1)
    fig_combined.add_trace(fig2.data[0], row=1, col=2)
    fig_combined.add_trace(fig3.data[0], row=2, col=1)
    fig_combined.update_layout(title={
        'text': state,
        'x': 0.5, # center title horizontally
        'y': 0.95 # position title slightly above the top of the plot
    },height=800, width=1000, showlegend=False)
    # Update the subplot axes titles
    fig_combined.update_xaxes(title_text='Hour of Day', row=1, col=1)
    fig_combined.update_yaxes(title_text='Demand (GW)', row=1, col=1)
    
    fig_combined.update_xaxes(title_text='Percentage of Time', row=1, col=2)
    fig_combined.update_yaxes(title_text='Load (GW)', row=1, col=2)

    fig_combined.update_xaxes(title_text='Day', row=2, col=1)
    fig_combined.update_yaxes(title_text='Demand (GW)', row=2, col=1)

    
    
    # Add the subplot to the list
    subplots.append(fig_combined)

# Write the subplots for each state to separate HTML files
for i in range(len(top_10_states)):
    filename = f"G:/My Drive/Work/Vasudha/Demand_Projection/graphs/top10/{top_10_states[i]}_combined.html"
    subplots[i].write_html(filename)




#%% Correlation Graph

import plotly.graph_objects as go
import pandas as pd
import numpy as np



# Load data
india_correlation = pd.DataFrame(index=[2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022], 
                                  columns=['Peak Demand','Demand'])

#Added values from CEA historic data


cea_data = {
    2004: [77.652, 548.115],
    2005: [81.792, 578.819],
    2006: [86.818, 624.495],
    2007: [90.793, 666.007],
    2008: [96.785, 691.038],
    2009: [104.009, 746.644],
    2010: [110.256, 788.355],
    2011: [116.191, 857.886],
    2012: [123.294, 908.652],
    2013: [129.815, 959.829],
    2014: [141.16, 1030.785],
    2015: [148.463, 1090.85],
    2016: [156.934, 1135.332]
}

# Load data
india_correlation = pd.DataFrame(index=cea_data.keys(), columns=['Peak Demand','Demand'])

# Added values from the dictionary
for year in cea_data.keys():
    india_correlation.loc[year] = cea_data[year]


years = [2017,2018,2019,2020,2021,2022]

for year in years:
    demand = india_hourly_demand['Data'].loc[india_hourly_demand['Calendar Year'] == year].sum() / 1000000
    india_correlation.loc[year, 'Demand'] = demand
    india_correlation.loc[year,'Peak Demand'] = india_peak_demand_matrix.loc[year,'Peak Demand']/1000


# Convert columns to numeric
india_correlation['Demand'] = pd.to_numeric(india_correlation['Demand'])
india_correlation['Peak Demand'] = pd.to_numeric(india_correlation['Peak Demand'])

# Convert columns to numpy arrays
peak_demand = india_correlation['Peak Demand'].to_numpy()
demand = india_correlation['Demand'].to_numpy()
years = india_correlation.index.tolist()

# Calculate correlation
corr = np.corrcoef(peak_demand,demand)[0, 1]

# Calculate the trendline equation
z = np.polyfit(demand, peak_demand, 1)
p = np.poly1d(z)
trendline_x = np.linspace(demand.min(), demand.max(), 100)
trendline_y = p(trendline_x)

# Create a scatter plot of 'demand' vs 'peak demand'
fig = go.Figure()
fig.add_trace(go.Scatter(x=demand, y=peak_demand, mode='markers+text', 
                         text=years, textposition='top center',
                         marker=dict(size=10, color='red', opacity=0.5)))

# Add the trendline equation to the plot
fig.add_trace(go.Scatter(x=trendline_x, y=trendline_y, mode='lines', name='Trendline', 
                         line=dict(dash='dash')))

# Add titles to the axes
fig.update_layout(title='Correlation between Peak Demand and Demand', 
                  xaxis_title='Demand(TWh)', yaxis_title='Peak Demand(GW)', showlegend=False, template='plotly_white')




# Add a correlation coefficient annotation to the plot
fig.add_annotation(x=0.5, y=0.9, text=f'Correlation Coefficient: {corr:.3f}', showarrow=False, 
                   xref='paper', yref='paper', align='center')



# Save the plot to an HTML file
filename = "G:/My Drive/Work/Vasudha/Demand_Projection/graphs/india_correlation.html"
fig.write_html(filename)










