# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 20:45:44 2023

@author: Nancssee
"""

# Purpose: Wind turbine failure prediction. 
#          (To know under what conditions wind turbine will fail.)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/Users/hanna/Desktop/Nancssee/Courses/Diploma In Practical Data Analytics/Project 2-Wind Turbine Failure/Project Templates_Project 2/EDA AND PREPROCESSING USING PYTHON/EDA AND PREPROCESSING USING PYTHON/cleaned_data.csv')
exclude_column = ['Unnamed: 0']
data = df.drop(columns=exclude_column)

data['Wind_speed']= abs(data['Wind_speed'])
data['Power']= abs(data['Power'])
data['Rotor_Speed']= abs(data['Rotor_Speed'])
data['Generator_speed']= abs(data['Generator_speed'])
data['Yaw_angle']= abs(data['Yaw_angle'])
data['Wind_direction']= abs(data['Wind_direction'])

summary= data.describe()  
data.info()  

"""(A) Exploratory data analysis (EDA)"""
'''1) 1st moment business decision'''
'''Mean'''
data.Wind_speed.mean() 
data.Power.mean() 
data.Nacelle_ambient_temperature.mean() 
data.Generator_bearing_temperature.mean() 
data.Gear_oil_temperature.mean() 
data.Ambient_temperature.mean() 
data.Rotor_Speed.mean() 
data.Nacelle_temperature.mean() 
data.Generator_speed.mean() 
data.Yaw_angle.mean() 
data.Wind_direction.mean()
data.Gear_box_inlet_temperature.mean()
data.Bearing_temperature.mean() 
data.Wheel_hub_temperature.mean()

'''Median'''
data.Wind_speed.median() 
data.Power.median() 
data.Nacelle_ambient_temperature.median() 
data.Generator_bearing_temperature.median() 
data.Gear_oil_temperature.median() 
data.Ambient_temperature.median() 
data.Rotor_Speed.median() 
data.Nacelle_temperature.median() 
data.Generator_speed.median() 
data.Yaw_angle.median()
data.Wind_direction.median() 
data.Gear_box_inlet_temperature.median()
data.Bearing_temperature.median() 
data.Wheel_hub_temperature.median()

'''Mode'''
data.Wind_speed.mode() 
data.Power.mode() 
data.Nacelle_ambient_temperature.mode() 
data.Generator_bearing_temperature.mode() 
data.Gear_oil_temperature.mode()
data.Ambient_temperature.mode()
data.Rotor_Speed.mode() 
data.Nacelle_temperature.mode() 
data.Generator_speed.mode()
data.Yaw_angle.mode() 
data.Wind_direction.mode() 
data.Gear_box_inlet_temperature.mode() 
data.Bearing_temperature.mode() 
data.Wheel_hub_temperature.mode() 
data.Failure_status.mode()

'''Count & frequency (in %) for "Failure_status"'''
data.Failure_status.value_counts() 
data.Failure_status.value_counts(normalize=True)*100 

'''Auto EDA'''
import sweetviz as sv
s = sv.analyze(data)
s.show_html()

import dtale
d = dtale.show(data)
d.open_browser()

# From Sweetviz:
# 1) Strongly related to 'Failure_status': 
#    Gear_oil_temperature, Yaw_angle, Gear_box_inlet_temperature.

# 2) Moderately related to 'Failure_status': 
#    Wind_direction, Nacelle_ambient_temperature, Ambient_temperature.

# 3) Mildly related to 'Failure_status': 
#    Wind_speed, Rotor_Speed.

'''2) 2nd moment business decision'''
'''Variance'''
data.Wind_speed.var() 
data.Power.var() 
data.Nacelle_ambient_temperature.var() 
data.Generator_bearing_temperature.var() 
data.Gear_oil_temperature.var() 
data.Ambient_temperature.var()
data.Rotor_Speed.var() 
data.Nacelle_temperature.var() 
data.Generator_speed.var() 
data.Yaw_angle.var() 
data.Wind_direction.var() 
data.Gear_box_inlet_temperature.var() 
data.Bearing_temperature.var() 
data.Wheel_hub_temperature.var() 

'''Standard Deviation'''
data.Wind_speed.std() 
data.Power.std() 
data.Nacelle_ambient_temperature.std() 
data.Generator_bearing_temperature.std() 
data.Gear_oil_temperature.std() 
data.Ambient_temperature.std() 
data.Rotor_Speed.std() 
data.Nacelle_temperature.std()
data.Generator_speed.std() 
data.Yaw_angle.std() 
data.Wind_direction.std() 
data.Gear_box_inlet_temperature.std() 
data.Bearing_temperature.std()
data.Wheel_hub_temperature.std() 

'''Maximum value'''
data.Wind_speed.max() 
data.Power.max() 
data.Nacelle_ambient_temperature.max() 
data.Generator_bearing_temperature.max() 
data.Gear_oil_temperature.max() 
data.Ambient_temperature.max() 
data.Rotor_Speed.max() 
data.Nacelle_temperature.max() 
data.Generator_speed.max() 
data.Yaw_angle.max() 
data.Wind_direction.max() 
data.Gear_box_inlet_temperature.max() 
data.Bearing_temperature.max() 
data.Wheel_hub_temperature.max() 

'''Minimum value'''
data.Wind_speed.min() 
data.Power.min()
data.Nacelle_ambient_temperature.min() 
data.Generator_bearing_temperature.min() 
data.Gear_oil_temperature.min() 
data.Ambient_temperature.min() 
data.Rotor_Speed.min() 
data.Nacelle_temperature.min() 
data.Generator_speed.min() 
data.Yaw_angle.min()
data.Wind_direction.min() 
data.Gear_box_inlet_temperature.min() 
data.Bearing_temperature.min() 
data.Wheel_hub_temperature.min()

'''Range'''
max(data.Wind_speed)-min(data.Wind_speed) 
max(data.Power)-min(data.Power) 
max(data.Nacelle_ambient_temperature)-min(data.Nacelle_ambient_temperature) 
max(data.Generator_bearing_temperature)-min(data.Generator_bearing_temperature) 
max(data.Gear_oil_temperature)-min(data.Gear_oil_temperature) 
max(data.Ambient_temperature)-min(data.Ambient_temperature) 
max(data.Rotor_Speed)-min(data.Rotor_Speed) 
max(data.Nacelle_temperature)-min(data.Nacelle_temperature) 
max(data.Generator_speed)-min(data.Generator_speed) 
max(data.Yaw_angle)-min(data.Yaw_angle) 
max(data.Wind_direction)-min(data.Wind_direction) 
max(data.Gear_box_inlet_temperature)-min(data.Gear_box_inlet_temperature) 
max(data.Bearing_temperature)-min(data.Bearing_temperature) 
max(data.Wheel_hub_temperature)-min(data.Wheel_hub_temperature) 
# Range different quite a lot among features.

'''3) 3rd moment business decision'''
'''Skewness'''
data.Wind_speed.skew() 
data.Power.skew() 
data.Nacelle_ambient_temperature.skew() 
data.Generator_bearing_temperature.skew() 
data.Gear_oil_temperature.skew() 
data.Ambient_temperature.skew() 
data.Rotor_Speed.skew() 
data.Nacelle_temperature.skew() 
data.Generator_speed.skew() 
data.Yaw_angle.skew() 
data.Wind_direction.skew() 
data.Gear_box_inlet_temperature.skew() 
data.Bearing_temperature.skew() 
data.Wheel_hub_temperature.skew() 

'''4) 4th moment business decision'''
'''Kurtosis'''
data.Wind_speed.kurt() 
data.Power.kurt() 
data.Nacelle_ambient_temperature.kurt() 
data.Generator_bearing_temperature.kurt() 
data.Gear_oil_temperature.kurt() 
data.Ambient_temperature.kurt() 
data.Rotor_Speed.kurt() 
data.Nacelle_temperature.kurt() 
data.Generator_speed.kurt() 
data.Yaw_angle.kurt() 
data.Wind_direction.kurt() 
data.Gear_box_inlet_temperature.kurt() 
data.Bearing_temperature.kurt() 
data.Wheel_hub_temperature.kurt() 

'''5) Data Visualization'''
'''Boxplots'''
sns.boxplot(x=data.Wind_speed) #No outlier.
sns.boxplot(x=data.Power) #No outlier. 
sns.boxplot(x=data.Nacelle_ambient_temperature) #No outlier.
sns.boxplot(x=data.Generator_bearing_temperature) #2 outliers in both side.
sns.boxplot(x=data.Gear_oil_temperature) #1 outlier in right side.
sns.boxplot(x=data.Ambient_temperature) #No outlier.
sns.boxplot(x=data.Rotor_Speed) #No outlier.
sns.boxplot(x=data.Nacelle_temperature) #2 outliers at both side.
sns.boxplot(x=data.Generator_speed) #1 outlier in left side.
sns.boxplot(x=data.Yaw_angle) #No outlier.
sns.boxplot(x=data.Wind_direction) #1 outlier at right.
sns.boxplot(x=data.Gear_box_inlet_temperature) #1 outlier in right side.
sns.boxplot(x=data.Bearing_temperature) #98 outliers at both sides.
sns.boxplot(x=data.Wheel_hub_temperature) #No outlier.

'''Histograms & density plots'''
plt.hist(data.Wind_speed, edgecolor='black')
plt.title('Wind_speed') 
sns.kdeplot(data.Wind_speed, fill = True)

plt.hist(data.Power, edgecolor='black')
plt.title('Power') 
sns.kdeplot(data.Power, fill = True)

plt.hist(data.Nacelle_ambient_temperature, edgecolor='black')
plt.title('Nacelle_ambient_temperature') 
sns.kdeplot(data.Nacelle_ambient_temperature, fill = True)

plt.hist(data.Generator_bearing_temperature, edgecolor='black')
plt.title('Generator_bearing_temperature') 
sns.kdeplot(data.Generator_bearing_temperature, fill = True)

plt.hist(data.Gear_oil_temperature, edgecolor='black')
plt.title('Gear_oil_temperature') 
sns.kdeplot(data.Gear_oil_temperature, fill = True)

plt.hist(data.Ambient_temperature, edgecolor='black')
plt.title('Ambient_temperature') 
sns.kdeplot(data.Ambient_temperature, fill = True)

plt.hist(data.Rotor_Speed, edgecolor='black')
plt.title('Rotor_Speed') 
sns.kdeplot(data.Rotor_Speed, fill = True)

plt.hist(data.Nacelle_temperature, edgecolor='black')
plt.title('Nacelle_temperature') 
sns.kdeplot(data.Nacelle_temperature, fill = True)

plt.hist(data.Generator_speed, edgecolor='black')
plt.title('Generator_speed') 
sns.kdeplot(data.Generator_speed, fill = True)

plt.hist(data.Yaw_angle, edgecolor='black')
plt.title('Yaw_angle') 
sns.kdeplot(data.Yaw_angle, fill = True)

plt.hist(data.Wind_direction, edgecolor='black')
plt.title('Wind_direction') 
sns.kdeplot(data.Wind_direction, fill = True)

plt.hist(data.Gear_box_inlet_temperature, edgecolor='black')
plt.title('Gear_box_inlet_temperature') 
sns.kdeplot(data.Gear_box_inlet_temperature, fill = True)

plt.hist(data.Bearing_temperature, edgecolor='black')
plt.title('Bearing_temperature') 
sns.kdeplot(data.Bearing_temperature, fill = True)

plt.hist(data.Wheel_hub_temperature, edgecolor='black')
plt.title('Wheel_hub_temperature') 
sns.kdeplot(data.Wheel_hub_temperature, fill = True)

"""(B) Data cleaning"""
'''1) Missing value & duplicated rows'''
data.isna().sum() #No missing value
data.duplicated().sum() #No row duplicates

'''2) Duplicated columns'''
exclude_columns = ['Failure_status']
columns_for_correlation = [col for col in data.columns if col not in exclude_columns]
correlation_matrix = data[columns_for_correlation].corr() #No high similarity columns

'''3) Zero variance'''
'''For numeric columns'''
from sklearn.feature_selection import VarianceThreshold
var0 = VarianceThreshold(threshold=0)
exclude_columns = ['Failure_status']
columns_for_var0 = [col for col in data.columns if col not in exclude_columns]
var0.fit(data[columns_for_var0])
var0.get_support() #No zero variance

'''For non-numeric columns'''
categorical_columns = data.select_dtypes(include=['object']).columns
var0_categorical_columns = data[categorical_columns].columns[data[categorical_columns].nunique() == 1]
#No zero variance

'''4) Winsorization - IQR method'''
from feature_engine.outliers import Winsorizer
columns_to_winsorize = ['Generator_bearing_temperature', 'Gear_oil_temperature',
                        'Nacelle_temperature', 'Generator_speed', 'Wind_direction',
                        'Gear_box_inlet_temperature', 'Bearing_temperature']

winsor_iqr = Winsorizer(capping_method='iqr', tail='both', fold=1.5, 
                        variables=columns_to_winsorize)

data_winsor = winsor_iqr.fit_transform(data)

# Check outliers after winsorization
sns.boxplot(data_winsor[['Generator_bearing_temperature', 'Gear_oil_temperature', 
                              'Nacelle_temperature', 'Generator_speed', 'Wind_direction',
                              'Gear_box_inlet_temperature', 'Bearing_temperature']])
plt.title('After Winsorization')

data_winsor.isna().sum() #No missing value
data_winsor.duplicated().sum() #No row duplicates

"""(C) Save final data"""
data_winsor.to_csv('Final Data_Project 2.csv', index=False, encoding='utf-8')

# Searching & accessing the saved file
import os
current_directory = os.getcwd()
print("Current Directory:", current_directory)

