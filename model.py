import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats as stats

df = pd.read_csv('D:\\Deployment\\Heart_Attack\\dataset.csv')
df.head()

# Feature Selection:-
df.drop(['oldpeak','slp','thall'], axis = 1, inplace = True)

# Feature Scaling:-
x = df.drop('output', axis = 1)
y = df['output']
scal = MinMaxScaler()
scal_df = scal.fit_transform(x)
df1 = pd.DataFrame(scal_df, columns = df.columns[:-1])

x = df1
y = df['output']


# Outliers Treatment:-

def outlier_data(df):
    df_outlier = df.copy()
    for column in df.columns:
        upper_limit = df[column].mean() + 3*df[column].std()
        lower_limit = df[column].mean() - 3*df[column].std()
        
        df_outlier[column] = df_outlier[column].apply(lambda x: upper_limit if x > upper_limit else x)
        
        df_outlier[column] = df_outlier[column].apply(lambda x: lower_limit if x < lower_limit else x)
    
    return df_outlier  

df = outlier_data(df1)

# Balancing data:-
x = df
from imblearn.under_sampling import ClusterCentroids
cc = ClusterCentroids(random_state = 42)
x_res,y_res = cc.fit_resample(x,y)

# Splite the data:-
x = x_res
y = y_res
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

# With Logistic Regression:-

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train,y_train)

import pickle
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scal, open("scaler.pkl", "wb"))

import os
os.getcwd()