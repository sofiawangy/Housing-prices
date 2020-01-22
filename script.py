#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 19:07:48 2020

@author: sofiawangy
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

path = '/home/sofiawangy/Github/Housing Project//'

train = pd.read_csv(path + 'train.csv', index_col = 0)
test = pd.read_csv(path + 'test.csv', index_col = 0)

x_train1 = train.iloc[:, :-1]
y_train = train.iloc[:, -1]

cont =['LotFrontage', 'LotArea', 'MasVnrArea', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 
       'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 
       'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath']

cat = list(set(x_train1.columns) - set(cont))

#encode categorical

x_train1[cat] = x_train1[cat].astype(str)
test[cat] = test[cat].astype(str)

le = LabelEncoder()

x_train2= pd.DataFrame()
test2= pd.DataFrame()

for i in cat:
    x_train2[i] = le.fit_transform(x_train1[i])
    test2[i] = le.fit_transform(test[i])

#merge encoded and continuous variables
x_train2 = np.concatenate((x_train2, x_train1[cont]), axis=1)
test2 = np.concatenate((test2, test[cont]), axis=1)

columnNames  = cat + cont
x_train2 = pd.DataFrame(x_train2, columns = columnNames)
test2 = pd.DataFrame(test2, columns = columnNames)

x_train2.fillna(0, inplace=True)
test2.fillna(0, inplace=True)

#get R^2 with multiple linear
lm = LinearRegression()
lm.fit(x_train2, y_train)
residuals = y_train - lm.predict(x_train2)
plt.hist(residuals)

lm.score(x_train2, y_train) #0.8462

#VIF
vif = {'factor': [], 'features': []}

for i in range(x_train2.shape[1]):
    factor = variance_inflation_factor(x_train2.values, i)
    
    if factor <= 5:
        vif['factor'].append(factor)
        vif['features'].append(x_train2.columns[i])
        
#ANOVA




























