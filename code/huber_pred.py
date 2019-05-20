# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 20:04:45 2019

@author: MangeshAher-PC
"""

#import packages
import pandas as pd
import numpy as np

#to plot within notebook
import matplotlib.pyplot as plt

from sklearn import preprocessing
import matplotlib.pyplot as plt 
import math 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
#setting figure size
from matplotlib.pylab import rcParams
from matplotlib.pyplot import figure
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error,median_absolute_error
from sklearn.preprocessing import MinMaxScaler
data_scaler = MinMaxScaler(feature_range = (0, 1))

figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')



df = pd.read_csv('classification_featuresss.csv')
#df = pd.read_csv('price_pred_features.csv')
"""
df[['Open','High','Low','Close','Volume_(BTC)','Websearch',
    'Newssearch','Youtubesearch','Weighted_Price']]=data_scaler.fit_transform(df[['Open','High','Low','Close','Volume_(BTC)','Websearch',
    'Newssearch','Youtubesearch','Weighted_Price']])
"""
#lis = [1,2,3]
lis = [100,101,102,106,107,108,112,113,114]
liss = [103,104,105,109,110,111,115,116,117]
X = df.iloc[:,lis]
X = data_scaler.fit_transform(X)
#print(X)
#y33 means n+5   ie 97, y22 means n+10 ie 98, y11 means n+20 ie 99
y = df.iloc[:,99]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size= 0.05 ,shuffle=False,random_state=0)
"""
pca = PCA(n_components = 30)


#pca.fit_transform(X_train_scaled)
#dpca = pd.DataFrame(pca.components_,columns=X.columns)



pca.fit(X_train)
#print(pca.n_components_)
X_train = pca.transform(X_train)

#X_test_scaled = data_scaler.transform(X_test)

X_test = pca.transform(X_test)

#X_train_scaled = data_scaler.fit_transform(X_train)
#y_train_scaled = data_scaler.fit_transform(y_train)
#print(X_train)
"""
hr = HuberRegressor().fit(X_train,y_train)

#X_test_scaled = data_scaler.transform(X_test)
#y_test_scaled = data_scaler.transform(y_test)
y_pred = hr.predict(X_test)
plt.plot(list(range(0,7101)),y_test-y_pred,label='Error in Bitcoin Price')
plt.xlabel('Row iteration')
plt.ylabel('Price of Bitcoin in $')
plt.legend(loc='best')
plt.show()

dff = pd.DataFrame({'pred': y_pred , 'real' : y_test})
print('Predicted Test sample for n + 20 is :-')
print(dff)
print('Huber Regression Prediction for n + 20 is :-')
print('Mean Absolute Error is :- ' , mean_absolute_error(y_pred,y_test))
print('Median Absolute Error is :- ' , median_absolute_error(y_pred,y_test))
print('Mean Squared Error is :- ' , mean_squared_error(y_pred,y_test))
print('R2 score is :- ' , r2_score(y_pred,y_test))
"""
X_test['pred']=y_pred
dfff = data_scaler.inverse_transform(X_test.reshape(-1,1)).flatten()
print(dfff)
#y_predd = data_scaler.inverse_transform(y_pred.reshape(0,1)).flatten()
#print(mean_squared_error(y_pred,y_test))
#print(r2_score(y_pred,y_test))
#y_predd = data_scaler.transform(y_pred)
#print(y_predd,y_test)
"""