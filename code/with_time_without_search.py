# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 19:44:20 2019

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
from sklearn.preprocessing import MinMaxScaler

data_scaler = MinMaxScaler(feature_range = (0, 1))

figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
val = [1,2,5,10,20,40,80]
#read the file
df = pd.read_csv('classification_features.csv')

#print the head
#plt.plot(df['Volume'], label='Volume of Bitcoin Traded')
#plt.plot(df['Search_Rate'], label='Search Popularity')
#plt.show()
#df.hist()

lis = [7,8,9,10,11,12,13,14,15,19,20,21,22,23,24,25,26,27,31,32,33,34,35,36,37,38,39,43,44,45,46,47,48,49,50,51,55,56,57,58,59,60,61,62,63,67,68,69,70,71,72,73,74,75,79,80,81,82,83,84,85,86,87]
X = df.iloc[:,lis]
X = data_scaler.fit_transform(X)

y = df.iloc[:,92]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.05,shuffle=False,random_state=0)

X_train_scaled = data_scaler.fit_transform(X_train)

print(X_train.shape)
classifier = LogisticRegression(random_state=0,solver='lbfgs',max_iter= 2000)
classifier.fit(X_train_scaled, y_train)

X_test_scaled = data_scaler.transform(X_test)

y_pred = classifier.predict(X_test_scaled)

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print("""Accuracy of logistic regression with time features and 
      without search features  for 20 classifier on test set:
    {:.4f}"""  .format(classifier.score(X_test, y_test)))



#df.to_csv('wwsearch.csv',sep=",")
        
