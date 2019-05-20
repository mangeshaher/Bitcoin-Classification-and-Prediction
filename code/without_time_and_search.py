# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 17:14:24 2019

@author: MangeshAher-PC
"""

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


  
        
X = df.iloc[:,1:6]
X = data_scaler.fit_transform(X)

y = df.iloc[:,92]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size= 0.05,shuffle=False,random_state=0)
print(X_train.shape)
classifier = LogisticRegression(random_state=0,solver='lbfgs',max_iter= 2000)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print("""Accuracy of logistic regression without time features and without search
      for 5 classifier on test set: {:.4f}
      """.format(classifier.score(X_test, y_test)))



#df.to_csv('wwsearch.csv',sep=",")



