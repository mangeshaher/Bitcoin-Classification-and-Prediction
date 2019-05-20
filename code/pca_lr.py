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
from sklearn.decomposition import PCA
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
X = df.iloc[:,7:91]
X = data_scaler.fit_transform(X)
print(X)
y = df.iloc[:,96]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size= 0.05,shuffle=False,random_state=0)

#data_scaled = pd.DataFrame(preprocessing.scale(df),columns = df.col) 

#X_train_scaled = data_scaler.fit_transform(X_train)

pca = PCA(n_components = 30)

"""
pca.fit_transform(X_train_scaled)
dpca = pd.DataFrame(pca.components_,columns=X.columns)
"""


pca.fit(X_train)
print(pca.n_components_)
X_train = pca.transform(X_train)

#X_test_scaled = data_scaler.transform(X_test)

X_test = pca.transform(X_test)

print(X_train.shape)
classifier = LogisticRegression(random_state=0,solver='lbfgs',max_iter=2000)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print("""Accuracy of logistic regression after PCA 30 with time
      features and with search features  for 5 classifier
      on test set: {:.4f}""".format(classifier.score(X_test, y_test)))


#dpca.to_csv('short_dataset.csv',sep=",",index=False)
#df.to_csv('finall.csv',sep=",")
        
