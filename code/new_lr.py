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
scaler = MinMaxScaler(feature_range = (0, 1))
figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
val = [1,2,5,10,20,40,80]
#read the file
df = pd.read_csv('data_features_allsearch.csv')

#data_scaled = pd.DataFrame(preprocessing.scale(df),columns = df.col) 
"""
pca = PCA(n_components = 30)
pca.fit_transform(X)
dpca = pd.DataFrame(pca.components_,columns=X.columns)



pca.fit(X_train)
print(pca.n_components_)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
"""

X = df.iloc[:,9:134]
y = df.iloc[:,136]
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle='false')
print(X_train.shape)

pca = PCA(n_components = 40)
pca.fit_transform(X)
dpca = pd.DataFrame(pca.components_,columns=X.columns)



pca.fit(X_train)
print(pca.n_components_)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

classifier = LogisticRegression(solver='lbfgs',max_iter= 1000)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print("""Accuracy of logistic regression with time features and 
      with search features  for 10 classifier on test set:
    {:.4f}"""  .format(classifier.score(X_test, y_test)))

