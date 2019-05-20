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
from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
#setting figure size
from matplotlib.pylab import rcParams
from matplotlib.pyplot import figure
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

data_scaler = MinMaxScaler(feature_range = (0, 1))

figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
val = [1,2,5,10,20,40,80]
#read the file


#nsfdata is 9000 records
#classification_features is 64k records


df = pd.read_csv('classification_features.csv')

#print the head
#plt.plot(df['Volume'], label='Volume of Bitcoin Traded')
#plt.plot(df['Search_Rate'], label='Search Popularity')
#plt.show()
#df.hist()
        
#learning_rate = 'adaptive'
X = df.iloc[:,7:91]
X = data_scaler.fit_transform(X)

y = df.iloc[:,92]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size= 0.05,shuffle=False,random_state=0)

#X_train_scaled = data_scaler.fit_transform(X_train)

pca = PCA(n_components = 30)
pca.fit(X_train)
print(pca.n_components_)
X_train = pca.transform(X_train)

#X_test_scaled = data_scaler.transform(X_test)

X_test = pca.transform(X_test)

print(X_train.shape)
#print(X_train)
#print(X_test)

classifier = MLPClassifier(random_state=0,hidden_layer_sizes=(30,),batch_size=100,
                           solver='sgd', max_iter=1000,
                           alpha=0.0001 ,learning_rate = 'adaptive',
                           learning_rate_init=0.1,early_stopping=True,
                           shuffle=False,n_iter_no_change=25)

#n_iter_no_change=25,early_stopping=False,n_iter_no_change=25,learning_rate='adaptive',,tol=0.0001,n_iter_no_change=25,
#early_stopping,n_iter_no_change,tol,learning_rate_init,early_stopping=True, 
classifier.fit(X_train , y_train)


y_pred = classifier.predict(X_test)
#classifier.predict(X_test)

#cv = ShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
 

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print("""Accuracy of Neural Network with time
      features and with search features  for 20 classifier
      having hidden layer (30,) batch as 100 alpha .0001 and n_iter_no_change=20
   on test set: {:.4f}""".format(classifier.score(X_test,y_test)))



#df.to_csv('finall.csv',sep=",")
        
