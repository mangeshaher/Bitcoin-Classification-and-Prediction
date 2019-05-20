# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:19:27 2019

@author: MangeshAher-PC
"""

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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

data_scaler = MinMaxScaler(feature_range = (0, 1))

figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
val = [1,2,5,10,20,40,80]
#read the file


#nsfdata is 9000 records
#classification_features is 64k records

#df = pd.read_csv('two_features.csv')
df = pd.read_csv('classification_featuress.csv')
lis = [97,98,99]
li = [97]
X = data_scaler.fit_transform(df.iloc[:,lis])
print(X)

plt.plot(X[0], label='Search_Rate of Bitcoin')
plt.plot(df['P2'], label='Search_Rate of Bitcoin')
plt.plot(df['R2'], label='Search_Rate of Bitcoin')
plt.show()
        
