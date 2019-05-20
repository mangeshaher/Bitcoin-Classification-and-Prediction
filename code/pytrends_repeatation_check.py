# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 23:42:42 2019

@author: MangeshAher-PC
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 14:22:35 2019

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
from pytrends.request import TrendReq
figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')

df = pd.read_csv('web_search.csv')
count=0
li = []

for i in range(df.shape[0]):
    x = (df.loc[i,'date']) 
    y = (df.loc[i-1,'date'])
    if x == y :
        print(x)
        count = count + 1
        #df.loc[i-1,'Bitcoin']=(df.loc[i,'Bitcoin']+df.loc[i-1,'Bitcoin'])/2
        li.append(i)

#dff = df.drop(df.index[li])
#dff.to_csv('youtube_search.csv',sep=",",index=False)
#plt.plot(df['Timestamp'], label='Timestamp')
#plt.show()
