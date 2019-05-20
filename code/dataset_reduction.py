# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 23:45:51 2019

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

figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
val = [1,2,5,10,20,40,80]
#read the file
df = pd.read_csv('data_allsearch_end.csv')

#print the head
#plt.plot(df['Volume'], label='Volume of Bitcoin Traded')
#plt.plot(df['Search_Rate'], label='Search Popularity')
#plt.show()
#df.hist()
        

#dff = df.drop(['Volume_(Currency)', 'Weighted_Price'], axis=1)
dff = df.drop(['Weighted_Price'], axis=1)

dff.to_csv('data_allsearch_end.csv',sep=",",index=False)
        
