# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 19:08:29 2019

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
#figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')

df = pd.read_csv('data_allsearch_normalized.csv')
dm = df.corr()
dm.to_csv('my_check.csv',sep=",",index=False)