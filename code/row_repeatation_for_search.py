# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 00:19:13 2019

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

df = pd.read_csv('youtube_search.csv')
dff = df.loc[df.index.repeat(60)]
dff.to_csv('mul_youtube_search.csv',sep=",",index=False)
