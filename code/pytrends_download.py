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

pytrends = TrendReq(hl='en-US', tz=360)
kw_list = ["Bitcoin"]
df = pytrends.get_historical_interest(kw_list, 
    year_start=2018, month_start=1, day_start=1, 
    hour_start=0, year_end=2018, month_end=1, day_end=16,
    hour_end=10,cat=0, geo='', gprop='', sleep=0)
df.to_csv('youtube_search.csv',sep=",")
