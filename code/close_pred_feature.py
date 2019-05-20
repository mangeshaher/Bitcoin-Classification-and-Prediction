# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 23:38:50 2019

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

figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
val = [1,2,5,10,20,40,80]
#read the file
df = pd.read_csv('data_allsearch_end.csv')

#print the head
#plt.plot(df['Volume'], label='Volume of Bitcoin Traded')
#plt.plot(df['Search_Rate'], label='Search Popularity')
#plt.show()
#df.hist()
rows = df.shape[0]
for i in range(rows):
    if(i+20<rows):
        df.loc[i,'Y1']=df.loc[i+20 ,'Close']
    else :
        df.loc[i,'Y1'] = df.loc[i ,'Close']
    if(i+10<rows):
        df.loc[i,'Y2']=df.loc[i+10 ,'Close']
    else :
        df.loc[i,'Y2'] = df.loc[i ,'Close']
    if(i+5<rows):
        df.loc[i,'Y3']=df.loc[i+5 ,'Close']
    else :
        df.loc[i,'Y3'] = df.loc[i ,'Close']
    if(i-20>=0):
        df.loc[i,'O1']=df.loc[i ,'Open']-df.loc[i-20,'Open']
        df.loc[i,'H1']=df.loc[i ,'High']-df.loc[i-20,'High']
        df.loc[i,'L1']=df.loc[i,'Low']-df.loc[i-20,'Low']
        df.loc[i,'V1']=df.loc[i,'Volume_(BTC)']-df.loc[i-20,'Volume_(BTC)']
        df.loc[i,'WS1']=df.loc[i ,'Websearch']-df.loc[i-20,'Websearch']
        df.loc[i,'NS1']=df.loc[i ,'Newssearch']-df.loc[i-20,'Newssearch']
        df.loc[i,'YS1']=df.loc[i ,'Youtubesearch']-df.loc[i-20,'Youtubesearch']
    else :
        df.loc[i,'O1'] = 0
        df.loc[i,'H1'] = 0
        df.loc[i,'L1']= 0
        df.loc[i,'V1']= 0
        df.loc[i,'WS1']=0
        df.loc[i,'NS1']=0
        df.loc[i,'YS1']=0
    if(i-10>=0):
        df.loc[i,'O2']=df.loc[i ,'Open']-df.loc[i-10,'Open']
        df.loc[i,'H2']=df.loc[i ,'High']-df.loc[i-10,'High']
        df.loc[i,'L2']=df.loc[i ,'Low']-df.loc[i-10,'Low']
        df.loc[i,'V2']=df.loc[i ,'Volume_(BTC)']-df.loc[i-10,'Volume_(BTC)']
        df.loc[i,'WS2']=df.loc[i ,'Websearch']-df.loc[i-10,'Websearch']
        df.loc[i,'NS2']=df.loc[i ,'Newssearch']-df.loc[i-10,'Newssearch']
        df.loc[i,'YS2']=df.loc[i ,'Youtubesearch']-df.loc[i-10,'Youtubesearch']
    else :
        df.loc[i,'O2'] = 0
        df.loc[i,'H2'] = 0
        df.loc[i,'L2']=0
        df.loc[i,'V2']= 0
        df.loc[i,'WS2']=0
        df.loc[i,'NS2']=0
        df.loc[i,'YS2']=0
    if(i-5>=0):
        df.loc[i,'O3']=df.loc[i ,'Open']-df.loc[i-5,'Open']
        df.loc[i,'H3']=df.loc[i,'High']-df.loc[i-5,'High']
        df.loc[i,'L3']=df.loc[i,'Low']-df.loc[i-5,'Low']
        df.loc[i,'V3']=df.loc[i,'Volume_(BTC)']-df.loc[i-5,'Volume_(BTC)']
        df.loc[i,'WS3']=df.loc[i ,'Websearch']-df.loc[i-5,'Websearch']
        df.loc[i,'NS3']=df.loc[i ,'Newssearch']-df.loc[i-5,'Newssearch']
        df.loc[i,'YS3']=df.loc[i,'Youtubesearch']-df.loc[i-5,'Youtubesearch']
    else :
        df.loc[i,'O3'] = 0
        df.loc[i,'H3'] = 0
        df.loc[i,'L3']=0
        df.loc[i,'V3']= 0
        df.loc[i,'WS3']=0
        df.loc[i,'NS3']=0
        df.loc[i,'YS3']=0
    if i % 1000 == 0 :
        df.to_csv('data_allsearch_end_features.csv',sep=",",index=False)
        print('*')
        

    