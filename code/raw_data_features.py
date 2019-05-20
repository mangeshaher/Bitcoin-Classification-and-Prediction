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
df = pd.read_csv('raw_data.csv')

#print the head
#plt.plot(df['Volume'], label='Volume of Bitcoin Traded')
#plt.plot(df['Search_Rate'], label='Search Popularity')
#plt.show()
#df.hist()
rows = df.shape[0]
for i in range(rows):
    for j in range(7):
        if i >= val[j] and df.loc[i-val[j],'Open']!=0 :  #Proportion of Increase in Price 
            df.loc[i,'P'+chr(48+j)]= (df.loc[i,'Open']-df.loc[i - val[j],'Open'])/df.loc[i - val[j],'Open'] 
        elif i < val[j] and df.loc[0,'Open']!=0 :
            df.loc[i,'P'+chr(48+j)]= (df.loc[i,'Open']-df.loc[0,'Open'])/df.loc[0,'Open'] 
        else :
            df.loc[i,'P'+chr(48+j)]=100  
        if i >= val[j] and df.loc[i,'Open']!=0 :  #Proportion of Increase in Change in Price 
            df.loc[i,'R'+chr(48+j)]= (df.loc[i - val[j],'Open'])/df.loc[i ,'Open'] 
        elif i < val[j] and df.loc[0,'Open']!=0 :
            df.loc[i,'R'+chr(48+j)]= (df.loc[0,'Open'])/df.loc[i ,'Open'] 
        else :
            df.loc[i,'R'+chr(48+j)]=100
        if i >= 2*val[j] : #Price change proportion from t-2n,t-n to t-n,t
            df.loc[i,'WAP'+chr(48+j)]=(df.loc[i - 2*val[j] : i - val[j] , 'Open'].mean())/(df.loc[i - val[j] : i  , 'Open'].mean())
        elif i >= val[j] :
            df.loc[i,'WAP'+chr(48+j)]=(df.loc[0 : i - val[j] , 'Open'].mean())/(df.loc[i - val[j] : i  , 'Open'].mean())
        else :
            df.loc[i,'WAP'+chr(48+j)]=1
        
        if i >= 2*val[j] and df.loc[i,'Open']!=0 : #Price change proportion from t-2n,t-n to t-n,t 
            df.loc[i,'AP'+chr(48+j)]=(df.loc[i - 2*val[j] : i - val[j] , 'Open'].mean())/df.loc[i,'Open']   
        elif i >= val[j] and df.loc[i,'Open']!=0:
            df.loc[i,'AP'+chr(48+j)]=(df.loc[0 : i - val[j] , 'Open'].mean())/df.loc[i,'Open']
        elif i < val[j] and df.loc[i,'Open']!=0 :
            df.loc[i,'AP'+chr(48+j)]=(df.loc[0 , 'Open'].mean())/df.loc[i,'Open']
        else :
            df.loc[i,'AP'+chr(48+j)]=100
    if i % 1000 == 0 :
        df.to_csv('raw_data_features.csv',sep=",",index=False)
        
#if i >= val[j] :  #Proportion of Increase in Change in Price 
        #    df.loc[i,'Ac'+chr(48+j)]= (df.loc[i,'P']-df.loc[i - val[j],'P'])/df.loc[i - val[j],'P'] 
        #else :
        #    df.loc[i,'Ac'+chr(48+j)]=0  
        

