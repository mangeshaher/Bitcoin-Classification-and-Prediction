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
df = pd.read_csv('data_allsearch_normalized_modified.csv')

#print the head
#plt.plot(df['Volume'], label='Volume of Bitcoin Traded')
#plt.plot(df['Search_Rate'], label='Search Popularity')
#plt.show()
#df.hist()
rows = df.shape[0]
for i in range(11700,rows):
    for j in range(7):
        if i >= val[j] and df.loc[i,'Open']!=0:
            df.loc[i,'H'+chr(48+j)]=(df.loc[i - val[j] : i , 'High'].max())/df.loc[i,'Open']   #High Price feature
        elif i < val[j] and df.loc[i,'Open']!=0 :
            df.loc[i,'H'+chr(48+j)]=df.loc[i,'High']/df.loc[i,'Open']
        else :
            df.loc[i,'H'+chr(48+j)]=100
        if i >= val[j] and df.loc[i,'Open']!=0:
            df.loc[i,'L'+chr(48+j)]=(df.loc[i - val[j] : i , 'Low'].max())/df.loc[i,'Open']    #Low Price feature
        elif i < val[j] and df.loc[i,'Open']!=0 :
            df.loc[i,'L'+chr(48+j)]=df.loc[i , 'Low']/df.loc[i,'Open']
        else :
            df.loc[i,'L'+chr(48+j)]=100
        if i >= val[j] and df.loc[i,'Open']!=0:
            df.loc[i,'Av'+chr(48+j)]=(df.loc[i - val[j] : i , 'Open'].mean())/df.loc[i,'Open'] #Avg Price feature
        elif i < val[j] and df.loc[i,'Open']!=0 :
            df.loc[i,'Av'+chr(48+j)]=1
        else :
            df.loc[i,'Av'+chr(48+j)]=100
        if i >= val[j] :
            df.loc[i,'V'+chr(48+j)]=df.loc[i - val[j] : i , 'Volume'].sum()  #Volume feature
        else :
            df.loc[i,'V'+chr(48+j)]=df.loc[i,'Volume']
        if i >= val[j] and df.loc[i,'Volume']!=0:  #Volume n min ago to Volume now 
            df.loc[i,'Vr'+chr(48+j)]= (df.loc[i - val[j] ,'Volume'].mean())/df.loc[i,'Volume'] 
        elif i < val[j] and df.loc[i,'Volume']!=0 :
            df.loc[i,'Vr'+chr(48+j)]=1
        else :
            df.loc[i,'Vr'+chr(48+j)]=100
        if i >= val[j] and df.loc[i-val[j],'Open']!=0 :  #Proportion of Increase in Price 
            df.loc[i,'P'+chr(48+j)]= (df.loc[i,'Open']-df.loc[i - val[j],'Open'])/df.loc[i - val[j],'Open'] 
        elif i < val[j] and df.loc[0,'Open']!=0 :
            df.loc[i,'P'+chr(48+j)]= (df.loc[i,'Open']-df.loc[0,'Open'])/df.loc[0,'Open'] 
        else :
            df.loc[i,'P'+chr(48+j)]=100
        #if i >= val[j] :  #Proportion of Increase in Change in Price 
         #   df.loc[i,'Ac'+chr(48+j)]= (df.loc[i,'P']-df.loc[i - val[j],'P'])/df.loc[i - val[j],'P'] 
        #else :
         #   df.loc[i,'Ac'+chr(48+j)]=0
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
        
        if i >= val[j] and df.loc[i - val[j] ,'Websearch']!=0:  #Proportion of change in search rate
            df.loc[i,'PSW'+chr(48+j)]= (df.loc[i,'Websearch']-df.loc[i - val[j],'Websearch'])/(df.loc[i - val[j] ,'Websearch']) 
        elif i < val[j] and df.loc[0 ,'Websearch']!=0 :
            df.loc[i,'PSW'+chr(48+j)]=(df.loc[i,'Websearch']-df.loc[0,'Websearch'])/(df.loc[0 ,'Websearch'])  
        else :
            df.loc[i,'PSW'+chr(48+j)]=100
        if i >= val[j] and df.loc[i  ,'Websearch']!=0:  #max search rate over interval
            df.loc[i,'MaxSW'+chr(48+j)]= (df.loc[i - val[j] : i,'Websearch'].max())/(df.loc[i ,'Websearch']) 
        elif i < val[j] and df.loc[i ,'Websearch']!=0 :
            df.loc[i,'MaxSW'+chr(48+j)]=(df.loc[0 : i,'Websearch'].max())/(df.loc[i ,'Websearch']) 
        else :
            df.loc[i,'MaxSW'+chr(48+j)]=100
        if i >= val[j] and df.loc[i  ,'Websearch']!=0:  #ratio of Avg Search Rate over interval
            df.loc[i,'AvSW'+chr(48+j)]= (df.loc[i - val[j] : i,'Websearch'].mean())/(df.loc[i ,'Websearch']) 
        elif i < val[j] and df.loc[i ,'Websearch']!=0  :
            df.loc[i,'AvSW'+chr(48+j)]= (df.loc[0: i,'Websearch'].mean())/(df.loc[i ,'Websearch']) 
        else :
            df.loc[i,'AvSW'+chr(48+j)]=100

        if i >= val[j] and df.loc[i - val[j] ,'Newssearch']!=0:  #Proportion of change in search rate
            df.loc[i,'PSN'+chr(48+j)]= (df.loc[i,'Newssearch']-df.loc[i - val[j],'Newssearch'])/(df.loc[i - val[j] ,'Newssearch']) 
        elif i < val[j] and df.loc[0 ,'Newssearch']!=0 :
            df.loc[i,'PSN'+chr(48+j)]=(df.loc[i,'Newssearch']-df.loc[0,'Newssearch'])/(df.loc[0 ,'Newssearch'])  
        else :
            df.loc[i,'PSN'+chr(48+j)]=100
        if i >= val[j] and df.loc[i  ,'Newssearch']!=0:  #max search rate over interval
            df.loc[i,'MaxSN'+chr(48+j)]= (df.loc[i - val[j] : i,'Newssearch'].max())/(df.loc[i ,'Newssearch']) 
        elif i < val[j] and df.loc[i ,'Newssearch']!=0 :
            df.loc[i,'MaxSN'+chr(48+j)]=(df.loc[0 : i,'Newssearch'].max())/(df.loc[i ,'Newssearch']) 
        else :
            df.loc[i,'MaxSN'+chr(48+j)]=100
        if i >= val[j] and df.loc[i  ,'Newssearch']!=0:  #ratio of Avg Search Rate over interval
            df.loc[i,'AvSN'+chr(48+j)]= (df.loc[i - val[j] : i,'Newssearch'].mean())/(df.loc[i ,'Newssearch']) 
        elif i < val[j] and df.loc[i ,'Newssearch']!=0  :
            df.loc[i,'AvSN'+chr(48+j)]= (df.loc[0: i,'Newssearch'].mean())/(df.loc[i ,'Newssearch']) 
        else :
            df.loc[i,'AvSN'+chr(48+j)]=100

        if i >= val[j] and df.loc[i - val[j] ,'Youtubesearch']!=0:  #Proportion of change in search rate
            df.loc[i,'PSY'+chr(48+j)]= (df.loc[i,'Youtubesearch']-df.loc[i - val[j],'Youtubesearch'])/(df.loc[i - val[j] ,'Youtubesearch']) 
        elif i < val[j] and df.loc[0 ,'Youtubesearch']!=0 :
            df.loc[i,'PSY'+chr(48+j)]=(df.loc[i,'Youtubesearch']-df.loc[0,'Youtubesearch'])/(df.loc[0 ,'Youtubesearch'])  
        else :
            df.loc[i,'PSY'+chr(48+j)]=100
        if i >= val[j] and df.loc[i  ,'Youtubesearch']!=0:  #max search rate over interval
            df.loc[i,'MaxSY'+chr(48+j)]= (df.loc[i - val[j] : i,'Youtubesearch'].max())/(df.loc[i ,'Youtubesearch']) 
        elif i < val[j] and df.loc[i ,'Youtubesearch']!=0 :
            df.loc[i,'MaxSY'+chr(48+j)]=(df.loc[0 : i,'Youtubesearch'].max())/(df.loc[i ,'Youtubesearch']) 
        else :
            df.loc[i,'MaxSY'+chr(48+j)]=100
        if i >= val[j] and df.loc[i  ,'Youtubesearch']!=0:  #ratio of Avg Search Rate over interval
            df.loc[i,'AvSY'+chr(48+j)]= (df.loc[i - val[j] : i,'Youtubesearch'].mean())/(df.loc[i ,'Youtubesearch']) 
        elif i < val[j] and df.loc[i ,'Youtubesearch']!=0  :
            df.loc[i,'AvSY'+chr(48+j)]= (df.loc[0: i,'Youtubesearch'].mean())/(df.loc[i ,'Youtubesearch']) 
        else :
            df.loc[i,'AvSY'+chr(48+j)]=100
    if(i+20<rows) and df.loc[i  ,'Close']!=0:
        df.loc[i,'Y1']=math.log10(df.loc[i+20 , 'Close']/df.loc[i ,'Close'])
        if(math.log10(df.loc[i+20 , 'Close']/df.loc[i ,'Close']) > 0):
            df.loc[i,'Yt1'] = 1
        else:
            df.loc[i,'Yt1'] = 0
    elif(i+20<=rows) and df.loc[0  ,'Close']==0 :
        df.loc[i,'Y1']=math.log10(df.loc[i+20 , 'Close']/df.loc[0 ,'Close'])
        if(math.log10(df.loc[i+20 , 'Close']/df.loc[0 ,'Close']) > 0):
            df.loc[i,'Yt1'] = 1
        else:
            df.loc[i,'Yt1'] = 0
    else :
        df.loc[i,'Y1'] = 1
        df.loc[i,'Yt1'] = 1
    if(i+10<rows) and df.loc[i  ,'Close']!=0:
        df.loc[i,'Y2']=math.log10(df.loc[i+10 , 'Close']/df.loc[i ,'Close'])
        if(math.log10(df.loc[i+10 , 'Close']/df.loc[i ,'Close']) > 0):
            df.loc[i,'Yt2'] = 1
        else:
            df.loc[i,'Yt2'] = 0
    elif(i+10>=rows) and df.loc[0  ,'Close']!=0:
        df.loc[i,'Y2']=math.log10(df.loc[i+10 , 'Close']/df.loc[0 ,'Close'])
        if(math.log10(df.loc[i+10 , 'Close']/df.loc[0 ,'Close']) > 0):
            df.loc[i,'Yt2'] = 1
        else:
            df.loc[i,'Yt2'] = 0
    else :
        df.loc[i,'Y2'] = 1
        df.loc[i,'Yt2'] = 1
    if(i+5<rows) and df.loc[i  ,'Close']!=0:
        df.loc[i,'Y3']=math.log10(df.loc[i+5 , 'Close']/df.loc[i ,'Close'])
        if(math.log10(df.loc[i+5 , 'Close']/df.loc[i ,'Close']) > 0):
            df.loc[i,'Yt3'] = 1
        else:
            df.loc[i,'Yt3'] = 0
    elif(i+5>=rows) and df.loc[0  ,'Close']!=0:
        df.loc[i,'Y3']=math.log10(df.loc[i+5 , 'Close']/df.loc[0 ,'Close'])
        if(math.log10(df.loc[i+5 , 'Close']/df.loc[0 ,'Close']) > 0):
            df.loc[i,'Yt3'] = 1
        else:
            df.loc[i,'Yt3'] = 0
    else :
        df.loc[i,'Y3'] = 1
        df.loc[i,'Yt3'] = 1
    if i % 100 == 0:
        df.to_csv('temp_data_featues_allsearch.csv',sep=",",index=False)
        print('*')

