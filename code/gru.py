# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 14:23:09 2019

@author: MangeshAher-PC
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:38:58 2019

@author: MangeshAher-PC
"""

#import packages
import pandas as pd
import numpy 

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
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error,median_absolute_error
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.preprocessing.text import Tokenizer
from collections import defaultdict
from keras.layers.convolutional import Convolution1D
from keras import backend as K




data_scaler = MinMaxScaler(feature_range = (0, 1))

EPOCH = 480
BATCH = 1124

df = pd.read_csv('classification_featuresss.csv')

lis = [100,101,102,106,107,108,112,113,114]
liss = [103,104,105,109,110,111,115,116,117]

X = df.iloc[:,lis]
#print(X)
X = data_scaler.fit_transform(X)

y = df.iloc[:,99]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size= 0.05
                                                    ,shuffle=False)

smodel = Sequential()
smodel.add(GRU(100,input_shape = (1,9),activation='relu',go_backwards=True,return_sequences=False))

#,input_shape = (1,3)

smodel.add(Dense(1))
smodel.compile(loss='mean_squared_error',optimizer='adam',metrics=['mean_squared_error','mean_absolute_error'])
#X_train_scaled = data_scaler.fit_transform(X_train)

xtrain = numpy.reshape(X_train,(X_train.shape[0],1,X_train.shape[1]))
xtest = numpy.reshape(X_test,(X_test.shape[0],1,X_test.shape[1]))
#ytrain = numpy.reshape(y_train,(y_train.shape[0],1,y_train.shape[1]))
#ytest = numpy.reshape(y_test,(y_test.shape[0],1,y_test.shape[1]))
#ytrain_data = y_train.reshape(1,210750,1)


smodel.fit(xtrain,y_train,batch_size=150,epochs=400,shuffle=False)
    
#X_test_scaled = data_scaler.transform(X_test)
y_pred = smodel.predict(xtest)
#dff = pd.DataFrame({'pred': y_pred , 'real' : y_test})
#print('Predicted Test sample for n + 20 is :-')
#print(dff)
score = smodel.evaluate(xtest, y_test, batch_size=150)
print(score)




