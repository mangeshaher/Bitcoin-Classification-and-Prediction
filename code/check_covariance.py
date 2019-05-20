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
dff = df.cov()
dff.to_csv('covariance.csv',sep=",",index=False)