#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 18:57:50 2020

@author: ranjan.gsk
"""



#importing the libraries required for the clustering

#since the datasets are od the type arff we import this library
from scipy.io import arff
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

#importing the birch algorithm from scikit learn
from sklearn.cluster import Birch



##########################


#Considering the birch dataset
filename= 'birch-rg1.arff'
file=filename.split('.')
data = arff.loadarff('../dataset/'+filename)
X = pd.DataFrame(data[0])
X.head()


#scatter the points of the dataset to understand the points nature
plt.scatter(X.iloc[:,0], X.iloc[:,1], alpha=0.7, edgecolors='black')
plt.savefig('../plots/'+file[0]+'.png')

#threshold: The radius of the subcluster obtained by merging a new sample and the closest subcluster should be lesser than the threshold.    
#branching_factor: Maximum number of CF subclusters in each node
#n_clusters: Number of clusters after the final clustering step, which treats the subclusters from the 
#leaves as new samples. If set to None, the final clustering step is not performed and the subclusters are returned as they are.


#using the birch algorithm
brc = Birch(branching_factor=50, n_clusters=None, threshold=1.5)
brc.fit(X)
    
labels = brc.predict(X)

#plotting the datapoints with the clusters
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=labels, cmap='rainbow', alpha=0.7, edgecolors='black')
plt.savefig('../plots/'+file[0]+' clusters'+'.png')


############################
#Considering the cure dataset
filename= 'cure-t0-2000n-2D.arff'
file=filename.split('.')
data = arff.loadarff('../dataset/'+filename)
X = pd.DataFrame(data[0])
X.head()

plt.scatter(X.iloc[:,0], X.iloc[:,1], alpha=0.7, edgecolors='black')
plt.savefig('../plots/'+file[0]+'.png')
    
brc = Birch(branching_factor=50, n_clusters=None, threshold=0.8)
brc.fit(X)
    
labels = brc.predict(X)

plt.scatter(X.iloc[:,0], X.iloc[:,1], c=labels, cmap='rainbow', alpha=0.7, edgecolors='black')
plt.savefig('../plots/'+file[0]+' clusters'+'.png')

##############################
#Considering the disk dataset
filename= 'disk-1000n.arff'
file=filename.split('.')
data = arff.loadarff('../dataset/'+filename)
X = pd.DataFrame(data[0])
X.head()

plt.scatter(X.iloc[:,0], X.iloc[:,1], alpha=0.7, edgecolors='black')
plt.savefig('../plots/'+file[0]+'.png')
    
brc = Birch(branching_factor=50, n_clusters=None, threshold=0.5)
brc.fit(X)
    
labels = brc.predict(X)

plt.scatter(X.iloc[:,0], X.iloc[:,1], c=labels, cmap='rainbow', alpha=0.7, edgecolors='black')
plt.savefig('../plots/'+file[0]+' clusters'+'.png')

##############################
#Considering the donut dataset
filename= 'donut1.arff'
file=filename.split('.')
data = arff.loadarff('../dataset/'+filename)
X = pd.DataFrame(data[0])
X.head()

plt.scatter(X.iloc[:,0], X.iloc[:,1], alpha=0.7, edgecolors='black')
plt.savefig('../plots/'+file[0]+'.png')
    
brc = Birch(branching_factor=50, n_clusters=None, threshold=0.1)
brc.fit(X)
    
labels = brc.predict(X)

plt.scatter(X.iloc[:,0], X.iloc[:,1], c=labels, cmap='rainbow', alpha=0.7, edgecolors='black')
plt.savefig('../plots/'+file[0]+' clusters'+'.png')

##############################
#Considering the square dataset
filename= 'square1.arff'
file=filename.split('.')
data = arff.loadarff('../dataset/'+filename)
X = pd.DataFrame(data[0])
X.head()

plt.scatter(X.iloc[:,0], X.iloc[:,1], alpha=0.7, edgecolors='black')
plt.savefig('../plots/'+file[0]+'.png')
    
brc = Birch(branching_factor=50, n_clusters=None, threshold=2)
brc.fit(X)
    
labels = brc.predict(X)

plt.scatter(X.iloc[:,0], X.iloc[:,1], c=labels, cmap='rainbow', alpha=0.7, edgecolors='black')
plt.savefig('../plots/'+file[0]+' clusters'+'.png')

###############################
