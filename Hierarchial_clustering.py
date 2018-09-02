# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 11:13:44 2018

@author: ASUS
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:\\Users\\ASUS\Desktop\\Linear_algebra\\Mall_Customers.csv')
x = dataset.iloc[:, [3,4]].values

#finding the optimal clustering using dendogram method
import scipy.cluster.hierarchy as sch
dendogram =sch.dendrogram(sch.linkage(x,method = 'ward', metric ='euclidean'))
plt.title("DENDOGRAM")
plt.xlabel("customers")
plt.ylabel("euclidean distance")
plt.show()

#fitting the optimal number of cluster in the data set
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters= 5,affinity = 'euclidean',linkage ='ward')
y_hc = hc.fit_predict(x)

#visualizing the clusters
plt.scatter(x[y_hc==0, 0],x[y_hc==0, 1], c="cyan", label= "cluster1")
plt.scatter(x[y_hc==1, 0],x[y_hc==1, 1], c="r", label= "cluster2")
plt.scatter(x[y_hc==2, 0],x[y_hc==2, 1], c="b", label= "cluster3")
plt.scatter(x[y_hc==3, 0],x[y_hc==3, 1], c="y", label= "cluster4")
plt.scatter(x[y_hc==4, 0],x[y_hc==4, 1], c="magenta", label= "cluster5")
plt.title("HC_clustering")
plt.xlabel("annual income")
plt.ylabel("spending score")
plt.show()

