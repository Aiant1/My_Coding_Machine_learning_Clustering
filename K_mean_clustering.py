# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 19:15:57 2018

@author: Antika
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:\\Users\\ASUS\Desktop\\Linear_algebra\\Mall_Customers.csv')
x = dataset.iloc[:, [3,4]].values

#using elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters =i,init='k-means++',n_init =10,max_iter=300,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("elbow method")
plt.xlabel("number of clusters")
plt.ylabel("wcss")

#apply k mean clustering
kmeans = KMeans(n_clusters =5,init='k-means++',n_init =10,max_iter=300,random_state=0)
y_kmean = kmeans.fit_predict(x)

#visualizing the clusters
plt.scatter(x[y_kmean==0, 0],x[y_kmean==0, 1], c="cyan", label= "cluster1")
plt.scatter(x[y_kmean==1, 0],x[y_kmean==1, 1], c="r", label= "cluster2")
plt.scatter(x[y_kmean==2, 0],x[y_kmean==2, 1], c="b", label= "cluster3")
plt.scatter(x[y_kmean==3, 0],x[y_kmean==3, 1], c="y", label= "cluster4")
plt.scatter(x[y_kmean==4, 0],x[y_kmean==4, 1], c="magenta", label= "cluster5")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s = 350,color = "green" )
plt.title("k_mean_clustering")
plt.xlabel("annual income")
plt.ylabel("spending score")
plt.show()




