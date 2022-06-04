#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.metrics import jaccard_score
from scipy import spatial
import h5py
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
import os
from Explain.Predict_new_game import start


# In[ ]:


def averaging(a):
    
    avg_a = []
    first=0
    center=0
    last=0
    for i in range(0, 4):
        for j in range(0, 2):
            avg_ = ((a[i][j] + a[i+1][j] + a[i][j+1] + a[i+1][j+1])/4)
            avg_a.append(avg_)
            
            if j == 0:
                first=first+a[i][j]
            if j == 1:
                center=center+a[i][j]
                last=last+a[i][j+1]
    
    avg_a.append(first)
    avg_a.append(center)
    avg_a.append(last)
    
    return avg_a


# In[ ]:


def transform_and_concat(dt):

    print("Trasforming data and extracting centroids")
    
    res = np.zeros((len(dt['pred']), 2, 11))

    for i in tqdm(range(0, len(dt['pred']))):
        for j in range(0, 2):
            res[i][j]=averaging(dt['pred'][i][j])
            
    sum_ = np.zeros((len(dt['pred']), 22))
    for i in range(0, len(res)):
        sum_[i]=np.concatenate((res[i][0], res[i][1]))
        
    index = [i for i in range(0, len(dt['pred']))]

    print("\n")
    
    return res, sum_, index


# In[ ]:


def make_cluster(data, frame, n_cluster):

    cluster = KMeans(n_clusters=n_cluster)
    result=cluster.fit_predict(data[:frame])
    
    return cluster


# In[ ]:


def main_get_centroid(Team):
    
    if not(os.path.isfile("Files/"+Team+"/Testing/"+Team+"_testing_pred.h5")):

        print("\n")
        print("Dataset for testing doesn't exist, start extracting")
        print("\n")
        
        start(Team)

        print("\n")
        print("Extracted")
        print("\n")

    else:
        print("Data for testing already created")
        print("\n")
    
    dt = h5py.File("Files/"+Team+"/Testing/"+Team+"_testing_pred.h5", 'r')
    
    res, sum_, index = transform_and_concat(dt)
    
    res2, sum_2, index2 = shuffle(res, sum_, index)
    
    cluster = make_cluster(sum_2, 10000, 4)
    
    
    centroid = np.round(cluster.cluster_centers_, 2)
    
    return centroid


# In[ ]:




