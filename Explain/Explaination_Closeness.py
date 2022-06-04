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
import copy as cp
import params
import torch
from ST_ResNet import STResNet
from path import *
from mplsoccer.pitch import Pitch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from Explain.Cluster_centroid import main_get_centroid


# In[2]:


np.set_printoptions(suppress=True)


# In[4]:


#LIMIT

def construct_limit(row, col):
    
    """
    Get as input the number of rows and columns of the grid and construct the coordinates of the grid.
    """

    #Coordinates of the grid [1:4] and ID of the cell [0]
    
    #Start limits
    start_col=-52.5
    start_row=34
    limit = np.zeros((row*col, 5))

    #Set the ID in the coordinates vector
    for i in range(0, col*row):
        limit[i][0] = i

    #Set the shift for row and column
    col_shift = 105/col
    row_shift = 68/row
    
    #Impose column limits coordinates
    for i in range(0, row):
        start_col=-52.5
        i = i*col
        for j in range(0, col):
            limit[i+j][1] = start_col + col_shift
            limit[i+j][2] = start_col
            start_col = start_col + col_shift

    #Impose row coordinates
    for i in range (0, row):
        i = i*col
        for j in range(0, col):
            limit[i+j][3] = start_row
            limit[i+j][4] = start_row - row_shift
        start_row = start_row - row_shift
    
    #Increase the boundaries limit to include also the players out of the field.
    for i in range(0, col*row):
        for j in range(0, 5):
            if(limit[i][j] == -34 or limit[i][j] == -52.5):
                limit[i][j] = limit[i][j]-2
            if(limit[i][j] == 34 or limit[i][j] == 52.5):
                limit[i][j] = limit[i][j]+2
    
    return limit

def construct_limit_no_inc(row, col):
    
    """
    Get as input the number of rows and columns of the grid and construct the coordinates of the grid.
    """

    #Coordinates of the grid [1:4] and ID of the cell [0]
    
    #Start limits
    start_col=-52.5
    start_row=34
    limit = np.zeros((row*col, 5))

    #Set the ID in the coordinates vector
    for i in range(0, col*row):
        limit[i][0] = i

    #Set the shift for row and column
    col_shift = 105/col
    row_shift = 68/row
    
    #Impose column limits coordinates
    for i in range(0, row):
        start_col=-52.5
        i = i*col
        for j in range(0, col):
            limit[i+j][1] = start_col + col_shift
            limit[i+j][2] = start_col
            start_col = start_col + col_shift

    #Impose row coordinates
    for i in range (0, row):
        i = i*col
        for j in range(0, col):
            limit[i+j][3] = start_row
            limit[i+j][4] = start_row - row_shift
        start_row = start_row - row_shift
    
    return limit




# In[6]:


#FUNCTION TRANSFORM

def get_frames(dt):
    
    """
    Get the data of the first closeness frame
    """
    
    c = np.zeros((len(dt['X_c']), 3, 5, 3))
    for i in (range(0, len(dt['X_c']))):
        c[i] = dt['X_c'][i][:3]
    
    return c


def averaging(a):
    
    """
    Used to construct the index in transform and concat for each matrix
    """
    
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


def transform_and_concat(c):
    
    """
    Transoform the data of the frame used into the index constructed and concate the indexes of 
     analyzed team and opponent team.
    """
    
    """
    The index is composed by the average of the subzone(2x2 cells) and at the end the sum of the 
     number of player in each third
    """
    
    res = np.zeros((len(c), 2, 11))

    for i in (range(0, len(c))):
        for j in range(0, 2):
            res[i][j]=averaging(c[i][j])
            
    sum_ = np.zeros((len(c), 22))
    for i in range(0, len(res)):
        sum_[i]=np.concatenate((res[i][0], res[i][1]))
        
    index = [i for i in range(0, len(c))]
    
    return res, sum_, index 


# In[7]:


#FUNCTION CLUSTER

def get_cluster(sum_, centroids):
    
    """
    Assign each frame to a cluster basing on the cosine distance
    """
    
    print("Assigning each frame to a cluster")
    
    cluster = []
    for i in tqdm(range(0, len(sum_))):
        max_=-99999
        max_ind=None
        for j in range(0, len(centroids)):
            if (1 - spatial.distance.cosine(centroids[j], sum_[i]))>max_:
                max_= (1 - spatial.distance.cosine(centroids[j], sum_[i]))
                max_ind=j
        cluster.append(max_ind)
    
    return cluster 


def append_cluster(dt, cluster, meta):
    
    """
    Based on the cluster constructed create the different training data
    """
    
    c0 = []
    p0 = []
    meta0 = []

    c1 = []
    p1 = []
    meta1 = []

    c2 = []
    p2 = []
    meta2 = []

    c3 = []
    p3 = []
    meta3 = []

    for i in (range(0, len(dt['X_c']))):
        if cluster[i]==0:
            c0.append(dt['X_c'][i])
            p0.append(dt['X_p'][i])
            meta0.append(meta[i])
        elif cluster[i]==1:
            c1.append(dt['X_c'][i])
            p1.append(dt['X_p'][i])
            meta1.append(meta[i])
        elif cluster[i]==2:
            c2.append(dt['X_c'][i])
            p2.append(dt['X_p'][i])
            meta2.append(meta[i])
        elif cluster[i]==3:
            c3.append(dt['X_c'][i])
            p3.append(dt['X_p'][i])
            meta3.append(meta[i])
    
    print("\n")
            
    return c0, p0, meta0, c1, p1, meta1, c2, p2, meta2, c3, p3, meta3 


# In[8]:


#FUNCTION VISUALIZATION 

def average_cluster(c0):
    
    """
    See the average disposition of the players in each the cluster
    """
    
    
    add = np.zeros((2, 5, 3))
    ball_c0 = np.zeros((5, 3))
    for i in range(0, len(c0)):
        add = add + c0[i][:2]
        ball_c0 = ball_c0 + c0[i][2]
    avg_c0 = np.round(add/len(c0), 0)
    
    return avg_c0, ball_c0 


def plot(c):
    
    #Get the prediction we need
    pred = c
    
    row=params.row
    col=params.col
    
    #Get the limits
    limit=construct_limit(params.row, params.col)

    #Get vertical line to plot
    ver = set()
    for i in range(0, len(limit)):
        if limit[i][1] != 54.5 and limit[i][1] != -54.5:
            ver.add(limit[i][1])

    #Get horizontal line to plot
    hor = set()
    for i in range(0, len(limit)):
        if limit[i][3] != 36 and limit[i][3] != -36:
            hor.add(limit[i][3])

    #Get the shift for printing in the X axis
    i=0
    for el in sorted(ver):
        if i == 0:
            firstV = el
        elif i == 1:
            secondV = el
        i=i+1

    #Get the shift for printing in the Y axis
    i=0
    for el in sorted(hor):
        if i == 0:
            firstH = el
        elif i == 1:
            secondH = el
        i=i+1

    #Calculate the shift
    shiftX=(max(firstV, secondV) - min(firstV, secondV))*100  #col
    shiftY=(max(firstH, secondH) - min(firstH, secondH))*100  #row


    #Get the index for print the information
    indexes = []
    for i in range(0, row):
        for j in range(0, col):
            indexes.append([i, j])

    #Get the start value of the indication and of the number to print
    X_start = np.average([-5600, min(ver)*100])
    Y_start = np.average([3400, max(hor)*100])
    
    name = ['analyzed team', 'opponent team']

    #for c in range(0, 2):

    #print('Prediction of the occupancy of pitch for: ', name[c])

    #Plot the pitch
    pitch = Pitch(pitch_type='tracab', pitch_length=105, pitch_width=68, pitch_color='grass')
    fig, ax = pitch.draw()
    #Plot vertical lines of the grid
    for i in ver:
        plt.vlines(i*100, 3400, -3400)
    #Plot horizontal lines of the grid
    for i in hor:
        plt.hlines(i*100, -5250, 5250)
    #Plot the final result in the pitch
    for r in range(0, row):
        for t in range(0, col):
            plt.annotate(int(pred[0][r][t]), 
                         xy=(X_start+shiftX*t-1000, Y_start-shiftY*r), 
                         fontsize=20,  color='Black', label='Analyzed')
            plt.annotate(int(pred[1][r][t]), 
                         xy=(X_start+shiftX*t+1000, Y_start-shiftY*r), 
                         fontsize=20,  color='white', label='Opponent')
    
    
    w = mpatches.Patch(color='black', alpha=1, label='Analyzed', fill=True)
    b = mpatches.Patch(color='black', alpha=1, label='Opponent', fill=False)
    legend1 = plt.legend(handles=[w, b], bbox_to_anchor=(1.3, 1),
                         ncol=1, edgecolor="black")
    plt.gca().add_artist(legend1)
    
    plt.show()
    print("\n")  
    
    
def plot_changes(steps, point, direction, col_, row_, value, g, team_, ind):
    
    row=params.row
    col=params.col

    #Construct the limits
    limit=construct_limit(params.row, params.col)

    #Get vertical line to plot
    ver = set()
    for i in range(0, len(limit)):
        if limit[i][1] != 54.5 and limit[i][1] != -54.5:
            ver.add(limit[i][1])

    #Get horizontal line to plot
    hor = set()
    for i in range(0, len(limit)):
        if limit[i][3] != 36 and limit[i][3] != -36:
            hor.add(limit[i][3])

    #Get the shift for printing in the X axis
    i=0
    for el in sorted(ver):
        if i == 0:
            firstV = el
        elif i == 1:
            secondV = el
        i=i+1

    #Get the shift for printing in the Y axis
    i=0
    for el in sorted(hor):
        if i == 0:
            firstH = el
        elif i == 1:
            secondH = el
        i=i+1

    #Calculate the shift
    shiftX=(max(firstV, secondV) - min(firstV, secondV))*100  #col
    shiftY=(max(firstH, secondH) - min(firstH, secondH))*100  #row


    #Get the index for print the information
    indexes = []
    for i in range(0, row):
        for j in range(0, col):
            indexes.append([i, j])

    #Get the start value of the indication and of the number to print
    X_start = np.average([-5600, min(ver)*100])
    Y_start = np.average([3400, max(hor)*100])

    #Plot the pitch
    fig, ax =plt.subplots(figsize=(15,10))
    pitch = Pitch(pitch_type='tracab', figsize=(50,10), pitch_length=105, pitch_width=68) 
    pitch.draw(ax=ax)

    #Plot vertical lines of the grid
    for i in ver:
        plt.vlines(i*100, 3400, -3400, linestyles='dashed', linewidth=0.5)
    #Plot horizontal lines of the grid
    for i in hor:
        plt.hlines(i*100, -5250, 5250, linestyles='dashed', linewidth=0.5)


    limit_no_inc = construct_limit_no_inc(row, col)
    
    if g == [1, 4]:
        color = ['darkred', 'palegreen', 'limegreen', 'seagreen', 'darkgreen']
    elif g == [2, 3]:
        color = ['darkred', 'tomato', 'limegreen', 'seagreen', 'darkgreen']
    elif g == [3, 2]:
        color = ['darkred', 'red', 'tomato', 'limegreen', 'darkgreen']
    elif g == [4, 1]:
        color = ['darkred', 'red', 'tomato', 'lightsalmon', 'darkgreen']
        
     
    for r in range(0, row):
        for t in range(0, col):
            if point[0][r][t]==0:
                rect=mpatches.Rectangle((limit_no_inc[r*3+t][2]*100, limit_no_inc[r*3+t][3]*100-shiftY),shiftX, shiftY, 
                        #fill=False,
                        alpha=0.5,
                       facecolor=color[0])
                plt.gca().add_patch(rect)
            elif point[0][r][t]==1:
                rect=mpatches.Rectangle((limit_no_inc[r*3+t][2]*100, limit_no_inc[r*3+t][3]*100-shiftY),shiftX, shiftY, 
                        #fill=False,
                        alpha=0.5,
                       facecolor=color[1])
                plt.gca().add_patch(rect)
            elif point[0][r][t]==2:
                rect=mpatches.Rectangle((limit_no_inc[r*3+t][2]*100, limit_no_inc[r*3+t][3]*100-shiftY),shiftX, shiftY, 
                        #fill=False,
                        alpha=0.5,
                       facecolor=color[2])
                plt.gca().add_patch(rect)
            elif point[0][r][t]==3:
                rect=mpatches.Rectangle((limit_no_inc[r*3+t][2]*100, limit_no_inc[r*3+t][3]*100-shiftY),shiftX, shiftY, 
                        #fill=False,
                        alpha=0.5,
                       facecolor=color[3])
                plt.gca().add_patch(rect)
            elif point[0][r][t]==4:
                rect=mpatches.Rectangle((limit_no_inc[r*3+t][2]*100, limit_no_inc[r*3+t][3]*100-shiftY),shiftX, shiftY, 
                        #fill=False,
                        alpha=0.5,
                       facecolor=color[4])
                plt.gca().add_patch(rect)

    
    if team_==0:
        
        if direction == 'forward':
            row_2=row_
            col_2=col_+1
            plt.arrow(X_start+shiftX*col_, Y_start-shiftY*row_, shiftX, 0, 
                  width=0.5, head_width=100, length_includes_head=True, color='blue')
            plt.annotate("-"+str(value), xy=(X_start+shiftX*col_-330, Y_start-shiftY*row_-100), fontsize=20)
            plt.annotate("+"+str(value), xy=(X_start+shiftX*col_2+150, Y_start-shiftY*row_-100), fontsize=20)
        elif direction == 'behind':
            row_2=row_
            col_2=col_-1
            plt.arrow(X_start+shiftX*col_, Y_start-shiftY*row_, -shiftX, 0, 
                  width=0.5, head_width=100, length_includes_head=True, color='blue')
            plt.annotate("-"+str(value), xy=(X_start+shiftX*col_+100, Y_start-shiftY*row_-80), fontsize=20)
            plt.annotate("+"+str(value), xy=(X_start+shiftX*col_2-580, Y_start-shiftY*row_-80), fontsize=20)
        elif direction == 'right':
            row_2=row_+1
            col_2=col_
            plt.arrow(X_start+shiftX*col_, Y_start-shiftY*row_, 0, -shiftY, 
                  width=0.5, head_width=100, length_includes_head=True, color='blue')
            plt.annotate("-"+str(value), xy=(X_start+shiftX*col_-150, Y_start-shiftY*row_+180), fontsize=20)
            plt.annotate("+"+str(value), xy=(X_start+shiftX*col_-200, Y_start-shiftY*row_2-250), fontsize=20)
        elif direction == 'left':
            row_2=row_-1
            col_2=col_
            plt.arrow(X_start+shiftX*col_, Y_start-shiftY*row_, 0, shiftY, 
                  width=30, head_width=100, length_includes_head=True, color='blue')
            plt.annotate("-"+str(value), xy=(X_start+shiftX*col_-180, Y_start-shiftY*row_-250), fontsize=20)
            plt.annotate("+"+str(value), xy=(X_start+shiftX*col_-200, Y_start-shiftY*row_2+200), fontsize=20)
        elif direction == 'forward-left':
            row_2=row_-1
            col_2=col_+1
            plt.arrow(X_start+shiftX*col_, Y_start-shiftY*row_, shiftX, shiftY, 
                  width=0.5, head_width=100, length_includes_head=True, color='blue')
            plt.annotate("-"+str(value), xy=(X_start+shiftX*col_-550, Y_start-shiftY*row_-100), fontsize=20)
            plt.annotate("+"+str(value), xy=(X_start+shiftX*col_2-100, Y_start-shiftY*row_2), fontsize=20)
        elif direction == 'forward-right':
            row_2=row_+1
            col_2=col_+1
            plt.arrow(X_start+shiftX*col_, Y_start-shiftY*row_, shiftX, -shiftY, 
                  width=0.5, head_width=100, length_includes_head=True, color='blue')
            plt.annotate("-"+str(value), xy=(X_start+shiftX*col_-550, Y_start-shiftY*row_-100), fontsize=20)
            plt.annotate("+"+str(value), xy=(X_start+shiftX*col_2-100, Y_start-shiftY*row_2-250), fontsize=20)
        elif direction == 'behind-left':
            row_2=row_-1
            col_2=col_-1
            plt.arrow(X_start+shiftX*col_, Y_start-shiftY*row_, -shiftX, shiftY, 
                  width=0.5, head_width=100, length_includes_head=True, color='blue')
            plt.annotate("-"+str(value), xy=(X_start+shiftX*col_+100, Y_start-shiftY*row_-250), fontsize=20)
            plt.annotate("+"+str(value), xy=(X_start+shiftX*col_2-580, Y_start-shiftY*row_2+200), fontsize=20)
        elif direction == 'behind-right':
            row_2=row_+1
            col_2=col_-1
            plt.arrow(X_start+shiftX*col_, Y_start-shiftY*row_, -shiftX, -shiftY, 
                  width=0.5, head_width=100, length_includes_head=True, color='blue')
            plt.annotate("-"+str(value), xy=(X_start+shiftX*col_+100, Y_start-shiftY*row_+180), fontsize=20)
            plt.annotate("+"+str(value), xy=(X_start+shiftX*col_2-580, Y_start-shiftY*row_2-250), fontsize=20)




    g0 = mpatches.Patch(color=color[0], alpha=.5, label=str(np.round(steps[0][1], 3))+"/"+str(np.round(steps[0][2], 3)))
    g1 = mpatches.Patch(color=color[1], alpha=.5, label=str(np.round(steps[1][1], 3))+"/"+str(np.round(steps[1][2], 3)))
    g2 = mpatches.Patch(color=color[2], alpha=.5, label=str(np.round(steps[2][1], 3))+"/"+str(np.round(steps[2][2], 3)))
    g3 = mpatches.Patch(color=color[3], alpha=.5, label=str(np.round(steps[3][1], 3))+"/"+str(np.round(steps[3][2], 3)))
    g4 = mpatches.Patch(color=color[4], alpha=.5, label=str(np.round(steps[4][1], 3))+"/"+str(np.round(steps[4][2], 3)))

    legend1 = plt.legend(handles=[g0, g1, g2, g3, g4], bbox_to_anchor=(0.99, -0.01), 
                         ncol=5, edgecolor="black", prop={'size': 15})
    plt.gca().add_artist(legend1)

    plt.title(f"Average changes in each cell after the changes of the players location, Analyzed team, first closeness frame, Cluster {ind}", fontsize=15)
    plt.show()
    

def plot_changes2(steps, point, direction, col_, row_, value, g, team_, ind):
    row=params.row
    col=params.col

    #Construct the limits
    limit=construct_limit(params.row, params.col)

    #Get vertical line to plot
    ver = set()
    for i in range(0, len(limit)):
        if limit[i][1] != 54.5 and limit[i][1] != -54.5:
            ver.add(limit[i][1])

    #Get horizontal line to plot
    hor = set()
    for i in range(0, len(limit)):
        if limit[i][3] != 36 and limit[i][3] != -36:
            hor.add(limit[i][3])

    #Get the shift for printing in the X axis
    i=0
    for el in sorted(ver):
        if i == 0:
            firstV = el
        elif i == 1:
            secondV = el
        i=i+1

    #Get the shift for printing in the Y axis
    i=0
    for el in sorted(hor):
        if i == 0:
            firstH = el
        elif i == 1:
            secondH = el
        i=i+1

    #Calculate the shift
    shiftX=(max(firstV, secondV) - min(firstV, secondV))*100  #col
    shiftY=(max(firstH, secondH) - min(firstH, secondH))*100  #row


    #Get the index for print the information
    indexes = []
    for i in range(0, row):
        for j in range(0, col):
            indexes.append([i, j])

    #Get the start value of the indication and of the number to print
    X_start = np.average([-5600, min(ver)*100])
    Y_start = np.average([3400, max(hor)*100])

    #Plot the pitch
    fig, ax =plt.subplots(figsize=(15,10))
    pitch = Pitch(pitch_type='tracab', figsize=(50,10), pitch_length=105, pitch_width=68) 
    pitch.draw(ax=ax)

    #Plot vertical lines of the grid
    for i in ver:
        plt.vlines(i*100, 3400, -3400, linestyles='dashed', linewidth=0.5)
    #Plot horizontal lines of the grid
    for i in hor:
        plt.hlines(i*100, -5250, 5250, linestyles='dashed', linewidth=0.5)


    limit_no_inc = construct_limit_no_inc(row, col)
    
    if g == [1, 4]:
        color = ['darkred', 'palegreen', 'limegreen', 'seagreen', 'darkgreen']
    elif g == [2, 3]:
        color = ['darkred', 'tomato', 'limegreen', 'seagreen', 'darkgreen']
    elif g == [3, 2]:
        color = ['darkred', 'red', 'tomato', 'limegreen', 'darkgreen']
    elif g == [4, 1]:
        color = ['darkred', 'red', 'tomato', 'lightsalmon', 'darkgreen']
        
     
    for r in range(0, row):
        for t in range(0, col):
            if point[1][r][t]==0:
                rect=mpatches.Rectangle((limit_no_inc[r*3+t][2]*100, limit_no_inc[r*3+t][3]*100-shiftY),shiftX, shiftY, 
                        #fill=False,
                        alpha=0.5,
                       facecolor=color[0])
                plt.gca().add_patch(rect)
            elif point[1][r][t]==1:
                rect=mpatches.Rectangle((limit_no_inc[r*3+t][2]*100, limit_no_inc[r*3+t][3]*100-shiftY),shiftX, shiftY, 
                        #fill=False,
                        alpha=0.5,
                       facecolor=color[1])
                plt.gca().add_patch(rect)
            elif point[1][r][t]==2:
                rect=mpatches.Rectangle((limit_no_inc[r*3+t][2]*100, limit_no_inc[r*3+t][3]*100-shiftY),shiftX, shiftY, 
                        #fill=False,
                        alpha=0.5,
                       facecolor=color[2])
                plt.gca().add_patch(rect)
            elif point[1][r][t]==3:
                rect=mpatches.Rectangle((limit_no_inc[r*3+t][2]*100, limit_no_inc[r*3+t][3]*100-shiftY),shiftX, shiftY, 
                        #fill=False,
                        alpha=0.5,
                       facecolor=color[3])
                plt.gca().add_patch(rect)
            elif point[1][r][t]==4:
                rect=mpatches.Rectangle((limit_no_inc[r*3+t][2]*100, limit_no_inc[r*3+t][3]*100-shiftY),shiftX, shiftY, 
                        #fill=False,
                        alpha=0.5,
                       facecolor=color[4])
                plt.gca().add_patch(rect)

    
    if team_==1:
        if direction == 'forward':
            row_2=row_
            col_2=col_+1
            plt.arrow(X_start+shiftX*col_, Y_start-shiftY*row_, shiftX, 0, 
                  width=0.5, head_width=100, length_includes_head=True, color='blue')
            plt.annotate("-"+str(value), xy=(X_start+shiftX*col_-330, Y_start-shiftY*row_-100), fontsize=20)
            plt.annotate("+"+str(value), xy=(X_start+shiftX*col_2+150, Y_start-shiftY*row_-100), fontsize=20)
        elif direction == 'behind':
            row_2=row_
            col_2=col_-1
            plt.arrow(X_start+shiftX*col_, Y_start-shiftY*row_, -shiftX, 0, 
                  width=0.5, head_width=100, length_includes_head=True, color='blue')
            plt.annotate("-"+str(value), xy=(X_start+shiftX*col_+100, Y_start-shiftY*row_-80), fontsize=20)
            plt.annotate("+"+str(value), xy=(X_start+shiftX*col_2-580, Y_start-shiftY*row_-80), fontsize=20)
        elif direction == 'right':
            row_2=row_+1
            col_2=col_
            plt.arrow(X_start+shiftX*col_, Y_start-shiftY*row_, 0, -shiftY, 
                  width=0.5, head_width=100, length_includes_head=True, color='blue')
            plt.annotate("-"+str(value), xy=(X_start+shiftX*col_-150, Y_start-shiftY*row_+180), fontsize=20)
            plt.annotate("+"+str(value), xy=(X_start+shiftX*col_-200, Y_start-shiftY*row_2-250), fontsize=20)
        elif direction == 'left':
            row_2=row_-1
            col_2=col_
            plt.arrow(X_start+shiftX*col_, Y_start-shiftY*row_, 0, shiftY, 
                  width=30, head_width=100, length_includes_head=True, color='blue')
            plt.annotate("-"+str(value), xy=(X_start+shiftX*col_-180, Y_start-shiftY*row_-250), fontsize=20)
            plt.annotate("+"+str(value), xy=(X_start+shiftX*col_-200, Y_start-shiftY*row_2+200), fontsize=20)
        elif direction == 'forward-left':
            row_2=row_-1
            col_2=col_+1
            plt.arrow(X_start+shiftX*col_, Y_start-shiftY*row_, shiftX, shiftY, 
                  width=0.5, head_width=100, length_includes_head=True, color='blue')
            plt.annotate("-"+str(value), xy=(X_start+shiftX*col_-550, Y_start-shiftY*row_-100), fontsize=20)
            plt.annotate("+"+str(value), xy=(X_start+shiftX*col_2-100, Y_start-shiftY*row_2), fontsize=20)
        elif direction == 'forward-right':
            row_2=row_+1
            col_2=col_+1
            plt.arrow(X_start+shiftX*col_, Y_start-shiftY*row_, shiftX, -shiftY, 
                  width=0.5, head_width=100, length_includes_head=True, color='blue')
            plt.annotate("-"+str(value), xy=(X_start+shiftX*col_-550, Y_start-shiftY*row_-100), fontsize=20)
            plt.annotate("+"+str(value), xy=(X_start+shiftX*col_2-100, Y_start-shiftY*row_2-250), fontsize=20)
        elif direction == 'behind-left':
            row_2=row_-1
            col_2=col_-1
            plt.arrow(X_start+shiftX*col_, Y_start-shiftY*row_, -shiftX, shiftY, 
                  width=0.5, head_width=100, length_includes_head=True, color='blue')
            plt.annotate("-"+str(value), xy=(X_start+shiftX*col_+100, Y_start-shiftY*row_-250), fontsize=20)
            plt.annotate("+"+str(value), xy=(X_start+shiftX*col_2-580, Y_start-shiftY*row_2+200), fontsize=20)
        elif direction == 'behind-right':
            row_2=row_+1
            col_2=col_-1
            plt.arrow(X_start+shiftX*col_, Y_start-shiftY*row_, -shiftX, -shiftY, 
                  width=0.5, head_width=100, length_includes_head=True, color='blue')
            plt.annotate("-"+str(value), xy=(X_start+shiftX*col_+100, Y_start-shiftY*row_+180), fontsize=20)
            plt.annotate("+"+str(value), xy=(X_start+shiftX*col_2-580, Y_start-shiftY*row_2-250), fontsize=20)




    g0 = mpatches.Patch(color=color[0], alpha=.5, label=str(np.round(steps[0][1], 3))+"/"+str(np.round(steps[0][2], 3)))
    g1 = mpatches.Patch(color=color[1], alpha=.5, label=str(np.round(steps[1][1], 3))+"/"+str(np.round(steps[1][2], 3)))
    g2 = mpatches.Patch(color=color[2], alpha=.5, label=str(np.round(steps[2][1], 3))+"/"+str(np.round(steps[2][2], 3)))
    g3 = mpatches.Patch(color=color[3], alpha=.5, label=str(np.round(steps[3][1], 3))+"/"+str(np.round(steps[3][2], 3)))
    g4 = mpatches.Patch(color=color[4], alpha=.5, label=str(np.round(steps[4][1], 3))+"/"+str(np.round(steps[4][2], 3)))

    legend1 = plt.legend(handles=[g0, g1, g2, g3, g4], bbox_to_anchor=(0.99, -0.01), 
                         ncol=5, edgecolor="black", prop={'size': 15})
    plt.gca().add_artist(legend1)

    plt.title(f"Average changes in each cell after the changes of the players location, Opponent team, first closeness frame, Cluster {ind}", fontsize=15)
    plt.show()
    

def plot_cluster(c0, c1, c2, c3):
    
    print("Cluster 0")
    plot(c0)
    print("Cluster 1")
    plot(c1)
    print("Cluster 2")
    plot(c2)
    print("Cluster 3")
    plot(c3)


# In[9]:


#FUNCTION PREDICT

class MinMaxNormalization(object):
    """
        MinMax Normalization --> [-1, 1]
        x = (x - min) / (max - min).
        x = x * 2 - 1
    """

    def __init__(self):
        pass

    def fit(self,):

        """
        Fit the data to the normalizer, get maximum and minimum
        """

        self._min = 0
        self._max = 10

        

        #print("min:", self._min, "max:", self._max)

    def transform(self, X):

        """
        Normalize the data
        """

        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1. #to put the value between -1 and 1
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):

        """
        De-Normalize the data
        """

        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X 
    

def load_model(min_, max_, ext, team):

    """
    Get as input the maximum value and the minimum value of the training data used to train the net
    Resume the model saved from the training phase
    Return the model
    """

    #Instanciate the model with the right hyperparameters 
    model = STResNet(learning_rate=params.lr,
        epoches=params.nb_epoch,
        batch_size=params.batch_size,
        len_closeness=params.len_closeness,
        len_period=params.len_period,
        #len_trend=len_trend,
        external_dim=np.shape(ext)[1],
        map_heigh=params.map_height,
        map_width=params.map_width,
        nb_flow=params.nb_flow,
        nb_residual_unit=params.nb_residual_unit,
        data_min = min_, 
        data_max = max_,
        team = team

    )
    
    
    #Load the satet of the model, thus all the weights trained
    model.load_state_dict(torch.load(get_model(team, params.nb_residual_unit, params.len_closeness, params.len_period)))
    
    return model


def predict(c, p, meta, team):
    
    """
    Predict
    """
    
    mmn = MinMaxNormalization()
    mmn.fit()

    c_norm = [mmn.transform(d) for d in c]
    p_norm = [mmn.transform(d) for d in p]


    X = []
    #Merge the internal component in one array
    for l, X_ in zip([params.len_clos, params.len_per], [c_norm, p_norm]):
        X.append(X_)

    X_torch = [torch.Tensor(x) for x in X]
    model1=load_model(0, 10, meta, team)

    y_pred = model1(X_torch[0], X_torch[1], torch.Tensor(np.array(meta)))

    #DE-NORMALIZED PREDICTION
    y_pred = (y_pred + 1.) / 2.
    y_pred = 1. * y_pred * (10 - 0) + 0

    y_pred = np.round(y_pred.detach())

    return y_pred 


def change_position(c, team, row_, col_, direction, value):
    
    """
    Switch the position between ceels
    """
    
    """
    Input:
    c: data
    team: 0-> analyzed, 1-> opponent
    row_: row of the cell to substract
    col_: col of the cell to substract
    direction: direct of the cell to increment
    value: value to change
    """
    

    global c_mod
    c_mod = cp.deepcopy(c)
    
    if direction == 'forward':
        row_2=row_
        col_2=col_+1
    elif direction == 'behind':
        row_2=row_
        col_2=col_-1
    elif direction == 'right':
        row_2=row_+1
        col_2=col_
    elif direction == 'left':
        row_2=row_-1
        col_2=col_
    elif direction == 'forward-left':
        row_2=row_-1
        col_2=col_+1
    elif direction == 'forward-right':
        row_2=row_+1
        col_2=col_+1
    elif direction == 'behind-left':
        row_2=row_-1
        col_2=col_-1
    elif direction == 'behind-right':
        row_2=row_+1
        col_2=col_-1

        
    for i in range(0, len(c_mod)):
        if c_mod[i][team][row_][col_] >= value:

            c_mod[i][team][row_][col_] = c_mod[i][team][row_][col_]-value
            c_mod[i][team][row_2][col_2] = c_mod[i][team][row_2][col_2]+value
        else:
            temp=c_mod[i][team][row_][col_]
            c_mod[i][team][row_][col_] = 0
            c_mod[i][team][row_2][col_2] = c_mod[i][team][row_2][col_2] + temp
    
    return c_mod 


def compare_step(diff):
    
    g = [[1, 4], [2, 3], [3, 2], [4, 1]]

    compare = 999999

    for i in range(0, len(g)):
        min_ = np.min(diff)
        max_ = np.max(diff)
        mi_d = min_/g[i][0]
        ma_d = max_/g[i][1]
        dif = np.absolute(mi_d-ma_d)
        
        if dif<compare:
            compare = dif
            out = i
    
    return g[out]


def get_step(diff):
    
    
    g = compare_step(diff)
    
    
    
    st_mi = np.absolute(np.min(diff)/g[0])
    st_ma = np.max(diff)/g[1]
    steps = []
    base = np.min(diff)
    for i in range(0, g[0]):
        steps.append([i, base, base+st_mi])
        base=base+st_mi
    steps[-1][2]=0
    
    base = 0
    for i in range(0, g[1]):
        steps.append([i+g[0], base, base+st_ma])
        base=base+st_ma
    
    return steps, g


def get_intervall(diff, steps):

    point = np.zeros((np.shape(diff)))

    for c in range(0, len(diff)):
        for r in range(0, len(diff[0])):
            for co in range(0, len(diff[0][0])):
                for l in range(1, len(steps)):
                    if diff[c][r][co] >= steps[l][1] and diff[c][r][co] <= steps[l][2]:
                        point[c][r][co] = steps[l][0]
    return point


def check_movements(col_, row_, direction):
    
    check = True
    
    if direction == 'forward':
        row_2=row_
        col_2=col_+1
    elif direction == 'behind':
        row_2=row_
        col_2=col_-1
    elif direction == 'right':
        row_2=row_+1
        col_2=col_
    elif direction == 'left':
        row_2=row_-1
        col_2=col_
    elif direction == 'forward-left':
        row_2=row_-1
        col_2=col_+1
    elif direction == 'forward-right':
        row_2=row_+1
        col_2=col_+1
    elif direction == 'behind-left':
        row_2=row_-1
        col_2=col_-1
    elif direction == 'behind-right':
        row_2=row_+1
        col_2=col_-1
        
    if row_2 >4 or row_2 <0 or col_2 >2 or col_2<0 or row_ >4 or row_ <0 or col_ >2 or col_ <0:
        check = False
        
    return check


# In[10]:


def main_tranform(dt):
    
    c = get_frames(dt)
    
    res, sum_, index = transform_and_concat(c)
    
    return res, sum_, index


# In[11]:


def main_cluster(dt, sum_, meta, centroids):
    
    clusters_ = []
    
    cluster = get_cluster(sum_, centroids)
    
    c0, p0, meta0, c1, p1, meta1, c2, p2, meta2, c3, p3, meta3 = append_cluster(dt, cluster, meta)
    
    avg_c0, ball_c0 = average_cluster(c0)
    avg_c1, ball_c1 = average_cluster(c1)
    avg_c2, ball_c2 = average_cluster(c2)
    avg_c3, ball_c3 = average_cluster(c3)
    
    clusters_.append([c0, p0, meta0])
    clusters_.append([c1, p1, meta1])
    clusters_.append([c2, p2, meta2])
    clusters_.append([c3, p3, meta3])
    
    
    return clusters_, avg_c0, ball_c0, avg_c1, ball_c1, avg_c2, ball_c2, avg_c3, ball_c3


# In[12]:


def main_predict(clusters_, ind, team_, row_, col_, direction, value, team):
    
    if check_movements(col_, row_, direction):
    
        c = clusters_[ind][0]

        p = clusters_[ind][1]

        meta = clusters_[ind][2]

        y_pred = predict(c, p, meta, team)

        c_mod = change_position(c, team_, row_, col_, direction, value)

        y_pred_mod = predict(c_mod, p, meta, team)


        diff = np.zeros((2, 5, 3))
        for i in range(0, len(y_pred)):
            diff = np.add(diff, (y_pred_mod[i][:2]-y_pred[i][:2]))
        diff = diff/len(y_pred)

        diff=diff.numpy()

        steps, g = get_step(diff)

        point = get_intervall(diff, steps)

        plot_changes(steps, point, direction, col_, row_, value, g, team_, ind)

        plot_changes2(steps, point, direction, col_, row_, value, g, team_, ind)
        
        print(diff)
    
    else:
        
        print("Invalid value of row and column accoridng to the direction, the resulting cell is out of the pitch")
    


# In[13]:


def main_explain_clos(team):
    
    centroids = main_get_centroid(team)
    
    dt = h5py.File("Files/"+team+"/Testing/"+team+"Testing.h5", 'r')
    
    meta=list(dt['meta'])
    
    res, sum_, index = main_tranform(dt)
    
    
    
    clusters_, avg_c0, ball_c0, avg_c1, ball_c1, avg_c2, ball_c2, avg_c3, ball_c3 = main_cluster(dt, sum_, meta, centroids)
    
    plot_cluster(avg_c0, avg_c1, avg_c2, avg_c3)
    
    
    
    return clusters_





