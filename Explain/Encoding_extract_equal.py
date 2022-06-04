#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import h5py
from tqdm import tqdm
import os
import pandas as pd
from mplsoccer.pitch import Pitch
import matplotlib.pyplot as plt
import params
import matplotlib.patches as mpatches
import random
from path import *


# In[2]:


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


# In[3]:


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


# In[4]:


def encode(dt):
    
    """
    Get as input the prediction done by the model.
    Encode the output into matrix with the first column wich refer to the analyzed team,
     the second refer to the opposite team and the third to the ball.
     Thus instead to have the the grid we will have a matrix [test_sample, channles (3), number_of_cells].
     res[0][0] means the number of player of the analyzed team in the cell 0.
     res[1][0] means the number of player of the opposite team in the cell 0.
     res[2][0] is 1 if the ball is in the cell 0, 0 otherwise.
    Return the encoded prediction. 
    """

    res = np.zeros((len(dt['pred']), 15, 2))

    for i in tqdm(range(0, len(dt['pred']))):
        for f in range(0, len(dt['pred'][0])-1):
            for r in range(0, 5):
                for c in range(0, 3):
                    res[i][r*3+c][f]=dt['pred'][i][f][r][c]
                    
    return res


# In[5]:


def delete_equal(couple):
    
    """
    Get as input the list of tuple which indicates the indexes of the equal prediction.
    Delete the tuple which reversed are equal.
    Return the list of tuple without duplicates. 
    """
    
    s = set()
    out = []
    for i in couple:
        t = tuple(i)
        if t in s or tuple(reversed(t)) in s:
            continue
        s.add(t)
        out.append(i)
        
    return out


# In[6]:


def check(i, j, team):
    
    ch = True
    
    tr=h5py.File("Files/"+str(team)+"/Testing/"+str(team)+"Testing.h5", 'r')
    
    if tr['Timestamp_y'][i][0].decode('utf8')==tr['Timestamp_y'][j][0].decode('utf8'):
        #print(tr['Timestamp_y'][i][0].decode('utf8'))
        #print(tr['Timestamp_y'][i][0].decode('utf8'))
        
        if tr['Timestamp_y'][i][2][:3].decode('utf8') == tr['Timestamp_y'][j][2][0:3].decode('utf8') or        int(tr['Timestamp_y'][i][2][:3].decode('utf8'))-1 == int(tr['Timestamp_y'][j][2][:3].decode('utf8')) or        int(tr['Timestamp_y'][i][2][:3].decode('utf8'))+1 == int(tr['Timestamp_y'][j][2][:3].decode('utf8')):
            ch = False
    
    return ch


# In[7]:


def get_equal(res, team):
    
    """
    Get as input the encoded prediction
    Construct the tuple which includes the indexes of two equal prediction.
    Return the list of tuples without duplicates
    """
    
    couple=[]
    for i in tqdm(range(0, len(res))):
        for j in range(i, len(res)):
            if i != j:
                if np.allclose(res[i], res[j]):
                    if check(i, j, team):
                        couple.append([i, j])
    
    #Delete duplicates
    out = delete_equal(couple)
    
    return out


# In[8]:


def list_equal(out):

    """
    Get as input the list of tuple without duplicates.
    Check if there is some case where more than two predictions are equal and list them.
    Return the final list record of indexes which indicates equal predicitons.
    """


    key = [[out[i][0]] for i in range(0, len(out))]

    key=np.unique([key])

    equal = []
    for i in range(0, len(key)):
        equal.append([key[i]])

    for i in range(0, len(equal)):
        for j in range(0, len(out)):
            if equal[i][0]==out[j][0]:
                equal[i].append(out[j][1])
    
    return equal


# In[9]:


def delete_closest_prediction(equal, team):
    
    """
    Get as input the list of record of indexes which indicates equal predicitons.
    Delete from this list the indexes that areclose each other and return this list. 
    """
    
    #Open the file for the timestamp
    tr=h5py.File("Files/"+str(team)+"/Testing/"+str(team)+"Testing.h5", 'r')
    
    #List that will contain the indexes which are close each other
    to_del2 = []
    for i in range(0, len(equal)):
        to_del = []
        for i1 in equal[i]:
            for i2 in equal[i]:
                if i1 != i2:
                    #Check if the timestamps are in the same minute or +/-1
                    if tr['Timestamp_y'][i1][0].decode('utf8')==tr['Timestamp_y'][i2][0].decode('utf8'):
                        if tr['Timestamp_y'][i1][2][:3].decode('utf8') == tr['Timestamp_y'][i2][2][:3].decode('utf8') or                            int(tr['Timestamp_y'][i1][2][:3].decode('utf8'))-1 == int(tr['Timestamp_y'][i2][2][:3].decode('utf8')) or                            int(tr['Timestamp_y'][i1][2][:3].decode('utf8'))+1 == int(tr['Timestamp_y'][i2][2][:3].decode('utf8')):
                                    to_del.append(i1)
        to_del2.append(to_del)
    
    #Get the unique indexes
    for i in range(0, len(to_del2)):
        to_del2[i]=np.unique(to_del2[i])
        to_del2[i]=to_del2[i].tolist()
    
    #Select randomly 1 element between all the indexes that are close
    for i in range(0, len(to_del2)):
        if len(to_del2[i]) != 0:
            to_del2[i].remove(random.choice(to_del2[i]))
    
    #Delete the indexes close except for the one previously selected
    for i in range(0, len(equal)):
        equal[i] = [a for a in equal[i] if a not in to_del2[i]]
    
    
    return equal


# In[10]:


def main_encode(dt, team):
    
    """
    Get as input the file with the predictions.
    Call all the functions to encode the predictions and extract the indexes of equal predictions.
    Return the final list of record of indexes which indicates equal predicitons.
    """

    #Encode the predictions 
    res = encode(dt)

    #Get tuples of equal predictions
    out = get_equal(res, team)

    #Check for more than two equal prediction.
    equal = list_equal(out)

    equal2 = delete_closest_prediction(equal, team)
    
    return equal2


# In[11]:


def see_equal_prediction(dt, indexes):
    
    """
    Get as input the file with the prediction and a list with indexes of equa prediction.
    Plot the predicted number of players in each cell divided for team and the ball.
    """
    
    #Get the prediction we need
    pred = dt['pred'][indexes[0]]
    
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
    
    name = ['analyzed team', 'opponent team', 'ball']

    for c in range(0, 3):

        print('Prediction of the occupancy of pitch for: ', name[c])

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
                plt.annotate(int(pred[c][r][t]), 
                             xy=(X_start+shiftX*t, Y_start-shiftY*r), 
                             fontsize=20,  color='Black')
        plt.show()
        print("\n")  


# In[12]:


def plot_movements_pred(dt, tr_data, match_id, mult_all, team_id, index_p):
    
    """
    Get as input the data of training for a given prediction and the match ID.
    Plot the movements of the players through the various frames.
    """
    
    #Open the file with the player informations
    info = pd.read_json("Matches/Data/"+str(match_id)+"Data_match.json", lines=True)
    
    team=[]

    #Obtain the id of the two teams and get the players of each of them
    for i in range(0, len(info['players'][0])):
        team.append(info['players'][0][i]['team_id'])
    team=np.unique(team)
    team=np.setdiff1d(team, team_id)

    team_0_id=team_id
    team_1_id=team[0]

    team_0 = []
    team_1 = []
    for i in range(0, len(info['players'][0])):
        #if info['players'][0][i]['player_role']['acronym'] != 'GK':
        if info['players'][0][i]['team_id'] == team_0_id:
            team_0.append(info['players'][0][i]['trackable_object'])
        elif info['players'][0][i]['team_id'] == team_1_id:
            team_1.append(info['players'][0][i]['trackable_object'])
                
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
    pitch = Pitch(pitch_type='tracab', figsize=(50,10), pitch_length=105, pitch_width=68) #, axis=True, label=True
    pitch.draw(ax=ax)

    #Plot vertical lines of the grid
    for i in ver:
        plt.vlines(i*100, 3400, -3400, linestyles='dashed', linewidth=0.5)
    #Plot horizontal lines of the grid
    for i in hor:
        plt.hlines(i*100, -5250, 5250, linestyles='dashed', linewidth=0.5)
    
    #color1 = ['red', 'orange', 'black']
    #color2 = ['blue', 'green', 'yellow']
    
    col=0
    
    for j in range(0, len(tr_data)):
        
        #Get the data of one index
        data=tr_data[j]['data']
        mult=mult_all[j]
        
        if j==0 or j==2 or j==4:
            
            #Iterate though the tracking data of each players
            for i in range(0, len(data)):
                
                #Plot players of analyzed team
                if data[i]['trackable_object'] in team_0:
                    ax.scatter(data[i]["x"]*100*mult, data[i]["y"]*100*mult, color='red', marker='.', s=50)
                
                #Plot players of opponent team
                elif data[i]['trackable_object'] in team_1:
                    ax.scatter(data[i]["x"]*100*mult, data[i]["y"]*100*mult, color='blue', marker='1', s=50)
                    
                #Plot the ball position
                elif data[i]['trackable_object'] == 55:
                    ax.scatter(data[i]["x"]*100*mult, data[i]["y"]*100*mult, color='orange', s=50)
            
            col=col+1
        
        #Plot the arrow
        if j==2 or j==4:
            
            #Get the data of the start of the arrow
            data_st=tr_data[j-2]['data']
            mult=mult_all[j]
            
            for i in range(0, len(data)):
                
                #Plot arrow for analyzed team
                if data[i]['trackable_object'] in team_0 and data_st[i]['trackable_object'] in team_0:
                    plt.arrow(data_st[i]["x"]*100*mult, data_st[i]["y"]*100*mult, 
                      (-data_st[i]["x"]*100*mult+data[i]["x"]*100*mult), 
                              (-data_st[i]["y"]*100*mult+data[i]["y"]*100*mult), 
                      width=0.05, head_width=100, length_includes_head=True, color='red')
            
                #Plot arrow for opponent team
                elif data[i]['trackable_object'] in team_1 and data_st[i]['trackable_object'] in team_1:
                    plt.arrow(data_st[i]["x"]*100*mult, data_st[i]["y"]*100*mult, 
                      (-data_st[i]["x"]*100*mult+data[i]["x"]*100*mult),
                              (-data_st[i]["y"]*100*mult+data[i]["y"]*100*mult), 
                      width=0.05, head_width=100, length_includes_head=True, color='blue')
                
                #Plot arrow for the ball
                elif data[i]['trackable_object'] == 55 and data_st[i]['trackable_object'] == 55:
                    plt.arrow(data_st[i]["x"]*100*mult, data_st[i]["y"]*100*mult, 
                      (-data_st[i]["x"]*100*mult+data[i]["x"]*100*mult),
                              (-data_st[i]["y"]*100*mult+data[i]["y"]*100*mult), 
                      width=0.05, head_width=100, length_includes_head=True, color='orange')
    
    
    """
    pred_an = dt['pred'][index_p][0]
    for r in range(0, row):
        for t in range(0, col):
            if int(pred_an[r][t])==1:
                ax.scatter(X_start+shiftX*t+1500, Y_start-shiftY*r-300, color='green')
            elif int(pred_an[r][t])>1:
                for a in range(0, int(pred_an[r][t])):
                    ax.scatter(X_start+shiftX*t+1500-(a*150), Y_start-shiftY*r-300, color='green')
    
    pred_op = dt['pred'][index_p][1]
    for r in range(0, row):
        for t in range(0, col):
            if int(pred_op[r][t])==1:
                ax.scatter(X_start+shiftX*t+1500, Y_start-shiftY*r-500, color='purple')
            elif int(pred_op[r][t])>1:
                for a in range(0, int(pred_op[r][t])):
                    ax.scatter(X_start+shiftX*t+1500-(a*150), Y_start-shiftY*r-500, color='purple')
    """
    
    limit_no_inc = construct_limit_no_inc(row, col)
    pred_op = dt['pred'][index_p][1]
    pred_an = dt['pred'][index_p][0]
    for r in range(0, row):
        for t in range(0, col):
            if int(pred_an[r][t])==0 and int(pred_op[r][t])==0:
                rect=mpatches.Rectangle((limit_no_inc[r*3+t][2]*100, limit_no_inc[r*3+t][3]*100-shiftY),shiftX, shiftY, 
                        #fill=False,
                        alpha=0.2,
                       facecolor="white")
                plt.gca().add_patch(rect)
            elif int(pred_an[r][t])-int(pred_op[r][t])==1:
                rect=mpatches.Rectangle((limit_no_inc[r*3+t][2]*100, limit_no_inc[r*3+t][3]*100-shiftY),shiftX, shiftY, 
                        #fill=False,
                        alpha=0.2,
                       facecolor="green")
                plt.gca().add_patch(rect)
            elif int(pred_an[r][t])-int(pred_op[r][t])==2:
                rect=mpatches.Rectangle((limit_no_inc[r*3+t][2]*100, limit_no_inc[r*3+t][3]*100-shiftY),shiftX, shiftY, 
                        #fill=False,
                        alpha=0.35,
                       facecolor="green")
                plt.gca().add_patch(rect)
            elif int(pred_an[r][t])-int(pred_op[r][t])==3:
                rect=mpatches.Rectangle((limit_no_inc[r*3+t][2]*100, limit_no_inc[r*3+t][3]*100-shiftY),shiftX, shiftY, 
                        #fill=False,
                        alpha=0.5,
                       facecolor="green")
                plt.gca().add_patch(rect)
            elif int(pred_an[r][t])-int(pred_op[r][t])==-1:
                rect=mpatches.Rectangle((limit_no_inc[r*3+t][2]*100, limit_no_inc[r*3+t][3]*100-shiftY),shiftX, shiftY, 
                        #fill=False,
                        alpha=0.2,
                       facecolor="purple")
                plt.gca().add_patch(rect)
            elif int(pred_an[r][t])-int(pred_op[r][t])==-2:
                rect=mpatches.Rectangle((limit_no_inc[r*3+t][2]*100, limit_no_inc[r*3+t][3]*100-shiftY),shiftX, shiftY, 
                        #fill=False,
                        alpha=0.35,
                       facecolor="purple")
                plt.gca().add_patch(rect)
            elif int(pred_an[r][t])-int(pred_op[r][t])==-3:
                rect=mpatches.Rectangle((limit_no_inc[r*3+t][2]*100, limit_no_inc[r*3+t][3]*100-shiftY),shiftX, shiftY, 
                        #fill=False,
                        alpha=0.5,
                       facecolor="purple")
                plt.gca().add_patch(rect)
            elif int(pred_an[r][t])-int(pred_op[r][t])==0:
                rect=mpatches.Rectangle((limit_no_inc[r*3+t][2]*100, limit_no_inc[r*3+t][3]*100-shiftY),shiftX, shiftY, 
                        #fill=False,
                        alpha=0.2,
                       facecolor="gray")
                plt.gca().add_patch(rect)
            elif int(pred_an[r][t])-int(pred_op[r][t])>3:
                rect=mpatches.Rectangle((limit_no_inc[r*3+t][2]*100, limit_no_inc[r*3+t][3]*100-shiftY),shiftX, shiftY, 
                        #fill=False,
                        alpha=0.8,
                       facecolor="green")
                plt.gca().add_patch(rect)
            elif int(pred_an[r][t])-int(pred_op[r][t])<-3:
                rect=mpatches.Rectangle((limit_no_inc[r*3+t][2]*100, limit_no_inc[r*3+t][3]*100-shiftY),shiftX, shiftY, 
                        #fill=False,
                        alpha=0.8,
                       facecolor="purple")
                plt.gca().add_patch(rect)
    
    b = dt['pred'][index_p][2]
    for r in range(0, row):
        for t in range(0, col):
            if int(b[r][t]) == 1:
                rect=mpatches.Rectangle((limit_no_inc[r*3+t][2]*100, limit_no_inc[r*3+t][3]*100-shiftY),shiftX, shiftY, 
                        fill=False,
                        alpha=1,
                        edgecolor="orange",
                        linewidth=5)
                plt.gca().add_patch(rect)
    
    red_patch = mpatches.Patch(color='red', label='Analyzed Team')
    blue_patch = mpatches.Patch(color='blue', label='Opponent Team')
    orange_patch = mpatches.Patch(color='orange', label='Ball')
    
    
    g1 = mpatches.Patch(color='green', alpha=.2, label='+1')
    g2 = mpatches.Patch(color='green', alpha=.35, label='+2')
    g3 = mpatches.Patch(color='green', alpha=.5, label='+3')
    g4 = mpatches.Patch(color='green', alpha=.8, label='>4')
    p1 = mpatches.Patch(color='purple', alpha=.2, label='-1')
    p2 = mpatches.Patch(color='purple', alpha=.35, label='-2')
    p3 = mpatches.Patch(color='purple', alpha=.5, label='-3')
    p4 = mpatches.Patch(color='purple', alpha=.8, label='<4')
    g = mpatches.Patch(color='gray', alpha=.2, label='Even')
    w = mpatches.Patch(color='black', alpha=.2, label='No one', fill=False)
    b = mpatches.Patch(color='orange', alpha=1, label='Ball', fill=False)
    legend1 = plt.legend(handles=[g1, g2, g3, g4, g, w, p1, p2, p3, p4, b], bbox_to_anchor=(0.92, -0.01), 
                         ncol=11, edgecolor="black",
                         title="Pitch coloration based on the prediction of the number of players of the analyzed team with respect to the opponent team")
    plt.gca().add_artist(legend1)
    plt.legend(handles=[red_patch, blue_patch, orange_patch], bbox_to_anchor=(1, 0.9))
    
    plt.title("Predicted pitch occupancy based on the previous movement of the teams.")
    plt.show()


# In[37]:


def plot_movements_real(se, tr_data, match_id, mult_all, team_id, index_p):
    
    """
    Get as input the data of training for a given prediction and the match ID.
    Plot the movements of the players through the various frames.
    """
    
    #Open the file with the player informations
    info = pd.read_json("Matches/Data/"+str(match_id)+"Data_match.json", lines=True)
    
    team=[]

    #Obtain the id of the two teams and get the players of each of them
    for i in range(0, len(info['players'][0])):
        team.append(info['players'][0][i]['team_id'])
    team=np.unique(team)
    team=np.setdiff1d(team, team_id)

    team_0_id=team_id
    team_1_id=team[0]

    team_0 = []
    team_1 = []
    for i in range(0, len(info['players'][0])):
        #if info['players'][0][i]['player_role']['acronym'] != 'GK':
        if info['players'][0][i]['team_id'] == team_0_id:
            team_0.append(info['players'][0][i]['trackable_object'])
        elif info['players'][0][i]['team_id'] == team_1_id:
            team_1.append(info['players'][0][i]['trackable_object'])
                
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
    pitch = Pitch(pitch_type='tracab', figsize=(50,10), pitch_length=105, pitch_width=68) #, axis=True, label=True
    pitch.draw(ax=ax)

    #Plot vertical lines of the grid
    for i in ver:
        plt.vlines(i*100, 3400, -3400, linestyles='dashed', linewidth=0.5)
    #Plot horizontal lines of the grid
    for i in hor:
        plt.hlines(i*100, -5250, 5250, linestyles='dashed', linewidth=0.5)
    
    #color1 = ['red', 'orange', 'black']
    #color2 = ['blue', 'green', 'yellow']
    
    col=0
    
    for j in range(0, len(tr_data)):
        
        #Get the data of one index
        data=tr_data[j]['data']
        mult=mult_all[j]
        
        if j==0 or j==2 or j==4:
            
            #Iterate though the tracking data of each players
            for i in range(0, len(data)):
                
                #Plot players of analyzed team
                if data[i]['trackable_object'] in team_0:
                    ax.scatter(data[i]["x"]*100*mult, data[i]["y"]*100*mult, color='red', marker='.', s=50)
                
                #Plot players of opponent team
                elif data[i]['trackable_object'] in team_1:
                    ax.scatter(data[i]["x"]*100*mult, data[i]["y"]*100*mult, color='blue', marker='1', s=50)
                    
                #Plot the ball position
                elif data[i]['trackable_object'] == 55:
                    ax.scatter(data[i]["x"]*100*mult, data[i]["y"]*100*mult, color='orange', s=50)
            
            col=col+1
        
        #Plot the arrow
        if j==2 or j==4:
            
            #Get the data of the start of the arrow
            data_st=tr_data[j-2]['data']
            mult=mult_all[j]
            
            for i in range(0, len(data)):
                
                #Plot arrow for analyzed team
                if data[i]['trackable_object'] in team_0 and data_st[i]['trackable_object'] in team_0:
                    plt.arrow(data_st[i]["x"]*100*mult, data_st[i]["y"]*100*mult, 
                      (-data_st[i]["x"]*100*mult+data[i]["x"]*100*mult), 
                              (-data_st[i]["y"]*100*mult+data[i]["y"]*100*mult), 
                      width=0.05, head_width=100, length_includes_head=True, color='red')
            
                #Plot arrow for opponent team
                elif data[i]['trackable_object'] in team_1 and data_st[i]['trackable_object'] in team_1:
                    plt.arrow(data_st[i]["x"]*100*mult, data_st[i]["y"]*100*mult, 
                      (-data_st[i]["x"]*100*mult+data[i]["x"]*100*mult),
                              (-data_st[i]["y"]*100*mult+data[i]["y"]*100*mult), 
                      width=0.05, head_width=100, length_includes_head=True, color='blue')
                
                #Plot arrow for the ball
                elif data[i]['trackable_object'] == 55 and data_st[i]['trackable_object'] == 55:
                    plt.arrow(data_st[i]["x"]*100*mult, data_st[i]["y"]*100*mult, 
                      (-data_st[i]["x"]*100*mult+data[i]["x"]*100*mult),
                              (-data_st[i]["y"]*100*mult+data[i]["y"]*100*mult), 
                      width=0.05, head_width=100, length_includes_head=True, color='orange')
    
    
    """
    pred_an = se['Y'][index_p][0]
    for r in range(0, row):
        for t in range(0, col):
            if int(pred_an[r][t])==1:
                ax.scatter(X_start+shiftX*t+1500, Y_start-shiftY*r-300, color='green')
            elif int(pred_an[r][t])>1:
                for a in range(0, int(pred_an[r][t])):
                    ax.scatter(X_start+shiftX*t+1500-(a*150), Y_start-shiftY*r-300, color='green')
    
    pred_op = se['Y'][index_p][1]
    for r in range(0, row):
        for t in range(0, col):
            if int(pred_op[r][t])==1:
                ax.scatter(X_start+shiftX*t+1500, Y_start-shiftY*r-500, color='purple')
            elif int(pred_op[r][t])>1:
                for a in range(0, int(pred_op[r][t])):
                    ax.scatter(X_start+shiftX*t+1500-(a*150), Y_start-shiftY*r-500, color='purple')
    """
    
    limit_no_inc = construct_limit_no_inc(row, col)
    pred_op = se['Y'][index_p][1]
    pred_an = se['Y'][index_p][0]
    for r in range(0, row):
        for t in range(0, col):
            if int(pred_an[r][t])==0 and int(pred_op[r][t])==0:
                rect=mpatches.Rectangle((limit_no_inc[r*3+t][2]*100, limit_no_inc[r*3+t][3]*100-shiftY),shiftX, shiftY, 
                        #fill=False,
                        alpha=0.2,
                       facecolor="white")
                plt.gca().add_patch(rect)
            elif int(pred_an[r][t])-int(pred_op[r][t])==1:
                rect=mpatches.Rectangle((limit_no_inc[r*3+t][2]*100, limit_no_inc[r*3+t][3]*100-shiftY),shiftX, shiftY, 
                        #fill=False,
                        alpha=0.2,
                       facecolor="green")
                plt.gca().add_patch(rect)
            elif int(pred_an[r][t])-int(pred_op[r][t])==2:
                rect=mpatches.Rectangle((limit_no_inc[r*3+t][2]*100, limit_no_inc[r*3+t][3]*100-shiftY),shiftX, shiftY, 
                        #fill=False,
                        alpha=0.35,
                       facecolor="green")
                plt.gca().add_patch(rect)
            elif int(pred_an[r][t])-int(pred_op[r][t])==3:
                rect=mpatches.Rectangle((limit_no_inc[r*3+t][2]*100, limit_no_inc[r*3+t][3]*100-shiftY),shiftX, shiftY, 
                        #fill=False,
                        alpha=0.5,
                       facecolor="green")
                plt.gca().add_patch(rect)
            elif int(pred_an[r][t])-int(pred_op[r][t])==-1:
                rect=mpatches.Rectangle((limit_no_inc[r*3+t][2]*100, limit_no_inc[r*3+t][3]*100-shiftY),shiftX, shiftY, 
                        #fill=False,
                        alpha=0.2,
                       facecolor="purple")
                plt.gca().add_patch(rect)
            elif int(pred_an[r][t])-int(pred_op[r][t])==-2:
                rect=mpatches.Rectangle((limit_no_inc[r*3+t][2]*100, limit_no_inc[r*3+t][3]*100-shiftY),shiftX, shiftY, 
                        #fill=False,
                        alpha=0.35,
                       facecolor="purple")
                plt.gca().add_patch(rect)
            elif int(pred_an[r][t])-int(pred_op[r][t])==-3:
                rect=mpatches.Rectangle((limit_no_inc[r*3+t][2]*100, limit_no_inc[r*3+t][3]*100-shiftY),shiftX, shiftY, 
                        #fill=False,
                        alpha=0.5,
                       facecolor="purple")
                plt.gca().add_patch(rect)
            elif int(pred_an[r][t])-int(pred_op[r][t])==0:
                rect=mpatches.Rectangle((limit_no_inc[r*3+t][2]*100, limit_no_inc[r*3+t][3]*100-shiftY),shiftX, shiftY, 
                        #fill=False,
                        alpha=0.2,
                       facecolor="gray")
                plt.gca().add_patch(rect)
            elif int(pred_an[r][t])-int(pred_op[r][t])>3:
                rect=mpatches.Rectangle((limit_no_inc[r*3+t][2]*100, limit_no_inc[r*3+t][3]*100-shiftY),shiftX, shiftY, 
                        #fill=False,
                        alpha=0.8,
                       facecolor="green")
                plt.gca().add_patch(rect)
            elif int(pred_an[r][t])-int(pred_op[r][t])<-3:
                rect=mpatches.Rectangle((limit_no_inc[r*3+t][2]*100, limit_no_inc[r*3+t][3]*100-shiftY),shiftX, shiftY, 
                        #fill=False,
                        alpha=0.8,
                       facecolor="purple")
                plt.gca().add_patch(rect)
    
    b = se['Y'][index_p][2]
    for r in range(0, row):
        for t in range(0, col):
            if int(b[r][t]) == 1:
                rect=mpatches.Rectangle((limit_no_inc[r*3+t][2]*100, limit_no_inc[r*3+t][3]*100-shiftY),shiftX, shiftY, 
                        fill=False,
                        alpha=1,
                        edgecolor="orange",
                        linewidth=5)
                plt.gca().add_patch(rect)
    
    red_patch = mpatches.Patch(color='red', label='Analyzed Team')
    blue_patch = mpatches.Patch(color='blue', label='Opponent Team')
    orange_patch = mpatches.Patch(color='orange', label='Ball')
    
    
    g1 = mpatches.Patch(color='green', alpha=.2, label='+1')
    g2 = mpatches.Patch(color='green', alpha=.35, label='+2')
    g3 = mpatches.Patch(color='green', alpha=.5, label='+3')
    g4 = mpatches.Patch(color='green', alpha=.8, label='>4')
    p1 = mpatches.Patch(color='purple', alpha=.2, label='-1')
    p2 = mpatches.Patch(color='purple', alpha=.35, label='-2')
    p3 = mpatches.Patch(color='purple', alpha=.5, label='-3')
    p4 = mpatches.Patch(color='purple', alpha=.8, label='<4')
    g = mpatches.Patch(color='gray', alpha=.2, label='Even')
    w = mpatches.Patch(color='black', alpha=.2, label='No one', fill=False)
    b = mpatches.Patch(color='orange', alpha=1, label='Ball', fill=False)
    legend1 = plt.legend(handles=[g1, g2, g3, g4, g, w, p1, p2, p3, p4, b], bbox_to_anchor=(0.92, -0.01), 
                         ncol=11, edgecolor="black",
                         title="Pitch coloration based on the real number of players of the analyzed team with respect to the opponent team")
    plt.gca().add_artist(legend1)
    plt.legend(handles=[red_patch, blue_patch, orange_patch], bbox_to_anchor=(1, 0.9))
    
    plt.title("Real pitch occupancy based on the previous movement of the teams.")
    plt.show()


# In[38]:


def invert_pitch(period, match_id, analyzed):
    
    """
    Get as input the array with the period of each frame, the match ID and it the analyzed team
     plays at home or away.
    For each frame get the variable to multiplicate to the 'X' coordinate. It is -1 if the frame must be inverted,
     1 otherwise. This to normalize the position of analyzed team always attacking left to right.
    Return this array of -1 or 1.
    """

    
    #Open the file with the information about the teams 
    dt = pd.read_json(get_match_info(match_id), lines=True)


    mult=[]

    for i in range(0, len(period)):
        
        #For frame before the start of the game and between the two periods
        if period[i]==None:
            mult.append('Null')
            continue
    
        #Case if the analyzed team played at home
        if analyzed=='home team':
            
            if period[i]==1:
                if dt['home_team_side'][0][0]=='left_to_right':
                    mult.append(1)
                else:
                    mult.append(-1)

            elif period[i]==2:
                if dt['home_team_side'][0][1]=='left_to_right':
                    mult.append(1)
                else:
                    mult.append(-1)
        
        #Case if the analyzed team played away
        else:

            if period[i]==1:
                if dt['home_team_side'][0][0]=='left_to_right':
                    mult.append(-1)
                else:
                    mult.append(1)

            elif period[i]==2:
                if dt['home_team_side'][0][1]=='left_to_right':
                    mult.append(-1)
                else:
                    mult.append(1)
    
    return mult


# In[39]:


def get_team(match_id, team_id):

    """
    Get as input the match_id
    Open the file with the match data and construct a dictionary with as keys the team_id and the trackable_object
     of each team inside.
    Return the dictionary and the team_id
    """

    
    #Open the file with the information about the teams 
    dt = pd.read_json(get_match_info(match_id), lines=True)

    #Extract Wyscout IDs used in the construction
    wys_id = dt['wyscout_id'][0]
    
    #Extract the two teams
    for j in range(0, len(dt['players'][0])):
        if dt['players'][0][j]['team_id'] != team_id:
            other_id = dt['players'][0][j]['team_id']
            break
    
    
    #Create a dictionary and store the trackable object of each team
    team_0 = []
    team_1 = []
    for i in range(0, len(dt['players'][0])):
        if dt['players'][0][i]['player_role']['acronym'] != 'GK':
            if dt['players'][0][i]['team_id'] == team_id:
                team_0.append(dt['players'][0][i]['trackable_object'])
            elif dt['players'][0][i]['team_id'] == other_id:
                team_1.append(dt['players'][0][i]['trackable_object'])

    
    return team_0, team_1, other_id, wys_id


# In[47]:

def get_id_short(team):

    """
    Get as input the name of the team to analyze.
    Return the SkillCorner ID and the short name of the team to analyze. 
    """

    #Array with the teamID, the short name of the team used in the name of the file and the name of the team
    id_team = [[158, 'SAS', 'Sassuolo'],
             [148, 'UDI', 'Udinese'],
             [147, 'LEC', 'Lecce'],
             [152, 'PAR', 'Parma'],
             [135, 'GEN', 'Genoa'],
             [157, 'VER', 'Verona'],
             [141, 'BOL', 'Bologna'],
             [146, 'TOR', 'Torino'],
             [315, 'SPA', 'Spal'],
             [138, 'FIO', 'Fiorentina'],
             [130, 'ATA', 'Atalanta'],
             [145, 'INT', 'Inter'],
             [143, 'NAP', 'Napoli'],
             [133, 'LAZ', 'Lazio'],
             [140, 'ACM', 'Milan'],
             [132, 'CAG', 'Cagliari'],
             [139, 'JUV', 'Juventus'],
             [142, 'ROM', 'Roma'],
             [154, 'BRE', 'Brescia'],
             [144, 'SAM', 'Sampdoria']]

    #Get the teamID and the short name of the team
    for i in range(0, len(id_team)):
        if id_team[i][2]==team:
            team_id = id_team[i][0]
            team_short = id_team[i][1]
    
    return team_id, team_short


def main_real(se, list_, team):
    
    """
    Get as input the one list of index which represents the inexes of equal prediction.
    Extract the files and call the functions to plot the movements.
    """
    
    #Open file with the dataset
    #tr=h5py.File("Files/"+team+"/Testing/"+team+"EEE.h5", 'r')

    team_id, team_short = get_id_short(team)


    index_list = 0
    
    #Iterate through the indexes get as input
    for i in list_: 

        #Get the timestamp of the index specified to get the frame to access.
        this = se['Timestamp_c'][i]
        #y = tr['Timestamp_test_0'][i]

        #Get match_ID
        match_id=this[0][0].decode('utf8')

        directory='Matches'

        #Iterate though the files of the tracking data
        for filename in os.listdir(directory):

            #Get the name of the file
            f = os.path.join(directory, filename)

            #Get match_ID of the file obtained through the loop
            match_id_all = f[8:12]

            home=f[12:15]
            away=f[15:18]

            #Check if the match_ID of the prediction match with the match_ID of the file obtained
            # through the prediction
            if match_id_all == match_id:

                #Open file of tracking data
                track = pd.read_json(f, lines=True)
                tr_data = []
                y_data = []
                mult_all = []


                #Extract the players of the two teams involved, 
                # team_0 includes the players of the analyzed team
                team_0, team_1, other_id, wys_id = get_team(match_id, team_id)

                #check if the analyzed team is the home team or the away team, used for the ball possession
                if home==team_short:
                    analyzed = 'home team'
                    other = 'away team'
                elif away==team_short:
                    analyzed = 'away team'
                    other = 'home team'


                #For each frame in the data, get the period of it
                period = []
                for ind in range(0, np.shape(track)[1]):
                    period.append(track[ind][0]['period'])

                mult = invert_pitch(period, match_id, analyzed)

                #Iterate through the frame of training set used for the specified index
                for ind in this:

                    #Append the frames
                    tr_data.append(track[int(ind[1].decode('utf8'))][0])  #['data']
                    mult_all.append(mult[track[int(ind[1].decode('utf8'))][0]['frame']])

                tr_data.reverse()

                #print(tr['Timestamp_y'][i])


                #Plot the movements for the given frames. 
                plot_movements_real(se, tr_data, match_id, mult_all, team_id, list_[index_list])
                
                index_list = index_list +1


# In[41]:


def main_pred(dt, list_, team):
    
    """
    Get as input the one list of index which represents the inexes of equal prediction.
    Extract the files and call the functions to plot the movements.
    """
    
    #Open file with the dataset
    tr=h5py.File("Files/"+team+"/Testing/"+team+"EEE.h5", 'r')

    team_id, team_short = get_id_short(team)


    #Iterate through the indexes get as input
    for i in list_: 

        #Get the timestamp of the index specified to get the frame to access.
        this = tr['Timestamp_c'][i]
        #y = tr['Timestamp_test_0'][i]

        #Get match_ID
        match_id=this[0][0].decode('utf8')

        directory='Matches'

        #Iterate though the files of the tracking data
        for filename in os.listdir(directory):

            #Get the name of the file
            f = os.path.join(directory, filename)

            #Get match_ID of the file obtained through the loop
            match_id_all = f[8:12]

            home=f[12:15]
            away=f[15:18]

            #Check if the match_ID of the prediction match with the match_ID of the file obtained
            # through the prediction
            if match_id_all == match_id:

                #Open file of tracking data
                track = pd.read_json(f, lines=True)
                tr_data = []
                y_data = []
                mult_all = []


                #Extract the players of the two teams involved, 
                # team_0 includes the players of the analyzed team
                team_0, team_1, other_id, wys_id = get_team(match_id, team_id)

                #check if the analyzed team is the home team or the away team, used for the ball possession
                if home==team_short:
                    analyzed = 'home team'
                    other = 'away team'
                elif away==team_short:
                    analyzed = 'away team'
                    other = 'home team'


                #For each frame in the data, get the period of it
                period = []
                for ind in range(0, np.shape(track)[1]):
                    period.append(track[ind][0]['period'])

                mult = invert_pitch(period, match_id, analyzed)

                #Iterate through the frame of training set used for the specified index
                for ind in this:

                    #Append the frames
                    tr_data.append(track[int(ind[1].decode('utf8'))][0])  #['data']
                    mult_all.append(mult[track[int(ind[1].decode('utf8'))][0]['frame']])

                tr_data.reverse()

                #print(tr['Timestamp_y'][i])


                #Plot the movements for the given frames. 
                plot_movements_pred(dt, tr_data, match_id, mult_all, team_id, list_[0])


# In[42]:


def extract_data(team):
    
    
    if not(os.path.isfile("Files/"+team+"/Testing/"+team+"_Equal.h5")):
        
        dt = h5py.File("Files/"+team+"/Testing/"+team+"EEE_pred.h5", 'r')
        
        equal = main_encode(dt, team)

        h5 = h5py.File("Files/"+team+"/Testing/"+team+"_Equal.h5", 'w')
        h5.create_dataset('equal', data=equal)
        h5.close()

    else:

        h5 = h5py.File("Files/"+team+"/Testing/"+team+"_Equal.h5", 'r')
        equal = list(h5['equal'])
        h5.close()
    
        dt = h5py.File("Files/"+team+"/Testing/"+team+"EEE_pred.h5", 'r')
    
    se = h5py.File("Files/"+team+"/Testing/"+team+"EEE.h5", 'r')
        
    return dt, se, equal


