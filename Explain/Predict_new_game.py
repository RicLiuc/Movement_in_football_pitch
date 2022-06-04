#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import h5py
import json
import os
import params
from tqdm import tqdm
from sklearn.utils import shuffle
from path import *
import random
import torch
#from External_element2.External import create_external
from ST_ResNet import STResNet
from sklearn.metrics import jaccard_score
import copy as cp


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


def check_sequence(dt, i):

    """
    Get as input che file with the data and the index to analyze.
    Check if the index passed can be eligible for the construction of the dataset. Thus check if the sequences for period and closeness
     have data inside and if the period of the shifted frames are in the same period.
    Return True or False. 
    """
    
    check = True
    
    to_pred = i-50
    
    #Check if the frame used to calc the start and the end for the adjacent matrix
    # have data inside. 
    if to_pred>=0:
        if len(dt[to_pred][0]['data'])==0 or len(dt[i][0]['data'])==0: #Shifted to calculate the adjacent
            check=False
    
    #Shifting to start the closeness sequence
    clos_ind = to_pred
    
    for j in range(0, params.len_clos):
        
        this_index = clos_ind - j * params.shift_clos
        
        #check if going back we get an index which is less than zero
        if this_index <0:
            check = False
            continue
        
        #period of the closeness frames must be the same of the frame to predict.
        #frames of closeness must have data inside.
        #period of the shifted frame used to construct the matrix must be the same.
        #shifted frame must have data inside
        
        if dt[to_pred+50][0]['period'] != dt[this_index][0]['period']        or len(dt[this_index][0]['data'])==0:
            check=False
    
    per_ind = to_pred+50
    
    for j in range(1, params.len_per+1):
        
        this_index = per_ind - j * params.shift_per
        
        #check if going back we get an index which is less than zero
        if this_index <0 or per_ind<0:
            check = False
            continue
        
        #period of the closeness frames must be the same of the frame to predict.
        #frames of closeness must have data inside.
        #period of the shifted frame used to construct the matrix must be the same.
        #shifted frame must have data inside
        
        if dt[to_pred+50][0]['period'] != dt[this_index][0]['period']        or len(dt[this_index][0]['data'])==0:
            check=False
    
    
    
    return check


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


def get_team(match_id, team_id):

    """
    Get as input the match_id
    Open the file with the match data and construct a dictionary with as keys the team_id and the trackable_object
     of each team inside.
    Return the dictionary and the team_id
    """
    
    #name = "to_pred/Data/"+str(match_id)+"Data_match.json"
    name = "Matches/Data/"+str(match_id)+"Data_match.json"
    
    #Open the file with the information about the teams 
    dt = pd.read_json(name, lines=True)

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


def create_timestamp(data):

    """
    Get as input the timestamp of a specific frame.
    Trasform the timestamp into a string composed by 'minute'+'second'+'millisecond'. 
    If we are in the second period it trasform the minute subtracting 45.
    Return the timestamp formatted as stated.
    """
    
    
    hour = int(data['timestamp'][0:2])
    minute = int(data['timestamp'][3:5])
    period = str(data['period'])
    
    #Sum the minute if we pass the hour of play in order to have just minute
    if(hour>0):
        minute = hour*60 + minute
    
    #Substract 45 if we are in the second period
    if data['period']==2:
        minute = minute - 45
        
    second = str(data['timestamp'][6:8]).rjust(2, '0')
    milli = str(data['timestamp'][9:]).rjust(2, '0')
    minute = str(minute).rjust(2, '0')
    
    #Merge everything
    timestamp=period+minute+second+milli
    
    return timestamp


def construct_matrix(dt, i, team_0, team_1, match_id, analyzed, other, mult):

    """
    Get as input the file with the data, the index to predict, the match ID and the dictionary with the teams composition.
    Construct the different sequences of matrices for the specified index which is already checked.
     It constructs the closeness and period sequences for the index passed as input.
    Return the sequences and the timestamps of the frames used. 
    """
    
    to_pred = i-50
    
    col = params.col
    row = params.row
    
    #Vectors for timestamp
    time_c = []
    time_p = []
    
    #Construct limits for the grid division
    limit=construct_limit(params.row, params.col)
    
    #Vectors for the sequences
    x_c = []
    x_p = []
    ball_possession = []
    
    #CLOSENESS SEQUENCE

    #Set up the index for the closeness sequence
    clos_ind = to_pred
    
    for j in range(0, params.len_clos):
        
        #Get the index to start to go back
        this_index = clos_ind - j * params.shift_clos
        
        #Get the data of the specified index
        data = dt[this_index][0]['data']
        
        #Create the timestamp of this frame of closeness
        #The timestamp is constructed as an array with each row composed by three rows: ['match_id', 'frame', 'period+minute+second+millisecond']
        # this construction is done for each frame of closeness, and the same is for the period. 
        time = create_timestamp(dt[this_index][0])
        time_c.append([match_id, str(this_index), str(time)])
        
        #Prepare the array for the results
        position_team0 = np.zeros(col*row)
        position_team1 = np.zeros(col*row)
        ball = np.zeros(col*row)
        
        #Initialize the array of the results
        for i in range(0, col*row):
            position_team0[i]=0
            position_team1[i]=0
            ball[i]=0


        for i in range(0, len(data)):
            ty =list(data[i].keys())[1] #get 'trackable object' or 'group name'
            if ty != 'group_name': #check if the player has been identified
                for j in range(0, len(limit)):
                    #Assign the cell to each player
                    if data[i]["x"]*mult[this_index]<=limit[j][1]                     and data[i]["x"]*mult[this_index]>limit[j][2]                     and data[i]["y"]*mult[this_index]<=limit[j][3]                     and data[i]["y"]*mult[this_index]>limit[j][4]:
                        
                        #check the team of each object and the ball position
                        if data[i]["trackable_object"] in team_0:
                            position_team0[int(limit[j][0])] = position_team0[int(limit[j][0])] +1
                        elif data[i]["trackable_object"] in team_1:
                            position_team1[int(limit[j][0])] = position_team1[int(limit[j][0])]+1
                        elif data[i]["trackable_object"] == 55: #get ball position
                            ball[int(limit[j][0])] = ball[int(limit[j][0])] +1


        #Get the team with the ball possession for each frame using 0 if it is not known,
        # 1 if the analyzed team has the ball and 2 if the opponent team has the ball. 
        if dt[this_index][0]['possession']['group'] == analyzed:
            ball_possession.append(1)
        elif dt[this_index][0]['possession']['group'] == other:
            ball_possession.append(2)
        else:
            ball_possession.append(0)

        #Dictionary to convert from list to grid.
        conv = {}
        for i in range(0, row):
            r = i*col
            for j in range(0, col):
                conv[r+j] = [i, j]

        position_team0_f = np.zeros((row, col))
        position_team1_f = np.zeros((row, col))
        ball_f = np.zeros((row, col))

        #Conversion from list to the grid.
        for i in range(0, len(position_team0)):
            position_team0_f[conv[i][0]][conv[i][1]] = position_team0[i]
            position_team1_f[conv[i][0]][conv[i][1]] = position_team1[i]
            ball_f[conv[i][0]][conv[i][1]] = ball[i]
        
        #Append the result
        x_c.append([position_team0_f, position_team1_f, ball_f])
        
    
    #PERIOD SEQUENCE
   
    #Index for period
    per_ind = to_pred+50
    
    for j in range(1, params.len_per+1):
        
        #Get the index to start to go back
        this_index = per_ind - j * params.shift_per
        
        #Get the data of the specified index
        data = dt[this_index][0]['data']
        
        #Create the timestamp of this frame of peiod
        time = create_timestamp(dt[this_index][0])
        time_p.append([match_id, str(this_index), str(time)])
        
        #Prepare the array for the results
        position_team0 = np.zeros(col*row)
        position_team1 = np.zeros(col*row)
        ball = np.zeros(col*row)
        
        #Initialize the array of the results
        for i in range(0, col*row):
            position_team0[i]=0
            position_team1[i]=0
            ball[i]=0


        for i in range(0, len(data)):
            ty =list(data[i].keys())[1] #get 'trackable object' or 'group name'
            if ty != 'group_name': #check if the player has been identified
                for j in range(0, len(limit)):
                    #Assign the cell to each player
                    if data[i]["x"]*mult[this_index]<=limit[j][1]                     and data[i]["x"]*mult[this_index]>limit[j][2]                     and data[i]["y"]*mult[this_index]<=limit[j][3]                     and data[i]["y"]*mult[this_index]>limit[j][4]:
                        
                        #check the team of each object and the ball position
                        if data[i]["trackable_object"] in team_0:
                            position_team0[int(limit[j][0])] = position_team0[int(limit[j][0])] +1
                        elif data[i]["trackable_object"] in team_1:
                            position_team1[int(limit[j][0])] = position_team1[int(limit[j][0])]+1
                        elif data[i]["trackable_object"] == 55: #get ball position
                            ball[int(limit[j][0])] = ball[int(limit[j][0])] +1

        #Dictionary to convert from list to grid.
        conv = {}
        for i in range(0, row):
            r = i*col
            for j in range(0, col):
                conv[r+j] = [i, j]

        position_team0_f = np.zeros((row, col))
        position_team1_f = np.zeros((row, col))
        ball_f = np.zeros((row, col))

        #Conversion from list to the grid.
        for i in range(0, len(position_team0)):
            position_team0_f[conv[i][0]][conv[i][1]] = position_team0[i]
            position_team1_f[conv[i][0]][conv[i][1]] = position_team1[i]
            ball_f[conv[i][0]][conv[i][1]] = ball[i]
            
        
        #Append the result
        x_p.append([position_team0_f, position_team1_f, ball_f])
        
    
    return x_c, x_p, time_c, time_p, ball_possession  


def construct_target(dt, i, team_0, team_1, mult):

    """
    Get as input the file with the data, the index to predict and the dictionary with the teams composition.
    Construct the matrices of the target for the specified index.
    Return the target to predict. 
    """
    
    #Data where extract information of the target
    data = dt[i][0]['data']
    
    col = params.col
    row = params.row
    
    #Prepare the array for the results
    Y = []
    
    #Construct the limits to grid the pitch
    limit=construct_limit(params.row, params.col)
    
    #Prepare the array fot he different flows. 
    flow_0 = np.zeros((params.row*params.col, params.row*params.col))
    flow_1 = np.zeros((params.row*params.col, params.row*params.col))
    flow_b = np.zeros((params.row*params.col, params.row*params.col))
    
    
    position_team0 = np.zeros(col*row)
    position_team1 = np.zeros(col*row)
    ball = np.zeros(col*row)

    for i in range(0, col*row):
        position_team0[i]=0
        position_team1[i]=0
        ball[i]=0


    for i in range(0, len(data)):
        ty =list(data[i].keys())[1] #get 'trackable object' or 'group name'
        if ty != 'group_name': #check if the player has been identified
            for j in range(0, len(limit)):
                #Assign the cell to each player
                if data[i]["x"]*mult<=limit[j][1]                 and data[i]["x"]*mult>limit[j][2]                 and data[i]["y"]*mult<=limit[j][3]                 and data[i]["y"]*mult>limit[j][4]:

                    #check the team of each object and the ball position
                    if data[i]["trackable_object"] in team_0:
                        position_team0[int(limit[j][0])] = position_team0[int(limit[j][0])] +1
                    elif data[i]["trackable_object"] in team_1:
                        position_team1[int(limit[j][0])] = position_team1[int(limit[j][0])]+1
                    elif data[i]["trackable_object"] == 55: #get ball position
                        ball[int(limit[j][0])] = ball[int(limit[j][0])] +1

    #Dictionary to convert from list to grid.
    conv = {}
    for i in range(0, row):
        r = i*col
        for j in range(0, col):
            conv[r+j] = [i, j]

    position_team0_f = np.zeros((row, col))
    position_team1_f = np.zeros((row, col))
    ball_f = np.zeros((row, col))

    #Conversion from list to the grid.
    for i in range(0, len(position_team0)):
        position_team0_f[conv[i][0]][conv[i][1]] = position_team0[i]
        position_team1_f[conv[i][0]][conv[i][1]] = position_team1[i]
        ball_f[conv[i][0]][conv[i][1]] = ball[i]

    Y.append([position_team0_f, position_team1_f, ball_f])
    
    return Y


def extract_valid(dt):

    """
    Get as input the file with the data.
    Extract all the index that are valid for the construction of the dataset.
    Return the valid index that we have to use for this match. 
    """
    
    valid = []
    
    #Extract all the valid samples
    for i in tqdm(range(0, np.shape(dt)[1])):
        if(check_sequence(dt, i)):
            valid.append(i)
    
    
    #Extract randomly 'sample' frame from the list of valid index
    extracted_frame = []
    
    
    #Extract randomly the right number of samples of this match. 
    for i in range(0, 10000):
        if len(valid) != 0:
            cho = random.choice(valid)

            if cho not in extracted_frame:
                extracted_frame.append(cho)

            valid.remove(cho)
        else:
            print(f"Not enough data, extracted {len(extracted_frame)} samples")
            break
    
    
    return extracted_frame
    #return valid


def split_test(XC, XP, Y, time_Y, time_C, time_P, ball, len_test, len_clos, len_per):

    
    
    """
    Get as input the data of closeness, period, the target, the length of test set
     and the length of closeness and period sequencies. Futhermore we pass also the timestamp of each component
     for futher analysis and the ball possession for the external component.
    Split train and test and merge all the iternal component in one array. 
    Return the data train, the data test and the timestamps
    """

    print("\n")
    print("Splitting train and test")

    #Shuffle the data before split them
    XC, XP, Y, time_Y, time_C, time_P, ball = shuffle(XC, XP, Y, time_Y, time_C, time_P, ball)
    
    #Split train and test for each component and for the timestamps
    XC_train, XP_train, Y_train = XC[:-len_test], XP[:-len_test], Y[:-len_test]
    XC_test, XP_test, Y_test = XC[-len_test:], XP[-len_test:], Y[-len_test:]
    timeC_train, timeP_train, timeY_train = time_C[:-len_test], time_P[:-len_test], time_Y[:-len_test]
    timeC_test, timeP_test, timeY_test = time_C[-len_test:], time_P[-len_test:], time_Y[-len_test:]
    ball_train = ball[:-len_test]
    ball_test = ball[-len_test:]
    
    X_train = []
    X_test = []
    time_train = []
    time_test = []
    
    #Merge the internal component in one array
    for l, X_ in zip([len_clos, len_per], [XC_train, XP_train]):
        X_train.append(X_)
    
    for l, X_ in zip([len_clos, len_per], [XC_test, XP_test]):
        X_test.append(X_)

    #Merge the timestamps of internal component in one array
    for l, X_ in zip([len_clos, len_per], [timeC_train, timeP_train]):
        time_train.append(X_)

    for l, X_ in zip([len_clos, len_per], [timeC_test, timeP_test]):
        time_test.append(X_)
    
    return X_train, Y_train, X_test, Y_test, time_train, timeY_train, time_test, timeY_test, ball_train, ball_test


def write_dataset(X_c, X_p, Y, ball, Timestamp_c, Timestamp_p, Timestamp_y, team):
    
    print("Writing dataset in the file")
    
    

    #Open file
    file = h5py.File("Files/"+str(team)+"/"+str(team)+"Testing.h5", 'w')
    
    #Write closeness
    file.create_dataset('X_c', data=X_c)
    
    #Write period
    file.create_dataset('X_p', data=X_p)
        
    #Write Y
    file.create_dataset('Y', data=Y)
    
    #Write the timestamps of closeness
    file.create_dataset('Timestamp_c', data=np.string_(Timestamp_c))
    
    #Write the timestamps of period
    file.create_dataset('Timestamp_p', data=np.string_(Timestamp_p))
    
    #Write the timestamps of target test
    file.create_dataset('Timestamp_y', data=np.string_(Timestamp_y))
    
    #Write the ball possession for the train
    file.create_dataset('Ball', data=ball)
    
    file.close()

    
def write_file(X_c, X_p, Y, ball, Timestamp_c, Timestamp_p, Timestamp_y, team):
    
    """
    Get as input the 4D tensor with the occupancy data, the list with the timestamps and the control array.
    Popolate the datasets with the data regarding all the match, excluding the first one.
    """
    

    h5 = h5py.File("Files/"+str(team)+"/"+str(team)+"Testing.h5", 'r+')
    
    #Popolate the timestamp dataset
    h5["X_c"].resize((h5["X_c"].shape[0] + len(X_c)), axis = 0)
    h5["X_c"][-len(X_c):] = X_c
    
    #Popolate the matrixs dataset with the occupancy data
    h5["X_p"].resize((h5["X_p"].shape[0] + len(X_p)), axis = 0)
    h5["X_p"][-len(X_p):] = X_p
    
    #Popolate the matrixs dataset with the ball possession data
    h5["Y"].resize((h5["Y"].shape[0] + len(Y)), axis = 0)
    h5["Y"][-len(Y):] = Y
    
    #Popolate the matrixs dataset with the occupancy data
    h5["Timestamp_c"].resize((h5["Timestamp_c"].shape[0] + len(Timestamp_c)), axis = 0)
    h5["Timestamp_c"][-len(Timestamp_c):] = Timestamp_c
    
    #Popolate the matrixs dataset with the occupancy data
    h5["Timestamp_p"].resize((h5["X_p"].shape[0] + len(Timestamp_p)), axis = 0)
    h5["Timestamp_p"][-len(Timestamp_p):] = Timestamp_p
    
    #Popolate the matrixs dataset with the occupancy data
    h5["Timestamp_y"].resize((h5["Timestamp_y"].shape[0] + len(Timestamp_y)), axis = 0)
    h5["Timestamp_y"][-len(Timestamp_y):] = Timestamp_y
    
    #Popolate the matrixs dataset with the occupancy data
    h5["Ball"].resize((h5["Ball"].shape[0] + len(Ball)), axis = 0)
    h5["Ball"][-len(Ball):] = Ball
    
    h5.close()
    

def write_id(wyscout_id, skillcorner_id, team):

    """
    Get as input the list with the Wyscout IDs used in the creation of the dataset and the name of the team analyzed
    Write in a h5 file the IDs used. 
    """


    h5 = h5py.File(get_ID(team), 'w')

    #Dataset with the IDs
    h5.create_dataset('WS_ID', data=wyscout_id)
    h5.create_dataset('SC_ID', data=skillcorner_id)

    h5.close()


def create_data(team, f):

    team_id, team_short = get_id_short(team)
    print("\n")


    #All the array for the results
    X_c = []
    X_p = []
    ball = []
    Y = []
    Time_Xc = []
    Time_Xp = []
    Timestamp = []
    Time_Y = []

    wyscout_id = []
    skillcorner_id = []
    

    #Get the name of teams involved in the file 
    home=f[12:15]
    away=f[15:18]

        
    #Extract Match_ID of the file
    match_id = f[8:12]

    #Open the file
    dt = pd.read_json(f, lines=True)
    
    #For each frame in the data, get the period of it
    period = []
    for i in range(0, np.shape(dt)[1]):
        period.append(dt[i][0]['period'])

    #Extract the players of the two teams involved, 
    # team_0 includes the players of the analyzed team
    team_0, team_1, other_id, wys_id=get_team(match_id, team_id)

    #check if the analyzed team is the home team or the away team, used for the ball possession
    if home==team_short:
        analyzed = 'home team'
        other = 'away team'
    elif away==team_short:
        analyzed = 'away team'
        other = 'home team'
    
    mult = invert_pitch(period, match_id, analyzed)

    print("Extracting valid frames for", f)

    #Extract the valid frame of the match
    arr = extract_valid(dt)

    wyscout_id.append(wys_id)
    skillcorner_id.append(match_id)


    print("Constructing dataset for valid frames")

    #Iterate through the extracted valid frames
    for i in tqdm(arr):

        #Construct the dataset for the sequences
        xc, xp, time_c, time_p, ball_possession = construct_matrix(dt, i, team_0, team_1, match_id, analyzed, other, mult)

        #Construct the target
        y = construct_target(dt, i, team_0, team_1, mult[i])

        ball.append(ball_possession)

        x_c = np.vstack(xc)
        x_p = np.vstack(xp)

        y_ = np.vstack(y)

        Time_Xc.append(time_c)
        Time_Xp.append(time_p)

        #Create timestamp for the target
        t_y=create_timestamp(dt[i][0])
        Time_Y.append([match_id, str(i), str(t_y)])


        X_c.append(x_c)
        X_p.append(x_p)
        Y.append(y_)

    print("\n")

    Timestamp.append(Time_Xc)
    Timestamp.append(Time_Xp)
    Timestamp.append(Time_Y)

    """
    #Split training and test
    X_train, Y_train, X_test, Y_test, time_train, timeY_train, time_test, \
    timeY_test, ball_train, ball_test = split_test(X_c, X_p, Y, Timestamp[2], Timestamp[0], Timestamp[1], ball, \
               params.len_test, params.len_clos, params.len_per)
    """


    #Write data in a file for further use
    write_dataset(X_c, X_p, Y, ball, Timestamp[0], Timestamp[1], Timestamp[2], team)
    #write_file(X_c, X_p, Y, ball, Timestamp[0], Timestamp[1], Timestamp[2], team)

    #Write all the Wyscout id used in the construction of the dataset in a file
    #write_id(wyscout_id, skillcorner_id, team)

    print("Written")
    print("\n")

    return X_c, X_p, Y, ball, Timestamp


# In[33]:


def create_timeslot(data):
    
    """
    Get as input the data to transform.
    Create the arrary with the timestamp one-hot-encoded for the external component. 
    The array is composed by 3 timeslot for each period. 
    Array[i][0] = period 1 timeslot 0-15
    Array[i][1] = period 1 timeslot 16-30
    Array[i][2] = period 1 timeslot 31-45
    Array[i][3] = period 2 timeslot 0-15
    Array[i][4] = period 2 timeslot 16-30
    Array[i][5] = period 2 timeslot 31-45
    Return the array.
    """
    
    #Initialize the array
    timeslot = np.zeros((len(data), 6)).astype(int)
    
    for i in range(0, len(timeslot)):
        
        #Chcek the period
        if int(data[i][2][:1])==1:
            minute=int(data[i][2][1:3])
            if minute <= 15:
                timeslot[i][0] = 1
            elif minute >= 16 and minute <=30:
                timeslot[i][1] = 1
            elif minute >= 31:
                timeslot[i][2] = 1
                
        elif int(data[i][2][:1])==2:
            minute=int(data[i][2][1:3])
            if minute <= 15:
                timeslot[i][3] = 1
            elif minute >= 16 and minute <=30:
                timeslot[i][4] = 1
            elif minute >= 31:
                timeslot[i][5] = 1
    
    return timeslot


def construct_timeslot(team):
    
    """
    Get as input the analyzed team.
    Function used to construct the call the function create_timeslot for both training and testing.
    Return the two arrays with the timeslots. 
    """
    
    dt = h5py.File("Files/"+str(team)+"/"+str(team)+"Testing.h5", 'r')

    train = dt['Timestamp_y']
    timeslot = create_timeslot(train)
    
    return timeslot
    

def get_possession(team):
    
    """
    Get as input the analyzed team.
    Function used to extract the array with the ball possession for each closeness component elaborated
    in the Make_dataset module
    """
    
    dt = h5py.File("Files/"+str(team)+"/"+str(team)+"Testing.h5", 'r')
    
    ball = np.array(dt['Ball'])
    
    
    return ball


def concat(arr1, arr2):
    
    """
    Get as input the external component separated.
    Used to merge all the external component preparated. 
    """
    
    meta = np.concatenate((arr1, arr2), axis=1)
    
    return meta


def create_external(team, sc_id, ws_id):
    
    """
    Main of the module
    """

    print("\n")
    print("=================================")
    print("Start extracting external element")
    print("=================================")
    print("\n")

    #Get the timeslots
    timeslot = construct_timeslot(team)
    
    #Get the ball possession
    ball = get_possession(team)

    #Merge both training and testing
    meta = concat(timeslot, ball)

    #Get the external attributes related to the events
    res = extract_advanced_external(team, sc_id, ws_id)

    #Merge the external elemente realted to the events with those already calculated
    meta = concat(meta, res)
    
    return meta


# In[34]:


def get_match(matchId):
    
    """
    Get as input the WhyScout match ID to consider
    Extract events in the match that we are considering
    Return the vector with the events of the match
    """
    
    #Open the event file
    f = open('External_element/Files/event.json')
    data = json.load(f)

    #extract the event for the specific match
    res = []
    for dt in data:
        if dt['matchId']==matchId:
            res.append(dt)
    
    return res


def extract_events(res):
    
    """
    Get as input the events of match
    Extract events in the match that are relevant for the players movement, like corner and penalty
    Return the vector with corner and penalty event data
    """
    
    
    corner = []
    penalty = []
    
    #extract conrer and penalty event data
    for i in range(0, len(res)):
        #30 is the subEventId of the corner
        if res[i]['subEventId'] == 30:
            corner.append(res[i])
        #35 is the subEventId of the penalty
        if res[i]['subEventId'] == 35:
            penalty.append(res[i])
    
    return corner, penalty


def extract_timestamp(vect):
    
    """
    Get as input the event data
    Extract the timestamp for each event data
    Return the timestamp vector
    """
    
    timestamp = []

    for i in range(0, len(vect)):
        #Extract period
        period = 1 if vect[i]['matchPeriod']=='1H' else 2
        #Extract minute
        minute = int(vect[i]['eventSec']/60)
        #Extract second
        second = int(vect[i]['eventSec']-minute*60)
        
        #construct timestamp
        timestamp.append((str(period)+str(minute).rjust(2,'0')+str(second).rjust(2,'0')))
    
    return timestamp


def extract_gol(res):
    
    """
    Get as input the events of match
    Extract from the events the goals
    Return the vector with goals event data
    """
    
    gol=[]
    
    for i in range(0, len(res)):
        #The goals are in the eventName=Shot with tag 101
        if res[i]['eventName']=='Shot':
            for j in range(0, len(res[i]['tags'])):
                if res[i]['tags'][j]['id']==101:
                    gol.append(res[i])
    
    return gol


def get_gol_limit(analyzed_team_ID, gol, timestamp_gol):
    
    """
    Get as input the ID of the team that we are analyzing, the data of the event goal and the timestamp of the goal
    We construct an array containing the limit of the goal in timestamp and the goal difference wrt to the analyzed team
    Return the vector
    """
    
    #Initialize the array that will contain the data, the array is structured in order to
    # delimitate the time between two different goals, so the first element will start from the start of the game
    # and will end when the first goal is scored. Additionally we have also the goal difference wrt to the analyzed team
    #It has shape(#goal+1, 3)
    #The array is composed by:
    #limit_gol[i][0] it is the lower bound limit of the time between two goals
    #limit_gol[i][1] it is the upper bound limit of the time between two goals
    #limit_gol[i][2] it is the goal difference wrt to the analyzed team in that time window between the goals
    limit_gol = np.empty([len(timestamp_gol)+1, 3], dtype="S7")
    
    #Sort the timestamp of the gol
    to_sort = [int(x) for x in timestamp_gol]
    to_sort.sort()
    timestamp_gol = [str(x) for x in to_sort]

    #Get the time limits, with the special case when we don't have goals
    if len(timestamp_gol)==0:
        limit_gol[0][0] = '10000' #Start game
        limit_gol[-1][1] = '29959' #End game
        limit_gol[-1][2] = '0'
    else:
        limit_gol[0][0] = '10000' #Start game
        limit_gol[0][1] = timestamp_gol[0]
        limit_gol[-1][0] = timestamp_gol[-1]
        limit_gol[-1][1] = '29959' #End game
        for i in range(1, len(timestamp_gol)):
            limit_gol[i][0] = timestamp_gol[i-1]
            limit_gol[i][1] = timestamp_gol[i]

    #set up the goal difference using the data in gol vector
    
    #Special case with 0 goals
    limit_gol[0][2] = '0'
    for i in range(0, len(gol)):
        if gol[i]['teamId']==str(analyzed_team_ID):
            limit_gol[i+1][2] = str(int(limit_gol[i][2])+1)
        else:
            limit_gol[i+1][2] = str(int(limit_gol[i][2])-1)
            
    return limit_gol


def get_match_id(team):
    
    """
    Get as input the name of the analyzed team.
    Extract the wyscout ID of the matches used in the construction of the dataset
    Return the IDs
    """
    
    
    ids_WY = 2852683
    ids_SC = 4120
    
    return ids_WY, ids_SC
    

def get_team_id(team):
    
    """
    Get as input the name of the analyzed team.
    Extract the Wyscout ID of the team that we are analyzing.
    Return the ID.
    """
    
    #Conversion between team name and WyscoutID
    id_team = [[3315, 'Sassuolo'],
             [3163, 'Udinese'],
             [3168, 'Lecce'],
             [3160, 'Parma'],
             [3193, 'Genoa'],
             [3194, 'Verona'],
             [3166, 'Bologna'],
             [3185, 'Torino'],
             [3204, 'Spal'],
             [3176, 'Fiorentina'],
             [3172, 'Atalanta'],
             [3161, 'Inter'],
             [3187, 'Napoli'],
             [3162, 'Lazio'],
             [3157, 'Milan'],
             [3173, 'Cagliari'],
             [3159, 'Juventus'],
             [3158, 'Roma'],
             [3167, 'Brescia'],
             [3164, 'Sampdoria']]
    
    for i in range(0, len(id_team)):
        if id_team[i][1] == team:
            analyzed_team_ID = id_team[i][0]
        
    return analyzed_team_ID


def extract_external(team, sc_id, ws_id):

    """
    Get as input the analyzed team.
    Function used to call the function to extract the event data of each match.
    Return the two arrays with the events. 
    """
    
    #Get the Wyscout ID of the analysis team
    analyzed_team_ID = get_team_id(team)
    
    #Get the IDs of the matches used in the dataset
    ids_WY, ids_SC = ws_id, sc_id
    
    corner_ = []
    penalty_ = []
    gol_ = []
    
    
    #Extract event of a match
    res = get_match(ids_WY)

    #Extract corner and penalty event data
    corner, penalty = extract_events(res)
    #Construct corner and penalty timestamp
    timestamp_cor = extract_timestamp(corner)
    timestamp_pen = extract_timestamp(penalty)
    
    #Extract gol event data
    gol = extract_gol(res)
    #Construct gol timestamp
    timestamp_gol = extract_timestamp(gol)
    #Construct gol limit
    limit_gol = get_gol_limit(analyzed_team_ID, gol, timestamp_gol)
    
    corner_.append([ids_SC, timestamp_cor])
    penalty_.append([ids_SC, timestamp_pen])
    gol_.append([ids_SC, limit_gol])
    
    return corner_, penalty_, gol_
        

def get_timestamp(team):
    
    """
    Get as input the analyzed team.
    Function used to construct the call the function create_timeslot for both training and testing.
    Return the two arrays with the timeslots. 
    """
    
    dt = h5py.File("Files/"+str(team)+"/"+str(team)+"Testing.h5", 'r')

    train = np.array(dt['Timestamp_c'])
    
    
    return train


def match_events(data, event):
    
    """
    Get as input the array with length equal to the number of sample in training/test
     with refer to the one frame of closeness and the event data (not gol).
    Match the timestamp of the trainig/testing instance with the timestamps of the event.
    Return the array with 1 or 0 if the event happend close to the timestamp of the training/testing
     closeness frame passed as input. 
    """
    
    #Array that will contain the output with size: len(data)x1
    out= np.zeros((len(data)))

    #Iterate through the data
    for i in range(0, len(data)):
        #Get the match_id of the timestamp
        match=data[i][0].decode("UTF-8")
        #iterate through the event
        #print(len(event))
        for j in range(0, len(event)):
            #Get the event list of the match of the data
            #print(type(match), data[i], type(event[j][0]), event[j][1])
            if event[j][0] == int(match): #.decode('UTF-8')
                #Extract period, minute and second of the data
                period_time = data[i][2].decode("UTF-8")[:1] 
                minute_time = data[i][2].decode("UTF-8")[1:3]
                second_time = data[i][2].decode("UTF-8")[3:5]

                #period_time = data[i][2][:1] 
                #minute_time = data[i][2][1:3]
                #second_time = data[i][2][3:5]
                #iterate through the event timestamp of the specific match
                for s in range(0, len(event[j][1])):
                    #Extract period, minute and second of the event
                    period_eve = event[j][1][s][0:1]
                    minute_eve = event[j][1][s][1:3]
                    second_eve = event[j][1][s][3:]
                    #Check if match with precision of second (and confidence +/- 1 second)
                    if period_time==period_eve and minute_time==minute_eve and (second_time==second_eve or str(int(second_eve)+1)==second_time or str(int(second_eve)-1)==second_time):
                        out[i]=1
            
    return out


def is_in(data, gol):
    
    """
    Get as input a specific gol limit of a given match and the timestamp analyzed.
    The match between the match data and the event data has been alrready checked. Check if the timestamp
     passed as input is inside the limit passed as input in the variable gol. The accuracy is on the minutes, 
     not the seconds. 
    Return True or False, depending if the timestamp is inside or not. 
    """
    
    #Default value
    check=False
    
    #Extract period, minute and second of the data
    period_time = int(data[2].decode("UTF-8")[:1])
    minute_time = int(data[2].decode("UTF-8")[1:3])
    second_time = int(data[2].decode("UTF-8")[3:5])
    
    #Extract period, minute and second of the lower bound (goal event timestamp)
    period_lower = int(gol[0][0:1].decode('UTF-8'))
    minute_lower = int(gol[0][1:3].decode('UTF-8'))
    second_lower = int(gol[0][3:].decode('UTF-8'))
    
     #Extract period, minute and second of the upper bound (goal event timestamp)
    period_upper = int(gol[1][0:1].decode('UTF-8'))
    minute_upper = int(gol[1][1:3].decode('UTF-8'))
    second_upper = int(gol[1][3:].decode('UTF-8'))
    
    #Limits both in one period
    if period_time==period_upper and period_time==period_lower:
        if minute_time>minute_lower and minute_time<=minute_upper:
            check=True
            
    #lower limit in the first period, upper limit in the second period and timestamp of first period
    elif period_time==period_lower and (period_time+1)==period_upper:
        if minute_time>minute_lower:
            check=True
            
    #lower limit in the first period, upper limit in the second period and timestamp of second period
    elif (period_time-1)==period_lower and period_time==period_upper:
        if minute_time<minute_upper:
            check=True
    
    
    return check
        

def match_goals(data, event):
    
    """
    Get as input the array with length equal to the number of sample in training/test
     with refer to the one frame of closeness and the gol event data.
    Match the timestamp of the trainig/testing instance with the timestamps of the gol.
    Return the array with the gol difference wrt to the analyzed team in that timestamp.
    """
    
    #array that will contain the output
    out = np.zeros((len(data)))
    
    #Iterate through the data
    for i in range(0, len(data)):
        #Get the match_id of the timestamp
        match=data[i][0].decode("UTF-8")
        #iterate through the event
        for j in range(0, len(event)):
            #Get the event list of the match of the data
            if event[j][0] == int(match):
                #Iterate through the timestamp limit of the gol
                for s in range(0, len(event[j][1])):
                    #Call function 'is_in' to see if the data is in the specifed limit
                    check = is_in(data[i], event[j][1][s])
                    if check:
                        #Append goal difference to the array
                        out[i]=int(event[j][1][s][2])
    
    return out
        

def extract_advanced_external(team, sc_id, ws_id):
    
    """
    Main of the module
    """
    
    #Vectors to arrange the list
    vect_train = []
    
    #Extract timestamp
    train = get_timestamp(team)
    
    #Extract external 
    corner_, penalty_, gol_ = extract_external(team, sc_id, ws_id)
    
    #Tranform the list to have a list for each closenes frame containing all the closeness frame for training/testing
    for i in range(0, train.shape[1]):
        vect_train.append([train[j][i] for j in range(0, len(train))])
    
    
    #train_res and test_res will contain the output of the overall process.
    #The shape will be: (3*len_closeness, number sample), then we will transpose it.
    #The first element of each frame (0, 3, 6, ...) will contain 1 if the timestamp is close to a coner, 0 otherwise
    #The second element of each frame (1, 4, 7, ...) will contain 1 if the timestamp is close to a penalty, 0 otherwise
    #the third element of each frame (2, 5, 8, ...) will contain the goals difference in that frame wrt the analyzed team
    train_res = np.zeros((3*np.shape(vect_train)[0], np.shape(vect_train)[1]))   

    #Prepare the output data for training 
    for j in range(0, np.shape(vect_train)[0]):
        #Extract the corner data
        corner = match_events(vect_train[j], corner_)
        train_res[0+j*3] = corner
        #Extract the penalty data
        penalty = match_events(vect_train[j], penalty_)
        train_res[1+j*3] = penalty
        #Extract the gol data
        gol = match_goals(vect_train[j], gol_)
        train_res[2+j*3] = gol
    
    #Transpose the data
    train_res = np.transpose(train_res)
    
    
    return train_res


# In[35]:


import numpy as np


np.random.seed(1337)  # for reproducibility


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

        

        print("min:", self._min, "max:", self._max)

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


# In[36]:


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


# In[37]:


def invert_context(data, frame):
    
    """
    0-> più vicino
    4-> più lontano
    Frame_0: 0-1, 2
    Frame_1: 3-4, 5
    Frame_2: 6-7, 8
    Frame_3: 9-10, 11
    Frame_4: 12-13, 14
    """
    global temp
    global temp_r
    
    c=data
    
    indexes = [[0, 1, 2],
              [3, 4, 5],
              [6, 7, 8],
              [9, 10, 11],
              [12, 13, 14]]
    
    
    an = indexes[frame][0]
    op = indexes[frame][1]
    ba = indexes[frame][2]

    for i in range(0, len(c)):

        temp = cp.deepcopy(c[i][an])
        c[i][an] = c[i][op]
        c[i][op] = temp

        for r in range(0, 5):
            temp_r=cp.deepcopy(c[i][an][r][0])
            c[i][an][r][0] = c[i][an][r][2]
            c[i][an][r][2] = temp_r

            temp_r=cp.deepcopy(c[i][1][r][0])
            c[i][op][r][0] = c[i][op][r][2]
            c[i][op][r][2] = temp_r


        for r in range(0, 5):
            for co in range(0, 3):
                if c[i][ba][r][co]==1:
                    if co==0:
                        c[i][ba][r][2]=1.0
                        c[i][ba][r][0]=0.0
                        break
                    if co==2:
                        c[i][ba][r][2]=0.0
                        c[i][ba][r][0]=1.0
                        break
             
    return c


# In[110]:


def invert_meta_time(meta_l):
    
    for j in range(0, len(meta_l)):
        back=0
        for i in range(0, len(meta_l[j][:6])):
            if meta_l[j][:6][i] == 1:
                back=i

        if back>=3:
            meta_l[j][back-3] = 1
            meta_l[j][back]=0
        else:
            meta_l[j][back+3]=1
            meta_l[j][back]=0
            
    return meta_l


# In[111]:


def invert_meta_ball(meta_l):
    
    for j in range(0, len(meta_l)):
        for i in range(6, 11):
            if meta_l[j][i] == 1:
                meta_l[j][i] = 0
                continue
            elif meta_l[j][i] == 2:
                meta_l[j][i] = 1
                continue
            elif meta_l[j][i] == 0:
                meta_l[j][i] = 2
                continue
    
    return meta_l


# In[108]:


def invert_meta_event(meta_l):
    
    ind = 11
    for j in range(0, len(meta_l)):
        for i in range(0, 5):
            for c in range(0, 3):
                if c==2:
                    meta_l[j][ind+c+i*3] = np.negative(meta_l[j][ind+c+i*3])
                else:
                    if meta_l[j][ind+c+i*3] == 0:
                        meta_l[j][ind+c+i*3]=1
                    else:
                        meta_l[j][ind+c+i*3]=0
    return meta_l     


# In[39]:


def calc_simil(y_pred, y_pred_inv):
    
    tot = []
    for i in tqdm(range(0, len(y_pred))):
        for f in range(0, len(y_pred[0])):
            a1 = []
            a2 = []
            for r in range(0, 5):
                for c in range(0, 3):
                    a1.append(y_pred[i][f][r][c])
                    a2.append(y_pred_inv[i][f][r][c])
            tot.append(jaccard_score(a1, a2, average='micro')) 
    
    return np.mean(tot)


# In[42]:


def get_ID_list(team):
    
    out = []
    file_name = []

    directory = "Matches"

    h5 = h5py.File(get_ID(team), 'r')

    id_list = []

    for i in h5['SC_ID']:
        id_list.append(i.decode('UTF-8'))

    for filename in os.listdir(directory):

        #Get the name of the file
        f = os.path.join(directory, filename)

        # checking if it is a file and ignore some file
        if os.path.isfile(f) and f != 'Matches/.DS_Store':

            id_=f[8:12]

            if id_ in id_list:

                fil = pd.read_json(get_match_info(id_), lines=True)

                wy_id = fil['wyscout_id'][0]

                out.append([id_, wy_id])
                file_name.append(f)
    
    return out, file_name


# In[ ]:


def write_all(team, C, P, Y_, Ball, Timestamp_, Meta, Prediction):
    
    tot_c = []
    for i in range(0, len(C)):
        for c in range(0, len(C[i])):
            tot_c.append(C[i][c])

    tot_p = []
    for i in range(0, len(P)):
        for c in range(0, len(P[i])):
            tot_p.append(P[i][c])

    tot_y = []
    for i in range(0, len(Y_)):
        for c in range(0, len(Y_[i])):
            tot_y.append(Y_[i][c])

    tot_b = []
    for i in range(0, len(Ball)):
        for c in range(0, len(Ball[i])):
            tot_b.append(Ball[i][c])

    t_c = []
    for m in range(0, len(Timestamp_)):
        for i in range(0, len(Timestamp_[m][0])):
            t_c.append(Timestamp_[m][0][i])

    t_p = []
    for m in range(0, len(Timestamp_)):
        for i in range(0, len(Timestamp_[m][1])):
            t_p.append(Timestamp_[m][1][i])

    t_y = []
    for m in range(0, len(Timestamp_)):
        for i in range(0, len(Timestamp_[m][2])):
            t_y.append(Timestamp_[m][2][i])

    Meta_t = []
    for m in range(0, len(Meta)):
        for i in range(0, len(Meta[m])):
            Meta_t.append(Meta[m][i])

    Prediction_=(np.vstack(Prediction))


    file = h5py.File("Files/"+team+"/Testing/"+team+"Testing.h5", 'w')

    #Write closeness
    file.create_dataset('X_c', data=tot_c)

    #Write period
    file.create_dataset('X_p', data=tot_p)

    #Write Y
    file.create_dataset('Y', data=tot_y)

    #Dataset with the metadata
    file.create_dataset('meta', data=Meta_t)

    #Write the timestamps of closeness
    file.create_dataset('Timestamp_c', data=np.string_(t_c))

    #Write the timestamps of period
    file.create_dataset('Timestamp_p', data=np.string_(t_p))

    #Write the timestamps of target test
    file.create_dataset('Timestamp_y', data=np.string_(t_y))

    #Write the ball possession for the train
    file.create_dataset('Ball', data=tot_b)

    file.close()


    h5 = h5py.File("Files/"+team+"/Testing/"+team+"_testing_pred.h5", 'w')

    h5.create_dataset('pred', data=Prediction_)

    h5.close()


# In[49]:


def start(team):

    ids_, file = get_ID_list(team)


    mmn = MinMaxNormalization()
    mmn.fit()

    C = []
    P = []
    Y_ = []
    Ball = []
    Timestamp_ = []
    Prediction = []
    Meta = []

    Inverted = []

    global P_inv
    global C_inv
    global Meta_inv

    for m in range(0, 5):

        c, p, Y, ball, Timestamp  = create_data(team, file[m])

        #print(c[0][:3])


        C.append(c)
        P.append(p)
        Y_.append(Y)
        Ball.append(ball)
        Timestamp_.append(Timestamp)

        #C_inv = cp.deepcopy(c)
        #c_inv = invert_context(C_inv, 4)

        #P_inv = cp.deepcopy(p)
        #p_inv = invert_context(P_inv, 4)


        #Inverted.append(c_inv)
        #Inverted.append(p_inv)

        #print(c_inv[0][:3])


        meta = create_external(team, ids_[m][0], ids_[m][1])

        Meta_inv = cp.deepcopy(meta)
        meta_inv = invert_meta_time(Meta_inv)
        meta_inv = invert_meta_ball(meta_inv)
        meta_inv = invert_meta_event(meta_inv)

        Meta.append(meta)

        c_norm = [mmn.transform(d) for d in c]
        p_norm = [mmn.transform(d) for d in p]

        #c_inv_norm = [mmn.transform(d) for d in c_inv]
        #p_inv_norm = [mmn.transform(d) for d in p_inv]


        X = []
        #Merge the internal component in one array
        for l, X_ in zip([params.len_clos, params.len_per], [c_norm, p_norm]):
            X.append(X_)

        """
        X_inv = []
        #Merge the internal component in one array
        for l, X_ in zip([params.len_clos, params.len_per], [c_inv_norm, p_norm]):
            X_inv.append(X_)




        X_inv = []
        #Merge the internal component in one array
        for l, X_ in zip([params.len_clos, params.len_per], [c_norm, p_inv_norm]):
            X_inv.append(X_)
        """


        X_torch = [torch.Tensor(x) for x in X]
        #X_torch_inv = [torch.Tensor(x) for x in X_inv]
        ext_torch = torch.Tensor(meta)
        #ext_torch_inv = torch.Tensor(meta_inv)

        model1=load_model(0, 10, meta, team)
        #model2=load_model(0, 10, meta_inv, team)

        y_pred = model1(X_torch[0], X_torch[1], torch.Tensor(np.array(meta)))
        #y_pred_inv = model2(X_torch[0], X_torch[1], torch.Tensor(np.array(meta_inv)))

        #DE-NORMALIZED PREDICTION
        y_pred = (y_pred + 1.) / 2.
        y_pred = 1. * y_pred * (10 - 0) + 0

        y_pred = np.round(y_pred.detach())

        """
        #DE-NORMALIZED PREDICTION
        y_pred_inv = (y_pred_inv + 1.) / 2.
        y_pred_inv = 1. * y_pred_inv * (10 - 0) + 0

        y_pred_inv = np.round(y_pred_inv.detach())
        """

        print(file[m])
        print("RMSE ", np.sqrt(np.mean((y_pred.detach().numpy() - Y)**2)))
        #print("Context changed ", np.sqrt(np.mean((y_pred_inv.detach().numpy() - Y)**2)))

        #similarity = calc_simil(y_pred, y_pred_inv)

        #print("Similarity: ", similarity)

        Prediction.append(y_pred)
        
    write_all(team, C, P, Y_, Ball, Timestamp_, Meta, Prediction)


# In[ ]:




