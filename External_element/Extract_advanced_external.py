import numpy as np
import pandas as pd
import h5py
import json
from tqdm import tqdm
from path import *



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
    
    dt = h5py.File(get_ID(team), 'r')
    
    ids_WY = dt["WS_ID"]
    ids_SC = dt["SC_ID"]
    
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




def extract_external(team):

    """
    Get as input the analyzed team.
    Function used to call the function to extract the event data of each match.
    Return the two arrays with the events. 
    """
    
    #Get the Wyscout ID of the analysis team
    analyzed_team_ID = get_team_id(team)
    
    #Get the IDs of the matches used in the dataset
    ids_WY, ids_SC = get_match_id(team)
    
    corner_ = []
    penalty_ = []
    gol_ = []
    
    for i in tqdm(range(0, len(ids_WY))):
    
        #Extract event of a match
        res = get_match(ids_WY[i])

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
        
        corner_.append([ids_SC[i], timestamp_cor])
        penalty_.append([ids_SC[i], timestamp_pen])
        gol_.append([ids_SC[i], limit_gol])
    
    return corner_, penalty_, gol_
        




def get_timestamp(team):
    
    """
    Get as input the analyzed team.
    Function used to construct the call the function create_timeslot for both training and testing.
    Return the two arrays with the timeslots. 
    """
    
    dt = h5py.File(get_dataset(team), 'r')

    train = np.array(dt['Timestamp_train_0'])
    
    test = np.array(dt['Timestamp_test_0'])
    
    return train, test




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
        for j in range(0, len(event)):
            #Get the event list of the match of the data
            if event[j][0].decode('UTF-8') == match:
                #Extract period, minute and second of the data
                period_time = data[i][2].decode("UTF-8")[:1]
                minute_time = data[i][2].decode("UTF-8")[1:3]
                second_time = data[i][2].decode("UTF-8")[3:5]
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
            if event[j][0].decode('UTF-8') == match:
                #Iterate through the timestamp limit of the gol
                for s in range(0, len(event[j][1])):
                    #Call function 'is_in' to see if the data is in the specifed limit
                    check = is_in(data[i], event[j][1][s])
                    if check:
                        #Append goal difference to the array
                        out[i]=int(event[j][1][s][2])
    
    return out
        




def extract_advanced_external(team):
    
    """
    Main of the module
    """
    
    #Vectors to arrange the list
    vect_train = []
    vect_test = []
    
    #Extract timestamp
    train, test = get_timestamp(team)
    
    #Extract external 
    corner_, penalty_, gol_ = extract_external(team)
    
    #Tranform the list to have a list for each closenes frame containing all the closeness frame for training/testing
    for i in range(0, train.shape[1]):
        vect_train.append([train[j][i] for j in range(0, len(train))])
    
    for i in range(0, test.shape[1]):
        vect_test.append([test[j][i] for j in range(0, len(test))])
    
    #train_res and test_res will contain the output of the overall process.
    #The shape will be: (3*len_closeness, number sample), then we will transpose it.
    #The first element of each frame (0, 3, 6, ...) will contain 1 if the timestamp is close to a coner, 0 otherwise
    #The second element of each frame (1, 4, 7, ...) will contain 1 if the timestamp is close to a penalty, 0 otherwise
    #the third element of each frame (2, 5, 8, ...) will contain the goals difference in that frame wrt the analyzed team
    train_res = np.zeros((3*np.shape(vect_train)[0], np.shape(vect_train)[1]))
    test_res = np.zeros((3*np.shape(vect_test)[0], np.shape(vect_test)[1]))    

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
    
    #Prepare the output data for testing 
    for j in range(0, np.shape(vect_test)[0]):
        #Extract the corner data
        corner = match_events(vect_test[j], corner_)
        test_res[0+j*3] = corner
        #Extract the penalty data
        penalty = match_events(vect_test[j], penalty_)
        test_res[1+j*3] = penalty
        #Extract the gol data
        gol = match_goals(vect_test[j], gol_)
        test_res[2+j*3] = gol
    
    #Transpose the data
    test_res = np.transpose(test_res)
    
    return train_res, test_res







