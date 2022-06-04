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


def select_team():

    """
    Function to get as user-input the name of the team to analyze.
    Return the name of the team to analyze.
    """

    #List with all the possible valid team name
    team_l = ['Sassuolo',
                 'Udinese',
                 'Lecce',
                 'Parma',
                 'Genoa',
                 'Verona',
                 'Bologna',
                 'Torino',
                 'Spal',
                 'Fiorentina',
                 'Atalanta',
                 'Inter',
                 'Napoli',
                 'Lazio',
                 'Milan',
                 'Cagliari',
                 'Juventus',
                 'Roma',
                 'Brescia',
                 'Sampdoria']


    print('Choose a team to analyze:')
    print('Type help to see the list of the team.')
    print("\n")
    while(True):
        team = input() #Get the team name as input
        if (team =='help'): #To see the list of the valid name
            print(team_l)
        elif team not in team_l: #Team not valid case
            print('Invalid team')
        elif team in team_l: #Valid team case
            break
    
    return team


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
        
        if dt[to_pred+50][0]['period'] != dt[this_index][0]['period']\
        or len(dt[this_index][0]['data'])==0:
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
        
        if dt[to_pred+50][0]['period'] != dt[this_index][0]['period']\
        or len(dt[this_index][0]['data'])==0:
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
    Get as input the file with the data, the index to predict, the match ID, the dictionary with the teams composition, 
     where plays the nalayzed team and the other and the array with the variable to multiplicate to normalize everything. 
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
                    if data[i]["x"]*mult[this_index]<=limit[j][1] \
                    and data[i]["x"]*mult[this_index]>limit[j][2] \
                    and data[i]["y"]*mult[this_index]<=limit[j][3] \
                    and data[i]["y"]*mult[this_index]>limit[j][4]:
                        
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
                    if data[i]["x"]*mult[this_index]<=limit[j][1] \
                    and data[i]["x"]*mult[this_index]>limit[j][2] \
                    and data[i]["y"]*mult[this_index]<=limit[j][3] \
                    and data[i]["y"]*mult[this_index]>limit[j][4]:
                        
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
    Get as input the file with the data, the index to predict, the dictionary with the teams composition 
     and the variable to multiplicate to normalize the positions.
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
                if data[i]["x"]*mult<=limit[j][1] \
                and data[i]["x"]*mult>limit[j][2] \
                and data[i]["y"]*mult<=limit[j][3] \
                and data[i]["y"]*mult>limit[j][4]:

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
    for i in range(0, int(params.sample/(params.num_matches))):
        if len(valid) != 0:
            cho = random.choice(valid)

            if cho not in extracted_frame:
                extracted_frame.append(cho)

            valid.remove(cho)
        else:
            print(f"Not enough data, extracted {len(extracted_frame)} samples")
            break

    return extracted_frame


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




def write_dataset(X_train, Y_train, X_test, Y_test, ball_train, ball_test, time_train, timeY_train, time_test, timeY_test, team):
    
    print("Writing dataset in the file")
    
    

    #Open file
    file = h5py.File(get_dataset(team), 'w')
    
    #Write X_train
    for i, data in enumerate(X_train):
        file.create_dataset('X_train_%i' % i, data=data)
        
    #Write X_test
    for i, data in enumerate(X_test):
        file.create_dataset('X_test_%i' % i, data=data)
        
    #Write Y_train
    file.create_dataset('Y_train', data=Y_train)
    
    #Write Y_test
    file.create_dataset('Y_test', data=Y_test)
    
    #Write the timestamps of the train
    for i, data in enumerate(time_train):
        file.create_dataset('Timestamp_train_%i' % i, data=np.string_(data))
        
    #Write the timestamps of the test
    for i, data in enumerate(time_test):
        file.create_dataset('Timestamp_test_%i' % i, data=np.string_(data))
        
    #Write the timestamps of traget train
    file.create_dataset('Y_timestamps_train', data=np.string_(timeY_train))
    
    #Write the timestamps of target test
    file.create_dataset('Y_timestamps_test', data=np.string_(timeY_test))
    
    #Write the ball possession for the train
    file.create_dataset('Ball_train', data=ball_train)
    #Write the ball possession for the test
    file.create_dataset('Ball_test', data=ball_test)
    
    file.close()



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



def create_data(team):
    

    team_id, team_short = get_id_short(team)
    print("\n")

    directory = "Matches"

    #num_matches = 1
    num_matches = params.num_matches
    j=0

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


    #Iterate through the files
    for filename in os.listdir(directory):

        #Get the name of the file
        f = os.path.join(directory, filename)

        # checking if it is a file and ignore some file
        if os.path.isfile(f) and f != 'Matches/.DS_Store':

            #Get the name of teams involved in the file 
            home=f[12:15]
            away=f[15:18]

            #Check if the file regards the team selected
            if(home==team_short or away==team_short) and j<num_matches:

                j=j+1

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
                team_0, team_1, other_id, wys_id = get_team(match_id, team_id)

                #check if the analyzed team is the home team or the away team, used for the ball possession
                if home==team_short:
                    analyzed = 'home team'
                    other = 'away team'
                elif away==team_short:
                    analyzed = 'away team'
                    other = 'home team'

                #Call the function to obtain the variable to multiplicate to the 'X' coordinates if we need to invert the data
                # normalizing the analyzed team always attacking 'left to right'
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


    #Split training and test
    X_train, Y_train, X_test, Y_test, time_train, timeY_train, time_test, \
    timeY_test, ball_train, ball_test = split_test(X_c, X_p, Y, Timestamp[2], Timestamp[0], Timestamp[1], ball, \
               params.len_test, params.len_clos, params.len_per)


    #Write data in a file for further use
    write_dataset(X_train, Y_train, X_test, Y_test, ball_train, ball_test, time_train, timeY_train, time_test, timeY_test, team)

    #Write all the Wyscout id used in the construction of the dataset in a file
    write_id(wyscout_id, skillcorner_id, team)


    print("Written")
    print("\n")

    return X_train, Y_train, X_test, Y_test



