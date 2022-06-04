import numpy as np
import h5py
import pandas as pd
from External_element.Extract_advanced_external import extract_advanced_external
from path import *




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
    
    dt = h5py.File(get_dataset(team), 'r')

    train = dt['Y_timestamps_train']
    timeslot_train = create_timeslot(train)
    
    test = dt['Y_timestamps_test']
    timeslot_test = create_timeslot(test)
    
    return timeslot_train, timeslot_test
    
    




def get_possession(team):
    
    """
    Get as input the analyzed team.
    Function used to extract the array with the ball possession for each closeness component elaborated
    in the Make_dataset module
    """
    
    dt = h5py.File(get_dataset(team), 'r')
    
    ball_train = np.array(dt['Ball_train'])
    
    ball_test = np.array(dt['Ball_test'])
    
    return ball_train, ball_test




def concat(arr1, arr2):
    
    """
    Get as input the external component separated.
    Used to merge all the external component preparated. 
    """
    
    meta = np.concatenate((arr1, arr2), axis=1)
    
    return meta




def create_external(team):
    
    """
    Main of the module
    """

    print("\n")
    print("=================================")
    print("Start extracting external element")
    print("=================================")
    print("\n")

    #Get the timeslots
    timeslot_train, timeslot_test = construct_timeslot(team)
    
    #Get the ball possession
    ball_train, ball_test = get_possession(team)

    #Merge both training and testing
    meta_train = concat(timeslot_train, ball_train)
    meta_test = concat(timeslot_test, ball_test)

    #Get the external attributes related to the events
    train_res, test_res = extract_advanced_external(team)

    #Merge the external elemente realted to the events with those already calculated
    meta_train = concat(meta_train, train_res)
    meta_test = concat(meta_test, test_res)
    
    return meta_train, meta_test





