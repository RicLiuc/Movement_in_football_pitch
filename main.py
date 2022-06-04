import numpy as np 
from Create_Dataset import create_data
from torch.utils.data import TensorDataset, DataLoader
import torch
from ST_ResNet import STResNet
from path import *
from MinMaxNorm import MinMaxNormalization
from utils import *
import params
from External_element.External import create_external
import os



#List with all the possible valid team name
team_l = ['Sassuolo', 'Udinese',
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


#Check if the dataset has been already created, to save time
if os.path.isfile(get_dataset(team)):

    #If the dataset exists, extract the data already presented
    print("\n")
    print("Dataset already created, start extracting existed data")
    
    X_train = []
    X_test = []

    h5 = h5py.File(get_dataset(team), 'r')
    
    X_train_0 = h5["X_train_0"]
    X_train_1 = h5["X_train_1"]
    
    X_test_0 = h5["X_test_0"]
    X_test_1 = h5["X_test_1"]
    
    Y_train = h5['Y_train']
    Y_test = h5['Y_test']
    
    for l, X_ in zip([params.len_clos, params.len_per], [X_train_0, X_train_1]):
        X_train.append(X_)
    
    for l, X_ in zip([params.len_clos, params.len_per], [X_test_0, X_test_1]):
        X_test.append(X_)
    
    h5_meta = h5py.File(get_external(team), 'r')

    meta_train = h5_meta['ext_train']

    meta_test = h5_meta['ext_test']

    
    
else:


    X_train, Y_train, X_test, Y_test = create_data(team)

    #Extracting the external components
    meta_train, meta_test = create_external(team)

    write_external(meta_train, meta_test, team)


train_loader, mmn, X_test, Y_test_N = normalize_and_tensor(X_train, Y_train, X_test, Y_test, meta_train)

model = STResNet(
        learning_rate=params.lr,
        epoches=params.nb_epoch,
        batch_size=params.batch_size,
        len_closeness=params.len_clos,
        len_period=params.len_per,
        #len_trend=len_trend,
        external_dim=np.shape(meta_train)[1],
        map_heigh=params.map_height,
        map_width=params.map_width,
        nb_flow=params.nb_flow,
        nb_residual_unit=params.nb_residual_unit,
        data_min = mmn._min,
        data_max = mmn._max,
        team = team
        #mmn = mmn
    )


y_pred = get_prediction(train_loader, X_test, Y_test_N, model, meta_test, team)


write_prediction(team, y_pred, Y_test)










