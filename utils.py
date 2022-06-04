import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch
from ST_ResNet import STResNet
from path import *
from MinMaxNorm import MinMaxNormalization
import params
import h5py
from path import *


def normalize_and_tensor(X_train, Y_train, X_test, Y_test, meta_train):

    """
    Get as input training and test data for both sequences and external data
    Normalze the data and transform the data in tensor.
    Return the data loader for the training phase and the test data.
    """

    #Normalizing dataset
    mmn = MinMaxNormalization()
    mmn.fit(X_train, X_test, Y_train, Y_test)

    X_train = [mmn.transform(d) for d in X_train]
    X_test = [mmn.transform(d) for d in X_test]
    Y_train = [mmn.transform(d) for d in Y_train]
    Y_test_N = [mmn.transform(d) for d in Y_test]


    # Put everything in a single tensor, in order to use Dataloader
    train_set = TensorDataset(torch.Tensor(np.array(X_train[0])), torch.Tensor(np.array(X_train[1])), torch.Tensor(meta_train), torch.Tensor(np.array(Y_train)))
    # corresponding to closeness, period, trend, external, y
    train_loader = DataLoader(train_set, batch_size = params.batch_size, shuffle = True)
    
    return train_loader, mmn, X_test, Y_test_N



def get_prediction(train_loader, X_test, Y_test_N, model, meta_test, team):

    """
    Get as input the model and the data for training.
    Train the model.
    Return the prediction.
    """
    
    
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        gpu = torch.device("cuda:0")

    if gpu_available:
        model = model.to(gpu)

    #Transform the test set into a tensor
    X_test_torch = [torch.Tensor(x) for x in X_test]

    # Train the model calling train_model
    rmse_list,mae_list = model.train_model(train_loader, X_test_torch, torch.Tensor(np.array(Y_test_N)), torch.Tensor(np.array(meta_test)))


    #Save the model
    model.save_model("best")
    model.load_model("best")

    #Evaluate the model
    rmse, mae, y_pred = model.evaluate(X_test_torch, torch.Tensor(np.array(Y_test_N)), torch.Tensor(np.array(meta_test)))

    y_pred = np.round(y_pred.cpu(), 0)
    
    return y_pred


def write_prediction(team, y_pred, y_test):


    #Write the prediction and the real value in a file .h5
    h5 = h5py.File(prediction(team), 'w')

    h5.create_dataset('y_test', data=y_test)
    h5.create_dataset('y_pred', data=y_pred)

    h5.close()


def write_external(meta_train, meta_test, team):

    #Write the prediction and the real value in a file .h5
    h5 = h5py.File("Files/"+str(team)+"/"+str(team)+"External.h5", 'w')

    h5.create_dataset('ext_train', data=meta_train)
    h5.create_dataset('ext_test', data=meta_test)

    h5.close()







