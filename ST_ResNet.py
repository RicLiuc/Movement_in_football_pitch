import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os
import torch.optim as optim
import time
from path import *

class ResUnit(nn.Module):
    # Definition of the residual unit composed by BatchNormalization, Convolutional layer, 
    # another Batch nomarlization and a final Convolutional layer
    def __init__(self, in_channels, out_channels, lng, lat):
        # It takes in a four dimensional input (B, C, lng, lat)
        super(ResUnit, self).__init__()
        
        #First batch normalization plus convolutional layer
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        
        #Second batch normalization plus convolutional layer
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
    
    def forward(self, x):
        
        z = self.bn1(x)
        z = F.relu(z)
        z = self.conv1(z)
       
        z = self.bn2(z)
        z = F.relu(z)
        z = self.conv2(z)

        return z + x #Sum of the residual and the input 




class STResNet(nn.Module):
    
    #Get as input all the hyperparameters set in the main.py
    #It instanciates the ST-ResNet and then creates it. 


    #All the parameters pre-setted
    def __init__(self,
                 #mmn,
                 learning_rate=0.0001,
                 epoches=50,
                 batch_size=32,
                 len_closeness=3,
                 len_period=1,
                 #len_trend=3,
                 external_dim=11,
                 map_heigh=2,
                 map_width=5,
                 nb_flow=2,
                 nb_residual_unit=2,
                 data_min = -999, 
                 data_max = 999,
                 team = 'Lazio'
                 ):

        super(STResNet, self).__init__()
        self.epoches = epoches
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.len_closeness = len_closeness
        self.len_period = len_period
        #self.len_trend = len_trend
        self.external_dim = external_dim
        self.map_heigh = map_heigh
        self.map_width = map_width
        self.nb_flow = nb_flow
        self.nb_residual_unit = nb_residual_unit
        self.logger = logging.getLogger(__name__)
        self.data_min = data_min
        self.data_max = data_max
        self.team = team
        #self.mmn = mmn

        #setting the location of the training, GPU if cuda is available, CPU in the other case
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu = torch.device("cuda:0")

        #Call function build_stresnet to construct the ST-ResNet with the residual unit setted for all the internal component
        #And the fully connected Neurla Network for the external component.
        self._build_stresnet()

        self.save_path =get_model_path(self.team, self.nb_residual_unit, self.len_closeness, self.len_period)
        self.best_rmse = 999
        self.best_mae = 999


    def _build_stresnet(self):
        #The actual construnction of the ST-ResNet

        branches = ['c', 'p'] #Closeness, Period 


        #Setting the CLOSENESS ResNet 
        #Creating the first convolutional out of the residual 
        self.c_net = nn.ModuleList([ #ModuleList holds submodels in a list
            nn.Conv2d(self.len_closeness * self.nb_flow, self.map_heigh*self.map_width*2, kernel_size=3, stride = 1, padding = 1), 
        ])
        #Append to the closeness model the number of the residual unit needed
        for i in range(self.nb_residual_unit):
            self.c_net.append(ResUnit(self.map_heigh*self.map_width*2, self.map_heigh*self.map_width*2, self.map_heigh, self.map_width))
        #Append tthe last convolutional layer out of the residual unit
        self.c_net.append(nn.Conv2d(self.map_heigh*self.map_width*2, self.nb_flow, kernel_size=3, stride = 1, padding = 1))


        #Setting the PERIOD ResNet 
        #Creating the first convolutional out of the residual 
        self.p_net = nn.ModuleList([
            nn.Conv2d(self.len_period * self.nb_flow, self.map_heigh*self.map_width*2, kernel_size=3, stride = 1, padding = 1), 
        ])
        #Append to the closeness model the number of the residual unit needed
        for i in range(self.nb_residual_unit):
            self.p_net.append(ResUnit(self.map_heigh*self.map_width*2, self.map_heigh*self.map_width*2, self.map_heigh, self.map_width))
        #Append tthe last convolutional layer out of the residual unit
        self.p_net.append(nn.Conv2d(self.map_heigh*self.map_width*2, self.nb_flow, kernel_size=3, stride = 1, padding = 1))

        #Setting the TREND ResNet 
        """
        #Creating the first convolutional out of the residual 
        self.t_net = nn.ModuleList([
            nn.Conv2d(self.len_trend * self.nb_flow, 32, kernel_size=3, stride = 1, padding = 1), 
        ])
        #Append to the closeness model the number of the residual unit needed
        for i in range(self.nb_residual_unit):
            self.t_net.append(ResUnit(32, 32, self.map_heigh, self.map_width))
        #Append tthe last convolutional layer out of the residual unit
        self.t_net.append(nn.Conv2d(32, self.nb_flow, kernel_size=3, stride = 1, padding = 1))
        """




        #Setting EXTERNAL COMPONENT Fully Connected Network
        
        self.ext_net = nn.Sequential(
            nn.Linear(self.external_dim, 10), 
            nn.ReLU(inplace = True),
            nn.Linear(10, self.nb_flow * self.map_heigh * self.map_width)
        )
        

        #Setting random parameter (between 0 and 1) for each of the ResNet
        self.w_c = nn.Parameter(torch.rand((self.nb_flow, self.map_heigh, self.map_width)), requires_grad=True)
        self.w_p = nn.Parameter(torch.rand((self.nb_flow, self.map_heigh, self.map_width)), requires_grad=True)
        #self.w_t = nn.Parameter(torch.rand((self.nb_flow, self.map_heigh, self.map_width)), requires_grad=True)



    def forward_branch(self, branch, x_in):
        #Iterate through the submodels in the ModuleList previously created passing the input in order to obtain the output for a given input
        for layer in branch:
            x_in = layer(x_in)
        return x_in

    def forward(self, xc, xp, ext): #,xt

        #Called from train_model, pass the input to the ST-ResNet

        #For Closeness, Period and Trend pass the input and the ModuleList created to implement it
        c_out = self.forward_branch(self.c_net, xc)
        p_out = self.forward_branch(self.p_net, xp)
        #t_out = self.forward_branch(self.t_net, xt)


        #Run the external component
        ext_out = self.ext_net(ext).view([-1, self.nb_flow, self.map_heigh, self.map_width])

        # FUSION

        #First fusion between Closeness, Period and Trend
        res = self.w_c.unsqueeze(0) * c_out + \
                self.w_p.unsqueeze(0) * p_out 
                #+ \ self.w_t.unsqueeze(0) * t_out


        #Fusion between Closeness and Period merged and the external component
        res += ext_out


        #print(res)

        #return torch.tanh(res)
        return (res)


    def train_model(self, train_loader, test_x, test_y, ext_test):

        print("\n")
        print("================================")
        print("Start training the model")
        print("================================")
        print("\n")

        #Trainig of the model
        #Get the training data through a data loader and the test data

        #Setting Adam as optimizer
        optimizer = optim.Adam(self.parameters(), lr = self.learning_rate)

        criterion = nn.MSELoss()

        start_time = time.time()
        epoch_loss = []
        rmse_list = []
        mae_list = []

        to_plot = []

        #Loop for the training 
        for ep in range(self.epoches):
            torch.manual_seed(0)
            self.train()
            
            for i, (xc, xp, ext, y) in enumerate(train_loader): #xt, ext, 

                #Pass to GPU if it is available
                if self.gpu_available:
                    xc = xc.to(self.gpu)
                    xp = xp.to(self.gpu)
                    #xt = xt.to(self.gpu)
                    ext = ext.to(self.gpu)
                    y = y.to(self.gpu)

                #Get the prediction calling function forward
                ypred = self.forward(xc, xp, ext) #, xt

                #Calc the loss (RMSE denormalized)
                #DE-NORMALIZED PREDICTION
                ypred_DN = (ypred + 1.) / 2.
                ypred_DN = 1. * ypred_DN * (self.data_max - self.data_min) + self.data_min   


                #DE-NORMALIZED REAL VALUE
                Y_test_DN = (y + 1.) / 2.
                Y_test_DN = 1. * Y_test_DN * (self.data_max - self.data_min) + self.data_min             

                #Calc the errors
                #loss = ((ypred_DN - Y_test_DN) **2).mean().pow(1/2)
                loss = torch.sqrt(criterion(ypred_DN, Y_test_DN))



                #Calc the gradient and update the weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #Vector for the loss
                epoch_loss.append(loss.item())

                
                if i % 50 == 0:
                    print("[%.2fs] ep %d it %d, loss %.4f"%(time.time() - start_time, ep, i, loss.item()))
            print("[%.2fs] ep %d, loss %.4f"%(time.time() - start_time, ep, np.mean(epoch_loss)))


            
            test_rmse, test_mae, y_pred = self.evaluate(test_x, test_y, ext_test)
            rmse_list.append(test_rmse)
            mae_list.append(test_mae)
            print("[%.2fs] ep %d test rmse %.4f, mae %.4f" % (time.time() - start_time, ep, test_rmse, test_mae))
            print("\n")

            to_plot.append([np.round(np.mean(epoch_loss), 4), np.round(test_rmse.item(), 4)])
            epoch_loss = []

            # Save best model
            if test_rmse < self.best_rmse:
                self.save_model("best")
                self.best_rmse = test_rmse
                self.best_mae = test_mae
        

        self.save_model("best")

        print(to_plot)

        return rmse_list,mae_list


    def evaluate(self, X_test, Y_test, ext):
        #Evaluation of the test set
        """
        X_test: a quadruplle: (xc, xp) , xt, ext
        y_test: a label
        mmn: minmax scaler, has attribute _min and _max
        """
        self.eval()

        if self.gpu_available:
            for i in range(3):
                X_test[i] = X_test[i].to(self.gpu)
            Y_test = Y_test.to(self.gpu)
        with torch.no_grad():
            ypred = self.forward(X_test[0], X_test[1], ext) #, X_test[3]
            

            #We de-normalise the prediction values and the real values in order to calc the error as stated in the original paper

            #DE-NORMALIZED PREDICTION
            ypred_DN = (ypred + 1.) / 2.
            ypred_DN = 1. * ypred_DN * (self.data_max - self.data_min) + self.data_min
            #ypred_DN = np.round(ypred_DN, 0)     


            #DE-NORMALIZED REAL VALUE
            Y_test_DN = (Y_test + 1.) / 2.
            Y_test_DN = 1. * Y_test_DN * (self.data_max - self.data_min) + self.data_min             

            #Calc the errors
            rmse_DN = ((ypred_DN - Y_test_DN) **2).mean().pow(1/2)
            mae_DN = ((ypred_DN - Y_test_DN).abs()).mean()

            xtest_DN1 = (X_test[0] + 1.) / 2.
            xtest_DN1 = 1. * xtest_DN1 * (self.data_max - self.data_min) + self.data_min
            xtest_DN1 = np.round(xtest_DN1.cpu(), 0)

            
            return rmse_DN , mae_DN, ypred_DN


    def pred(self, X, ext):
        self.eval()
        with torch.no_grad():
            ypred = self.forward(X[0], X[1], ext) #, X_test[2], X_test[3]
        return ypred

    
    def save_model(self, name):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        torch.save(self.state_dict(), self.save_path + name + ".pt")
    
    def load_model(self, name):
        if not name.endswith(".pt"):
            name += ".pt"
        self.load_state_dict(torch.load(self.save_path + name))
            
