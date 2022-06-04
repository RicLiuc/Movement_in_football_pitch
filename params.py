#SET HYPERPARAMETERS

#Number of row of the grid to create
row = 5
#Number of column of the grid to create
col = 3

#Team to analyze, no need to set this parameter, it is an input in the execution of the program
team = 'Lazio'

#Number of matches to analyze of the specified team
num_matches = 5

#
shift_test = 50

#REMEMBER
# The original shift between the frame is 0.1 second, so setting 'offset', 'shift_cos' and 'shift_per' 
# take into account this in order to impose the right value. 

#Length of the closeness sequences
len_clos = 5
#Shift between the frame in the closeness sequences
shift_clos = 8 #Make sure is greater than offset

#Length of the period sequences
len_per = 5
#Shift between the frame in the closeness sequences
shift_per = 170 #Make sure is greater of len_clos*shift_clos

#Number of samples in the test set
len_test = 2000
#Number of samples to extract from the data, it is the length of the all dataset that then will be splitted in training and test
sample = 25000

#SET HYPERPARAMETERS OF THE MODEL

#Number of epoch at training stage
nb_epoch = 30 
#Batch size
batch_size = 32  
#Length of closeness dependent sequence
len_closeness = len_clos  
#Length of  peroid dependent sequence
len_period = len_per  
#Length of trend dependent sequence 
#len_trend = 1  
#Grid size
map_height, map_width = row, col  
#Number of channels (2, inflow and outflow)
nb_flow = 3 
# learning rate
lr = 0.001
#Number of residual units
nb_residual_unit = 12