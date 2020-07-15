#PROGRAMMER:   Joshua Cayanan
# DATE CREATED: June 29, 2020                     
# REVISED DATE: 
# PURPOSE: Train a new network on a dataset of flower images

#   Example call:
#    python train.py --data_dir flowers --save_dir saves 

#Import functions created for this program
from get_input_args import get_input_args_train
from train_model import train_model
from save_checkpoint import save_checkpoint


#Funnel command line arguments into program
in_arg = get_input_args_train()

#Assign input arguments to variables
data_dir = in_arg.data_dir
save_dir = in_arg.save_dir
arch = in_arg.arch
learning_rate = in_arg.learning_rate
hidden_units = in_arg.hidden_units
epochs = in_arg.epochs
gpu = in_arg.gpu
if gpu:
    gpu = 'cuda'
else:
    gpu = 'cpu'

#Call function to train model, return model parameters
classifier_layers, model_state, optimizer_state = train_model(data_dir, arch, learning_rate, hidden_units, epochs, gpu)

#save the trained network to a checkpoint file
save_checkpoint(save_dir, arch, epochs, learning_rate, classifier_layers, model_state, optimizer_state)


